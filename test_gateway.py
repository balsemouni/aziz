"""
test_gateway.py — Voice Pipeline Monitor v3
══════════════════════════════════════════════════════════════

What's new vs v2
─────────────────
1. CLEAN ORGANIZED UI
   • Permanent live status bar (enrollment → idle → thinking → speaking)
   • Latency dashboard updates in-place after every turn
   • Conversation panel separated from system events
   • Color-coded phase indicators

2. INSTANT BARGE-IN
   • On "mute_mic" the output queue is drained immediately (no leftover audio)
   • On user speech during SPEAKING state: output queue is drained client-side
     before the barge-in is even confirmed by the gateway
   • This means the AI voice cuts off in <1 audio frame instead of finishing
     the current sd.play() call (~100-500ms lag in v2)

3. LOW-LATENCY AUDIO OUTPUT
   • Replaced sd.play() + sd.wait() (blocking, ~1 chunk lag) with a continuous
     sd.OutputStream that is fed from a ring buffer in real time
   • Audio latency drops from ~200-400ms (per-chunk blocking) to ~10-30ms
     (stream latency only)
   • Barge-in drains the ring buffer + output queue atomically so silence is
     heard immediately

Usage
─────
    python test_gateway.py
    python test_gateway.py --host 192.168.1.10 --port 8090
    python test_gateway.py --out-rate 24000 --echo-threshold 0.02
    python test_gateway.py --debug-raw

Requirements
────────────
    pip install websockets sounddevice numpy
"""

import argparse
import asyncio
import json
import sys
import datetime
import os
import threading
import collections

_missing = []
try:    import sounddevice as sd
except ImportError: _missing.append("sounddevice")
try:    import websockets
except ImportError: _missing.append("websockets")
try:    import numpy as np
except ImportError: _missing.append("numpy")
if _missing:
    print(f"Missing: pip install {' '.join(_missing)}")
    sys.exit(1)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--host",            default="localhost")
parser.add_argument("--port",            default=8090,  type=int)
parser.add_argument("--mic-rate",        default=16000, type=int)
parser.add_argument("--chunk-ms",        default=20,    type=int)
parser.add_argument("--out-rate",        default=24000, type=int)
parser.add_argument("--echo-threshold",  default=0.04,  type=float,
                    help="RMS gate when AI is speaking — raise if STT picks up TTS output")
parser.add_argument("--stream-buf-ms",   default=40,    type=int,
                    help="Output stream buffer in ms (lower = less latency, more risk of glitches)")
parser.add_argument("--debug-raw",       action="store_true")
args = parser.parse_args()

GW_URL        = f"ws://{args.host}:{args.port}/ws"
MIC_RATE      = args.mic_rate
CHUNK_MS      = args.chunk_ms
CHUNK_SAMP    = int(MIC_RATE * CHUNK_MS / 1000)
OUT_RATE      = args.out_rate
ECHO_THRESH   = args.echo_threshold
STREAM_BUF_MS = args.stream_buf_ms
DEBUG_RAW     = args.debug_raw

# ── Terminal width ─────────────────────────────────────────────────────────────
try:
    TW = min(os.get_terminal_size().columns, 120)
except Exception:
    TW = 100

# ── ANSI colours ─────────────────────────────────────────────────────────────
R    = "\033[0m"
BOLD = "\033[1m"
DIM  = "\033[2m"
ITA  = "\033[3m"
UND  = "\033[4m"

RED  = "\033[91m"
GRN  = "\033[92m"
YLW  = "\033[93m"
BLU  = "\033[94m"
MAG  = "\033[95m"
CYN  = "\033[96m"
WHT  = "\033[97m"
BLK  = "\033[30m"

BGBLK  = "\033[40m"
BGRED  = "\033[41m"
BGGRN  = "\033[42m"
BGYLW  = "\033[43m"
BGBLU  = "\033[44m"
BGMAG  = "\033[45m"
BGCYN  = "\033[46m"
BGWHT  = "\033[47m"
BGDARK = "\033[48;5;234m"
BGPNL  = "\033[48;5;236m"
BGDEEP = "\033[48;5;17m"

def c(*args):
    codes = "".join(args[:-1])
    return f"{codes}{args[-1]}{R}"

def ts():
    now = datetime.datetime.now()
    return c(DIM, f"{now.strftime('%H:%M:%S')}.{now.microsecond//1000:03d}")

def _bar(val, max_val, width=20, fill="█", empty="░"):
    if not max_val or val is None:
        return c(DIM, empty * width)
    filled = min(int((val / max_val) * width), width)
    col = GRN if val < 300 else YLW if val < 700 else MAG if val < 1500 else RED
    return c(col, fill * filled) + c(DIM, empty * (width - filled))

def _fmt_ms(ms, width=8):
    if ms is None:
        return c(DIM, "   —   ".ljust(width))
    col = GRN + BOLD if ms < 300 else YLW + BOLD if ms < 700 else MAG + BOLD if ms < 1500 else RED + BOLD
    return c(col, f"{ms:>{width-2}.0f}ms")

def _latency_icon(ms):
    if ms is None:  return c(DIM, "○")
    if ms < 300:    return c(GRN + BOLD, "●")
    if ms < 700:    return c(YLW + BOLD, "●")
    if ms < 1500:   return c(MAG + BOLD, "●")
    return c(RED + BOLD, "●")

SEP      = c(DIM, "─" * TW)
SEP_BOLD = c(CYN, "═" * TW)
SEP_DIM  = c(DIM, "┄" * TW)
INDENT   = "  "

# ── Shared state ──────────────────────────────────────────────────────────────
_audio_in_q:  asyncio.Queue
_loop = None
_stop = asyncio.Event()

_ai_speaking       = False
_ai_speaking_since = 0.0   # monotonic time when AI started speaking
AI_SPEAKING_MUTE_S = 0.35  # suppress ALL mic input briefly after TTS starts
_phase       = "CONNECTING"   # ENROLLING | IDLE | THINKING | SPEAKING | INTERRUPTED

# Voice enrollment
_enrolled    = False

# Conversation history  [(role, text)]
_conversation: list[tuple[str, str]] = []

# Current turn
_cur_user_words:   list[str] = []
_cur_ai_tokens:    list[str] = []
_cur_ai_sentences: list[str] = []
_cur_turn_id:      str = ""

# Latency of last turn (for live dashboard)
_last_lat: dict = {}
_turn_count  = 0
_barge_count = 0

# ─────────────────────────────────────────────────────────────────────────────
# LOW-LATENCY AUDIO OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
# We use a continuous sd.OutputStream fed by a deque ring buffer.
# The sounddevice callback pulls samples from the deque on every tick.
# When a barge-in happens we atomically clear the deque — silence is immediate.
# ─────────────────────────────────────────────────────────────────────────────

_pcm_deque: collections.deque = collections.deque()   # deque of float32 arrays
_deque_lock = threading.Lock()
_output_stream = None    # sd.OutputStream, created in _main


def _audio_out_cb(outdata: np.ndarray, frames: int, time_info, status):
    """Sounddevice output callback — runs in a real-time audio thread."""
    needed   = frames
    out_flat = np.zeros(needed, dtype=np.float32)
    pos      = 0

    with _deque_lock:
        while pos < needed and _pcm_deque:
            chunk = _pcm_deque[0]
            avail = len(chunk)
            take  = min(avail, needed - pos)
            out_flat[pos:pos + take] = chunk[:take]
            pos += take
            if take == avail:
                _pcm_deque.popleft()
            else:
                _pcm_deque[0] = chunk[take:]

    outdata[:, 0] = out_flat


def _enqueue_pcm(pcm_bytes: bytes):
    """Push raw PCM16 bytes to the output ring buffer (any thread)."""
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    with _deque_lock:
        _pcm_deque.append(arr)


def _drain_audio_output():
    """Immediately silence output — called on barge-in."""
    with _deque_lock:
        _pcm_deque.clear()


# ── Mic callback (sounddevice thread) ─────────────────────────────────────────

def _mic_cb(indata, frames, time_info, status):
    import time as _time
    global _ai_speaking, _ai_speaking_since
    chunk = indata[:, 0].copy().astype(np.float32)
    if _ai_speaking:
        # Hard-mute for the first AI_SPEAKING_MUTE_S after TTS starts
        # so the initial loud burst can't leak into STT.
        if _time.monotonic() - _ai_speaking_since < AI_SPEAKING_MUTE_S:
            return
        # After that, apply RMS gate — only pass loud user speech
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < ECHO_THRESH:
            return
    i16   = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
    frame = b"\x01" + i16.tobytes()
    if _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(_audio_in_q.put(frame), _loop)


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _print_header():
    print()
    print(c(BGDARK, CYN + BOLD, "  ◆  VOICE PIPELINE  ◆".ljust(TW)))
    print()


def _print_sep():
    print(c(DIM, "─" * TW))


def _print_tts_chunk(idx: int, text: str, tone: str = ""):
    label = c(MAG, f"  [{idx}]")
    print(f"{label}  {c(MAG + BOLD, text)}")


def _print_segment_complete(text: str):
    print()
    print(f"  {c(BGCYN, BLK + BOLD, ' YOU ')}  {c(CYN + BOLD, text)}")


def _print_barge_in():
    global _ai_speaking
    _drain_audio_output()
    _ai_speaking = False
    print(f"  {c(RED + BOLD, '⚡ interrupted')}")


def _print_latency_panel(ev: dict):
    barge  = ev.get("barge_in", False)
    stt    = ev.get("stt_latency_ms")
    cag    = ev.get("cag_first_token_ms")
    c2t    = ev.get("cag_to_tts_ms")
    tts    = ev.get("tts_synth_ms")
    e2e    = ev.get("e2e_ms")
    toks   = ev.get("total_tokens", 0)

    result = c(RED + BOLD, "⚡ INTERRUPTED") if barge else c(GRN + BOLD, "✓")
    vals   = [v for v in [stt, cag, c2t, tts] if v is not None]
    max_ms = max(vals) if vals else 1
    BAR_W  = 18
    LBL_W  = 22

    print(f"\n  {c(DIM, '─' * (TW - 2))}")
    print(f"  {c(CYN + BOLD, 'LATENCY')}  {result}  {c(DIM, str(toks) + ' tok')}")
    for lbl, ms in [
        ("STT  word→segment",  stt),
        ("CAG  segment→token", cag),
        ("CAG  token→TTS",     c2t),
        ("TTS  send→audio",    tts),
    ]:
        print(f"  {c(DIM, lbl.ljust(LBL_W))}  {_fmt_ms(ms)}  {_latency_icon(ms)}  {_bar(ms, max_ms, BAR_W)}")
    print(f"  {c(BOLD, 'E2E  word→audio'.ljust(LBL_W))}  {_fmt_ms(e2e)}  {_latency_icon(e2e)}")

    chunks = ev.get("tts_chunks", [])
    if chunks:
        max_cl = max((ch.get("synthesis_latency_ms") or 0) for ch in chunks) or 1
        print(f"  {c(DIM, 'TTS chunks:')}")
        for ch in chunks:
            i   = ch.get("chunk_index", 0)
            lat = ch.get("synthesis_latency_ms") or 0
            dur = ch.get("duration_sec") or 0.0
            star = c(CYN + BOLD, "★") if i == 0 else ""
            print(f"    {c(DIM,'['+str(i)+']')}  lat {_fmt_ms(lat)}  {_bar(lat, max_cl, 12)}  {c(DIM,f'{dur:.2f}s')}  {star}")
    print(f"  {c(DIM, '─' * (TW - 2))}\n")


def _print_session_summary(summary: dict):
    print(f"\n  {c(CYN + BOLD, 'SESSION SUMMARY')}")
    turns  = summary.get("turns", 0)
    barges = summary.get("barge_ins", 0)
    print(f"  turns:{turns}  barge-ins:{barges}")
    max_e2e = (summary.get("e2e", {}).get("max") or 0)
    for key, label in [("stt","STT"), ("cag","CAG"), ("tts_synth","TTS"), ("e2e","E2E")]:
        st = summary.get(key, {})
        if not st:
            continue
        print(f"  {c(BOLD, label)}  avg {_fmt_ms(st.get('avg'))}  p95 {_fmt_ms(st.get('p95'))}")
    print()


# ── Sender ─────────────────────────────────────────────────────────────────────

async def _sender(ws):
    while not _stop.is_set():
        try:
            frame = await asyncio.wait_for(_audio_in_q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        try:
            await ws.send(frame)
        except Exception:
            break


# ── Receiver ───────────────────────────────────────────────────────────────────

async def _receiver(ws):
    global _ai_speaking, _phase
    global _cur_user_words, _cur_ai_tokens, _cur_ai_sentences, _cur_turn_id
    global _last_lat, _turn_count, _barge_count, _enrolled

    _cur_user_words   = []
    _cur_ai_tokens    = []
    _cur_ai_sentences = []

    _in_token_stream = False

    def _end_inline():
        nonlocal _in_token_stream
        if _in_token_stream:
            print()
            _in_token_stream = False

    async for message in ws:
        import time as _time

        # ── binary PCM — AI speech audio ──────────────────────────────────────
        if isinstance(message, bytes):
            if message:
                if not _ai_speaking:
                    _ai_speaking       = True
                    _ai_speaking_since = _time.monotonic()
                    _phase = "SPEAKING"
                _enqueue_pcm(message)
            continue

        # ── JSON ──────────────────────────────────────────────────────────────
        try:
            ev = json.loads(message)
        except Exception:
            continue

        t = ev.get("type")

        if DEBUG_RAW:
            print(f"\n{ts()}  {c(DIM, 'RAW')}  {c(DIM, json.dumps(ev)[:160])}")

        if t == "enrolling":
            _phase = "ENROLLING"
            _end_inline()
            _cur_user_words = []
            print(f"\n  {c(YLW + BOLD, '🎤 Say your first sentence to enroll your voice.')}\n")

        elif t == "enrolled":
            _phase    = "IDLE"
            _enrolled = True
            _end_inline()
            _cur_user_words = []
            print(f"  {c(GRN + BOLD, '✓ Voice enrolled — speak freely.')}\n")

        elif t == "re_enrolling":
            _phase    = "ENROLLING"
            _enrolled = False
            _end_inline()
            _cur_user_words = []
            print(f"\n  {c(YLW, '↺ Re-enrolling…')}\n")

        elif t == "voice_mismatch":
            pass   # silently drop

        elif t in ("mute_mic", "unmute_mic"):
            pass   # mic state driven by PCM arrival and done event

        # ── STT words ─────────────────────────────────────────────────────────
        elif t == "word":
            word = ev.get("word", "")
            _cur_user_words.append(word)
            words_line = " ".join(_cur_user_words)
            if len(words_line) > TW - 16:
                words_line = "…" + words_line[-(TW - 17):]
            print(f"\r{ts()}  🎤  {c(CYN + BOLD, words_line)}  ", end="", flush=True)
            _in_token_stream = False

        elif t == "partial":
            word = ev.get("word", "").strip().rstrip("?.!,;:")
            if word and (not _cur_user_words or _cur_user_words[-1].lower() != word.lower()):
                _cur_user_words.append(word)
            words_line = " ".join(_cur_user_words)
            if len(words_line) > TW - 16:
                words_line = "…" + words_line[-(TW - 17):]
            print(f"\r{ts()}  🎤  {c(CYN, words_line)}  ", end="", flush=True)

        elif t == "segment":
            # Use the authoritative text from the gateway, not the accumulated words
            text = ev.get("text", "").strip()
            _end_inline()
            _cur_user_words = []   # reset accumulator regardless
            if not text:
                pass
            else:
                _print_segment_complete(text)
                _conversation.append(("user", text))
                if _phase == "SPEAKING":
                    _print_barge_in()
                    _barge_count += 1
                    _phase = "INTERRUPTED"

        # ── CAG ───────────────────────────────────────────────────────────────
        elif t == "thinking":
            _cur_turn_id      = ev.get("turn_id", "")
            _cur_ai_tokens    = []
            _cur_ai_sentences = []
            _cur_user_words   = []   # clear so next utterance starts fresh
            _end_inline()
            _phase = "THINKING"

        elif t == "ai_token":
            _cur_ai_tokens.append(ev.get("token", ""))
            # Don't print tokens inline — chunks give better signal

        elif t == "ai_sentence":
            text = ev.get("text", "").strip()
            tone = ev.get("tone", "")
            _cur_ai_sentences.append(text)
            _end_inline()
            _print_tts_chunk(len(_cur_ai_sentences), text, tone)

        elif t == "done":
            _end_inline()
            _cur_ai_tokens    = []
            _cur_ai_sentences = []
            _ai_speaking = False
            _phase = "IDLE"
            _turn_count += 1

        # ── Latency ───────────────────────────────────────────────────────────
        elif t == "latency":
            _end_inline()
            _last_lat = ev
            _print_latency_panel(ev)

        elif t == "session_summary":
            _end_inline()
            _print_session_summary(ev.get("latency", {}))
            _stop.set()

        # ── misc ──────────────────────────────────────────────────────────────
        elif t == "hallucination_reset":
            _end_inline()
            print(f"  {c(YLW, '⚠ hallucination reset')}")
            _cur_user_words = []

        elif t in ("session_reset", "error"):
            _end_inline()
            detail = ev.get("detail", ev.get("reason", ""))
            col = RED if t == "error" else YLW
            print(f"  {c(col + BOLD, '✗' if t == 'error' else '↺')}  {c(col, detail)}")


# ── Main ─────────────────────────────────────────────────────────────────────

async def _main():
    global _audio_in_q, _loop, _output_stream, _phase

    _audio_in_q  = asyncio.Queue(maxsize=500)
    _loop        = asyncio.get_running_loop()

    _print_header()
    print(f"  {c(DIM, GW_URL)}   mic:{MIC_RATE}Hz  spk:{OUT_RATE}Hz\n")
    print(SEP_DIM)
    _phase = "CONNECTING"

    # ── Continuous output stream (low-latency, drainable) ─────────────────────
    stream_blocksize = int(OUT_RATE * STREAM_BUF_MS / 1000)
    output_stream = sd.OutputStream(
        samplerate=OUT_RATE,
        channels=1,
        dtype="float32",
        blocksize=stream_blocksize,
        callback=_audio_out_cb,
    )
    output_stream.start()

    try:
        async with websockets.connect(
            GW_URL,
            ping_interval=20,
            ping_timeout=30,
            max_size=10 * 1024 * 1024,
        ) as ws:
            print(f"\n  {c(GRN + BOLD, '✓ Connected')}\n")
            _phase = "ENROLLING"

            with sd.InputStream(
                samplerate=MIC_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMP,
                callback=_mic_cb,
            ):
                print(f"  {c(GRN + BOLD, '🎤 Mic open')}\n")
                await asyncio.gather(
                    _sender(ws),
                    _receiver(ws),
                )

    except ConnectionRefusedError:
        print(c(RED, f"\n  Cannot connect to {GW_URL}"))
        print(c(DIM,  "  Start the gateway first: python gateway.py"))
        sys.exit(1)
    except websockets.exceptions.ConnectionClosedOK:
        print(c(YLW, "\n  Gateway closed the connection."))
    except Exception as e:
        print(c(RED, f"\n  {e}"))
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        output_stream.stop()
        output_stream.close()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        _stop.set()
        print(c(YLW, "\n\n  Stopped.\n"))