"""
test_gateway.py — Gateway Test Client v2
══════════════════════════════════════════════════════════════
Beautiful terminal UI showing the full conversation flow with
live latency visualization and echo suppression.

ECHO SUPPRESSION
────────────────
The mic is always on and always streams to the gateway.
Echo cancellation is handled entirely inside the STT pipeline:
  • AECGate      — timing-based gate (suppresses during AI speaking + 1200ms tail)
  • TTSVoiceGate — acoustic fingerprint gate (cosine similarity vs enrolled TTS voice)
No client-side muting needed.

Usage
─────
    python test_gateway.py
    python test_gateway.py --host 192.168.1.10 --port 8090
    python test_gateway.py --out-rate 24000 --echo-threshold 0.02

Requirements
────────────
    pip install websockets sounddevice numpy
"""

import argparse
import asyncio
import json
import sys
import datetime
import textwrap
import os

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
parser.add_argument("--debug-raw",       action="store_true")
args = parser.parse_args()

GW_URL        = f"ws://{args.host}:{args.port}/ws"
MIC_RATE      = args.mic_rate
CHUNK_MS      = args.chunk_ms
CHUNK_SAMP    = int(MIC_RATE * CHUNK_MS / 1000)
OUT_RATE      = args.out_rate
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

BLK  = "\033[30m"
RED  = "\033[91m"
GRN  = "\033[92m"
YLW  = "\033[93m"
BLU  = "\033[94m"
MAG  = "\033[95m"
CYN  = "\033[96m"
WHT  = "\033[97m"

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

def c(*args):
    codes = "".join(args[:-1])
    return f"{codes}{args[-1]}{R}"

def ts():
    now = datetime.datetime.now()
    return c(DIM, f"{now.strftime('%H:%M:%S')}.{now.microsecond//1000:03d}")

# ── State ─────────────────────────────────────────────────────────────────────
_audio_in_q:  asyncio.Queue
_audio_out_q: asyncio.Queue
_loop = None
_stop = asyncio.Event()

# Mic is always on — TTSVoiceGate in STT suppresses echo acoustically

# Conversation history  [(role, text)]
_conversation: list[tuple[str, str]] = []

# Current turn accumulator
_cur_user_words: list[str] = []
_cur_ai_tokens:  list[str] = []
_cur_ai_sentences: list[str] = []

# Latency of last completed turn
_last_latency: dict = {}

# ── Layout constants ──────────────────────────────────────────────────────────
SEP      = c(DIM, "─" * TW)
SEP_BOLD = c(CYN, "═" * TW)
INDENT   = "  "

# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(val, max_val, width=24, fill="█", empty="░"):
    if not max_val or val is None:
        return c(DIM, empty * width)
    filled = min(int((val / max_val) * width), width)
    col = GRN if val < 200 else YLW if val < 500 else MAG if val < 1000 else RED
    return c(col, fill * filled) + c(DIM, empty * (width - filled))

def _fmt_ms(ms):
    if ms is None:
        return c(DIM, "    —   ")
    col = GRN + BOLD if ms < 200 else YLW + BOLD if ms < 500 else MAG + BOLD if ms < 1000 else RED + BOLD
    return c(col, f"{ms:>6.0f}ms")

def _icon(ms):
    if ms is None:  return "○"
    if ms < 200:    return c(GRN, "●")
    if ms < 500:    return c(YLW, "●")
    if ms < 1000:   return c(MAG, "●")
    return c(RED, "●")

def _wrap(text, width, indent=""):
    lines = textwrap.wrap(text, width=width - len(indent))
    return ("\n" + indent).join(lines)

# ── Screen sections ──────────────────────────────────────────────────────────

def _print_header():
    print()
    print(c(BGDARK, CYN + BOLD, " " * TW))
    title = "  ◆  VOICE PIPELINE MONITOR  ◆"
    sub   = f"  {GW_URL}   mic:{MIC_RATE}Hz  out:{OUT_RATE}Hz"
    print(c(BGDARK, CYN + BOLD, title.ljust(TW)))
    print(c(BGDARK, DIM,        sub.ljust(TW)))
    print(c(BGDARK, CYN + BOLD, " " * TW))
    print()


def _print_conversation_entry(role: str, text: str):
    """Print one conversation message in a clean bubble style."""
    if role == "user":
        tag   = c(BGCYN, BLK + BOLD, " USER ")
        col   = CYN
        pfx   = "  "
    else:
        tag   = c(BGMAG, WHT + BOLD, "  AI  ")
        col   = MAG
        pfx   = "  "

    wrapped = _wrap(text, TW - 10, indent=" " * 10)
    print(f"\n{pfx}{tag}  {c(col, wrapped)}")


def _print_live_words(words: list, prefix="🎤"):
    """Overwrite current line with growing word stream."""
    line = " ".join(words)
    max_w = TW - 12
    if len(line) > max_w:
        line = "…" + line[-(max_w-1):]
    print(f"\r{ts()}  {prefix}  {c(GRN, line)}  ", end="", flush=True)


def _print_live_tokens(tokens: list, prefix="💬"):
    """Overwrite current line with growing token stream."""
    line = "".join(tokens)
    max_w = TW - 12
    if len(line) > max_w:
        line = "…" + line[-(max_w-1):]
    print(f"\r{ts()}  {prefix}  {c(BLU, line)}  ", end="", flush=True)


def _print_segment_banner(text: str):
    """Show a clean segment completion banner."""
    quoted = '"' + text + '"'
    print()
    print(SEP)
    print(f"  {c(GRN + BOLD, 'USER SAID')}   {c(GRN, quoted)}")
    print(SEP)


def _print_thinking_banner(tid: str):
    short_tid = tid[:8]
    print(f"\n{SEP}")
    print(f"  {c(YLW + BOLD, 'THINKING')}  {c(DIM, '[' + short_tid + ']')}")
    print(SEP)


def _print_tts_chunk(idx: int, text: str):
    idx_label = "[" + str(idx) + "]"
    print(f"\n  {c(MAG + BOLD, idx_label)}  {c(MAG, text)}")


def _print_latency_panel(ev: dict):
    """Full latency panel after each turn."""
    barge = ev.get("barge_in", False)
    stage = ev.get("stage", "")
    stt   = ev.get("stt_latency_ms")
    cag   = ev.get("cag_first_token_ms")
    c2t   = ev.get("cag_to_tts_ms")
    tts   = ev.get("tts_synth_ms")
    e2e   = ev.get("e2e_ms")
    toks  = ev.get("total_tokens", 0)
    tid   = ev.get("turn_id", "")[:8]

    label = c(RED + BOLD, "INTERRUPTED") if barge else c(GRN + BOLD, "COMPLETE")
    max_ms = max((v for v in [stt, cag, c2t, tts] if v), default=1)

    tid_label  = "[" + tid + "]"
    tok_label  = str(toks) + " tokens"

    print(f"\n{SEP_BOLD}")
    print(f"  {c(BOLD, 'LATENCY REPORT')}  {label}  {c(DIM, tid_label)}  {c(DIM, tok_label)}")
    print(SEP_BOLD)

    rows = [
        ("STT  word → segment  ", stt,  "Last spoken word until STT fires"),
        ("CAG  seg  → token    ", cag,  "STT segment until first AI token"),
        ("CAG  tok  → TTS send ", c2t,  "First token until first TTS chunk queued"),
        ("TTS  send → audio    ", tts,  "TTS chunk queued until audio arrives"),
    ]
    for lbl, ms, desc in rows:
        bar = _bar(ms, max_ms)
        print(f"  {c(DIM, lbl)}  {_fmt_ms(ms)}  {bar}  {c(DIM, desc)}")

    print()
    print(f"  {c(BOLD, 'E2E  word → audio     ')}  {_fmt_ms(e2e)}  {_icon(e2e)}")

    # TTS chunk breakdown
    chunks = ev.get("tts_chunks", [])
    if chunks:
        max_clag = max((ch.get("synthesis_latency_ms") or 0) for ch in chunks) or 1
        print(f"\n  {c(DIM, 'Chunk  Synth dur   Synth lat   Bar                       Audio dur   Offset')}")
        print(f"  {c(DIM, '─' * (TW - 4))}")
        for ch in chunks:
            idx_  = ch.get("chunk_index", 0)
            clag  = ch.get("synthesis_latency_ms") or 0
            cdur  = ch.get("synth_duration_ms")    or 0
            dsec  = ch.get("duration_sec")          or 0.0
            fcl   = ch.get("first_chunk_latency_ms")
            star  = c(CYN + BOLD, "FIRST") if idx_ == 0 else c(DIM, "+" + f"{fcl:.0f}" + "ms")
            cbar  = _bar(clag, max_clag, 20)
            chunk_label = "[" + str(idx_) + "]"
            dsec_label  = f"{dsec:.2f}" + "s"
            print(
                f"  {c(CYN, chunk_label)}  "
                f"{_fmt_ms(cdur)}    "
                f"{_fmt_ms(clag)}  "
                f"{cbar}  "
                f"{c(DIM, dsec_label)}       "
                f"{star}"
            )

    print(SEP_BOLD)



def _print_session_summary(summary: dict):
    print(f"\n{SEP_BOLD}")
    print(c(BOLD + CYN, "   SESSION SUMMARY"))
    print(SEP_BOLD)
    print(f"  Turns     : {c(BOLD, str(summary.get('turns', 0)))}")
    print(f"  Barge-ins : {c(BOLD, str(summary.get('barge_ins', 0)))}")

    for key, label in [
        ("stt",       "STT  word → segment "),
        ("cag",       "CAG  → first token  "),
        ("tts_synth", "TTS  → audio out    "),
        ("e2e",       "E2E  word → audio   "),
    ]:
        st = summary.get(key, {})
        if not st:
            continue
        avg = st.get("avg"); p95 = st.get("p95")
        mn  = st.get("min"); mx  = st.get("max")
        bar = _bar(avg, 2000, 20)
        print(
            f"\n  {c(BOLD, label)}\n"
            f"    avg {_fmt_ms(avg)}  p95 {_fmt_ms(p95)}  "
            f"min {_fmt_ms(mn)}  max {_fmt_ms(mx)}\n"
            f"    {bar}"
        )
    print(SEP_BOLD)


# ── Mic callback (sounddevice thread) ─────────────────────────────────────────

def _mic_cb(indata, frames, time_info, status):
    # Mic is always on — never mute client-side.
    # TTSVoiceGate inside the STT pipeline suppresses TTS echo acoustically.
    chunk = indata[:, 0].copy().astype(np.float32)
    i16   = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
    frame = b"\x01" + i16.tobytes()
    if _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(_audio_in_q.put(frame), _loop)


# ── Sender ────────────────────────────────────────────────────────────────────

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


# ── Speaker ───────────────────────────────────────────────────────────────────

async def _speaker():
    loop = asyncio.get_running_loop()
    while not _stop.is_set():
        try:
            pcm_bytes = await asyncio.wait_for(_audio_out_q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        await loop.run_in_executor(None, _play_sync, pcm)

def _play_sync(pcm: np.ndarray):
    sd.play(pcm, samplerate=OUT_RATE)
    sd.wait()


# ── Receiver ─────────────────────────────────────────────────────────────────

async def _receiver(ws):
    global _cur_user_words, _cur_ai_tokens, _cur_ai_sentences

    _cur_user_words   = []
    _cur_ai_tokens    = []
    _cur_ai_sentences = []

    _in_token_stream = False   # whether we're mid-stream printing tokens

    async for message in ws:

        # ── binary PCM ────────────────────────────────────────────────────────
        if isinstance(message, bytes):
            if message:
                await _audio_out_q.put(message)
            continue

        # ── JSON ──────────────────────────────────────────────────────────────
        try:
            ev = json.loads(message)
        except Exception:
            continue

        t = ev.get("type")

        if DEBUG_RAW:
            raw_preview = json.dumps(ev)[:160]
            # FIX: extract t before embedding in f-string (Python ≤ 3.11 compat)
            print(f"\n{ts()}  {c(DIM, 'RAW [' + str(t) + ']')}  {c(DIM, raw_preview)}")

        # ── STT words ─────────────────────────────────────────────────────────
        if t == "word":
            _cur_user_words.append(ev.get("word", ""))
            _print_live_words(_cur_user_words, "🎤")

        elif t == "partial":
            word = ev.get("word", "").strip().rstrip("?.!,;:")
            if word and (not _cur_user_words or _cur_user_words[-1].lower() != word.lower()):
                _cur_user_words.append(word)
            _print_live_words(_cur_user_words, "🎤")

        elif t == "segment":
            text = ev.get("text", "").strip()
            print()  # end inline word stream
            _print_segment_banner(text)
            _conversation.append(("user", text))
            _cur_user_words = []

        # ── CAG ───────────────────────────────────────────────────────────────
        elif t == "thinking":
            tid = ev.get("turn_id", "")[:8]
            _cur_ai_tokens    = []
            _cur_ai_sentences = []
            _print_thinking_banner(tid)
            # Start inline token stream
            print(f"{ts()}  {c(BLU + BOLD, 'AI')}  ", end="", flush=True)
            _in_token_stream = True

        elif t == "ai_token":
            tok = ev.get("token", "")
            _cur_ai_tokens.append(tok)
            print(c(BLU, tok), end="", flush=True)
            _in_token_stream = True

        elif t == "ai_sentence":
            text = ev.get("text", "").strip()
            _cur_ai_sentences.append(text)
            if _in_token_stream:
                print()
                _in_token_stream = False
            idx = len(_cur_ai_sentences)
            _print_tts_chunk(idx, text)
            # Restart inline token stream for next tokens
            print(f"{ts()}  {c(BLU + BOLD, 'AI')}  ", end="", flush=True)
            _in_token_stream = True

        elif t == "done":
            if _in_token_stream:
                print()
                _in_token_stream = False
            full = "".join(_cur_ai_tokens).strip()
            if full:
                _conversation.append(("ai", full))
                _print_conversation_entry("ai", full)
            _cur_ai_tokens    = []
            _cur_ai_sentences = []

        # ── Latency ───────────────────────────────────────────────────────────
        elif t == "latency":
            if _in_token_stream:
                print()
                _in_token_stream = False
            _print_latency_panel(ev)

        elif t == "session_summary":
            _print_session_summary(ev.get("latency", {}))
            _stop.set()

        # ── misc ──────────────────────────────────────────────────────────────
        elif t == "hallucination_reset":
            if _in_token_stream:
                print()
                _in_token_stream = False
            print(f"\n  {c(YLW + BOLD, 'HALLUCINATION GUARD')}  {c(YLW, ev.get('detail', ''))}")
            print(c(DIM, "  STT context reset. Discarding accumulated words."))
            _cur_user_words = []

        elif t == "session_reset":
            reason = ev.get("reason", "")
            print(f"\n  {c(YLW, 'Session reset: ' + reason)}")

        elif t == "error":
            if _in_token_stream:
                print()
                _in_token_stream = False
            print(f"\n  {c(RED + BOLD, 'ERROR')}  {c(RED, ev.get('detail', ''))}")

        elif t == "ping":
            pass

        else:
            if DEBUG_RAW:
                print(f"\n  {c(YLW, '?')}  {json.dumps(ev)[:120]}")


# ── Main ─────────────────────────────────────────────────────────────────────

async def _main():
    global _audio_in_q, _audio_out_q, _loop

    _audio_in_q  = asyncio.Queue(maxsize=500)
    _audio_out_q = asyncio.Queue(maxsize=500)
    _loop        = asyncio.get_running_loop()

    _print_header()

    print(f"  {c(DIM, 'Gateway')}  {c(BOLD, GW_URL)}")
    print(f"  {c(DIM, 'Echo suppression')}  {c(BOLD, 'TTSVoiceGate — acoustic fingerprint in STT pipeline')}")
    print(f"\n  {c(YLW, 'Speak naturally. Barge in any time to interrupt the AI.')}")
    print(f"  {c(DIM, 'Ctrl+C to quit.')}\n")
    print(SEP)

    try:
        async with websockets.connect(
            GW_URL,
            ping_interval=20,
            ping_timeout=30,
            max_size=10 * 1024 * 1024,
        ) as ws:
            print(f"\n  {c(GRN + BOLD, 'Connected')}\n")
            print(SEP_BOLD)

            with sd.InputStream(
                samplerate=MIC_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMP,
                callback=_mic_cb,
            ):
                print(f"\n  {c(GRN + BOLD, 'Mic open — start speaking...')}\n")
                await asyncio.gather(
                    _sender(ws),
                    _receiver(ws),
                    _speaker(),
                )

    except ConnectionRefusedError:
        print(c(RED, f"\n  Cannot connect to {GW_URL}"))
        print(c(DIM,  "     Start the gateway first: python gateway.py"))
        sys.exit(1)
    except websockets.exceptions.ConnectionClosedOK:
        print(c(YLW, "\n  Gateway closed the connection."))
    except Exception as e:
        print(c(RED, f"\n  {e}"))
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        _stop.set()
        print(c(YLW, "\n\n  Stopped.\n"))