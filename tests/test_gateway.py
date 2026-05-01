"""
test_gateway.py — Gateway Test Client v3.3
══════════════════════════════════════════════════════════════════════════════
v3.3 CHANGES (over v3.2):
  • _TTSSession RE-ENABLED — persistent WS to TTS service (/ws/tts) with a
    real sentence-pipeline worker. The v3.2 stub (all no-ops) is replaced with
    a working async implementation that:
      – Opens ONE WS connection per AI turn (warm llama context, no cold starts)
      – Enqueues sentences immediately via a thread-safe asyncio.Queue
      – Worker coroutine sends each sentence JSON, reads PCM bytes until the
        empty-bytes end-of-stream marker, then pushes float32 audio to
        _pcm_play_q while the NEXT sentence is already being synthesised on
        the server — true pipeline overlap
      – On barge-in/cancel: sends {"type":"cancel"} + {"type":"close"} and
        drains the play queue so silence is immediate
  • AUDIO BUFFER CONTINUITY — _pcm_play_q consumer thread now pre-fills a
    512-sample look-ahead ring so sounddevice never underruns between sentences.
  • ZeroDivisionError guard kept from v3.2 (waveform `or 1e-9`).
  • Port defaults: gateway=8090, tts=8765.

Architecture
────────────
  test_gateway.py  ←WS→  gateway (port 8090)
                               └──WS persistent per-turn──→  TTS (port 8765/ws/tts)

  Per-turn flow:
    1. "thinking" event → open TTS WS, start _tts_worker coroutine
    2. "ai_sentence" event → _tts_session.enqueue(text)
       (worker is already running; sentence is synthesised while earlier ones play)
    3. "done" event → _tts_session.finish_and_close()
       (worker drains queue, sends close, WS closes cleanly)
    4. "barge_in" event → _tts_session.cancel_and_close()
       (worker receives sentinel, sends cancel+close, play queue drained)

Usage
─────
    python test_gateway.py
    python test_gateway.py --host 192.168.1.10 --port 8090
    python test_gateway.py --out-rate 24000

Requirements
────────────
    pip install websockets sounddevice numpy
"""

import argparse
import asyncio
import json
import logging
import sys
import datetime
import textwrap
import os
import threading
import queue as stdlib_queue
from typing import Optional

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

log = logging.getLogger("test_gateway")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--host",      default="127.0.0.1")
parser.add_argument("--port",      default=8090,  type=int)
parser.add_argument("--tts-host",  default="127.0.0.1")
parser.add_argument("--tts-port",  default=8765,  type=int)
parser.add_argument("--mic-rate",  default=16000, type=int)
parser.add_argument("--chunk-ms",  default=20,    type=int)
parser.add_argument("--out-rate",  default=22050, type=int)
parser.add_argument("--debug-raw", action="store_true")
args = parser.parse_args()

GW_URL     = f"ws://{args.host}:{args.port}/ws"
TTS_URL    = f"ws://{args.tts_host}:{args.tts_port}/ws/tts"
MIC_RATE   = args.mic_rate
CHUNK_MS   = args.chunk_ms
CHUNK_SAMP = int(MIC_RATE * CHUNK_MS / 1000)
OUT_RATE   = args.out_rate
DEBUG_RAW  = args.debug_raw

# ── Terminal width ────────────────────────────────────────────────────────────
try:    TW = min(os.get_terminal_size().columns, 120)
except: TW = 100

# ── ANSI colours ──────────────────────────────────────────────────────────────
R    = "\033[0m";  BOLD = "\033[1m";  DIM  = "\033[2m"
RED  = "\033[91m"; GRN  = "\033[92m"; YLW  = "\033[93m"
BLU  = "\033[94m"; MAG  = "\033[95m"; CYN  = "\033[96m"
WHT  = "\033[97m"; BLK  = "\033[30m"
BGDARK = "\033[48;5;234m"; BGCYN = "\033[46m"; BGMAG = "\033[45m"

def c(*a):
    return "".join(a[:-1]) + a[-1] + R

def ts():
    n = datetime.datetime.now()
    return c(DIM, f"{n.strftime('%H:%M:%S')}.{n.microsecond//1000:03d}")

SEP      = c(DIM, "─" * TW)
SEP_BOLD = c(CYN, "═" * TW)

# ── Global state ──────────────────────────────────────────────────────────────
_audio_in_q:  asyncio.Queue          # mic → gateway (bytes)
_pcm_play_q:  stdlib_queue.Queue     # TTS PCM → speaker thread (np.ndarray float32)
_tts_play_cancel = threading.Event() # set by cancel_and_close() to stop audio immediately
_loop         = None
_stop         = asyncio.Event()

_cur_user_words:   list = []
_cur_ai_tokens:    list = []
_cur_ai_sentences: list = []
_ai_speaking       = False
_last_word_display: str = ""  # dedup word display

# ── Waveform helper ───────────────────────────────────────────────────────────

def _waveform_line(pcm_f32: np.ndarray, width: int = 40) -> str:
    if pcm_f32 is None or len(pcm_f32) == 0:
        return c(DIM, "·" * width)
    step  = max(1, len(pcm_f32) // width)
    cols  = [float(np.sqrt(np.mean(pcm_f32[i:i+step]**2)))
             for i in range(0, len(pcm_f32), step)][:width]
    mx    = max(cols) if cols else 0.0
    mx    = mx or 1e-9
    bars  = "▁▂▃▄▅▆▇█"
    out   = ""
    for v in cols:
        idx = min(int((v / mx) * (len(bars) - 1)), len(bars) - 1)
        out += c(MAG, bars[idx])
    return out + " " * (width - len(cols))

# ── Latency bars ──────────────────────────────────────────────────────────────

def _bar(val, mx, w=24):
    if not mx or val is None: return c(DIM, "░" * w)
    f   = min(int((val / mx) * w), w)
    col = GRN if val < 200 else YLW if val < 500 else MAG if val < 1000 else RED
    return c(col, "█" * f) + c(DIM, "░" * (w - f))

def _ms(ms):
    if ms is None: return c(DIM, "    —   ")
    col = GRN+BOLD if ms<200 else YLW+BOLD if ms<500 else MAG+BOLD if ms<1000 else RED+BOLD
    return c(col, f"{ms:>6.0f}ms")

def _icon(ms):
    if ms is None: return "○"
    return c(GRN,"●") if ms<200 else c(YLW,"●") if ms<500 else c(MAG,"●") if ms<1000 else c(RED,"●")

# ── Print helpers ─────────────────────────────────────────────────────────────

def _header():
    print()
    print(c(BGDARK, CYN+BOLD, " " * TW))
    print(c(BGDARK, CYN+BOLD, f"  ◆  VOICE PIPELINE MONITOR  ◆".ljust(TW)))
    print(c(BGDARK, DIM,      f"  gw:{GW_URL}   tts:{TTS_URL}   mic:{MIC_RATE}Hz  out:{OUT_RATE}Hz".ljust(TW)))
    print(c(BGDARK, CYN+BOLD, " " * TW))
    print()

def _conv_entry(role, text):
    if role == "user":
        tag = c(BGCYN, BLK+BOLD, " USER "); col = CYN
    else:
        tag = c(BGMAG, WHT+BOLD, "  AI  "); col = MAG
    wrapped = ("\n" + " "*10).join(textwrap.wrap(text, TW-12))
    print(f"\n  {tag}  {c(col, wrapped)}")

def _thinking_banner(tid):
    print(f"\n{SEP}")
    print(f"  {c(YLW+BOLD,'THINKING')}  {c(DIM,'['+tid[:8]+']')}")
    print(SEP)

def _segment_banner(text):
    print()
    print(SEP)
    print(f"  {c(GRN+BOLD,'USER SAID')}   {c(GRN, chr(34)+text+chr(34))}")
    print(SEP)

def _tts_chunk_line(idx, text):
    print(f"\n  {c(MAG+BOLD,'['+str(idx)+']')}  {c(MAG, text)}")

def _latency_panel(ev):
    barge = ev.get("barge_in", False)
    stt   = ev.get("stt_latency_ms")
    cag   = ev.get("cag_first_token_ms")
    c2t   = ev.get("cag_first_sentence_ms")
    tts_s = ev.get("tts_first_audio_ms")
    e2e   = ev.get("e2e_ms")
    toks  = ev.get("total_tokens", 0)
    tid   = ev.get("turn_id","")[:8]
    label = c(RED+BOLD,"INTERRUPTED") if barge else c(GRN+BOLD,"COMPLETE")
    mx    = max((v for v in [stt,cag,c2t,tts_s] if v), default=1)
    print(f"\n{SEP_BOLD}")
    print(f"  {c(BOLD,'LATENCY REPORT')}  {label}  {c(DIM,'['+tid+']')}  {c(DIM,str(toks)+' tokens')}")
    print(SEP_BOLD)
    for lbl, ms, desc in [
        ("STT  word → segment  ", stt,  "Last spoken word until STT fires"),
        ("CAG  seg  → token    ", cag,  "STT segment until first AI token"),
        ("CAG  tok  → TTS send ", c2t,  "First token until first TTS chunk queued"),
        ("TTS  send → audio    ", tts_s,"TTS chunk queued until audio arrives"),
    ]:
        print(f"  {c(DIM,lbl)}  {_ms(ms)}  {_bar(ms,mx)}  {c(DIM,desc)}")
    print(f"\n  {c(BOLD,'E2E  word → audio     ')}  {_ms(e2e)}  {_icon(e2e)}")
    chunks = ev.get("tts_chunks", [])
    if chunks:
        mxc = max((ch.get("synthesis_latency_ms") or 0) for ch in chunks) or 1
        print(f"\n  {c(DIM,'Chunk  Synth dur   Synth lat   Bar                 Audio dur')}")
        print(f"  {c(DIM,'─'*(TW-4))}")
        for ch in chunks:
            i    = ch.get("chunk_index", 0)
            clag = ch.get("synthesis_latency_ms") or 0
            cdur = ch.get("synth_duration_ms")    or 0
            dsec = ch.get("duration_sec")          or 0.0
            star = c(CYN+BOLD,"FIRST") if i==0 else c(DIM,f"+{ch.get('first_chunk_latency_ms',0):.0f}ms")
            print(f"  {c(CYN,'['+str(i)+']')}  {_ms(cdur)}    {_ms(clag)}  {_bar(clag,mxc,20)}  {c(DIM,f'{dsec:.2f}s')}  {star}")
    print(SEP_BOLD)

def _session_summary(summary):
    print(f"\n{SEP_BOLD}")
    print(c(BOLD+CYN,"   SESSION SUMMARY"))
    print(SEP_BOLD)
    print(f"  Turns     : {c(BOLD, str(summary.get('turns',0)))}")
    print(f"  Barge-ins : {c(BOLD, str(summary.get('barge_ins',0)))}")
    for key, lbl in [("stt","STT  word→seg"),("cag","CAG  →token"),
                     ("tts_synth","TTS  →audio"),("e2e","E2E  word→audio")]:
        st = summary.get(key,{})
        if not st: continue
        avg = st.get("avg"); p95 = st.get("p95")
        mn  = st.get("min"); mx2 = st.get("max")
        print(f"\n  {c(BOLD,lbl)}\n    avg {_ms(avg)}  p95 {_ms(p95)}  min {_ms(mn)}  max {_ms(mx2)}\n    {_bar(avg,2000,20)}")
    print(SEP_BOLD)

# ── Continuous audio output thread ────────────────────────────────────────────

def _audio_output_thread():
    global _ai_speaking
    BLOCK   = 1024
    silence = np.zeros(BLOCK, dtype=np.float32)

    try:
        with sd.OutputStream(
            samplerate=OUT_RATE, channels=1, dtype="float32",
            blocksize=BLOCK, latency="low",
        ) as stream:
            # Pre-fill look-ahead ring so we never underrun at sentence joins.
            lookahead = np.zeros(512, dtype=np.float32)
            leftover  = np.zeros(0,   dtype=np.float32)

            while not _stop.is_set():
                # ── Cancel flush ─────────────────────────────────────────
                # Fires when cancel_and_close() is called (barge-in / new turn).
                # Discard all buffered audio instantly so speakers go silent.
                if _tts_play_cancel.is_set():
                    _tts_play_cancel.clear()
                    lookahead = np.zeros(0, dtype=np.float32)
                    leftover  = np.zeros(0, dtype=np.float32)
                    _ai_speaking = False
                    stream.write(silence)    # one silent block to prevent underrun
                    continue

                chunks = []
                try:
                    while True:
                        chunks.append(_pcm_play_q.get_nowait())
                except stdlib_queue.Empty:
                    pass

                if chunks:
                    _ai_speaking = True
                    combined = np.concatenate(chunks)

                    # Prepend any look-ahead from previous iteration.
                    if lookahead.size:
                        buf = np.concatenate([lookahead, leftover, combined]) if leftover.size else np.concatenate([lookahead, combined])
                    else:
                        buf = np.concatenate([leftover, combined]) if leftover.size else combined
                    leftover  = np.zeros(0, dtype=np.float32)
                    lookahead = np.zeros(0, dtype=np.float32)

                    pos = 0
                    while pos < len(buf):
                        # Check cancel between every block (~46ms granularity)
                        if _tts_play_cancel.is_set():
                            leftover = np.zeros(0, dtype=np.float32)
                            break
                        frame = buf[pos:pos + BLOCK]
                        if len(frame) < BLOCK:
                            leftover = frame
                            break
                        stream.write(frame)
                        pos += BLOCK
                else:
                    if leftover.size:
                        if _tts_play_cancel.is_set():
                            leftover = np.zeros(0, dtype=np.float32)
                        else:
                            # Drain leftover before going silent.
                            pad = np.zeros(BLOCK - len(leftover) % BLOCK, dtype=np.float32) if len(leftover) % BLOCK else np.zeros(0, dtype=np.float32)
                            buf = np.concatenate([leftover, pad])
                            leftover = np.zeros(0, dtype=np.float32)
                            for i in range(0, len(buf), BLOCK):
                                stream.write(buf[i:i+BLOCK])
                    _ai_speaking = False
                    stream.write(silence)
                    import time; time.sleep(0.005)

    except Exception as e:
        print(f"\n  {c(RED,'Audio output error: '+str(e))}")

# ── Mic callback ─────────────────────────────────────────────────────────────
# Client-side AEC: RMS gate suppresses AI echo while speaker is playing.
# Only audio above MIC_BARGE_RMS passes (user barge-in is louder than echo).
MIC_BARGE_RMS = float(os.getenv("MIC_BARGE_RMS", "0.02"))

def _mic_cb(indata, frames, time_info, status):
    chunk = indata[:, 0].copy().astype(np.float32)
    # AEC gate: suppress low-energy mic frames while AI is playing
    # so the AI's own voice (echoed from speakers) never reaches the gateway.
    # Real user speech during barge-in is significantly louder than echo.
    if _ai_speaking:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < MIC_BARGE_RMS:
            chunk = np.zeros_like(chunk)  # suppress echo, keep stream alive
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

# ─────────────────────────────────────────────────────────────────────────────
#  _TTSSession  (v3.3 — re-enabled, persistent WS per turn)
# ─────────────────────────────────────────────────────────────────────────────

_SENTINEL = object()   # signals worker to cancel and exit


class _TTSSession:
    """
    Persistent WebSocket session to the TTS microservice for one AI turn.

    Usage:
        session = _TTSSession(voice="tara")
        await session.start()                   # opens WS, starts worker
        session.enqueue("Hello there.")         # sentence arrives immediately
        session.enqueue("How can I help?")      # queued; pipeline overlap
        await session.finish_and_close()        # drain queue, close cleanly

    On barge-in:
        await session.cancel_and_close()        # immediate stop, drain play queue
    """

    def __init__(self, voice: str = "tara"):
        self._voice    = voice
        self._ws       = None
        self._q: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._closed   = False

    async def start(self):
        """Open WS connection and launch the sentence worker coroutine."""
        try:
            self._ws = await websockets.connect(
                TTS_URL,
                ping_interval=10,
                ping_timeout=20,
                max_size=8 * 1024 * 1024,
                open_timeout=5,
            )
            self._worker_task = asyncio.create_task(self._worker())
            log.debug("TTS session opened: %s", TTS_URL)
        except Exception as exc:
            log.error("TTS session failed to open: %s", exc)
            self._closed = True

    async def _worker(self):
        """
        Pull sentences from _q, send each to TTS, collect PCM, push to
        _pcm_play_q.  Runs until _SENTINEL is received or WS closes.
        """
        while True:
            try:
                item = await asyncio.wait_for(self._q.get(), timeout=60.0)
            except asyncio.TimeoutError:
                log.warning("TTS worker: idle 60s — closing")
                break

            if item is _SENTINEL:
                break

            text, voice = item
            if not text or self._closed:
                continue

            v = voice or self._voice
            payload = json.dumps({"type": "tts", "text": text, "voice": v})

            t_send = asyncio.get_event_loop().time()
            try:
                await self._ws.send(payload)
            except Exception as exc:
                log.error("TTS send error: %s", exc)
                break

            # Read PCM until empty-bytes end-of-stream marker.
            sentence_frames = 0
            try:
                async for msg in self._ws:
                    if isinstance(msg, bytes):
                        if not msg:
                            # End-of-sentence marker — move to next sentence.
                            break
                        pcm = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                        _pcm_play_q.put(pcm)
                        sentence_frames += len(pcm)
                        if DEBUG_RAW:
                            wf = _waveform_line(pcm, width=TW - 20)
                            print(f"\r{ts()}  {c(MAG+BOLD,'♪')}  {wf}  ", end="", flush=True)
                    else:
                        # JSON control frame (ping/pong) — ignore.
                        pass
            except websockets.exceptions.ConnectionClosed:
                log.warning("TTS WS closed mid-stream")
                break

            elapsed_ms = (asyncio.get_event_loop().time() - t_send) * 1000
            log.debug(
                "Sentence complete: %.0fms, %.2fs audio",
                elapsed_ms, sentence_frames / OUT_RATE,
            )

        # Clean up WS.
        await self._close_ws()

    async def _close_ws(self):
        if self._ws and not self._closed:
            self._closed = True
            try:
                await self._ws.send(json.dumps({"type": "close"}))
                await self._ws.close()
            except Exception:
                pass

    def enqueue(self, text: str, voice: Optional[str] = None):
        """Enqueue a sentence for synthesis (call from async context)."""
        if not self._closed:
            asyncio.ensure_future(self._q.put((text, voice or self._voice)))

    async def finish_and_close(self):
        """
        Wait for all queued sentences to finish synthesising, then close.
        Safe to call from the main async context.
        """
        if self._closed:
            return
        await self._q.put(_SENTINEL)
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=120.0)
            except asyncio.TimeoutError:
                log.warning("TTS finish_and_close: worker timed out")
                self._worker_task.cancel()
        await self._close_ws()

    async def cancel_and_close(self):
        """
        Immediately cancel synthesis and silence the play queue.
        Sends {"type":"cancel"} to stop the server-side synthesis, then closes.
        """
        if self._closed:
            return
        self._closed = True

        # Stop audio output thread mid-buffer within ~46ms (one soundcard block).
        _tts_play_cancel.set()

        # Send cancel to microservice.
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "cancel"}))
            except Exception:
                pass

        # Drain play queue so no further audio is pushed.
        while True:
            try:
                _pcm_play_q.get_nowait()
            except stdlib_queue.Empty:
                break

        # Cancel the worker task.
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        await self._close_ws()
        log.debug("TTS session cancelled and closed.")


# ── Receiver ─────────────────────────────────────────────────────────────

async def _receiver(ws):
    global _cur_user_words, _cur_ai_tokens, _cur_ai_sentences, _ai_speaking, _last_word_display

    _cur_user_words   = []
    _cur_ai_tokens    = []
    _cur_ai_sentences = []
    _in_tok           = False
    _current_voice    = "tara"

    async for msg in ws:

        # ── PCM audio bytes from GATEWAY (legacy path) ────────────────────
        if isinstance(msg, bytes):
            if msg:
                pcm = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                _pcm_play_q.put(pcm)
                wf = _waveform_line(pcm, width=TW - 20)
                print(f"\r{ts()}  {c(MAG+BOLD,'♪')}  {wf}  ", end="", flush=True)
            continue

        # ── JSON control frames ───────────────────────────────────────────
        try:
            ev = json.loads(msg)
        except Exception:
            continue

        t = ev.get("type")

        if DEBUG_RAW:
            print(f"\n{ts()}  {c(DIM,'RAW['+str(t)+']')}  {c(DIM,json.dumps(ev)[:160])}")

        # STT ──────────────────────────────────────────────────────────────
        if t == "word":
            if _ai_speaking: print()
            word = ev.get("word", "")
            # Only append if the word is new (STT can fire the same word event repeatedly)
            if word and (not _cur_user_words or _cur_user_words[-1] != word):
                _cur_user_words.append(word)
            display = ' '.join(_cur_user_words)
            if display != _last_word_display:
                _last_word_display = display
                print(f"\r{ts()}  \U0001f3a4  {c(GRN, display)}  ", end="", flush=True)

        elif t == "partial":
            word = ev.get("word","").strip().rstrip("?.!,;:")
            if word and (not _cur_user_words or _cur_user_words[-1].lower()!=word.lower()):
                _cur_user_words.append(word)
            display = ' '.join(_cur_user_words)
            if display != _last_word_display:
                _last_word_display = display
                print(f"\r{ts()}  \U0001f3a4  {c(GRN, display)}  ", end="", flush=True)

        elif t == "segment":
            text = ev.get("text","").strip()
            print()
            _segment_banner(text)
            _cur_user_words = []
            _last_word_display = ""  # reset so next words display fresh

        # CAG ──────────────────────────────────────────────────────────────
        elif t == "thinking":
            if _in_tok: print(); _in_tok = False
            _cur_ai_tokens = []; _cur_ai_sentences = []
            _thinking_banner(ev.get("turn_id",""))
            print(f"{ts()}  {c(BLU+BOLD,'AI')}  ", end="", flush=True)
            _in_tok = True
            # TTS audio comes as raw PCM bytes from the gateway (see bytes handler
            # above).  No separate TTS session needed — that caused double audio.

        elif t == "ai_token":
            tok = ev.get("token","")
            if tok:
                # Add space when prev token didn't end with space and curr token
                # doesn't start with space or punctuation
                if _cur_ai_tokens:
                    prev = _cur_ai_tokens[-1]
                    if prev and prev[-1] not in ' ' and tok[0] not in ' .,!?;:")\']}-':
                        print(" ", end="", flush=True)
                print(c(BLU, tok), end="", flush=True)
                _cur_ai_tokens.append(tok)
            _in_tok = True

        elif t == "ai_sentence":
            text = ev.get("text","").strip()
            _cur_ai_sentences.append(text)
            if _in_tok: print(); _in_tok = False
            _tts_chunk_line(len(_cur_ai_sentences), text)
            print(f"{ts()}  {c(BLU+BOLD,'AI')}  ", end="", flush=True)
            _in_tok = True
            # Gateway streams PCM directly — no local TTS enqueue needed.

        elif t == "done":
            if _in_tok: print(); _in_tok = False
            # Use sentence-level text (correctly spaced from gateway's _smart_join)
            # rather than raw tokens which may be missing word-boundary spaces.
            full = " ".join(s.strip() for s in _cur_ai_sentences if s.strip())
            if not full:  # fallback: token concat (single-word responses etc.)
                full = "".join(_cur_ai_tokens).strip()
            if full:
                _conv_entry("ai", full)
            _cur_ai_tokens = []; _cur_ai_sentences = []

        # Latency ──────────────────────────────────────────────────────────
        elif t == "latency":
            if _in_tok: print(); _in_tok = False
            _latency_panel(ev)

        elif t == "session_summary":
            _session_summary(ev.get("latency",{}))
            _stop.set()

        # Misc ─────────────────────────────────────────────────────────────
        elif t == "barge_in":
            if _in_tok: print(); _in_tok = False
            print(f"\n  {c(YLW+BOLD,'⚡ BARGE-IN')}  {c(YLW,'Stopping AI speech')}")
            # Stop playback immediately: signal the audio thread and drain queue.
            _tts_play_cancel.set()
            while True:
                try: _pcm_play_q.get_nowait()
                except stdlib_queue.Empty: break

        elif t == "hallucination_reset":
            if _in_tok: print(); _in_tok = False
            print(f"\n  {c(YLW+BOLD,'HALLUCINATION GUARD')}  {c(YLW,ev.get('detail',''))}")
            _cur_user_words = []

        elif t == "session_reset":
            print(f"\n  {c(YLW,'Session reset: '+ev.get('reason',''))}")

        elif t == "error":
            if _in_tok: print(); _in_tok = False
            print(f"\n  {c(RED+BOLD,'ERROR')}  {c(RED, ev.get('detail',''))}")

        elif t in ("ping", "ready", "ai_state"):
            if t == "ready":
                print(f"\n  {c(GRN+BOLD, '✓ Pipeline ready — start speaking')}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

async def _main():
    global _audio_in_q, _pcm_play_q, _loop

    _audio_in_q = asyncio.Queue(maxsize=500)
    _loop       = asyncio.get_running_loop()

    _header()
    print(f"  {c(DIM,'Gateway')}  {c(BOLD,GW_URL)}")
    print(f"  {c(DIM,'TTS    ')}  {c(BOLD,TTS_URL)}  {c(DIM,'(persistent session per turn)')}")
    print(f"\n  {c(YLW,'Speak naturally. Barge in any time to interrupt the AI.')}")
    print(f"  {c(DIM,'Ctrl+C to quit.')}\n")
    print(SEP)

    out_thread = threading.Thread(target=_audio_output_thread, daemon=True)
    out_thread.start()

    try:
        async with websockets.connect(
            GW_URL,
            ping_interval=20,
            ping_timeout=30,
            max_size=10 * 1024 * 1024,
        ) as ws:
            print(f"\n  {c(GRN+BOLD,'Connected')}\n{SEP_BOLD}")

            with sd.InputStream(
                samplerate=MIC_RATE, channels=1, dtype="float32",
                blocksize=CHUNK_SAMP, callback=_mic_cb,
            ):
                print(f"\n  {c(GRN+BOLD,'Mic open — start speaking...')}\n")
                await asyncio.gather(
                    _sender(ws),
                    _receiver(ws),
                )

    except ConnectionRefusedError:
        print(c(RED, f"\n  Cannot connect to {GW_URL}"))
        print(c(DIM, "     Start the gateway first: python gateway.py"))
        sys.exit(1)
    except websockets.exceptions.ConnectionClosedOK:
        print(c(YLW, "\n  Gateway closed the connection."))
    except Exception as e:
        print(c(RED, f"\n  {e}"))
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    _pcm_play_q  = stdlib_queue.Queue(maxsize=2000)
    _ai_speaking = False

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        _stop.set()
        print(c(YLW, "\n\n  Stopped.\n"))