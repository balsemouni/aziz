"""
test_ws_latency.py — STT Microservice WebSocket Latency Tester
──────────────────────────────────────────────────────────────
Measures what actually matters:

  ┌─────────────────────────────────────────────────────┐
  │  PROCESSING LATENCY                                 │
  │  = time from your LAST WORD → result appears        │
  │                                                     │
  │  You:    "hello how are you ..."                    │
  │                              ↑                      │
  │                         last chunk sent             │
  │                              │── processing ──▶     │
  │  Server:                    "hello how are you"     │
  └─────────────────────────────────────────────────────┘

Usage
─────
    python test_ws_latency.py
    python test_ws_latency.py --url ws://10.0.0.5:8001
    python test_ws_latency.py --chunk-ms 20 --rms-threshold 0.004

Requirements
────────────
    pip install sounddevice websockets numpy
"""

import argparse
import asyncio
import json
import sys
import time
import numpy as np

# ── Dependency check ──────────────────────────────────────────────────────────
missing = []
try:    import sounddevice as sd
except ImportError: missing.append("sounddevice")
try:    import websockets
except ImportError: missing.append("websockets")

if missing:
    print(f"❌  Missing: pip install {' '.join(missing)}")
    sys.exit(1)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--url",           default="ws://localhost:8001/stream/mux")
parser.add_argument("--chunk-ms",      type=int,   default=20)
parser.add_argument("--rate",          type=int,   default=16000)
parser.add_argument("--rms-threshold", type=float, default=0.004,
                    help="RMS level to consider as voice (default 0.004)")
args = parser.parse_args()

WS_URL        = args.url
SAMPLE_RATE   = args.rate
CHUNK_MS      = args.chunk_ms
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
RMS_THRESH    = args.rms_threshold

# ── Colours ───────────────────────────────────────────────────────────────────
R = "\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
GREEN="\033[92m"; CYAN="\033[96m"; YELLOW="\033[93m"; RED="\033[91m"; BLUE="\033[94m"
def c(col, t): return f"{col}{t}{R}"

# ── Shared state ──────────────────────────────────────────────────────────────
audio_queue: asyncio.Queue
stop_event = asyncio.Event()
_loop: asyncio.AbstractEventLoop | None = None

# Timestamp of the most recent voice chunk sent this utterance.
# NOT reset between bursts — only reset after a segment arrives.
# This ensures multi-burst sentences measure from the true last word.
_last_voice_chunk_ts: float | None = None
_segment_received = False   # set by receiver; tells sender to allow ts reset

# Session stats
session_segments: list[dict] = []
session_words:    list[str]  = []


# ── Audio frame builder ───────────────────────────────────────────────────────
def _make_frame(pcm: np.ndarray) -> bytes:
    i16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    return b"\x01" + i16.tobytes()


# ── Mic callback ──────────────────────────────────────────────────────────────
def _mic_callback(indata, frames, time_info, status):
    if status:
        print(c(YELLOW, f"[mic] {status}"), flush=True)
    chunk = indata[:, 0].copy().astype(np.float32)
    if _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(audio_queue.put(chunk), _loop)


# ── Sender ────────────────────────────────────────────────────────────────────
async def _sender(ws):
    global _last_voice_chunk_ts, _segment_received

    _in_voice      = False
    _silence_count = 0
    SILENCE_END    = 3   # consecutive silent chunks before flagging end of speech

    while not stop_event.is_set():
        try:
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue

        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-10))
        now = time.monotonic()

        if rms > RMS_THRESH:
            if not _in_voice:
                _in_voice      = True
                _silence_count = 0
                # Only print "speaking" indicator; do NOT reset _last_voice_chunk_ts
                # here unless a segment was already received (new sentence)
                if _segment_received:
                    _last_voice_chunk_ts = None   # fresh sentence
                    _segment_received    = False
                print(c(DIM, "\n  🎤"), end="", flush=True)

            # Always update — this is the timestamp of the last voice chunk
            _last_voice_chunk_ts = now
            _silence_count       = 0

        else:
            if _in_voice:
                _silence_count += 1
                if _silence_count >= SILENCE_END:
                    _in_voice = False
                    print(c(DIM, " ⏸ waiting…"), flush=True)

        # Always send all chunks — server needs silence to detect end-of-utterance
        try:
            await ws.send(_make_frame(chunk))
        except Exception:
            break


# ── Receiver ──────────────────────────────────────────────────────────────────
async def _receiver(ws):
    global _segment_received
    current_words:  list[str] = []
    _enrolled       = False
    _last_pct       = -1
    _rejected_count = 0

    async for raw in ws:
        now = time.monotonic()
        try:
            ev = json.loads(raw)
        except Exception:
            continue

        etype = ev.get("type")

        # ── enrollment ───────────────────────────────────────────────────────
        # Server fires this every ~1% progress.
        # While enrolling: words and segments are blocked server-side — only
        # enrollment events arrive.  Once enrolled=True, ASR opens up.
        if etype == "enrollment":
            enrolled = ev.get("enrolled", False)
            progress = ev.get("progress", 0.0)
            pct      = int(progress * 100)

            if enrolled and not _enrolled:
                _enrolled = True
                bar = c(GREEN, "█" * 20)
                print(f"\r  🔐 Enrolling  [{bar}] 100%", flush=True)
                print(c(GREEN + BOLD, "\n  ✅ Voice enrolled — ASR is now LIVE."))
                print(c(DIM,          "  Only your voice will be transcribed. Others are blocked.\n"))
                print("─" * 52)
            elif not enrolled and pct != _last_pct:
                _last_pct = pct
                filled    = int(20 * progress)
                bar       = c(CYAN, "█" * filled) + c(DIM, "░" * (20 - filled))
                print(f"\r  🔐 Enrolling  [{bar}] {pct}%   ", end="", flush=True)
            continue

        # ── segment_rejected ─────────────────────────────────────────────────
        if etype == "segment_rejected":
            _rejected_count += 1
            sim     = ev.get("similarity")
            sim_str = f"  sim={sim:.2f}" if sim is not None else ""
            print(c(YELLOW, f"\n  🚫 Blocked — not your voice{sim_str}  (#{_rejected_count} total blocked)"), flush=True)
            continue

        # ── word ─────────────────────────────────────────────────────────────
        # Only reaches here after enrollment is confirmed server-side.
        if etype == "word":
            word = ev.get("word", "").strip()
            if not word:
                continue
            current_words.append(word)
            session_words.append(word)
            print(c(GREEN, f"  {word}"), end=" ", flush=True)

        # ── segment ──────────────────────────────────────────────────────────
        elif etype == "segment":
            text = ev.get("text", "").strip()
            if not text:
                continue

            proc_ms: float | None = None
            if _last_voice_chunk_ts is not None:
                proc_ms = (now - _last_voice_chunk_ts) * 1000

            print()
            print(c(BOLD + GREEN, f'\n  ✅ "{text}"'))

            if proc_ms is not None:
                col    = GREEN if proc_ms < 400 else (YELLOW if proc_ms < 800 else RED)
                avg_ms = (
                    (sum(s["proc_ms"] for s in session_segments) + proc_ms)
                    / (len(session_segments) + 1)
                )
                print(
                    f"\n  {c(DIM, 'processing latency')}  "
                    f"{c(col + BOLD, f'{proc_ms:.0f} ms')}"
                    f"  {c(DIM, '← last word → result')}"
                )
                print(
                    f"  {c(DIM, 'running average   ')}  "
                    f"{c(BLUE, f'{avg_ms:.0f} ms')}  "
                    f"{c(DIM, f'({len(session_segments)+1} sentence(s))')}"
                )
                session_segments.append({
                    "text":    text,
                    "proc_ms": proc_ms,
                    "words":   list(current_words),
                })

            _segment_received = True
            print("\n" + "─" * 52)
            current_words.clear()

        elif etype == "partial":
            pass

        elif etype == "error":
            print(c(RED, f"\n  ❌ Server: {ev.get('detail')}"))


# ── Summary ───────────────────────────────────────────────────────────────────
def _print_summary():
    print(c(CYAN, "\n══════════════════════════════════════════════"))
    print(c(BOLD,  "   Session Summary"))
    print(c(CYAN, "══════════════════════════════════════════════"))
    print(f"  Words transcribed : {len(session_words)}")
    if session_words:
        print(f"  Full transcript   : {' '.join(session_words)}")

    if not session_segments:
        print("  No sentences completed.")
        print(c(CYAN, "══════════════════════════════════════════════\n"))
        return

    lats = [s["proc_ms"] for s in session_segments]
    print(f"\n  Sentences : {len(lats)}")
    print(f"\n  ── Processing latency (last word → result) ──")
    print(f"     Min  : {min(lats):.0f} ms")
    print(f"     Max  : {max(lats):.0f} ms")
    print(f"     Avg  : {sum(lats)/len(lats):.0f} ms")

    buckets = [
        ("< 200ms",   lambda l: l < 200,          GREEN),
        ("200–400ms", lambda l: 200 <= l < 400,   GREEN),
        ("400–800ms", lambda l: 400 <= l < 800,   YELLOW),
        ("> 800ms",   lambda l: l >= 800,          RED),
    ]
    print(f"\n  ── Distribution ──")
    for label, fn, col in buckets:
        cnt = sum(1 for l in lats if fn(l))
        bar = "█" * cnt
        print(f"    {label:>12}  {c(col, bar or '·')} {cnt}")

    print(f"\n  ── Per-sentence breakdown ──")
    print(f"  {'#':>3}  {'Latency':>10}  Text")
    for i, s in enumerate(session_segments, 1):
        col = GREEN if s["proc_ms"] < 400 else (YELLOW if s["proc_ms"] < 800 else RED)
        lat_str = c(col, f"{s['proc_ms']:.0f} ms")
        print(f"  {i:>3}  {lat_str:>10}  {s['text'][:55]}")

    print(c(CYAN, "\n══════════════════════════════════════════════\n"))


# ── Main ──────────────────────────────────────────────────────────────────────
async def _main():
    global audio_queue, _loop
    audio_queue = asyncio.Queue(maxsize=200)
    _loop       = asyncio.get_running_loop()

    print(c(CYAN,  "\n══════════════════════════════════════════════"))
    print(c(BOLD,   "   STT Microservice — Latency Tester"))
    print(c(CYAN,  "══════════════════════════════════════════════"))
    print(f"  Server : {WS_URL}")
    print(f"  Chunk  : {CHUNK_MS} ms  |  Rate: {SAMPLE_RATE} Hz  |  RMS thresh: {RMS_THRESH}")
    print(c(YELLOW, "\n  HOW IT WORKS:"))
    print(c(DIM,    "  1. Speak for ~2s → your voice profile is built"))
    print(c(DIM,    "  2. ASR turns ON — only your voice gets transcribed"))
    print(c(DIM,    "  3. Anyone else speaking is silently blocked"))
    print(c(DIM,   "\n  Connecting…"))

    try:
        async with websockets.connect(
            WS_URL,
            ping_interval = 20,
            ping_timeout  = 20,
            max_size      = 2**20,
        ) as ws:
            print(c(GREEN, "  ✅ Connected\n"))
            print(c(YELLOW, "  🎙  Start speaking — building your voice profile…\n"))
            print("─" * 52)

            with sd.InputStream(
                samplerate = SAMPLE_RATE,
                channels   = 1,
                dtype      = "float32",
                blocksize  = CHUNK_SAMPLES,
                callback   = _mic_callback,
            ):
                print(c(GREEN, "  🎙  Mic open — listening…\n"))
                await asyncio.gather(_sender(ws), _receiver(ws))

    except ConnectionRefusedError:
        print(c(RED, f"\n  ❌  Connection refused at {WS_URL}"))
        print(c(DIM,  "      Start server:  uvicorn main:app --port 8001"))
        sys.exit(1)
    except Exception as e:
        print(c(RED, f"\n  ❌  {e}"))
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        stop_event.set()
        print(c(YELLOW, "\n\n  Stopping…"))
        _print_summary()