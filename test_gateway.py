"""
test_gateway.py — Gateway Test Client v3.2
══════════════════════════════════════════════════════════════════════════════
v3.2 CHANGES:
  • Connects directly to TTS microservice (/ws/tts) via a PERSISTENT SESSION
    for the duration of each AI turn.  All sentences from one turn share ONE
    WebSocket connection → warm llama context → natural, gapless speech.
  • Removed the per-sentence new-connection pattern that caused:
      – Cold llama context on every sentence  (→ robotic audio)
      – Queue serialisation across sentences  (→ ~100s latency)
  • chunk_tone now always sent as "auto" — microservice v5.4 auto-detects
    the correct type from the text, ignoring caller hint.
  • ZeroDivisionError fix: waveform mx guarded with `or 1e-9`.
  • Port default changed to 8090 (gateway) but TTS_URL points to 8765.

Architecture
────────────
  test_gateway.py  ←WS→  gateway (port 8090)
                              └──WS persistent session──→  TTS (port 8765/ws/tts)

  When the gateway receives AI sentences it opens ONE WS to the TTS service,
  sends each sentence JSON, reads PCM+sentinel, then sends the next sentence.
  At turn end (or on barge-in) it sends { "type":"close" } to cleanly end
  the session.

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
parser.add_argument("--host",      default="localhost")
parser.add_argument("--port",      default=8090,  type=int)
parser.add_argument("--tts-host",  default="localhost")
parser.add_argument("--tts-port",  default=8765,  type=int)
parser.add_argument("--mic-rate",  default=16000, type=int)
parser.add_argument("--chunk-ms",  default=20,    type=int)
parser.add_argument("--out-rate",  default=24000, type=int)
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
_audio_in_q:  asyncio.Queue          # mic  → gateway  (bytes)
_pcm_play_q:  stdlib_queue.Queue     # gateway → speaker thread (np.ndarray float32)
_loop         = None
_stop         = asyncio.Event()

_cur_user_words:   list = []
_cur_ai_tokens:    list = []
_cur_ai_sentences: list = []
_ai_speaking       = False

# ── Waveform helper ───────────────────────────────────────────────────────────

def _waveform_line(pcm_f32: np.ndarray, width: int = 40) -> str:
    if pcm_f32 is None or len(pcm_f32) == 0:
        return c(DIM, "·" * width)
    step  = max(1, len(pcm_f32) // width)
    cols  = [float(np.sqrt(np.mean(pcm_f32[i:i+step]**2)))
             for i in range(0, len(pcm_f32), step)][:width]
    mx    = max(cols) if cols else 0.0
    mx    = mx or 1e-9    # guard against all-zero silent chunks
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
    c2t   = ev.get("cag_to_tts_ms")
    tts_s = ev.get("tts_synth_ms")
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
            leftover = np.zeros(0, dtype=np.float32)

            while not _stop.is_set():
                chunks = []
                try:
                    while True:
                        chunks.append(_pcm_play_q.get_nowait())
                except stdlib_queue.Empty:
                    pass

                if chunks:
                    _ai_speaking = True
                    buf = (np.concatenate([leftover] + chunks)
                           if leftover.size else np.concatenate(chunks))
                    leftover = np.zeros(0, dtype=np.float32)
                    pos = 0
                    while pos < len(buf):
                        frame = buf[pos:pos+BLOCK]
                        if len(frame) < BLOCK:
                            leftover = frame
                            break
                        stream.write(frame)
                        pos += BLOCK
                else:
                    _ai_speaking = False
                    stream.write(silence)
                    import time; time.sleep(0.005)

    except Exception as e:
        print(f"\n  {c(RED,'Audio output error: '+str(e))}")

# ── Waveform display coroutine ────────────────────────────────────────────────

async def _waveform_display():
    last_pcm: np.ndarray = np.zeros(512, dtype=np.float32)
    while not _stop.is_set():
        await asyncio.sleep(0.08)
        if _ai_speaking:
            wf = _waveform_line(last_pcm, width=TW - 20)
            print(f"\r{ts()}  {c(MAG+BOLD,'♪')}  {wf}  ", end="", flush=True)

# ── Mic callback ─────────────────────────────────────────────────────────────

def _mic_cb(indata, frames, time_info, status):
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

# ── TTS session manager ───────────────────────────────────────────────────────
#
# PIPELINE DESIGN (v3.3):
#
#   Gateway emits ai_sentence events as the LLM streams tokens.
#   We want sentence N+1 to START GENERATING on the TTS while sentence N is
#   still being played back — so there is no gap between sentences.
#
#   Architecture:
#     _sentence_q  ←  ai_sentence events push text here immediately
#     _tts_worker  →  single coroutine, pulls from _sentence_q, sends to TTS
#                     WS one at a time (sequential on same connection),
#                     pushes PCM to _pcm_play_q as bytes arrive
#
#   This gives us:
#     • No concurrent sends on one WS (no lock contention, no races)
#     • Sentences pipeline: S2 starts generating the moment S1's last byte
#       is sent to _pcm_play_q — the audio output thread plays S1 while S2
#       is being synthesised
#     • Cancellation: pushing _SENTINEL into _sentence_q unblocks the worker
#       and the worker sends {"type":"cancel"} + {"type":"close"} to TTS

_SENTINEL = object()   # signals worker to cancel and exit

class _TTSSession:
    """
    One persistent WS session to TTS microservice per AI turn.
    Sentences are queued and synthesised in order by a single worker coroutine.
    """
    WAV_HEADER_LEN = 44

    def __init__(self, voice: str = "tara"):
        self._voice       = voice
        self._ws          = None
        self._sentence_q: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Connect and start the background worker."""
        try:
            self._ws = await websockets.connect(
                TTS_URL,
                ping_interval=None,
                ping_timeout=None,
                max_size=32 * 1024 * 1024,
                open_timeout=10,
            )
        except Exception as exc:
            print(f"\n  {c(RED,'TTS connect error: ')}{exc}")
            self._ws = None
            return
        self._worker_task = asyncio.create_task(self._worker())

    def enqueue(self, text: str, voice: Optional[str] = None):
        """Queue a sentence for synthesis. Non-blocking."""
        self._sentence_q.put_nowait((text, voice or self._voice))

    async def cancel_and_close(self):
        """Abort current synthesis and close the session."""
        # Drain the queue first so worker doesn't pick up stale sentences
        while not self._sentence_q.empty():
            try:
                self._sentence_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Push sentinel to unblock the worker
        await self._sentence_q.put(_SENTINEL)
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()

    async def finish_and_close(self):
        """Wait for all queued sentences to finish, then close."""
        await self._sentence_q.put(_SENTINEL)
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=120.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()

    async def _worker(self):
        """
        Single coroutine — pulls sentences from queue, synthesises them
        sequentially on one WS connection.  Exits on _SENTINEL.
        """
        if not self._ws:
            return
        try:
            while True:
                item = await self._sentence_q.get()

                if item is _SENTINEL:
                    # Cancel any in-progress synthesis and close
                    try:
                        await self._ws.send(json.dumps({"type": "cancel"}))
                        await self._ws.send(json.dumps({"type": "close"}))
                        await self._ws.close()
                    except Exception:
                        pass
                    return

                text, voice = item
                await self._synthesise_one(text, voice)

        except Exception as exc:
            print(f"\n  {c(RED,'TTS worker error: ')}{exc}")
        finally:
            if self._ws and not self._ws.closed:
                try:
                    await self._ws.send(json.dumps({"type": "close"}))
                    await self._ws.close()
                except Exception:
                    pass

    async def _synthesise_one(self, text: str, voice: str):
        """Send one sentence, stream PCM to playback queue, wait for sentinel."""
        if not self._ws or self._ws.closed:
            return
        try:
            await self._ws.send(json.dumps({
                "text":       text,
                "voice":      voice,
                "chunk_tone": "auto",
            }))

            got_header = False
            async for msg in self._ws:
                if isinstance(msg, bytes):
                    if not msg:
                        # end-of-request sentinel — this sentence is done
                        return
                    if not got_header:
                        if len(msg) == self.WAV_HEADER_LEN:
                            got_header = True
                            continue
                        if len(msg) > self.WAV_HEADER_LEN:
                            msg = msg[self.WAV_HEADER_LEN:]
                        got_header = True
                    pcm = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                    _pcm_play_q.put(pcm)
                    wf = _waveform_line(pcm, width=TW - 20)
                    print(f"\r{ts()}  {c(MAG+BOLD,'♪')}  {wf}  ", end="", flush=True)
                else:
                    try:
                        ev = json.loads(msg)
                        t  = ev.get("type", "")
                        if t == "error":
                            print(f"\n  {c(RED,'TTS error: ')}{ev.get('detail','')}")
                        elif t == "ping":
                            pass   # server heartbeat, ignore
                    except Exception:
                        pass

        except Exception as exc:
            print(f"\n  {c(RED,'TTS synthesise error: ')}{exc}")
            self._ws = None


_tts_session: Optional[_TTSSession] = None

# ── Receiver ─────────────────────────────────────────────────────────────────

async def _receiver(ws):
    global _cur_user_words, _cur_ai_tokens, _cur_ai_sentences, _ai_speaking
    global _tts_session

    _cur_user_words   = []
    _cur_ai_tokens    = []
    _cur_ai_sentences = []
    _in_tok           = False

    # Active TTS voice for this session
    _current_voice = "tara"

    async for msg in ws:

        # ── PCM audio bytes from GATEWAY (legacy path if gateway handles TTS) ─
        if isinstance(msg, bytes):
            if msg:
                pcm = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                _pcm_play_q.put(pcm)
                wf = _waveform_line(pcm, width=TW - 20)
                print(f"\r{ts()}  {c(MAG+BOLD,'♪')}  {wf}  ", end="", flush=True)
            continue

        # ── JSON control frames ───────────────────────────────────────────────
        try:
            ev = json.loads(msg)
        except Exception:
            continue

        t = ev.get("type")

        if DEBUG_RAW:
            print(f"\n{ts()}  {c(DIM,'RAW['+str(t)+']')}  {c(DIM,json.dumps(ev)[:160])}")

        # STT
        if t == "word":
            if _ai_speaking: print()
            _cur_user_words.append(ev.get("word",""))
            print(f"\r{ts()}  🎤  {c(GRN,' '.join(_cur_user_words))}  ", end="", flush=True)

        elif t == "partial":
            word = ev.get("word","").strip().rstrip("?.!,;:")
            if word and (not _cur_user_words or _cur_user_words[-1].lower()!=word.lower()):
                _cur_user_words.append(word)
            print(f"\r{ts()}  🎤  {c(GRN,' '.join(_cur_user_words))}  ", end="", flush=True)

        elif t == "segment":
            text = ev.get("text","").strip()
            print()
            _segment_banner(text)
            _cur_user_words = []

        # CAG
        elif t == "thinking":
            if _in_tok: print(); _in_tok = False
            _cur_ai_tokens = []; _cur_ai_sentences = []
            _thinking_banner(ev.get("turn_id",""))
            print(f"{ts()}  {c(BLU+BOLD,'AI')}  ", end="", flush=True)
            _in_tok = True
            # Open fresh TTS session for this turn
            if _tts_session:
                await _tts_session.cancel_and_close()
            _tts_session = _TTSSession(voice=_current_voice)
            await _tts_session.start()

        elif t == "ai_token":
            print(c(BLU, ev.get("token","")), end="", flush=True)
            _cur_ai_tokens.append(ev.get("token",""))
            _in_tok = True

        elif t == "ai_sentence":
            text = ev.get("text","").strip()
            _cur_ai_sentences.append(text)
            if _in_tok: print(); _in_tok = False
            _tts_chunk_line(len(_cur_ai_sentences), text)
            print(f"{ts()}  {c(BLU+BOLD,'AI')}  ", end="", flush=True)
            _in_tok = True
            # Enqueue immediately — worker pipelines synthesis with playback
            if _tts_session:
                voice = ev.get("voice", _current_voice)
                _tts_session.enqueue(text, voice=voice)

        elif t == "done":
            if _in_tok: print(); _in_tok = False
            full = "".join(_cur_ai_tokens).strip()
            if full:
                _conv_entry("ai", full)
            _cur_ai_tokens = []; _cur_ai_sentences = []
            # Let worker finish remaining sentences, then close
            if _tts_session:
                asyncio.create_task(_tts_session.finish_and_close())
                _tts_session = None

        # Latency
        elif t == "latency":
            if _in_tok: print(); _in_tok = False
            _latency_panel(ev)

        elif t == "session_summary":
            _session_summary(ev.get("latency",{}))
            _stop.set()

        # Misc
        elif t == "barge_in":
            if _in_tok: print(); _in_tok = False
            print(f"\n  {c(YLW+BOLD,'⚡ BARGE-IN')}  {c(YLW,'Stopping AI speech')}")
            if _tts_session:
                await _tts_session.cancel_and_close()
                _tts_session = None

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