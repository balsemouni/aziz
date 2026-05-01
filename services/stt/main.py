"""
main.py — STT Microservice  v3.2
══════════════════════════════════════════════════════════════════════════════

Changes from v3.1
──────────────────
  Fix 1 — Non-blocking session creation
      _build_pipeline() loads Silero VAD + Whisper, which takes 2-5s on first
      call even with a warmup (the warmup instance is a *different* object;
      model weights are already in GPU VRAM so the second load is faster, but
      PyTorch still has to allocate new tensors, re-JIT, etc.).
      When this happened inside the WebSocket coroutine it blocked the entire
      event loop: the gateway was flooding the STT _audio_q but nobody was
      reading it, frames overflowed and were dropped, and the pipeline never
      saw any audio.
      Fix: sessions are pre-created at startup (pool of 1) and reused on
      first connect. Subsequent sessions are built in a thread executor so
      the event loop stays free.

  Fix 2 — LOG_LEVEL env var respected at uvicorn level too

  Fix 3 — Startup warmup now reuses the pre-built session (no second load)

GATEWAY PATCH (still required — see bottom of file):
  Change STT_WS_URL connect call to  f"{STT_WS_URL}?sid={self.sid}"
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import logging
import os
import time
import uuid
from typing import Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from pipeline import STTPipeline

_DEEPFILTER_AVAILABLE = False
try:
    from deepfilter import DeepFilterNoiseReducer   # noqa: F401
    _DEEPFILTER_AVAILABLE = True
except ImportError:
    pass


# ─── Configuration ────────────────────────────────────────────────────────────

HOST                 = os.getenv("STT_HOST",              "0.0.0.0")
PORT                 = int(os.getenv("STT_PORT",          "8001"))
SAMPLE_RATE          = int(os.getenv("SAMPLE_RATE",       "16000"))
WHISPER_MODEL        = os.getenv("WHISPER_MODEL",         "base.en")
DEVICE               = os.getenv("DEVICE",                "")

VAD_IDLE_THRESH      = float(os.getenv("VAD_IDLE_THRESH", "0.15"))
VAD_BARGE_IN_THRESH  = float(os.getenv("VAD_BARGE_IN",    "0.40"))
VAD_PRE_GAIN         = float(os.getenv("VAD_PRE_GAIN",    "5.0"))

ASR_OVERLAP_S        = float(os.getenv("ASR_OVERLAP_S",   "0.8"))
ASR_WORD_GAP_MS      = float(os.getenv("ASR_WORD_GAP_MS", "60.0"))
ASR_CONTEXT_WORDS    = int(os.getenv("ASR_CONTEXT_WORDS", "10"))
ASR_HISTORY_TURNS    = int(os.getenv("ASR_HISTORY_TURNS", "3"))

ENABLE_AEC           = os.getenv("ENABLE_AEC",          "true").lower()  == "true"
ENABLE_VOICE_GATE    = os.getenv("ENABLE_VOICE_GATE",    "true").lower()  == "true"
ENABLE_DEEPFILTER    = os.getenv("ENABLE_DEEPFILTER",    "false").lower() == "true"
SESSION_IDLE_TIMEOUT = float(os.getenv("SESSION_IDLE_TIMEOUT", "300.0"))
TTS_SAMPLE_RATE      = int(os.getenv("TTS_SAMPLE_RATE", "22050"))   # Piper TTS output rate


# ─── Logging ──────────────────────────────────────────────────────────────────

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level  = getattr(logging, _log_level, logging.INFO),
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("stt")


class _NoMetricsFilter(logging.Filter):
    """Drop uvicorn access log lines for /metrics so they don't spam the console."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "/metrics" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_NoMetricsFilter())

app = FastAPI(title="STT Microservice", version="3.2.0")

# Thread pool for building pipelines off the event loop
_build_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="pipeline-build"
)


# ─── Pipeline / noise factory (blocking — always call in executor) ────────────

def _make_pipeline() -> STTPipeline:
    import torch
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    return STTPipeline(
        sample_rate        = SAMPLE_RATE,
        device             = device,
        idle_threshold     = VAD_IDLE_THRESH,
        barge_in_threshold = VAD_BARGE_IN_THRESH,
        vad_pre_gain       = VAD_PRE_GAIN,
        whisper_model_size = WHISPER_MODEL,
        overlap_seconds    = ASR_OVERLAP_S,
        word_gap_ms        = ASR_WORD_GAP_MS,
        max_context_words  = ASR_CONTEXT_WORDS,
        max_history_turns  = ASR_HISTORY_TURNS,
        enable_aec         = ENABLE_AEC,
        enable_voice_gate  = ENABLE_VOICE_GATE,
    )


def _make_noise_reducer():
    if not ENABLE_DEEPFILTER or not _DEEPFILTER_AVAILABLE:
        return None
    try:
        from deepfilter import DeepFilterNoiseReducer
        return DeepFilterNoiseReducer(sample_rate=SAMPLE_RATE)
    except Exception as exc:
        log.warning(f"DeepFilter failed: {exc}")
        return None


def _pcm_to_f32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ─── Session ──────────────────────────────────────────────────────────────────

class _Session:
    """
    One per gateway WebSocket (?sid= keyed).

    NEVER construct directly from the event loop — use _Session.create()
    which runs the blocking pipeline load in a thread executor.

    Threading
    ─────────
    process_chunk is dispatched to a single-worker executor so:
      • It never blocks the asyncio event loop.
      • Frames are always processed serially (queue depth = 1 thread).
    asyncio.Lock prevents a second frame entering run_in_executor while the
    first is still running (belt-and-suspenders for high-throughput streams).
    """

    def __init__(self, sid: str, pipeline: STTPipeline, noise):
        self.sid      = sid
        self.pipeline = pipeline
        self.noise    = noise
        self._last_rx = time.monotonic()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"stt-{sid[:8]}"
        )
        self._lock = asyncio.Lock()

    @classmethod
    async def create(cls, sid: str) -> "_Session":
        """Build a session without blocking the event loop."""
        loop     = asyncio.get_event_loop()
        pipeline = await loop.run_in_executor(_build_executor, _make_pipeline)
        noise    = await loop.run_in_executor(_build_executor, _make_noise_reducer)
        return cls(sid, pipeline, noise)

    def touch(self):
        self._last_rx = time.monotonic()

    def idle_s(self) -> float:
        return time.monotonic() - self._last_rx

    def close(self):
        self._executor.shutdown(wait=False)


# ─── Session registry ─────────────────────────────────────────────────────────

_sessions: Dict[str, _Session] = {}

# Pre-warmed session ready for the FIRST incoming connection.
# After startup completes this holds a fully loaded _Session.
# The first WebSocket that has no existing sid claims it instantly
# (zero model-load latency on first call).
_warm_session: Optional[_Session] = None


async def _reap_idle_sessions():
    while True:
        await asyncio.sleep(60)
        dead = [sid for sid, s in _sessions.items() if s.idle_s() > SESSION_IDLE_TIMEOUT]
        for sid in dead:
            sess = _sessions.pop(sid, None)
            if sess:
                sess.close()
            log.info(f"[reaper] evicted {sid}")


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/stream/mux")
async def stream_mux(ws: WebSocket):
    global _warm_session
    await ws.accept()

    sid = ws.query_params.get("sid") or str(uuid.uuid4())

    if sid in _sessions:
        # Reconnect — reuse existing session (keeps Whisper history)
        sess = _sessions[sid]
        log.info(f"[{sid}] reconnected — pipeline preserved")
    elif _warm_session is not None:
        # Claim the pre-warmed session (zero latency on first connect)
        sess = _warm_session
        sess.sid = sid
        _warm_session = None
        _sessions[sid] = sess
        log.info(f"[{sid}] claimed pre-warmed session")
        # Kick off building the next warm session in the background
        asyncio.create_task(_pre_warm())
    else:
        # Build a new session off the event loop (non-blocking)
        log.info(f"[{sid}] building session (first connect — models loading)…")
        sess = await _Session.create(sid)
        _sessions[sid] = sess
        log.info(f"[{sid}] session ready")

    try:
        async for raw in ws.iter_bytes():
            if not raw:
                continue

            sess.touch()
            ftype   = raw[0]
            payload = raw[1:]

            # ── 0x01  Audio ───────────────────────────────────────────────
            if ftype == 0x01:
                if len(payload) < 2:
                    continue

                audio = _pcm_to_f32(payload)

                if sess.noise is not None:
                    audio = sess.noise.process(audio)

                async with sess._lock:
                    events = await asyncio.get_event_loop().run_in_executor(
                        sess._executor,
                        sess.pipeline.process_chunk,
                        audio,
                    )

                for ev in events:
                    await ws.send_json(ev)
                    log.debug(f"[{sid}] ← {ev}")

            # ── 0x02  Control ─────────────────────────────────────────────
            elif ftype == 0x02:
                try:
                    ctrl = json.loads(payload)
                except Exception:
                    await ws.send_json({"type": "error", "message": "bad control JSON"})
                    continue

                mtype = ctrl.get("type", "")

                if mtype == "reset_context":
                    sess.pipeline.reset()
                    log.info(f"[{sid}] reset")
                    await ws.send_json({"type": "reset_ok"})

                elif mtype == "hard_reset":
                    sess.pipeline.reset()
                    sess.pipeline.realtime_asr.reset()
                    log.info(f"[{sid}] hard reset")
                    await ws.send_json({"type": "reset_ok"})

                elif mtype in ("assistant_turn", "add_assistant_turn"):
                    # Gateway v14 sends "assistant_turn" after each CAG reply
                    text = ctrl.get("text", "").strip()
                    if text:
                        sess.pipeline.realtime_asr.add_assistant_turn(text)
                        log.debug(f"[{sid}] assistant turn: {text[:80]!r}")

                elif mtype == "get_stats":
                    stats = sess.pipeline.get_stats()
                    stats.update({"sid": sid, "idle_s": round(sess.idle_s(), 1)})
                    await ws.send_json({"type": "stats", **stats})

                elif mtype == "ping":
                    await ws.send_json({"type": "pong"})

                elif mtype == "ai_state":
                    speaking = bool(ctrl.get("speaking", False))
                    sess.pipeline.notify_ai_speaking(speaking)
                    log.debug(f"[{sid}] ai_state speaking={speaking}")

                else:
                    log.debug(f"[{sid}] unknown ctrl: {mtype!r}")

            # ── 0x03  TTS reference audio (fingerprint enrollment) ────────
            elif ftype == 0x03:
                if len(payload) >= 2 and sess.pipeline.voice_gate is not None:
                    try:
                        # TTS audio is at TTS_SAMPLE_RATE (22050 Hz); resample to SAMPLE_RATE
                        pcm_tts = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
                        if TTS_SAMPLE_RATE != SAMPLE_RATE:
                            from math import gcd as _gcd
                            from scipy.signal import resample_poly as _rp
                            _g = _gcd(SAMPLE_RATE, TTS_SAMPLE_RATE)
                            pcm_tts = _rp(pcm_tts, SAMPLE_RATE // _g, TTS_SAMPLE_RATE // _g).astype(np.float32)
                        sess.pipeline.push_ai_reference(pcm_tts)
                    except Exception as _e:
                        log.debug(f"[{sid}] TTS ref enroll error: {_e}")

    except WebSocketDisconnect:
        log.info(f"[{sid}] disconnected")
    except Exception as exc:
        log.error(f"[{sid}] error: {exc}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ─── REST ─────────────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    t0  = time.monotonic()
    raw = await file.read()
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if sr != SAMPLE_RATE:
            from math import gcd
            from scipy.signal import resample_poly
            g    = gcd(SAMPLE_RATE, sr)
            data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
        audio = data if data.ndim == 1 else data[:, 0]
    except Exception:
        audio = _pcm_to_f32(raw)

    pipe   = await asyncio.get_event_loop().run_in_executor(_build_executor, _make_pipeline)
    text   = pipe.transcribe_full(audio)
    lat_ms = (time.monotonic() - t0) * 1000
    return {"text": text, "latency_ms": round(lat_ms, 1)}


@app.post("/reset/{sid}")
async def reset_session(sid: str):
    sess = _sessions.get(sid)
    if not sess:
        raise HTTPException(404, f"session {sid!r} not found")
    sess.pipeline.reset()
    return {"status": "reset", "sid": sid}


@app.get("/stats/{sid}")
async def session_stats(sid: str):
    sess = _sessions.get(sid)
    if not sess:
        raise HTTPException(404, f"session {sid!r} not found")
    stats = sess.pipeline.get_stats()
    stats.update({"sid": sid, "idle_s": round(sess.idle_s(), 1)})
    return stats


@app.get("/sessions")
async def list_sessions():
    return {
        "sessions": [
            {"sid": sid, "idle_s": round(s.idle_s(), 1)}
            for sid, s in _sessions.items()
        ]
    }


@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "version":    "3.2.0",
        "model":      WHISPER_MODEL,
        "deepfilter": ENABLE_DEEPFILTER and _DEEPFILTER_AVAILABLE,
        "sessions":   len(_sessions),
        "warm":       _warm_session is not None,
    }


@app.get("/metrics")
async def metrics():
    """Stub metrics endpoint — silences 404 from external Prometheus scrapers."""
    return JSONResponse(content={})


# ─── Pre-warm helper ──────────────────────────────────────────────────────────

async def _pre_warm():
    """Build one session in the background and park it as _warm_session."""
    global _warm_session
    if _warm_session is not None:
        return
    try:
        log.info("[warm] Building next warm session…")
        sess = await _Session.create("__warm__")
        # Quick smoke-test so VAD and Whisper tensors are allocated
        await asyncio.get_event_loop().run_in_executor(
            sess._executor,
            sess.pipeline.process_chunk,
            np.zeros(512, dtype=np.float32),
        )
        _warm_session = sess
        log.info("[warm] ✅ Warm session ready")
    except Exception as exc:
        log.warning(f"[warm] Failed: {exc}")


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    log.info("=" * 60)
    log.info("  STT Microservice v3.2")
    log.info(f"  Listen     : {HOST}:{PORT}")
    log.info(f"  Whisper    : {WHISPER_MODEL}")
    log.info(f"  DeepFilter : {'ON' if (ENABLE_DEEPFILTER and _DEEPFILTER_AVAILABLE) else 'OFF'}")
    log.info(f"  AEC        : {'ON (passive)' if ENABLE_AEC else 'OFF'}")
    log.info(f"  VoiceGate  : {'ON (passive)' if ENABLE_VOICE_GATE else 'OFF'}")
    log.info("=" * 60)

    # Build and smoke-test the warm session during startup.
    # This means the FIRST incoming gateway connection gets a pipeline with
    # zero model-load latency — models are already in GPU memory.
    log.info("[startup] Pre-warming pipeline (loading Silero VAD + Whisper)…")
    await _pre_warm()

    asyncio.create_task(_reap_idle_sessions())
    log.info("[startup] Ready — session reaper running")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host      = HOST,
        port      = PORT,
        workers   = 1,       # MUST be 1 — torch models not fork-safe
        reload    = False,
        log_level = _log_level.lower(),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  REQUIRED ONE-LINE PATCH TO gateway.py  (_stt_loop, line ~957)
# ══════════════════════════════════════════════════════════════════════════════
#
#  BEFORE:
#      stt_ws = await _ws_connect(
#          STT_WS_URL, max_retries=STT_MAX_RETRIES,
#
#  AFTER:
#      stt_ws = await _ws_connect(
#          f"{STT_WS_URL}?sid={self.sid}", max_retries=STT_MAX_RETRIES,
#
#  Without this, all gateway sessions share one pipeline and corrupt each
#  other's VAD accumulator and ASR buffer.
# ══════════════════════════════════════════════════════════════════════════════