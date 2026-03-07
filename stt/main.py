"""
main.py — STT microservice  v3.4.1
────────────────────────────────────────────────────────────────────────────
PHONE-CALL DESIGN: speak immediately, enroll in background
──────────────────────────────────────────────────────────
Enrollment and live transcription run in PARALLEL from the very first word.
The caller hears words appearing on-screen the moment they start speaking —
no waiting for enrollment to complete.  This mirrors a real phone call where
conversation begins immediately.

FIXES IN v3.4.1
───────────────
1. DEAD CODE REMOVED in _make_pipeline()
   Duplicate `logger.info` + `return pipeline` lines after the real return
   were unreachable.  Removed.

2. asr_min_buffer_ms EXPOSED in _make_pipeline()
   pipeline.py now accumulates voice chunks until 600ms before calling
   Whisper.  This is the root-cause fix for words never appearing live:
   Whisper was receiving 20ms micro-chunks and producing no stable word
   timestamps.  600ms gives it enough context to emit words one-by-one
   as the speaker talks, instead of waiting until end-of-utterance silence.

FIXES IN v3.4.0
───────────────
1. PARALLEL ENROLLMENT + TRANSCRIPTION (phone-call UX)
2. SMOOTH ENROLLMENT PROGRESS EVENTS
   Progress is now sent on every percentage-point change.

FIXES IN v3.3.0
───────────────
1. WHISPER HALLUCINATION FILTER
2. CONTEXT BLEED fix (reset_context after each segment)
3. ENROLLMENT logging cleaned up.
"""

import base64
import json
import logging
import io

import numpy as np
import uvicorn
import soundfile as sf
import re

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from pipeline import STTPipeline

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("speaker_enrollment").setLevel(logging.INFO)
logging.getLogger("pipeline").setLevel(logging.INFO)

app = FastAPI(title="STT Microservice", version="3.4.1")

# ── Hallucination filter config ───────────────────────────────────────────────
MIN_SEGMENT_CHARS = 3
MAX_REPEATS       = 3

# Set True to log RMS/peak/VAD stats every ~1s per connection.
_LOG_AUDIO_DIAG   = True


def _filter_segment(text: str) -> str | None:
    """
    Returns cleaned text if it looks like real speech, None to drop it.
    1. Strip punctuation-only content
    2. Must contain at least one alphabetic word
    3. Length gate
    4. Repetition gate (Whisper hallucination loop catcher)
    """
    text = text.strip().strip(".,!?;:-\u2013\u2014").strip()
    if not text:
        return None
    if not any(c.isalpha() for c in text):
        logger.debug(f"[filter] dropped (no alpha): {text!r}")
        return None
    if len(text) < MIN_SEGMENT_CHARS:
        logger.debug(f"[filter] dropped (too short): {text!r}")
        return None
    words = text.lower().split()
    if len(words) >= 2:
        for phrase_len in range(1, min(7, len(words) // 2 + 1)):
            for i in range(len(words) - phrase_len + 1):
                phrase = " ".join(words[i:i + phrase_len])
                count  = sum(
                    1 for j in range(0, len(words) - phrase_len + 1)
                    if " ".join(words[j:j + phrase_len]) == phrase
                )
                if count > MAX_REPEATS:
                    logger.debug(f"[filter] dropped (repetition x{count}): {text[:80]!r}")
                    return None
    return text


# ── Pipeline factory ──────────────────────────────────────────────────────────

def _make_pipeline() -> STTPipeline:
    logger.info("🔧 Creating STTPipeline...")
    pipeline = STTPipeline(
        sample_rate            = 16000,
        whisper_model_size     = "base.en",
        idle_threshold         = 0.10,
        barge_in_threshold     = 0.30,
        vad_pre_gain           = 15.0,
        enable_ai_filtering    = False,
        ai_detector_model_path = None,
        ai_detection_threshold = 0.7,
        enable_aec             = True,
        enable_enrollment      = True,   # v7: first-sentence enrollment
        similarity_threshold   = 0.72,   # v7: tighter — full-sentence centroid is reliable
        overlap_seconds        = 0.8,
        word_gap_ms            = 80.0,
        max_context_words      = 40,
        max_history_turns      = 3,
        # FIX: accumulate 600ms of voice before calling Whisper.
        # Was the root cause of no live words: micro-chunks (20ms) gave
        # Whisper too little audio to produce stable word timestamps.
        asr_min_buffer_ms      = 600.0,
        speculative_vad_threshold = 0.08,
    )
    logger.info("✅ STTPipeline ready")
    return pipeline


_rest_pipeline: STTPipeline | None = None

def _get_rest_pipeline() -> STTPipeline:
    global _rest_pipeline
    if _rest_pipeline is None:
        _rest_pipeline = _make_pipeline()
    return _rest_pipeline


def _reset_asr_context(pipeline: STTPipeline):
    """
    Clear Whisper's context window so the next segment doesn't inherit
    tokens from the previous one.
    """
    try:
        pipeline.realtime_asr._history.clear()
        logger.debug("[context] ASR history cleared")
        return
    except AttributeError:
        pass
    for attr in ("_context", "_prompt", "_prev_text", "context"):
        obj = getattr(pipeline, attr, None) or getattr(
            getattr(pipeline, "realtime_asr", None), attr, None
        )
        if obj is not None:
            try:
                if isinstance(obj, list):
                    obj.clear()
                elif isinstance(obj, str):
                    setattr(pipeline.realtime_asr, attr, "")
                logger.debug(f"[context] cleared .{attr}")
                return
            except Exception:
                pass


# ── REST endpoint ─────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode audio: {e}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(16000, sr)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        except ImportError:
            raise HTTPException(
                status_code=422,
                detail="scipy not installed — upload 16 kHz audio.",
            )
    return {"transcript": _get_rest_pipeline().transcribe_full(audio)}


# ── WebSocket /stream/mux ─────────────────────────────────────────────────────

@app.websocket("/stream/mux")
async def stream_audio_mux(websocket: WebSocket):
    await websocket.accept()
    client = websocket.client
    logger.info(f"🔌 WebSocket connected: {client}")

    pipeline = _make_pipeline()
    pipeline.reset()

    _last_enrolled_state:    bool = False
    _enrollment_log_counter: int  = 0
    _diag_chunk_count:       int  = 0

    try:
        async for message in websocket.iter_bytes():
            if len(message) < 1:
                continue

            frame_type = message[0]
            payload    = message[1:]

            # ── Audio frame ───────────────────────────────────────────────
            if frame_type == 0x01:
                pcm = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
                _diag_chunk_count += 1

                if _LOG_AUDIO_DIAG and _diag_chunk_count % 100 == 1:
                    raw_rms  = float(np.sqrt(np.mean(pcm**2))) if len(pcm) else 0.0
                    raw_peak = float(np.max(np.abs(pcm)))      if len(pcm) else 0.0
                    pre_gain = getattr(pipeline.vad, "pre_gain", 15.0)
                    est_agc_rms = min(raw_rms * pre_gain, 0.05)
                    vad_state = pipeline.vad.get_state() if hasattr(pipeline, "vad") else {}
                    thr = getattr(pipeline.vad, "idle_threshold", 0.10)
                    logger.info(
                        f"[diag] chunk={_diag_chunk_count}  "
                        f"raw_rms={raw_rms:.4f}  raw_peak={raw_peak:.3f}  "
                        f"est_agc_rms≈{est_agc_rms:.4f}  "
                        f"vad_prob={vad_state.get('prob', 0):.3f}  "
                        f"threshold={thr:.2f}  "
                        f"voice={vad_state.get('was_voice', False)}"
                    )

                events = pipeline.process_chunk(pcm)

                for event in events:
                    etype = event.get("type")

                    # ── Enrollment ────────────────────────────────────────
                    if etype == "enrollment":
                        enrolled = event.get("enrolled", False)
                        progress = event.get("progress", 0.0)
                        reason   = event.get("reason", "")
                        pct      = int(progress * 100)

                        # Only fire enrolled confirmation when the profile locked
                        # from a real first sentence (not noise, not single chunk).
                        # reason="enrolled_first_sentence" is the only valid lock.
                        is_real_lock = (
                            enrolled
                            and not _last_enrolled_state
                            and reason == "enrolled_first_sentence"
                        )

                        if is_real_lock:
                            _last_enrolled_state    = True
                            _enrollment_log_counter = 100
                            logger.info("[enroll] Speaker enrolled (first sentence)")
                            await websocket.send_json({
                                "type":     "enrollment",
                                "enrolled": True,
                                "progress": 1.0,
                                "message":  "Speaker enrolled",
                            })
                        elif not enrolled and pct != _enrollment_log_counter:
                            # Show collecting progress (0-99%) while user speaks
                            _enrollment_log_counter = pct
                            await websocket.send_json(event)
                        continue

                    # ── Word ─────────────────────────────────────────────
                    elif etype == "word":
                        word = event.get("word", "").strip()
                        if word:
                            print(word, flush=True)
                        await websocket.send_json(event)
                        continue

                    # ── Partial ───────────────────────────────────────────
                    elif etype == "partial":
                        await websocket.send_json(event)
                        continue

                    # ── Segment ───────────────────────────────────────────
                    elif etype == "segment":
                        raw_text = event.get("text", "").strip()
                        clean    = _filter_segment(raw_text)
                        if clean is None:
                            continue
                        if clean != raw_text:
                            event = {**event, "text": clean}
                        logger.info(f"SEGMENT: {clean}")
                        await websocket.send_json(event)
                        _reset_asr_context(pipeline)
                        continue

                    # ── Everything else — forward silently ────────────────
                    await websocket.send_json(event)

            # ── Control frame ─────────────────────────────────────────────
            elif frame_type == 0x02:
                try:
                    ctrl     = json.loads(payload.decode("utf-8"))
                    msg_type = ctrl.get("type")
                    logger.debug(f"[ctrl] {msg_type}")

                    if msg_type == "ai_state":
                        speaking = bool(ctrl.get("speaking", False))
                        logger.info(f"[ctrl] AI speaking → {speaking}")
                        pipeline.notify_ai_speaking(speaking)

                    elif msg_type == "ai_reference":
                        b64 = ctrl.get("pcm", "")
                        if b64:
                            raw_bytes = base64.b64decode(b64)
                            ref = (
                                np.frombuffer(raw_bytes, dtype=np.int16)
                                .astype(np.float32) / 32768.0
                            )
                            pipeline.push_ai_reference(ref)

                    elif msg_type == "assistant_turn":
                        text = ctrl.get("text", "").strip()
                        if text:
                            logger.info(f"[ctrl] assistant_turn: {text[:60]}")
                            pipeline.add_assistant_turn(text)

                    elif msg_type == "reset_context":
                        _reset_asr_context(pipeline)
                        logger.info("[ctrl] context reset")
                        await websocket.send_json({"type": "context_reset"})

                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif msg_type == "get_stats":
                        stats = pipeline.get_stats()
                        logger.info(f"[ctrl] stats: {stats}")
                        await websocket.send_json({"type": "stats", **stats})

                    elif msg_type == "reset_enrollment":
                        logger.info("[ctrl] enrollment reset")
                        if pipeline.enrollment:
                            pipeline.enrollment.reset()
                        _last_enrolled_state    = False
                        _enrollment_log_counter = 0
                        await websocket.send_json({
                            "type":     "enrollment_status",
                            "enrolled": False,
                            "progress": 0.0,
                        })

                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"[ctrl] bad payload: {e}")

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {client}")
        final = pipeline.flush()
        if final:
            clean = _filter_segment(final)
            if clean:
                try:
                    await websocket.send_json({"type": "segment", "text": clean})
                except Exception:
                    pass

    except Exception as e:
        logger.exception(f"[MUX] Unhandled error: {e}")
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
            await websocket.close()
        except Exception:
            pass


# ── WebSocket /stream/binary ──────────────────────────────────────────────────

@app.websocket("/stream/binary")
async def stream_audio_binary(websocket: WebSocket):
    await websocket.accept()
    pipeline = _make_pipeline()
    pipeline.reset()
    try:
        async for message in websocket.iter_bytes():
            pcm    = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            events = pipeline.process_chunk(pcm)
            for event in events:
                if event.get("type") == "segment":
                    text = _filter_segment(event.get("text", ""))
                    if not text:
                        continue
                    event = {**event, "text": text}
                    _reset_asr_context(pipeline)
                await websocket.send_json(event)
    except WebSocketDisconnect:
        final = pipeline.flush()
        if final:
            clean = _filter_segment(final)
            if clean:
                await websocket.send_json({"type": "segment", "text": clean})
    except Exception as e:
        await websocket.send_json({"type": "error", "detail": str(e)})


# ── Health / stats ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    return _get_rest_pipeline().get_stats()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )