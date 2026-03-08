"""
main.py — STT microservice  v5.3.0
────────────────────────────────────────────────────────────────────────────

FIXES vs v5.2
─────────────
  Fix 1 — /enroll_tts now PROPERLY enrolls TTSVoiceGate
      v5.2 said "AEC-only mode" and only called push_ai_reference(), which
      feeds the AEC reference buffer but the TTSVoiceGate.enroll() call
      inside push_ai_reference() was still being called — HOWEVER the pipeline
      was built with enable_voice_gate=False by default in _build_pipeline().
      
      FIX: enable_voice_gate=True is now set in _build_pipeline(), and
      /enroll_tts correctly calls push_ai_reference() which feeds BOTH
      AEC reference AND TTSVoiceGate.enroll(). The response now reflects
      the real voice_gate enrollment status.

  Fix 2 — AI speaking → False spam eliminated
      The gateway sends ai_state=False multiple times when TTS stops
      (from multiple tasks: synth_worker, play_worker, _finalize_turn).
      main.py now de-duplicates: only calls notify_ai_speaking() when the
      state actually CHANGES. This stops the log flood of repeated
      "[ctrl] AI speaking → False" lines.

  Fix 3 — DELETE /enroll_tts now does a FULL voice gate reset
      Previously it was a no-op. Now it calls voice_gate.full_reset()
      so the TTS service can re-enroll after a speaker/voice change.

  Fix 4 — /enroll_tts returns real enrolled status
      Returns voice_gate.is_ready instead of hardcoded True, so the TTS
      enrollment manager knows the actual state and can retry if needed.

ENDPOINTS
─────────
  WS  /stream/mux      — main streaming endpoint (mux audio + control)
  WS  /stream/binary   — binary-only audio (no control frames)
  POST /transcribe     — REST full-file transcription
  POST /enroll_tts     — receive TTS voice PCM → enroll TTSVoiceGate + AEC
  DELETE /enroll_tts   — full reset of TTS voice profile
  GET  /health         — version + status
  GET  /stats          — pipeline stats
"""

import asyncio
import base64
import io
import json
import logging

import numpy as np
import uvicorn
import soundfile as sf

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from pydantic import BaseModel
from pipeline import STTPipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("pipeline").setLevel(logging.INFO)

app = FastAPI(title="STT Microservice", version="5.3.0")

MIN_SEGMENT_CHARS = 3
MAX_REPEATS       = 3
_LOG_AUDIO_DIAG   = True

# ── Single shared pipeline — loaded once at startup ───────────────────────────
_pipeline: STTPipeline | None = None


@app.on_event("startup")
async def _startup():
    global _pipeline
    logger.info("🔧 Loading STTPipeline (one-time startup)…")
    loop = asyncio.get_event_loop()
    _pipeline = await loop.run_in_executor(None, _build_pipeline)
    logger.info("✅ STTPipeline ready — accepting connections")


def _build_pipeline() -> STTPipeline:
    return STTPipeline(
        sample_rate           = 16000,
        whisper_model_size    = "base.en",
        idle_threshold        = 0.10,
        barge_in_threshold    = 0.30,
        vad_pre_gain          = 15.0,
        enable_aec            = True,
        # FIX 1: Enable TTSVoiceGate so /enroll_tts actually works
        enable_voice_gate     = True,
        voice_gate_threshold  = 0.70,
        voice_gate_barge_in   = 0.82,
        voice_gate_min_frames = 8,
        overlap_seconds       = 0.8,
        word_gap_ms           = 80.0,
        max_context_words     = 40,
        max_history_turns     = 3,
        asr_min_buffer_ms     = 600.0,
    )


def _get_pipeline() -> STTPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not ready — startup not complete")
    return _pipeline


# ── Hallucination / garbage filter ───────────────────────────────────────────

def _filter_segment(text: str) -> str | None:
    text = text.strip().strip(".,!?;:-\u2013\u2014").strip()
    if not text:
        return None
    if not any(c.isalpha() for c in text):
        return None
    if len(text) < MIN_SEGMENT_CHARS:
        return None
    words = text.lower().split()
    if len(words) >= 2:
        for phrase_len in range(1, min(7, len(words) // 2 + 1)):
            for i in range(len(words) - phrase_len + 1):
                phrase = " ".join(words[i:i + phrase_len])
                count  = sum(
                    1 for j in range(len(words) - phrase_len + 1)
                    if " ".join(words[j:j + phrase_len]) == phrase
                )
                if count > MAX_REPEATS:
                    return None
    return text


def _reset_asr_context(pipeline: STTPipeline):
    try:
        pipeline.realtime_asr._history.clear()
    except AttributeError:
        pass


# ── TTS self-enrollment endpoints ─────────────────────────────────────────────

class _EnrollReq(BaseModel):
    pcm_b64:     str
    sample_rate: int  = 16000
    commit:      bool = True


@app.post("/enroll_tts")
async def enroll_tts(req: _EnrollReq):
    """
    Called by tts_microservice background enrollment at startup.

    Feeds TTS audio into BOTH:
      • AECGate      — spectral subtraction reference
      • TTSVoiceGate — acoustic fingerprint enrollment

    Both gates receive audio via pipeline.push_ai_reference(), which
    internally calls both aec.push_reference() and voice_gate.enroll().

    Returns enrolled=True once TTSVoiceGate has enough frames (>= min_enroll_frames).
    Returns enrolled=False while still accumulating (TTS service will retry).
    """
    try:
        raw   = base64.b64decode(req.pcm_b64)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample to 16 kHz if needed
        if req.sample_rate != 16000 and len(audio) > 0:
            try:
                from math import gcd
                from scipy.signal import resample_poly
                g     = gcd(16000, req.sample_rate)
                audio = resample_poly(audio, 16000 // g, req.sample_rate // g).astype(np.float32)
            except ImportError:
                n     = int(len(audio) * 16000 / req.sample_rate)
                audio = np.interp(
                    np.linspace(0, 1, n), np.linspace(0, 1, len(audio)), audio
                ).astype(np.float32)

        pipeline = _get_pipeline()

        # This calls BOTH aec.push_reference() AND voice_gate.enroll() internally
        pipeline.push_ai_reference(audio)

        # Check if the voice gate is now ready (has enough enrolled frames)
        voice_gate_ready = (
            pipeline.voice_gate is not None and pipeline.voice_gate.is_ready
        )

        enrolled_frames = (
            pipeline.voice_gate._n_enrolled
            if pipeline.voice_gate is not None else 0
        )

        logger.info(
            f"[enroll_tts] fed {len(audio)} samples ({len(audio)/16000:.2f}s) "
            f"→ voice_gate enrolled={voice_gate_ready} "
            f"frames={enrolled_frames}"
        )

        # FIX 4: Return REAL enrolled status (not hardcoded True)
        # TTS service retries until enrolled=True, so this is safe.
        return {
            "enrolled":       voice_gate_ready,
            "samples":        len(audio),
            "duration_s":     round(len(audio) / 16000, 2),
            "enrolled_frames": enrolled_frames,
        }

    except Exception as exc:
        logger.error(f"[enroll_tts] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/enroll_tts")
async def reset_enroll_tts():
    """
    FIX 3: Full reset of TTSVoiceGate voice profile.
    Called by TTS service before re-enrollment after a speaker/voice change.
    """
    pipeline = _get_pipeline()
    if pipeline.voice_gate is not None:
        pipeline.voice_gate.full_reset()
        logger.info("[enroll_tts] TTSVoiceGate voice profile fully reset")
    if pipeline.aec is not None:
        pipeline.aec.reset()
        logger.info("[enroll_tts] AECGate reference buffer reset")
    return {"status": "ok", "voice_gate_reset": pipeline.voice_gate is not None}


# ── REST transcribe ───────────────────────────────────────────────────────────

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
            raise HTTPException(status_code=422, detail="scipy required for resampling")
    return {"transcript": _get_pipeline().transcribe_full(audio)}


# ── WebSocket /stream/mux ─────────────────────────────────────────────────────

@app.websocket("/stream/mux")
async def stream_audio_mux(websocket: WebSocket):
    await websocket.accept()
    client = websocket.client
    logger.info(f"🔌 WS connected: {client}")

    pipeline = _get_pipeline()
    pipeline.reset()
    _reset_asr_context(pipeline)

    diag_count:    int  = 0
    # FIX 2: Track last known ai_speaking state to de-duplicate notifications
    _last_ai_speaking: bool = False

    try:
        async for message in websocket.iter_bytes():
            if len(message) < 1:
                continue

            # Gateway strips 0x01 before sending to STT (gateway.py _push).
            # Audio arrives as raw PCM (no header). Control frames keep 0x02.
            if message[0] == 0x02:
                frame_type = 0x02
                payload    = message[1:]
            else:
                frame_type = 0x01  # raw PCM from gateway
                payload    = message

            # ── Audio frame (0x01) ────────────────────────────────────────
            if frame_type == 0x01:
                # int16 needs exactly 2 bytes per sample — drop stray odd byte
                if len(payload) % 2 != 0:
                    payload = payload[: len(payload) - 1]
                if not payload:
                    continue
                pcm = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
                diag_count += 1

                if _LOG_AUDIO_DIAG and diag_count % 200 == 1:
                    rms       = float(np.sqrt(np.mean(pcm**2))) if len(pcm) else 0.0
                    vad_state = pipeline.vad.get_state() if hasattr(pipeline, "vad") else {}
                    logger.debug(
                        f"[diag] chunk={diag_count}  rms={rms:.4f}  "
                        f"vad_prob={vad_state.get('prob', 0):.3f}"
                    )

                events = pipeline.process_chunk(pcm)

                for event in events:
                    etype = event.get("type")

                    if etype == "word":
                        word = event.get("word", "").strip()
                        if word:
                            logger.debug(f"WORD: {word}")
                        await websocket.send_json(event)

                    elif etype == "partial":
                        await websocket.send_json(event)

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

                    else:
                        await websocket.send_json(event)

            # ── Control frame (0x02) ──────────────────────────────────────
            elif frame_type == 0x02:
                try:
                    ctrl     = json.loads(payload.decode("utf-8"))
                    msg_type = ctrl.get("type")

                    if msg_type == "ai_state":
                        speaking = bool(ctrl.get("speaking", False))

                        # FIX 2: Only act + log when the state actually changes.
                        # The gateway sends ai_state=False from multiple tasks
                        # (synth_worker, play_worker, finalize_turn). Without
                        # this guard the log fills with repeated False entries
                        # and notify_ai_speaking() resets barge-in state
                        # multiple times unnecessarily.
                        if speaking != _last_ai_speaking:
                            _last_ai_speaking = speaking
                            logger.info(f"[ctrl] AI speaking → {speaking}")
                            pipeline.notify_ai_speaking(speaking)
                        # else: silently ignore duplicate

                    elif msg_type == "ai_reference":
                        b64 = ctrl.get("pcm", "")
                        if b64:
                            raw_bytes = base64.b64decode(b64)
                            ref = (
                                np.frombuffer(raw_bytes, dtype=np.int16)
                                .astype(np.float32) / 32768.0
                            )
                            tts_sr = ctrl.get("sample_rate", 24000)
                            if tts_sr != 16000 and len(ref) > 0:
                                try:
                                    from math import gcd as _g
                                    from scipy.signal import resample_poly as _rp
                                    g   = _g(16000, tts_sr)
                                    ref = _rp(ref, 16000//g, tts_sr//g).astype(np.float32)
                                except ImportError:
                                    n   = int(len(ref) * 16000 / tts_sr)
                                    ref = np.interp(
                                        np.linspace(0,1,n),
                                        np.linspace(0,1,len(ref)), ref
                                    ).astype(np.float32)
                            pipeline.push_ai_reference(ref)

                    elif msg_type == "assistant_turn":
                        text = ctrl.get("text", "").strip()
                        if text:
                            pipeline.add_assistant_turn(text)

                    elif msg_type == "reset_context":
                        _reset_asr_context(pipeline)
                        await websocket.send_json({"type": "context_reset"})

                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif msg_type == "get_stats":
                        await websocket.send_json({"type": "stats", **pipeline.get_stats()})

                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"[ctrl] bad payload: {e}")

            # ── Unknown frame type — silently drop ────────────────────────
            # WebSocket library ping (0x89) / pong (0x8a) / close (0x88)
            # frames can surface here as raw bytes. They are not audio
            # or control frames — just ignore them.
            else:
                logger.debug(f"[mux] dropping unknown frame_type=0x{frame_type:02x} len={len(payload)}")

    except WebSocketDisconnect:
        logger.info(f"🔌 WS disconnected: {client}")
        final = pipeline.flush()
        if final:
            clean = _filter_segment(final)
            if clean:
                try:
                    await websocket.send_json({"type": "segment", "text": clean})
                except Exception:
                    pass

    except Exception as e:
        logger.exception(f"[MUX] error: {e}")
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
            await websocket.close()
        except Exception:
            pass


# ── WebSocket /stream/binary ──────────────────────────────────────────────────

@app.websocket("/stream/binary")
async def stream_audio_binary(websocket: WebSocket):
    await websocket.accept()
    pipeline = _get_pipeline()
    pipeline.reset()
    try:
        async for message in websocket.iter_bytes():
            if not message:
                continue
            # int16 needs exactly 2 bytes per sample — drop stray odd byte
            if len(message) % 2 != 0:
                message = message[: len(message) - 1]
            if not message:
                continue
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


# ── Health / Stats ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    pipeline = _pipeline
    vg_ready = (
        pipeline.voice_gate is not None and pipeline.voice_gate.is_ready
        if pipeline else False
    )
    return {
        "status":              "ok",
        "version":             "5.3.0",
        "pipeline_ready":      pipeline is not None,
        "voice_gate_enrolled": vg_ready,
    }


@app.get("/stats")
def stats():
    return _get_pipeline().get_stats()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )