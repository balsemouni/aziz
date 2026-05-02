"""
main.py — Piper / XTTS TTS WebSocket server.

Endpoint: ws://0.0.0.0:8765/ws/tts  (same URL the gateway already targets)

Protocol (per connection):
  Client → server:  UTF-8 text frame containing JSON
    Synthesis request:
    {
      "text":       "<phrase to synthesize>",
      "voice":      "<speaker name>",          // optional, default: tara
      "language":   "<lang tag>",              // optional, "en" | "fr" | "auto"
      "engine":     "xtts" | "piper",          // optional, default: AUTO
      "emotion":    "<tag>",                   // optional, prosody hint
      "speed":      <float>,                   // optional, 0.5 .. 1.5
      "chunk_tone": "tone" | "logic"           // optional, ignored
    }

    Cancel control frame (mid-synth barge-in):
    {"type": "cancel"}

  Server → client:
    First frame (binary 0x02 + JSON UTF-8):  metadata
      {"type":"meta","sample_rate":<int>,"engine":"<piper|xtts>"}
    Then one or more binary frames of raw 16-bit signed PCM chunks.
    Final frame: empty binary frame (b"") = end-of-stream sentinel.

  On JSON parse error or empty text the server sends b"" immediately (no audio).

Run:
    python main.py
    # or
    uvicorn main:app --host 0.0.0.0 --port 8765
"""

import asyncio
import json
import logging
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import piper_engine

# XTTS is optional — load lazily so the server still starts on machines
# without GPU / Coqui-TTS installed.
try:
    import xtts_engine        # type: ignore
    _XTTS_AVAILABLE = True
except Exception as _e:        # noqa: BLE001
    xtts_engine = None         # type: ignore
    _XTTS_AVAILABLE = False
    logging.getLogger("tts_server").warning(
        f"xtts_engine not available — falling back to Piper only ({_e})"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tts_server")

app = FastAPI(title="Piper / XTTS TTS Server", version="2.0.0")

HOST = "0.0.0.0"
PORT = 8765

# Maximum bytes to send in a single WebSocket binary frame.
_MAX_FRAME_BYTES = 32 * 1024  # 32 KB

# Default engine — "piper" by default for stability; set TTS_DEFAULT_ENGINE=xtts
# (or =auto) once XTTS reference wavs are placed under voices/refs/.
DEFAULT_ENGINE = os.getenv("TTS_DEFAULT_ENGINE", "piper").lower().strip()


def _resolve_engine(requested: Optional[str]) -> str:
    req = (requested or DEFAULT_ENGINE or "auto").lower().strip()
    if req == "xtts" and _XTTS_AVAILABLE:
        return "xtts"
    if req == "piper":
        return "piper"
    if req == "auto":
        return "xtts" if _XTTS_AVAILABLE else "piper"
    # Unknown / unavailable -> Piper
    return "piper"


@app.websocket("/ws/tts")
async def tts_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client = ws.client
    log.info(f"Client connected: {client}")

    # Per-connection cancel event — set whenever a {"type":"cancel"} frame
    # arrives mid-synth.  Cleared at the start of every new synth request.
    cancel_event = asyncio.Event()
    # Channel of pending synth requests (parsed JSON dicts).  A single
    # receiver task multiplexes the WebSocket so we can read cancel frames
    # WHILE audio is streaming.
    synth_q: asyncio.Queue = asyncio.Queue()
    closed = asyncio.Event()

    async def _receiver():
        try:
            while not closed.is_set():
                try:
                    raw = await ws.receive_text()
                except WebSocketDisconnect:
                    closed.set()
                    cancel_event.set()
                    await synth_q.put(None)
                    return
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning(f"[{client}] non-JSON frame ignored")
                    continue
                if (msg.get("type") or "").strip().lower() == "cancel":
                    cancel_event.set()
                    log.info(f"[{client}] cancel frame received")
                    continue
                # Treat anything else as a synth request
                await synth_q.put(msg)
        except asyncio.CancelledError:
            return

    async def _send_meta(sample_rate: int, engine: str) -> None:
        meta = json.dumps({
            "type":        "meta",
            "sample_rate": sample_rate,
            "engine":      engine,
        }).encode("utf-8")
        try:
            await ws.send_bytes(b"\x02" + meta)
        except Exception:
            pass

    receiver_task = asyncio.create_task(_receiver())

    try:
        while not closed.is_set():
            msg = await synth_q.get()
            if msg is None:
                break

            text     = (msg.get("text") or "").strip()
            voice    = msg.get("voice") or msg.get("speaker") or None
            emotion  = msg.get("emotion") or None
            language = (msg.get("language") or "").strip().lower() or None
            speed    = msg.get("speed")
            engine   = _resolve_engine(msg.get("engine"))

            if not text:
                await ws.send_bytes(b"")
                continue

            log.info(
                f"[{client}] synth engine={engine} voice={voice!r} "
                f"lang={language!r} len={len(text)} — {text[:60]!r}"
            )

            # Reset cancel state for this request.
            cancel_event.clear()

            # ── Synthesize and stream PCM chunks ──────────────────────────────
            sent_meta = False
            total_bytes = 0
            try:
                if engine == "xtts" and xtts_engine is not None:
                    sr = xtts_engine.sample_rate(voice)
                    await _send_meta(sr, "xtts")
                    sent_meta = True
                    chunk_iter = xtts_engine.stream(
                        text, voice_name=voice, language=language,
                        emotion=emotion, speed=speed,
                    )
                else:
                    sr = piper_engine.sample_rate(voice)
                    await _send_meta(sr, "piper")
                    sent_meta = True
                    chunk_iter = piper_engine.stream(
                        text, voice_name=voice, emotion=emotion, speed=speed,
                    )

                # piper/xtts iterators are sync generators — pull each chunk
                # in a thread so we can interleave cancel checks cleanly.
                loop = asyncio.get_event_loop()
                it = iter(chunk_iter)
                while not cancel_event.is_set():
                    pcm_chunk = await loop.run_in_executor(None, _next_or_none, it)
                    if pcm_chunk is None:
                        break
                    if not pcm_chunk:
                        continue
                    for off in range(0, len(pcm_chunk), _MAX_FRAME_BYTES):
                        if cancel_event.is_set():
                            break
                        await ws.send_bytes(pcm_chunk[off: off + _MAX_FRAME_BYTES])
                    total_bytes += len(pcm_chunk)
            except Exception as exc:
                log.error(f"[{client}] synthesis error ({engine}): {exc}")
                # ── C5: auto-fallback Piper on XTTS failure ─────────────
                if engine == "xtts" and not cancel_event.is_set():
                    try:
                        log.info(f"[{client}] falling back to Piper")
                        if not sent_meta:
                            sr = piper_engine.sample_rate(voice)
                            await _send_meta(sr, "piper")
                        loop = asyncio.get_event_loop()
                        it2 = iter(piper_engine.stream(
                            text, voice_name=voice, emotion=emotion, speed=speed,
                        ))
                        while not cancel_event.is_set():
                            pcm_chunk = await loop.run_in_executor(None, _next_or_none, it2)
                            if pcm_chunk is None:
                                break
                            for off in range(0, len(pcm_chunk), _MAX_FRAME_BYTES):
                                if cancel_event.is_set():
                                    break
                                await ws.send_bytes(pcm_chunk[off: off + _MAX_FRAME_BYTES])
                            total_bytes += len(pcm_chunk)
                    except Exception as exc2:
                        log.error(f"[{client}] Piper fallback also failed: {exc2}")

            # End-of-stream sentinel
            try:
                await ws.send_bytes(b"")
            except Exception:
                closed.set()
                break
            log.info(
                f"[{client}] done — {total_bytes} PCM bytes "
                f"{'(cancelled)' if cancel_event.is_set() else ''}"
            )

    except Exception as exc:
        log.error(f"[{client}] handler error: {exc}")
    finally:
        closed.set()
        if receiver_task and not receiver_task.done():
            receiver_task.cancel()
            try:
                await receiver_task
            except Exception:
                pass
        log.info(f"[{client}] disconnected")


def _next_or_none(it):
    """
    Pull the next chunk from a sync generator.  Returns None on StopIteration
    but RE-RAISES other exceptions so the caller can run the fallback path.
    """
    try:
        return next(it)
    except StopIteration:
        return None


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
