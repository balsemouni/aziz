"""
main.py — Piper TTS WebSocket server.

Endpoint: ws://0.0.0.0:8765/ws/tts  (same URL the gateway already targets)

Protocol (per connection):
  Client → server:  UTF-8 text frame containing JSON
    {
      "text":       "<phrase to synthesize>",
      "voice":      "<speaker name>",          // optional, default: tara
      "language":   "<lang tag>",              // optional, informational only
      "chunk_tone": "tone" | "logic"           // optional, ignored
    }

  Server → client:  one or more binary frames of raw 16-bit signed PCM
                    followed by a single empty binary frame (b"") that signals
                    end-of-stream.

  On JSON parse error or empty text the server sends b"" immediately (no audio).

Run:
    python main.py
    # or
    uvicorn main:app --host 0.0.0.0 --port 8765
"""

import json
import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import piper_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tts_server")

app = FastAPI(title="Piper TTS Server", version="1.0.0")

HOST = "0.0.0.0"
PORT = 8765

# Maximum bytes to send in a single WebSocket binary frame.
# Keeping this below typical WebSocket frame limits avoids fragmentation.
_MAX_FRAME_BYTES = 32 * 1024  # 32 KB


@app.websocket("/ws/tts")
async def tts_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client = ws.client
    log.info(f"Client connected: {client}")

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                break

            # ── Parse request ─────────────────────────────────────────────────
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f"[{client}] non-JSON frame — sending empty terminator")
                await ws.send_bytes(b"")
                continue

            text = (msg.get("text") or "").strip()
            voice = msg.get("voice") or msg.get("speaker") or None
            emotion = msg.get("emotion") or None

            if not text:
                await ws.send_bytes(b"")
                continue

            log.info(f"[{client}] synthesize voice={voice!r} len={len(text)} — {text[:60]!r}")

            # ── Synthesize ────────────────────────────────────────────────────
            try:
                pcm = piper_engine.synthesize(text, voice_name=voice, emotion=emotion)
            except Exception as exc:
                log.error(f"[{client}] synthesis error: {exc}")
                await ws.send_bytes(b"")
                continue

            # ── Stream PCM back in chunks, then signal end-of-stream ──────────
            if pcm:
                for offset in range(0, len(pcm), _MAX_FRAME_BYTES):
                    await ws.send_bytes(pcm[offset : offset + _MAX_FRAME_BYTES])

            # Empty frame = end-of-stream sentinel (gateway polls for this)
            await ws.send_bytes(b"")
            log.info(f"[{client}] done — {len(pcm)} PCM bytes sent")

    except Exception as exc:
        log.error(f"[{client}] handler error: {exc}")
    finally:
        log.info(f"[{client}] disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
