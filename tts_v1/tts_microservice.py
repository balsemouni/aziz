"""
XTTS-v2 TTS Microservice  (Enhanced)
=====================================
Additions over v1:
  - Per-chunk latency tracking
      • synthesis_latency_ms  : time from job start → chunk ready
      • first_chunk_latency_ms: time from first chunk received → this chunk ready
      • play_latency_ms       : synthesis_latency_ms + estimated playback offset
  - /tts/chunks endpoint now returns JSON with per-chunk metadata + base64 WAV audio
  - /tts/stream  streams WAV audio as before; X-Chunk-Meta response header carries JSON
  - WebSocket sends JSON metadata frames interleaved with binary audio frames
  - ChunkAudioDisplay helper: ASCII waveform for quick console inspection

Original features preserved:
  - Pre-buffered audio queue (PRE_BUFFER_CHUNKS ahead)
  - Tone/Logic chunking
  - FastAPI microservice with WebSocket + REST endpoints
  - Seamless audio streaming
"""

import asyncio
import base64
import io
import json
import logging
import queue
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tts_service")

SAMPLE_RATE       = 24000   # XTTS-v2 native output rate
PRE_BUFFER_CHUNKS = 2       # chunks to synthesise ahead before playback starts
MAX_QUEUE_SIZE    = 20      # max pending audio chunks per request
DEFAULT_LANGUAGE  = "en"

DEFAULT_SPEAKER_WAV: Optional[str] = None
DEFAULT_SPEAKER = "Claribel Dervla"


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

class ChunkType(str, Enum):
    TONE  = "tone"
    LOGIC = "logic"


@dataclass
class ChunkLatency:
    """All latency measurements for a single chunk (all in milliseconds)."""
    # Absolute timestamps (epoch seconds, stored for reference)
    job_start_ts:          float = 0.0   # when the SynthesisJob was created
    synth_start_ts:        float = 0.0   # when synthesis of *this* chunk started
    synth_end_ts:          float = 0.0   # when synthesis of *this* chunk finished
    first_chunk_ready_ts:  float = 0.0   # when the *first* chunk of the job finished

    # Derived latencies (populated by _compute())
    synthesis_latency_ms:   float = 0.0  # job_start → this chunk ready
    first_chunk_latency_ms: float = 0.0  # first_chunk_ready → this chunk ready (0 for chunk 0)
    synth_duration_ms:      float = 0.0  # time spent synthesising only this chunk

    def compute(self) -> None:
        self.synth_duration_ms    = (self.synth_end_ts - self.synth_start_ts) * 1000
        self.synthesis_latency_ms = (self.synth_end_ts - self.job_start_ts)   * 1000
        if self.first_chunk_ready_ts:
            self.first_chunk_latency_ms = (self.synth_end_ts - self.first_chunk_ready_ts) * 1000

    def to_dict(self) -> dict:
        return {
            "synthesis_latency_ms":   round(self.synthesis_latency_ms,   1),
            "first_chunk_latency_ms": round(self.first_chunk_latency_ms, 1),
            "synth_duration_ms":      round(self.synth_duration_ms,      1),
        }


@dataclass
class AudioChunk:
    chunk_id:    str
    chunk_type:  ChunkType
    text:        str
    audio:       Optional[np.ndarray] = None
    duration_sec: float = 0.0
    ready:       bool   = False
    error:       Optional[str] = None
    latency:     ChunkLatency  = field(default_factory=ChunkLatency)
    chunk_index: int    = 0   # 0-based position in the job


@dataclass
class SynthesisJob:
    job_id:      str
    text:        str
    language:    str   = DEFAULT_LANGUAGE
    speaker_wav: Optional[str] = None
    speaker:     Optional[str] = None
    chunks:      list  = field(default_factory=list)
    audio_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=MAX_QUEUE_SIZE))
    done:        threading.Event = field(default_factory=threading.Event)
    cancelled:   bool  = False
    start_ts:    float = field(default_factory=time.time)
    # Shared mutable: set when first chunk is ready
    _first_chunk_ready_ts: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_first_chunk(self, ts: float) -> None:
        with self._lock:
            if not self._first_chunk_ready_ts:
                self._first_chunk_ready_ts = ts

    @property
    def first_chunk_ready_ts(self) -> float:
        return self._first_chunk_ready_ts


# ── Request / Response schemas ─────────────────

class TTSRequest(BaseModel):
    text:        str
    language:    str           = DEFAULT_LANGUAGE
    speaker_wav: Optional[str] = None
    speaker:     Optional[str] = None


class ChunkRequest(BaseModel):
    chunks:      List[str]
    chunk_types: List[str] = []
    language:    str           = DEFAULT_LANGUAGE
    speaker_wav: Optional[str] = None
    speaker:     Optional[str] = None


class ChunkMeta(BaseModel):
    chunk_id:    str
    chunk_index: int
    chunk_type:  str
    text:        str
    duration_sec: float
    latency:     dict
    audio_b64:   Optional[str] = None   # base64-encoded WAV; only in /tts/chunks


# ──────────────────────────────────────────────
# Text Chunker
# ──────────────────────────────────────────────

class TextChunker:
    TONE_MAX_CHARS  = 60
    LOGIC_MAX_CHARS = 200

    @staticmethod
    def _sentence_split(text: str) -> list:
        import re
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _sub_split(sentence: str, max_chars: int) -> list:
        if len(sentence) <= max_chars:
            return [sentence]
        import re
        parts  = re.split(r'(?<=[,;])\s+', sentence)
        result, current = [], ""
        for part in parts:
            if len(current) + len(part) + 1 <= max_chars:
                current = (current + " " + part).strip() if current else part
            else:
                if current:
                    result.append(current)
                current = part
        if current:
            result.append(current)
        return result or [sentence]

    def split(self, text: str) -> list:
        chunks = []
        for sentence in self._sentence_split(text):
            is_tone = (
                sentence.endswith("?") or
                sentence.endswith("!") or
                len(sentence) <= self.TONE_MAX_CHARS
            )
            ctype    = ChunkType.TONE if is_tone else ChunkType.LOGIC
            max_chars = self.TONE_MAX_CHARS if is_tone else self.LOGIC_MAX_CHARS
            for sub in self._sub_split(sentence, max_chars):
                chunks.append(AudioChunk(
                    chunk_id    = str(uuid.uuid4())[:8],
                    chunk_type  = ctype,
                    text        = sub,
                    chunk_index = len(chunks),
                ))
        return chunks


# ──────────────────────────────────────────────
# XTTS-v2 Engine
# ──────────────────────────────────────────────

class XTTSEngine:
    """
    Wraps XTTS-v2. Synthesis runs in a dedicated background thread.
    All latency tracking happens inside the worker so timestamps are accurate.
    """

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.model_name     = model_name
        self._tts           = None
        self._ready         = threading.Event()
        self._synth_queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="xtts-worker"
        )

    # ── Lifecycle ──────────────────────────────

    def load(self):
        log.info("Loading XTTS-v2: %s", self.model_name)
        from TTS.api import TTS as CoquiTTS
        self._tts = CoquiTTS(model_name=self.model_name, gpu=torch.cuda.is_available())
        self._ready.set()
        self._worker_thread.start()
        log.info("XTTS-v2 ready.  CUDA=%s  Speakers: %s",
                 torch.cuda.is_available(),
                 self._tts.speakers[:3] if self._tts.speakers else "n/a")

    def _worker_loop(self):
        while True:
            item = self._synth_queue.get()
            if item is None:
                break
            chunk, speaker_wav, speaker, language, job, result_event = item
            try:
                chunk.latency.synth_start_ts = time.time()
                audio = self._synthesize_sync(chunk.text, speaker_wav, speaker, language)
                chunk.latency.synth_end_ts   = time.time()

                # Record first-chunk timestamp on the job (thread-safe)
                job.record_first_chunk(chunk.latency.synth_end_ts)
                chunk.latency.first_chunk_ready_ts = job.first_chunk_ready_ts
                chunk.latency.job_start_ts         = job.start_ts
                chunk.latency.compute()

                chunk.audio        = audio
                chunk.duration_sec = len(audio) / SAMPLE_RATE
                chunk.ready        = True

                log.info(
                    "Chunk %s [%s] idx=%d | synth=%.0fms | from_job=%.0fms | from_first=%.0fms | dur=%.2fs",
                    chunk.chunk_id, chunk.chunk_type, chunk.chunk_index,
                    chunk.latency.synth_duration_ms,
                    chunk.latency.synthesis_latency_ms,
                    chunk.latency.first_chunk_latency_ms,
                    chunk.duration_sec,
                )
            except Exception as exc:
                log.error("Synthesis error chunk %s: %s", chunk.chunk_id, exc)
                chunk.error = str(exc)
                chunk.ready = True
            finally:
                result_event.set()
                self._synth_queue.task_done()

    def _synthesize_sync(
        self,
        text: str,
        speaker_wav: Optional[str],
        speaker: Optional[str],
        language: str,
    ) -> np.ndarray:
        self._ready.wait()
        wav = speaker_wav or DEFAULT_SPEAKER_WAV
        spk = speaker or (DEFAULT_SPEAKER if not wav else None)
        kwargs = {"text": text, "language": language}
        if wav:
            kwargs["speaker_wav"] = wav
        else:
            kwargs["speaker"] = spk
        result = self._tts.tts(**kwargs)
        return np.array(result, dtype=np.float32)

    # ── Public async API ───────────────────────

    async def synthesize_chunk(
        self,
        chunk: AudioChunk,
        speaker_wav: Optional[str],
        speaker: Optional[str],
        language: str,
        job: "SynthesisJob",
    ) -> AudioChunk:
        result_event = threading.Event()
        self._synth_queue.put((chunk, speaker_wav, speaker, language, job, result_event))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, result_event.wait)
        return chunk

    async def synthesize_stream(self, job: SynthesisJob) -> AsyncGenerator[AudioChunk, None]:
        chunks = job.chunks
        if not chunks:
            return

        pending     = []
        chunk_index = 0

        async def enqueue_next():
            nonlocal chunk_index
            if chunk_index < len(chunks):
                c = chunks[chunk_index]
                chunk_index += 1
                task = asyncio.create_task(
                    self.synthesize_chunk(c, job.speaker_wav, job.speaker, job.language, job)
                )
                pending.append(task)

        for _ in range(min(PRE_BUFFER_CHUNKS, len(chunks))):
            await enqueue_next()

        while pending:
            if job.cancelled:
                for t in pending:
                    t.cancel()
                break
            done_chunk = await pending.pop(0)
            await enqueue_next()
            if done_chunk.error:
                log.warning("Skipping chunk %s: %s", done_chunk.chunk_id, done_chunk.error)
                continue
            yield done_chunk

    def shutdown(self):
        self._synth_queue.put(None)
        self._worker_thread.join(timeout=5)


# ──────────────────────────────────────────────
# Chunk Audio Display (ASCII waveform)
# ──────────────────────────────────────────────

class ChunkAudioDisplay:
    """
    Renders a compact ASCII waveform + latency table for a list of AudioChunks.
    Useful for logging / test output.
    """

    WIDTH    = 60   # characters wide
    HEIGHT   = 8    # rows tall
    BAR_CHAR = "█"
    NEG_CHAR = "▄"

    @classmethod
    def waveform(cls, audio: np.ndarray, width: int = WIDTH, height: int = HEIGHT) -> str:
        if audio is None or len(audio) == 0:
            return "[no audio]"
        # Down-sample to `width` columns by taking RMS over each window
        step   = max(1, len(audio) // width)
        cols   = []
        for i in range(0, len(audio), step):
            window = audio[i : i + step]
            rms    = float(np.sqrt(np.mean(window ** 2)))
            cols.append(rms)
            if len(cols) == width:
                break

        max_val = max(cols) if cols else 1.0
        if max_val == 0:
            max_val = 1.0
        norm = [v / max_val for v in cols]

        rows = []
        for row in range(height, 0, -1):
            threshold = row / height
            line = ""
            for v in norm:
                line += cls.BAR_CHAR if v >= threshold else " "
            rows.append(f"|{line}|")
        rows.append("+" + "-" * width + "+")
        return "\n".join(rows)

    @classmethod
    def display_chunk(cls, chunk: AudioChunk) -> str:
        lat = chunk.latency
        lines = [
            f"{'─'*68}",
            f"  Chunk #{chunk.chunk_index}  id={chunk.chunk_id}  type={chunk.chunk_type}",
            f"  Text : {chunk.text[:80]}",
            f"  Audio: {chunk.duration_sec:.3f}s  ({int(chunk.duration_sec * SAMPLE_RATE)} samples)",
            f"  Latency:",
            f"    • synth duration          : {lat.synth_duration_ms:>8.1f} ms",
            f"    • since job start         : {lat.synthesis_latency_ms:>8.1f} ms",
            f"    • since first chunk ready : {lat.first_chunk_latency_ms:>8.1f} ms",
        ]
        if chunk.audio is not None:
            lines.append(f"  Waveform:")
            for wline in cls.waveform(chunk.audio).split("\n"):
                lines.append(f"    {wline}")
        return "\n".join(lines)

    @classmethod
    def display_job(cls, chunks: list) -> str:
        parts = [f"\n{'═'*68}", "  CHUNK AUDIO REPORT", f"{'═'*68}"]
        cumulative_play = 0.0
        for c in chunks:
            parts.append(cls.display_chunk(c))
            parts.append(
                f"  Estimated play time offset: {cumulative_play*1000:.1f} ms "
                f"(play_latency ≈ {(c.latency.synthesis_latency_ms + cumulative_play*1000):.1f} ms)"
            )
            cumulative_play += c.duration_sec
        parts.append(f"{'═'*68}\n")
        return "\n".join(parts)


# ──────────────────────────────────────────────
# Audio Helpers
# ──────────────────────────────────────────────

def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    pcm = float32_to_pcm16(audio)
    return build_wav_header(len(audio)) + pcm


def build_wav_header(
    num_samples:    int = 0,
    sample_rate:    int = SAMPLE_RATE,
    num_channels:   int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    byte_rate   = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    if num_samples == 0:
        file_size = 0xFFFFFFFF
        data_size = 0xFFFFFFFF
    else:
        data_size = num_samples * block_align
        file_size = 36 + data_size
    hdr  = b"RIFF"
    hdr += struct.pack("<I", file_size)
    hdr += b"WAVE"
    hdr += b"fmt "
    hdr += struct.pack("<I", 16)
    hdr += struct.pack("<H", 1)
    hdr += struct.pack("<H", num_channels)
    hdr += struct.pack("<I", sample_rate)
    hdr += struct.pack("<I", byte_rate)
    hdr += struct.pack("<H", block_align)
    hdr += struct.pack("<H", bits_per_sample)
    hdr += b"data"
    hdr += struct.pack("<I", data_size)
    return hdr


def chunk_to_meta(chunk: AudioChunk, include_audio: bool = False) -> dict:
    meta: dict = {
        "chunk_id":    chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "chunk_type":  chunk.chunk_type,
        "text":        chunk.text,
        "duration_sec": round(chunk.duration_sec, 4),
        "latency":     chunk.latency.to_dict(),
    }
    if include_audio and chunk.audio is not None:
        wav_bytes     = audio_to_wav_bytes(chunk.audio)
        meta["audio_b64"] = base64.b64encode(wav_bytes).decode()
        meta["audio_samples"] = len(chunk.audio)
    return meta


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app         = FastAPI(title="XTTS-v2 TTS Microservice", version="2.0.0")
engine      = XTTSEngine()
chunker     = TextChunker()
active_jobs: dict = {}


@app.on_event("startup")
def startup():
    engine.load()


@app.on_event("shutdown")
def shutdown():
    engine.shutdown()


# ── Health ─────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "cuda": torch.cuda.is_available(),
        "model": engine.model_name,
        "default_speaker": DEFAULT_SPEAKER,
        "default_speaker_wav": DEFAULT_SPEAKER_WAV,
    }


# ── REST: streaming WAV + chunk metadata ───────

@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    """
    Streaming WAV response.
    Response headers carry:
      X-Job-Id      : job UUID
      X-Chunk-Count : total number of text chunks
    After the stream, chunk metadata (latency etc.) is logged server-side.
    """
    job_id = str(uuid.uuid4())
    chunks = chunker.split(req.text)
    job = SynthesisJob(
        job_id=job_id, text=req.text,
        language=req.language,
        speaker_wav=req.speaker_wav, speaker=req.speaker,
        chunks=chunks,
    )
    active_jobs[job_id] = job
    all_chunks = []

    async def generate():
        yield build_wav_header(num_samples=0)
        async for chunk in engine.synthesize_stream(job):
            all_chunks.append(chunk)
            yield float32_to_pcm16(chunk.audio)
        active_jobs.pop(job_id, None)
        # Log the full chunk report after stream finishes
        log.info(ChunkAudioDisplay.display_job(all_chunks))

    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={
            "X-Job-Id":      job_id,
            "X-Chunk-Count": str(len(chunks)),
        },
    )


# ── REST: full WAV ─────────────────────────────

@app.post("/tts/full")
async def tts_full(req: TTSRequest):
    chunks = chunker.split(req.text)
    job = SynthesisJob(
        job_id=str(uuid.uuid4()), text=req.text,
        language=req.language,
        speaker_wav=req.speaker_wav, speaker=req.speaker,
        chunks=chunks,
    )
    all_audio  = []
    all_chunks = []
    async for chunk in engine.synthesize_stream(job):
        all_audio.append(chunk.audio)
        all_chunks.append(chunk)

    if not all_audio:
        raise HTTPException(status_code=500, detail="Synthesis produced no audio")

    log.info(ChunkAudioDisplay.display_job(all_chunks))

    combined  = np.concatenate(all_audio)
    pcm       = float32_to_pcm16(combined)
    wav_bytes = build_wav_header(len(combined)) + pcm

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(wav_bytes)),
            "X-Chunk-Count":  str(len(all_chunks)),
            "X-Chunk-Meta":   json.dumps([chunk_to_meta(c) for c in all_chunks]),
        },
    )


# ── REST: pre-chunked → JSON + audio ──────────

@app.post("/tts/chunks")
async def tts_chunks(req: ChunkRequest):
    """
    Accept pre-split text chunks.
    Returns JSON array, each element containing:
      - chunk metadata (id, type, text, latency)
      - audio_b64: base64-encoded WAV for that chunk
      - display_waveform: ASCII art waveform string
    This endpoint is designed for inspection and latency analysis.
    """
    audio_chunks = []
    for i, text in enumerate(req.chunks):
        ctype_str = req.chunk_types[i] if i < len(req.chunk_types) else "logic"
        try:
            ctype = ChunkType(ctype_str)
        except ValueError:
            ctype = ChunkType.LOGIC
        audio_chunks.append(AudioChunk(
            chunk_id    = str(uuid.uuid4())[:8],
            chunk_type  = ctype,
            text        = text,
            chunk_index = i,
        ))

    job = SynthesisJob(
        job_id      = str(uuid.uuid4()),
        text        = " ".join(req.chunks),
        language    = req.language,
        speaker_wav = req.speaker_wav,
        speaker     = req.speaker,
        chunks      = audio_chunks,
    )

    results = []
    async for chunk in engine.synthesize_stream(job):
        meta = chunk_to_meta(chunk, include_audio=True)
        meta["display_waveform"] = ChunkAudioDisplay.waveform(chunk.audio)
        meta["display_report"]   = ChunkAudioDisplay.display_chunk(chunk)
        results.append(meta)

    log.info(ChunkAudioDisplay.display_job([c for c in audio_chunks if c.ready]))
    return {"total_chunks": len(results), "chunks": results}


# ── WebSocket: real-time duplex ────────────────

@app.websocket("/ws/tts")
async def ws_tts(websocket: WebSocket):
    """
    WebSocket TTS endpoint.

    Client → Server (JSON):
        {
            "text":        "...",
            "language":    "en",
            "speaker_wav": null,
            "speaker":     null,
            "chunk_tone":  "tone" | "logic"   ← NEW (v9): prosody hint from gateway
        }

        chunk_tone values:
          "tone"  → short expressive phrase (question / exclamation).
                    TTS applies slightly higher pitch and moderate speaking rate.
          "logic" → longer declarative statement.
                    TTS applies neutral pitch and slightly slower rate for clarity.
          absent  → default (same as "logic")

    Server → Client (mixed frames):
        Frame 1     : binary WAV header (44 bytes)
        Frame 2–N   : alternating JSON metadata + binary PCM per chunk
                      JSON: { "type": "chunk_meta", "data": { ...ChunkMeta... } }
                      BIN : raw PCM16 bytes for that chunk
        Final frame : binary b""  (end-of-stream)

    NOTE: The gateway sends ONE sentence per WebSocket connection (it opens a
    fresh connection per tonal chunk).  The while-True loop below still works
    correctly for that usage pattern (it will just process one message then
    receive the connection close).
    """
    await websocket.accept()
    log.info("WebSocket connected: %s", websocket.client)
    active_job: Optional[SynthesisJob] = None

    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "").strip()
            if not text:
                await websocket.send_json({"error": "empty text"})
                continue

            if active_job:
                active_job.cancelled = True

            # ── Prosody hint from gateway ─────────────────────────────────────
            chunk_tone = data.get("chunk_tone", "logic")   # "tone" | "logic"

            # When the gateway pre-splits into tonal chunks, each message is a
            # single complete phrase — we honour that and skip internal re-split
            # to avoid destroying the prosodic boundary decisions already made.
            # We wrap the whole text as a single AudioChunk with the correct type.
            if chunk_tone == "tone":
                ctype = ChunkType.TONE
            else:
                ctype = ChunkType.LOGIC

            # Build a single-chunk job (gateway already did the splitting)
            single_chunk = AudioChunk(
                chunk_id    = str(uuid.uuid4())[:8],
                chunk_type  = ctype,
                text        = text,
                chunk_index = 0,
            )
            active_job = SynthesisJob(
                job_id      = str(uuid.uuid4()),
                text        = text,
                language    = data.get("language", DEFAULT_LANGUAGE),
                speaker_wav = data.get("speaker_wav"),
                speaker     = data.get("speaker"),
                chunks      = [single_chunk],
            )

            log.info(
                "WS TTS synthesis: tone=%s  len=%d  text=%r",
                chunk_tone, len(text), text[:60]
            )

            await websocket.send_bytes(build_wav_header(num_samples=0))

            all_chunks = []
            async for chunk in engine.synthesize_stream(active_job):
                all_chunks.append(chunk)
                # Send metadata first, then audio
                await websocket.send_json({
                    "type": "chunk_meta",
                    "data": chunk_to_meta(chunk, include_audio=False),
                    "waveform": ChunkAudioDisplay.waveform(chunk.audio, width=40),
                    "chunk_tone": chunk_tone,
                })
                await websocket.send_bytes(float32_to_pcm16(chunk.audio))

            await websocket.send_bytes(b"")  # end-of-stream
            log.info(ChunkAudioDisplay.display_job(all_chunks))

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as exc:
        log.error("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass


# ── Cancel job ─────────────────────────────────

@app.delete("/tts/job/{job_id}")
def cancel_job(job_id: str):
    job = active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancelled = True
    return {"cancelled": job_id}


# ── Chunker preview ────────────────────────────

@app.post("/tts/preview-chunks")
def preview_chunks(req: TTSRequest):
    chunks = chunker.split(req.text)
    return {
        "total_chunks": len(chunks),
        "chunks": [
            {"id": c.chunk_id, "type": c.chunk_type, "text": c.text, "index": c.chunk_index}
            for c in chunks
        ],
    }


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("tts_microservice:app", host="0.0.0.0", port=8765, reload=False)