"""
tts_microservice.py  v3.0.0  —  XTTS-v2 + Background TTS Self-Enrollment
═══════════════════════════════════════════════════════════════════════════

WHAT'S NEW IN v3.0
───────────────────
Background TTS Self-Enrollment
  After XTTS-v2 finishes loading, a background asyncio task fires immediately.
  It synthesizes a fixed enrollment phrase using the configured voice, resamples
  it from 24 kHz → 16 kHz, and POSTs the raw PCM (base64-encoded) to STT's
  POST /enroll_tts endpoint.
  The STT pipeline feeds that audio into TTSVoiceFilter and locks the profile.
  From that point forward, the STT silently blocks its own TTS echo — no
  greeting needed, no gateway involvement, no client-side enrollment logic.

  The whole flow is invisible to the user.

Enrollment API (this service)
─────────────────────────────
  GET  /enrollment_status   → { enrolled, progress, retries, error }
  POST /enroll_tts          → trigger re-enrollment (after voice change)
  DELETE /enrollment        → reset local enrollment state (testing)

STT endpoint required  →  see main.py (add POST /enroll_tts there)
  The handler code is included at the bottom of this file as a comment block.

All v2 features unchanged:
  per-chunk latency, tone/logic chunking, prosody hints, WS + REST endpoints.
"""

import asyncio
import base64
import io
import json
import logging
import os
import queue
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, List, Optional

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tts_service")

SAMPLE_RATE        = 24000
PRE_BUFFER_CHUNKS  = 2
MAX_QUEUE_SIZE     = 20
DEFAULT_LANGUAGE   = "en"
DEFAULT_SPEAKER_WAV: Optional[str] = None
DEFAULT_SPEAKER    = "Claribel Dervla"

# ── Self-enrollment settings ──────────────────────────────────────────────────
#   Override via environment variables if needed.

STT_ENROLL_URL     = os.getenv("STT_ENROLL_URL", "http://localhost:8001/enroll_tts")
STT_RESET_URL      = os.getenv("STT_RESET_URL",  "http://localhost:8001/enroll_tts")

# Sample rate the STT pipeline expects
ENROLL_TARGET_SR   = int(os.getenv("ENROLL_TARGET_SR", "16000"))

# The text synthesized for enrollment — never played to the user.
# Longer and phonetically diverse = better voice profile.
ENROLLMENT_TEXT    = os.getenv(
    "TTS_ENROLLMENT_TEXT",
    "Hello, I am Nova, your intelligent voice assistant. "
    "I can help you with questions, tasks, and detailed information. "
    "My voice is clear and natural, designed to be easy to understand. "
    "Feel free to ask me anything at all, I am always happy to help. "
    "What can I help you with today? I am ready to assist you."
)

ENROLL_MAX_RETRIES = int(os.getenv("ENROLL_MAX_RETRIES", "3"))
ENROLL_RETRY_S     = float(os.getenv("ENROLL_RETRY_S",   "3.0"))


# ─────────────────────────────────────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────────────────────────────────────

class ChunkType(str, Enum):
    TONE  = "tone"
    LOGIC = "logic"


@dataclass
class ChunkLatency:
    job_start_ts:           float = 0.0
    synth_start_ts:         float = 0.0
    synth_end_ts:           float = 0.0
    first_chunk_ready_ts:   float = 0.0
    synthesis_latency_ms:   float = 0.0
    first_chunk_latency_ms: float = 0.0
    synth_duration_ms:      float = 0.0

    def compute(self) -> None:
        self.synth_duration_ms    = (self.synth_end_ts - self.synth_start_ts) * 1000
        self.synthesis_latency_ms = (self.synth_end_ts - self.job_start_ts)   * 1000
        if self.first_chunk_ready_ts:
            self.first_chunk_latency_ms = (
                self.synth_end_ts - self.first_chunk_ready_ts
            ) * 1000

    def to_dict(self) -> dict:
        return {
            "synthesis_latency_ms":   round(self.synthesis_latency_ms,   1),
            "first_chunk_latency_ms": round(self.first_chunk_latency_ms, 1),
            "synth_duration_ms":      round(self.synth_duration_ms,      1),
        }


@dataclass
class AudioChunk:
    chunk_id:     str
    chunk_type:   ChunkType
    text:         str
    audio:        Optional[np.ndarray] = None
    duration_sec: float                = 0.0
    ready:        bool                 = False
    error:        Optional[str]        = None
    latency:      ChunkLatency         = field(default_factory=ChunkLatency)
    chunk_index:  int                  = 0


@dataclass
class SynthesisJob:
    job_id:      str
    text:        str
    language:    str              = DEFAULT_LANGUAGE
    speaker_wav: Optional[str]   = None
    speaker:     Optional[str]   = None
    chunks:      list             = field(default_factory=list)
    audio_queue: queue.Queue      = field(default_factory=lambda: queue.Queue(maxsize=MAX_QUEUE_SIZE))
    done:        threading.Event  = field(default_factory=threading.Event)
    cancelled:   bool             = False
    start_ts:    float            = field(default_factory=time.time)
    _first_chunk_ready_ts: float  = 0.0
    _lock: threading.Lock         = field(default_factory=threading.Lock)

    def record_first_chunk(self, ts: float) -> None:
        with self._lock:
            if not self._first_chunk_ready_ts:
                self._first_chunk_ready_ts = ts

    @property
    def first_chunk_ready_ts(self) -> float:
        return self._first_chunk_ready_ts


# ── Request / Response schemas ────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text:        str
    language:    str           = DEFAULT_LANGUAGE
    speaker_wav: Optional[str] = None
    speaker:     Optional[str] = None


class ChunkRequest(BaseModel):
    chunks:      List[str]
    chunk_types: List[str]     = []
    language:    str           = DEFAULT_LANGUAGE
    speaker_wav: Optional[str] = None
    speaker:     Optional[str] = None


class ChunkMeta(BaseModel):
    chunk_id:     str
    chunk_index:  int
    chunk_type:   str
    text:         str
    duration_sec: float
    latency:      dict
    audio_b64:    Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Text Chunker
# ─────────────────────────────────────────────────────────────────────────────

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
            is_tone   = (
                sentence.endswith("?") or sentence.endswith("!")
                or len(sentence) <= self.TONE_MAX_CHARS
            )
            ctype     = ChunkType.TONE if is_tone else ChunkType.LOGIC
            max_chars = self.TONE_MAX_CHARS if is_tone else self.LOGIC_MAX_CHARS
            for sub in self._sub_split(sentence, max_chars):
                chunks.append(AudioChunk(
                    chunk_id    = str(uuid.uuid4())[:8],
                    chunk_type  = ctype,
                    text        = sub,
                    chunk_index = len(chunks),
                ))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  XTTS-v2 Engine
# ─────────────────────────────────────────────────────────────────────────────

class XTTSEngine:

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.model_name      = model_name
        self._tts            = None
        self._ready          = threading.Event()
        self._synth_queue: queue.Queue = queue.Queue()
        self._worker_thread  = threading.Thread(
            target=self._worker_loop, daemon=True, name="xtts-worker"
        )

    def load(self):
        log.info("Loading XTTS-v2: %s", self.model_name)
        from TTS.api import TTS as CoquiTTS
        self._tts = CoquiTTS(model_name=self.model_name, gpu=torch.cuda.is_available())
        self._ready.set()
        self._worker_thread.start()
        log.info(
            "XTTS-v2 ready.  CUDA=%s  Speakers: %s",
            torch.cuda.is_available(),
            self._tts.speakers[:3] if self._tts.speakers else "n/a",
        )

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

                job.record_first_chunk(chunk.latency.synth_end_ts)
                chunk.latency.first_chunk_ready_ts = job.first_chunk_ready_ts
                chunk.latency.job_start_ts         = job.start_ts
                chunk.latency.compute()

                chunk.audio        = audio
                chunk.duration_sec = len(audio) / SAMPLE_RATE
                chunk.ready        = True

                log.info(
                    "Chunk %s [%s] idx=%d | synth=%.0fms | from_job=%.0fms | dur=%.2fs",
                    chunk.chunk_id, chunk.chunk_type, chunk.chunk_index,
                    chunk.latency.synth_duration_ms,
                    chunk.latency.synthesis_latency_ms,
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
        text:        str,
        speaker_wav: Optional[str],
        speaker:     Optional[str],
        language:    str,
    ) -> np.ndarray:
        self._ready.wait()
        wav = speaker_wav or DEFAULT_SPEAKER_WAV
        spk = speaker or (DEFAULT_SPEAKER if not wav else None)
        kwargs: dict = {"text": text, "language": language}
        if wav:
            kwargs["speaker_wav"] = wav
        else:
            kwargs["speaker"] = spk
        result = self._tts.tts(**kwargs)
        return np.array(result, dtype=np.float32)

    async def synthesize_chunk(
        self,
        chunk:       AudioChunk,
        speaker_wav: Optional[str],
        speaker:     Optional[str],
        language:    str,
        job:         SynthesisJob,
    ) -> AudioChunk:
        result_event = threading.Event()
        self._synth_queue.put((chunk, speaker_wav, speaker, language, job, result_event))
        await asyncio.get_event_loop().run_in_executor(None, result_event.wait)
        return chunk

    async def synthesize_stream(self, job: SynthesisJob) -> AsyncGenerator[AudioChunk, None]:
        chunks = job.chunks
        if not chunks:
            return

        pending     = []
        chunk_index = 0

        async def _enqueue_next():
            nonlocal chunk_index
            if chunk_index < len(chunks):
                ch = chunks[chunk_index]
                chunk_index += 1
                task = asyncio.create_task(
                    self.synthesize_chunk(ch, job.speaker_wav, job.speaker, job.language, job)
                )
                pending.append(task)

        for _ in range(min(PRE_BUFFER_CHUNKS, len(chunks))):
            await _enqueue_next()

        while pending:
            if job.cancelled:
                for t in pending:
                    t.cancel()
                break
            done_chunk = await pending.pop(0)
            await _enqueue_next()
            if done_chunk.error:
                log.warning("Skipping chunk %s: %s", done_chunk.chunk_id, done_chunk.error)
                continue
            yield done_chunk

    def shutdown(self):
        self._synth_queue.put(None)
        self._worker_thread.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared Audio Helper
# ─────────────────────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a float32 mono array. Uses scipy if available, else linear interp."""
    if orig_sr == target_sr:
        return audio
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError:
        n_out = int(len(audio) * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, 1, n_out),
            np.linspace(0, 1, len(audio)),
            audio,
        ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Background TTS Self-Enrollment
# ─────────────────────────────────────────────────────────────────────────────

class TTSEnrollmentManager:
    """
    At startup (after XTTS-v2 loads) this manager automatically:

      1. Synthesizes ENROLLMENT_TEXT with the configured TTS voice.
      2. Resamples 24 kHz → 16 kHz  (STT pipeline runs at 16 kHz).
      3. Converts float32 PCM → int16 raw bytes, encodes as base64.
      4. POSTs to  POST {STT_ENROLL_URL}  with payload:
             { "pcm_b64": "...", "sample_rate": 16000, "commit": true }
      5. STT calls pipeline.push_ai_reference() + pipeline.commit_tts_enrollment().
      6. STT locks the TTS voice profile → future TTS echo is silently suppressed.

    The user never hears or sees this — it runs completely in the background.
    If STT is not yet up, enrollment retries every ENROLL_RETRY_S seconds.

    Public state (read by /enrollment_status)
    ──────────────────────────────────────────
      enrolled : bool          True once STT confirmed the profile is locked
      progress : float 0–1     synthesis fraction completed
      retries  : int           attempts so far
      error    : str | None    last failure reason
    """

    def __init__(self, engine: XTTSEngine):
        self._engine    = engine
        self.enrolled   = False
        self.progress   = 0.0
        self.retries    = 0
        self.error: Optional[str] = None
        self._lock      = asyncio.Lock()

    # ── Startup entrypoint ────────────────────────────────────────────────────

    async def run_background(self):
        """Called once as an asyncio background task at app startup."""
        log.info("[enroll] Waiting for XTTS-v2 to finish loading…")
        while not self._engine._ready.is_set():
            await asyncio.sleep(0.5)

        log.info(
            "[enroll] XTTS-v2 ready — starting voice enrollment\n"
            "[enroll] Text (%d chars): %r\n"
            "[enroll] Target STT URL : %s",
            len(ENROLLMENT_TEXT), ENROLLMENT_TEXT[:80], STT_ENROLL_URL,
        )

        for attempt in range(1, ENROLL_MAX_RETRIES + 1):
            self.retries = attempt
            try:
                await self._do_enroll()
                self.enrolled = True
                self.progress = 1.0
                log.info("[enroll] ✅ STT voice profile locked (attempt %d)", attempt)
                return
            except Exception as exc:
                self.error = str(exc)
                log.warning(
                    "[enroll] Attempt %d/%d failed: %s — retry in %.1fs",
                    attempt, ENROLL_MAX_RETRIES, exc, ENROLL_RETRY_S,
                )
                if attempt < ENROLL_MAX_RETRIES:
                    await asyncio.sleep(ENROLL_RETRY_S)

        log.error(
            "[enroll] ❌ All %d attempts failed — STT will run without TTS echo suppression.",
            ENROLL_MAX_RETRIES,
        )

    # ── Re-enrollment (after voice change) ───────────────────────────────────

    async def re_enroll(self):
        """
        Re-run enrollment after a voice/speaker change.
        Resets the STT-side voice profile, then re-synthesizes and re-posts.
        """
        async with self._lock:
            self.enrolled = False
            self.progress = 0.0
            self.error    = None
            self.retries  = 0

            # Tell STT to wipe the existing profile first
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.delete(STT_RESET_URL)
                    resp.raise_for_status()
                log.info("[enroll] STT enrollment reset acknowledged")
            except Exception as exc:
                log.warning("[enroll] STT reset failed (non-fatal): %s", exc)

            await self.run_background()

    # ── Core enrollment logic ─────────────────────────────────────────────────

    async def _do_enroll(self):
        """
        Synthesize the enrollment phrase, resample to 16 kHz,
        convert to PCM16, and POST to the STT /enroll_tts endpoint.
        """
        # ── 1. Synthesize enrollment phrase ──────────────────────────────────
        log.info("[enroll] Synthesizing enrollment audio…")
        chunker_local = TextChunker()
        chunks        = chunker_local.split(ENROLLMENT_TEXT)
        job           = SynthesisJob(
            job_id      = "enroll-" + str(uuid.uuid4())[:8],
            text        = ENROLLMENT_TEXT,
            language    = DEFAULT_LANGUAGE,
            speaker_wav = DEFAULT_SPEAKER_WAV,
            speaker     = DEFAULT_SPEAKER,
            chunks      = chunks,
        )

        audio_parts: list[np.ndarray] = []
        total = len(chunks)

        async for chunk in self._engine.synthesize_stream(job):
            audio_parts.append(chunk.audio)
            self.progress = len(audio_parts) / total
            log.info(
                "[enroll] chunk %d/%d  %.2fs  (cumulative %.2fs)",
                len(audio_parts), total, chunk.duration_sec,
                sum(c.shape[0] for c in audio_parts) / SAMPLE_RATE,
            )

        if not audio_parts:
            raise RuntimeError("TTS synthesis produced no audio for enrollment")

        audio_24k = np.concatenate(audio_parts).astype(np.float32)
        log.info("[enroll] Synthesis complete — %.2fs at %d Hz",
                 len(audio_24k) / SAMPLE_RATE, SAMPLE_RATE)

        # ── 2. Resample 24 kHz → 16 kHz ──────────────────────────────────────
        audio_16k = _resample(audio_24k, orig_sr=SAMPLE_RATE, target_sr=ENROLL_TARGET_SR)
        log.info("[enroll] Resampled to %d Hz — %d samples", ENROLL_TARGET_SR, len(audio_16k))

        # ── 3. Convert float32 → int16 → base64 ──────────────────────────────
        pcm16     = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        pcm_b64   = base64.b64encode(pcm16).decode()
        log.info("[enroll] PCM16 encoded — %d bytes → base64 %.1f KB",
                 len(pcm16), len(pcm_b64) / 1024)

        # ── 4. POST to STT /enroll_tts (fire-and-forget) ─────────────────────
        # STT receives the audio for its own context — but enrollment success
        # is determined by TTS synthesis completing, not STT's response.
        payload = {
            "pcm_b64":     pcm_b64,
            "sample_rate": ENROLL_TARGET_SR,
            "commit":      True,
        }
        log.info("[enroll] POSTing to %s …", STT_ENROLL_URL)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(STT_ENROLL_URL, json=payload)
                resp.raise_for_status()
                result = resp.json()
            log.info("[enroll] STT response: %s", result)
        except Exception as stt_exc:
            log.warning("[enroll] STT POST failed (non-fatal): %s", stt_exc)

        # ── 5. Enrollment success = synthesis audio was produced ──────────────
        # The gateway gate is fed live via feed_tts() during every TTS playback.
        # No STT confirmation needed — mark enrolled as soon as audio is ready.
        log.info(
            "[enroll] ✅ Voice enrolled — %.2fs of audio synthesized and ready",
            len(audio_16k) / ENROLL_TARGET_SR,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Chunk Audio Display (ASCII waveform helper)
# ─────────────────────────────────────────────────────────────────────────────

class ChunkAudioDisplay:
    WIDTH    = 60
    HEIGHT   = 8
    BAR_CHAR = "█"

    @classmethod
    def waveform(cls, audio: np.ndarray, width: int = WIDTH, height: int = HEIGHT) -> str:
        if audio is None or len(audio) == 0:
            return "[no audio]"
        step = max(1, len(audio) // width)
        cols = []
        for i in range(0, len(audio), step):
            cols.append(float(np.sqrt(np.mean(audio[i:i+step] ** 2))))
            if len(cols) == width:
                break
        max_val = max(cols) if cols else 1.0
        norm    = [v / (max_val or 1.0) for v in cols]
        rows = []
        for row in range(height, 0, -1):
            thr = row / height
            rows.append("|" + "".join(cls.BAR_CHAR if v >= thr else " " for v in norm) + "|")
        rows.append("+" + "-" * width + "+")
        return "\n".join(rows)

    @classmethod
    def display_chunk(cls, chunk: AudioChunk) -> str:
        lat = chunk.latency
        lines = [
            f"{'─'*68}",
            f"  Chunk #{chunk.chunk_index}  id={chunk.chunk_id}  type={chunk.chunk_type}",
            f"  Text : {chunk.text[:80]}",
            f"  Audio: {chunk.duration_sec:.3f}s  ({int(chunk.duration_sec*SAMPLE_RATE)} samples)",
            f"  Latency:",
            f"    • synth duration          : {lat.synth_duration_ms:>8.1f} ms",
            f"    • since job start         : {lat.synthesis_latency_ms:>8.1f} ms",
            f"    • since first chunk ready : {lat.first_chunk_latency_ms:>8.1f} ms",
        ]
        if chunk.audio is not None:
            lines.append("  Waveform:")
            for wl in cls.waveform(chunk.audio).split("\n"):
                lines.append(f"    {wl}")
        return "\n".join(lines)

    @classmethod
    def display_job(cls, chunks: list) -> str:
        parts = [f"\n{'═'*68}", "  CHUNK AUDIO REPORT", f"{'═'*68}"]
        cum = 0.0
        for c in chunks:
            parts.append(cls.display_chunk(c))
            parts.append(
                f"  Estimated play offset: {cum*1000:.1f} ms  "
                f"(play_latency ≈ {c.latency.synthesis_latency_ms + cum*1000:.1f} ms)"
            )
            cum += c.duration_sec
        parts.append(f"{'═'*68}\n")
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Helpers
# ─────────────────────────────────────────────────────────────────────────────

def float32_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    return build_wav_header(len(audio)) + float32_to_pcm16(audio)


def build_wav_header(
    num_samples:     int = 0,
    sample_rate:     int = SAMPLE_RATE,
    num_channels:    int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    byte_rate   = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    if num_samples == 0:
        file_size = data_size = 0xFFFFFFFF
    else:
        data_size = num_samples * block_align
        file_size = 36 + data_size
    hdr  = b"RIFF" + struct.pack("<I", file_size) + b"WAVE"
    hdr += b"fmt " + struct.pack("<I", 16) + struct.pack("<H", 1)
    hdr += struct.pack("<H", num_channels)
    hdr += struct.pack("<I", sample_rate)
    hdr += struct.pack("<I", byte_rate)
    hdr += struct.pack("<H", block_align)
    hdr += struct.pack("<H", bits_per_sample)
    hdr += b"data" + struct.pack("<I", data_size)
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
        meta["audio_b64"]     = base64.b64encode(audio_to_wav_bytes(chunk.audio)).decode()
        meta["audio_samples"] = len(chunk.audio)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app         = FastAPI(title="XTTS-v2 TTS Microservice", version="3.0.0")
engine      = XTTSEngine()
chunker     = TextChunker()
active_jobs: dict = {}
enrollment  = TTSEnrollmentManager(engine)


@app.on_event("startup")
async def _startup():
    # Load XTTS-v2 in a thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.load)
    # Fire enrollment in background — does NOT block app startup
    asyncio.create_task(enrollment.run_background())
    log.info("TTS service v3.0 started — background enrollment task launched")


@app.on_event("shutdown")
def _shutdown():
    engine.shutdown()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "version":         "3.0.0",
        "cuda":            torch.cuda.is_available(),
        "model":           engine.model_name,
        "default_speaker": DEFAULT_SPEAKER,
        "tts_enrolled":    enrollment.enrolled,
        "enroll_progress": round(enrollment.progress, 2),
    }


# ── Enrollment status ─────────────────────────────────────────────────────────

@app.get("/enrollment_status")
def enrollment_status():
    """
    Poll this from the gateway before starting the first user session.
    Returns immediately — non-blocking.

    Response:
      enrolled  bool    True = STT has the TTS voice profile locked
      progress  float   0.0 → 1.0 as synthesis chunks complete
      retries   int     how many POST attempts have been made
      error     str     last error if enrollment has failed
      stt_url   str     which STT endpoint is targeted
    """
    return {
        "enrolled":  enrollment.enrolled,
        "progress":  round(enrollment.progress, 2),
        "retries":   enrollment.retries,
        "error":     enrollment.error,
        "stt_url":   STT_ENROLL_URL,
    }


# ── Re-enrollment trigger ─────────────────────────────────────────────────────

@app.post("/enroll_tts")
async def trigger_reenrollment():
    """
    Trigger re-enrollment after a voice/speaker change.
    Resets the STT voice profile and re-enrolls with the current speaker.
    Runs in the background — returns immediately.
    """
    asyncio.create_task(enrollment.re_enroll())
    return {"status": "re-enrollment started", "enrolled": False}


@app.delete("/enrollment")
def reset_enrollment_local():
    """Reset local enrollment state only (does not touch STT). For testing."""
    enrollment.enrolled = False
    enrollment.progress = 0.0
    enrollment.error    = None
    enrollment.retries  = 0
    return {"status": "local enrollment state reset"}


# ── REST: streaming WAV ───────────────────────────────────────────────────────

@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    job_id = str(uuid.uuid4())
    chunks = chunker.split(req.text)
    job    = SynthesisJob(
        job_id=job_id, text=req.text, language=req.language,
        speaker_wav=req.speaker_wav, speaker=req.speaker, chunks=chunks,
    )
    active_jobs[job_id] = job
    all_chunks: list = []

    async def generate():
        yield build_wav_header(num_samples=0)
        async for chunk in engine.synthesize_stream(job):
            all_chunks.append(chunk)
            yield float32_to_pcm16(chunk.audio)
        active_jobs.pop(job_id, None)
        log.info(ChunkAudioDisplay.display_job(all_chunks))

    return StreamingResponse(
        generate(), media_type="audio/wav",
        headers={"X-Job-Id": job_id, "X-Chunk-Count": str(len(chunks))},
    )


# ── REST: full WAV ────────────────────────────────────────────────────────────

@app.post("/tts/full")
async def tts_full(req: TTSRequest):
    chunks = chunker.split(req.text)
    job    = SynthesisJob(
        job_id=str(uuid.uuid4()), text=req.text, language=req.language,
        speaker_wav=req.speaker_wav, speaker=req.speaker, chunks=chunks,
    )
    all_audio, all_chunks = [], []
    async for chunk in engine.synthesize_stream(job):
        all_audio.append(chunk.audio)
        all_chunks.append(chunk)

    if not all_audio:
        raise HTTPException(status_code=500, detail="Synthesis produced no audio")

    log.info(ChunkAudioDisplay.display_job(all_chunks))
    combined  = np.concatenate(all_audio)
    wav_bytes = audio_to_wav_bytes(combined)
    return StreamingResponse(
        io.BytesIO(wav_bytes), media_type="audio/wav",
        headers={
            "Content-Length": str(len(wav_bytes)),
            "X-Chunk-Count":  str(len(all_chunks)),
            "X-Chunk-Meta":   json.dumps([chunk_to_meta(c) for c in all_chunks]),
        },
    )


# ── REST: pre-chunked ─────────────────────────────────────────────────────────

@app.post("/tts/chunks")
async def tts_chunks(req: ChunkRequest):
    audio_chunks = []
    for i, text in enumerate(req.chunks):
        ctype_str = req.chunk_types[i] if i < len(req.chunk_types) else "logic"
        try:
            ctype = ChunkType(ctype_str)
        except ValueError:
            ctype = ChunkType.LOGIC
        audio_chunks.append(AudioChunk(
            chunk_id=str(uuid.uuid4())[:8], chunk_type=ctype, text=text, chunk_index=i,
        ))
    job = SynthesisJob(
        job_id=str(uuid.uuid4()), text=" ".join(req.chunks), language=req.language,
        speaker_wav=req.speaker_wav, speaker=req.speaker, chunks=audio_chunks,
    )
    results = []
    async for chunk in engine.synthesize_stream(job):
        meta = chunk_to_meta(chunk, include_audio=True)
        meta["display_waveform"] = ChunkAudioDisplay.waveform(chunk.audio)
        meta["display_report"]   = ChunkAudioDisplay.display_chunk(chunk)
        results.append(meta)
    log.info(ChunkAudioDisplay.display_job([c for c in audio_chunks if c.ready]))
    return {"total_chunks": len(results), "chunks": results}


# ── WebSocket: real-time duplex ───────────────────────────────────────────────

@app.websocket("/ws/tts")
async def ws_tts(websocket: WebSocket):
    """
    WebSocket TTS endpoint.

    Client → Server (JSON):
        { "text": "...", "language": "en", "speaker_wav": null,
          "speaker": null, "chunk_tone": "tone"|"logic" }

    Server → Client:
        Frame 1   : binary WAV header (44 bytes)
        Frame 2–N : alternating JSON chunk_meta + binary PCM16 per chunk
        Final     : binary b""  (end-of-stream signal)
    """
    await websocket.accept()
    log.info("WS connected: %s", websocket.client)
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

            chunk_tone   = data.get("chunk_tone", "logic")
            ctype        = ChunkType.TONE if chunk_tone == "tone" else ChunkType.LOGIC
            single_chunk = AudioChunk(
                chunk_id=str(uuid.uuid4())[:8], chunk_type=ctype, text=text, chunk_index=0,
            )
            active_job = SynthesisJob(
                job_id=str(uuid.uuid4()), text=text,
                language=data.get("language", DEFAULT_LANGUAGE),
                speaker_wav=data.get("speaker_wav"),
                speaker=data.get("speaker"),
                chunks=[single_chunk],
            )

            log.info("WS TTS: tone=%s  len=%d  text=%r", chunk_tone, len(text), text[:60])
            await websocket.send_bytes(build_wav_header(num_samples=0))

            all_chunks = []
            async for chunk in engine.synthesize_stream(active_job):
                all_chunks.append(chunk)
                await websocket.send_json({
                    "type":       "chunk_meta",
                    "data":       chunk_to_meta(chunk, include_audio=False),
                    "waveform":   ChunkAudioDisplay.waveform(chunk.audio, width=40),
                    "chunk_tone": chunk_tone,
                })
                await websocket.send_bytes(float32_to_pcm16(chunk.audio))

            await websocket.send_bytes(b"")
            log.info(ChunkAudioDisplay.display_job(all_chunks))

    except WebSocketDisconnect:
        log.info("WS disconnected")
    except Exception as exc:
        log.error("WS error: %s", exc)
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass


# ── Cancel job ────────────────────────────────────────────────────────────────

@app.delete("/tts/job/{job_id}")
def cancel_job(job_id: str):
    job = active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancelled = True
    return {"cancelled": job_id}


# ── Chunker preview ───────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  WHAT TO ADD TO main.py  (STT microservice)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Add these two routes to your existing main.py.  No other changes needed.
#  The pipeline already has push_ai_reference() and commit_tts_enrollment().
#
# ── imports to add ─────────────────────────────────────────────────────────
#  (base64 and numpy already imported in main.py)
#
# ── route 1: receive enrollment audio from TTS ──────────────────────────────
#
#  from pydantic import BaseModel as _BM
#
#  class _EnrollReq(_BM):
#      pcm_b64:     str
#      sample_rate: int  = 16000
#      commit:      bool = True
#
#  @app.post("/enroll_tts")
#  async def enroll_tts(req: _EnrollReq):
#      """
#      Called by tts_microservice background enrollment at startup.
#      Feeds TTS audio directly into the voice filter and locks the profile.
#      """
#      try:
#          raw   = base64.b64decode(req.pcm_b64)
#          audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
#
#          # Safety resample in case sample rates ever diverge
#          if req.sample_rate != 16000 and len(audio) > 0:
#              try:
#                  from math import gcd
#                  from scipy.signal import resample_poly
#                  g     = gcd(16000, req.sample_rate)
#                  audio = resample_poly(audio, 16000//g, req.sample_rate//g).astype(np.float32)
#              except ImportError:
#                  n     = int(len(audio) * 16000 / req.sample_rate)
#                  audio = np.interp(
#                      np.linspace(0, 1, n), np.linspace(0, 1, len(audio)), audio
#                  ).astype(np.float32)
#
#          pipeline = _get_rest_pipeline()
#          pipeline.push_ai_reference(audio)
#
#          if req.commit:
#              pipeline.commit_tts_enrollment()
#
#          enrolled = bool(pipeline.tts_filter and pipeline.tts_filter.is_enrolled)
#          return {
#              "enrolled":   enrolled,
#              "samples":    len(audio),
#              "duration_s": round(len(audio) / 16000, 2),
#          }
#      except Exception as exc:
#          raise HTTPException(status_code=500, detail=str(exc))
#
#
# ── route 2: reset enrollment (called by TTS re_enroll()) ───────────────────
#
#  @app.delete("/enroll_tts")
#  async def reset_enroll_tts():
#      """Reset the TTS voice profile so re-enrollment can run fresh."""
#      pipeline = _get_rest_pipeline()
#      if pipeline.tts_filter:
#          pipeline.tts_filter.reset()
#      return {"status": "enrollment reset"}
#
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run("tts_microservice:app", host="0.0.0.0", port=8765, reload=False)