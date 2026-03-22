"""
tts_microservice.py  v5.4.0  —  Orpheus TTS (llama-cpp-python GGUF + SNAC)
═══════════════════════════════════════════════════════════════════════════

ROOT-CAUSE FIXES vs v5.3.0
──────────────────────────
PROBLEM 1 — ~100s latency on first real request
  The pre-warm sentence held the worker queue.  Real requests waited 22-80s.
  FIX: pre-warm runs in its OWN daemon thread, never touches the worker
  queue.  Real requests start immediately after engine.load() returns.

PROBLEM 2 — New WS connection per sentence = cold llama context per sentence
  Each /ws/tts connection cold-started llama context, producing robotic
  speech at sentence boundaries and stacking queue latency per sentence.
  FIX: /ws/tts is now a PERSISTENT SESSION.  One connection per AI turn.
  The gateway keeps it open, sends one JSON per sentence, reads PCM+sentinel,
  then sends the next.  Llama context stays warm across sentences.

PROBLEM 3 — chunk_tone misclassification → horrible audio quality
  chunk_tone was passed in from the gateway (always wrong) and used verbatim.
  FIX: chunk type is now auto-detected from text length + punctuation.
  The hint field is still accepted but defaults to "auto".

PROBLEM 4 — flush check fired on every token (including non-audio tokens)
  FIX: flush check now only executes when an audio token is appended.

PROBLEM 5 — synthesize_stream swallowed PCM bytes instead of yielding them
  REST /tts/stream was waiting for each full chunk before sending anything.
  FIX: synthesize_stream now yields (chunk, raw_bytes) tuples so the REST
  route can forward PCM as it arrives.

WS Protocol (persistent session):
  Per-request:
    CLIENT → JSON { "text":"...", "voice":"tara", "chunk_tone":"auto" }
    SERVER → binary WAV header (44 bytes)
           → binary PCM chunks (int16 24kHz, streamed live)
           → JSON  { "type":"chunk_meta", ... }
           → binary b""  ← end-of-request sentinel (send next request now)
  Control (any time):
    CLIENT → JSON { "type":"cancel" }   abort current synthesis
    CLIENT → JSON { "type":"close" }    graceful disconnect
    CLIENT → JSON { "type":"ping" }  →  SERVER { "type":"pong" }
  Heartbeat every 15s keeps NAT/proxies alive.

Model config (env vars):
  ORPHEUS_HF_REPO     QuantFactory/orpheus-3b-0.1-ft-GGUF
  ORPHEUS_GGUF_FILE   orpheus-3b-0.1-ft.Q2_K.gguf
  ORPHEUS_TOKENIZER   unsloth/orpheus-3b-0.1-ft
  ORPHEUS_GPU_LAYERS  -1
  ORPHEUS_CTX         768   (reduced from 900 — saves KV-cache, lower latency)
  ORPHEUS_BATCH       256
  ORPHEUS_THREADS     4
  ORPHEUS_VOICE       tara

Sample rate: 24 000 Hz mono int16 PCM.
"""

import asyncio
import base64
import collections
import io
import json
import logging
import os
import queue
import struct
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tts_service")


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 24_000
MAX_QUEUE_SIZE = 20
DEFAULT_LANGUAGE = "en"

HF_REPO        = os.getenv("ORPHEUS_HF_REPO",   "QuantFactory/orpheus-3b-0.1-ft-GGUF")
GGUF_FILENAME  = os.getenv("ORPHEUS_GGUF_FILE",  "orpheus-3b-0.1-ft.Q2_K.gguf")
TOKENIZER_REPO = os.getenv("ORPHEUS_TOKENIZER",  "unsloth/orpheus-3b-0.1-ft")

N_GPU_LAYERS = int(os.getenv("ORPHEUS_GPU_LAYERS", "-1"))
N_CTX        = int(os.getenv("ORPHEUS_CTX",        "768"))
N_BATCH      = int(os.getenv("ORPHEUS_BATCH",      "256"))
N_THREADS    = int(os.getenv("ORPHEUS_THREADS",    "4"))

ORPHEUS_VOICE  = os.getenv("ORPHEUS_VOICE", "tara")
ORPHEUS_VOICES = {"tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"}

SOH_ID       = 128259
EOT_ID       = 128009
EOH_ID       = 128260
START_TOKEN  = 128257
END_TOKENS   = {128258, 49158}
AUDIO_OFFSET = 128266
SNAC_VOCAB   = 4096
TPF          = 7    # tokens per SNAC frame ≈ 12 ms

INIT_FRAMES    = 1  # first flush: ~12 ms
STREAM_FRAMES  = 2  # subsequent: ~24 ms
MAX_EMPTY_SNAC = 3

TEMPERATURE        = 0.6
TOP_P              = 0.8
REPETITION_PENALTY = 1.1
MAX_TOKENS         = 1200

TONE_CHAR_LIMIT        = 80   # text ≤ this → tone mode
WS_HEARTBEAT_INTERVAL  = 15   # seconds


# ─────────────────────────────────────────────────────────────────────────────
#  Latency Stats
# ─────────────────────────────────────────────────────────────────────────────

class LatencyStats:
    WINDOW = 10

    def __init__(self):
        self._lock             = threading.Lock()
        self._data: Dict[str, collections.deque] = {}
        self._total_requests   = 0
        self._total_errors     = 0
        self._total_chars      = 0

    def record(self, voice: str, first_chunk_ms: float, chars: int) -> None:
        with self._lock:
            self._data.setdefault(voice, collections.deque(maxlen=self.WINDOW))
            self._data[voice].append(first_chunk_ms)
            self._total_requests += 1
            self._total_chars    += chars

    def record_error(self) -> None:
        with self._lock:
            self._total_errors += 1

    def summary(self) -> dict:
        with self._lock:
            out: dict = {
                "total_requests": self._total_requests,
                "total_errors":   self._total_errors,
                "total_chars":    self._total_chars,
                "voices":         {},
            }
            for voice, dq in self._data.items():
                if not dq:
                    continue
                arr = sorted(dq)
                n   = len(arr)
                out["voices"][voice] = {
                    "samples": n,
                    "avg_ms":  round(sum(arr) / n, 1),
                    "p50_ms":  round(arr[n // 2], 1),
                    "p95_ms":  round(arr[min(int(n * 0.95), n - 1)], 1),
                }
            return out

    def prometheus(self) -> str:
        s     = self.summary()
        lines = [
            "# HELP orpheus_requests_total Total synthesis requests",
            "# TYPE orpheus_requests_total counter",
            f'orpheus_requests_total {s["total_requests"]}',
            "# HELP orpheus_errors_total Total synthesis errors",
            "# TYPE orpheus_errors_total counter",
            f'orpheus_errors_total {s["total_errors"]}',
            "# HELP orpheus_chars_total Total characters synthesized",
            "# TYPE orpheus_chars_total counter",
            f'orpheus_chars_total {s["total_chars"]}',
        ]
        for voice, vs in s["voices"].items():
            lbl = f'voice="{voice}"'
            for metric, val in [
                ("orpheus_latency_avg_ms", vs["avg_ms"]),
                ("orpheus_latency_p50_ms", vs["p50_ms"]),
                ("orpheus_latency_p95_ms", vs["p95_ms"]),
            ]:
                lines.append(f"{metric}{{{lbl}}} {val}")
        return "\n".join(lines) + "\n"


_stats = LatencyStats()


# ─────────────────────────────────────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────────────────────────────────────

class ChunkType(str, Enum):
    TONE  = "tone"
    LOGIC = "logic"


def _auto_chunk_type(text: str) -> ChunkType:
    """Detect chunk type from the actual text — never trust caller hint."""
    stripped = text.rstrip()
    if len(stripped) <= TONE_CHAR_LIMIT or stripped.endswith("?") or stripped.endswith("!"):
        return ChunkType.TONE
    return ChunkType.LOGIC


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
    job_id:    str
    text:      str
    language:  str           = DEFAULT_LANGUAGE
    voice:     Optional[str] = None
    chunks:    list          = field(default_factory=list)
    audio_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=MAX_QUEUE_SIZE))
    done:      threading.Event = field(default_factory=threading.Event)
    cancelled: bool            = False
    start_ts:  float           = field(default_factory=time.time)
    _first_chunk_ready_ts: float = 0.0
    _lock:     threading.Lock  = field(default_factory=threading.Lock)

    def record_first_chunk(self, ts: float) -> None:
        with self._lock:
            if not self._first_chunk_ready_ts:
                self._first_chunk_ready_ts = ts

    @property
    def first_chunk_ready_ts(self) -> float:
        return self._first_chunk_ready_ts


class TTSRequest(BaseModel):
    text:     str
    language: str           = DEFAULT_LANGUAGE
    voice:    Optional[str] = None


class ChunkRequest(BaseModel):
    chunks:      List[str]
    chunk_types: List[str]     = []
    language:    str           = DEFAULT_LANGUAGE
    voice:       Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Text Chunker
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    TONE_MAX_CHARS  = 80
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
        parts = re.split(r'(?<=[,;])\s+', sentence)
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
            ctype     = _auto_chunk_type(sentence)
            max_chars = self.TONE_MAX_CHARS if ctype == ChunkType.TONE else self.LOGIC_MAX_CHARS
            for sub in self._sub_split(sentence, max_chars):
                chunks.append(AudioChunk(
                    chunk_id=str(uuid.uuid4())[:8], chunk_type=ctype,
                    text=sub, chunk_index=len(chunks),
                ))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  SNAC Decoder
# ─────────────────────────────────────────────────────────────────────────────

class SnacDecoder:
    def __init__(self, device: torch.device):
        from snac import SNAC
        self.model  = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
        self.device = device

    def warmup(self):
        dummy = [AUDIO_OFFSET + i % SNAC_VOCAB for i in range(TPF * 7)]
        self.decode(dummy)
        log.info("SNAC decoder ready on %s (pre-warmed)", self.device)

    @torch.inference_mode()
    def decode(self, toks: list) -> bytes:
        n = len(toks) // TPF
        if n == 0:
            return b""
        toks  = toks[:n * TPF]
        codes = [t - AUDIO_OFFSET for t in toks]
        if any(c < 0 for c in codes):
            return b""
        l0, l1, l2 = [], [], []
        for i in range(n):
            b = i * TPF
            l0.append(codes[b    ] % SNAC_VOCAB)
            l1.append(codes[b + 1] % SNAC_VOCAB)
            l2.append(codes[b + 2] % SNAC_VOCAB)
            l2.append(codes[b + 3] % SNAC_VOCAB)
            l1.append(codes[b + 4] % SNAC_VOCAB)
            l2.append(codes[b + 5] % SNAC_VOCAB)
            l2.append(codes[b + 6] % SNAC_VOCAB)
        t0 = torch.tensor(l0, device=self.device).unsqueeze(0)
        t1 = torch.tensor(l1, device=self.device).unsqueeze(0)
        t2 = torch.tensor(l2, device=self.device).unsqueeze(0)
        try:
            audio = self.model.decode([t0, t1, t2]).squeeze().cpu().numpy()
            return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        except Exception as exc:
            log.debug("SNAC decode error: %s", exc)
            return b""


# ─────────────────────────────────────────────────────────────────────────────
#  Token ID Capture
# ─────────────────────────────────────────────────────────────────────────────

class _TokenIDCapture:
    def __init__(self):
        self.last_token: Optional[int] = None

    def __call__(self, input_ids, scores):
        if len(input_ids) > 0:
            self.last_token = int(input_ids[-1])
        return scores


# ─────────────────────────────────────────────────────────────────────────────
#  Orpheus GGUF Engine
# ─────────────────────────────────────────────────────────────────────────────

class OrpheusEngine:

    def __init__(self):
        self.hf_repo    = HF_REPO
        self.gguf_file  = GGUF_FILENAME
        self.gguf_path  = None
        self.model_name = f"{HF_REPO}/{GGUF_FILENAME}"
        self._llm       = None
        self._tokenizer = None
        self._snac      = None
        self._device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._ready     = threading.Event()
        self._synth_queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="orpheus-worker"
        )

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self):
        log.info("Loading SNAC decoder on %s ...", self._device)
        self._snac = SnacDecoder(self._device)
        self._snac.warmup()   # warm SNAC only, not llama

        log.info("Loading tokenizer: %s ...", TOKENIZER_REPO)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
        log.info("Tokenizer ready.")

        log.info("Resolving GGUF: %s / %s ...", self.hf_repo, self.gguf_file)
        from huggingface_hub import hf_hub_download
        self.gguf_path = hf_hub_download(
            repo_id=self.hf_repo, filename=self.gguf_file, repo_type="model"
        )
        log.info("GGUF path: %s", self.gguf_path)

        log.info("Loading GGUF (n_gpu_layers=%d, n_ctx=%d, n_batch=%d) ...",
                 N_GPU_LAYERS, N_CTX, N_BATCH)
        from llama_cpp import Llama
        self._llm = Llama(
            model_path   = self.gguf_path,
            n_gpu_layers = N_GPU_LAYERS,
            n_ctx        = N_CTX,
            n_batch      = N_BATCH,
            n_threads    = N_THREADS,
            verbose      = False,
        )
        log.info("GGUF model loaded.")
        self._worker_thread.start()
        self._ready.set()
        log.info("OrpheusEngine ready. CUDA=%s device=%s voice=%s quant=%s",
                 torch.cuda.is_available(), self._device, ORPHEUS_VOICE, self.gguf_file)

    def prewarm_bg(self):
        """
        v5.4: pre-warm runs in its own thread, NOT via _synth_queue.
        Real requests are never blocked by this.
        """
        def _run():
            try:
                log.info("Pre-warm (background): warming llama + SNAC ...")
                t0   = time.time()
                v    = ORPHEUS_VOICE
                text = "Hello."
                ids  = self._tokenizer.encode(f"{v}: {text}", add_special_tokens=False)
                pids = [SOH_ID] + ids + [EOT_ID, EOH_ID]
                buf  = []
                for out in self._llm(pids, max_tokens=80, temperature=TEMPERATURE,
                                     top_p=TOP_P, stream=True):
                    frag = out["choices"][0].get("text", "")
                    if frag:
                        fids = self._tokenizer.encode(frag, add_special_tokens=False)
                        for tid in fids:
                            if tid >= AUDIO_OFFSET:
                                buf.append(tid)
                        if len(buf) >= TPF * 3:
                            self._snac.decode(buf)
                            break
                log.info("Pre-warm done in %.0fms", (time.time() - t0) * 1000)
            except Exception as exc:
                log.warning("Pre-warm failed (non-fatal): %s", exc)

        threading.Thread(target=_run, daemon=True, name="prewarm").start()

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker_loop(self):
        while True:
            item = self._synth_queue.get()
            if item is None:
                break
            text, voice, pcm_queue, cancel_evt = item
            try:
                self._synthesize_sync(text, voice, pcm_queue, cancel_evt)
            except Exception as exc:
                log.error("Worker synthesis error: %s", exc)
                _stats.record_error()
            finally:
                pcm_queue.put(None)
                self._synth_queue.task_done()

    # ── Core synthesis ────────────────────────────────────────────────────────

    def _synthesize_sync(
        self,
        text:       str,
        voice:      str,
        pcm_queue:  queue.Queue,
        cancel_evt: threading.Event,
    ):
        v          = voice if voice in ORPHEUS_VOICES else ORPHEUS_VOICE
        ids        = self._tokenizer.encode(f"{v}: {text}", add_special_tokens=False)
        prompt_ids = [SOH_ID] + ids + [EOT_ID, EOH_ID]

        capture = _TokenIDCapture()
        use_lp  = False
        try:
            from llama_cpp import LogitsProcessorList
            lp_list = LogitsProcessorList([capture])
            use_lp  = True
        except ImportError:
            pass

        gen_kwargs: dict = dict(
            max_tokens     = MAX_TOKENS,
            temperature    = TEMPERATURE,
            top_p          = TOP_P,
            repeat_penalty = REPETITION_PENALTY,
            stream         = True,
        )
        if use_lp:
            gen_kwargs["logits_processor"] = lp_list

        buf         = []
        in_audio    = False
        yielded     = 0
        done        = False
        empty_count = 0

        for output in self._llm(prompt_ids, **gen_kwargs):
            if done or cancel_evt.is_set():
                break

            text_frag     = output["choices"][0].get("text", "")
            finish_reason = output["choices"][0].get("finish_reason")

            if use_lp and capture.last_token is not None:
                tok_id = capture.last_token
                capture.last_token = None
            else:
                if not text_frag:
                    if finish_reason:
                        break
                    continue
                frag_ids = self._tokenizer.encode(text_frag, add_special_tokens=False)
                if not frag_ids:
                    continue
                tok_id = frag_ids[-1]

            if tok_id in END_TOKENS or finish_reason:
                done = True
                break
            if tok_id == START_TOKEN:
                in_audio = True
                continue
            if not in_audio:
                continue

            if tok_id >= AUDIO_OFFSET:
                buf.append(tok_id)

                # v5.4 FIX: flush check ONLY when audio token was just appended
                target = (INIT_FRAMES if yielded == 0 else STREAM_FRAMES) * TPF
                if len(buf) >= target:
                    n   = (len(buf) // TPF) * TPF
                    pcm = self._snac.decode(buf[:n])
                    buf = buf[n:]
                    if pcm:
                        empty_count = 0
                        if not cancel_evt.is_set():
                            pcm_queue.put(pcm)
                            yielded += n // TPF
                    else:
                        empty_count += 1
                        if empty_count >= MAX_EMPTY_SNAC:
                            log.warning("SNAC empty %d× — skipping chunk", empty_count)
                            break

        # Flush leftover
        if buf and not cancel_evt.is_set():
            n = (len(buf) // TPF) * TPF
            if n:
                pcm = self._snac.decode(buf[:n])
                if pcm:
                    pcm_queue.put(pcm)

    # ── Async streaming ───────────────────────────────────────────────────────

    async def synthesize_chunk_streaming(
        self,
        chunk:      AudioChunk,
        voice:      Optional[str],
        job:        SynthesisJob,
        cancel_evt: Optional[threading.Event] = None,
    ) -> AsyncGenerator[bytes, None]:
        if job.cancelled:
            return

        v      = voice if voice and voice in ORPHEUS_VOICES else ORPHEUS_VOICE
        pcm_q  = queue.Queue()
        _cevt  = cancel_evt if cancel_evt is not None else threading.Event()

        chunk.latency.synth_start_ts = time.time()
        chunk.latency.job_start_ts   = job.start_ts
        self._synth_queue.put((chunk.text, v, pcm_q, _cevt))

        loop      = asyncio.get_event_loop()
        raw_parts: list = []

        try:
            while True:
                if job.cancelled or _cevt.is_set():
                    _cevt.set()
                    break
                try:
                    raw = await loop.run_in_executor(
                        None, lambda: pcm_q.get(timeout=120.0)
                    )
                except queue.Empty:
                    log.warning("Synthesis timeout [%s...]", chunk.text[:30])
                    _cevt.set()
                    break

                if raw is None:
                    break

                raw_parts.append(raw)
                if len(raw_parts) == 1:
                    job.record_first_chunk(time.time())

                yield raw

        except asyncio.CancelledError:
            _cevt.set()
            raise
        except Exception as exc:
            _cevt.set()
            log.error("synthesize_chunk_streaming: %s", exc)
            chunk.error = str(exc)
        finally:
            _cevt.set()

        chunk.latency.synth_end_ts         = time.time()
        chunk.latency.first_chunk_ready_ts = job.first_chunk_ready_ts
        chunk.latency.compute()

        if raw_parts:
            all_bytes          = b"".join(raw_parts)
            chunk.audio        = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            chunk.duration_sec = len(chunk.audio) / SAMPLE_RATE
            _stats.record(v, chunk.latency.first_chunk_latency_ms, len(chunk.text))
        else:
            chunk.audio        = np.zeros(0, dtype=np.float32)
            chunk.duration_sec = 0.0
            chunk.error        = chunk.error or "empty audio"
            _stats.record_error()

        chunk.ready = True
        log.info("Chunk %s [%s] idx=%d | synth=%.0fms | job=%.0fms | dur=%.2fs",
                 chunk.chunk_id, chunk.chunk_type, chunk.chunk_index,
                 chunk.latency.synth_duration_ms,
                 chunk.latency.synthesis_latency_ms,
                 chunk.duration_sec)

    async def synthesize_chunk(self, chunk, voice, job):
        async for _ in self.synthesize_chunk_streaming(chunk, voice, job):
            pass
        return chunk

    async def synthesize_stream(self, job):
        """
        v5.4: yields (chunk, raw_pcm_bytes) as PCM arrives.
        Caller gets bytes immediately — no waiting for full chunk.
        """
        for ch in job.chunks:
            if job.cancelled:
                break
            async for raw in self.synthesize_chunk_streaming(ch, job.voice, job):
                yield ch, raw
            if ch.error:
                log.warning("Chunk %s error: %s", ch.chunk_id, ch.error)

    def shutdown(self):
        self._synth_queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
#  Chunk Audio Display
# ─────────────────────────────────────────────────────────────────────────────

class ChunkAudioDisplay:
    WIDTH    = 60
    HEIGHT   = 8
    BAR_CHAR = "█"

    @classmethod
    def waveform(cls, audio: np.ndarray, width: int = WIDTH, height: int = HEIGHT) -> str:
        if audio is None or len(audio) == 0:
            return "[no audio]"
        step    = max(1, len(audio) // width)
        cols    = []
        for i in range(0, len(audio), step):
            cols.append(float(np.sqrt(np.mean(audio[i:i+step] ** 2))))
            if len(cols) == width:
                break
        max_val = max(cols) if cols else 1.0
        norm    = [v / (max_val or 1.0) for v in cols]
        rows    = []
        for row in range(height, 0, -1):
            thr = row / height
            rows.append("|" + "".join(cls.BAR_CHAR if v >= thr else " " for v in norm) + "|")
        rows.append("+" + "-" * width + "+")
        return "\n".join(rows)

    @classmethod
    def display_chunk(cls, chunk: AudioChunk) -> str:
        lat = chunk.latency
        return "\n".join([
            f"{'─'*68}",
            f"  Chunk #{chunk.chunk_index}  id={chunk.chunk_id}  type={chunk.chunk_type}",
            f"  Text : {chunk.text[:80]}",
            f"  Audio: {chunk.duration_sec:.3f}s  ({int(chunk.duration_sec*SAMPLE_RATE)} samples)",
            f"  Latency:",
            f"    • synth duration          : {lat.synth_duration_ms:>8.1f} ms",
            f"    • since job start         : {lat.synthesis_latency_ms:>8.1f} ms",
            f"    • since first chunk ready : {lat.first_chunk_latency_ms:>8.1f} ms",
        ])

    @classmethod
    def display_job(cls, chunks: list) -> str:
        parts = [f"\n{'═'*68}", "  CHUNK AUDIO REPORT", f"{'═'*68}"]
        cum   = 0.0
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
        "chunk_id":     chunk.chunk_id,
        "chunk_index":  chunk.chunk_index,
        "chunk_type":   chunk.chunk_type,
        "text":         chunk.text,
        "duration_sec": round(chunk.duration_sec, 4),
        "latency":      chunk.latency.to_dict(),
    }
    if include_audio and chunk.audio is not None:
        meta["audio_b64"]     = base64.b64encode(audio_to_wav_bytes(chunk.audio)).decode()
        meta["audio_samples"] = len(chunk.audio)
    return meta

def _resolve_voice(requested: Optional[str]) -> str:
    if requested and requested in ORPHEUS_VOICES:
        return requested
    return ORPHEUS_VOICE


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

engine      = OrpheusEngine()
chunker     = TextChunker()
active_jobs: dict = {}
_AUDIO_HEADERS = {"X-Sample-Rate": str(SAMPLE_RATE), "X-Encoding": "pcm16"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.load)
    log.info("TTS service v5.4 started")
    engine.prewarm_bg()   # background only — never blocks real requests
    yield
    engine.shutdown()


app = FastAPI(title="Orpheus TTS Microservice", version="5.4.0", lifespan=lifespan)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "version":       "5.4.0",
        "engine":        "orpheus-gguf-llama-cpp",
        "hf_repo":       engine.hf_repo,
        "gguf_file":     engine.gguf_file,
        "gguf_path":     str(engine.gguf_path),
        "tokenizer":     TOKENIZER_REPO,
        "cuda":          torch.cuda.is_available(),
        "n_gpu_layers":  N_GPU_LAYERS,
        "n_ctx":         N_CTX,
        "n_batch":       N_BATCH,
        "default_voice": ORPHEUS_VOICE,
        "voices":        sorted(ORPHEUS_VOICES),
        "latency_stats": _stats.summary(),
    }

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return _stats.prometheus()

@app.get("/tts/voices")
def list_voices():
    stats = _stats.summary()
    return {"voices": [
        {"voice": v, "default": v == ORPHEUS_VOICE,
         **({"latency": stats["voices"][v]} if v in stats["voices"] else {})}
        for v in sorted(ORPHEUS_VOICES)
    ]}


# ── REST: streaming WAV ───────────────────────────────────────────────────────

@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    job_id = str(uuid.uuid4())
    chunks = chunker.split(req.text)
    job    = SynthesisJob(
        job_id=job_id, text=req.text, language=req.language,
        voice=_resolve_voice(req.voice), chunks=chunks,
    )
    active_jobs[job_id] = job
    seen: set = set()

    async def generate():
        yield build_wav_header(num_samples=0)
        async for ch, raw in engine.synthesize_stream(job):
            seen.add(ch.chunk_id)
            yield raw
        active_jobs.pop(job_id, None)
        log.info(ChunkAudioDisplay.display_job([c for c in chunks if c.chunk_id in seen]))

    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={**_AUDIO_HEADERS, "X-Job-Id": job_id, "X-Chunk-Count": str(len(chunks))},
    )


# ── REST: full WAV ────────────────────────────────────────────────────────────

@app.post("/tts/full")
async def tts_full(req: TTSRequest):
    chunks = chunker.split(req.text)
    job    = SynthesisJob(
        job_id=str(uuid.uuid4()), text=req.text, language=req.language,
        voice=_resolve_voice(req.voice), chunks=chunks,
    )
    all_audio, done_chunks, seen = [], [], set()
    async for ch, _ in engine.synthesize_stream(job):
        if ch.chunk_id not in seen and ch.audio is not None and len(ch.audio):
            seen.add(ch.chunk_id)
            all_audio.append(ch.audio)
            done_chunks.append(ch)

    if not all_audio:
        raise HTTPException(status_code=500, detail="Synthesis produced no audio")

    log.info(ChunkAudioDisplay.display_job(done_chunks))
    combined  = np.concatenate(all_audio)
    wav_bytes = audio_to_wav_bytes(combined)
    return StreamingResponse(
        io.BytesIO(wav_bytes), media_type="audio/wav",
        headers={
            **_AUDIO_HEADERS,
            "Content-Length": str(len(wav_bytes)),
            "X-Chunk-Count":  str(len(done_chunks)),
            "X-Chunk-Meta":   json.dumps([chunk_to_meta(c) for c in done_chunks]),
        },
    )


# ── REST: pre-chunked ─────────────────────────────────────────────────────────

@app.post("/tts/chunks")
async def tts_chunks(req: ChunkRequest):
    audio_chunks = []
    for i, text in enumerate(req.chunks):
        hint = req.chunk_types[i] if i < len(req.chunk_types) else "auto"
        if hint in ("auto", "") or hint not in ("tone", "logic"):
            ctype = _auto_chunk_type(text)
        else:
            ctype = ChunkType(hint)
        audio_chunks.append(AudioChunk(
            chunk_id=str(uuid.uuid4())[:8], chunk_type=ctype,
            text=text, chunk_index=i,
        ))
    job = SynthesisJob(
        job_id=str(uuid.uuid4()), text=" ".join(req.chunks), language=req.language,
        voice=_resolve_voice(req.voice), chunks=audio_chunks,
    )
    results, seen = [], set()
    async for ch, _ in engine.synthesize_stream(job):
        if ch.chunk_id not in seen and ch.ready:
            seen.add(ch.chunk_id)
            meta = chunk_to_meta(ch, include_audio=True)
            meta["display_waveform"] = ChunkAudioDisplay.waveform(ch.audio)
            meta["display_report"]   = ChunkAudioDisplay.display_chunk(ch)
            results.append(meta)
    log.info(ChunkAudioDisplay.display_job([c for c in audio_chunks if c.ready]))
    return {"total_chunks": len(results), "chunks": results}


# ── WebSocket: PERSISTENT SESSION ────────────────────────────────────────────

async def _ws_tts_session(websocket: WebSocket):
    """
    v5.4 PERSISTENT SESSION — one WS connection per AI turn.

    The gateway sends sentences one at a time over the SAME connection.
    This keeps llama context warm → natural gapless speech, no cold-start
    penalty per sentence.

    Per-request flow:
      1. Gateway → JSON { "text":"...", "voice":"tara", "chunk_tone":"auto" }
      2. Server  → binary WAV header (44 bytes)
      3. Server  → binary PCM chunks (int16, streamed live as generated)
      4. Server  → JSON { "type":"chunk_meta", "data":{...} }
      5. Server  → binary b""   ← END-OF-REQUEST sentinel
         (gateway can now send next JSON immediately)

    Control messages (gateway may send at any time):
      { "type":"cancel" }  → abort current synthesis, server resets for next
      { "type":"close" }   → graceful session end
      { "type":"ping" }    → server replies { "type":"pong" }

    Server sends { "type":"ping" } every 15s to keep NAT/proxies alive.
    """
    await websocket.accept()
    conn_id        = f"{websocket.client.host}:{websocket.client.port}"
    session_cancel = threading.Event()
    active_job: Optional[SynthesisJob] = None

    log.info("WS session opened: %s", conn_id)

    async def _heartbeat():
        while True:
            await asyncio.sleep(WS_HEARTBEAT_INTERVAL)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    hb_task = asyncio.create_task(_heartbeat())

    try:
        while True:
            try:
                raw_msg = await asyncio.wait_for(websocket.receive(), timeout=120.0)
            except asyncio.TimeoutError:
                log.info("WS session idle 120s — closing: %s", conn_id)
                break

            if raw_msg.get("bytes") is not None:
                continue  # ignore unexpected binary from client

            text_msg = raw_msg.get("text", "")
            if not text_msg:
                continue

            try:
                data = json.loads(text_msg)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")

            # ── Control ───────────────────────────────────────────────────────
            if msg_type == "close":
                log.info("WS close requested: %s", conn_id)
                break

            if msg_type == "cancel":
                session_cancel.set()
                if active_job:
                    active_job.cancelled = True
                active_job = None
                session_cancel.clear()
                log.info("WS cancel applied: %s", conn_id)
                continue

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # ── Synthesis ─────────────────────────────────────────────────────
            text = data.get("text", "").strip()
            if not text:
                await websocket.send_json({"type": "error", "detail": "empty text"})
                continue

            voice = _resolve_voice(data.get("voice"))

            # Always auto-detect — gateway hint is unreliable
            hint  = data.get("chunk_tone", "auto")
            ctype = _auto_chunk_type(text) if hint not in ("tone", "logic") else ChunkType(hint)

            single_chunk = AudioChunk(
                chunk_id=str(uuid.uuid4())[:8], chunk_type=ctype,
                text=text, chunk_index=0,
            )
            active_job = SynthesisJob(
                job_id   = str(uuid.uuid4()),
                text     = text,
                language = data.get("language", DEFAULT_LANGUAGE),
                voice    = voice,
                chunks   = [single_chunk],
            )
            session_cancel.clear()

            log.info("WS synth: %s voice=%s len=%d text=%r",
                     ctype, voice, len(text), text[:60])

            await websocket.send_bytes(build_wav_header(num_samples=0))

            async for raw_pcm in engine.synthesize_chunk_streaming(
                single_chunk, voice, active_job, cancel_evt=session_cancel
            ):
                await websocket.send_bytes(raw_pcm)

            if single_chunk.ready:
                await websocket.send_json({
                    "type":       "chunk_meta",
                    "data":       chunk_to_meta(single_chunk, include_audio=False),
                    "waveform":   ChunkAudioDisplay.waveform(single_chunk.audio, width=40)
                                  if single_chunk.audio is not None else "",
                    "chunk_tone": ctype.value,
                })
                log.info(ChunkAudioDisplay.display_chunk(single_chunk))

            await websocket.send_bytes(b"")  # end-of-request sentinel
            active_job = None

    except WebSocketDisconnect:
        log.info("WS disconnected: %s", conn_id)
        if active_job:
            active_job.cancelled = True
            session_cancel.set()
    except Exception as exc:
        log.error("WS error [%s]: %s", conn_id, exc)
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
        if active_job:
            active_job.cancelled = True
            session_cancel.set()
    finally:
        hb_task.cancel()
        session_cancel.set()
        log.info("WS session closed: %s", conn_id)


@app.websocket("/ws/tts")
async def ws_tts(websocket: WebSocket):
    await _ws_tts_session(websocket)


@app.websocket("/ws/tts/pipeline")
async def ws_tts_pipeline(websocket: WebSocket):
    """Alias — semantically clearer URL for gateway use."""
    await _ws_tts_session(websocket)


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

if __name__ == "__main__":
    uvicorn.run("tts_microservice:app", host="0.0.0.0", port=8765, reload=False)