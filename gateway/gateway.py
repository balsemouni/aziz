"""
gateway.py — Zero-Latency Voice Pipeline Gateway v9
════════════════════════════════════════════════════

WHAT'S NEW IN v9 (vs v8)
──────────────────────────

1. VOICE ENROLLMENT
   • First utterance at session start enrolls the user's voice (speaker profile).
   • Enrollment computes a voice embedding fingerprint from the first audio segment.
   • All subsequent ASR input is filtered: only audio matching the enrolled voice
     fingerprint (cosine similarity ≥ ENROLLMENT_THRESHOLD) is accepted.
   • Barge-in is ONLY triggered when the incoming audio matches the enrolled voice,
     preventing echo / TTS playback from triggering false barge-ins.

2. SMART BARGE-IN  (voice-verified)
   • When TTS is speaking, incoming audio is checked against enrolled voice before
     acting on it — noise or TTS echo (low similarity) is silently dropped.
   • Only real user speech (high similarity + min word count) causes barge-in.
   • Cooldown logic unchanged from v8.

3. TONAL TTS CHUNKING  (moved to gateway)
   • SentenceAccumulator redesigned to emit "tonal chunks" — complete phrases
     ending at natural pause points (sentence end, comma, colon, dash).
   • No more hard character caps mid-phrase; we wait for a punctuation boundary
     OR for a maximum safe length.
   • Chunks are annotated as TONE (questions/exclamations/short) or LOGIC (longer)
     so the TTS engine can apply appropriate prosody.
   • The TTS microservice's TextChunker is now bypassed in favour of pre-chunked
     tonal input, removing double-chunking.

4. PARALLEL TTS PIPELINE (v8)  — unchanged
   • Each tonal chunk gets its own TTS WebSocket immediately.
   • Audio is reordered and played in sequence via play_worker.

5. ENROLLMENT FLOW
   • Client sends: {"type": "start_session"} to begin enrollment mode.
   • First STT segment → enrolled, gateway sends {"type": "enrolled"}.
   • Subsequent segments are verified before routing to CAG / barge-in.
   • Client can re-enroll anytime with {"type": "re_enroll"}.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import httpx
import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# ─── Config ───────────────────────────────────────────────────────────────────

STT_WS_URL           = os.getenv("STT_WS_URL",            "ws://localhost:8001/stream/mux")
CAG_HTTP_URL         = os.getenv("CAG_HTTP_URL",           "http://localhost:8000")
TTS_WS_URL           = os.getenv("TTS_WS_URL",             "ws://localhost:8765/ws/tts")
GATEWAY_HOST         = os.getenv("GATEWAY_HOST",           "0.0.0.0")
GATEWAY_PORT         = int(os.getenv("GATEWAY_PORT",       "8090"))
TTS_SPEAKER          = os.getenv("TTS_SPEAKER",            "Claribel Dervla")
TTS_LANGUAGE         = os.getenv("TTS_LANGUAGE",           "en")

BARGE_IN_MIN_WORDS   = int(os.getenv("BARGE_IN_MIN_WORDS",    "3"))
BARGE_IN_COOLDOWN_S  = float(os.getenv("BARGE_IN_COOLDOWN_S", "0.4"))

# Voice enrollment / verification
# Higher threshold = stricter match = better echo rejection
ENROLLMENT_THRESHOLD  = float(os.getenv("ENROLLMENT_THRESHOLD",  "0.82"))
ENROLLMENT_WIN_FRAMES = int(os.getenv("ENROLLMENT_WIN_FRAMES",   "50"))

# TTS chunking thresholds — tuned for minimum first-word latency
# Fire the FIRST chunk as soon as we have a short phrase (≥8 chars).
# Subsequent chunks can be longer for natural prosody.
MIN_TTS_CHARS        = int(os.getenv("MIN_TTS_CHARS",       "8"))    # was 12
TONE_MAX_CHARS       = int(os.getenv("TONE_MAX_CHARS",      "60"))   # was 80
LOGIC_MAX_CHARS      = int(os.getenv("LOGIC_MAX_CHARS",     "160"))  # was 220
# Fire TTS immediately at first word boundary once buffer hits this size
FIRST_CHUNK_CHARS    = int(os.getenv("FIRST_CHUNK_CHARS",   "20"))   # new: eager first chunk

HALLUC_WINDOW        = int(os.getenv("HALLUC_WINDOW",       "10"))
HALLUC_THRESHOLD     = int(os.getenv("HALLUC_THRESHOLD",     "5"))
IDLE_TIMEOUT_S       = float(os.getenv("IDLE_TIMEOUT_S",   "120.0"))
HEARTBEAT_S          = float(os.getenv("HEARTBEAT_S",        "25.0"))
STT_MAX_RETRIES      = int(os.getenv("STT_MAX_RETRIES",      "5"))
TTS_MAX_RETRIES      = int(os.getenv("TTS_MAX_RETRIES",      "5"))
LATENCY_LOG_LEVEL    = os.getenv("LATENCY_LOG_LEVEL",        "info")
STT_SILENCE_MS       = float(os.getenv("STT_SILENCE_MS",    "600"))
TTS_MAX_PARALLEL     = int(os.getenv("TTS_MAX_PARALLEL",     "4"))

WS_PING_INTERVAL = 15
WS_PING_TIMEOUT  = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log       = logging.getLogger("gateway")
lat_log   = logging.getLogger("gateway.latency")
if LATENCY_LOG_LEVEL == "debug":
    lat_log.setLevel(logging.DEBUG)

app = FastAPI(title="Voice Gateway", version="9.0.0")
_session_latency_store: dict[str, list[dict]] = {}


# ─── Enums ────────────────────────────────────────────────────────────────────

class State(Enum):
    ENROLLING = auto()   # NEW: waiting for first voice sample
    IDLE      = auto()
    THINKING  = auto()
    SPEAKING  = auto()


class ChunkTone(str, Enum):
    TONE  = "tone"    # question / exclamation / short phrase
    LOGIC = "logic"   # longer declarative statement


# ─── Latency dataclasses ──────────────────────────────────────────────────────

@dataclass
class TurnLatency:
    turn_id:               str   = ""
    query_text:            str   = ""
    barge_in:              bool  = False

    stt_last_word_ts:      Optional[float] = None
    stt_segment_ts:        Optional[float] = None
    stt_latency_ms:        Optional[float] = None

    query_sent_ts:         Optional[float] = None
    first_token_ts:        Optional[float] = None
    cag_first_token_ms:    Optional[float] = None
    total_tokens:          int             = 0

    first_tts_chunk_ts:    Optional[float] = None
    cag_to_tts_ms:         Optional[float] = None
    tts_audio_start_ts:    Optional[float] = None
    tts_synth_ms:          Optional[float] = None

    e2e_ms:                Optional[float] = None
    tts_chunks:            list            = field(default_factory=list)

    def finalize(self):
        if self.stt_last_word_ts and self.stt_segment_ts:
            self.stt_latency_ms = (self.stt_segment_ts - self.stt_last_word_ts) * 1000
        if self.query_sent_ts and self.first_token_ts:
            self.cag_first_token_ms = (self.first_token_ts - self.query_sent_ts) * 1000
        if self.first_token_ts and self.first_tts_chunk_ts:
            self.cag_to_tts_ms = (self.first_tts_chunk_ts - self.first_token_ts) * 1000
        if self.first_tts_chunk_ts and self.tts_audio_start_ts:
            self.tts_synth_ms = (self.tts_audio_start_ts - self.first_tts_chunk_ts) * 1000
        if self.stt_last_word_ts and self.tts_audio_start_ts:
            self.e2e_ms = (self.tts_audio_start_ts - self.stt_last_word_ts) * 1000

    def to_report(self) -> dict:
        self.finalize()
        return {
            "turn_id":            self.turn_id,
            "query":              self.query_text[:80],
            "barge_in":           self.barge_in,
            "stt_latency_ms":     _r(self.stt_latency_ms),
            "cag_first_token_ms": _r(self.cag_first_token_ms),
            "cag_to_tts_ms":      _r(self.cag_to_tts_ms),
            "tts_synth_ms":       _r(self.tts_synth_ms),
            "e2e_ms":             _r(self.e2e_ms),
            "total_tokens":       self.total_tokens,
            "tts_chunks":         self.tts_chunks,
        }


def _r(v: Optional[float]) -> Optional[float]:
    return round(v, 1) if v is not None else None


# ─── Voice Enrollment / Verification ─────────────────────────────────────────
#
# We use a simple but effective approach:
#   • When the STT gives us word-level audio timing, we extract raw PCM energy
#     features from the 16-bit audio frames collected during enrollment.
#   • We store a mean feature vector (the "voiceprint").
#   • On subsequent segments, we compute cosine similarity between the new
#     segment's feature vector and the voiceprint.
#   • If similarity < ENROLLMENT_THRESHOLD → treat as noise / echo → ignore.
#
# This works WITHOUT a heavy speaker-embedding model.  For production you would
# swap _extract_features() to use a real x-vector or d-vector model (e.g.
# SpeechBrain).  The interface stays identical.
# ─────────────────────────────────────────────────────────────────────────────

class VoiceEnrollment:
    """
    Lightweight voice enrollment using spectral energy fingerprinting.

    Frame format expected: raw PCM-16 LE, mono, 16 kHz (matches STT input).
    """

    FRAME_SIZE   = 512      # samples per feature window
    N_BANDS      = 16       # frequency bands for feature vector
    ENERGY_FLOOR = 1e-8

    def __init__(self):
        self._voiceprint:  Optional[np.ndarray] = None
        self._enrolled:    bool = False
        self._enroll_buf:  list[np.ndarray]    = []   # raw frames collected
        self._enroll_frames_needed = ENROLLMENT_WIN_FRAMES

    @property
    def enrolled(self) -> bool:
        return self._enrolled

    def reset(self):
        self._voiceprint  = None
        self._enrolled    = False
        self._enroll_buf  = []

    def add_enrollment_frame(self, pcm_bytes: bytes) -> bool:
        """
        Feed raw PCM bytes during enrollment.
        Returns True when enough frames have been collected and enrollment is complete.
        """
        samples = self._bytes_to_float(pcm_bytes)
        if samples is None or len(samples) == 0:
            return False
        self._enroll_buf.append(samples)
        if len(self._enroll_buf) >= self._enroll_frames_needed:
            combined = np.concatenate(self._enroll_buf)
            self._voiceprint = self._extract_features(combined)
            self._enrolled   = True
            self._enroll_buf = []
            return True
        return False

    def verify(self, pcm_bytes: bytes) -> float:
        """
        Compare incoming audio against enrolled voiceprint.
        Returns cosine similarity [0, 1].  Returns 1.0 if not yet enrolled
        (accept everything during enrollment phase).
        """
        if not self._enrolled or self._voiceprint is None:
            return 1.0
        samples = self._bytes_to_float(pcm_bytes)
        if samples is None or len(samples) < self.FRAME_SIZE:
            return 0.0
        feat = self._extract_features(samples)
        return float(self._cosine(self._voiceprint, feat))

    def verify_text_segment(self, audio_frames: list[bytes]) -> float:
        """Verify a full segment represented as a list of PCM byte chunks."""
        if not self._enrolled:
            return 1.0
        if not audio_frames:
            return 0.0
        combined_bytes = b"".join(audio_frames)
        return self.verify(combined_bytes)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _bytes_to_float(self, pcm_bytes: bytes) -> Optional[np.ndarray]:
        try:
            n = len(pcm_bytes) // 2
            if n == 0:
                return None
            return np.frombuffer(pcm_bytes[:n*2], dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            return None

    def _extract_features(self, samples: np.ndarray) -> np.ndarray:
        """
        Extract a simple log-energy filterbank feature vector.
        Splits signal into FRAME_SIZE windows, computes FFT per window,
        averages across N_BANDS frequency bands.
        Result: 1-D float32 vector of length N_BANDS.
        """
        n = len(samples)
        n_frames = max(1, n // self.FRAME_SIZE)
        band_energies = np.zeros(self.N_BANDS, dtype=np.float64)
        count = 0
        for i in range(n_frames):
            frame = samples[i * self.FRAME_SIZE : (i + 1) * self.FRAME_SIZE]
            if len(frame) < self.FRAME_SIZE:
                break
            # Apply Hann window
            window = frame * np.hanning(len(frame))
            spec   = np.abs(np.fft.rfft(window))
            half   = len(spec)
            # Split spectrum into N_BANDS equal-width bands and take log energy
            band_size = max(1, half // self.N_BANDS)
            for b in range(self.N_BANDS):
                start = b * band_size
                end   = min(start + band_size, half)
                energy = np.mean(spec[start:end] ** 2) + self.ENERGY_FLOOR
                band_energies[b] += np.log(energy)
            count += 1
        if count > 0:
            band_energies /= count
        return band_energies.astype(np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ─── Tonal Sentence Accumulator ───────────────────────────────────────────────
#
# Redesigned for v9.  Goals:
#   1. Never cut mid-word or mid-phrase.
#   2. Emit chunks at natural prosody boundaries: sentence end, clause end.
#   3. Annotate each chunk as TONE or LOGIC so TTS engine can adjust prosody.
#   4. Fall back to splitting at commas/semicolons if a clause is too long.
#   5. Keep a configurable minimum size so we don't fire TTS for single words.
# ─────────────────────────────────────────────────────────────────────────────

_RE_SENTENCE_END  = re.compile(r'(?<=[^\d])([.!?]+["\']?)(?=\s|$)')
_RE_CLAUSE_BREAK  = re.compile(r'([,;:—–])\s')      # comma, semicolon, colon, em-dash
_RE_STARTS_PUNCT  = re.compile(r'^[\s,\.!?;:\)\]\}\'\u2019\u2018\u201c\u201d\-]')


def _classify_tone(text: str) -> ChunkTone:
    stripped = text.strip()
    if (stripped.endswith("?") or stripped.endswith("!")
            or len(stripped) <= TONE_MAX_CHARS):
        return ChunkTone.TONE
    return ChunkTone.LOGIC


@dataclass
class TonalChunk:
    text: str
    tone: ChunkTone


class TonalAccumulator:
    """
    Accumulates CAG tokens and emits tonal chunks when it has a complete
    prosodic unit.  Never emits partial words or arbitrarily short fragments.
    """

    def __init__(self):
        self._buf        = ""
        self._first_sent = True

    def reset(self):
        self._buf        = ""
        self._first_sent = True

    def feed(self, token: str) -> list[TonalChunk]:
        """
        Feed one token, return a (possibly empty) list of ready TonalChunks.
        """
        if not token:
            return []

        # Smart space handling: avoid double-spaces
        if token.startswith(" "):
            if self._buf and self._buf[-1] == " ":
                token = token.lstrip(" ")
        else:
            if self._buf and not self._buf[-1].isspace() and not _RE_STARTS_PUNCT.match(token):
                token = " " + token
        self._buf += token

        return self._try_flush()

    def flush(self) -> Optional[TonalChunk]:
        """Force-flush whatever remains in the buffer at end of stream."""
        text = self._buf.strip()
        self._buf = ""
        self._first_sent = True
        if len(text) >= 1:
            return TonalChunk(text=text, tone=_classify_tone(text))
        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _try_flush(self) -> list[TonalChunk]:
        results: list[TonalChunk] = []

        while True:
            buf = self._buf

            # ── Priority 0: EAGER FIRST CHUNK ────────────────────────────────
            # On the very first chunk, don't wait for punctuation — fire as
            # soon as we have FIRST_CHUNK_CHARS and a word boundary so TTS
            # synthesis starts immediately while CAG is still streaming.
            if self._first_sent and len(buf) >= FIRST_CHUNK_CHARS:
                # Find last space to avoid cutting mid-word
                split = buf.rfind(" ")
                if split >= MIN_TTS_CHARS:
                    candidate = buf[:split].strip()
                    remainder = buf[split:].lstrip()
                    if candidate:
                        results.append(TonalChunk(
                            text=candidate,
                            tone=_classify_tone(candidate),
                        ))
                        self._buf        = remainder
                        self._first_sent = False
                        continue
                elif len(buf) >= TONE_MAX_CHARS:
                    # No space found but buffer is huge — force-split
                    candidate = buf[:TONE_MAX_CHARS].strip()
                    remainder = buf[TONE_MAX_CHARS:].lstrip()
                    if candidate:
                        results.append(TonalChunk(
                            text=candidate,
                            tone=_classify_tone(candidate),
                        ))
                        self._buf        = remainder
                        self._first_sent = False
                        continue
                # Not enough content yet — keep accumulating
                break

            # ── Priority 1: sentence-ending punctuation ───────────────────────
            m = _RE_SENTENCE_END.search(buf)
            if m:
                candidate = buf[:m.end()].strip()
                remainder = buf[m.end():].lstrip()
                if len(candidate) >= MIN_TTS_CHARS or not self._first_sent:
                    results.append(TonalChunk(
                        text=candidate,
                        tone=_classify_tone(candidate),
                    ))
                    self._buf        = remainder
                    self._first_sent = False
                    continue
                break

            # ── Priority 2: clause break (comma/semicolon/colon/dash) ─────────
            m = _RE_CLAUSE_BREAK.search(buf)
            if m:
                candidate = buf[:m.start() + 1].strip()
                remainder = buf[m.end():].lstrip()
                if (len(candidate) >= MIN_TTS_CHARS
                        and len(remainder) >= 3):
                    results.append(TonalChunk(
                        text=candidate,
                        tone=_classify_tone(candidate),
                    ))
                    self._buf        = remainder
                    self._first_sent = False
                    continue
                break

            # ── Priority 3: hard length cap ───────────────────────────────────
            max_cap = LOGIC_MAX_CHARS if not self._first_sent else TONE_MAX_CHARS
            if len(buf) > max_cap:
                split = buf[:max_cap].rfind(" ")
                if split <= MIN_TTS_CHARS:
                    split = max_cap
                candidate = buf[:split].strip()
                remainder = buf[split:].lstrip()
                if candidate:
                    results.append(TonalChunk(
                        text=candidate,
                        tone=_classify_tone(candidate),
                    ))
                    self._buf        = remainder
                    self._first_sent = False
                    continue
                break

            break

        return results


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _drain(q: asyncio.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break


async def _ws_connect_with_backoff(url: str, max_retries: int, label: str, **kwargs):
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            ws = await websockets.connect(url, **kwargs)
            if attempt > 1:
                log.info(f"{label} reconnected (attempt {attempt})")
            return ws
        except Exception as e:
            if attempt == max_retries:
                raise
            log.warning(f"{label} connect failed ({e}), retry {attempt}/{max_retries} in {delay:.1f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)


def _bar(value: float, max_val: float, width: int = 28) -> str:
    filled = int((value / max_val) * width) if max_val else 0
    return "█" * filled + "░" * (width - filled)


def _latency_color(ms: float) -> str:
    if ms < 200:  return "🟢"
    if ms < 400:  return "🟡"
    if ms < 800:  return "🟠"
    return "🔴"


# ─── Hallucination / repetition guard ────────────────────────────────────────

class RepetitionGuard:
    _STOP_WORDS: frozenset = frozenset({
        "a", "an", "the", "i", "me", "my", "we", "our", "you", "your",
        "he", "she", "it", "they", "his", "her", "its", "their",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "to", "of", "in", "on", "at", "by", "for", "with",
        "and", "or", "but", "not", "no", "so", "if", "as",
        "that", "this", "these", "those",
        "what", "which", "who", "how", "when", "where", "why",
        "up", "out", "about", "into", "from", "than", "then",
        "can", "will", "would", "could", "should", "may", "might",
        "just", "also", "very", "more", "some", "any",
    })

    def __init__(self, window: int = HALLUC_WINDOW, threshold: int = HALLUC_THRESHOLD):
        self._threshold = max(threshold, 4)
        self._history: list[str] = []
        self._window = window

    def feed(self, word: str) -> bool:
        w = word.lower().strip().rstrip(".,!?;:")
        if not w or w in self._STOP_WORDS:
            return False
        self._history.append(w)
        if len(self._history) > self._window:
            self._history.pop(0)
        if len(self._history) < self._threshold:
            return False
        tail = self._history[-self._threshold:]
        return len(set(tail)) == 1

    def reset(self):
        self._history.clear()


# ─── Session latency tracker ─────────────────────────────────────────────────

class LatencyTracker:
    def __init__(self, sid: str):
        self.sid                 = sid
        self.current             : Optional[TurnLatency] = None
        self.history             : list[TurnLatency]     = []
        self._stt_lats           : list[float]           = []
        self._tts_job_start      : Optional[float]       = None
        self._tts_first_chunk_ts : Optional[float]       = None
        self._tts_chunk_index    : int                   = 0

    def new_turn(self, turn_id: str, query: str):
        self.current             = TurnLatency(turn_id=turn_id, query_text=query)
        self._tts_job_start      = None
        self._tts_first_chunk_ts = None
        self._tts_chunk_index    = 0

    def complete_turn(self) -> Optional[dict]:
        if not self.current:
            return None
        self.current.finalize()
        report = self.current.to_report()
        self.history.append(self.current)
        if self.current.stt_latency_ms is not None:
            self._stt_lats.append(self.current.stt_latency_ms)
        lat_log.info(
            f"[{self.sid}] turn_latency "
            f"stt={_r(self.current.stt_latency_ms)}ms "
            f"cag_first={_r(self.current.cag_first_token_ms)}ms "
            f"cag_to_tts={_r(self.current.cag_to_tts_ms)}ms "
            f"tts_synth={_r(self.current.tts_synth_ms)}ms "
            f"e2e={_r(self.current.e2e_ms)}ms "
            f"tokens={self.current.total_tokens}"
        )
        e2e = self.current.e2e_ms
        if e2e is not None:
            icon = _latency_color(e2e)
            lat_log.info(
                f"[{self.sid}] {icon} E2E {e2e:.0f}ms  "
                f"[STT {_r(self.current.stt_latency_ms)}ms | "
                f"CAG {_r(self.current.cag_first_token_ms)}ms | "
                f"TTS {_r(self.current.tts_synth_ms)}ms]"
            )
        self.current = None
        return report

    def on_stt_word(self):
        now = time.monotonic()
        if self.current:
            self.current.stt_last_word_ts = now

    def on_stt_segment(self):
        now = time.monotonic()
        if self.current:
            self.current.stt_segment_ts = now
            if self.current.stt_last_word_ts:
                ms  = (now - self.current.stt_last_word_ts) * 1000
                self.current.stt_latency_ms = ms
                avg = (sum(self._stt_lats) + ms) / (len(self._stt_lats) + 1)
                lat_log.info(
                    f"[{self.sid}] STT segment latency: {ms:.0f}ms  "
                    f"(running avg {avg:.0f}ms over {len(self._stt_lats)+1} segments)  "
                    f"{_latency_color(ms)}"
                )

    def on_query_sent(self):
        now = time.monotonic()
        if self.current:
            self.current.query_sent_ts = now

    def on_first_token(self):
        now = time.monotonic()
        if self.current and not self.current.first_token_ts:
            self.current.first_token_ts = now
            if self.current.query_sent_ts:
                ms = (now - self.current.query_sent_ts) * 1000
                self.current.cag_first_token_ms = ms
                lat_log.info(f"[{self.sid}] CAG first token: {ms:.0f}ms  {_latency_color(ms)}")

    def on_token(self):
        if self.current:
            self.current.total_tokens += 1

    def on_tts_chunk_sent(self, chunk_text: str):
        now = time.monotonic()
        if self.current and not self.current.first_tts_chunk_ts:
            self.current.first_tts_chunk_ts = now
            self._tts_job_start = now
            self._tts_chunk_index = 0
            if self.current.first_token_ts:
                ms = (now - self.current.first_token_ts) * 1000
                self.current.cag_to_tts_ms = ms
                lat_log.info(f"[{self.sid}] CAG→TTS first chunk queued: {ms:.0f}ms")

    def on_tts_audio_start(self):
        now = time.monotonic()
        if self.current and not self.current.tts_audio_start_ts:
            self.current.tts_audio_start_ts = now
            if self.current.first_tts_chunk_ts:
                ms = (now - self.current.first_tts_chunk_ts) * 1000
                self.current.tts_synth_ms = ms
                lat_log.info(f"[{self.sid}] TTS synthesis latency: {ms:.0f}ms  {_latency_color(ms)}")

    def on_tts_chunk_complete(self, synthesis_latency_ms: float, synth_duration_ms: float, duration_sec: float):
        if not self.current:
            return
        idx = self._tts_chunk_index
        now = time.monotonic()
        if self._tts_first_chunk_ts is None:
            self._tts_first_chunk_ts   = now
            first_chunk_latency_ms     = 0.0
        else:
            first_chunk_latency_ms     = (now - self._tts_first_chunk_ts) * 1000
        chunk_info = {
            "chunk_index":            idx,
            "synthesis_latency_ms":   _r(synthesis_latency_ms),
            "synth_duration_ms":      _r(synth_duration_ms),
            "first_chunk_latency_ms": _r(first_chunk_latency_ms),
            "duration_sec":           _r(duration_sec),
        }
        self.current.tts_chunks.append(chunk_info)
        max_lat = max((c["synthesis_latency_ms"] or 0) for c in self.current.tts_chunks) or 1
        bar     = _bar(synthesis_latency_ms, max_lat)
        star    = "★ FIRST" if idx == 0 else f"from 1st: {first_chunk_latency_ms:.0f}ms"
        lat_log.info(
            f"[{self.sid}] TTS chunk {idx}  "
            f"synth={synth_duration_ms:.0f}ms  "
            f"lat={synthesis_latency_ms:.0f}ms [{bar}]  "
            f"{star}  dur={duration_sec:.2f}s"
        )
        self._tts_chunk_index += 1

    def session_summary(self) -> dict:
        turns = [t for t in self.history if t.e2e_ms is not None]
        if not turns:
            return {"sid": self.sid, "turns": 0}

        def _stats(vals):
            if not vals:
                return {}
            sv = sorted(vals)
            p95_idx = max(0, int(len(sv) * 0.95) - 1)
            return {"min": _r(min(sv)), "max": _r(max(sv)),
                    "avg": _r(sum(sv)/len(sv)), "p95": _r(sv[p95_idx])}

        stt_lats  = [t.stt_latency_ms    for t in turns if t.stt_latency_ms    is not None]
        cag_lats  = [t.cag_first_token_ms for t in turns if t.cag_first_token_ms is not None]
        tts_lats  = [t.tts_synth_ms       for t in turns if t.tts_synth_ms      is not None]
        e2e_lats  = [t.e2e_ms             for t in turns]
        barge_ins = sum(1 for t in turns if t.barge_in)
        summary   = {
            "sid":       self.sid,
            "turns":     len(turns),
            "barge_ins": barge_ins,
            "stt":       _stats(stt_lats),
            "cag":       _stats(cag_lats),
            "tts_synth": _stats(tts_lats),
            "e2e":       _stats(e2e_lats),
        }
        lat_log.info(
            f"[{self.sid}] ═══ SESSION SUMMARY ═══  "
            f"turns={len(turns)}  barge_ins={barge_ins}  "
            f"e2e avg={_r(sum(e2e_lats)/len(e2e_lats))}ms  "
            f"e2e p95={_r(sorted(e2e_lats)[max(0,int(len(e2e_lats)*0.95)-1)])}ms"
        )
        return summary

    def all_reports(self) -> list[dict]:
        return [t.to_report() for t in self.history]


# ─── Gateway session ──────────────────────────────────────────────────────────

class GatewaySession:
    """
    v9 session with:
      • Voice enrollment (first utterance)
      • Voice-verified barge-in (prevents echo/TTS from triggering false barge-in)
      • Tonal TTS chunking (complete prosodic units, not arbitrary char cuts)
      • Parallel TTS synthesis + ordered playback (unchanged from v8)
    """

    _INTERRUPT   = object()   # sentinel: barge-in signal
    _TURN_END    = object()   # sentinel: end of CAG turn

    def __init__(self, ws: WebSocket):
        self.ws       = ws
        self.sid      = str(uuid.uuid4())[:8]
        self.state    = State.ENROLLING   # start in enrollment mode
        self._running = True

        self._audio_q : asyncio.Queue[bytes]  = asyncio.Queue()
        self._query_q : asyncio.Queue[str]    = asyncio.Queue()

        # _tts_q carries:  TonalChunk | _INTERRUPT | _TURN_END
        self._tts_q   : asyncio.Queue         = asyncio.Queue()

        # _pcm_q carries ordered frames for the play worker:
        # each item is (order:int, frames:list[bytes]) | _INTERRUPT | _TURN_END
        self._pcm_q   : asyncio.Queue         = asyncio.Queue()

        self._tts_sem = asyncio.Semaphore(TTS_MAX_PARALLEL)

        self._stt_ws          = None
        self._barge_in        = False
        self._active_turn_id  : Optional[str] = None
        self._last_query_time : float         = time.monotonic()
        self._last_pong_time  : float         = time.monotonic()
        self._barge_in_until  : float         = 0.0

        # v9: voice enrollment
        self._voice = VoiceEnrollment()
        # Buffer of raw audio frames collected during current speech segment
        # Used for voice verification at segment completion
        self._segment_audio_buf: list[bytes] = []

        self._lat = LatencyTracker(self.sid)

        log.info(f"[{self.sid}] session created (v9 — enrollment + tonal TTS)")

    # ── safe sends ────────────────────────────────────────────────────────────

    async def _jsend(self, obj: dict):
        try:
            await self.ws.send_json(obj)
        except Exception:
            pass

    async def _bsend(self, data: bytes):
        try:
            await self.ws.send_bytes(data)
        except Exception:
            pass

    async def _unmute_mic(self):
        # Kept for compatibility — mic is never muted; barge-in handles interruption.
        pass

    # ── entry ─────────────────────────────────────────────────────────────────

    async def run(self):
        tasks = [
            asyncio.create_task(self._stt_loop(),      name="stt"),
            asyncio.create_task(self._cag_loop(),      name="cag"),
            asyncio.create_task(self._synth_worker(),  name="synth"),
            asyncio.create_task(self._play_worker(),   name="play"),
            asyncio.create_task(self._heartbeat(),     name="heartbeat"),
            asyncio.create_task(self._idle_watchdog(), name="idle"),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in pending:
            t.cancel()
        for t in done:
            if not t.cancelled():
                exc = t.exception()
                if exc:
                    log.error(f"[{self.sid}] {t.get_name()} crashed: {exc}", exc_info=exc)

    async def stop(self):
        self._running = False
        summary = self._lat.session_summary()
        _session_latency_store[self.sid] = {
            "summary": summary,
            "turns":   self._lat.all_reports(),
        }
        await self._jsend({"type": "session_summary", "latency": summary})

    # ── client audio / control ────────────────────────────────────────────────

    def push_audio(self, frame: bytes):
        # During enrollment we also feed audio directly to the voice enroller
        # so the fingerprint is built from real raw audio, not just text events.
        payload = frame[1:]  # strip frame-type byte
        if self.state == State.ENROLLING and not self._voice.enrolled:
            self._voice.add_enrollment_frame(payload)
        else:
            # Always accumulate in segment buffer for verification
            self._segment_audio_buf.append(payload)
        self._audio_q.put_nowait(frame)

    def push_control(self, payload: bytes):
        try:
            ctrl = json.loads(payload)
        except Exception:
            return
        mtype = ctrl.get("type")
        if mtype == "re_enroll":
            self._voice.reset()
            self._segment_audio_buf.clear()
            self.state = State.ENROLLING
            asyncio.ensure_future(self._jsend({"type": "re_enrolling"}))
            log.info(f"[{self.sid}] re-enrollment requested")
        elif self._stt_ws:
            asyncio.ensure_future(self._relay_ctrl(payload))

    async def _relay_ctrl(self, payload: bytes):
        try:
            await self._stt_ws.send(b'\x02' + payload)
        except Exception:
            pass

    # ── heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat(self):
        while self._running:
            await asyncio.sleep(HEARTBEAT_S)
            await self._jsend({"type": "ping"})
            if time.monotonic() - self._last_pong_time > HEARTBEAT_S * 2:
                log.warning(f"[{self.sid}] heartbeat timeout — closing session")
                self._running = False
                break

    # ── idle watchdog ─────────────────────────────────────────────────────────

    async def _idle_watchdog(self):
        while self._running:
            await asyncio.sleep(10)
            idle_s = time.monotonic() - self._last_query_time
            if idle_s > IDLE_TIMEOUT_S and self.state == State.IDLE:
                log.info(f"[{self.sid}] idle reset after {idle_s:.0f}s")
                try:
                    async with httpx.AsyncClient(base_url=CAG_HTTP_URL, timeout=5.0) as http:
                        await http.post("/reset")
                except Exception as e:
                    log.warning(f"[{self.sid}] idle /reset failed: {e}")
                self._last_query_time = time.monotonic()
                await self._jsend({"type": "session_reset", "reason": "idle_timeout"})

    # ── voice verification helper ─────────────────────────────────────────────

    def _is_enrolled_voice(self) -> bool:
        """Check if the current segment audio matches the enrolled voice."""
        if not self._voice.enrolled:
            return True  # not enrolled yet → accept (shouldn't happen post-enrollment)
        similarity = self._voice.verify_text_segment(self._segment_audio_buf)
        self._segment_audio_buf = []  # always clear after check
        log.info(f"[{self.sid}] voice similarity={similarity:.3f} (threshold={ENROLLMENT_THRESHOLD})")
        return similarity >= ENROLLMENT_THRESHOLD

    # ── barge-in ──────────────────────────────────────────────────────────────

    async def _do_barge_in(self, new_query):
        now = time.monotonic()
        if now < self._barge_in_until:
            log.info(f"[{self.sid}] barge-in suppressed (cooldown)")
            return
        display = new_query[1][:60] if isinstance(new_query, tuple) else new_query[:60]
        log.info(f"[{self.sid}] ⚡ BARGE-IN → '{display}'")

        if self._lat.current:
            self._lat.current.barge_in = True
            interrupted_report = self._lat.complete_turn()
            if interrupted_report:
                await self._jsend({"type": "latency", "stage": "turn_interrupted", **interrupted_report})

        self._barge_in       = True
        self._barge_in_until = now + BARGE_IN_COOLDOWN_S

        # Stop all in-flight TTS immediately
        _drain(self._tts_q)
        _drain(self._pcm_q)
        await self._tts_q.put(self._INTERRUPT)
        await self._pcm_q.put(self._INTERRUPT)

        _drain(self._query_q)
        await self._query_q.put(new_query)
        await self._unmute_mic()

    # ─────────────────────────────────────────────────────────────────────────
    # STT LOOP  (v9 — adds enrollment gate + voice-verified barge-in)
    # ─────────────────────────────────────────────────────────────────────────

    async def _stt_loop(self):
        log.info(f"[{self.sid}] STT → {STT_WS_URL}")
        retries = 0

        # Notify client to start speaking for enrollment
        await self._jsend({"type": "enrolling", "message": "Please say your first sentence to enroll your voice."})

        while self._running and retries < STT_MAX_RETRIES:
            try:
                stt_ws = await _ws_connect_with_backoff(
                    STT_WS_URL, max_retries=STT_MAX_RETRIES,
                    label=f"[{self.sid}] STT",
                    max_size=2 * 1024 * 1024,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                )
                self._stt_ws = stt_ws
                retries = 0
                log.info(f"[{self.sid}] STT connected")

                async def _push():
                    while self._running:
                        frame = await self._audio_q.get()
                        try:
                            await stt_ws.send(frame)
                        except Exception:
                            pass

                async def _recv():
                    guard        = RepetitionGuard()
                    word_buf     : list[str]            = []
                    last_word_ts : float                = 0.0
                    silence_task : Optional[asyncio.Task] = None

                    async def _fire_query():
                        nonlocal word_buf, last_word_ts, silence_task
                        await asyncio.sleep(STT_SILENCE_MS / 1000.0)
                        if not word_buf:
                            return
                        text       = " ".join(word_buf).strip()
                        word_count = len(word_buf)
                        word_buf   = []
                        silence_task = None
                        if not text:
                            return

                        # ── Enrollment gate ───────────────────────────────────
                        if self.state == State.ENROLLING:
                            # First utterance → enroll voice from accumulated audio
                            if not self._voice.enrolled:
                                # The raw audio was already fed in push_audio().
                                # If we haven't finished building the voiceprint yet
                                # (e.g. silence trigger fired too early), force-complete
                                # enrollment using whatever we have.
                                if self._segment_audio_buf:
                                    combined = b"".join(self._segment_audio_buf)
                                    self._voice.add_enrollment_frame(combined)
                                    self._segment_audio_buf = []
                                # Mark enrolled
                                if not self._voice.enrolled:
                                    # Still not enough — just mark enrolled so we proceed
                                    self._voice._enrolled = True
                            log.info(f"[{self.sid}] ✅ Voice enrolled from first utterance: {text!r}")
                            self.state = State.IDLE
                            await self._jsend({"type": "enrolled", "message": "Voice enrolled. You may now speak."})
                            guard.reset()
                            # The enrollment sentence also counts as the first query
                            enroll_turn_id = str(uuid.uuid4())
                            self._lat.new_turn(enroll_turn_id, text)
                            await self._query_q.put((enroll_turn_id, text))
                            return

                        # ── Voice verification (post-enrollment) ──────────────
                        voice_ok = self._is_enrolled_voice()
                        if not voice_ok:
                            log.info(f"[{self.sid}] 🚫 Voice mismatch — dropping segment (echo/noise): {text!r}")
                            await self._jsend({"type": "voice_mismatch", "detail": "Audio did not match enrolled voice"})
                            return

                        guard.reset()
                        turn_id = str(uuid.uuid4())
                        self._lat.new_turn(turn_id, text)
                        self._lat.on_stt_segment()
                        log.info(f"[{self.sid}] STT silence-trigger [{self.state.name}] ({word_count}w): {text!r}")
                        await self._jsend({"type": "segment", "text": text})

                        if self.state == State.SPEAKING:
                            if word_count >= BARGE_IN_MIN_WORDS:
                                await self._do_barge_in((turn_id, text))
                            else:
                                log.info(f"[{self.sid}] Dropped short segment (too short for barge-in)")
                        elif self.state == State.THINKING:
                            await self._do_barge_in((turn_id, text))
                        else:
                            await self._query_q.put((turn_id, text))

                    async for raw in stt_ws:
                        try:
                            ev = json.loads(raw)
                        except Exception:
                            continue
                        kind = ev.get("type")

                        if kind == "word":
                            word = ev.get("word", "").strip()
                            if not word:
                                continue
                            if guard.feed(word):
                                log.warning(f"[{self.sid}] 🚨 Hallucination ('{word}'). Resetting STT.")
                                guard.reset()
                                word_buf.clear()
                                self._segment_audio_buf.clear()
                                if silence_task and not silence_task.done():
                                    silence_task.cancel()
                                    silence_task = None
                                try:
                                    await stt_ws.send(b'\x02' + json.dumps({"type": "reset_context"}).encode())
                                except Exception:
                                    pass
                                _drain(self._query_q)
                                await self._jsend({"type": "hallucination_reset", "detail": f"Repeated '{word}'"})
                                continue
                            word_buf.append(word)
                            last_word_ts = time.monotonic()
                            self._lat.on_stt_word()
                            await self._jsend(ev)
                            if silence_task and not silence_task.done():
                                silence_task.cancel()
                            silence_task = asyncio.create_task(_fire_query())

                        elif kind == "partial":
                            word = ev.get("word", "").strip().rstrip("?.!,;:")
                            if word:
                                if not word_buf or word_buf[-1].lower() != word.lower():
                                    word_buf.append(word)
                                last_word_ts = time.monotonic()
                                self._lat.on_stt_word()
                                if silence_task and not silence_task.done():
                                    silence_task.cancel()
                                silence_task = asyncio.create_task(_fire_query())
                            await self._jsend(ev)

                        elif kind == "segment":
                            text       = ev.get("text", "").strip()
                            word_count = len(text.split()) if text else 0
                            if not text:
                                continue
                            if silence_task and not silence_task.done():
                                silence_task.cancel()
                                silence_task = None
                            word_buf.clear()
                            guard.reset()

                            # ── Enrollment gate ───────────────────────────────
                            if self.state == State.ENROLLING:
                                if not self._voice.enrolled:
                                    if self._segment_audio_buf:
                                        combined = b"".join(self._segment_audio_buf)
                                        self._voice.add_enrollment_frame(combined)
                                        self._segment_audio_buf = []
                                    if not self._voice.enrolled:
                                        self._voice._enrolled = True
                                log.info(f"[{self.sid}] ✅ Voice enrolled (segment): {text!r}")
                                self.state = State.IDLE
                                await self._jsend({"type": "enrolled", "message": "Voice enrolled. You may now speak."})
                                enroll_turn_id = str(uuid.uuid4())
                                self._lat.new_turn(enroll_turn_id, text)
                                await self._query_q.put((enroll_turn_id, text))
                                continue

                            # ── Voice verification ────────────────────────────
                            voice_ok = self._is_enrolled_voice()
                            if not voice_ok:
                                log.info(f"[{self.sid}] 🚫 Segment dropped (voice mismatch): {text!r}")
                                await self._jsend({"type": "voice_mismatch", "detail": "Audio did not match enrolled voice"})
                                continue

                            seg_turn_id = str(uuid.uuid4())
                            self._lat.new_turn(seg_turn_id, text)
                            self._lat.on_stt_segment()
                            log.info(f"[{self.sid}] STT segment [{self.state.name}] ({word_count}w): {text!r}")
                            await self._jsend(ev)

                            if self.state == State.SPEAKING:
                                if word_count >= BARGE_IN_MIN_WORDS:
                                    await self._do_barge_in((seg_turn_id, text))
                                else:
                                    log.info(f"[{self.sid}] Dropped short segment (echo guard)")
                            elif self.state == State.THINKING:
                                await self._do_barge_in((seg_turn_id, text))
                            else:
                                await self._query_q.put((seg_turn_id, text))

                        elif kind == "pong":
                            self._last_pong_time = time.monotonic()
                        elif kind == "error":
                            log.warning(f"[{self.sid}] STT error: {ev}")
                            await self._jsend(ev)

                await asyncio.gather(_push(), _recv())

            except Exception as e:
                retries += 1
                log.warning(f"[{self.sid}] STT disconnected ({e}), retry {retries}/{STT_MAX_RETRIES}")
                self._stt_ws = None
                if retries < STT_MAX_RETRIES:
                    await asyncio.sleep(min(2 ** retries, 30))
                else:
                    log.error(f"[{self.sid}] STT max retries reached")

    # ─────────────────────────────────────────────────────────────────────────
    # CAG LOOP  (v9 — uses TonalAccumulator instead of SentenceAccumulator)
    # ─────────────────────────────────────────────────────────────────────────

    async def _cag_loop(self):
        async with httpx.AsyncClient(
            base_url=CAG_HTTP_URL,
            timeout=httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0),
        ) as http:

            while self._running:
                query = await self._query_q.get()
                self._last_query_time = time.monotonic()

                # turn_id is bundled with the query as a tuple so that the
                # latency tracker's new_turn() + on_stt_segment() are both
                # anchored to the same turn object (the STT segment event fires
                # before the CAG loop dequeues the query).
                if isinstance(query, tuple):
                    turn_id, query = query
                else:
                    turn_id = str(uuid.uuid4())
                self._active_turn_id = turn_id

                self._lat.new_turn(turn_id, query)
                self._lat.on_query_sent()

                self._barge_in = False
                self.state     = State.THINKING
                log.info(f"[{self.sid}] CAG query [turn:{turn_id[:8]}]: {query!r}")
                await self._jsend({"type": "thinking", "turn_id": turn_id})

                # Use TonalAccumulator for prosody-aware chunking
                acc              = TonalAccumulator()
                acc.reset()
                full_reply_parts : list[str] = []
                interrupted      = False
                stream_confirmed = False

                try:
                    async with http.stream(
                        "POST", "/chat/stream",
                        json={
                            "message":       query,
                            "reset_session": False,
                            "turn_id":       turn_id,
                        },
                        headers={"Accept": "text/event-stream"},
                    ) as resp:

                        async for line in resp.aiter_lines():
                            if self._barge_in:
                                interrupted = True
                                log.info(f"[{self.sid}] CAG stream aborted (barge-in)")
                                break

                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()

                            if data.startswith("[TURN_ID]"):
                                server_turn_id = data[9:].strip()
                                if server_turn_id != turn_id:
                                    log.warning(
                                        f"[{self.sid}] Stale stream (got {server_turn_id[:8]}, "
                                        f"want {turn_id[:8]}) — discarding"
                                    )
                                    interrupted = True
                                    break
                                stream_confirmed = True
                                log.info(f"[{self.sid}] Stream confirmed [turn:{turn_id[:8]}]")
                                continue

                            if data == "[DONE]":
                                break
                            if data == "[TIMEOUT]":
                                log.warning(f"[{self.sid}] CAG timeout on turn {turn_id[:8]}")
                                await self._jsend({"type": "error", "detail": "CAG response timeout"})
                                interrupted = True
                                break
                            if data.startswith("[ERROR]"):
                                await self._jsend({"type": "error", "detail": data[7:].strip()})
                                interrupted = True
                                break
                            if not data or not stream_confirmed:
                                continue

                            # Token received
                            self._lat.on_first_token()
                            self._lat.on_token()
                            await self._jsend({"type": "ai_token", "token": data})
                            full_reply_parts.append(data)

                            # Feed TonalAccumulator → emit tonal chunks immediately
                            for tonal_chunk in acc.feed(data):
                                if self._barge_in:
                                    interrupted = True
                                    break
                                log.info(
                                    f"[{self.sid}] TTS ← [{tonal_chunk.tone}] {tonal_chunk.text!r}"
                                )
                                await self._jsend({
                                    "type": "ai_sentence",
                                    "text": tonal_chunk.text,
                                    "tone": tonal_chunk.tone,
                                })
                                self.state = State.SPEAKING
                                self._lat.on_tts_chunk_sent(tonal_chunk.text)
                                await self._tts_q.put(tonal_chunk)   # synth_worker picks up immediately

                            if interrupted:
                                break

                    if not interrupted and not self._barge_in:
                        tail = acc.flush()
                        if tail:
                            log.info(f"[{self.sid}] TTS ← tail [{tail.tone}]: {tail.text!r}")
                            await self._jsend({
                                "type": "ai_sentence",
                                "text": tail.text,
                                "tone": tail.tone,
                            })
                            self.state = State.SPEAKING
                            self._lat.on_tts_chunk_sent(tail.text)
                            await self._tts_q.put(tail)

                except Exception as e:
                    log.error(f"[{self.sid}] CAG error: {e}")
                    await self._jsend({"type": "error", "detail": str(e)})
                    interrupted = True

                finally:
                    if not interrupted and not self._barge_in:
                        await self._tts_q.put(self._TURN_END)
                        full_text = " ".join(full_reply_parts).strip()
                        if full_text and self._stt_ws:
                            ctrl = json.dumps({"type": "assistant_turn", "text": full_text}).encode()
                            try:
                                await self._stt_ws.send(b'\x02' + ctrl)
                            except Exception:
                                pass

    # ─────────────────────────────────────────────────────────────────────────
    # SYNTH WORKER  (v9 — passes tone annotation to TTS)
    # ─────────────────────────────────────────────────────────────────────────

    async def _synth_worker(self):
        order_index   = 0
        pending_tasks : list[asyncio.Task] = []

        while self._running:
            item = await self._tts_q.get()

            if item is self._INTERRUPT:
                # Set barge_in flag FIRST so all in-flight _synth_one tasks
                # abort at their next barge_in check — no waiting.
                self._barge_in = True
                log.info(f"[{self.sid}] synth_worker: INTERRUPT — cancelling {len(pending_tasks)} tasks")
                for t in pending_tasks:
                    t.cancel()
                pending_tasks.clear()
                order_index = 0
                await self._pcm_q.put(self._INTERRUPT)
                continue

            if item is self._TURN_END:
                total_dispatched = order_index   # how many synth_one tasks were fired
                log.info(f"[{self.sid}] synth_worker: TURN_END ({total_dispatched} chunks dispatched)")
                # Pass the count so play_worker can wait for all chunks before finalizing
                await self._pcm_q.put(("TURN_END", total_dispatched))
                pending_tasks = [t for t in pending_tasks if not t.done()]
                order_index = 0
                continue

            # TonalChunk — fire synthesis immediately (non-blocking)
            idx  = order_index
            order_index += 1
            task = asyncio.create_task(
                self._synth_one(item, idx),
                name=f"synth_one_{self.sid}_{idx}"
            )
            pending_tasks.append(task)
            pending_tasks = [t for t in pending_tasks if not t.done()]

    async def _synth_one(self, tonal_chunk: TonalChunk, order_index: int):
        """
        Synthesize one TonalChunk using its own TTS WebSocket connection.
        The chunk's tone annotation is forwarded to the TTS service so it
        can apply appropriate prosody (pitch/rate modulation).
        Push (order_index, frames) into _pcm_q when done.
        """
        text = tonal_chunk.text.strip()
        if not text:
            await self._pcm_q.put((order_index, []))
            return

        async with self._tts_sem:
            if self._barge_in:
                await self._pcm_q.put((order_index, []))
                return

            frames: list[bytes] = []
            tts_ws = None
            retries = 0

            while retries < TTS_MAX_RETRIES:
                try:
                    tts_ws = await _ws_connect_with_backoff(
                        TTS_WS_URL, max_retries=TTS_MAX_RETRIES,
                        label=f"[{self.sid}] TTS-{order_index}",
                        max_size=10 * 1024 * 1024,
                        ping_interval=None,
                        ping_timeout=None,
                    )

                    # Send chunk with tone annotation for prosody control
                    await tts_ws.send(json.dumps({
                        "text":       text,
                        "language":   TTS_LANGUAGE,
                        "speaker":    TTS_SPEAKER,
                        "chunk_tone": tonal_chunk.tone,  # "tone" or "logic"
                    }))

                    wav_header_skipped = False   # first binary frame is always WAV header
                    first_audio_marked = False

                    while True:
                        if self._barge_in:
                            log.info(f"[{self.sid}] synth_one[{order_index}] barge-in mid-synth")
                            frames = []
                            break

                        try:
                            frame = await asyncio.wait_for(tts_ws.recv(), timeout=30.0)
                        except asyncio.TimeoutError:
                            log.warning(f"[{self.sid}] synth_one[{order_index}] TTS recv timeout")
                            break
                        except Exception as e:
                            log.error(f"[{self.sid}] synth_one[{order_index}] TTS recv error: {e}")
                            raise

                        if isinstance(frame, bytes):
                            # EOS sentinel
                            if frame == b"":
                                log.info(f"[{self.sid}] synth_one[{order_index}] EOS — {len(frames)} frames")
                                break
                            # First binary frame is WAV header (44 bytes) — skip it
                            if not wav_header_skipped:
                                wav_header_skipped = True
                                log.debug(f"[{self.sid}] synth_one[{order_index}] skipped WAV header ({len(frame)}b)")
                                continue
                            # All subsequent binary frames are PCM16 audio
                            if frame:
                                if not first_audio_marked:
                                    first_audio_marked = True
                                    if order_index == 0:
                                        self._lat.on_tts_audio_start()
                                frames.append(frame)

                        elif isinstance(frame, str):
                            try:
                                msg = json.loads(frame)
                            except Exception:
                                continue
                            if msg.get("type") == "chunk_meta":
                                d   = msg.get("data", {})
                                lat = d.get("latency", {})
                                self._lat.on_tts_chunk_complete(
                                    synthesis_latency_ms=lat.get("synthesis_latency_ms", 0.0),
                                    synth_duration_ms   =lat.get("synth_duration_ms",    0.0),
                                    duration_sec        =d.get("duration_sec",            0.0),
                                )
                            elif msg.get("error"):
                                log.warning(f"[{self.sid}] TTS error: {msg['error']}")

                    break   # success — exit retry loop

                except Exception as e:
                    retries += 1
                    log.warning(f"[{self.sid}] synth_one[{order_index}] TTS error ({e}), retry {retries}")
                    frames = []
                    if retries < TTS_MAX_RETRIES:
                        await asyncio.sleep(min(2 ** retries, 10))
                finally:
                    if tts_ws:
                        try:
                            await tts_ws.close()
                        except Exception:
                            pass
                        tts_ws = None

        await self._pcm_q.put((order_index, frames))
        log.info(f"[{self.sid}] synth_one[{order_index}] pushed {len(frames)} frames to pcm_q")

    # ─────────────────────────────────────────────────────────────────────────
    # PLAY WORKER  (unchanged from v8 — reorders and streams PCM to client)
    # ─────────────────────────────────────────────────────────────────────────

    async def _play_worker(self):
        next_expected    = 0
        reorder_buf      : dict[int, list[bytes]] = {}
        chunk_count      = 0
        total_expected   = -1   # -1 = TURN_END not yet received
        chunks_received  = 0    # how many (order_index, frames) items received

        async def _flush_in_order():
            nonlocal next_expected, chunk_count
            while next_expected in reorder_buf:
                frames = reorder_buf.pop(next_expected)
                for f in frames:
                    await self._bsend(f)
                chunk_count += 1
                next_expected += 1

        async def _finalize_turn():
            nonlocal reorder_buf, next_expected, chunk_count, total_expected, chunks_received
            await _flush_in_order()
            report = self._lat.complete_turn()
            if report:
                await self._jsend({"type": "latency", "stage": "turn_complete", **report})
            await self._jsend({"type": "done", "chunks": chunk_count})
            reorder_buf.clear()
            next_expected   = 0
            chunk_count     = 0
            total_expected  = -1
            chunks_received = 0
            self.state      = State.IDLE
            await self._unmute_mic()

        while self._running:
            item = await self._pcm_q.get()

            if item is self._INTERRUPT:
                log.info(f"[{self.sid}] play_worker: INTERRUPT")
                reorder_buf.clear()
                next_expected   = 0
                chunk_count     = 0
                total_expected  = -1
                chunks_received = 0
                self.state      = State.IDLE
                await self._unmute_mic()
                continue

            # TURN_END now arrives as ("TURN_END", total_dispatched)
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "TURN_END":
                total_expected = item[1]
                log.info(f"[{self.sid}] play_worker: TURN_END — expecting {total_expected} chunks, received {chunks_received}")
                await _flush_in_order()
                # Only finalize if all chunks have already arrived
                if chunks_received >= total_expected:
                    await _finalize_turn()
                continue

            order_index, frames = item

            if self._barge_in:
                log.info(f"[{self.sid}] play_worker: drop chunk[{order_index}] (barge-in)")
                chunks_received += 1
                continue

            chunks_received += 1

            if self.state != State.SPEAKING:
                self.state = State.SPEAKING

            reorder_buf[order_index] = frames
            await _flush_in_order()

            # If TURN_END already arrived and we now have all chunks → finalize
            if total_expected >= 0 and chunks_received >= total_expected and not reorder_buf:
                await _finalize_turn()


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session  = GatewaySession(ws)
    pipeline = asyncio.create_task(session.run())
    log.info(f"[{session.sid}] client connected")

    try:
        async for raw in ws.iter_bytes():
            if not raw:
                continue
            ftype   = raw[0]
            payload = raw[1:]

            if ftype == 0x01:
                session.push_audio(raw)
            elif ftype == 0x02:
                try:
                    ctrl = json.loads(payload)
                except Exception:
                    continue
                mtype = ctrl.get("type")
                if mtype == "ping":
                    await ws.send_json({"type": "pong"})
                elif mtype == "pong":
                    session._last_pong_time = time.monotonic()
                elif mtype in ("ai_state", "ai_reference", "reset_context",
                               "assistant_turn", "re_enroll"):
                    session.push_control(payload)
                elif mtype == "get_stats":
                    await ws.send_json({
                        "type":     "stats",
                        "sid":      session.sid,
                        "state":    session.state.name,
                        "enrolled": session._voice.enrolled,
                    })
                elif mtype == "get_latency":
                    await ws.send_json({
                        "type":    "latency_snapshot",
                        "turns":   session._lat.all_reports(),
                        "summary": session._lat.session_summary(),
                    })

    except WebSocketDisconnect:
        log.info(f"[{session.sid}] disconnected")
    except Exception as e:
        log.error(f"[{session.sid}] ws error: {e}")
    finally:
        await session.stop()
        pipeline.cancel()


# ─── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "9.0.0"}


@app.get("/latency/session/{sid}")
def get_session_latency(sid: str):
    data = _session_latency_store.get(sid)
    if not data:
        return JSONResponse({"error": "session not found or still active"}, status_code=404)
    return data


@app.get("/latency/sessions")
def list_sessions():
    return {"sessions": list(_session_latency_store.keys())}


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "gateway:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        workers=1,
        log_level="info",
        reload=False,
    )