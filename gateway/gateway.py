"""
gateway.py — Zero-Latency Voice Pipeline Gateway  v14
══════════════════════════════════════════════════════

ARCHITECTURE
────────────
  Client mic audio ──► STT microservice  (WebSocket, persistent, priority drain)
                            │
                     word / segment events
                            │
                            ▼
                       word_buf  ──► 600ms silence ──► CAG microservice (WebSocket, streaming)
                                                            │
                                                     token stream
                                                            │
                                                     TonalAccumulator
                                                            │
                                               sentence chunks ──► TTS microservice (WebSocket, parallel)
                                                                         │
                                                                    PCM frames
                                                                         │
                                                                    play_worker ──► client

KEY IMPROVEMENTS v14
────────────────────
  1. TTS Voice fingerprint gate moved INTO the gateway (was in STT pipeline).
     Gateway builds a log-mel centroid from every TTS PCM frame it plays.
     Every incoming mic frame is checked: if cosine-similarity >= threshold
     the frame is silently dropped and never forwarded to STT.
     This means the STT service receives clean human-only audio always.
  2. STT no longer receives ai_reference / ai_state control frames.
     AECGate and TTSVoiceGate are removed from the STT pipeline.
  3. Barge-in safety: while state=SPEAKING a higher similarity threshold
     is used so real human speech overlapping TTS echo is not suppressed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# ─── Configuration ────────────────────────────────────────────────────────────

STT_WS_URL          = os.getenv("STT_WS_URL",          "ws://localhost:8001/stream/mux")
CAG_WS_URL          = os.getenv("CAG_WS_URL",          "ws://localhost:8000/chat/ws")   # NEW: WebSocket
CAG_HTTP_URL        = os.getenv("CAG_HTTP_URL",         "http://localhost:8000")         # fallback for /reset
TTS_WS_URL          = os.getenv("TTS_WS_URL",           "ws://localhost:8765/ws/tts")
TTS_ENROLL_URL      = os.getenv("TTS_ENROLL_URL",       "http://localhost:8765/enrollment_status")
# Path to the MFCC fingerprint file saved by tts_microservice enrollment.
# Must match TTS_FINGERPRINT_PATH in tts_microservice.py.
TTS_FINGERPRINT_PATH = os.getenv("TTS_FINGERPRINT_PATH", "tts_voice_fingerprint.npy")
GATEWAY_HOST        = os.getenv("GATEWAY_HOST",         "0.0.0.0")
GATEWAY_PORT        = int(os.getenv("GATEWAY_PORT",     "8090"))
TTS_SPEAKER         = os.getenv("TTS_SPEAKER",          "Claribel Dervla")
TTS_LANGUAGE        = os.getenv("TTS_LANGUAGE",         "en")

BARGE_IN_MIN_WORDS  = int(os.getenv("BARGE_IN_MIN_WORDS",    "1"))
BARGE_IN_COOLDOWN_S = float(os.getenv("BARGE_IN_COOLDOWN_S", "0.6"))
ECHO_TAIL_GUARD_S   = float(os.getenv("ECHO_TAIL_GUARD_S",   "1.5"))
STT_SILENCE_MS      = float(os.getenv("STT_SILENCE_MS",      "600"))

MIN_TTS_CHARS       = int(os.getenv("MIN_TTS_CHARS",          "8"))
TONE_MAX_CHARS      = int(os.getenv("TONE_MAX_CHARS",         "60"))
LOGIC_MAX_CHARS     = int(os.getenv("LOGIC_MAX_CHARS",        "160"))
FIRST_CHUNK_CHARS   = int(os.getenv("FIRST_CHUNK_CHARS",      "20"))

TTS_MAX_PARALLEL    = int(os.getenv("TTS_MAX_PARALLEL",       "4"))
TTS_MAX_RETRIES     = int(os.getenv("TTS_MAX_RETRIES",        "3"))
STT_MAX_RETRIES     = int(os.getenv("STT_MAX_RETRIES",        "5"))
CAG_MAX_RETRIES     = int(os.getenv("CAG_MAX_RETRIES",        "3"))

# STT audio queue: drop oldest frame when full (priority drain)
STT_AUDIO_QUEUE_MAX = int(os.getenv("STT_AUDIO_QUEUE_MAX",   "200"))

HALLUC_WINDOW       = int(os.getenv("HALLUC_WINDOW",          "10"))
HALLUC_THRESHOLD    = int(os.getenv("HALLUC_THRESHOLD",       "5"))
IDLE_TIMEOUT_S      = float(os.getenv("IDLE_TIMEOUT_S",       "120.0"))
HEARTBEAT_S         = float(os.getenv("HEARTBEAT_S",          "25.0"))
ENROLL_WAIT_S       = float(os.getenv("ENROLL_WAIT_S",        "10.0"))

TTS_GREETING        = os.getenv("TTS_GREETING", "")   # Empty = user starts the conversation
TEST_MODE           = os.getenv("TEST_MODE", "0").strip() in ("1", "true", "yes")

WS_PING_INTERVAL = 15
WS_PING_TIMEOUT  = 20

# Echo gate config — see GatewayEchoGate class for details
# (ECHO_* constants are defined alongside the class below)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log     = logging.getLogger("gateway")
lat_log = logging.getLogger("gateway.latency")

app = FastAPI(title="Voice Gateway", version="14.0.0")
_session_latency_store: dict[str, dict] = {}

# ── Module-level shared voice fingerprint ────────────────────────────────────
# Loaded once at startup from the file written by tts_microservice enrollment.
# All GatewaySession instances share this fingerprint via _load_shared_fingerprint().
_shared_fingerprint: Optional[np.ndarray] = None


def _load_shared_fingerprint() -> Optional[np.ndarray]:
    """
    Load the MFCC fingerprint saved by tts_microservice.
    Returns a unit-norm float32 array of shape (N_MFCC,), or None if not found.
    Called once at app startup and whenever a session initialises its gate.
    """
    global _shared_fingerprint
    if _shared_fingerprint is not None:
        return _shared_fingerprint
    try:
        fp = np.load(TTS_FINGERPRINT_PATH).astype(np.float32)
        norm = np.linalg.norm(fp)
        if norm > 1e-9:
            fp = fp / norm
        _shared_fingerprint = fp
        log.info(
            "[fingerprint] ✅ Loaded from %s — shape=%s  norm=%.4f",
            TTS_FINGERPRINT_PATH, fp.shape, float(norm),
        )
        return fp
    except FileNotFoundError:
        log.warning(
            "[fingerprint] File not found: %s — gate will build fingerprint "
            "live from first TTS audio (less accurate). "
            "Start tts_microservice first so it can enroll.", TTS_FINGERPRINT_PATH
        )
        return None
    except Exception as e:
        log.warning("[fingerprint] Load error: %s — falling back to live enrollment", e)
        return None


# ─── Enums ────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE     = auto()
    THINKING = auto()
    SPEAKING = auto()


class ChunkTone(str, Enum):
    TONE  = "tone"
    LOGIC = "logic"


# ─── Latency tracking ─────────────────────────────────────────────────────────

def _r(v: Optional[float]) -> Optional[float]:
    return round(v, 1) if v is not None else None


def _latency_color(ms: float) -> str:
    if ms < 200: return "🟢"
    if ms < 400: return "🟡"
    if ms < 800: return "🟠"
    return "🔴"


def _bar(v: float, mx: float, w: int = 28) -> str:
    f = int((v / mx) * w) if mx else 0
    return "█" * f + "░" * (w - f)


@dataclass
class TurnLatency:
    turn_id:            str   = ""
    query_text:         str   = ""
    barge_in:           bool  = False

    # STT stage
    stt_first_word_ts:  Optional[float] = None
    stt_segment_ts:     Optional[float] = None
    stt_latency_ms:     Optional[float] = None

    # CAG stage
    query_sent_ts:      Optional[float] = None
    first_token_ts:     Optional[float] = None
    cag_first_token_ms: Optional[float] = None
    total_tokens:       int             = 0

    # CAG → TTS handoff
    first_tts_chunk_ts: Optional[float] = None
    cag_to_tts_ms:      Optional[float] = None

    # TTS synthesis
    tts_audio_start_ts: Optional[float] = None
    tts_synth_ms:       Optional[float] = None

    # End-to-end
    e2e_ms:             Optional[float] = None

    tts_chunks:         list = field(default_factory=list)

    def finalize(self):
        if self.stt_first_word_ts and self.stt_segment_ts:
            self.stt_latency_ms     = (self.stt_segment_ts     - self.stt_first_word_ts) * 1000
        if self.query_sent_ts and self.first_token_ts:
            self.cag_first_token_ms = (self.first_token_ts     - self.query_sent_ts)     * 1000
        if self.first_token_ts and self.first_tts_chunk_ts:
            self.cag_to_tts_ms      = (self.first_tts_chunk_ts - self.first_token_ts)    * 1000
        if self.first_tts_chunk_ts and self.tts_audio_start_ts:
            self.tts_synth_ms       = (self.tts_audio_start_ts - self.first_tts_chunk_ts) * 1000
        if self.stt_first_word_ts and self.tts_audio_start_ts:
            self.e2e_ms             = (self.tts_audio_start_ts - self.stt_first_word_ts) * 1000

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


class LatencyTracker:
    def __init__(self, sid: str):
        self.sid      = sid
        self.current: Optional[TurnLatency] = None
        self.history: list[TurnLatency]     = []
        self._tts_first_chunk_ts: Optional[float] = None
        self._tts_chunk_index: int = 0

    def new_turn(self, turn_id: str, query: str):
        self.current             = TurnLatency(turn_id=turn_id, query_text=query)
        self._tts_first_chunk_ts = None
        self._tts_chunk_index    = 0

    def on_stt_first_word(self):
        if self.current and not self.current.stt_first_word_ts:
            self.current.stt_first_word_ts = time.monotonic()

    def on_stt_segment(self):
        if self.current:
            self.current.stt_segment_ts = time.monotonic()

    def on_query_sent(self):
        if self.current:
            self.current.query_sent_ts = time.monotonic()

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

    def on_tts_chunk_sent(self, text: str):
        now = time.monotonic()
        if self.current and not self.current.first_tts_chunk_ts:
            self.current.first_tts_chunk_ts = now
            if self.current.first_token_ts:
                ms = (now - self.current.first_token_ts) * 1000
                self.current.cag_to_tts_ms = ms
                lat_log.info(f"[{self.sid}] CAG→TTS first chunk: {ms:.0f}ms")

    def on_tts_audio_start(self):
        now = time.monotonic()
        if self.current and not self.current.tts_audio_start_ts:
            self.current.tts_audio_start_ts = now
            if self.current.first_tts_chunk_ts:
                ms = (now - self.current.first_tts_chunk_ts) * 1000
                self.current.tts_synth_ms = ms
                lat_log.info(f"[{self.sid}] TTS synth latency: {ms:.0f}ms  {_latency_color(ms)}")

    def on_tts_chunk_complete(
        self,
        synthesis_latency_ms: float,
        synth_duration_ms: float,
        duration_sec: float,
    ):
        if not self.current:
            return
        idx = self._tts_chunk_index
        now = time.monotonic()
        if self._tts_first_chunk_ts is None:
            self._tts_first_chunk_ts = now
            first_chunk_latency_ms   = 0.0
        else:
            first_chunk_latency_ms   = (now - self._tts_first_chunk_ts) * 1000
        self.current.tts_chunks.append({
            "chunk_index":            idx,
            "synthesis_latency_ms":   _r(synthesis_latency_ms),
            "synth_duration_ms":      _r(synth_duration_ms),
            "first_chunk_latency_ms": _r(first_chunk_latency_ms),
            "duration_sec":           _r(duration_sec),
        })
        mx = max((c["synthesis_latency_ms"] or 0) for c in self.current.tts_chunks) or 1
        lat_log.info(
            f"[{self.sid}] TTS chunk {idx}  "
            f"synth={synth_duration_ms:.0f}ms  "
            f"lat={synthesis_latency_ms:.0f}ms [{_bar(synthesis_latency_ms, mx)}]  "
            f"dur={duration_sec:.2f}s"
        )
        self._tts_chunk_index += 1

    def complete_turn(self) -> Optional[dict]:
        if not self.current:
            return None
        self.current.finalize()
        report = self.current.to_report()
        self.history.append(self.current)
        e2e = self.current.e2e_ms
        if e2e is not None:
            lat_log.info(
                f"[{self.sid}] {_latency_color(e2e)} E2E {e2e:.0f}ms  "
                f"[STT {_r(self.current.stt_latency_ms)}ms | "
                f"CAG {_r(self.current.cag_first_token_ms)}ms | "
                f"TTS {_r(self.current.tts_synth_ms)}ms]"
            )
        self.current = None
        return report

    def session_summary(self) -> dict:
        turns = [t for t in self.history if t.e2e_ms is not None]
        if not turns:
            return {"sid": self.sid, "turns": 0}

        def _stats(vals: list) -> dict:
            if not vals:
                return {}
            sv  = sorted(vals)
            p95 = sv[max(0, int(len(sv) * 0.95) - 1)]
            return {
                "min": _r(min(sv)), "max": _r(max(sv)),
                "avg": _r(sum(sv) / len(sv)), "p95": _r(p95),
            }

        stt_lats  = [t.stt_latency_ms     for t in turns if t.stt_latency_ms     is not None]
        cag_lats  = [t.cag_first_token_ms for t in turns if t.cag_first_token_ms is not None]
        tts_lats  = [t.tts_synth_ms       for t in turns if t.tts_synth_ms       is not None]
        e2e_lats  = [t.e2e_ms             for t in turns]
        barge_ins = sum(1 for t in turns if t.barge_in)
        lat_log.info(
            f"[{self.sid}] ══ SESSION SUMMARY ══  turns={len(turns)}  "
            f"barge_ins={barge_ins}  "
            f"e2e avg={_r(sum(e2e_lats)/len(e2e_lats))}ms"
        )
        return {
            "sid": self.sid,
            "turns": len(turns),
            "barge_ins": barge_ins,
            "stt":      _stats(stt_lats),
            "cag":      _stats(cag_lats),
            "tts_synth":_stats(tts_lats),
            "e2e":      _stats(e2e_lats),
        }

    def all_reports(self) -> list[dict]:
        return [t.to_report() for t in self.history]


# ─── Tonal sentence accumulator ───────────────────────────────────────────────

_RE_SENTENCE_END = re.compile(r'(?<=[^\d])([.!?]+["\']?)(?=\s|$)')
_RE_CLAUSE_BREAK = re.compile(r'([,;:—–])\s')
_RE_STARTS_PUNCT = re.compile(r'^[\s,\.!?;:\)\]\}\'\"\\u2019\\u2018\\u201c\\u201d\-]')


def _classify_tone(text: str) -> ChunkTone:
    s = text.strip()
    if s.endswith("?") or s.endswith("!") or len(s) <= TONE_MAX_CHARS:
        return ChunkTone.TONE
    return ChunkTone.LOGIC


@dataclass
class TonalChunk:
    text: str
    tone: ChunkTone


class TonalAccumulator:
    def __init__(self):
        self._buf        = ""
        self._first_sent = True

    def reset(self):
        self._buf        = ""
        self._first_sent = True

    def feed(self, token: str) -> list[TonalChunk]:
        if not token:
            return []
        if token.startswith(" "):
            if self._buf and self._buf[-1] == " ":
                token = token.lstrip(" ")
        else:
            if self._buf and not self._buf[-1].isspace() and not _RE_STARTS_PUNCT.match(token):
                token = " " + token
        self._buf += token
        return self._try_flush()

    def flush(self) -> Optional[TonalChunk]:
        text = self._buf.strip()
        self._buf        = ""
        self._first_sent = True
        if len(text) >= 1:
            return TonalChunk(text=text, tone=_classify_tone(text))
        return None

    def _try_flush(self) -> list[TonalChunk]:
        results: list[TonalChunk] = []
        while True:
            buf = self._buf

            if self._first_sent and len(buf) >= FIRST_CHUNK_CHARS:
                split = buf.rfind(" ")
                if split >= MIN_TTS_CHARS:
                    candidate = buf[:split].strip()
                    remainder = buf[split:].lstrip()
                    if candidate:
                        results.append(TonalChunk(text=candidate, tone=_classify_tone(candidate)))
                        self._buf        = remainder
                        self._first_sent = False
                        continue
                elif len(buf) >= TONE_MAX_CHARS:
                    candidate = buf[:TONE_MAX_CHARS].strip()
                    remainder = buf[TONE_MAX_CHARS:].lstrip()
                    if candidate:
                        results.append(TonalChunk(text=candidate, tone=_classify_tone(candidate)))
                        self._buf        = remainder
                        self._first_sent = False
                        continue
                break

            m = _RE_SENTENCE_END.search(buf)
            if m:
                candidate = buf[:m.end()].strip()
                remainder = buf[m.end():].lstrip()
                if len(candidate) >= MIN_TTS_CHARS or not self._first_sent:
                    results.append(TonalChunk(text=candidate, tone=_classify_tone(candidate)))
                    self._buf        = remainder
                    self._first_sent = False
                    continue
                break

            m = _RE_CLAUSE_BREAK.search(buf)
            if m:
                candidate = buf[:m.start() + 1].strip()
                remainder = buf[m.end():].lstrip()
                if len(candidate) >= MIN_TTS_CHARS and len(remainder) >= 3:
                    results.append(TonalChunk(text=candidate, tone=_classify_tone(candidate)))
                    self._buf        = remainder
                    self._first_sent = False
                    continue
                break

            max_cap = LOGIC_MAX_CHARS if not self._first_sent else TONE_MAX_CHARS
            if len(buf) > max_cap:
                split = buf[:max_cap].rfind(" ")
                if split <= MIN_TTS_CHARS:
                    split = max_cap
                candidate = buf[:split].strip()
                remainder = buf[split:].lstrip()
                if candidate:
                    results.append(TonalChunk(text=candidate, tone=_classify_tone(candidate)))
                    self._buf        = remainder
                    self._first_sent = False
                    continue
                break
            break
        return results


# ─── Repetition guard ─────────────────────────────────────────────────────────

class RepetitionGuard:
    _STOP = frozenset({
        "a","an","the","i","me","my","we","our","you","your",
        "he","she","it","they","his","her","its","their",
        "is","are","was","were","be","been","being",
        "have","has","had","do","does","did",
        "to","of","in","on","at","by","for","with",
        "and","or","but","not","no","so","if","as",
        "that","this","these","those","what","which","who",
        "how","when","where","why","up","out","about","into",
        "from","than","then","can","will","would","could",
        "should","may","might","just","also","very","more","some","any",
    })

    def __init__(self, window=HALLUC_WINDOW, threshold=HALLUC_THRESHOLD):
        self._threshold = max(threshold, 4)
        self._history: list[str] = []
        self._window = window

    def feed(self, word: str) -> bool:
        w = word.lower().strip().rstrip(".,!?;:")
        if not w or w in self._STOP:
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


# ─── AI Text Echo Filter ───────────────────────────────────────────────────────
#
# PROBLEM
# ───────
# When the AI speaks via TTS, the speaker output is picked up by the open mic.
# Even with acoustic cross-correlation gating, some echo leaks through and is
# transcribed by STT verbatim (e.g. STT returns "I'm so glad you reached out").
# These words then trigger false barge-ins, creating a feedback loop.
#
# SOLUTION — Text-layer echo guard
# ─────────────────────────────────
# Keep a rolling buffer of all words the AI has *actually said* in the last
# AI_TEXT_ECHO_WINDOW_S seconds.  When STT returns a word or segment, compute
# the overlap ratio between the STT result and that buffer.  If the ratio
# exceeds AI_TEXT_ECHO_RATIO the result is classified as echo and dropped.
#
# This is complementary to (not a replacement for) the acoustic gate:
#   • Acoustic gate catches echo before it reaches STT.
#   • Text filter catches anything that slips through (cheap, zero false-positive
#     risk for genuine human speech about unrelated topics).
#
# Tuning
# ──────
#   AI_TEXT_ECHO_WINDOW_S = 6.0   # seconds of AI speech to remember
#   AI_TEXT_ECHO_RATIO    = 0.60  # word-overlap ratio to count as echo
#   AI_TEXT_ECHO_MIN_WORDS= 1     # minimum STT words to bother checking

AI_TEXT_ECHO_WINDOW_S  = float(os.getenv("AI_TEXT_ECHO_WINDOW_S",  "30.0"))  # STT can hallucinate AI words long after TTS
AI_TEXT_ECHO_RATIO     = float(os.getenv("AI_TEXT_ECHO_RATIO",     "0.55"))  # slightly more aggressive
AI_TEXT_ECHO_MIN_WORDS = int(os.getenv("AI_TEXT_ECHO_MIN_WORDS",   "1"))

_ECHO_STOP = frozenset({
    "a","an","the","i","me","my","we","our","you","your",
    "is","are","was","were","to","of","in","on","at","by",
    "and","or","but","not","so","it","its","be","do","did",
})


class AITextEchoFilter:
    """
    Text-level echo filter.  Records words the AI speaks; drops STT results
    that are predominantly drawn from recent AI speech.

    Usage:
        f = AITextEchoFilter()
        # When AI says a TTS sentence:
        f.feed_ai_text("Hello! I'm so glad you reached out to Ask Novation.")
        # When STT returns a word:
        if f.is_echo_word("reached"):   # True → drop
            ...
        # When STT returns a segment:
        if f.is_echo_segment("reached out to"):  # True → drop
            ...
    """

    def __init__(self):
        # list of (timestamp, word_lowercase)
        self._ai_words: list[tuple[float, str]] = []

    def feed_ai_text(self, text: str):
        """Call with every sentence/chunk the AI sends to TTS."""
        now = time.monotonic()
        words = [w.lower().strip(".,!?;:\"'") for w in text.split()]
        for w in words:
            if w:
                self._ai_words.append((now, w))
        self._expire()

    def _expire(self):
        cutoff = time.monotonic() - AI_TEXT_ECHO_WINDOW_S
        self._ai_words = [(t, w) for t, w in self._ai_words if t >= cutoff]

    def _recent_ai_set(self) -> frozenset:
        self._expire()
        return frozenset(w for _, w in self._ai_words)

    def is_echo_word(self, word: str) -> bool:
        """Return True if a single STT word came from recent AI speech."""
        w = word.lower().strip(".,!?;:\"'")
        if not w or w in _ECHO_STOP:
            return False
        return w in self._recent_ai_set()

    def is_echo_segment(self, text: str) -> bool:
        """
        Return True if an STT segment is predominantly composed of recent AI words.
        Uses content-word overlap ratio.
        """
        if not text:
            return False
        words = [w.lower().strip(".,!?;:\"'") for w in text.split()]
        # Filter to content words only (skip stop words for ratio calc)
        content = [w for w in words if w and w not in _ECHO_STOP]
        if len(content) < AI_TEXT_ECHO_MIN_WORDS:
            return False
        ai_set = self._recent_ai_set()
        overlap = sum(1 for w in content if w in ai_set)
        ratio = overlap / len(content)
        if ratio >= AI_TEXT_ECHO_RATIO:
            log.info(
                f"[AITextEchoFilter] ECHO segment ({ratio:.0%} overlap): {text!r}"
            )
            return True
        return False

    def reset(self):
        self._ai_words.clear()


# ─── TTS Voice Fingerprint Gate ───────────────────────────────────────────────
#
# DESIGN
# ──────
# Cross-correlation (old approach) requires the mic to capture audio at nearly
# the exact same moment — it fails with room delay, reverb, or any buffering
# lag.  This gate instead identifies the AI's voice by SPECTRAL IDENTITY using
# MFCC features, which is timing-independent: it asks "does this mic frame
# sound like the AI voice?" regardless of when it arrives.
#
# HOW IT WORKS
# ────────────
# Phase 1 — ENROLLMENT (feed_tts calls while AI is speaking):
#   Every TTS PCM chunk is resampled to 16 kHz, split into 25 ms frames,
#   and 13 MFCC coefficients are extracted per frame.
#   These are averaged into a running centroid = the AI VOICE FINGERPRINT.
#   The fingerprint is a single 13-d unit vector representing the spectral
#   identity of the AI's voice across all utterances so far.
#
# Phase 2 — GATE CHECK (check calls from mic):
#   Each mic frame is analysed identically.  Cosine similarity is computed
#   between the mic MFCCs and the fingerprint.  If the highest similarity
#   frame >= threshold AND the gate is armed (TTS active or tail guard)
#   → DROP (AI echo).  Otherwise → PASS (human voice or silence).
#
# BARGE-IN SAFETY
# ────────────────
#   When the human speaks over the AI, the human voice has a different
#   spectral identity and will score below threshold → passes through.
#   Only frames that genuinely sound like the AI voice are dropped.
#
# TUNING (env vars)
# ──────
#   VFGATE_SIM_THRESHOLD = 0.85   cosine similarity to classify as AI
#   VFGATE_TAIL_S        = 2.0    seconds gate stays armed after TTS stops
#   VFGATE_ENERGY_FLOOR  = 0.002  RMS below this → silence → always pass
#   VFGATE_N_MFCC        = 13     MFCC coefficient count
#   VFGATE_FRAME_MS      = 25     analysis window in milliseconds

VFGATE_SIM_THRESHOLD = float(os.getenv("VFGATE_SIM_THRESHOLD", "0.85"))
VFGATE_TAIL_S        = float(os.getenv("VFGATE_TAIL_S",        "2.0"))
VFGATE_ENERGY_FLOOR  = float(os.getenv("VFGATE_ENERGY_FLOOR",  "0.002"))
VFGATE_N_MFCC        = int(os.getenv("VFGATE_N_MFCC",          "13"))
VFGATE_FRAME_MS      = int(os.getenv("VFGATE_FRAME_MS",        "25"))

# Alias used by existing ECHO_TAIL_GUARD_S logic elsewhere in the file
ECHO_TAIL_S = VFGATE_TAIL_S


class TTSVoiceFingerprintGate:
    """
    MFCC-based TTS voice fingerprint gate.  Drop-in replacement for the old
    cross-correlation GatewayEchoGate — same public API.

    Usage:
        gate = TTSVoiceFingerprintGate()
        gate.feed_tts(pcm_bytes_24k_int16)   # called for every TTS frame played
        gate.tts_stopped()                    # called when TTS finishes
        if gate.check(mic_pcm_bytes_16k):     # True → drop mic frame
            continue
    """

    MIC_SR = 16_000
    TTS_SR = 24_000

    def __init__(self):
        self._sr        = self.MIC_SR
        self._frame_len = int(self._sr * VFGATE_FRAME_MS / 1000)
        self._n_mfcc    = VFGATE_N_MFCC

        # Running mean of all MFCC vectors seen from TTS audio
        self._fp_sum   = np.zeros(self._n_mfcc, dtype=np.float64)
        self._fp_count = 0
        self._fingerprint: Optional[np.ndarray] = None   # unit-norm centroid

        self._tts_active     = False
        self._tts_stopped_at = 0.0

        self.frames_checked = 0
        self.frames_dropped = 0
        self.last_sim       = 0.0

        # Accumulation buffer — mic sends ~160 samples (10ms) per chunk but
        # MFCC extraction needs at least _frame_len (400) samples. We buffer
        # chunks until we have enough, then run the check on the full window.
        self._mic_buf = np.array([], dtype=np.float32)

    # ── MFCC helpers (pure numpy — no librosa required) ───────────────────────

    @staticmethod
    def _preemph(s: np.ndarray, c: float = 0.97) -> np.ndarray:
        return np.append(s[0], s[1:] - c * s[:-1])

    @staticmethod
    def _mel_fb(n_fft: int, n_mels: int, sr: int) -> np.ndarray:
        def h2m(h): return 2595 * np.log10(1 + h / 700)
        def m2h(m): return 700 * (10 ** (m / 2595) - 1)
        pts  = m2h(np.linspace(h2m(0), h2m(sr / 2), n_mels + 2))
        bins = np.floor((n_fft + 1) * pts / sr).astype(int)
        nb   = n_fft // 2 + 1
        fb   = np.zeros((n_mels, nb))
        for m in range(1, n_mels + 1):
            lo, mid, hi = bins[m-1], bins[m], bins[m+1]
            if mid > lo:
                fb[m-1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
            if hi > mid:
                fb[m-1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
        return fb

    def _extract_mfcc(self, pcm: np.ndarray) -> Optional[np.ndarray]:
        """Return (n_frames, n_mfcc) MFCC matrix or None if audio too short."""
        if len(pcm) < self._frame_len:
            return None
        n_mels  = 26
        n_fft   = self._frame_len
        hop     = n_fft // 2
        fb      = self._mel_fb(n_fft, n_mels, self._sr)
        sig     = self._preemph(pcm)
        out     = []
        for i in range(0, len(sig) - n_fft + 1, hop):
            w   = sig[i:i+n_fft] * np.hanning(n_fft)
            ps  = np.abs(np.fft.rfft(w, n=n_fft)) ** 2
            mel = np.maximum(np.dot(fb, ps), 1e-10)
            lm  = np.log(mel)
            N   = len(lm)
            k   = np.arange(self._n_mfcc)
            # Vectorised DCT-II
            dct = np.sum(lm[:, None] * np.cos(np.pi * k[None, :] * (np.arange(N)[:, None] + 0.5) / N), axis=0)
            out.append(dct)
        return np.array(out) if out else None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def feed_tts(self, pcm_int16_bytes: bytes):
        """Ingest one TTS PCM chunk (int16, 24 kHz) → update fingerprint."""
        if not pcm_int16_bytes or len(pcm_int16_bytes) < 2:
            return
        if len(pcm_int16_bytes) % 2 != 0:
            pcm_int16_bytes = pcm_int16_bytes[:-1]

        pcm24 = np.frombuffer(pcm_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        n_out = max(1, int(len(pcm24) * self.MIC_SR / self.TTS_SR))
        pcm16 = np.interp(
            np.linspace(0, 1, n_out), np.linspace(0, 1, len(pcm24)), pcm24,
        ).astype(np.float32)

        frames = self._extract_mfcc(pcm16)
        if frames is not None and len(frames):
            self._fp_sum   += frames.sum(axis=0)
            self._fp_count += len(frames)
            c  = self._fp_sum / self._fp_count
            n  = np.linalg.norm(c)
            if n > 1e-9:
                self._fingerprint = (c / n).astype(np.float32)

        self._tts_active = True
        log.debug(f"[VFGate] feed_tts: +{len(frames) if frames is not None else 0} frames  total={self._fp_count}")

    def tts_stopped(self):
        """Signal TTS playback ended — arms the tail guard."""
        self._tts_active     = False
        self._tts_stopped_at = time.monotonic()
        log.debug(f"[VFGate] tts_stopped — {self._fp_count} frames enrolled")

    def _is_armed(self) -> bool:
        if self._tts_active:
            return True
        if self._fingerprint is None:
            return False
        return (time.monotonic() - self._tts_stopped_at) < VFGATE_TAIL_S

    def check(self, mic_pcm_bytes: bytes, ai_speaking: bool = False) -> bool:
        """
        Returns True → DROP (AI voice detected in mic frame).
        mic_pcm_bytes : raw int16 PCM at 16 kHz, no WAV header.
        ai_speaking   : use tighter threshold while AI is actively playing.
        """
        self.frames_checked += 1

        if self._fingerprint is None or not self._is_armed() or not mic_pcm_bytes:
            return False

        if len(mic_pcm_bytes) % 2 != 0:
            mic_pcm_bytes = mic_pcm_bytes[:-1]

        # Accumulate into buffer — mic sends small chunks (~160 samples/10ms)
        # but MFCC needs at least _frame_len samples to work
        chunk = np.frombuffer(mic_pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._mic_buf = np.concatenate([self._mic_buf, chunk])

        # Not enough data yet — don't drop, keep buffering
        if len(self._mic_buf) < self._frame_len:
            return False

        # We have enough — work on the accumulated buffer then clear it
        mic = self._mic_buf.copy()
        self._mic_buf = np.array([], dtype=np.float32)

        # Energy floor — silence is never AI echo
        rms = float(np.sqrt(np.mean(mic ** 2)))
        if rms < VFGATE_ENERGY_FLOOR:
            self.last_sim = 0.0
            return False

        frames = self._extract_mfcc(mic)
        if frames is None or not len(frames):
            self.last_sim = 0.0
            return False

        # If ANY frame sounds like the AI voice → drop
        max_sim = max(self._cosine(f, self._fingerprint) for f in frames)
        self.last_sim = max_sim

        thresh = VFGATE_SIM_THRESHOLD * 0.90 if ai_speaking else VFGATE_SIM_THRESHOLD

        if max_sim >= thresh:
            self.frames_dropped += 1
            log.debug(f"[VFGate] DROP  sim={max_sim:.3f}  thresh={thresh:.2f}  rms={rms:.4f}")
            return True

        log.debug(f"[VFGate] PASS  sim={max_sim:.3f}  thresh={thresh:.2f}  rms={rms:.4f}")
        return False

    def reset(self):
        self._fp_sum[:]      = 0
        self._fp_count       = 0
        self._fingerprint    = None
        self._tts_active     = False
        self._tts_stopped_at = 0.0
        self.frames_checked  = 0
        self.frames_dropped  = 0
        self.last_sim        = 0.0
        self._mic_buf        = np.array([], dtype=np.float32)

    @property
    def is_enrolled(self) -> bool:
        return self._fingerprint is not None and self._fp_count >= 10

    def load_fingerprint(self, fp: np.ndarray):
        """
        Inject a pre-computed fingerprint (from tts_microservice enrollment file).
        This replaces any live-accumulated fingerprint immediately.
        fp: unit-norm float32 array of shape (n_mfcc,)
        """
        norm = np.linalg.norm(fp)
        if norm < 1e-9:
            log.warning("[VFGate] load_fingerprint: zero-norm vector ignored")
            return
        self._fingerprint = (fp / norm).astype(np.float32)
        # Fake a large frame count so is_enrolled returns True immediately
        self._fp_count    = 10_000
        log.info("[VFGate] Fingerprint injected — shape=%s  norm=%.4f",
                 fp.shape, float(norm))


# Alias so any external code still referencing GatewayEchoGate keeps working
GatewayEchoGate = TTSVoiceFingerprintGate



# ─── Helpers ──────────────────────────────────────────────────────────────────

def _drain_q(q: asyncio.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break


async def _ws_connect(url: str, max_retries: int, label: str, **kwargs):
    """Connect to a WebSocket with exponential backoff."""
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            return await websockets.connect(url, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise
            log.warning(f"{label} connect failed ({e}), retry {attempt}/{max_retries} in {delay:.1f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)


# ─── Gateway session ──────────────────────────────────────────────────────────

class GatewaySession:
    """
    Full voice pipeline session.

    Parallel tasks:
      _stt_loop      — streams mic audio to STT with priority drain; receives words/segments/barge_in
      _cag_loop      — receives queries, opens persistent WebSocket to CAG, streams tokens
      _synth_worker  — synthesizes TTS chunks in parallel (up to TTS_MAX_PARALLEL)
      _play_worker   — reorders + plays audio to client in chunk order
      _heartbeat     — keepalive pings
      _idle_watchdog — resets CAG after IDLE_TIMEOUT_S
    """

    _INTERRUPT = object()
    _TURN_END  = object()

    def __init__(self, ws: WebSocket):
        self.ws       = ws
        self.sid      = str(uuid.uuid4())[:8]
        self.state    = State.IDLE
        self._running = True

        # ── Queues ─────────────────────────────────────────────────────────────
        # STT audio: priority — drop oldest on overflow so real-time mic stays current
        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=STT_AUDIO_QUEUE_MAX)
        self._query_q: asyncio.Queue        = asyncio.Queue()
        self._tts_q:   asyncio.Queue        = asyncio.Queue()
        self._pcm_q:   asyncio.Queue        = asyncio.Queue()

        self._tts_sem = asyncio.Semaphore(TTS_MAX_PARALLEL)

        self._stt_ws: Optional[object]  = None
        self._cag_ws: Optional[object]  = None   # persistent CAG WebSocket

        self._barge_in        = False
        self._barge_in_until  = 0.0
        self._tts_stopped_at  = 0.0
        self._last_pong_time  = time.monotonic()
        self._last_query_time = time.monotonic()

        self._stt_ready  = asyncio.Event()
        self._lat        = LatencyTracker(self.sid)
        self._echo_gate        = TTSVoiceFingerprintGate()  # MFCC voice fingerprint gate (speaker-identity based)
        self._text_echo_filter = AITextEchoFilter()         # text-layer fallback
        self._gate_ready       = asyncio.Event()            # set once fingerprint is ready

        # Inject pre-computed fingerprint from TTS enrollment file (if available).
        # If the file exists the gate is immediately armed — no live synthesis needed.
        shared_fp = _load_shared_fingerprint()
        if shared_fp is not None:
            self._echo_gate.load_fingerprint(shared_fp)
            self._gate_ready.set()   # unblock _push() immediately
            log.info(f"[{self.sid}] Voice fingerprint loaded — gate armed from startup")

        log.info(f"[{self.sid}] session created")

    # ── Safe send helpers ─────────────────────────────────────────────────────

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

    # ── Entry ─────────────────────────────────────────────────────────────────

    async def run(self):
        tasks = [
            asyncio.create_task(self._stt_loop(),     name="stt"),
            asyncio.create_task(self._cag_loop(),      name="cag"),
            asyncio.create_task(self._synth_worker(),  name="synth"),
            asyncio.create_task(self._play_worker(),   name="play"),
            asyncio.create_task(self._heartbeat(),     name="heartbeat"),
            asyncio.create_task(self._idle_watchdog(), name="idle"),
        ]
        if not TEST_MODE:
            asyncio.create_task(self._startup_sequence(), name="startup")

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
        self._echo_gate.reset()
        self._text_echo_filter.reset()
        summary = self._lat.session_summary()
        _session_latency_store[self.sid] = {
            "summary": summary,
            "turns":   self._lat.all_reports(),
        }
        await self._jsend({"type": "session_summary", "latency": summary})
        # Close persistent CAG WS
        if self._cag_ws:
            try:
                await self._cag_ws.close()
            except Exception:
                pass

    # ── Startup & greeting ────────────────────────────────────────────────────

    async def _startup_sequence(self):
        log.info(f"[{self.sid}] Waiting for STT ready…")
        try:
            await asyncio.wait_for(self._stt_ready.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            log.warning(f"[{self.sid}] STT not ready after 15s — continuing anyway")

        # ── Step 1: Pre-enroll the voice gate ────────────────────────────────
        # Synthesize a short phrase via TTS, feed all PCM frames into the gate.
        # _push() is BLOCKED on _voice_gate_ready until this completes.
        # This guarantees the fingerprint is ready before any mic audio reaches STT.
        await self._pre_enroll_voice_gate()

        enrolled  = False
        deadline  = time.monotonic() + ENROLL_WAIT_S
        log.info(f"[{self.sid}] Polling TTS enrollment (max {ENROLL_WAIT_S:.0f}s)…")
        async with httpx.AsyncClient(timeout=3.0) as http:
            while time.monotonic() < deadline:
                try:
                    r    = await http.get(TTS_ENROLL_URL)
                    data = r.json()
                    if data.get("enrolled"):
                        enrolled = True
                        log.info(f"[{self.sid}] ✅ TTS voice profile confirmed")
                        break
                    progress = data.get("progress", 0)
                    await self._jsend({"type": "tts_enroll_progress", "pct": int(progress * 100)})
                except Exception as e:
                    log.debug(f"[{self.sid}] enrollment poll: {e}")
                await asyncio.sleep(0.5)

        if not enrolled:
            log.warning(f"[{self.sid}] TTS enrollment not confirmed after {ENROLL_WAIT_S}s — continuing")

        await self._jsend({"type": "tts_enrolled", "message": "TTS voice profile ready"})
        # No greeting — user starts the conversation.
        # Send a "ready" event so the client knows the pipeline is live.
        await self._jsend({"type": "ready", "message": "Pipeline ready — speak to begin"})


    async def _pre_enroll_voice_gate(self):
        """
        Ensure the voice fingerprint gate is armed before mic audio flows to STT.

        Fast path: if the fingerprint was already loaded from disk in __init__,
        the gate is already armed (_gate_ready is set) — nothing to do.

        Slow path: fingerprint file not found → synthesize a short phrase via TTS,
        extract MFCC frames live, and build the fingerprint on the fly.
        This is less accurate than the full enrollment but better than nothing.
        """
        if self._gate_ready.is_set():
            log.info(f"[{self.sid}] Voice gate already armed from fingerprint file — skipping live enrollment")
            return

        ENROLL_PHRASE = "Hello, I am Nova. I am happy to help you today."
        log.info(f"[{self.sid}] Fingerprint file not found — live-enrolling from TTS phrase…")
        tts_ws = None
        try:
            tts_ws = await _ws_connect(
                TTS_WS_URL, max_retries=3,
                label=f"[{self.sid}] echo-gate-init",
                max_size=10 * 1024 * 1024,
                ping_interval=None, ping_timeout=None,
            )
            await tts_ws.send(json.dumps({
                "text":     ENROLL_PHRASE,
                "language": TTS_LANGUAGE,
                "speaker":  TTS_SPEAKER,
            }))
            wav_skipped = False
            async for frame in tts_ws:
                if not isinstance(frame, bytes):
                    continue
                if frame == b"":
                    break
                if not wav_skipped:
                    wav_skipped = True
                    continue
                if frame:
                    self._echo_gate.feed_tts(frame)
            # Do NOT call tts_stopped() — that starts the 2s countdown immediately.
            # Keep _tts_active=True so the gate stays armed until real playback takes over.
            self._echo_gate._tts_active = True
            log.info(f"[{self.sid}] ✅ Live voice gate enrollment done ({self._echo_gate._fp_count} MFCC frames) — gate kept armed")
        except Exception as e:
            log.warning(f"[{self.sid}] Live voice gate enrollment failed: {e} — mic unblocked without fingerprint")
        finally:
            if tts_ws:
                try: await tts_ws.close()
                except Exception: pass
        self._gate_ready.set()   # always unblock, even on error

    async def _play_greeting(self):
        log.info(f"[{self.sid}] Playing greeting: {TTS_GREETING!r}")
        tts_ws = None
        try:
            tts_ws = await _ws_connect(
                TTS_WS_URL, max_retries=TTS_MAX_RETRIES,
                label=f"[{self.sid}] greeting",
                max_size=10 * 1024 * 1024,
                ping_interval=None, ping_timeout=None,
            )
            await tts_ws.send(json.dumps({
                "text":     TTS_GREETING,
                "language": TTS_LANGUAGE,
                "speaker":  TTS_SPEAKER,
            }))
            wav_skipped = False
            self.state  = State.SPEAKING
            await self._jsend({"type": "ai_sentence", "text": TTS_GREETING,
                                "tone": _classify_tone(TTS_GREETING)})
            async for frame in tts_ws:
                if isinstance(frame, bytes):
                    if frame == b"":
                        break
                    if not wav_skipped:
                        wav_skipped = True
                        continue
                    if frame:
                        await self._bsend(frame)
                        self._echo_gate.feed_tts(frame)   # buffer for echo detection
        except Exception as e:
            log.warning(f"[{self.sid}] Greeting error: {e}")
        finally:
            if tts_ws:
                try:
                    await tts_ws.close()
                except Exception:
                    pass
        self._tts_stopped_at = time.monotonic()
        self.state = State.IDLE
        await self._jsend({"type": "done", "chunks": 1})
        self._echo_gate.tts_stopped()
        if not self._gate_ready.is_set():
            self._gate_ready.set()   # ensure mic unblocked after greeting
        log.info(f"[{self.sid}] Greeting done")

    # ── STT control helpers ───────────────────────────────────────────────────

    # ── Audio push: priority drain ────────────────────────────────────────────

    def push_audio(self, frame: bytes):
        """
        Push mic audio frame with priority drain.
        If the queue is full, drop the OLDEST frame (head) so the newest
        audio (tail) is always available for STT — ensuring minimal latency
        when the user speaks.
        """
        self._last_pong_time = time.monotonic()
        if self._audio_q.full():
            try:
                self._audio_q.get_nowait()   # drop oldest — prioritize latest
            except asyncio.QueueEmpty:
                pass
        self._audio_q.put_nowait(frame)

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat(self):
        while self._running:
            await asyncio.sleep(HEARTBEAT_S)
            await self._jsend({"type": "ping"})
            if time.monotonic() - self._last_pong_time > HEARTBEAT_S * 2:
                log.warning(f"[{self.sid}] heartbeat timeout — closing")
                self._running = False
                break

    # ── Idle watchdog ─────────────────────────────────────────────────────────

    async def _idle_watchdog(self):
        while self._running:
            await asyncio.sleep(10)
            if (time.monotonic() - self._last_query_time > IDLE_TIMEOUT_S
                    and self.state == State.IDLE):
                log.info(f"[{self.sid}] idle reset")
                try:
                    async with httpx.AsyncClient(base_url=CAG_HTTP_URL, timeout=5.0) as http:
                        await http.post("/reset")
                except Exception as e:
                    log.debug(f"[{self.sid}] idle /reset: {e}")
                self._last_query_time = time.monotonic()
                await self._jsend({"type": "session_reset", "reason": "idle_timeout"})

    # ── Barge-in ──────────────────────────────────────────────────────────────

    async def _do_barge_in_immediate(self):
        now = time.monotonic()
        if now < self._barge_in_until:
            log.debug(f"[{self.sid}] barge-in suppressed (cooldown)")
            return
        log.info(f"[{self.sid}] ⚡ BARGE-IN — stopping TTS immediately")

        if self._lat.current:
            self._lat.current.barge_in = True

        self._barge_in       = True
        self._barge_in_until = now + BARGE_IN_COOLDOWN_S

        _drain_q(self._tts_q)
        _drain_q(self._pcm_q)
        await self._tts_q.put(self._INTERRUPT)
        await self._pcm_q.put(self._INTERRUPT)

        # Cancel in-flight CAG stream by closing the WS
        if self._cag_ws:
            try:
                await self._cag_ws.close()
            except Exception:
                pass
            self._cag_ws = None

        await self._jsend({"type": "barge_in"})

    # ─── STT loop ─────────────────────────────────────────────────────────────

    async def _stt_loop(self):
        """
        Streams mic audio to STT microservice over a persistent WebSocket.
        Audio frames are priority-drained: newest frames always get through first.
        Handles: word, partial, segment, barge_in events from STT.
        """
        log.info(f"[{self.sid}] STT → {STT_WS_URL}")
        retries = 0

        while self._running and retries < STT_MAX_RETRIES:
            try:
                stt_ws = await _ws_connect(
                    STT_WS_URL, max_retries=STT_MAX_RETRIES,
                    label=f"[{self.sid}] STT",
                    max_size=2 * 1024 * 1024,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                )
                self._stt_ws = stt_ws
                retries      = 0
                log.info(f"[{self.sid}] STT connected ✓")
                self._stt_ready.set()

                async def _push():
                    """
                    Push audio frames to STT — reads from priority-drain queue.
                    BLOCKS until the TTS voice gate is enrolled (min_frames seen).
                    Once enrolled, frames matching the TTS fingerprint are dropped.
                    """
                    # Wait until gate has enrolled enough TTS frames to be reliable.
                    # This prevents mic echo leaking to STT during first TTS response.
                    if not self._gate_ready.is_set():
                        log.info(f"[{self.sid}] _push waiting for echo gate…")
                        await self._gate_ready.wait()
                        log.info(f"[{self.sid}] _push unblocked — echo gate ready")
                    while self._running:
                        frame = await self._audio_q.get()
                        try:
                            pcm_bytes = frame[1:] if frame and frame[0] == 0x01 else frame
                            is_tts_voice = self._echo_gate.check(pcm_bytes, ai_speaking=(self.state == State.SPEAKING))
                            log.info(
                                "[%s] chunk | tts_voice=%s | sim=%.3f | armed=%s | drop=%s",
                                self.sid, is_tts_voice,
                                self._echo_gate.last_sim,
                                self._echo_gate._is_armed(),
                                is_tts_voice,
                            )
                            if is_tts_voice:
                                continue
                            await stt_ws.send(frame if (frame and frame[0] == 0x01) else b'\x01' + frame)
                        except Exception:
                            pass

                async def _recv():
                    guard = RepetitionGuard()
                    word_buf: list[str]              = []
                    silence_task: Optional[asyncio.Task] = None
                    barge_triggered_this_turn: bool  = False

                    async def _fire_query():
                        """
                        Fires 600ms after the last confirmed STT word.
                        Builds the full sentence and sends to CAG immediately.
                        """
                        nonlocal word_buf, silence_task, barge_triggered_this_turn
                        await asyncio.sleep(STT_SILENCE_MS / 1000.0)
                        if not word_buf:
                            return

                        text       = " ".join(word_buf).strip()
                        word_count = len(word_buf)
                        word_buf   = []
                        silence_task              = None
                        barge_triggered_this_turn = False

                        if not text:
                            return

                        in_echo_tail = (time.monotonic() - self._tts_stopped_at) < ECHO_TAIL_GUARD_S
                        if word_count < 2 and self.state == State.IDLE and in_echo_tail:
                            log.info(f"[{self.sid}] echo-tail drop: {text!r}")
                            return

                        # Text-layer echo gate: drop segments that are mostly AI's own words
                        if self._text_echo_filter.is_echo_segment(text):
                            log.info(f"[{self.sid}] text-echo drop (fire_query): {text!r}")
                            return

                        guard.reset()
                        turn_id = str(uuid.uuid4())
                        self._lat.new_turn(turn_id, text)
                        self._lat.on_stt_segment()
                        log.info(f"[{self.sid}] STT silence [{self.state.name}] ({word_count}w): {text!r}")
                        await self._jsend({"type": "segment", "text": text})

                        if self.state in (State.SPEAKING, State.THINKING):
                            if word_count >= BARGE_IN_MIN_WORDS and self._barge_in:
                                _drain_q(self._query_q)
                                await self._query_q.put((turn_id, text))
                                log.info(f"[{self.sid}] barge-in query: {text!r}")
                            elif word_count >= BARGE_IN_MIN_WORDS:
                                await self._do_barge_in_immediate()
                                _drain_q(self._query_q)
                                await self._query_q.put((turn_id, text))
                        else:
                            await self._query_q.put((turn_id, text))

                    async for raw in stt_ws:
                        if isinstance(raw, bytes) and len(raw) > 1:
                            ftype = raw[0]
                            if ftype != 0x01:
                                continue
                            payload = raw[1:]
                        elif isinstance(raw, str):
                            payload = raw.encode()
                        else:
                            continue

                        try:
                            ev   = json.loads(payload)
                        except Exception:
                            continue

                        kind = ev.get("type", "")

                        # ── Barge-in: human voice detected while AI speaking ─────
                        if kind == "barge_in":
                            if not barge_triggered_this_turn:
                                barge_triggered_this_turn = True
                                await self._do_barge_in_immediate()
                            continue

                        # ── Word: add to buffer, reset silence timer ─────────────
                        elif kind == "word":
                            word = ev.get("word", "").strip().rstrip("?.!,;:")
                            if word:
                                # Text-layer echo gate: drop words the AI just said.
                                # Applied regardless of state — STT can hallucinate AI
                                # words long after TTS finishes (primed by context).
                                if self._text_echo_filter.is_echo_word(word):
                                    log.debug(f"[{self.sid}] text-echo drop word: {word!r}")
                                    continue

                                if guard.feed(word):
                                    log.warning(f"[{self.sid}] hallucination reset")
                                    word_buf.clear()
                                    guard.reset()
                                    if silence_task and not silence_task.done():
                                        silence_task.cancel()
                                    silence_task = None
                                    await self._jsend({"type": "hallucination_reset"})
                                    continue

                                if not word_buf or word_buf[-1].lower() != word.lower():
                                    word_buf.append(word)
                                self._lat.on_stt_first_word()

                                # Reset silence timer on every new word
                                if silence_task and not silence_task.done():
                                    silence_task.cancel()
                                silence_task = asyncio.create_task(_fire_query())
                            await self._jsend(ev)

                        # ── Segment: full sentence from Whisper flush ────────────
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
                            barge_triggered_this_turn = False

                            in_echo_tail = (time.monotonic() - self._tts_stopped_at) < ECHO_TAIL_GUARD_S
                            if word_count < 2 and self.state == State.IDLE and in_echo_tail:
                                log.info(f"[{self.sid}] echo-tail drop (segment): {text!r}")
                                continue

                            # Text-layer echo gate: drop segments that are mostly AI's own words
                            if self._text_echo_filter.is_echo_segment(text):
                                log.info(f"[{self.sid}] text-echo drop segment: {text!r}")
                                continue

                            seg_turn_id = str(uuid.uuid4())
                            self._lat.new_turn(seg_turn_id, text)
                            self._lat.on_stt_segment()
                            log.info(f"[{self.sid}] STT segment [{self.state.name}] ({word_count}w): {text!r}")
                            await self._jsend(ev)

                            if self.state in (State.SPEAKING, State.THINKING):
                                if word_count >= BARGE_IN_MIN_WORDS:
                                    if self._barge_in:
                                        _drain_q(self._query_q)
                                        await self._query_q.put((seg_turn_id, text))
                                    else:
                                        await self._do_barge_in_immediate()
                                        _drain_q(self._query_q)
                                        await self._query_q.put((seg_turn_id, text))
                            else:
                                await self._query_q.put((seg_turn_id, text))

                        elif kind == "partial":
                            await self._jsend(ev)

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

    # ─── CAG loop — persistent WebSocket ─────────────────────────────────────

    async def _cag_loop(self):
        """
        Opens a persistent WebSocket to the CAG microservice.
        Each query is sent as a JSON frame; tokens arrive as individual frames.
        This eliminates HTTP/SSE framing overhead for ultra-low first-token latency.

        Protocol (CAG WS server must implement):
          → {"type": "query", "turn_id": "...", "message": "...", "reset": bool}
          ← {"type": "token",    "token": "...", "turn_id": "..."}
          ← {"type": "done",     "turn_id": "..."}
          ← {"type": "error",    "detail": "...", "turn_id": "..."}
          ← {"type": "turn_id",  "turn_id": "..."}   (first frame, confirms routing)

        Falls back to HTTP SSE if CAG WebSocket is unavailable.
        """
        self._cag_turn_count = 0
        retries = 0

        while self._running and retries < CAG_MAX_RETRIES:
            try:
                cag_ws = await _ws_connect(
                    CAG_WS_URL, max_retries=CAG_MAX_RETRIES,
                    label=f"[{self.sid}] CAG",
                    max_size=4 * 1024 * 1024,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                )
                self._cag_ws = cag_ws
                retries      = 0
                log.info(f"[{self.sid}] CAG WebSocket connected ✓")

                while self._running:
                    query = await self._query_q.get()
                    self._last_query_time = time.monotonic()

                    if isinstance(query, tuple):
                        turn_id, query_text = query
                    else:
                        turn_id    = str(uuid.uuid4())
                        query_text = query

                    self._cag_turn_count += 1
                    self._barge_in        = False
                    self.state            = State.THINKING

                    self._lat.new_turn(turn_id, query_text)
                    self._lat.on_query_sent()

                    log.info(f"[{self.sid}] CAG WS query: {query_text!r}")
                    await self._jsend({"type": "thinking", "turn_id": turn_id})

                    # Send query frame to CAG
                    try:
                        await cag_ws.send(json.dumps({
                            "type":    "query",
                            "turn_id": turn_id,
                            "message": query_text,
                            "reset":   self._cag_turn_count == 1,
                        }))
                    except Exception as e:
                        log.error(f"[{self.sid}] CAG send error: {e}")
                        raise  # triggers reconnect

                    await self._process_cag_stream_ws(cag_ws, turn_id)

            except asyncio.CancelledError:
                return
            except Exception as e:
                retries += 1
                self._cag_ws = None
                log.warning(
                    f"[{self.sid}] CAG WS disconnected ({e}), "
                    f"falling back to HTTP SSE, retry {retries}/{CAG_MAX_RETRIES}"
                )
                if retries < CAG_MAX_RETRIES:
                    await asyncio.sleep(min(2 ** retries, 10))
                    # Try HTTP SSE fallback for the pending query
                    await self._cag_loop_http_fallback()
                else:
                    log.error(f"[{self.sid}] CAG max retries reached — dropping queries")

    async def _process_cag_stream_ws(self, cag_ws, turn_id: str):
        """
        Receive token frames from CAG WebSocket for one turn.
        Dispatches to TTS accumulator immediately on each token.
        """
        acc              = TonalAccumulator()
        acc.reset()
        full_reply_parts: list[str] = []
        interrupted      = False
        stream_confirmed = False

        try:
            async for raw_frame in cag_ws:
                if self._barge_in:
                    interrupted = True
                    log.info(f"[{self.sid}] CAG WS aborted (barge-in)")
                    break

                try:
                    frame = json.loads(raw_frame) if isinstance(raw_frame, (str, bytes)) else {}
                except Exception:
                    continue

                ftype = frame.get("type", "")

                # First frame: turn_id confirmation
                if ftype == "turn_id":
                    server_tid = frame.get("turn_id", "")
                    if server_tid != turn_id:
                        log.warning(f"[{self.sid}] stale CAG stream — discarding")
                        interrupted = True
                        break
                    stream_confirmed = True
                    continue

                if ftype == "done":
                    break

                if ftype == "error":
                    await self._jsend({"type": "error", "detail": frame.get("detail", "CAG error")})
                    interrupted = True
                    break

                if ftype == "timeout":
                    await self._jsend({"type": "error", "detail": "CAG timeout"})
                    interrupted = True
                    break

                if ftype != "token" or not stream_confirmed:
                    continue

                token = frame.get("token", "")
                if not token:
                    continue

                self._lat.on_first_token()
                self._lat.on_token()
                full_reply_parts.append(token)
                await self._jsend({"type": "ai_token", "token": token})

                # Dispatch to TTS immediately on each accumulated sentence chunk
                for tc in acc.feed(token):
                    if self._barge_in:
                        interrupted = True
                        break
                    log.info(f"[{self.sid}] TTS← [{tc.tone}] {tc.text!r}")
                    await self._jsend({"type": "ai_sentence", "text": tc.text, "tone": tc.tone})
                    self._text_echo_filter.feed_ai_text(tc.text)  # register AI words for echo detection
                    self.state = State.SPEAKING
                    self._lat.on_tts_chunk_sent(tc.text)
                    await self._tts_q.put(tc)

                if interrupted:
                    break

        except Exception as e:
            log.error(f"[{self.sid}] CAG WS stream error: {e}")
            interrupted = True
            await self._jsend({"type": "error", "detail": str(e)})
            raise  # triggers reconnect at caller

        finally:
            if not interrupted and not self._barge_in:
                # Flush tail
                tail = acc.flush()
                if tail:
                    log.info(f"[{self.sid}] TTS← tail [{tail.tone}]: {tail.text!r}")
                    await self._jsend({"type": "ai_sentence", "text": tail.text, "tone": tail.tone})
                    self._text_echo_filter.feed_ai_text(tail.text)  # register AI words for echo detection
                    self.state = State.SPEAKING
                    self._lat.on_tts_chunk_sent(tail.text)
                    await self._tts_q.put(tail)

                await self._tts_q.put(self._TURN_END)

                # Send assistant turn text to STT for conversation context
                full_text = " ".join(full_reply_parts).strip()
                if full_text and self._stt_ws:
                    ctrl = json.dumps({"type": "assistant_turn", "text": full_text}).encode()
                    try:
                        await self._stt_ws.send(b'\x02' + ctrl)
                    except Exception:
                        pass

    async def _cag_loop_http_fallback(self):
        """
        HTTP SSE fallback for CAG when the WebSocket is unavailable.
        Processes a single query from the queue if one is pending.
        """
        try:
            query = self._query_q.get_nowait()
        except asyncio.QueueEmpty:
            return

        if isinstance(query, tuple):
            turn_id, query_text = query
        else:
            turn_id    = str(uuid.uuid4())
            query_text = query

        self._cag_turn_count += 1
        self._barge_in        = False
        self.state            = State.THINKING
        self._lat.new_turn(turn_id, query_text)
        self._lat.on_query_sent()

        log.info(f"[{self.sid}] CAG HTTP fallback: {query_text!r}")
        await self._jsend({"type": "thinking", "turn_id": turn_id})

        acc              = TonalAccumulator()
        acc.reset()
        full_reply_parts: list[str] = []
        interrupted      = False
        stream_confirmed = False

        async with httpx.AsyncClient(
            base_url=CAG_HTTP_URL,
            timeout=httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0),
        ) as http:
            try:
                async with http.stream(
                    "POST", "/chat/stream",
                    json={
                        "message":       query_text,
                        "reset_session": self._cag_turn_count == 1,
                        "turn_id":       turn_id,
                    },
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if self._barge_in:
                            interrupted = True
                            break
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()

                        if data.startswith("[TURN_ID]"):
                            server_tid = data[9:].strip()
                            if server_tid != turn_id:
                                interrupted = True
                                break
                            stream_confirmed = True
                            continue
                        if data == "[DONE]":
                            break
                        if data in ("[TIMEOUT]", "") or data.startswith("[ERROR]"):
                            interrupted = True
                            break
                        if not stream_confirmed:
                            continue

                        self._lat.on_first_token()
                        self._lat.on_token()
                        full_reply_parts.append(data)
                        await self._jsend({"type": "ai_token", "token": data})

                        for tc in acc.feed(data):
                            if self._barge_in:
                                interrupted = True
                                break
                            await self._jsend({"type": "ai_sentence", "text": tc.text, "tone": tc.tone})
                            self.state = State.SPEAKING
                            self._lat.on_tts_chunk_sent(tc.text)
                            await self._tts_q.put(tc)

                        if interrupted:
                            break

            except Exception as e:
                log.error(f"[{self.sid}] CAG HTTP fallback error: {e}")
                interrupted = True

            finally:
                if not interrupted and not self._barge_in:
                    tail = acc.flush()
                    if tail:
                        await self._jsend({"type": "ai_sentence", "text": tail.text, "tone": tail.tone})
                        self.state = State.SPEAKING
                        self._lat.on_tts_chunk_sent(tail.text)
                        await self._tts_q.put(tail)
                    await self._tts_q.put(self._TURN_END)

    # ─── Synth worker ─────────────────────────────────────────────────────────

    async def _synth_worker(self):
        """
        Picks TTS sentence chunks from tts_q, synthesizes them in parallel
        (up to TTS_MAX_PARALLEL), puts (order_index, frames) into pcm_q.
        """
        order_index    = 0
        pending_tasks: list[asyncio.Task] = []

        while self._running:
            item = await self._tts_q.get()

            if item is self._INTERRUPT:
                self._barge_in = True
                log.info(f"[{self.sid}] synth INTERRUPT")
                for t in pending_tasks:
                    t.cancel()
                pending_tasks.clear()
                order_index = 0
                await self._pcm_q.put(self._INTERRUPT)
                continue

            if item is self._TURN_END:
                total = order_index
                log.info(f"[{self.sid}] synth TURN_END ({total} chunks)")
                await self._pcm_q.put(("TURN_END", total))
                pending_tasks = [t for t in pending_tasks if not t.done()]
                order_index   = 0
                continue

            idx         = order_index
            order_index += 1
            task = asyncio.create_task(
                self._synth_one(item, idx),
                name=f"synth_{self.sid}_{idx}"
            )
            pending_tasks.append(task)
            pending_tasks = [t for t in pending_tasks if not t.done()]

    async def _synth_one(self, tc: TonalChunk, idx: int):
        """Synthesize one TTS chunk, put (idx, frames) into pcm_q."""
        text = tc.text.strip()
        if not text:
            await self._pcm_q.put((idx, []))
            return

        async with self._tts_sem:
            if self._barge_in:
                await self._pcm_q.put((idx, []))
                return

            frames:  list[bytes] = []
            tts_ws = None
            retries = 0
            t_start = time.monotonic()

            while retries < TTS_MAX_RETRIES:
                try:
                    tts_ws = await _ws_connect(
                        TTS_WS_URL, max_retries=TTS_MAX_RETRIES,
                        label=f"[{self.sid}] TTS-{idx}",
                        max_size=10 * 1024 * 1024,
                        ping_interval=None, ping_timeout=None,
                    )
                    await tts_ws.send(json.dumps({
                        "text":       text,
                        "language":   TTS_LANGUAGE,
                        "speaker":    TTS_SPEAKER,
                        "chunk_tone": tc.tone,
                    }))

                    wav_skipped    = False
                    first_audio_ok = False

                    while True:
                        if self._barge_in:
                            frames = []
                            break
                        try:
                            frame = await asyncio.wait_for(tts_ws.recv(), timeout=30.0)
                        except asyncio.TimeoutError:
                            log.warning(f"[{self.sid}] synth_one[{idx}] TTS timeout")
                            break
                        except Exception as e:
                            log.error(f"[{self.sid}] synth_one[{idx}] recv error: {e}")
                            raise

                        if isinstance(frame, bytes):
                            if frame == b"":
                                break
                            if not wav_skipped:
                                wav_skipped = True
                                continue
                            if frame:
                                if not first_audio_ok:
                                    first_audio_ok = True
                                    if idx == 0:
                                        self._lat.on_tts_audio_start()
                                frames.append(frame)
                                self._echo_gate.feed_tts(frame)   # keep ring buffer current

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
                                log.warning(f"[{self.sid}] TTS error msg: {msg['error']}")
                    break

                except Exception as e:
                    retries += 1
                    log.warning(f"[{self.sid}] synth_one[{idx}] error ({e}), retry {retries}")
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

        synth_ms = (time.monotonic() - t_start) * 1000
        log.debug(f"[{self.sid}] synth_one[{idx}] done {synth_ms:.0f}ms frames={len(frames)}")
        await self._pcm_q.put((idx, frames))

    # ─── Play worker ──────────────────────────────────────────────────────────

    async def _play_worker(self):
        """
        Receives (order_index, frames) from pcm_q, reorders them, and streams
        PCM bytes to the client in correct sentence order.
        On INTERRUPT: drains and resets.
        """
        next_expected   = 0
        reorder_buf:    dict[int, list[bytes]] = {}
        chunk_count     = 0
        total_expected  = -1
        chunks_received = 0

        async def _flush_ordered():
            nonlocal next_expected, chunk_count
            while next_expected in reorder_buf:
                frames = reorder_buf.pop(next_expected)
                for f in frames:
                    await self._bsend(f)
                    self._echo_gate.feed_tts(f)   # arm gate with audio playing RIGHT NOW
                chunk_count   += 1
                next_expected += 1

        async def _finalize_turn():
            nonlocal reorder_buf, next_expected, chunk_count
            nonlocal total_expected, chunks_received
            await _flush_ordered()
            await asyncio.sleep(ECHO_TAIL_GUARD_S)
            self._tts_stopped_at = time.monotonic()
            self._echo_gate.tts_stopped()   # start tail guard
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

        while self._running:
            item = await self._pcm_q.get()

            if item is self._INTERRUPT:
                log.info(f"[{self.sid}] play INTERRUPT")
                reorder_buf.clear()
                next_expected   = 0
                chunk_count     = 0
                total_expected  = -1
                chunks_received = 0
                self.state      = State.IDLE
                self._tts_stopped_at = time.monotonic()
                self._echo_gate.tts_stopped()   # start tail guard
                report = self._lat.complete_turn()
                if report:
                    report["barge_in"] = True
                    await self._jsend({"type": "latency", "stage": "turn_interrupted", **report})
                continue

            if isinstance(item, tuple) and item[0] == "TURN_END":
                total_expected = item[1]
                log.info(f"[{self.sid}] play TURN_END — expecting {total_expected} chunks")
                await _flush_ordered()
                if total_expected == 0 or chunks_received >= total_expected:
                    await _finalize_turn()
                continue

            order_idx, frames = item

            if self._barge_in:
                chunks_received += 1
                continue

            chunks_received += 1
            self.state       = State.SPEAKING
            reorder_buf[order_idx] = frames
            await _flush_ordered()

            if total_expected >= 0 and chunks_received >= total_expected:
                await _flush_ordered()
                if not reorder_buf:
                    await _finalize_turn()


# ─── App startup ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _gateway_startup():
    """Load the TTS voice fingerprint from disk at startup (once for all sessions)."""
    fp = _load_shared_fingerprint()
    if fp is not None:
        log.info(
            "[startup] Voice fingerprint loaded — all sessions will use it immediately. "
            "Gate armed: no per-session TTS synthesis needed."
        )
    else:
        log.warning(
            "[startup] No fingerprint file at %s — sessions will live-enroll from TTS. "
            "Start tts_microservice.py first and wait for enrollment to complete.",
            TTS_FINGERPRINT_PATH,
        )


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session  = GatewaySession(ws)
    pipeline = asyncio.create_task(session.run())
    log.info(f"[{session.sid}] client connected")

    if TEST_MODE:
        log.info(f"[{session.sid}] TEST_MODE — greeting skipped")
        await ws.send_json({"type": "ready", "message": "TEST MODE — ready"})

    try:
        async for raw in ws.iter_bytes():
            if not raw:
                continue
            ftype   = raw[0]
            payload = raw[1:]

            if ftype == 0x01:                           # audio frame
                session.push_audio(raw)
                session._last_pong_time = time.monotonic()

            elif ftype == 0x02:                         # control frame
                try:
                    ctrl  = json.loads(payload)
                except Exception:
                    continue
                mtype = ctrl.get("type")

                if mtype == "ping":
                    await ws.send_json({"type": "pong"})

                elif mtype == "pong":
                    session._last_pong_time = time.monotonic()

                elif mtype == "inject_query":
                    text = ctrl.get("text", "").strip()
                    if text:
                        turn_id = str(uuid.uuid4())
                        session._lat.new_turn(turn_id, text)
                        session._lat.on_stt_segment()
                        await ws.send_json({"type": "segment", "text": text})
                        await session._query_q.put((turn_id, text))
                        log.info(f"[{session.sid}] inject_query: {text!r}")

                elif mtype == "get_stats":
                    await ws.send_json({
                        "type":  "stats",
                        "sid":   session.sid,
                        "state": session.state.name,
                    })

                elif mtype == "get_latency":
                    await ws.send_json({
                        "type":    "latency_snapshot",
                        "turns":   session._lat.all_reports(),
                        "summary": session._lat.session_summary(),
                    })

                elif mtype == "reset_context" and session._stt_ws:
                    try:
                        await session._stt_ws.send(b'\x02' + payload)
                    except Exception:
                        pass

    except WebSocketDisconnect:
        log.info(f"[{session.sid}] disconnected")
    except Exception as e:
        log.error(f"[{session.sid}] ws error: {e}")
    finally:
        await session.stop()
        pipeline.cancel()


# ─── REST ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "14.0.0"}


@app.get("/latency/session/{sid}")
def get_session_latency(sid: str):
    data = _session_latency_store.get(sid)
    if not data:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return data


@app.get("/latency/sessions")
def list_sessions():
    return {"sessions": list(_session_latency_store.keys())}


if __name__ == "__main__":
    uvicorn.run(
        "gateway:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        workers=1,
        log_level="info",
        reload=False,
    )