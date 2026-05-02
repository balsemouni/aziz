# gateway.py — Parallel Streaming Gateway  v18.0
"""
gateway.py — Parallel Streaming Gateway  v18.0
════════════════════════════════════════════════════════

KEY CHANGES v18.0 — ULTRA-LOW LATENCY PARALLEL STREAMING
────────────────────────────────────────────────────────────────
  • IMMEDIATE sentence detection - send to TTS on punctuation
  • No buffering - stream each sentence as soon as complete
  • Parallel TTS instances for overlapping generation
  • Optimized token streaming with minimal overhead
  • Zero-wait first audio playback
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
from typing import Optional, List, Dict, Set
from collections import deque

import numpy as np

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# ─── Configuration ────────────────────────────────────────────────────────────

STT_WS_URL = os.getenv("STT_WS_URL", "ws://127.0.0.1:8001/stream/mux")
CAG_WS_URL = os.getenv("CAG_WS_URL", "ws://127.0.0.1:8000/chat/ws")
CAG_HTTP_URL = os.getenv("CAG_HTTP_URL", "http://127.0.0.1:8000")
TTS_WS_URL = os.getenv("TTS_WS_URL", "ws://127.0.0.1:8765/ws/tts")
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8090"))
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "tara")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en")
DEFAULT_LANG = os.getenv("GATEWAY_DEFAULT_LANG", "auto")  # forwarded to STT

BARGE_IN_MIN_WORDS = int(os.getenv("BARGE_IN_MIN_WORDS", "1"))
BARGE_IN_COOLDOWN_S = float(os.getenv("BARGE_IN_COOLDOWN_S", "0.05"))  # STT handles real debouncing
ECHO_TAIL_GUARD_S = float(os.getenv("ECHO_TAIL_GUARD_S", "0.3"))
STT_SILENCE_MS = float(os.getenv("STT_SILENCE_MS", "500"))

# Set to "0" if your client uses the streamed PCM as the only audio source
# (recommended). When "1" the gateway also tags ai_sentence/ai_token events
# as synthesizable — only enable that for text-only UI fallbacks.
EMIT_TEXT_FOR_CLIENT_TTS = os.getenv("EMIT_TEXT_FOR_CLIENT_TTS", "0").strip() in ("1", "true", "yes")

TTS_MAX_RETRIES = int(os.getenv("TTS_MAX_RETRIES", "2"))
TTS_POOL_SIZE = int(os.getenv("TTS_POOL_SIZE", "3"))  # Increased for parallel TTS
STT_MAX_RETRIES = int(os.getenv("STT_MAX_RETRIES", "3"))
CAG_MAX_RETRIES = int(os.getenv("CAG_MAX_RETRIES", "2"))

HALLUC_WINDOW = int(os.getenv("HALLUC_WINDOW", "8"))
HALLUC_THRESHOLD = int(os.getenv("HALLUC_THRESHOLD", "4"))
IDLE_TIMEOUT_S = float(os.getenv("IDLE_TIMEOUT_S", "60.0"))
HEARTBEAT_S = float(os.getenv("HEARTBEAT_S", "25.0"))

TTS_GREETING = os.getenv("TTS_GREETING", "")
TEST_MODE = os.getenv("TEST_MODE", "0").strip() in ("1", "true", "yes")

# Parallel streaming settings
MAX_CONCURRENT_TTS = int(os.getenv("MAX_CONCURRENT_TTS", "3"))  # Max parallel TTS tasks
MIN_SENTENCE_LENGTH = int(os.getenv("MIN_SENTENCE_LENGTH", "2"))  # Min words for TTS
SENTENCE_TIMEOUT_S = float(os.getenv("SENTENCE_TIMEOUT_S", "0.3"))  # Force flush after timeout
MIN_CLAUSE_WORDS   = int(os.getenv("MIN_CLAUSE_WORDS", "10"))       # Flush on , ; : after this many words
TTS_PCM_SAMPLE_RATE = int(os.getenv("TTS_PCM_SAMPLE_RATE", "22050"))  # Piper output rate (int16 = 2 bytes/sample)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("gateway")
lat_log = logging.getLogger("gateway.latency")

app = FastAPI(title="Voice Gateway", version="18.1.0")


def _smart_join(parts: list) -> str:
    """Join LLM token stream. Modern BPE tokenizers (Qwen, Llama, GPT-style)
    emit a *leading space* on every genuine word-start token. Tokens without
    a leading space are subword continuations and MUST be concatenated with
    no space — otherwise we get artefacts like 'abbrev iations', 'specifi c',
    'm aking'. Concatenate verbatim and trust the tokenizer's own spacing.
    """
    if not parts:
        return ""
    return "".join(p for p in parts if p).strip()
_session_latency_store: dict[str, dict] = {}


# ─── Enums ────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE = auto()
    THINKING = auto()
    SPEAKING = auto()


# ─── Ultra-fast sentence splitting for parallel streaming ───────────────────

# Sentence-ending punctuation, including French «…»/guillemets-aware closers.
# NBSP (\u00A0) and NARROW NBSP (\u202F) precede ?!:; in French typography —
# we treat them like ordinary spaces.
_SENTENCE_END_CHARS = ".!?…»"
_FR_NBSP_CHARS      = "\u00A0\u202F"
# Lower-case abbreviation set (English + a few French ones that end in '.').
_ABBREV = {
    "dr", "mr", "ms", "mrs", "prof", "rev", "st", "ave",
    "mme", "mlle", "etc", "p.s", "n.b", "cf",
}

def is_sentence_boundary(text: str, idx: int) -> bool:
    """Check if position idx is a sentence boundary (en + fr aware)."""
    if idx >= len(text) - 1:
        return False
    char = text[idx]

    if char not in _SENTENCE_END_CHARS:
        return False

    # Look-ahead: must be followed by whitespace (incl. NBSP) or end of string.
    nxt = text[idx + 1] if idx + 1 < len(text) else ''
    if nxt and not (nxt.isspace() or nxt in _FR_NBSP_CHARS or nxt in '«"\''):
        return False

    # Don't split on common abbreviations like "Dr.", "Mme.", "etc."
    if char == '.':
        # Walk back to grab the alpha run before the dot
        j = idx - 1
        while j >= 0 and (text[j].isalpha() or text[j] == '.'):
            j -= 1
        token = text[j + 1: idx].lower().rstrip('.')
        if token in _ABBREV:
            return False
    return True

def split_into_sentences_immediate(text: str) -> List[str]:
    """Split text into sentences IMMEDIATELY on boundaries (en + fr)."""
    if not text:
        return []

    sentences = []
    current = []

    for i, char in enumerate(text):
        current.append(char)

        if is_sentence_boundary(text, i):
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
                current = []

    # Don't return incomplete sentence
    return sentences


# ─── Latency tracking ─────────────────────────────────────────────────────────

def _r(v: Optional[float]) -> Optional[float]:
    return round(v, 1) if v is not None else None


def _latency_color(ms: float) -> str:
    if ms < 150:
        return "🟢"
    if ms < 300:
        return "🟡"
    if ms < 600:
        return "🟠"
    return "🔴"


@dataclass
class TurnLatency:
    turn_id: str = ""
    query_text: str = ""
    barge_in: bool = False
    
    stt_first_word_ts: Optional[float] = None
    stt_segment_ts: Optional[float] = None
    stt_latency_ms: Optional[float] = None
    
    query_sent_ts: Optional[float] = None
    first_token_ts: Optional[float] = None
    first_sentence_ts: Optional[float] = None
    first_audio_ts: Optional[float] = None
    cag_first_token_ms: Optional[float] = None
    cag_first_sentence_ms: Optional[float] = None
    tts_first_audio_ms: Optional[float] = None
    total_tokens: int = 0
    sentences: int = 0
    
    e2e_ms: Optional[float] = None
    
    def finalize(self):
        if self.stt_first_word_ts and self.stt_segment_ts:
            self.stt_latency_ms = (self.stt_segment_ts - self.stt_first_word_ts) * 1000
        if self.query_sent_ts and self.first_token_ts:
            self.cag_first_token_ms = (self.first_token_ts - self.query_sent_ts) * 1000
        if self.query_sent_ts and self.first_sentence_ts:
            self.cag_first_sentence_ms = (self.first_sentence_ts - self.query_sent_ts) * 1000
        if self.first_sentence_ts and self.first_audio_ts:
            self.tts_first_audio_ms = (self.first_audio_ts - self.first_sentence_ts) * 1000
        if self.stt_first_word_ts and self.first_audio_ts:
            self.e2e_ms = (self.first_audio_ts - self.stt_first_word_ts) * 1000
    
    def to_report(self) -> dict:
        self.finalize()
        return {
            "turn_id": self.turn_id,
            "query": self.query_text[:80],
            "barge_in": self.barge_in,
            "stt_latency_ms": _r(self.stt_latency_ms),
            "cag_first_token_ms": _r(self.cag_first_token_ms),
            "cag_first_sentence_ms": _r(self.cag_first_sentence_ms),
            "tts_first_audio_ms": _r(self.tts_first_audio_ms),
            "e2e_ms": _r(self.e2e_ms),
            "total_tokens": self.total_tokens,
            "sentences": self.sentences,
        }


class LatencyTracker:
    def __init__(self, sid: str):
        self.sid = sid
        self.current: Optional[TurnLatency] = None
        self.history: list[TurnLatency] = []
    
    def new_turn(self, turn_id: str, query: str):
        self.current = TurnLatency(turn_id=turn_id, query_text=query)
    
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
    
    def on_first_sentence(self):
        now = time.monotonic()
        if self.current and not self.current.first_sentence_ts:
            self.current.first_sentence_ts = now
            if self.current.query_sent_ts:
                ms = (now - self.current.query_sent_ts) * 1000
                self.current.cag_first_sentence_ms = ms
                lat_log.info(f"[{self.sid}] First sentence: {ms:.0f}ms  {_latency_color(ms)}")
    
    def on_token(self):
        if self.current:
            self.current.total_tokens += 1
    
    def on_sentence(self):
        if self.current:
            self.current.sentences += 1
    
    def on_first_audio(self):
        now = time.monotonic()
        if self.current and not self.current.first_audio_ts:
            self.current.first_audio_ts = now
            if self.current.first_sentence_ts:
                ms = (now - self.current.first_sentence_ts) * 1000
                self.current.tts_first_audio_ms = ms
                lat_log.info(f"[{self.sid}] First audio: {ms:.0f}ms after sentence  {_latency_color(ms)}")
    
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
                f"CAG token {_r(self.current.cag_first_token_ms)}ms | "
                f"CAG sent {_r(self.current.cag_first_sentence_ms)}ms | "
                f"TTS {_r(self.current.tts_first_audio_ms)}ms]"
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
            sv = sorted(vals)
            p95 = sv[max(0, int(len(sv) * 0.95) - 1)]
            return {
                "min": _r(min(vals)), "max": _r(max(vals)),
                "avg": _r(sum(vals) / len(vals)), "p95": _r(p95),
            }
        
        stt_lats = [t.stt_latency_ms for t in turns if t.stt_latency_ms is not None]
        cag_lats = [t.cag_first_token_ms for t in turns if t.cag_first_token_ms is not None]
        sent_lats = [t.cag_first_sentence_ms for t in turns if t.cag_first_sentence_ms is not None]
        tts_lats = [t.tts_first_audio_ms for t in turns if t.tts_first_audio_ms is not None]
        e2e_lats = [t.e2e_ms for t in turns]
        barge_ins = sum(1 for t in turns if t.barge_in)
        
        return {
            "sid": self.sid,
            "turns": len(turns),
            "barge_ins": barge_ins,
            "stt": _stats(stt_lats),
            "cag_first_token": _stats(cag_lats),
            "cag_first_sentence": _stats(sent_lats),
            "tts": _stats(tts_lats),
            "e2e": _stats(e2e_lats),
        }


# ─── Repetition guard ─────────────────────────────────────────────────────────

class RepetitionGuard:
    _STOP = frozenset({
        "a", "an", "the", "i", "me", "my", "we", "our", "you", "your",
        "he", "she", "it", "they", "his", "her", "its", "their",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "to", "of", "in", "on", "at", "by", "for", "with",
        "and", "or", "but", "not", "no", "so", "if", "as",
        "that", "this", "these", "those", "what", "which", "who",
        "how", "when", "where", "why", "up", "out", "about", "into",
        "from", "than", "then", "can", "will", "would", "could",
        "should", "may", "might", "just", "also", "very", "more", "some", "any",
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

_ECHO_STOP = frozenset({
    "a", "an", "the", "i", "me", "my", "we", "our", "you", "your",
    "is", "are", "was", "were", "to", "of", "in", "on", "at", "by",
    "and", "or", "but", "not", "so", "it", "its", "be", "do", "did",
})

AI_TEXT_ECHO_WINDOW_S = 3.0
AI_TEXT_ECHO_RATIO = 0.85
AI_TEXT_ECHO_MIN_WORDS = 4


class AITextEchoFilter:
    def __init__(self):
        self._ai_words: list[tuple[float, str]] = []
    
    def feed_ai_text(self, text: str):
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
        w = word.lower().strip(".,!?;:\"'")
        if not w or w in _ECHO_STOP:
            return False
        return w in self._recent_ai_set()
    
    def is_echo_segment(self, text: str) -> bool:
        if not text:
            return False
        words = [w.lower().strip(".,!?;:\"'") for w in text.split()]
        content = [w for w in words if w and w not in _ECHO_STOP]
        if len(content) < AI_TEXT_ECHO_MIN_WORDS:
            return False
        # Only flag as echo if the segment is short (likely captured TTS playback)
        # Long user utterances are almost certainly real speech, not echo
        if len(content) > 8:
            return False
        ai_set = self._recent_ai_set()
        overlap = sum(1 for w in content if w in ai_set)
        ratio = overlap / len(content)
        if ratio >= AI_TEXT_ECHO_RATIO:
            return True
        return False
    
    def reset(self):
        self._ai_words.clear()


# ─── Timing-based Echo Gate ────────────────────────────────────────────────────

class TimingEchoGate:
    def __init__(self):
        self._tts_active = False
        self._tts_stopped_at = 0.0
        self.frames_checked = 0
        self.frames_dropped = 0
    
    def feed_tts(self, pcm_bytes: bytes):
        self._tts_active = True
    
    def tts_stopped(self):
        self._tts_active = False
        self._tts_stopped_at = time.monotonic()
    
    def _is_armed(self) -> bool:
        if self._tts_active:
            return True
        return (time.monotonic() - self._tts_stopped_at) < ECHO_TAIL_GUARD_S
    
    def check(self, mic_pcm_bytes: bytes, ai_speaking: bool = False) -> bool:
        self.frames_checked += 1
        if not mic_pcm_bytes:
            return False
        if self._is_armed():
            self.frames_dropped += 1
            return True
        return False
    
    def reset(self):
        self._tts_active = False
        self._tts_stopped_at = 0.0
        self.frames_checked = 0
        self.frames_dropped = 0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _drain_q(q: asyncio.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break


async def _ws_connect(url: str, max_retries: int, label: str, **kwargs):
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
    Ultra-low latency parallel streaming gateway.
    
    v18.0: Immediate sentence detection and parallel TTS
      1. CAG streams tokens -> detect sentence boundaries in real-time
      2. IMMEDIATELY send complete sentences to TTS (no waiting)
      3. Multiple TTS instances run in parallel
      4. Audio streams with zero gaps
    """
    
    _INTERRUPT = object()
    _TURN_END = object()
    
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.sid = str(uuid.uuid4())[:8]
        self.state = State.IDLE
        self._running = True
        # Per-session language (en|fr|auto) negotiated at /ws connect.
        # Forwarded to STT via ?lang= and to TTS via the request payload.
        self.lang: str = DEFAULT_LANG
        
        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._query_q: asyncio.Queue = asyncio.Queue()
        self._tts_pcm_q: asyncio.Queue = asyncio.Queue()
        
        self._stt_ws: Optional[object] = None
        self._cag_ws: Optional[object] = None
        self._tts_ws: Optional[object] = None
        
        self._tts_pool: asyncio.Queue = asyncio.Queue()
        
        self._barge_in = False
        self._barge_in_until = 0.0
        self._tts_stopped_at = 0.0
        self._last_pong_time = time.monotonic()
        self._last_query_time = time.monotonic()
        
        self._stt_ready = asyncio.Event()
        self._lat = LatencyTracker(self.sid)
        self._echo_gate = TimingEchoGate()
        self._text_echo_filter = AITextEchoFilter()
        
        # Parallel streaming buffers
        self._active_tts_tasks: List[asyncio.Task] = []
        self._tts_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS)
        self._sentence_queue: asyncio.Queue = asyncio.Queue()
        self._current_turn_id: Optional[str] = None  # active CAG turn; TTS tasks abort if stale

        # Control-frame queue for sending AI-state signals to STT
        self._stt_ctrl_q: asyncio.Queue = asyncio.Queue(maxsize=20)
        # TTS reference audio queue — fed into STT's acoustic fingerprint gate
        self._stt_ref_q: asyncio.Queue = asyncio.Queue(maxsize=200)

        # Playback duration tracking: bytes received from TTS server this turn
        self._tts_pcm_bytes_this_turn: int = 0
        # Task that re-opens the mic after audio finishes playing
        self._open_mic_task: Optional[asyncio.Task] = None
        # Set True after AI playback finishes so the next iteration of _recv
        # discards any STT words that accumulated during AI speech (echo
        # leakage that wasn't fully suppressed by AEC). The _recv loop
        # consumes & resets this flag.
        self._stt_buf_invalidate: bool = False

        # ── A2 dual-trigger: predicted barge-in soft-mute ────────────────
        # Set True the instant STT emits {"type":"barge_in_predicted"}.
        # While True, _bsend drops outgoing TTS PCM frames so the user
        # immediately stops hearing the AI talk over them.  Cleared by:
        #   • a confirmed barge_in (which fully tears down TTS+CAG), or
        #   • _do_barge_in_immediate (defensive), or
        #   • the predict_unmute watchdog if no real word follows.
        self._audio_muted: bool = False
        self._predict_unmute_task: Optional[asyncio.Task] = None

        # ── A4: multi-interrupt safety ───────────────────────────────────
        # Serialises overlapping cancel/start transitions so a barge-in
        # arriving DURING another barge-in's cleanup can't race with it.
        # Combined with _current_turn_id invalidation this guarantees the
        # last user utterance is always the one answered.
        self._turn_lock: asyncio.Lock = asyncio.Lock()
        self._turn_seq: int = 0  # monotonic counter for diagnostics

        log.info(f"[{self.sid}] session created")
    
    # ── Safe send helpers ─────────────────────────────────────────────────────
    
    async def _jsend(self, obj: dict):
        try:
            await self.ws.send_json(obj)
        except Exception:
            pass
    
    async def _bsend(self, data: bytes):
        # A2: predicted barge-in mutes outgoing audio without tearing down
        # the TTS/CAG stack.  If the prediction is confirmed by a real
        # word, _do_barge_in_immediate runs next and cleans everything up;
        # otherwise the predict_unmute watchdog clears the flag and the
        # remaining audio continues flowing.
        if self._audio_muted:
            return
        try:
            await self.ws.send_bytes(data)
        except Exception:
            pass

    async def _broadcast_tts_cancel(self):
        """
        Send a {"type":"cancel"} JSON frame to every TTS WebSocket we own
        (single + pool) so any in-flight synth aborts immediately at the
        next chunk boundary.  Connections that respond cleanly stay in the
        pool; broken ones are closed and replaced by the pool manager.
        """
        cancel_frame = json.dumps({"type": "cancel"})
        targets = []
        if self._tts_ws is not None:
            targets.append(self._tts_ws)
        # Drain pool — any healthy connections get the cancel frame and are
        # then put back; broken ones are dropped and pool manager refills.
        drained = []
        while not self._tts_pool.empty():
            try:
                drained.append(self._tts_pool.get_nowait())
            except Exception:
                break
        targets.extend(drained)
        for ws in targets:
            try:
                await ws.send(cancel_frame)
            except Exception:
                try:
                    await ws.close()
                except Exception:
                    pass
                continue
        # Re-queue the pool connections we cancelled (still usable for next req)
        for ws in drained:
            try:
                await self._tts_pool.put(ws)
            except Exception:
                pass
    
    # ── Entry ─────────────────────────────────────────────────────────────────
    
    async def run(self):
        tasks = [
            asyncio.create_task(self._stt_loop(), name="stt"),
            asyncio.create_task(self._cag_loop(), name="cag"),
            asyncio.create_task(self._tts_pool_manager(), name="tts_pool"),
            asyncio.create_task(self._heartbeat(), name="heartbeat"),
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
        
        # Cancel all active TTS tasks
        for task in self._active_tts_tasks:
            if not task.done():
                task.cancel()
        self._active_tts_tasks.clear()
        
        while not self._tts_pool.empty():
            try:
                ws = self._tts_pool.get_nowait()
                try:
                    await ws.close()
                except Exception:
                    pass
            except asyncio.QueueEmpty:
                break
        
        summary = self._lat.session_summary()
        _session_latency_store[self.sid] = {
            "summary": summary,
            "turns": self._lat.history,
        }
        await self._jsend({"type": "session_summary", "latency": summary})
        
        if self._cag_ws:
            try:
                await self._cag_ws.close()
            except Exception:
                pass
    
    # ── Startup ───────────────────────────────────────────────────────────────
    
    async def _startup_sequence(self):
        log.info(f"[{self.sid}] Waiting for STT ready…")
        try:
            await asyncio.wait_for(self._stt_ready.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            log.warning(f"[{self.sid}] STT not ready after 15s — continuing anyway")
        
        asyncio.create_task(self._prewarm_tts(), name=f"prewarm_{self.sid}")
        await self._jsend({"type": "ready", "message": "Pipeline ready — speak to begin"})
    
    async def _prewarm_tts(self):
        tts_ws = None
        try:
            tts_ws = await _ws_connect(
                TTS_WS_URL, max_retries=1,
                label=f"[{self.sid}] prewarm",
                max_size=10 * 1024 * 1024,
                ping_interval=None, ping_timeout=None,
            )
            await tts_ws.send(json.dumps({
                "text": "Hello, I am ready.",
                "chunk_tone": "tone",
            }))
            while True:
                frame = await asyncio.wait_for(tts_ws.recv(), timeout=10.0)
                if isinstance(frame, bytes) and frame == b"":
                    break
            log.info(f"[{self.sid}] TTS pre-warm complete")
        except Exception as e:
            log.debug(f"[{self.sid}] TTS pre-warm skipped: {e}")
        finally:
            if tts_ws:
                try:
                    await tts_ws.close()
                except Exception:
                    pass
    
    # ── Audio push ────────────────────────────────────────────────────────────
    
    def push_audio(self, frame: bytes):
        self._last_pong_time = time.monotonic()
        if self._audio_q.full():
            try:
                self._audio_q.get_nowait()
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

    async def _cancel_tts_for_barge_in(self):
        """
        Phase-1 barge-in: stop TTS audio immediately so the user isn't heard
        over by the AI.  CAG is NOT cancelled yet — we wait for an actual
        transcribed segment before committing to the full barge-in.

        If no words are transcribed (too brief / false VAD spike), the CAG
        stream keeps running and the AI response is preserved.
        If words arrive as segment[barge_in=True], the segment handler calls
        _do_barge_in_immediate() which closes CAG and sends the new query.
        """
        async with self._turn_lock:
            await self._cancel_tts_for_barge_in_unlocked()

    async def _cancel_tts_for_barge_in_unlocked(self):
        log.info(f"[{self.sid}] 🎤 Barge-in: TTS stopped — waiting for transcribed words")

        # Abort in-flight TTS tasks (turn_id mismatch makes them exit cleanly)
        self._current_turn_id = None
        for task in self._active_tts_tasks:
            if not task.done():
                task.cancel()
        self._active_tts_tasks.clear()
        _drain_q(self._sentence_queue)

        # Send hard cancel to any currently-checked-out TTS connection so
        # the synth aborts mid-sentence (vs draining all remaining audio).
        await self._broadcast_tts_cancel()

        if self._tts_ws:
            try:
                await self._tts_ws.close()
            except Exception:
                pass
            self._tts_ws = None

        _drain_q(self._tts_pcm_q)
        _drain_q(self._stt_ref_q)

        if self._open_mic_task and not self._open_mic_task.done():
            self._open_mic_task.cancel()
        self._open_mic_task = None
        self._tts_pcm_bytes_this_turn = 0
        self._tts_stopped_at = time.monotonic()
        self._echo_gate.tts_stopped()

        # Open mic immediately so STT starts capturing the user's words
        self._stt_ctrl_q.put_nowait(
            b'\x02' + json.dumps({"type": "ai_state", "speaking": False}).encode()
        )

        # Tell client to drop buffered audio and show barge-in UI indicator
        await self._jsend({"type": "clear_audio"})
        await self._jsend({"type": "barge_in"})
        # NOTE: self.state intentionally stays SPEAKING/THINKING so the segment
        # handler can detect we are still mid-turn and call _do_barge_in_immediate().

    async def _do_barge_in_immediate(self):
        # A4: serialise overlapping barge-ins.  If a second barge-in arrives
        # while the first is still running cleanup, it queues here, then
        # runs to completion — guaranteeing the LATEST user utterance wins
        # and avoiding races on _current_turn_id / _cag_ws / _tts_ws.
        async with self._turn_lock:
            await self._do_barge_in_unlocked()

    async def _do_barge_in_unlocked(self):
        # Skip only if we're already mid-barge-in with no new turn started yet
        # (dedup guard — prevents double-firing from the same detection event).
        # A *new* barge-in while the previous one's segment is already queued
        # (state back to THINKING/SPEAKING) must always proceed.
        if self._barge_in and self._current_turn_id is None:
            log.debug(f"[{self.sid}] _do_barge_in_immediate: already active, skip")
            return

        self._turn_seq += 1
        log.info(f"[{self.sid}] ⚡ BARGE-IN #{self._turn_seq}")

        if self._lat.current:
            self._lat.current.barge_in = True

        now = time.monotonic()
        self._barge_in = True
        self._barge_in_until = now + BARGE_IN_COOLDOWN_S
        self.state = State.IDLE  # Reset immediately so finally blocks see correct state

        # Invalidate the current turn so in-flight TTS tasks abort themselves
        self._current_turn_id = None

        # Cancel all active TTS tasks
        for task in self._active_tts_tasks:
            if not task.done():
                task.cancel()
        self._active_tts_tasks.clear()
        
        # Clear sentence queue (legacy drain — kept for safety)
        _drain_q(self._sentence_queue)

        # A3: blast a cancel frame to all TTS connections so they abort
        # synthesis instantly instead of draining the remaining sentence.
        await self._broadcast_tts_cancel()
        
        # Close any single TTS connection (pooled connections are managed
        # separately by the pool manager).
        if self._tts_ws:
            try:
                await self._tts_ws.close()
            except Exception:
                pass
            self._tts_ws = None
        
        # Send cancel to CAG but KEEP the WebSocket open — closing it forces
        # a full TCP+WS reconnect for every barge-in (adds ~50-200ms delay).
        # Instead: the cancel frame makes CAG abort the LLM generator and
        # send a "done" frame; _stream_cag_immediate drains those leftover
        # frames and returns cleanly; _cag_loop then sends the new barge-in
        # query on the SAME connection with zero reconnect overhead.
        cag_ws = self._cag_ws
        if cag_ws is not None:
            cancel_tid = self._lat.current.turn_id if self._lat.current else None
            try:
                await cag_ws.send(json.dumps({"type": "cancel", "turn_id": cancel_tid}))
            except Exception:
                pass
            # Do NOT close cag_ws here.

        # Drain stale pending queries so the barge-in utterance is the next
        # (and only) query the CAG loop processes after reconnecting.
        _drain_q(self._query_q)

        # Clear any pending PCM
        _drain_q(self._tts_pcm_q)

        # Stop feeding stale TTS audio to the STT VoiceGate fingerprint so
        # the centroid doesn't keep drifting with cancelled-turn audio.
        _drain_q(self._stt_ref_q)

        # Cancel any pending mic-open task — open mic immediately on barge-in
        if self._open_mic_task and not self._open_mic_task.done():
            self._open_mic_task.cancel()
        self._open_mic_task = None
        self._tts_pcm_bytes_this_turn = 0
        self._tts_stopped_at = time.monotonic()
        self._echo_gate.tts_stopped()

        # Signal STT that AI stopped speaking immediately on barge-in
        self._stt_ctrl_q.put_nowait(
            b'\x02' + json.dumps({"type": "ai_state", "speaking": False}).encode()
        )

        # Tell the client to drop any audio it has buffered locally so the
        # cancelled sentence does not keep playing after the user has barged in.
        await self._jsend({"type": "clear_audio"})
        await self._jsend({"type": "barge_in"})

        # A2: clear the predicted-mute (the confirmed barge-in supersedes it).
        if self._predict_unmute_task and not self._predict_unmute_task.done():
            self._predict_unmute_task.cancel()
        self._predict_unmute_task = None
        if self._audio_muted:
            self._audio_muted = False
            await self._jsend({"type": "audio_mute", "muted": False})
    
    # ─── TTS WebSocket pool manager ───────────────────────────────────────────
    # On startup, creates 3 TTS connections
    # Periodically checks pool size
    # If connections die, refills the pool
    # Allows _sentence_processor to grab ready-to-use connections
    async def _tts_pool_manager(self):
        """Maintains warm TTS WebSocket connections."""
        for i in range(TTS_POOL_SIZE):
            asyncio.create_task(self._tts_pool_open_one(), name=f"tts_pool_init_{i}")
        
        while self._running:
            await asyncio.sleep(0.5)
            size = self._tts_pool.qsize()
            need = TTS_POOL_SIZE - size
            if need > 0:
                for _ in range(min(need, 2)):
                    asyncio.create_task(self._tts_pool_open_one(), name=f"tts_pool_refill")
    #Creates one TTS connection with retries
    async def _tts_pool_open_one(self):
        try:
            ws = await _ws_connect(
                TTS_WS_URL, max_retries=2,
                label=f"[{self.sid}] TTS-pool",
                max_size=10 * 1024 * 1024,
                ping_interval=None, ping_timeout=None,
            )
            await self._tts_pool.put(ws)
            log.debug(f"[{self.sid}] TTS pool +1 (size={self._tts_pool.qsize()})")
        except Exception as e:
            log.warning(f"[{self.sid}] TTS pool open failed: {e}")
    #_tts_pool_checkout(): Get a connection from pool (wait up to 1s)
    # _tts_pool_checkin(): Return connection to pool
    async def _tts_pool_checkout(self) -> Optional[object]:
        try:
            return await asyncio.wait_for(self._tts_pool.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def _tts_pool_checkin(self, ws, healthy: bool = True):
        if ws is None:
            return
        if not healthy:
            try:
                await ws.close()
            except Exception:
                pass
            return
        await self._tts_pool.put(ws)
    
    # ─── TTS sentence streaming (called directly from _stream_cag_immediate) ──
    async def _tts_stream_sentence(
        self,
        text: str,
        sentence_idx: int,
        turn_id: str,
        prev_done: Optional[asyncio.Event] = None,
        my_done: Optional[asyncio.Event] = None,
    ):
        """
        Synthesize one sentence and forward audio to the client.

        PARALLEL SYNTHESIS, SEQUENTIAL PLAYBACK
        ────────────────────────────────────────
        • TTS synthesis starts immediately (no waiting) → low first-audio latency.
        • Audio bytes are buffered locally while synthesizing.
        • Forwarding to the client is gated behind `prev_done` so sentence N+1
          never plays until sentence N has finished, eliminating double-audio.
        • `my_done` is always set in `finally` so downstream sentences unblock
          even on cancellation or barge-in (they check the barge-in flag and skip).
        """
        if not text or len(text.split()) < MIN_SENTENCE_LENGTH:
            if my_done is not None:
                my_done.set()
            return

        # Abort before touching TTS if this turn is already superseded
        if self._current_turn_id != turn_id:
            log.debug(f"[{self.sid}] TTS sentence {sentence_idx} dropped — stale turn")
            if my_done is not None:
                my_done.set()
            return

        tts_ws = None
        try:
            # ── Get a healthy TTS connection ───────────────────────────────
            tts_ws = await self._tts_pool_checkout()
            if tts_ws is not None:
                try:
                    await asyncio.wait_for(tts_ws.ping(), timeout=0.3)
                except Exception:
                    try:
                        await tts_ws.close()
                    except Exception:
                        pass
                    tts_ws = None

            if tts_ws is None:
                tts_ws = await _ws_connect(
                    TTS_WS_URL, max_retries=TTS_MAX_RETRIES,
                    label=f"[{self.sid}] TTS-{sentence_idx}",
                    max_size=10 * 1024 * 1024,
                    ping_interval=None, ping_timeout=None,
                )

            if self._current_turn_id != turn_id or self._barge_in:
                return

            stripped = text.strip()
            if stripped.endswith("?") or stripped.endswith("!") or len(stripped) <= 120:
                chunk_tone = "tone"
            else:
                chunk_tone = "logic"

            log.info(f"[{self.sid}] → TTS sentence {sentence_idx+1}: {text[:60]}…")

            # Signal STT BEFORE synthesis so mic is muted during AI speech
            if sentence_idx == 0 and self.state != State.SPEAKING:
                if self._open_mic_task and not self._open_mic_task.done():
                    self._open_mic_task.cancel()
                self._open_mic_task = None
                self._tts_pcm_bytes_this_turn = 0
                self.state = State.SPEAKING
                self._stt_ctrl_q.put_nowait(
                    b'\x02' + json.dumps({"type": "ai_state", "speaking": True}).encode()
                )

            await tts_ws.send(json.dumps({
                "text": text,
                "language": TTS_LANGUAGE,
                "voice": TTS_SPEAKER,
                "chunk_tone": chunk_tone,
            }))

            # ── Phase 1: Synthesize — collect all audio while TTS generates ──
            audio_buf: List[bytes] = []
            got_first = False
            seen_any_frame = False  # tracks whether we've consumed *any* frame yet
            async for frame in tts_ws:
                if self._barge_in or self._current_turn_id != turn_id:
                    break
                if isinstance(frame, bytes):
                    if frame == b"":
                        break
                    if not frame:
                        continue
                    # Meta frame is ALWAYS the first frame of a synth, prefixed
                    # with byte 0x02. Once we've seen a non-meta frame, never
                    # try to parse another as meta (PCM samples can legally
                    # start with 0x02, so byte-sniffing every frame is unsafe).
                    if not seen_any_frame and frame[:1] == b"\x02":
                        seen_any_frame = True
                        try:
                            meta = json.loads(frame[1:].decode("utf-8"))
                            if isinstance(meta, dict) and meta.get("type") == "meta":
                                await self._jsend({
                                    "type":        "tts_meta",
                                    "sample_rate": meta.get("sample_rate"),
                                    "engine":      meta.get("engine"),
                                })
                                continue
                        except Exception:
                            # Not actually a meta frame — fall through and
                            # treat the bytes as audio.
                            pass
                    seen_any_frame = True
                    if not got_first:
                        got_first = True
                        if sentence_idx == 0:
                            # Latency: first audio chunk arrived from TTS
                            self._lat.on_first_audio()
                    audio_buf.append(frame)

            # Return connection to pool before the playback wait so the pool
            # stays healthy even while we wait for the previous sentence to end.
            healthy = (not self._barge_in) and (self._current_turn_id == turn_id)
            await self._tts_pool_checkin(tts_ws, healthy=healthy)
            tts_ws = None

            # ── Phase 2: Wait for previous sentence to finish playing ────────
            # `prev_done` is set the moment the sentence before us finishes
            # forwarding its last byte. Waiting here keeps audio sequential.
            if prev_done is not None:
                await prev_done.wait()

            # ── Phase 3: Forward audio to client ────────────────────────────
            if not self._barge_in and self._current_turn_id == turn_id:
                for chunk in audio_buf:
                    if self._barge_in or self._current_turn_id != turn_id:
                        break
                    self._echo_gate.feed_tts(chunk)
                    self._tts_pcm_bytes_this_turn += len(chunk)
                    # Feed TTS audio to STT fingerprint gate (drop if queue full)
                    try:
                        self._stt_ref_q.put_nowait(chunk)
                    except asyncio.QueueFull:
                        pass
                    await self._bsend(chunk)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"[{self.sid}] TTS sentence {sentence_idx} error: {e}")
        finally:
            if tts_ws is not None:
                try:
                    await tts_ws.close()
                except Exception:
                    pass
            # Always unblock the next sentence, even on error / barge-in
            if my_done is not None:
                my_done.set()
    
    # ─── STT loop ─────────────────────────────────────────────────────────────
    
    async def _stt_loop(self):
        log.info(f"[{self.sid}] STT → {STT_WS_URL}")
        retries = 0
        
        while self._running and retries < STT_MAX_RETRIES:
            try:
                stt_url = f"{STT_WS_URL}?sid={self.sid}&lang={self.lang}"
                stt_ws = await _ws_connect(
                    stt_url, max_retries=STT_MAX_RETRIES,
                    label=f"[{self.sid}] STT",
                    max_size=2 * 1024 * 1024,
                )
                self._stt_ws = stt_ws
                retries = 0
                log.info(f"[{self.sid}] STT connected ✓")
                self._stt_ready.set()
                
                async def _push():
                    while self._running:
                        frame = await self._audio_q.get()
                        try:
                            await stt_ws.send(frame if (frame and frame[0] == 0x01) else b'\x01' + frame)
                        except Exception:
                            pass

                async def _ctrl_push():
                    """Forward AI-state control frames to STT."""
                    while self._running:
                        try:
                            ctrl = await asyncio.wait_for(self._stt_ctrl_q.get(), timeout=0.5)
                            await stt_ws.send(ctrl)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            log.debug(f"[{self.sid}] STT ctrl push error: {e}")
                            break

                async def _ref_push():
                    """Forward TTS reference audio (0x03) to STT for fingerprint enrollment."""
                    while self._running:
                        try:
                            chunk = await asyncio.wait_for(self._stt_ref_q.get(), timeout=0.5)
                            await stt_ws.send(b'\x03' + chunk)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            log.debug(f"[{self.sid}] STT ref push error: {e}")
                            break
                
                async def _recv():
                    guard = RepetitionGuard()
                    word_buf: list[str] = []
                    silence_task: Optional[asyncio.Task] = None
                    barge_triggered_this_turn = False
                    last_fired_text = ""
                    
                    async def _fire_query():
                        nonlocal word_buf, silence_task, barge_triggered_this_turn, last_fired_text
                        await asyncio.sleep(STT_SILENCE_MS / 1000.0)
                        if not word_buf:
                            return
                        text = " ".join(word_buf).strip()
                        word_count = len(word_buf)

                        # ── Always-on duplicate suppression ───────────────
                        # Identical text as the last fired query is ALWAYS
                        # an echo / hallucination repeat — never a real new
                        # user turn. Drop unconditionally regardless of state.
                        prev = last_fired_text.lower().strip()
                        cur  = text.lower().strip()
                        if prev and cur == prev:
                            log.debug(f"[{self.sid}] suppress duplicate fire: {text!r}")
                            word_buf = []
                            silence_task = None
                            return

                        # ── Partial-fragment debouncer ────────────────────
                        # While the AI is THINKING or SPEAKING, fragmented STT
                        # partials ("Do you" → "Do you me" → "you hear") would
                        # otherwise each spawn a brand-new turn. Suppress fires
                        # that are not real, distinct user utterances.
                        if self.state in (State.SPEAKING, State.THINKING):
                            if prev:
                                # Only suppress exact duplicates.  Even one
                                # new word ("you hear" → "you hear me") is a
                                # real new intent — barge-in must carry the
                                # full text to CAG, not just the partial.
                                if cur == prev:
                                    log.debug(
                                        f"[{self.sid}] suppress exact-dup extension: {text!r}"
                                    )
                                    word_buf = []
                                    silence_task = None
                                    return
                            if word_count < BARGE_IN_MIN_WORDS:
                                log.debug(
                                    f"[{self.sid}] suppress short barge-in attempt "
                                    f"({word_count}<{BARGE_IN_MIN_WORDS}): {text!r}"
                                )
                                word_buf = []
                                silence_task = None
                                return

                        word_buf = []
                        silence_task = None
                        barge_triggered_this_turn = False
                        if not text:
                            return
                        guard.reset()
                        last_fired_text = text
                        turn_id = str(uuid.uuid4())
                        self._lat.new_turn(turn_id, text)
                        self._lat.on_stt_segment()
                        log.info(f"[{self.sid}] STT: {text!r}")
                        await self._jsend({"type": "segment", "text": text})
                        if self.state in (State.SPEAKING, State.THINKING):
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
                            ev = json.loads(payload)
                        except Exception:
                            continue
                        kind = ev.get("type", "")
                        
                        if kind == "barge_in":
                            # Barge-in: stop BOTH TTS and CAG immediately so
                            # the AI cannot keep talking or generating while
                            # the user is speaking. Barge-in text is written
                            # as a new query the moment the STT segment arrives.
                            if not barge_triggered_this_turn:
                                barge_triggered_this_turn = True
                                await self._do_barge_in_immediate()
                                word_buf.clear()
                                last_fired_text = ""
                                if silence_task and not silence_task.done():
                                    silence_task.cancel()
                                silence_task = None
                            continue

                        elif kind == "barge_in_predicted":
                            # ── A2 dual-trigger: predicted (early) barge-in ──
                            # Sustained voice during AI speech, but no real
                            # word yet.  Mute outgoing audio instantly so the
                            # user doesn't hear the AI talk over them.  We do
                            # NOT cancel CAG yet — if no confirmed word
                            # follows within a short window we just unmute
                            # and the answer continues.
                            if not barge_triggered_this_turn and self.state == State.SPEAKING:
                                if not self._audio_muted:
                                    self._audio_muted = True
                                    await self._jsend({"type": "audio_mute", "muted": True})
                                    log.debug(f"[{self.sid}] predicted barge-in — audio muted")
                                # Restart the auto-unmute watchdog (~600ms).
                                if self._predict_unmute_task and not self._predict_unmute_task.done():
                                    self._predict_unmute_task.cancel()

                                async def _auto_unmute():
                                    try:
                                        await asyncio.sleep(0.6)
                                        if self._audio_muted and not barge_triggered_this_turn:
                                            self._audio_muted = False
                                            await self._jsend({"type": "audio_mute", "muted": False})
                                            log.debug(f"[{self.sid}] predict mute auto-cleared")
                                    except asyncio.CancelledError:
                                        pass

                                self._predict_unmute_task = asyncio.create_task(_auto_unmute())
                            continue
                        
                        elif kind == "word":
                            # Drop any words that landed during AI playback —
                            # these are echo leakage, not a real user turn.
                            if self._stt_buf_invalidate:
                                self._stt_buf_invalidate = False
                                word_buf.clear()
                                guard.reset()
                                if silence_task and not silence_task.done():
                                    silence_task.cancel()
                                silence_task = None
                                last_fired_text = ""
                            word = ev.get("word", "").strip().rstrip("?.!,;:")
                            if word:
                                if guard.feed(word):
                                    word_buf.clear()
                                    guard.reset()
                                    if silence_task and not silence_task.done():
                                        silence_task.cancel()
                                    silence_task = None
                                    await self._jsend({"type": "hallucination_reset"})
                                    continue
                                is_new_word = not word_buf or word_buf[-1].lower() != word.lower()
                                if is_new_word:
                                    word_buf.append(word)
                                self._lat.on_stt_first_word()
                                if silence_task and not silence_task.done():
                                    silence_task.cancel()
                                # During a barge-in, the STT pipeline will emit a
                                # complete `segment` event when the utterance ends.
                                # Don't also fire a partial word-timeout query —
                                # that would send "you hear" AND then "you hear me"
                                # as two separate CAG queries for the same utterance.
                                # Rely only on the segment event for the full text.
                                if not barge_triggered_this_turn:
                                    silence_task = asyncio.create_task(_fire_query())
                                else:
                                    silence_task = None
                                # Only forward to client when the word changes (stops rapid-fire spam)
                                if is_new_word:
                                    await self._jsend(ev)
                        
                        elif kind == "segment":
                            text = ev.get("text", "").strip()
                            word_count = len(text.split()) if text else 0
                            if not text:
                                continue
                            # Drop the very first segment after AI playback if it
                            # is identical/prefix of the last AI-time echo \u2014
                            # echo leakage tail. The flag is cleared on the next
                            # genuinely-new word arrival.
                            if self._stt_buf_invalidate:
                                self._stt_buf_invalidate = False
                                if last_fired_text and text.lower().strip() == last_fired_text.lower().strip():
                                    log.debug(f"[{self.sid}] drop echo-tail segment: {text!r}")
                                    word_buf.clear()
                                    if silence_task and not silence_task.done():
                                        silence_task.cancel()
                                    silence_task = None
                                    continue
                            if silence_task and not silence_task.done():
                                silence_task.cancel()
                                silence_task = None
                            word_buf.clear()
                            guard.reset()
                            barge_triggered_this_turn = False
                            if last_fired_text and last_fired_text.lower() == text.lower():
                                last_fired_text = ""
                                continue
                            last_fired_text = ""
                            seg_turn_id = str(uuid.uuid4())
                            self._lat.new_turn(seg_turn_id, text)
                            self._lat.on_stt_segment()
                            log.info(f"[{self.sid}] STT segment: {text!r}")
                            await self._jsend(ev)
                            if self.state in (State.SPEAKING, State.THINKING):
                                await self._do_barge_in_immediate()
                                _drain_q(self._query_q)
                                await self._query_q.put((seg_turn_id, text))
                            else:
                                await self._query_q.put((seg_turn_id, text))
                        
                        elif kind == "partial":
                            await self._jsend(ev)
                        elif kind == "pong":
                            self._last_pong_time = time.monotonic()
                
                await asyncio.gather(_push(), _recv(), _ctrl_push(), _ref_push())
                
            except Exception as e:
                retries += 1
                log.warning(f"[{self.sid}] STT disconnected ({e}), retry {retries}/{STT_MAX_RETRIES}")
                self._stt_ws = None
                if retries < STT_MAX_RETRIES:
                    await asyncio.sleep(min(2 ** retries, 30))
    
    # ─── CAG loop with immediate parallel TTS streaming ─────────────────────────────────
    
    async def _cag_loop(self):
        self._cag_turn_count = 0
        retries = 0
        
        while self._running and retries < CAG_MAX_RETRIES:
            try:
                cag_ws = await _ws_connect(
                    CAG_WS_URL, max_retries=CAG_MAX_RETRIES,
                    label=f"[{self.sid}] CAG",
                    max_size=4 * 1024 * 1024,
                )
                self._cag_ws = cag_ws
                retries = 0
                log.info(f"[{self.sid}] CAG WebSocket connected ✓")
                
                while self._running:
                    query = await self._query_q.get()
                    self._last_query_time = time.monotonic()
                    
                    if isinstance(query, tuple):
                        turn_id, query_text = query
                    else:
                        turn_id = str(uuid.uuid4())
                        query_text = query
                    
                    self._cag_turn_count += 1
                    came_from_barge_in = self._barge_in  # capture before reset
                    self._barge_in = False
                    self.state = State.THINKING
                    
                    self._lat.on_query_sent()
                    log.info(f"[{self.sid}] CAG query: {query_text!r}")
                    await self._jsend({"type": "thinking", "turn_id": turn_id})

                    # Send reset=True on the very first turn and on any turn
                    # that immediately follows a barge-in.  This clears CAG's
                    # in-flight context so the partial from the previous turn
                    # ("you hear") doesn't bleed into the barge-in reply
                    # ("you hear me").
                    do_reset = self._cag_turn_count == 1 or came_from_barge_in
                    self._barge_in = False  # consumed — clear for next turn

                    try:
                        await cag_ws.send(json.dumps({
                            "type":    "query",
                            "turn_id": turn_id,
                            "message": query_text,
                            "reset":   do_reset,
                        }))
                    except Exception as e:
                        log.error(f"[{self.sid}] CAG send error: {e}")
                        raise
                    
                    # Stream CAG tokens with immediate sentence detection
                    await self._stream_cag_immediate(cag_ws, turn_id)
                    
            except asyncio.CancelledError:
                return
            except Exception as e:
                self._cag_ws = None
                if self._barge_in:
                    # Barge-in intentionally closed the CAG WS — don't burn a
                    # retry. Reconnect immediately to serve the new query.
                    log.info(f"[{self.sid}] CAG WS closed (barge-in) — reconnecting")
                    retries = 0
                else:
                    retries += 1
                    log.warning(f"[{self.sid}] CAG WS disconnected ({e}), retry {retries}/{CAG_MAX_RETRIES}")
                if retries < CAG_MAX_RETRIES:
                    delay = 0.0 if self._barge_in else min(2 ** retries, 10)
                    if delay > 0:
                        await asyncio.sleep(delay)
    
    async def _stream_cag_immediate(self, cag_ws, turn_id: str):
        """Stream CAG tokens and IMMEDIATELY send complete sentences to TTS."""
        full_text_parts = []
        current_sentence_parts = []
        stream_confirmed = False
        sentences_sent = 0
        last_flush_time = time.monotonic()

        # Register this as the active turn — stale TTS tasks will abort themselves
        self._current_turn_id = turn_id
        # Cancel any TTS tasks left over from the previous (interrupted) turn
        for t in self._active_tts_tasks:
            if not t.done():
                t.cancel()
        self._active_tts_tasks.clear()

        # Playback-ordering chain: sentence N waits on done_events[N] before
        # forwarding audio, then sets done_events[N+1] when it finishes.
        # Sentence 0 gets a pre-set event so it plays immediately.
        _first_done = asyncio.Event()
        _first_done.set()
        _prev_done = _first_done
        
        try:
            async for raw_frame in cag_ws:
                if self._barge_in:
                    log.info(f"[{self.sid}] CAG stream aborted (barge-in) — draining")
                    break
                
                try:
                    frame = json.loads(raw_frame) if isinstance(raw_frame, (str, bytes)) else {}
                except Exception:
                    continue
                
                ftype = frame.get("type", "")
                
                if ftype == "turn_id":
                    server_tid = frame.get("turn_id", "")
                    if server_tid != turn_id:
                        return
                    stream_confirmed = True
                    continue
                
                if ftype == "done":
                    break
                
                if ftype in ("error", "timeout"):
                    detail = frame.get("detail", ftype)
                    await self._jsend({"type": "error", "detail": detail})
                    return
                
                if ftype != "token" or not stream_confirmed:
                    continue
                
                token = frame.get("token", "")
                if not token:
                    continue
                
                self._lat.on_first_token()
                self._lat.on_token()
                full_text_parts.append(token)
                current_sentence_parts.append(token)
                # display_only=True tells the client this text is for UI rendering;
                # the canonical audio is the PCM stream. Prevents double-TTS.
                await self._jsend({"type": "ai_token", "token": token,
                                   "display_only": not EMIT_TEXT_FOR_CLIENT_TTS})
                
                # Check for complete sentence IMMEDIATELY
                is_boundary = False
                tok_stripped = token.rstrip()
                if tok_stripped.endswith(('.', '!', '?')):
                    # True sentence boundary — flush even on short sentences
                    if len(current_sentence_parts) >= 2:
                        is_boundary = True

                # Soft clause boundary: comma / semicolon / colon after enough words.
                # This gets TTS started early on long answers instead of waiting
                # for a full stop or the 0.3s force-flush timeout.
                if not is_boundary and tok_stripped.endswith((',', ';', ':')):
                    word_count = len(_smart_join(current_sentence_parts).split())
                    if word_count >= MIN_CLAUSE_WORDS:
                        is_boundary = True

                # Force flush if no boundary detected after timeout
                if not is_boundary and time.monotonic() - last_flush_time > SENTENCE_TIMEOUT_S:
                    if current_sentence_parts:
                        is_boundary = True
                        log.debug(f"[{self.sid}] Force flush after timeout")
                
                if is_boundary:
                    sentence = _smart_join(current_sentence_parts).strip()
                    current_sentence_parts = []
                    
                    if sentence and len(sentence.split()) >= MIN_SENTENCE_LENGTH:
                        sentences_sent += 1
                        self._lat.on_sentence()
                        
                        if sentences_sent == 1:
                            self._lat.on_first_sentence()
                        
                        # Feed echo filter per-sentence (not just at end)
                        self._text_echo_filter.feed_ai_text(sentence)
                        # Notify client — display_only=True means "render text,
                        # do NOT synthesize" (audio comes via PCM stream).
                        await self._jsend({"type": "ai_sentence", "text": sentence,
                                           "idx": sentences_sent - 1,
                                           "display_only": not EMIT_TEXT_FOR_CLIENT_TTS})
                        # Dispatch to TTS — synthesis in parallel, playback sequential
                        _my_done = asyncio.Event()
                        task = asyncio.create_task(
                            self._tts_stream_sentence(
                                sentence, sentences_sent - 1, turn_id, _prev_done, _my_done
                            )
                        )
                        _prev_done = _my_done
                        self._active_tts_tasks.append(task)
                        self._active_tts_tasks = [t for t in self._active_tts_tasks if not t.done()]
                        
                        log.debug(f"[{self.sid}] Sentence {sentences_sent}: {sentence[:50]}…")
                    
                    last_flush_time = time.monotonic()
            
            # If barge-in: drain remaining CAG frames so the WS stays clean
            # for immediate reuse, then return (finally block still runs).
            if self._barge_in:
                try:
                    drain_limit = 100  # never block forever
                    async for drain_frame in cag_ws:
                        if drain_limit <= 0:
                            break
                        drain_limit -= 1
                        try:
                            df = json.loads(drain_frame) if isinstance(drain_frame, (str, bytes)) else {}
                        except Exception:
                            df = {}
                        if df.get("type") in ("done", "error", "timeout"):
                            break
                except Exception:
                    pass
                return  # finally runs; _cag_loop reuses the same WS immediately

            # Handle any remaining text
            if current_sentence_parts:
                sentence = _smart_join(current_sentence_parts).strip()
                if sentence and len(sentence.split()) >= MIN_SENTENCE_LENGTH:
                    sentences_sent += 1
                    self._lat.on_sentence()
                    
                    self._text_echo_filter.feed_ai_text(sentence)
                    await self._jsend({"type": "ai_sentence", "text": sentence,
                                       "idx": sentences_sent - 1,
                                       "display_only": not EMIT_TEXT_FOR_CLIENT_TTS})
                    _my_done = asyncio.Event()
                    task = asyncio.create_task(
                        self._tts_stream_sentence(
                            sentence, sentences_sent - 1, turn_id, _prev_done, _my_done
                        )
                    )
                    _prev_done = _my_done
                    self._active_tts_tasks.append(task)
                    self._active_tts_tasks = [t for t in self._active_tts_tasks if not t.done()]
            
            # Wait for all TTS to complete
            if self._active_tts_tasks:
                await asyncio.gather(*self._active_tts_tasks, return_exceptions=True)
            
            full_text = "".join(full_text_parts).strip()
            self._text_echo_filter.feed_ai_text(full_text)
            
        except Exception as e:
            if self._barge_in:
                # WS error during barge-in drain — return cleanly;
                # _cag_loop will reconnect naturally on next send().
                log.info(f"[{self.sid}] CAG stream interrupted by barge-in: {e}")
                return
            log.error(f"[{self.sid}] CAG stream error: {e}")
        
        finally:
            report = self._lat.complete_turn()
            if report:
                await self._jsend({"type": "latency", "stage": "turn_complete", **report})
            await self._jsend({"type": "done", "chunks": sentences_sent})
            # Tell STT what the AI said — Whisper context + text echo filter
            full_text = "".join(full_text_parts).strip()
            if full_text:
                self._stt_ctrl_q.put_nowait(
                    b'\x02' + json.dumps({"type": "assistant_turn", "text": full_text}).encode()
                )
            # Delay mic re-open until audio FINISHES PLAYING (not when synthesis ends).
            # Synthesis is fast (~125ms) but audio plays for pcm_bytes/(22050*2) seconds.
            pcm_bytes = self._tts_pcm_bytes_this_turn
            self._tts_pcm_bytes_this_turn = 0
            playback_s = pcm_bytes / (TTS_PCM_SAMPLE_RATE * 2) if pcm_bytes > 0 else 0.0
            was_speaking = (self.state == State.SPEAKING)
            self.state = State.IDLE

            if self._barge_in:
                # Mic was already re-opened immediately in _do_barge_in_immediate;
                # creating another _open_mic_task here would re-suppress the mic
                # after the old (now-cancelled) playback duration.
                self._tts_pcm_bytes_this_turn = 0
            else:
                async def _open_mic(playback_s=playback_s, was_speaking=was_speaking):
                    if playback_s > 0:
                        await asyncio.sleep(playback_s)
                    if was_speaking:
                        try:
                            self._stt_ctrl_q.put_nowait(
                                b'\x02' + json.dumps({"type": "ai_state", "speaking": False}).encode()
                            )
                        except asyncio.QueueFull:
                            pass
                    self._tts_stopped_at = time.monotonic()
                    self._echo_gate.tts_stopped()
                    # Only invalidate the STT buffer when the AI actually played
                    # audio that could have leaked into the mic. If TTS was
                    # cancelled before the first frame (barge-in), playback_s==0
                    # so there is nothing to echo-guard and we must NOT drop the
                    # user’s barge-in words.
                    if playback_s > 0:
                        self._stt_buf_invalidate = True
                    log.debug(f"[{self.sid}] Mic re-opened after {playback_s:.2f}s playback")

                self._open_mic_task = asyncio.create_task(_open_mic())
    
    async def _cag_loop_http_fallback(self):
        try:
            query = self._query_q.get_nowait()
        except asyncio.QueueEmpty:
            return
        
        if isinstance(query, tuple):
            turn_id, query_text = query
        else:
            turn_id = str(uuid.uuid4())
            query_text = query
        
        self._cag_turn_count += 1
        self._barge_in = False
        self.state = State.THINKING
        self._lat.new_turn(turn_id, query_text)
        self._lat.on_query_sent()
        
        log.info(f"[{self.sid}] CAG HTTP fallback: {query_text!r}")
        await self._jsend({"type": "thinking", "turn_id": turn_id})
        
        full_text_parts = []
        current_sentence_parts = []
        sentences_sent = 0
        last_flush_time = time.monotonic()
        self._current_turn_id = turn_id
        for t in self._active_tts_tasks:
            if not t.done():
                t.cancel()
        self._active_tts_tasks.clear()

        # Playback-ordering chain (same as WS path) so HTTP-fallback sentences
        # play sequentially instead of overlapping.
        _first_done = asyncio.Event()
        _first_done.set()
        _prev_done = _first_done
        
        async with httpx.AsyncClient(
            base_url=CAG_HTTP_URL,
            timeout=httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0),
        ) as http:
            try:
                async with http.stream(
                    "POST", "/chat/stream",
                    json={
                        "message": query_text,
                        "reset_session": self._cag_turn_count == 1,
                        "turn_id": turn_id,
                    },
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    stream_confirmed = False
                    async for line in resp.aiter_lines():
                        if self._barge_in:
                            return
                        if not line.startswith("data:"):
                            continue
                        # Preserve leading space in token content — Qwen2.5/tiktoken
                        # tokenizers embed a leading space on every genuine word-start
                        # token (e.g. " you"). Stripping it produces merged words like
                        # "assistyou". Only strip for control markers ([DONE] etc.).
                        data    = line[5:]          # raw content, space preserved
                        stripped = data.strip()     # used only for control checks
                        if stripped.startswith("[TURN_ID]"):
                            server_tid = stripped[9:].strip()
                            if server_tid != turn_id:
                                return
                            stream_confirmed = True
                            continue
                        if stripped == "[DONE]":
                            break
                        if stripped in ("[TIMEOUT]", "") or stripped.startswith("[ERROR]"):
                            return
                        if not stripped:
                            continue
                        if not stream_confirmed:
                            continue
                        
                        self._lat.on_first_token()
                        self._lat.on_token()
                        full_text_parts.append(data)
                        current_sentence_parts.append(data)
                        await self._jsend({"type": "ai_token", "token": data,
                                           "display_only": not EMIT_TEXT_FOR_CLIENT_TTS})
                        
                        # Check for complete sentence
                        is_boundary = False
                        if data.endswith(('.', '!', '?')):
                            if len(current_sentence_parts) >= 2:
                                is_boundary = True
                        
                        if not is_boundary and time.monotonic() - last_flush_time > SENTENCE_TIMEOUT_S:
                            if current_sentence_parts:
                                is_boundary = True
                        
                        if is_boundary:
                            sentence = _smart_join(current_sentence_parts).strip()
                            current_sentence_parts = []
                            
                            if sentence and len(sentence.split()) >= MIN_SENTENCE_LENGTH:
                                sentences_sent += 1
                                self._lat.on_sentence()
                                
                                if sentences_sent == 1:
                                    self._lat.on_first_sentence()
                                
                                self._text_echo_filter.feed_ai_text(sentence)
                                await self._jsend({"type": "ai_sentence", "text": sentence,
                                                   "idx": sentences_sent - 1,
                                                   "display_only": not EMIT_TEXT_FOR_CLIENT_TTS})
                                _my_done = asyncio.Event()
                                task = asyncio.create_task(
                                    self._tts_stream_sentence(
                                        sentence, sentences_sent - 1, turn_id,
                                        _prev_done, _my_done
                                    )
                                )
                                _prev_done = _my_done
                                self._active_tts_tasks.append(task)
                                self._active_tts_tasks = [t for t in self._active_tts_tasks if not t.done()]
                            
                            last_flush_time = time.monotonic()
                    
                    if current_sentence_parts:
                        sentence = _smart_join(current_sentence_parts).strip()
                        if sentence and len(sentence.split()) >= MIN_SENTENCE_LENGTH:
                            sentences_sent += 1
                            self._lat.on_sentence()
                            
                            self._text_echo_filter.feed_ai_text(sentence)
                            await self._jsend({"type": "ai_sentence", "text": sentence,
                                               "idx": sentences_sent - 1,
                                               "display_only": not EMIT_TEXT_FOR_CLIENT_TTS})
                            _my_done = asyncio.Event()
                            task = asyncio.create_task(
                                self._tts_stream_sentence(
                                    sentence, sentences_sent - 1, turn_id,
                                    _prev_done, _my_done
                                )
                            )
                            _prev_done = _my_done
                            self._active_tts_tasks.append(task)
                            self._active_tts_tasks = [t for t in self._active_tts_tasks if not t.done()]
                    
                    if self._active_tts_tasks:
                        await asyncio.gather(*self._active_tts_tasks, return_exceptions=True)
                    
            except Exception as e:
                log.error(f"[{self.sid}] CAG HTTP fallback error: {e}")
                return
        
        full_text = "".join(full_text_parts).strip()
        if full_text and not self._barge_in:
            self._text_echo_filter.feed_ai_text(full_text)
            self._stt_ctrl_q.put_nowait(
                b'\x02' + json.dumps({"type": "assistant_turn", "text": full_text}).encode()
            )
        
        report = self._lat.complete_turn()
        if report:
            await self._jsend({"type": "latency", "stage": "turn_complete", **report})
        await self._jsend({"type": "done", "chunks": sentences_sent})
        self.state = State.IDLE
        self._tts_stopped_at = time.monotonic()
        self._echo_gate.tts_stopped()


# ─── App startup ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _gateway_startup():
    log.info("[startup] Gateway v18.0 — ultra-low latency parallel streaming")


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = GatewaySession(ws)
    # Negotiate language from the client URL: /ws?lang=fr  (default 'auto')
    _req_lang = (ws.query_params.get("lang") or DEFAULT_LANG or "auto").strip().lower()
    if _req_lang in ("", "auto", "en", "fr"):
        session.lang = _req_lang or "auto"
    else:
        log.warning(f"[{session.sid}] unsupported lang={_req_lang!r}, falling back to auto")
        session.lang = "auto"
    pipeline = asyncio.create_task(session.run())
    log.info(f"[{session.sid}] client connected  lang={session.lang!r}")
    
    if TEST_MODE:
        await ws.send_json({"type": "ready", "message": "TEST MODE — ready"})
    
    try:
        async for raw in ws.iter_bytes():
            if not raw:
                continue
            ftype = raw[0]
            payload = raw[1:]
            
            if ftype == 0x01:
                session.push_audio(raw)
                session._last_pong_time = time.monotonic()
            
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
                elif mtype == "inject_query":
                    text = ctrl.get("text", "").strip()
                    if text:
                        turn_id = str(uuid.uuid4())
                        session._lat.new_turn(turn_id, text)
                        session._lat.on_stt_segment()
                        await ws.send_json({"type": "segment", "text": text})
                        await session._query_q.put((turn_id, text))
                elif mtype == "get_stats":
                    await ws.send_json({
                        "type": "stats",
                        "sid": session.sid,
                        "state": session.state.name,
                    })
                elif mtype == "get_latency":
                    await ws.send_json({
                        "type": "latency_snapshot",
                        "turns": session._lat.history,
                        "summary": session._lat.session_summary(),
                    })
    
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
    return {"status": "ok", "version": "18.0.0", "parallel_tts": True, "max_concurrent_tts": MAX_CONCURRENT_TTS}


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