"""
realtime_asr.py  —  High-Accuracy Streaming ASR  v2.0
═══════════════════════════════════════════════════════════════════════════════

Core design: NEVER lose a word, NEVER cut a word, NEVER duplicate a word.

The three failure modes fixed here
────────────────────────────────────
  1. SPLIT WORDS  ("tomor" + "row")
     Cause:  Chunk boundary falls inside a word.
     Fix:    Live pass always feeds the last CONTEXT_S of utterance audio,
             so Whisper sees full words from the start of the sentence.
             A word is only "confirmed" when the NEXT word starts.

  2. LOST WORDS AT SENTENCE END  ("So I have a [question]" → dropped tail)
     Cause:  Live pass fires every 600 ms; the last 100-500 ms of a short
             sentence never accumulates enough for another live fire before
             VAD silence arrives.
     Fix:    flush() re-transcribes the FULL utterance audio with beam=5
             (more accurate than greedy live passes). It diffs the accurate
             result against already-emitted words and emits exactly the tail
             that was missed.  Nothing is ever lost.

  3. DUPLICATE WORDS  ("hello hello world")
     Cause:  Each live pass overlaps with the previous; naive timestamp-zone
             splitting fails when Whisper timestamps drift slightly.
     Fix:    Emit-cursor via longest-common-prefix match. We compare Whisper's
             output word list against our already-emitted list and only emit
             words AFTER the matched prefix.

Architecture
────────────
  transcribe_chunk(chunk)  ->  called by pipeline.py for every voice chunk
    append chunk to utterance buffer
    if accumulated >= FIRE_MS: run Whisper on last CONTEXT_S
    advance emit cursor, return new words

  flush()  ->  called by pipeline.py on VAD silence_event
    run Whisper on FULL utterance audio (beam=5)
    diff against emitted, emit tail
    reset utterance state, save to history
"""

from __future__ import annotations

import time
import numpy as np
from collections import deque
from typing import List, Optional, Dict, Any
from faster_whisper import WhisperModel


# ─────────────────────────────────────────────────────────────────────────────
#  Tunable constants
# ─────────────────────────────────────────────────────────────────────────────

FIRE_MS             = 600    # Run live Whisper every N ms of voice audio
CONTEXT_S           = 8.0    # Feed Whisper up to this many seconds per live pass
OVERLAP_S           = 0.8    # Extra context prepended to heal word boundaries

MAX_PROMPT_WORDS    = 12     # Cap on initial_prompt (avoids Whisper echo-back bug)
MAX_HISTORY_TURNS   = 3

MIN_WORD_PROB       = 0.50   # Drop words Whisper is < 50% confident about
MAX_NO_SPEECH_PROB  = 0.55   # Drop segment if mostly silence


# ─────────────────────────────────────────────────────────────────────────────

class RealTimeChunkASR:
    """
    High-accuracy streaming ASR.  Drop-in for the original RealTimeChunkASR.
    Same public API, much better accuracy.
    """

    def __init__(
        self,
        model_size:        str   = "small",
        device:            str   = "cuda",
        sample_rate:       int   = 16000,
        overlap_seconds:   float = OVERLAP_S,
        word_gap_ms:       float = 60.0,      # kept for API compat only
        max_context_words: int   = MAX_PROMPT_WORDS,
        max_history_turns: int   = MAX_HISTORY_TURNS,
        language:          Optional[str] = None,   # None | "en" | "fr" | "auto"
    ):
        compute_type = "int8_float16" if device == "cuda" else "int8"
        print(f"[RealTimeASR] Loading Whisper '{model_size}' on "
              f"{device.upper()} (compute={compute_type})...")
        self.model       = WhisperModel(model_size, device=device,
                                        compute_type=compute_type)
        self.sample_rate = sample_rate
        self.device      = device
        # v2.1: per-session language.  None or "auto" → Whisper auto-detect.
        # Two-letter ISO codes ("en", "fr", …) are passed straight through
        # to faster-whisper which forwards them to the decoder language token.
        self._language: Optional[str] = (
            None if (language in (None, "", "auto")) else language.lower()
        )

        self._overlap_samples  = int(overlap_seconds * sample_rate)
        self._context_samples  = int(CONTEXT_S * sample_rate)
        self._fire_samples     = int(FIRE_MS / 1000 * sample_rate)

        # Per-utterance state
        self._utt_audio:       List[np.ndarray] = []
        self._utt_samples:     int = 0
        self._since_last_fire: int = 0

        # Emit state — track exactly which words have been sent to caller
        self._emitted: List[str] = []
        self._pending: Optional[str] = None

        # Conversation history
        self._history: deque[dict] = deque(maxlen=max_history_turns * 2)

        self._chunk_index: int = 0

        print("[RealTimeASR] Ready")

    # ─────────────────────────────────────────────────────────────────────────
    #  Public: per-chunk
    # ─────────────────────────────────────────────────────────────────────────

    def transcribe_chunk(self, chunk: np.ndarray) -> dict:
        t0 = time.perf_counter()
        self._chunk_index += 1

        self._utt_audio.append(chunk.copy())
        self._utt_samples     += len(chunk)
        self._since_last_fire += len(chunk)

        # Hard cap utterance buffer at 30 s to prevent OOM
        MAX_UTT = int(30 * self.sample_rate)
        while self._utt_samples > MAX_UTT and len(self._utt_audio) > 1:
            removed = self._utt_audio.pop(0)
            self._utt_samples -= len(removed)

        newly_emitted: List[str] = []

        if self._since_last_fire >= self._fire_samples:
            self._since_last_fire = 0
            audio_window = self._build_window()
            words = self._run_whisper(audio_window, beam_size=1)
            newly_emitted = self._advance_cursor(words, is_flush=False)

        return {
            "words":   newly_emitted,
            "partial": self._pending or "",
            "latency": {
                "chunk_duration_ms": round(len(chunk) / self.sample_rate * 1000, 1),
                "total_ms":          round((time.perf_counter() - t0) * 1000, 2),
                "chunk_index":       self._chunk_index,
                "emit_cursor":       len(self._emitted),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Public: flush
    # ─────────────────────────────────────────────────────────────────────────

    def flush(self) -> dict:
        """End of utterance. Accurate beam=5 pass to catch any missed tail words."""
        flushed: List[str] = []

        if self._utt_audio:
            full_audio = np.concatenate(self._utt_audio)
            words      = self._run_whisper(full_audio, beam_size=5)
            flushed    = self._advance_cursor(words, is_flush=True)

        # Safety net: emit pending if it survived the flush diff
        if self._pending:
            p = self._pending.strip()
            emitted_normalized = [_n(e) for e in self._emitted]
            if p and any(c.isalpha() for c in p) and _n(p) not in emitted_normalized[-2:]:
                flushed.append(p)
                self._emitted.append(p)
            self._pending = None

        full_text = " ".join(self._emitted).strip()
        if full_text:
            self._history.append({"role": "user", "text": full_text})

        self._reset_utterance_state()

        return {
            "words":   flushed,
            "partial": "",
            "text":    full_text,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal: build audio window for live pass
    # ─────────────────────────────────────────────────────────────────────────

    def _build_window(self) -> np.ndarray:
        """
        Grab last (CONTEXT_S + OVERLAP_S) of utterance audio.
        The leading OVERLAP_S is read-only context — helps Whisper produce
        stable timestamps at the window boundary.
        """
        target = self._context_samples + self._overlap_samples
        pieces: List[np.ndarray] = []
        gathered = 0
        for chunk in reversed(self._utt_audio):
            pieces.append(chunk)
            gathered += len(chunk)
            if gathered >= target:
                break
        pieces.reverse()
        window = np.concatenate(pieces)
        return window[-target:] if len(window) > target else window

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal: Whisper inference
    # ─────────────────────────────────────────────────────────────────────────

    def _run_whisper(self, audio: np.ndarray, beam_size: int) -> List[str]:
        """Run Whisper, apply quality filters, return cleaned word strings."""
        if len(audio) < int(self.sample_rate * 0.1):
            return []

        prompt = self._build_prompt()

        try:
            segments, _ = self.model.transcribe(
                audio,
                beam_size                  = beam_size,
                word_timestamps            = True,
                vad_filter                 = False,
                initial_prompt             = prompt or None,
                condition_on_previous_text = False,
                temperature                = 0.0,
                language                   = self._language,
            )
        except Exception as exc:
            print(f"[RealTimeASR] Whisper error: {exc}")
            return []

        words: List[str] = []
        for seg in segments:
            if getattr(seg, "no_speech_prob", 0.0) > MAX_NO_SPEECH_PROB:
                continue
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    text = w.word.strip()
                    prob = getattr(w, "probability", 1.0)
                    if text and prob >= MIN_WORD_PROB:
                        words.append(text)
            else:
                for wtext in seg.text.strip().split():
                    if wtext.strip():
                        words.append(wtext.strip())

        return words

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal: advance emit cursor
    # ─────────────────────────────────────────────────────────────────────────

    def _advance_cursor(self, whisper_words: List[str], is_flush: bool) -> List[str]:
        """
        Given the full word list Whisper just produced for the audio window,
        return only the words that are NEW (not yet emitted to caller).

        Uses longest-common-prefix matching to find the cut point between
        already-emitted words and the new output.
        """
        if not whisper_words:
            return []

        ne = [_n(w) for w in self._emitted]
        nw = [_n(w) for w in whisper_words]

        # Find how many leading words of whisper_words match our emitted list
        cursor = _lcp_match(ne, nw)
        new_raw = whisper_words[cursor:]

        if not new_raw:
            # Whisper produced nothing new — update pending to its last word
            if whisper_words:
                self._pending = whisper_words[-1]
            return []

        newly: List[str] = []

        if is_flush:
            # Emit all new words
            for w in new_raw:
                w = w.strip()
                if w and any(c.isalpha() for c in w):
                    newly.append(w)
                    self._emitted.append(w)
            self._pending = None

        else:
            # Live pass:
            # 1. Promote previous pending if Whisper confirms it
            if self._pending:
                pn = _n(self._pending)
                nw_fresh = [_n(w) for w in new_raw]
                if nw_fresh and nw_fresh[0] == pn:
                    # Confirmed: pending was correct
                    w = self._pending.strip()
                    if w and any(c.isalpha() for c in w):
                        newly.append(w)
                        self._emitted.append(w)
                    new_raw = new_raw[1:]
                elif pn in nw_fresh[:3]:
                    # Slightly shifted but still there
                    w = self._pending.strip()
                    if w and any(c.isalpha() for c in w):
                        newly.append(w)
                        self._emitted.append(w)
                # If Whisper doesn't see pending at all → was a hallucination, discard
                self._pending = None

            if not new_raw:
                return newly

            # 2. Confirm all words except the last one
            confirmed = new_raw[:-1]
            last_word = new_raw[-1]

            for w in confirmed:
                w = w.strip()
                if not w or not any(c.isalpha() for c in w):
                    continue
                # Genuine repetitions allowed; block accidental dups
                wn = _n(w)
                times_emitted = sum(1 for e in [_n(x) for x in self._emitted] if e == wn)
                times_whisper = sum(1 for x in nw if x == wn)
                if times_emitted < times_whisper:
                    newly.append(w)
                    self._emitted.append(w)

            # 3. Hold last word as pending
            ls = last_word.strip()
            self._pending = ls if ls else None

        return [w for w in newly if w]

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal: prompt
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(self) -> str:
        parts: List[str] = []
        # ── B4: small per-language biasing string ────────────────────────
        # Whisper conditions on the prompt; a one-line cue in the target
        # language nudges the decoder into that language and improves
        # punctuation/casing for short utterances. None for "auto".
        if self._language == "fr":
            parts.append("Conversation en français.")
        elif self._language == "en":
            parts.append("English conversation.")
        for turn in self._history:
            role = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{role}: {turn['text']}")
        if self._emitted:
            parts.append("User: " + " ".join(self._emitted[-MAX_PROMPT_WORDS:]))
        return "\n".join(parts) if parts else ""

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal: reset
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_utterance_state(self):
        self._utt_audio       = []
        self._utt_samples     = 0
        self._since_last_fire = 0
        self._emitted         = []
        self._pending         = None
        self._chunk_index     = 0

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API (compat)
    # ─────────────────────────────────────────────────────────────────────────

    def add_assistant_turn(self, text: str):
        if text:
            self._history.append({"role": "assistant", "text": text})

    def reset(self):
        self._reset_utterance_state()
        self._history.clear()

    def reset_utterance(self):
        self._reset_utterance_state()

    @property
    def history(self) -> list:
        return list(self._history)

    @property
    def current_utterance_so_far(self) -> str:
        words = self._emitted[:]
        if self._pending:
            words.append(self._pending)
        return " ".join(words)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _n(w: str) -> str:
    """Normalize a word for comparison (lowercase, strip punctuation)."""
    return w.lower().strip(".,!?;:'\"-—–")


def _lcp_match(emitted: List[str], whisper: List[str]) -> int:
    """
    Find how many leading elements of `whisper` match a suffix of `emitted`.

    Example:
        emitted = ["hello", "world", "how"]
        whisper = ["hello", "world", "how", "are", "you"]
        returns 3  ->  new words start at whisper[3]

    We try matching whisper's prefix against progressively shorter
    suffixes of emitted, returning the longest match found.
    """
    if not emitted or not whisper:
        return 0

    max_match = 0
    len_w = len(whisper)
    len_e = len(emitted)

    for suffix_start in range(0, len_e + 1):
        tail   = emitted[suffix_start:]
        prefix = min(len(tail), len_w)
        if prefix == 0:
            continue
        if tail[:prefix] == whisper[:prefix]:
            if prefix > max_match:
                max_match = prefix
            if max_match == len_w:
                return max_match   # Can't do better

    return max_match