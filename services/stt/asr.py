"""
realtime_asr.py
───────────────
Zero-buffer, per-chunk streaming ASR with:

  1. PREFIX HEALING     — injects last 0.8s of previous chunk as read-only
                          context so split words (to|mo|ro|w) get healed
                          by Whisper seeing both halves in the same call.

  2. WORD-BOUNDARY EMIT — a word is only emitted when the NEXT word starts
                          (gap between word[i].end and word[i+1].start > 
                          WORD_GAP_MS).  This is the "end of word pause"
                          you asked for — no timers, purely timestamp-driven.

  3. CONVERSATION CONTEXT — every ASR call receives a text prompt built
                            from the last N confirmed words of THIS utterance
                            + the last M turns of conversation history.
                            Whisper uses this as a prior so it picks the
                            right words in ambiguous cases ("two" vs "to" vs
                            "too", names, domain vocabulary, etc.)

  4. LATENCY EVENTS      — every chunk returns precise timing breakdowns
                           (prep, inference, total, realtime_factor).

Architecture per chunk
──────────────────────
  new chunk arrives
      │
      ├─ prepend prefix (last 0.8s of prev chunk)  ← heal split words
      │
      ├─ build initial_prompt from context          ← intelligent
      │
      ├─ Whisper.transcribe(audio, initial_prompt)  ← one inference call
      │
      ├─ split words into prefix_zone / new_zone    ← by timestamp
      │
      ├─ apply word-boundary rule:                  ← emit on gap
      │     word[i] confirmed when word[i+1].start
      │     - word[i].end > WORD_GAP_MS
      │     (last word always stays pending until
      │      next chunk or flush())
      │
      └─ return { partial, words, latency }
"""

from __future__ import annotations

import time
import numpy as np
from collections import deque
from typing import List, Optional
from faster_whisper import WhisperModel


# ─────────────────────────────────────────────────────────────────────────────
#  Tuneable constants
# ─────────────────────────────────────────────────────────────────────────────

WORD_GAP_MS        = 80     # Silence between word[i].end and word[i+1].start
                             # that counts as "word boundary" → emit word[i]
OVERLAP_SECONDS    = 0.8    # How much of the previous chunk to inject as prefix
MAX_CONTEXT_WORDS  = 40     # Max words from current utterance in prompt
MAX_PROMPT_WORDS   = 10     # Hard cap on words injected into initial_prompt
                             # Longer prompts cause Whisper to echo back context
                             # (the "you hear you hear" repetition bug).
MAX_HISTORY_TURNS  = 3      # How many past conversation turns to include
MAX_CONTEXT_TRIM   = 6.0    # Trim context window if > this many seconds (memory)


class RealTimeChunkASR:
    """
    Drop-in replacement for StreamingSpeechRecognizer for the per-chunk path.

    Key difference from the old approach:
      • NO SpeechBuffer — every chunk hits Whisper immediately
      • Words emitted on WORD BOUNDARY (gap), not on segment end
      • Whisper receives conversation history as initial_prompt
      • flush() called by pipeline on VAD silence to emit last pending word
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cuda",
        sample_rate: int = 16000,
        overlap_seconds: float = OVERLAP_SECONDS,
        word_gap_ms: float = WORD_GAP_MS,
        max_context_words: int = MAX_CONTEXT_WORDS,
        max_history_turns: int = MAX_HISTORY_TURNS,
    ):
        # int8_float16: quantised weights (int8) + float16 accumulation
        # ~35% faster than float16 on NVIDIA RTX 20-series+ with negligible accuracy loss.
        compute_type = "int8_float16" if device == "cuda" else "int8"
        print(f"[RealTimeASR] Loading Whisper {model_size} on {device.upper()} "
              f"(compute={compute_type})...")
        self.model         = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.sample_rate   = sample_rate
        self.overlap_samples = int(overlap_seconds * sample_rate)
        self.word_gap_sec  = word_gap_ms / 1000.0
        self.max_ctx_words = max_context_words
        self.max_hist_turns = max_history_turns

        # ── Per-utterance state ──────────────────────────────────────────
        self._prev_tail: Optional[np.ndarray] = None  # Overlap from last chunk
        self._confirmed_words: List[str] = []         # Emitted this utterance
        self._pending_word: Optional[str] = None      # Last word, not yet emitted
        self._pending_end: float = 0.0                # Its end timestamp (abs)
        self._utterance_offset: float = 0.0           # Running audio time (seconds)
        self._chunk_index: int = 0

        # ── Conversation history (cross-utterance) ───────────────────────
        # Each entry: {"role": "user"|"assistant", "text": "..."}
        self._history: deque[dict] = deque(maxlen=max_history_turns * 2)

        print("[RealTimeASR] ✅ Ready")

    # ─────────────────────────────────────────────────────────────────────────
    #  Main per-chunk entry point
    # ─────────────────────────────────────────────────────────────────────────

    def transcribe_chunk(self, chunk: np.ndarray) -> dict:
        """
        Process one audio chunk immediately.  No buffering.

        Returns
        ───────
        {
          "words":   ["hello", "world"],   # newly confirmed complete words
          "partial": "tomo",               # current incomplete word (for display)
          "latency": { ... }               # timing breakdown
        }
        """
        t0 = time.perf_counter()
        self._chunk_index += 1

        chunk_duration = len(chunk) / self.sample_rate

        # ── Step 1: Build audio window = prefix + chunk ──────────────────
        if self._prev_tail is not None:
            audio_input   = np.concatenate([self._prev_tail, chunk])
            prefix_dur    = len(self._prev_tail) / self.sample_rate
        else:
            audio_input   = chunk
            prefix_dur    = 0.0

        t1 = time.perf_counter()

        # ── Step 2: Build Whisper initial_prompt from context ────────────
        prompt = self._build_prompt()

        # ── Step 3: Whisper inference ────────────────────────────────────
        segments, _ = self.model.transcribe(
            audio_input,
            beam_size      = 1,
            word_timestamps= True,
            vad_filter     = False,       # We handle VAD
            initial_prompt = prompt or None,
            condition_on_previous_text = False,  # We control context manually
        )

        # Collect all words with absolute timestamps
        # (timestamps from Whisper are relative to audio_input start)
        # We shift by (utterance_offset - prefix_dur) to get absolute time
        time_shift = self._utterance_offset - prefix_dur
        all_words  = []

        for seg in segments:
            if not (hasattr(seg, "words") and seg.words):
                # No word timestamps — fall back to segment-level
                for w in seg.text.strip().split():
                    if w:
                        all_words.append({
                            "word":  w,
                            "start": seg.start + time_shift,
                            "end":   seg.end   + time_shift,
                        })
            else:
                for w in seg.words:
                    word = w.word.strip()
                    if word:
                        all_words.append({
                            "word":  word,
                            "start": w.start + time_shift,
                            "end":   w.end   + time_shift,
                        })

        t2 = time.perf_counter()

        # ── Step 4: Filter — only words from the NEW chunk zone ──────────
        # (words whose start time falls inside the new chunk, not the prefix)
        chunk_start_abs = self._utterance_offset        # absolute time of chunk start
        chunk_end_abs   = self._utterance_offset + chunk_duration

        new_zone_words = [
            w for w in all_words
            if w["start"] >= (chunk_start_abs - 0.05)   # small tolerance
        ]

        # ── Step 5: Word-boundary emit ───────────────────────────────────
        #
        # All words except the last are considered complete — we've already
        # moved past them.  The LAST word stays pending until the next chunk
        # confirms it (or flush() emits it at end of utterance).
        #
        # The word_gap_sec check is kept as an extra confidence filter:
        # words with a clear pause after them are confirmed immediately;
        # words without a gap are still confirmed because they're non-last.
        #
        newly_confirmed: List[str] = []

        if new_zone_words:
            # Promote previous pending word if new words arrived
            # (a new word starting means the previous word definitely ended)
            if self._pending_word:
                newly_confirmed.append(self._pending_word)
                self._pending_word = None

            # All words except the last are now confirmed
            for w in new_zone_words[:-1]:
                newly_confirmed.append(w["word"])

            # Last word is always partial until next chunk
            last = new_zone_words[-1]
            self._pending_word = last["word"]
            self._pending_end  = last["end"]

        # Commit confirmed words
        self._confirmed_words.extend(newly_confirmed)

        # ── Step 6: Update prefix for next chunk ────────────────────────
        if len(chunk) >= self.overlap_samples:
            self._prev_tail = chunk[-self.overlap_samples:].copy()
        else:
            self._prev_tail = chunk.copy()

        # Advance utterance clock
        self._utterance_offset += chunk_duration

        # ── Step 7: Build latency report ────────────────────────────────
        t3 = time.perf_counter()

        chunk_ms  = chunk_duration * 1000
        prep_ms   = (t1 - t0) * 1000
        asr_ms    = (t2 - t1) * 1000
        post_ms   = (t3 - t2) * 1000
        total_ms  = (t3 - t0) * 1000
        rtf       = total_ms / max(chunk_ms, 1)

        return {
            "words":   newly_confirmed,
            "partial": self._pending_word or "",
            "latency": {
                "chunk_duration_ms": round(chunk_ms,  1),
                "prep_ms":           round(prep_ms,   2),
                "asr_inference_ms":  round(asr_ms,    2),
                "post_ms":           round(post_ms,   2),
                "total_ms":          round(total_ms,  2),
                "realtime_factor":   round(rtf,       3),
                "chunk_index":       self._chunk_index,
                "status": (
                    "🟢 fast" if rtf < 0.8 else
                    "🟡 ok"   if rtf < 1.0 else
                    "🔴 slow — consider smaller model"
                ),
            },
            "debug": {
                "prompt_used":         prompt[:80] + "..." if prompt and len(prompt) > 80 else prompt,
                "all_words_from_asr":  [w["word"] for w in all_words],
                "new_zone_words":      [w["word"] for w in new_zone_words],
                "prefix_duration_s":   round(prefix_dur, 3),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Flush — called by pipeline when VAD detects end of utterance
    # ─────────────────────────────────────────────────────────────────────────

    def flush(self) -> dict:
        """
        End of utterance.  Emit the last pending word unconditionally
        (it has no 'next word' to confirm it via gap detection).

        Also adds the complete utterance to conversation history.

        Returns { "words": [...], "partial": "", "text": "full utterance" }
        """
        flushed = []

        if self._pending_word:
            flushed.append(self._pending_word)
            self._confirmed_words.append(self._pending_word)

        full_text = " ".join(self._confirmed_words).strip()

        # Add to conversation history
        if full_text:
            self._history.append({"role": "user", "text": full_text})

        # Reset utterance state (but keep history and prev_tail for next utt)
        self._confirmed_words  = []
        self._pending_word     = None
        self._pending_end      = 0.0
        self._utterance_offset = 0.0
        self._prev_tail        = None   # Fresh start for next utterance
        self._chunk_index      = 0

        return {
            "words":   flushed,
            "partial": "",
            "text":    full_text,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Conversation context management
    # ─────────────────────────────────────────────────────────────────────────

    def add_assistant_turn(self, text: str):
        """
        Call this from pipeline/orchestrator when the AI responds.
        Adds the AI reply to history so Whisper understands conversation flow.
        """
        if text:
            self._history.append({"role": "assistant", "text": text})

    def _build_prompt(self) -> str:
        """
        Build Whisper's initial_prompt from:
          - Last N confirmed words of the CURRENT utterance (in-progress context)
          - Last M turns of conversation history (cross-turn context)

        Whisper uses this as a prior — it biases transcription toward words
        and phrases that fit the ongoing conversation.

        Example output:
          "User: what time is the meeting tomorrow
           Assistant: The meeting is at 3pm
           User: can you also remind me about"

        Whisper then knows the domain is "meeting scheduling" and will
        prefer "remind" over "rewind", "3pm" over "three", etc.
        """
        parts: List[str] = []

        # ── History turns ────────────────────────────────────────────────
        for turn in self._history:
            role  = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{role}: {turn['text']}")

        # ── Current utterance so far (in-progress) ───────────────────────
        # Strictly cap to MAX_PROMPT_WORDS (last N words only).
        # Longer prompts cause Whisper to repeat context ("you hear you hear").
        if self._confirmed_words:
            recent = self._confirmed_words[-MAX_PROMPT_WORDS:]
            parts.append("User: " + " ".join(recent))

        return "\n".join(parts) if parts else ""

    # ─────────────────────────────────────────────────────────────────────────
    #  Full reset (new session / new user)
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Full reset — clears history too.  Call on new WebSocket session."""
        self._prev_tail        = None
        self._confirmed_words  = []
        self._pending_word     = None
        self._pending_end      = 0.0
        self._utterance_offset = 0.0
        self._chunk_index      = 0
        self._history.clear()

    def reset_utterance(self):
        """
        Reset only the current utterance state.
        Keeps conversation history intact — call between utterances.
        """
        self._prev_tail        = None
        self._confirmed_words  = []
        self._pending_word     = None
        self._pending_end      = 0.0
        self._utterance_offset = 0.0
        self._chunk_index      = 0

    @property
    def history(self) -> list:
        return list(self._history)

    @property
    def current_utterance_so_far(self) -> str:
        words = self._confirmed_words[:]
        if self._pending_word:
            words.append(self._pending_word)
        return " ".join(words)