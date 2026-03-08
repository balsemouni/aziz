"""
aec_gate.py  v2.0
─────────────────
Software-level Acoustic Echo Cancellation Gate

FIXES vs v1
───────────
  Fix 1 — Extended POST_STOP_BUFFER_MS (300 → 1200 ms)
      TTS echo rings in the mic for much longer than 300 ms, especially
      on speakers. Raising the tail to 1200 ms eliminates the "STT catches
      the last word of TTS" problem without hurting real barge-in (which is
      handled by identity, not timing).

  Fix 2 — Full suppression while AI speaking (no reference needed)
      Old code required a reference signal to suppress during the grace period.
      Without reference it still let audio through after GRACE_PERIOD_MS.
      New code: while ai_speaking=True → ALWAYS suppress regardless of reference.
      Reference is now ONLY used for the optional spectral-subtraction cleanup
      on non-suppressed chunks (outside the gate window).

  Fix 3 — Barge-in gate removed from AEC
      Barge-in is now detected purely by TTSVoiceFilter identity check in
      pipeline.py.  AEC no longer tries to "allow loud barge-ins" based on
      energy ratio — that was unreliable and caused echo leakage.
      The AEC gate now has one job: suppress while AI is speaking + tail.

  Fix 4 — reset() clears speaking state
      If the session resets mid-utterance the gate no longer stays open.
"""

from __future__ import annotations

import numpy as np
import time
import logging
from collections import deque
from typing import Optional

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

GRACE_PERIOD_MS      = 200    # Short grace before first AI word reaches mic
POST_STOP_BUFFER_MS  = 1200   # FIX: was 300 — cover full echo tail after TTS stops
SPECTRAL_ALPHA       = 0.85   # Spectral subtraction strength (outside gate window)
REFERENCE_QUEUE_MS   = 3000   # How much reference audio to keep


class AECGate:
    """
    Acoustic Echo Cancellation Gate.

    Two modes depending on whether a reference signal is available:

      WITHOUT reference  (most deployments):
        • While AI speaking + POST_STOP_BUFFER_MS tail → SUPPRESS everything.
        • Outside that window → PASS through unchanged.

      WITH reference (TTS forwards PCM back):
        • Same suppression window.
        • Outside window → light spectral subtraction for residual cleanup.

    Barge-in is NOT handled here.  Identity-based barge-in lives in
    pipeline.py / TTSVoiceFilter.  This gate's only job is echo suppression.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate      = sample_rate
        self._ai_speaking     = False
        self._ai_started_at: Optional[float]  = None
        self._ai_stopped_at: Optional[float]  = None

        _ref_samples = int(sample_rate * REFERENCE_QUEUE_MS / 1000)
        self._reference_buffer: deque = deque(maxlen=_ref_samples)
        self._has_reference = False

        self.chunks_suppressed = 0
        self.chunks_processed  = 0

    # ── AI state control ─────────────────────────────────────────────────────

    def set_ai_speaking(self, speaking: bool):
        if speaking and not self._ai_speaking:
            self._ai_speaking   = True
            self._ai_started_at = time.monotonic()
            self._ai_stopped_at = None
            logger.debug("[AEC] AI speaking — gate ACTIVE")

        elif not speaking and self._ai_speaking:
            self._ai_speaking   = False
            self._ai_stopped_at = time.monotonic()
            logger.debug(f"[AEC] AI stopped — echo tail {POST_STOP_BUFFER_MS}ms")

    # ── Reference signal ─────────────────────────────────────────────────────

    def push_reference(self, pcm_chunk: np.ndarray):
        self._reference_buffer.extend(pcm_chunk.astype(np.float32).tolist())
        self._has_reference = True

    # ── Main processing ───────────────────────────────────────────────────────

    def process(
        self,
        mic_chunk: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Process one mic chunk.

        Returns (cleaned_audio, suppressed).
        If suppressed=True do NOT forward to VAD/ASR.
        """
        self.chunks_processed += 1
        cleaned = mic_chunk.astype(np.float32)

        # ── Gate: suppress while AI is speaking + echo tail ───────────────
        # FIX: full suppression, no energy-ratio barge-in bypass.
        # Barge-in is handled by TTSVoiceFilter identity check.
        if self._ai_speaking or self._in_echo_tail():
            # Optional: spectral subtract to clean audio for downstream
            # (useful if caller still wants the cleaned signal for logging)
            if self._has_reference:
                cleaned = self._spectral_subtract(cleaned)
            self.chunks_suppressed += 1
            return cleaned, True   # SUPPRESSED

        # ── Outside gate window: light cleanup only ────────────────────────
        if self._has_reference:
            cleaned = self._spectral_subtract(cleaned, strength=0.25)

        return cleaned, False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _in_echo_tail(self) -> bool:
        """True for POST_STOP_BUFFER_MS after AI stops speaking."""
        if self._ai_speaking or self._ai_stopped_at is None:
            return False
        elapsed_ms = (time.monotonic() - self._ai_stopped_at) * 1000
        return elapsed_ms < POST_STOP_BUFFER_MS

    def _spectral_subtract(
        self,
        audio: np.ndarray,
        strength: float = SPECTRAL_ALPHA,
    ) -> np.ndarray:
        if not self._reference_buffer:
            return audio

        n = len(audio)
        ref_arr = np.array(list(self._reference_buffer)[-n:], dtype=np.float32)
        if len(ref_arr) < n:
            ref_arr = np.pad(ref_arr, (0, n - len(ref_arr)))

        if _TORCH_AVAILABLE and _torch.cuda.is_available():
            mic_t = _torch.from_numpy(audio).cuda()
            ref_t = _torch.from_numpy(ref_arr[:n]).cuda()
            mic_fft    = _torch.fft.rfft(mic_t)
            ref_fft    = _torch.fft.rfft(ref_t)
            mic_mag    = mic_fft.abs()
            ref_mag    = ref_fft.abs()
            mic_phase  = _torch.angle(mic_fft)
            sub_mag    = _torch.clamp(mic_mag - strength * ref_mag, min=0.0)
            result_fft = sub_mag * _torch.exp(1j * mic_phase)
            return _torch.fft.irfft(result_fft, n=n).cpu().numpy().astype(np.float32)

        mic_fft = np.fft.rfft(audio)
        ref_fft = np.fft.rfft(ref_arr[:n])
        mic_mag  = np.abs(mic_fft)
        ref_mag  = np.abs(ref_fft)
        mic_phase = np.angle(mic_fft)
        sub_mag = np.maximum(mic_mag - strength * ref_mag, 0.0)
        result_fft = sub_mag * np.exp(1j * mic_phase)
        return np.fft.irfft(result_fft, n=n).astype(np.float32)

    def get_stats(self) -> dict:
        return {
            "ai_speaking":        self._ai_speaking,
            "in_echo_tail":       self._in_echo_tail(),
            "has_reference":      self._has_reference,
            "chunks_processed":   self.chunks_processed,
            "chunks_suppressed":  self.chunks_suppressed,
            "suppression_rate":   round(
                self.chunks_suppressed / max(self.chunks_processed, 1), 3
            ),
            "post_stop_buffer_ms": POST_STOP_BUFFER_MS,
        }

    def reset(self):
        self._ai_speaking    = False
        self._ai_started_at  = None
        self._ai_stopped_at  = None
        self._reference_buffer.clear()
        self._has_reference  = False
        self.chunks_suppressed = 0
        self.chunks_processed  = 0