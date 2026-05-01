"""
tts_voice_gate.py  v1.0
────────────────────────
Acoustic-fingerprint-based TTS Voice Gate

HOW IT WORKS
────────────
  1.  ENROLLMENT  (background, at startup)
      The TTS microservice synthesizes a fixed phrase and POSTs the PCM to
      STT's /enroll_tts.  main.py already forwards that audio to the pipeline
      via  pipeline.push_ai_reference().

      TTSVoiceGate intercepts every push_ai_reference() call and builds a
      rolling spectral fingerprint of the AI voice:
        • Split the reference audio into 32 ms frames.
        • Compute log-mel-spectrogram (40 bands) for each frame.
        • Average across all enrolled frames → one "AI voice centroid" vector.

  2.  DETECTION  (per mic chunk, real-time)
      For every mic chunk that was NOT already suppressed by the timing gate:
        • Compute the same log-mel feature for the mic chunk.
        • Cosine-similarity between mic features and the AI centroid.
        • If similarity >= threshold  →  SUPPRESS  (return suppressed=True).

  3.  INTEGRATION
      TTSVoiceGate is a thin wrapper.  It lives inside STTPipeline alongside
      AECGate.  The processing order in pipeline.process_chunk() is:

        AECGate   →  TTSVoiceGate  →  VAD  →  ASR

      AECGate handles the easy cases (AI definitely speaking / echo tail).
      TTSVoiceGate catches residual echo that slips past the timing window:
        • Very reverberant rooms (echo tail > 1200 ms).
        • Quiet user speech that stops before TTS finishes.
        • Edge cases where ai_state=False arrives late.

  4.  BARGE-IN SAFETY
      When a real human voice interrupts (barge-in), the mic signal is a MIX
      of human + TTS echo.  To avoid suppressing a real barge-in:
        • Similarity must exceed a HIGHER threshold when ai_speaking=True
          (barge_in_threshold > detection_threshold).
        • The similarity score is also available for logging / tuning.

TUNING
──────
  detection_threshold   = 0.70   # lower = more aggressive echo killing
  barge_in_threshold    = 0.82   # higher = safer during barge-in
  min_enroll_frames     = 8      # enroll at least 256ms before activating

DEPENDENCIES
────────────
  numpy  (already in requirements)
  No torch required — pure numpy FFT is fast enough for 32ms frames.
"""

from __future__ import annotations

import logging
import numpy as np
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

FRAME_MS            = 32      # Analysis frame size
N_MELS              = 40      # Mel bands
CENTROID_ALPHA      = 0.05    # EMA update weight for live centroid updates
                              #   (each new reference frame shifts the centroid
                              #    5% toward it — slow drift to track voice changes)
MAX_ENROLL_FRAMES   = 500     # Cap stored frames to avoid unbounded memory


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Build a (n_mels, n_fft//2+1) mel filterbank matrix."""
    low_mel  = _hz_to_mel(80.0)
    high_mel = _hz_to_mel(sr / 2.0)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = np.array([_mel_to_hz(m) for m in mel_pts])
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m       = bin_pts[m]
        f_m_plus  = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fb


class TTSVoiceGate:
    """
    Acoustic fingerprint gate.

    Usage
    ─────
      gate = TTSVoiceGate(sample_rate=16000)

      # During enrollment (called by pipeline.push_ai_reference):
      gate.enroll(pcm_chunk)          # can be called many times

      # Per mic chunk in pipeline.process_chunk:
      suppressed, sim = gate.check(mic_chunk, ai_speaking=False)
      if suppressed:
          return []   # skip VAD + ASR
    """

    def __init__(
        self,
        sample_rate: int       = 16000,
        detection_threshold: float = 0.70,   # sim >= this → suppress
        barge_in_threshold:  float = 0.82,   # higher bar while ai_speaking
        min_enroll_frames:   int   = 8,      # min frames before gate activates
        enabled:             bool  = True,
    ):
        self.sample_rate          = sample_rate
        self.detection_threshold  = detection_threshold
        self.barge_in_threshold   = barge_in_threshold
        self.min_enroll_frames    = min_enroll_frames
        self.enabled              = enabled

        self._frame_samples = int(sample_rate * FRAME_MS / 1000)   # 512 @ 16kHz
        self._n_fft         = self._frame_samples
        self._filterbank    = _mel_filterbank(N_MELS, self._n_fft, sample_rate)

        # Centroid: running mean of enrolled mel features
        self._centroid:   Optional[np.ndarray] = None  # shape (N_MELS,)
        self._n_enrolled: int = 0
        self._is_ready:   bool = False

        # Stats
        self.chunks_checked    = 0
        self.chunks_suppressed = 0
        self.last_similarity   = 0.0

        logger.info(
            f"[TTSVoiceGate] init  sr={sample_rate}  "
            f"thresh={detection_threshold}  barge_thresh={barge_in_threshold}"
        )

    # ── Feature extraction ────────────────────────────────────────────────────

    def _log_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mean log-mel feature vector for an audio chunk.
        Returns shape (N_MELS,).
        """
        if len(audio) == 0:
            return np.zeros(N_MELS, dtype=np.float32)

        # Split into frames, compute power spectrum per frame
        frames = []
        for start in range(0, len(audio) - self._frame_samples + 1, self._frame_samples // 2):
            frame = audio[start : start + self._frame_samples].astype(np.float32)
            if len(frame) < self._frame_samples:
                frame = np.pad(frame, (0, self._frame_samples - len(frame)))
            window     = np.hanning(self._frame_samples).astype(np.float32)
            windowed   = frame * window
            spectrum   = np.abs(np.fft.rfft(windowed, n=self._n_fft)) ** 2
            mel_energy = self._filterbank @ spectrum
            log_mel    = np.log(mel_energy + 1e-8)
            frames.append(log_mel)

        if not frames:
            # Audio shorter than one frame — process as single frame
            frame    = np.pad(audio.astype(np.float32),
                              (0, max(0, self._frame_samples - len(audio))))
            window   = np.hanning(self._frame_samples).astype(np.float32)
            spectrum = np.abs(np.fft.rfft(frame[:self._frame_samples] * window,
                                          n=self._n_fft)) ** 2
            mel_energy = self._filterbank @ spectrum
            return np.log(mel_energy + 1e-8).astype(np.float32)

        return np.mean(frames, axis=0).astype(np.float32)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors, in [-1, 1]."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ── Enrollment ────────────────────────────────────────────────────────────

    def enroll(self, pcm_chunk: np.ndarray):
        """
        Feed a chunk of TTS reference audio to build the AI voice fingerprint.
        Called by pipeline.push_ai_reference() for every TTS frame.
        Safe to call from any thread.
        """
        if not self.enabled or len(pcm_chunk) == 0:
            return

        feat = self._log_mel(pcm_chunk)

        if self._centroid is None:
            # First frame — initialise centroid
            self._centroid   = feat.copy()
            self._n_enrolled = 1
        elif self._n_enrolled < self.min_enroll_frames:
            # Running mean during initial enrollment (equal weight)
            n = self._n_enrolled + 1
            self._centroid = (self._centroid * self._n_enrolled + feat) / n
            self._n_enrolled = n
        else:
            # EMA update — slow drift to follow voice over the session
            self._centroid = (
                (1.0 - CENTROID_ALPHA) * self._centroid + CENTROID_ALPHA * feat
            )
            self._n_enrolled = min(self._n_enrolled + 1, MAX_ENROLL_FRAMES)

        # Gate becomes active once we have enough frames
        if not self._is_ready and self._n_enrolled >= self.min_enroll_frames:
            self._is_ready = True
            logger.info(
                f"[TTSVoiceGate] ✅ READY after {self._n_enrolled} frames "
                f"({self._n_enrolled * FRAME_MS}ms of reference audio)"
            )

    # ── Detection ─────────────────────────────────────────────────────────────

    def check(
        self,
        mic_chunk:   np.ndarray,
        ai_speaking: bool = False,
    ) -> tuple[bool, float]:
        """
        Check if mic_chunk sounds like the enrolled AI voice.

        Returns
        ───────
        (suppressed, similarity)
          suppressed  — True if chunk should NOT be forwarded to VAD/ASR
          similarity  — cosine similarity score [0, 1] for logging/debug
        """
        if not self.enabled or not self._is_ready or len(mic_chunk) == 0:
            return False, 0.0

        self.chunks_checked += 1

        mic_feat = self._log_mel(mic_chunk)
        sim      = max(0.0, self._cosine_sim(mic_feat, self._centroid))
        self.last_similarity = sim

        # Choose threshold: be stricter during barge-in to avoid silencing
        # real human speech that overlaps with echo
        threshold = self.barge_in_threshold if ai_speaking else self.detection_threshold

        suppressed = sim >= threshold
        if suppressed:
            self.chunks_suppressed += 1
            logger.debug(
                f"[TTSVoiceGate] SUPPRESS  sim={sim:.3f}  thresh={threshold:.2f}  "
                f"ai_speaking={ai_speaking}"
            )

        return suppressed, sim

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_stats(self) -> dict:
        return {
            "enabled":          self.enabled,
            "is_ready":         self._is_ready,
            "enrolled_frames":  self._n_enrolled,
            "enrolled_ms":      self._n_enrolled * FRAME_MS,
            "chunks_checked":   self.chunks_checked,
            "chunks_suppressed": self.chunks_suppressed,
            "suppression_rate": round(
                self.chunks_suppressed / max(self.chunks_checked, 1), 3
            ),
            "last_similarity":  round(self.last_similarity, 4),
            "detection_threshold":  self.detection_threshold,
            "barge_in_threshold":   self.barge_in_threshold,
        }

    def reset(self):
        """Reset detection counters (does NOT wipe the enrolled voice profile)."""
        self.chunks_checked    = 0
        self.chunks_suppressed = 0
        self.last_similarity   = 0.0
        logger.debug("[TTSVoiceGate] counters reset (voice profile kept)")

    def full_reset(self):
        """Wipe the entire voice profile (use when TTS speaker changes)."""
        self._centroid    = None
        self._n_enrolled  = 0
        self._is_ready    = False
        self.reset()
        logger.info("[TTSVoiceGate] full reset — voice profile cleared")
