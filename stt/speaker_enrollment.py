"""
speaker_enrollment.py  (v7 — First-Sentence Enrollment)
─────────────────────────────────────────────────────────
Design goal
───────────
Build the speaker profile from the user's FIRST COMPLETE SENTENCE (i.e. all
voice chunks collected until the first silence_event), then lock ASR to that
voice permanently.  Any subsequent speech that does not match that voiceprint
is silently rejected — ASR never sees it.

How it works
────────────
Phase 1 — COLLECTING (not yet enrolled)
  • Every voice chunk with rms ≥ RMS_FLOOR is appended to an internal buffer.
  • ASR is NOT fed during this phase (send_to_asr=False).
  • The pipeline calls evaluate(..., silence_event=True) when the first pause
    is detected.  At that point:
      - All buffered chunks are concatenated into one audio segment.
      - _extract_mfcc_full() is called on the full segment → robust 39-dim
        centroid that spans many phonemes (unlike v6's single-chunk anchor).
      - _lock_profile() is called and the buffer is cleared.
      - The decision has send_to_asr=False and enrolled=True so the pipeline
        can replay the pre-enroll buffer into ASR.

Phase 2 — ENROLLED (profile locked)
  • Each incoming voice chunk goes through the standard two-layer similarity
    check (anchor + adaptive).
  • Accepted chunks are forwarded to ASR (send_to_asr=True).
  • Rejected chunks fire reason="speaker_mismatch" and block ASR.
  • The adaptive profile drifts slowly so the system adjusts to the speaker
    over the session (EMA with adapt_rate=0.05).

Integration with pipeline.py (no changes needed in pipeline)
─────────────────────────────────────────────────────────────
  pipeline.py already handles the just_locked transition:
    - It accumulates voice chunks in _pre_enroll_buffer during Phase 1.
    - On the first decision with enrolled=True it replays those chunks into ASR.
  The only change you must make in pipeline.py is to pass silence_event into
  evaluate():

    decision = self.enrollment.evaluate(
        audio, is_voice=is_voice, silence_event=silence_event
    )

  (The old signature had no silence_event arg — add it as shown above.)

Parameters you can tune in __init__
────────────────────────────────────
  similarity_threshold  float  0.72   Main gate (raised vs v6 because the anchor
                                      now covers a full sentence → better centroid)
  anchor_slack          float  0.20   anchor_threshold = 0.72 − 0.20 = 0.52
  adapt_rate            float  0.05   EMA blend rate for adaptive profile
  adapt_min_chunks      int    5      Start adapting after 5 accepted chunks
  min_enroll_chunks     int    3      Need at least 3 voice chunks before locking
                                      (guard against locking on a single cough)

Feature vector layout  (shape 39)
──────────────────────────────────
  [  0:13 ]  mean MFCCs    (c0–c12)
  [ 13:26 ]  mean deltas
  [ 26:39 ]  mean delta²
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import python_speech_features as psf
    PSF_AVAILABLE = True
except ImportError:
    PSF_AVAILABLE = False
    logger.warning("python_speech_features not installed — using spectral fallback")


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.72   # Raised vs v6: full-sentence centroid is more reliable
ANCHOR_SLACK         = 0.20   # anchor_threshold = 0.72 − 0.20 = 0.52
ADAPT_RATE           = 0.05
ADAPT_MIN_CHUNKS     = 5
RMS_FLOOR            = 0.020  # Real speech RMS is 0.02+; noise is typically below this
MIN_ENROLL_CHUNKS    = 8      # Need at least 8 real voice chunks (~500ms at 16kHz/chunk)
MIN_ENROLL_SECONDS   = 1.5    # Must collect at least 1.5s of qualifying speech
MIN_VAD_PROB_ENROLL  = 0.50   # VAD confidence must be >= 0.50 to count during enrollment
MIN_CONSECUTIVE_VOICE= 2      # Chunk must follow at least 1 other voice chunk (anti-spike)

N_MFCC               = 13
WINLEN               = 0.025
WINSTEP              = 0.010


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extractors
# ─────────────────────────────────────────────────────────────────────────────

def _extract_mfcc_full(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Returns shape (39,) = mean(MFCC) + mean(delta) + mean(delta²).

    Works on any length — pads to at least 200ms so even a short chunk
    produces a valid embedding.  For full-sentence audio this is not needed
    but the guard is kept for robustness.
    """
    min_len = max(int(sample_rate * 0.20), int(sample_rate * WINLEN * 4))
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    mfcc = psf.mfcc(
        audio,
        samplerate   = sample_rate,
        winlen       = WINLEN,
        winstep      = WINSTEP,
        numcep       = N_MFCC,
        nfilt        = 26,
        nfft         = 512,
        preemph      = 0.97,
        ceplifter    = 22,
        appendEnergy = True,
    )

    if mfcc.shape[0] < 3:
        return mfcc.mean(axis=0).astype(np.float32)

    delta  = _compute_delta(mfcc)
    delta2 = _compute_delta(delta)

    feat = np.concatenate([
        mfcc.mean(axis=0),
        delta.mean(axis=0),
        delta2.mean(axis=0),
    ]).astype(np.float32)

    return feat


def _compute_delta(features: np.ndarray, N: int = 2) -> np.ndarray:
    n_frames, n_feats = features.shape
    padded = np.pad(features, ((N, N), (0, 0)), mode="edge")
    denom  = 2 * sum(i**2 for i in range(1, N + 1))
    delta  = np.zeros_like(features)
    for t in range(n_frames):
        for n in range(1, N + 1):
            delta[t] += n * (padded[t + N + n] - padded[t + N - n])
    return delta / denom


def _extract_spectral(audio: np.ndarray) -> np.ndarray:
    """
    Returns shape (8,) — richer spectral fallback when psf is not installed.
    """
    n     = len(audio)
    spec  = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n)

    bands = np.logspace(np.log10(0.01), np.log10(0.5), 7)
    band_energies = []
    for i in range(len(bands) - 1):
        lo = int(bands[i]     * n)
        hi = int(bands[i + 1] * n)
        if hi > lo:
            band_energies.append(spec[lo:hi].mean())
        else:
            band_energies.append(0.0)

    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    return np.array(band_energies + [rms, zcr], dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _extract(audio: np.ndarray, sample_rate: int, use_mfcc: bool) -> np.ndarray:
    if use_mfcc:
        return _extract_mfcc_full(audio, sample_rate)
    return _extract_spectral(audio)


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerEnrollmentService:
    """
    First-sentence speaker enrollment (v7).

    Phase 1 — COLLECTING:
      Accumulates voice chunks until the first silence_event.
      ASR is blocked during this phase.
      On silence_event the full first sentence is used to build the anchor profile.

    Phase 2 — ENROLLED:
      Two-layer similarity check (anchor + adaptive) on every incoming chunk.
      Only matching chunks are forwarded to ASR.
    """

    def __init__(
        self,
        sample_rate: int            = 16000,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        anchor_slack: float         = ANCHOR_SLACK,
        adapt_rate: float           = ADAPT_RATE,
        adapt_min_chunks: int       = ADAPT_MIN_CHUNKS,
        min_enroll_chunks: int      = MIN_ENROLL_CHUNKS,
        min_enroll_seconds: float   = MIN_ENROLL_SECONDS,
        min_vad_prob: float         = MIN_VAD_PROB_ENROLL,
        min_consecutive_voice: int  = MIN_CONSECUTIVE_VOICE,
    ):
        self.sample_rate          = sample_rate
        self.similarity_threshold = similarity_threshold
        self.anchor_threshold     = max(0.0, similarity_threshold - anchor_slack)
        self.adapt_rate           = adapt_rate
        self.adapt_min_chunks     = adapt_min_chunks
        self.min_enroll_chunks       = min_enroll_chunks
        self.min_enroll_seconds      = min_enroll_seconds
        self.min_vad_prob             = min_vad_prob
        self.min_consecutive_voice   = min_consecutive_voice

        self._locked: bool                         = False
        self._use_mfcc: bool                       = PSF_AVAILABLE

        self._anchor_profile: Optional[np.ndarray]   = None
        self._adaptive_profile: Optional[np.ndarray] = None

        # ── Phase 1 buffer: voice chunks for first sentence ────────────────
        self._enroll_buffer: List[np.ndarray] = []
        self._enroll_chunk_count: int         = 0   # qualifying voice chunks seen
        self._enroll_audio_seconds: float     = 0.0 # total qualifying speech seconds
        self._consec_voice_count: int         = 0   # consecutive voice chunks (anti-spike)

        # ── Stats ──────────────────────────────────────────────────────────
        self.enrolled_seconds: float = 0.0
        self.last_similarity:  float = 1.0
        self.last_anchor_sim:  float = 1.0
        self.chunks_accepted:  int   = 0
        self.chunks_rejected:  int   = 0
        self._adapt_count:     int   = 0

        logger.info(
            f"[Enrollment] v7 initialized — FIRST-SENTENCE mode  "
            f"adaptive_threshold={similarity_threshold:.2f}  "
            f"anchor_threshold={self.anchor_threshold:.2f}  "
            f"min_enroll_chunks={min_enroll_chunks}  "
            f"extractor={'mfcc+delta+delta2' if self._use_mfcc else 'spectral'}"
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_enrolled(self) -> bool:
        return self._locked and self._anchor_profile is not None

    @property
    def enrollment_progress(self) -> float:
        if self.is_enrolled:
            return 1.0
        # Progress = max of chunk-based and seconds-based progress
        if self.min_enroll_chunks == 0 or self.min_enroll_seconds == 0:
            return 0.0
        chunk_progress   = self._enroll_chunk_count / self.min_enroll_chunks
        seconds_progress = self._enroll_audio_seconds / self.min_enroll_seconds
        return min(max(chunk_progress, seconds_progress), 0.99)

    # ── Main entry point ──────────────────────────────────────────────────────

    def evaluate(
        self,
        audio_chunk: np.ndarray,
        is_voice: bool,
        silence_event: bool = False,
        vad_prob: float = 0.0,
    ) -> "EnrollmentDecision":
        """
        Evaluate one audio chunk.

        Parameters
        ──────────
        audio_chunk    : raw mic audio (float32 or int16)
        is_voice       : VAD decision for this chunk
        silence_event  : True on the FIRST silent chunk after a voice segment
                         (pass this from pipeline.py — it is the signal to lock
                          the profile at the end of the first sentence)

        Returns
        ───────
        EnrollmentDecision with send_to_asr, reason, similarity, enrolled, progress
        """
        if len(audio_chunk) == 0:
            return EnrollmentDecision(send_to_asr=False, reason="empty_chunk")

        audio_chunk = audio_chunk.astype(np.float32)

        # ── Phase 1: COLLECTING — accumulate first sentence ──────────────────
        if not self.is_enrolled:
            # Track consecutive voice for anti-spike gate
            if is_voice:
                self._consec_voice_count += 1
            else:
                self._consec_voice_count = 0

            # Accumulate only REAL voice chunks — not noise spikes
            # A chunk must pass ALL four gates to count:
            #   1. VAD says is_voice=True
            #   2. VAD probability >= min_vad_prob (0.50) — not just marginal
            #   3. RMS >= RMS_FLOOR (0.020) — real speech energy
            #   4. At least min_consecutive_voice consecutive voice chunks — anti-spike
            if is_voice and vad_prob >= self.min_vad_prob and self._consec_voice_count >= self.min_consecutive_voice:
                rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                if rms >= RMS_FLOOR:
                    chunk_seconds = len(audio_chunk) / self.sample_rate
                    self._enroll_buffer.append(audio_chunk.copy())
                    self._enroll_chunk_count  += 1
                    self._enroll_audio_seconds += chunk_seconds
                    logger.debug(
                        f"[Enrollment] collecting chunk {self._enroll_chunk_count}  "
                        f"rms={rms:.4f}  vad_prob={vad_prob:.2f}  "
                        f"total={self._enroll_audio_seconds:.2f}s"
                    )

            # On silence — if not enough real speech yet, reset buffer to avoid
            # enrolling on fragmented noise across multiple small gaps.
            # Only lock if we have BOTH enough chunks AND enough seconds.
            if silence_event:
                has_enough = (
                    self._enroll_chunk_count >= self.min_enroll_chunks
                    and self._enroll_audio_seconds >= self.min_enroll_seconds
                )
                if not has_enough and self._enroll_chunk_count > 0:
                    logger.debug(
                        f"[Enrollment] silence_event — insufficient real speech "
                        f"({self._enroll_chunk_count} chunks / {self._enroll_audio_seconds:.2f}s), "
                        f"need {self.min_enroll_chunks} chunks / {self.min_enroll_seconds}s — resetting buffer"
                    )
                    self._enroll_buffer.clear()
                    self._enroll_chunk_count   = 0
                    self._enroll_audio_seconds = 0.0

            # Lock on silence_event IF we have enough chunks AND enough seconds
            if silence_event and self._enroll_chunk_count >= self.min_enroll_chunks and self._enroll_audio_seconds >= self.min_enroll_seconds:
                full_audio = np.concatenate(self._enroll_buffer)
                embedding  = _extract(full_audio, self.sample_rate, self._use_mfcc)
                self._lock_profile(embedding, len(full_audio))
                self._enroll_buffer.clear()

                logger.info(
                    f"[Enrollment] ✅ FIRST-SENTENCE LOCK — "
                    f"{self.enrolled_seconds*1000:.0f}ms of speech  "
                    f"chunks={self._enroll_chunk_count}  "
                    f"feat_dim={embedding.shape[0]}"
                )

                # Return enrolled=True, send_to_asr=False so pipeline replays
                # pre-enroll buffer into ASR (existing pipeline.py logic handles this)
                return EnrollmentDecision(
                    send_to_asr = False,
                    reason      = "enrolled_first_sentence",
                    similarity  = 1.0,
                    enrolled    = True,
                    progress    = 1.0,
                )

            # Still collecting — silence event but not enough chunks yet
            if silence_event and self._enroll_chunk_count > 0:
                logger.debug(
                    f"[Enrollment] silence_event but only {self._enroll_chunk_count} chunks "
                    f"(need {self.min_enroll_chunks}) — waiting for more speech"
                )

            return EnrollmentDecision(
                send_to_asr = False,
                reason      = "collecting_first_sentence",
                similarity  = None,
                enrolled    = False,
                progress    = self.enrollment_progress,
            )

        # ── Phase 2: ENROLLED — two-layer similarity check ───────────────────
        if not is_voice:
            return EnrollmentDecision(
                send_to_asr = False,
                reason      = "silence",
                enrolled    = True,
                progress    = 1.0,
            )

        embedding = _extract(audio_chunk, self.sample_rate, self._use_mfcc)

        # Layer 1: Anchor check (hard floor — never moves)
        anchor_sim = _cosine_similarity(self._anchor_profile, embedding)
        self.last_anchor_sim = anchor_sim

        if anchor_sim < self.anchor_threshold:
            self.chunks_rejected += 1
            logger.debug(
                f"[Enrollment] REJECTED by ANCHOR "
                f"anchor_sim={anchor_sim:.3f} < {self.anchor_threshold:.3f}"
            )
            return EnrollmentDecision(
                send_to_asr = False,
                reason      = "speaker_mismatch",
                similarity  = anchor_sim,
                enrolled    = True,
                progress    = 1.0,
            )

        # Layer 2: Adaptive check
        adaptive_sim = _cosine_similarity(self._adaptive_profile, embedding)
        self.last_similarity = adaptive_sim

        if adaptive_sim < self.similarity_threshold:
            self.chunks_rejected += 1
            logger.debug(
                f"[Enrollment] REJECTED by ADAPTIVE "
                f"adaptive_sim={adaptive_sim:.3f} < {self.similarity_threshold:.3f}  "
                f"anchor_sim={anchor_sim:.3f}"
            )
            return EnrollmentDecision(
                send_to_asr = False,
                reason      = "speaker_mismatch",
                similarity  = adaptive_sim,
                enrolled    = True,
                progress    = 1.0,
            )

        # ACCEPTED — blend into adaptive profile
        self.chunks_accepted += 1

        if self.chunks_accepted >= self.adapt_min_chunks:
            self._adapt_profile(embedding)

        logger.debug(
            f"[Enrollment] ACCEPTED  "
            f"adaptive={adaptive_sim:.3f}  anchor={anchor_sim:.3f}  "
            f"adapt_count={self._adapt_count}"
        )

        return EnrollmentDecision(
            send_to_asr = True,
            reason      = "speaker_match",
            similarity  = adaptive_sim,
            enrolled    = True,
            progress    = 1.0,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _lock_profile(self, embedding: np.ndarray, n_samples: int):
        """Lock the anchor and adaptive profiles from a full-sentence embedding."""
        if self._locked:
            return

        self._locked = True

        profile = embedding.astype(np.float32).copy()
        norm = np.linalg.norm(profile)
        if norm > 1e-9:
            profile /= norm

        self._anchor_profile   = profile.copy()
        self._adaptive_profile = profile.copy()
        self.enrolled_seconds  = n_samples / self.sample_rate

        logger.info(
            f"[Enrollment] Profile LOCKED (first-sentence) — "
            f"{self.enrolled_seconds*1000:.0f}ms  "
            f"feat_dim={profile.shape[0]}  "
            f"extractor={'mfcc+delta+delta2' if self._use_mfcc else 'spectral'}  "
            f"anchor_threshold={self.anchor_threshold:.2f}  "
            f"adaptive_threshold={self.similarity_threshold:.2f}"
        )

    def _adapt_profile(self, embedding: np.ndarray):
        """EMA blend into the adaptive profile. Anchor is NEVER touched."""
        self._adaptive_profile = (
            (1.0 - self.adapt_rate) * self._adaptive_profile
            + self.adapt_rate * embedding
        ).astype(np.float32)

        norm = np.linalg.norm(self._adaptive_profile)
        if norm > 1e-9:
            self._adaptive_profile /= norm

        self._adapt_count += 1

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self):
        self._anchor_profile   = None
        self._adaptive_profile = None
        self._locked           = False
        self._enroll_buffer.clear()
        self._enroll_chunk_count   = 0
        self._enroll_audio_seconds = 0.0
        self._consec_voice_count   = 0
        self.enrolled_seconds  = 0.0
        self.last_similarity   = 1.0
        self.last_anchor_sim   = 1.0
        self.chunks_accepted   = 0
        self.chunks_rejected   = 0
        self._adapt_count      = 0
        logger.info("[Enrollment] Reset — ready for new first-sentence enrollment")

    def get_stats(self) -> dict:
        return {
            "enrolled":             self.is_enrolled,
            "enrolled_seconds":     round(self.enrolled_seconds, 2),
            "progress":             round(self.enrollment_progress, 2),
            "enroll_chunks_so_far": self._enroll_chunk_count,
            "enroll_seconds_so_far": round(self._enroll_audio_seconds, 2),
            "last_similarity":      round(self.last_similarity, 3),
            "last_anchor_sim":      round(self.last_anchor_sim, 3),
            "chunks_accepted":      self.chunks_accepted,
            "chunks_rejected":      self.chunks_rejected,
            "adapt_count":          self._adapt_count,
            "adaptive_threshold":   self.similarity_threshold,
            "anchor_threshold":     round(self.anchor_threshold, 3),
            "feat_dim": (
                39 if (self._use_mfcc and self._anchor_profile is not None
                       and len(self._anchor_profile) == 39)
                else 13 if self._use_mfcc else 8
            ),
            "extractor": "mfcc+delta+delta2" if self._use_mfcc else "spectral",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Decision dataclass  (API unchanged — pipeline.py compatible)
# ─────────────────────────────────────────────────────────────────────────────

class EnrollmentDecision:
    __slots__ = ("send_to_asr", "reason", "similarity", "enrolled", "progress")

    def __init__(
        self,
        send_to_asr: bool,
        reason: str,
        similarity: Optional[float] = None,
        enrolled: bool  = False,
        progress: float = 0.0,
    ):
        self.send_to_asr = send_to_asr
        self.reason      = reason
        self.similarity  = similarity
        self.enrolled    = enrolled
        self.progress    = progress

    def to_dict(self) -> dict:
        return {
            "send_to_asr": self.send_to_asr,
            "reason":      self.reason,
            "similarity":  round(self.similarity, 3) if self.similarity is not None else None,
            "enrolled":    self.enrolled,
            "progress":    round(self.progress, 2),
        }