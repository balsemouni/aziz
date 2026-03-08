"""
speaker_enrollment.py  (v9 — TTS Voice Filter)
───────────────────────────────────────────────
NEW DESIGN (v9)
───────────────
Instead of enrolling the USER's voice and allowing only them through,
we now enroll the TTS (AI) voice at startup and BLOCK it from reaching ASR.

Flow
────
  STARTUP  →  TTS says its greeting  →  pipeline.enroll_tts_voice(pcm)
                                              ↓
                                    TTS voice profile locked

  RUNTIME  →  per mic chunk:
    • If chunk matches TTS profile  →  BLOCK  (it's AI audio leaking back)
    • If chunk does NOT match       →  ALLOW  (it's a real human)

Barge-in
────────
  Old: energy-ratio heuristic (noisy, unreliable)
  New: identity check — if the voice is NOT the TTS, it IS a human barge-in.
       This is far more robust because a human barge-in sounds nothing like TTS.

Phase states
────────────
  WAITING   — no TTS profile enrolled yet; all voice passes through to ASR
              (safe default — user can speak before enrollment completes)
  FILTERING — TTS profile locked; each chunk is identity-checked
              match  → suppressed (TTS echo)
              no-match → forwarded to ASR (human)

Backends (same priority chain as v8)
─────────────────────────────────────
  1. SpeechBrain ECAPA-TDNN   pip install speechbrain
  2. resemblyzer               pip install resemblyzer
  3. MFCC+delta²               pip install python_speech_features
  4. spectral (8-dim fallback) — always available

Parameters
──────────
  similarity_threshold  0.75  Cosine sim above this → classified as TTS voice
                              (intentionally lower than v8 user-enrollment
                               because TTS has very consistent timbre)
  anchor_slack          0.12  Hard floor = threshold − slack
  min_filter_seconds    0.20  Minimum audio length to run identity check
                              (shorter chunks are let through — too risky to block)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Backend detection  (ECAPA-TDNN > resemblyzer > MFCC > spectral)
# ─────────────────────────────────────────────────────────────────────────────

BACKEND = "mfcc"  # default — overwritten below

try:
    from speechbrain.pretrained import EncoderClassifier as _SBEncoder
    _sb_model = _SBEncoder.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/sb_ecapa",
        run_opts={"device": "cpu"},
    )
    BACKEND = "ecapa"
    logger.info("[TTSFilter] Backend: SpeechBrain ECAPA-TDNN (192-dim)")
except Exception:
    try:
        from resemblyzer import VoiceEncoder as _VE, preprocess_wav as _pp
        _ve_model = _VE()
        BACKEND = "resemblyzer"
        logger.info("[TTSFilter] Backend: resemblyzer GE2E (256-dim)")
    except Exception:
        try:
            import python_speech_features as _psf
            BACKEND = "mfcc"
            logger.warning(
                "[TTSFilter] Backend: MFCC fallback (39-dim) — "
                "install speechbrain or resemblyzer for best accuracy"
            )
        except Exception:
            BACKEND = "spectral"
            logger.warning(
                "[TTSFilter] Backend: spectral fallback (8-dim) — "
                "accuracy will be lower"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Tunable constants
# ─────────────────────────────────────────────────────────────────────────────

# TTS voices (neural TTS) are extremely consistent — lower threshold still
# catches them reliably while avoiding false-positives on human speech.
_THRESHOLDS = {
    "ecapa":       (0.75, 0.12),
    "resemblyzer": (0.73, 0.12),
    "mfcc":        (0.68, 0.15),
    "spectral":    (0.65, 0.15),
}

TTS_SIM_THRESHOLD, ANCHOR_SLACK = _THRESHOLDS[BACKEND]

# How many short TTS audio chunks to average into the anchor profile
# (more = more stable, but takes longer to lock at startup)
N_ENROLL_SAMPLES    = 3
MIN_ENROLL_SECONDS  = 0.25   # each sample must be at least this long
MIN_FILTER_SECONDS  = 0.20   # don't bother checking chunks shorter than this
RMS_FLOOR           = 0.005  # ignore near-silent chunks during enrollment
MIN_VAD_PROB_ENROLL = 0.40   # TTS produces very clear speech — low threshold ok

N_MFCC   = 13
WINLEN   = 0.025
WINSTEP  = 0.010
SAMPLE_RATE_DEFAULT = 16000


# ─────────────────────────────────────────────────────────────────────────────
#  Embedding extractors  (identical to v8 — just copied for self-containment)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_ecapa(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    import torch
    if sample_rate != 16000:
        audio = _resample(audio, sample_rate, 16000)
    wav_tensor = torch.tensor(audio).unsqueeze(0).float()
    with torch.no_grad():
        emb = _sb_model.encode_batch(wav_tensor)
    return emb.squeeze().numpy().astype(np.float32)


def _embed_resemblyzer(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 16000:
        audio = _resample(audio, sample_rate, 16000)
    wav = _pp(audio, source_sr=16000)
    return _ve_model.embed_utterance(wav).astype(np.float32)


def _embed_mfcc(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    min_len = max(int(sample_rate * 0.20), int(sample_rate * WINLEN * 4))
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))
    mfcc = _psf.mfcc(
        audio, samplerate=sample_rate, winlen=WINLEN, winstep=WINSTEP,
        numcep=N_MFCC, nfilt=26, nfft=512, preemph=0.97,
        ceplifter=22, appendEnergy=True,
    )
    if mfcc.shape[0] < 3:
        return mfcc.mean(axis=0).astype(np.float32)
    delta  = _compute_delta(mfcc)
    delta2 = _compute_delta(delta)
    return np.concatenate([
        mfcc.mean(0), delta.mean(0), delta2.mean(0)
    ]).astype(np.float32)


def _embed_spectral(audio: np.ndarray, _sr: int = None) -> np.ndarray:
    n    = len(audio)
    spec = np.abs(np.fft.rfft(audio))
    bands = np.logspace(np.log10(0.01), np.log10(0.5), 7)
    band_energies = []
    for i in range(len(bands) - 1):
        lo = int(bands[i] * n); hi = int(bands[i + 1] * n)
        band_energies.append(spec[lo:hi].mean() if hi > lo else 0.0)
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
    return np.array(band_energies + [rms, zcr], dtype=np.float32)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_out = int(len(audio) * ratio)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _compute_delta(features: np.ndarray, N: int = 2) -> np.ndarray:
    n_frames, _ = features.shape
    padded = np.pad(features, ((N, N), (0, 0)), mode="edge")
    denom  = 2 * sum(i**2 for i in range(1, N + 1))
    delta  = np.zeros_like(features)
    for t in range(n_frames):
        for n in range(1, N + 1):
            delta[t] += n * (padded[t + N + n] - padded[t + N - n])
    return delta / denom


def _extract_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if BACKEND == "ecapa":
        return _embed_ecapa(audio, sample_rate)
    if BACKEND == "resemblyzer":
        return _embed_resemblyzer(audio, sample_rate)
    if BACKEND == "mfcc":
        return _embed_mfcc(audio, sample_rate)
    return _embed_spectral(audio)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        m = min(len(a), len(b)); a, b = a[:m], b[:m]
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ─────────────────────────────────────────────────────────────────────────────
#  Decision dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TTSFilterDecision:
    """
    Result of evaluating one audio chunk against the TTS voice profile.

    Attributes
    ──────────
    send_to_asr   : True  → human voice, forward to ASR
                    False → TTS voice or not yet enrolled, suppress
    reason        : one of:
                      "waiting"        — TTS profile not yet enrolled
                      "tts_match"      — matches TTS voice → block
                      "human_voice"    — does not match TTS → allow
                      "too_short"      — chunk < min_filter_seconds → allow (safe)
                      "silence"        — not a voice chunk → suppress (normal)
                      "enrolling"      — currently accumulating TTS profile
    similarity    : cosine similarity score (None if not checked)
    tts_enrolled  : whether the TTS profile is locked
    is_barge_in   : True if this is a human speaking while TTS was expected to be speaking
    """
    __slots__ = ("send_to_asr", "reason", "similarity", "tts_enrolled", "is_barge_in")

    def __init__(
        self,
        send_to_asr:  bool,
        reason:       str,
        similarity:   Optional[float] = None,
        tts_enrolled: bool = False,
        is_barge_in:  bool = False,
    ):
        self.send_to_asr  = send_to_asr
        self.reason       = reason
        self.similarity   = similarity
        self.tts_enrolled = tts_enrolled
        self.is_barge_in  = is_barge_in

    def to_dict(self) -> dict:
        return {
            "send_to_asr":  self.send_to_asr,
            "reason":       self.reason,
            "similarity":   round(self.similarity, 3) if self.similarity is not None else None,
            "tts_enrolled": self.tts_enrolled,
            "is_barge_in":  self.is_barge_in,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class TTSVoiceFilter:
    """
    Filters out TTS (AI) voice from the microphone stream.

    Usage
    ─────
        filt = TTSVoiceFilter(sample_rate=16000)

        # At startup — feed TTS greeting audio to build the TTS profile:
        filt.enroll_tts_audio(pcm_chunk)   # call for each chunk of TTS greeting
        filt.commit_tts_enrollment()        # call when TTS finishes speaking

        # OR, feed it chunk-by-chunk while ai_is_playing=True and it
        # auto-commits when enough audio is accumulated (simpler integration).

        # Per mic chunk:
        decision = filt.evaluate(mic_chunk, is_voice=True, ai_is_speaking=False)
        if decision.send_to_asr:
            # forward to ASR
    """

    def __init__(
        self,
        sample_rate:          int   = SAMPLE_RATE_DEFAULT,
        similarity_threshold: float = TTS_SIM_THRESHOLD,
        anchor_slack:         float = ANCHOR_SLACK,
        n_enroll_samples:     int   = N_ENROLL_SAMPLES,
        min_enroll_seconds:   float = MIN_ENROLL_SECONDS,
        min_filter_seconds:   float = MIN_FILTER_SECONDS,
    ):
        self.sample_rate          = sample_rate
        self.similarity_threshold = similarity_threshold
        self.anchor_threshold     = max(0.0, similarity_threshold - anchor_slack)
        self.n_enroll_samples     = n_enroll_samples
        self.min_enroll_seconds   = min_enroll_seconds
        self.min_filter_seconds   = min_filter_seconds

        # Profile state
        self._locked:            bool                 = False
        self._anchor_profile:    Optional[np.ndarray] = None

        # Enrollment buffers
        self._enroll_buffer:     List[np.ndarray]     = []
        self._enroll_audio_sec:  float                = 0.0
        self._enroll_embeddings: List[np.ndarray]     = []

        # Stats
        self.chunks_blocked:   int   = 0
        self.chunks_allowed:   int   = 0
        self.last_similarity:  float = 0.0

        logger.info(
            f"[TTSFilter] v9 initialized  backend={BACKEND}  "
            f"sim_threshold={similarity_threshold:.2f}  "
            f"anchor_threshold={self.anchor_threshold:.2f}  "
            f"n_enroll_samples={n_enroll_samples}"
        )

    # ── TTS Enrollment API ────────────────────────────────────────────────────

    def enroll_tts_audio(self, pcm_chunk: np.ndarray):
        """
        Feed raw TTS output audio to build the voice profile.

        Call this for every chunk of audio that the TTS produces during its
        greeting utterance.  The filter accumulates these and builds the
        anchor profile once enough audio is collected.

        Auto-commits when n_enroll_samples worth of audio is accumulated
        (you can also call commit_tts_enrollment() explicitly).
        """
        if self._locked:
            return  # Already enrolled — ignore further feed

        chunk = pcm_chunk.astype(np.float32)
        rms   = float(np.sqrt(np.mean(chunk ** 2) + 1e-9))

        if rms < RMS_FLOOR:
            return  # Skip near-silent chunks (pauses between words)

        chunk_sec = len(chunk) / self.sample_rate
        self._enroll_buffer.append(chunk.copy())
        self._enroll_audio_sec += chunk_sec

        logger.debug(
            f"[TTSFilter] enroll buffer  total={self._enroll_audio_sec:.2f}s"
        )

        # Auto-commit when we have a full utterance segment
        if self._enroll_audio_sec >= self.min_enroll_seconds:
            self._commit_utterance()
            # Check if we've collected enough samples to lock
            if len(self._enroll_embeddings) >= self.n_enroll_samples:
                self._lock_profile()

    def commit_tts_enrollment(self):
        """
        Explicitly signal that the TTS has finished speaking its greeting.
        Commits any buffered audio and locks the profile if enough was collected.
        Call this from the ai_state=False control message handler.
        """
        if self._locked:
            return
        if self._enroll_audio_sec >= self.min_enroll_seconds * 0.5:
            # Accept shorter final segment
            self._commit_utterance(min_sec_override=self.min_enroll_seconds * 0.5)
        if self._enroll_embeddings:
            self._lock_profile()
        elif self._enroll_buffer:
            logger.warning(
                "[TTSFilter] TTS finished but not enough audio to enroll "
                f"({self._enroll_audio_sec:.2f}s collected) — filter stays in WAITING mode"
            )

    def _commit_utterance(self, min_sec_override: Optional[float] = None):
        """Extract embedding from current buffer and add to enroll_embeddings."""
        min_sec = min_sec_override if min_sec_override is not None else self.min_enroll_seconds
        if self._enroll_audio_sec < min_sec or not self._enroll_buffer:
            return

        audio = np.concatenate(self._enroll_buffer)
        emb   = _l2_normalize(_extract_embedding(audio, self.sample_rate))
        self._enroll_embeddings.append(emb)

        logger.info(
            f"[TTSFilter] TTS utterance {len(self._enroll_embeddings)}/{self.n_enroll_samples} "
            f"enrolled ({self._enroll_audio_sec:.2f}s)"
        )
        self._enroll_buffer.clear()
        self._enroll_audio_sec = 0.0

    def _lock_profile(self):
        if self._locked or not self._enroll_embeddings:
            return
        anchor = _l2_normalize(
            np.mean(self._enroll_embeddings, axis=0).astype(np.float32)
        )
        self._anchor_profile = anchor
        self._locked         = True
        logger.info(
            f"[TTSFilter] ✅ TTS voice profile LOCKED  "
            f"backend={BACKEND}  feat_dim={anchor.shape[0]}  "
            f"sim_threshold={self.similarity_threshold:.2f}  "
            f"samples={len(self._enroll_embeddings)}"
        )

    # ── Main evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        audio_chunk:   np.ndarray,
        is_voice:      bool,
        ai_is_speaking: bool = False,
        vad_prob:      float = 0.0,
    ) -> TTSFilterDecision:
        """
        Evaluate one mic chunk.

        Parameters
        ──────────
        audio_chunk    : mic audio (float32), any chunk size
        is_voice       : VAD decision
        ai_is_speaking : True when TTS is currently playing audio
        vad_prob       : VAD confidence [0, 1]

        Returns
        ───────
        TTSFilterDecision with send_to_asr=True for human voice
        """
        # ── Not a voice chunk ─────────────────────────────────────────────
        if not is_voice:
            return TTSFilterDecision(
                send_to_asr=False,
                reason="silence",
                tts_enrolled=self._locked,
            )

        # ── TTS profile not yet enrolled → let all voice through ──────────
        # This is the safe default: before we know what TTS sounds like,
        # we don't block anything (user can speak immediately).
        if not self._locked:
            return TTSFilterDecision(
                send_to_asr=True,
                reason="waiting",
                tts_enrolled=False,
            )

        # ── Chunk too short to reliably identify ──────────────────────────
        chunk_sec = len(audio_chunk) / self.sample_rate
        if chunk_sec < self.min_filter_seconds:
            self.chunks_allowed += 1
            return TTSFilterDecision(
                send_to_asr=True,
                reason="too_short",
                tts_enrolled=True,
            )

        # ── Identity check ────────────────────────────────────────────────
        try:
            embedding  = _extract_embedding(audio_chunk.astype(np.float32), self.sample_rate)
            similarity = _cosine_similarity(self._anchor_profile, embedding)
            self.last_similarity = similarity
        except Exception as exc:
            logger.warning(f"[TTSFilter] embedding failed: {exc} — allowing chunk through")
            self.chunks_allowed += 1
            return TTSFilterDecision(
                send_to_asr=True,
                reason="human_voice",
                tts_enrolled=True,
            )

        # Matches TTS voice → BLOCK
        if similarity >= self.anchor_threshold:
            self.chunks_blocked += 1
            logger.debug(
                f"[TTSFilter] BLOCKED  sim={similarity:.3f} >= {self.anchor_threshold:.3f}  "
                f"(TTS voice detected)"
            )
            return TTSFilterDecision(
                send_to_asr=False,
                reason="tts_match",
                similarity=similarity,
                tts_enrolled=True,
                is_barge_in=False,
            )

        # Does NOT match TTS → ALLOW (human barge-in or normal speech)
        # is_barge_in is True when the human speaks while TTS is supposedly playing
        is_barge_in = ai_is_speaking
        self.chunks_allowed += 1
        logger.debug(
            f"[TTSFilter] ALLOWED  sim={similarity:.3f} < {self.anchor_threshold:.3f}  "
            f"barge_in={is_barge_in}"
        )
        return TTSFilterDecision(
            send_to_asr=True,
            reason="human_voice",
            similarity=similarity,
            tts_enrolled=True,
            is_barge_in=is_barge_in,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @property
    def is_enrolled(self) -> bool:
        return self._locked

    @property
    def enrollment_progress(self) -> float:
        if self._locked:
            return 1.0
        sample_prog = len(self._enroll_embeddings) / max(self.n_enroll_samples, 1)
        partial     = min(self._enroll_audio_sec / max(self.min_enroll_seconds, 0.01), 1.0) / max(self.n_enroll_samples, 1)
        return min(max(sample_prog, partial), 0.99)

    def reset(self):
        self._locked            = False
        self._anchor_profile    = None
        self._enroll_buffer.clear()
        self._enroll_audio_sec  = 0.0
        self._enroll_embeddings.clear()
        self.chunks_blocked     = 0
        self.chunks_allowed     = 0
        self.last_similarity    = 0.0
        logger.info("[TTSFilter] Reset — ready for new TTS enrollment")

    def get_stats(self) -> dict:
        total = self.chunks_blocked + self.chunks_allowed
        return {
            "tts_enrolled":       self._locked,
            "backend":            BACKEND,
            "enroll_progress":    round(self.enrollment_progress, 2),
            "enroll_samples":     len(self._enroll_embeddings),
            "enroll_needed":      self.n_enroll_samples,
            "last_similarity":    round(self.last_similarity, 3),
            "chunks_blocked":     self.chunks_blocked,
            "chunks_allowed":     self.chunks_allowed,
            "block_rate":         round(self.chunks_blocked / max(total, 1), 3),
            "sim_threshold":      self.similarity_threshold,
            "anchor_threshold":   round(self.anchor_threshold, 3),
            "feat_dim": (
                192 if BACKEND == "ecapa"
                else 256 if BACKEND == "resemblyzer"
                else 39  if BACKEND == "mfcc"
                else 8
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Backwards-compat alias so old imports don't crash
# ─────────────────────────────────────────────────────────────────────────────

# Old code that does:  from speaker_enrollment import SpeakerEnrollmentService, EnrollmentDecision
# will still work — they just get the new class + a shim decision type.

class EnrollmentDecision(TTSFilterDecision):
    """Backwards-compat shim."""
    pass

SpeakerEnrollmentService = TTSVoiceFilter