"""
pipeline.py — STT Pipeline  v5.2  (AEC + TTSVoiceGate + Barge-In)
──────────────────────────────────────────────────────────────────

CHANGES vs v5.1
───────────────
  Added — TTSVoiceGate (acoustic fingerprint-based AI voice suppressor):

    The TTS microservice enrolls its own voice at startup by POSTing PCM
    to  POST /enroll_tts  in main.py.  That audio is forwarded to the
    pipeline via  push_ai_reference().

    NEW: every push_ai_reference() call ALSO feeds TTSVoiceGate.enroll(),
    building a rolling log-mel centroid of the AI voice.

    Processing order per mic chunk:

      1. AECGate       — timing-based suppression (ai_speaking + 1200ms tail)
      2. TTSVoiceGate  — acoustic-fingerprint suppression (cosine similarity)
      3. VAD           — voice activity detection
      4. Barge-in      — fire barge_in event if human voice detected
      5. ASR           — Whisper transcription

    TTSVoiceGate catches echo that slips past the AEC timing window:
      • Reverberant rooms where echo rings > 1200 ms
      • Edge cases where ai_state=False arrives late from gateway
      • The last word of a long TTS sentence (echo tail race)

    Barge-in safety: TTSVoiceGate uses a HIGHER similarity threshold while
    ai_speaking=True, so real human speech overlapping with echo is not
    suppressed.

Flow per chunk
──────────────
  1. AECGate       — if ai_speaking AND NOT barge_in_active → suppress
                     if ai_speaking AND barge_in_active     → pass through
  2. TTSVoiceGate  — if enrolled AND sim >= threshold → suppress
                     (skipped if already suppressed by AECGate)
  3. VAD           — always runs on cleaned signal
  4. Barge-in      — if ai_speaking and prob > barge_in_threshold → emit
  5. ASR           — accumulate voice, fire Whisper
  6. Flush         — on silence_event → drain + segment
"""

from __future__ import annotations

import os, sys, time, numpy as np, logging
from typing import List, Dict, Any, Optional

_HERE = os.path.dirname(__file__)
for _c in [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "agent")]:
    if _c not in sys.path:
        sys.path.insert(0, _c)

from vad import VoiceActivityDetector
from realtime_asr import RealTimeChunkASR
from aec_gate import AECGate
from tts_voice_gate import TTSVoiceGate

logger = logging.getLogger(__name__)


class STTPipeline:

    def __init__(
        self,
        sample_rate: int              = 16000,
        device: str | None            = None,
        # VAD
        idle_threshold: float         = 0.15,
        barge_in_threshold: float     = 0.40,
        vad_pre_gain: float           = 5.0,
        # ASR
        whisper_model_size: str       = "base.en",
        overlap_seconds: float        = 0.8,
        word_gap_ms: float            = 80.0,
        max_context_words: int        = 10,
        max_history_turns: int        = 3,
        asr_min_buffer_ms: float      = 600.0,
        # AEC (timing-based gate)
        enable_aec: bool              = True,
        # TTSVoiceGate (acoustic fingerprint gate)
        enable_voice_gate: bool       = True,
        voice_gate_threshold: float   = 0.70,   # suppress when sim >= this
        voice_gate_barge_in: float    = 0.82,   # stricter threshold during barge-in
        voice_gate_min_frames: int    = 8,      # min enrolled frames before gate activates
        # Barge-in tuning
        barge_in_debounce_frames: int = 2,
        barge_in_cooldown_ms: float   = 1500.0,
        # Compat params (ignored)
        enable_tts_filter: bool       = False,
        tts_sim_threshold: float      = 0.75,
        tts_n_enroll_samples: int     = 3,
        tts_min_enroll_seconds: float = 0.25,
        barge_in_min_words: int       = 1,
        enable_enrollment: bool       = False,
        similarity_threshold: float   = 0.82,
        n_enroll_samples: int         = 1,
        min_enroll_seconds: float     = 0.30,
        ai_detector_model_path        = None,
        ai_detection_threshold        = 0.7,
        enable_ai_filtering           = False,
        barge_in_energy_ratio         = 2.5,
        speculative_vad_threshold     = 0.08,
    ):
        import torch
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.sample_rate           = sample_rate
        self.barge_in_threshold    = barge_in_threshold
        self.barge_in_debounce     = barge_in_debounce_frames
        self.barge_in_cooldown_ms  = barge_in_cooldown_ms

        # State
        self._ai_speaking          = False
        self._last_is_voice        = False
        self._words_this_utterance: List[str] = []

        # Barge-in state
        self._barge_in_active      = False
        self._barge_in_fired_ts    = 0.0
        self._barge_in_voice_frames = 0

        # Latency tracking
        self._utterance_start_ts: Optional[float] = None
        self._latency_history: List[float] = []

        self.vad = VoiceActivityDetector(
            sample_rate        = sample_rate,
            device             = _device,
            idle_threshold     = idle_threshold,
            barge_in_threshold = barge_in_threshold,
            pre_gain           = vad_pre_gain,
        )

        self.realtime_asr = RealTimeChunkASR(
            model_size        = whisper_model_size,
            device            = _device,
            sample_rate       = sample_rate,
            overlap_seconds   = overlap_seconds,
            word_gap_ms       = word_gap_ms,
            max_context_words = max_context_words,
            max_history_turns = max_history_turns,
        )

        # ── Gate 1: AEC timing gate ───────────────────────────────────────────
        self.aec = AECGate(sample_rate=sample_rate) if enable_aec else None

        # ── Gate 2: Acoustic fingerprint voice gate ───────────────────────────
        self.voice_gate = TTSVoiceGate(
            sample_rate         = sample_rate,
            detection_threshold = voice_gate_threshold,
            barge_in_threshold  = voice_gate_barge_in,
            min_enroll_frames   = voice_gate_min_frames,
            enabled             = enable_voice_gate,
        ) if enable_voice_gate else None

        self.tts_filter = None
        self.enrollment = None

        if enable_voice_gate:
            logger.info(
                f"[pipeline] TTSVoiceGate enabled  "
                f"thresh={voice_gate_threshold}  barge_thresh={voice_gate_barge_in}"
            )

    # ── AI / TTS state ────────────────────────────────────────────────────────

    def notify_ai_speaking(self, speaking: bool):
        """
        Call when TTS starts (True) or stops (False).
        Resets barge-in state on every AI turn boundary.
        """
        self._ai_speaking = speaking
        if self.aec:
            if not speaking and self._barge_in_active:
                # User already confirmed barge-in — skip the echo tail so their
                # speech isn't blocked by the 1200ms post-stop suppression window.
                self.aec.stop_immediate()
            else:
                self.aec.set_ai_speaking(speaking)

        if not speaking:
            self._barge_in_active       = False
            self._barge_in_voice_frames = 0
            logger.debug("[pipeline] AI stopped — barge-in reset")

    def push_ai_reference(self, pcm: np.ndarray):
        """
        Feed TTS output PCM to both gates:
          • AECGate      — spectral subtraction reference
          • TTSVoiceGate — builds acoustic fingerprint of AI voice
        """
        if self.aec:
            self.aec.push_reference(pcm)

        # NEW: enroll every TTS frame so the voice gate learns the AI voice
        # continuously throughout the session (handles voice drift / speaker changes)
        if self.voice_gate:
            self.voice_gate.enroll(pcm)

    def add_assistant_turn(self, text: str):
        self.realtime_asr.add_assistant_turn(text)

    # ── Main processing ───────────────────────────────────────────────────────

    def process_chunk(self, audio_chunk: np.ndarray) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        # ── Step 1: AEC timing gate ────────────────────────────────────────
        # Suppress while AI is speaking + 1200ms echo tail.
        # BARGE-IN BYPASS: once barge-in is confirmed, skip AEC entirely so
        # the user's speech reaches VAD/ASR at full amplitude. Without this,
        # the AEC zeros out the mic (spectral subtraction on the TTS reference)
        # making VAD see silence and ASR produce nothing.
        if self.aec and not self._barge_in_active:
            cleaned, aec_suppressed = self.aec.process(audio_chunk)
        else:
            cleaned, aec_suppressed = audio_chunk, False

        # ── Step 2: TTSVoiceGate acoustic fingerprint check ───────────────
        # Only run if the AEC didn't already suppress AND the gate is enrolled.
        # Catches echo that slips past the timing window.
        voice_gate_suppressed = False
        voice_sim             = 0.0

        if not aec_suppressed and self.voice_gate and self.voice_gate.is_ready:
            # Use stricter threshold during barge-in to protect real speech
            voice_gate_suppressed, voice_sim = self.voice_gate.check(
                cleaned, ai_speaking=self._ai_speaking
            )
            if voice_gate_suppressed and not self._barge_in_active:
                logger.debug(
                    f"[pipeline] VoiceGate SUPPRESS  sim={voice_sim:.3f}  "
                    f"ai_speaking={self._ai_speaking}"
                )
                return events   # Silent drop — nothing goes to VAD or ASR

        # Overall suppressed flag (for barge-in check below)
        suppressed = aec_suppressed or voice_gate_suppressed

        # ── Step 3: VAD — always runs on cleaned signal ────────────────────
        # VAD runs even during AEC suppression to detect barge-in.
        # But if voice gate suppressed it (echo after timing window), skip VAD
        # to avoid false voice detections on residual echo.
        if voice_gate_suppressed:
            return events

        audio, is_voice, prob, rms, silence_event = self.vad.process_chunk(
            cleaned, ai_is_speaking=self._ai_speaking
        )

        # ── Step 4: Barge-in detection ─────────────────────────────────────
        if self._ai_speaking:
            if is_voice:
                self._barge_in_voice_frames += 1
            else:
                self._barge_in_voice_frames = 0

            now         = time.monotonic()
            cooldown_ok = (now - self._barge_in_fired_ts) * 1000 > self.barge_in_cooldown_ms
            debounce_ok = self._barge_in_voice_frames >= self.barge_in_debounce
            threshold_ok = prob >= self.barge_in_threshold

            if threshold_ok and debounce_ok and cooldown_ok and not self._barge_in_active:
                self._barge_in_active    = True
                self._barge_in_fired_ts  = now
                self._utterance_start_ts = now
                self._words_this_utterance.clear()
                self.realtime_asr.flush()
                logger.info(
                    f"[pipeline] ⚡ BARGE-IN fired  "
                    f"prob={prob:.3f}  frames={self._barge_in_voice_frames}  "
                    f"voice_sim={voice_sim:.3f}"
                )
                events.append({
                    "type":         "barge_in",
                    "prob":         round(prob, 3),
                    "voice_sim":    round(voice_sim, 3),
                    "words_so_far": len(self._words_this_utterance),
                })

            if not self._barge_in_active:
                return events

        # ── If suppressed by AEC and no barge-in yet → skip ASR ───────────
        if suppressed and not self._barge_in_active:
            return events

        # ── VAD state change event ─────────────────────────────────────────
        if is_voice != self._last_is_voice:
            events.append({
                "type":     "vad",
                "prob":     round(prob, 3),
                "is_voice": is_voice,
                "rms":      round(float(rms), 5),
            })
            self._last_is_voice = is_voice

        # ── Step 5: ASR ────────────────────────────────────────────────────
        if is_voice:
            if self._utterance_start_ts is None:
                self._utterance_start_ts = time.monotonic()

            result = self.realtime_asr.transcribe_chunk(audio)

            partial = result.get("partial", "").strip()
            if partial and hasattr(self.vad, "set_partial_text"):
                self.vad.set_partial_text(partial)

            for word in result.get("words", []):
                w = word if isinstance(word, str) else word.get("word", "")
                w = w.strip().strip(".,!?;:")
                if w and any(c.isalpha() for c in w):
                    self._words_this_utterance.append(w)
                    events.append({"type": "word", "word": w})

            if partial and any(c.isalpha() for c in partial):
                events.append({"type": "partial", "word": partial})

        # ── Step 6: Silence flush ──────────────────────────────────────────
        if silence_event:
            flush_result = self.realtime_asr.flush()

            for word in flush_result.get("words", []):
                w = word if isinstance(word, str) else word.get("word", "")
                w = w.strip().strip(".,!?;:")
                if w and any(c.isalpha() for c in w):
                    self._words_this_utterance.append(w)

            flush_text = flush_result.get("text", "").strip()
            if flush_text and not self._words_this_utterance:
                for w in flush_text.split():
                    w = w.strip().strip(".,!?;:")
                    if w and any(c.isalpha() for c in w):
                        self._words_this_utterance.append(w)

            if self._words_this_utterance:
                text = " ".join(self._words_this_utterance)

                latency_ms: Optional[float] = None
                if self._utterance_start_ts is not None:
                    latency_ms = (time.monotonic() - self._utterance_start_ts) * 1000
                    self._latency_history.append(latency_ms)
                    avg_ms = sum(self._latency_history) / len(self._latency_history)
                    print(
                        f"\n┌─ ASR Latency ({'#'+str(len(self._latency_history))})"
                        f"{'  [BARGE-IN]' if self._barge_in_active else ''}\n"
                        f"│  Text    : {text!r}\n"
                        f"│  Latency : {latency_ms:>7.1f} ms\n"
                        f"│  Average : {avg_ms:>7.1f} ms  "
                        f"({len(self._latency_history)} utterances)\n"
                        f"└{'─'*45}"
                    )
                self._utterance_start_ts = None

                events.append({
                    "type":       "segment",
                    "text":       text,
                    "latency_ms": latency_ms,
                    "barge_in":   self._barge_in_active,
                })
                logger.info(
                    f"[pipeline] segment{'[BARGE]' if self._barge_in_active else ''}: "
                    f"{text!r}  latency={latency_ms:.0f}ms"
                    if latency_ms else
                    f"[pipeline] segment: {text!r}"
                )

                self._barge_in_active       = False
                self._barge_in_voice_frames = 0

            self._words_this_utterance.clear()

        return events

    # ── REST path ─────────────────────────────────────────────────────────────

    def transcribe_full(self, audio: np.ndarray) -> str:
        result  = self.realtime_asr.transcribe_chunk(audio)
        flushed = self.realtime_asr.flush()
        words   = result.get("words", []) + flushed.get("words", [])
        return " ".join(words).strip() or flushed.get("text", "")

    def flush(self) -> str:
        result = self.realtime_asr.flush()
        words_from_live = list(self._words_this_utterance)
        for w in result.get("words", []):
            w = w.strip().strip(".,!?;:")
            if w and any(c.isalpha() for c in w):
                words_from_live.append(w)
        self._words_this_utterance.clear()
        full = result.get("text", "").strip()
        if full:
            return full
        return " ".join(words_from_live).strip()

    def reset(self):
        self.vad.reset()
        self.realtime_asr.reset()
        self._last_is_voice         = False
        self._words_this_utterance.clear()
        self._ai_speaking           = False
        self._barge_in_active       = False
        self._barge_in_voice_frames = 0
        self._barge_in_fired_ts     = 0.0
        self._utterance_start_ts    = None
        if self.aec:
            self.aec.reset()
        # Voice gate: reset counters but KEEP the enrolled voice profile
        # so it still works after a session reconnect without re-enrollment
        if self.voice_gate:
            self.voice_gate.reset()

    def get_stats(self) -> dict:
        stats = {
            "vad":             self.vad.get_state(),
            "barge_in_active": self._barge_in_active,
            "ai_speaking":     self._ai_speaking,
        }
        if self.aec:
            stats["aec"] = self.aec.get_stats()
        if self.voice_gate:
            stats["voice_gate"] = self.voice_gate.get_stats()
        if self._latency_history:
            stats["latency"] = {
                "last_ms": round(self._latency_history[-1], 1),
                "avg_ms":  round(sum(self._latency_history)/len(self._latency_history), 1),
                "min_ms":  round(min(self._latency_history), 1),
                "max_ms":  round(max(self._latency_history), 1),
                "samples": len(self._latency_history),
            }
        return stats