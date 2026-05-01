"""
vad.py — downloads Silero VAD from GitHub releases (direct URL, no auth needed)
"""

import os
import torch
import numpy as np
from threading import Thread
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from agc import SimpleAGC


# ─────────────────────────────────────────────────────────────────────────────
#  Download Silero VAD .jit file once, cache it locally
# ─────────────────────────────────────────────────────────────────────────────

_SILERO_URL   = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit"
_CACHE_DIR    = os.path.join(os.path.expanduser("~"), ".cache", "silero_vad")
_CACHE_FILE   = os.path.join(_CACHE_DIR, "silero_vad.jit")


def _get_silero_model_path() -> str:
    if os.path.exists(_CACHE_FILE):
        print(f"✅ Silero VAD loaded from cache: {_CACHE_FILE}")
        return _CACHE_FILE

    os.makedirs(_CACHE_DIR, exist_ok=True)
    print(f"📥 Downloading Silero VAD from GitHub releases...")

    import urllib.request
    try:
        urllib.request.urlretrieve(_SILERO_URL, _CACHE_FILE)
        print(f"✅ Silero VAD cached at: {_CACHE_FILE}")
        return _CACHE_FILE
    except Exception as e:
        # Clean up partial download
        if os.path.exists(_CACHE_FILE):
            os.remove(_CACHE_FILE)
        raise RuntimeError(f"Failed to download Silero VAD: {e}")


# Download once at import time
_SILERO_MODEL_PATH = _get_silero_model_path()


def _load_silero(device: str):
    model = torch.jit.load(_SILERO_MODEL_PATH, map_location=device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────

class VoiceActivityDetector:

    def __init__(
        self,
        sample_rate=16000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        idle_threshold=0.15,        # ↓ was 0.25 — more sensitive for quiet mics
        barge_in_threshold=0.40,    # ↓ was 0.50
        min_rms=0.001,              # ↓ was 0.005 — don't gate on volume alone
        pre_gain=5.0,               # 5× boost before AGC — lifts rms=0.002→0.010
                                    # before AGC amplifies further to target_rms=0.05
        silence_limit_ms=800,
        sentence_end_silence_ms=200,
        min_chunk_samples=512,
    ):
        self.sample_rate            = sample_rate
        self.device                 = device
        self.min_chunk_samples      = min_chunk_samples

        if torch.cuda.is_available() and device == "cpu":
            print("⚡ GPU available! Forcing CUDA")
            self.device = "cuda"

        print(f"🎯 Loading Silero VAD on {self.device.upper()}...")
        try:
            self.vad_model = _load_silero(self.device)

            if self.device == "cuda":
                self.vad_model = torch.jit.optimize_for_inference(self.vad_model)
                dummy = torch.randn(512, device=self.device)
                with torch.no_grad():
                    self.vad_model(dummy, self.sample_rate)
                torch.cuda.synchronize()
                print("✅ GPU warmed up")

            print(f"✅ VAD loaded on {self.device.upper()}")

        except Exception as e:
            print(f"❌ Failed to load VAD: {e}")
            raise

        self.agc = SimpleAGC(
            target_rms = 0.08,   # raised: gives Silero a stronger signal (was 0.05)
            max_gain   = 80.0,   # raised: allows up to 80× for very quiet mics
                                 #   raw rms=0.002 × pre_gain=15 = 0.030
                                 #   AGC gain = 0.08/0.030 = 2.7×  →  final ≈ 0.08
                                 #   well within Silero's reliable range (>0.05)
        )

        self.idle_threshold           = idle_threshold
        self.barge_in_threshold       = barge_in_threshold
        self.min_rms                  = min_rms
        self.pre_gain                 = pre_gain
        self.silence_limit_ms         = silence_limit_ms
        self.sentence_end_silence_ms  = sentence_end_silence_ms

        # Variable tail tracking
        self._last_partial_text: str  = ""    # updated by pipeline via set_partial_text()
        self._silence_frames: int     = 0     # consecutive silent frames
        self._total_voice_frames: int = 0     # voice frames in current utterance (debounce)

        self.last_vad_prob       = 0.0
        self.consecutive_voice   = 0
        self.consecutive_silence = 0
        self._was_voice          = False

        self.thread_pool      = ThreadPoolExecutor(max_workers=2, thread_name_prefix="VAD")
        self.vad_queue        = Queue(maxsize=20)  # large enough to never drop chunks
        self.vad_result_queue = Queue(maxsize=4)

        if self.device == "cuda":
            self.pinned_buffer = torch.zeros(self.min_chunk_samples, dtype=torch.float32).pin_memory()
        else:
            self.pinned_buffer = None

        self.gpu_tensor = torch.zeros(self.min_chunk_samples, dtype=torch.float32, device=self.device)

        self.running    = True
        self.vad_thread = Thread(target=self._vad_worker, daemon=True)
        self.vad_thread.start()

        print("✅ ASYNC VAD ready (BUFFER-FREE mode)")

    def _vad_worker(self):
        """
        Accumulate incoming audio until we have min_chunk_samples (512) of
        REAL speech, then run Silero on that window.

        Zero-padding short chunks was the root cause of vad_prob staying near
        zero: 320 samples of speech + 192 zeros = Silero sees diluted energy
        and outputs ~0.05 instead of ~0.80.
        """
        accumulator = np.zeros(0, dtype=np.float32)

        while self.running:
            try:
                audio_chunk = self.vad_queue.get(timeout=0.1)
                accumulator = np.concatenate([accumulator, audio_chunk])

                # Process as many full windows as we have accumulated
                while len(accumulator) >= self.min_chunk_samples:
                    vad_chunk   = accumulator[:self.min_chunk_samples]
                    accumulator = accumulator[self.min_chunk_samples:]

                    if self.device == "cuda" and self.pinned_buffer is not None:
                        self.pinned_buffer.copy_(torch.from_numpy(vad_chunk))
                        self.gpu_tensor.copy_(self.pinned_buffer, non_blocking=True)
                    else:
                        self.gpu_tensor.copy_(torch.from_numpy(vad_chunk).to(self.device))

                    with torch.no_grad():
                        prob = self.vad_model(self.gpu_tensor, self.sample_rate).item()

                    try:
                        self.vad_result_queue.get_nowait()
                    except Empty:
                        pass
                    self.vad_result_queue.put(prob)

            except Empty:
                continue
            except Exception as e:
                print(f"❌ VAD worker error: {e}")

    @staticmethod
    def rms(audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32)) + 1e-10))

    def process_chunk(self, audio_chunk: np.ndarray, ai_is_speaking: bool = False):
        if len(audio_chunk) == 0:
            return audio_chunk, False, 0.0, 0.0, False

        try:
            # Apply fixed pre-gain first (compensates for very quiet mics before
            # AGC gets involved — AGC alone can't recover a signal below ~0.001 RMS
            # because its max_gain cap limits amplification).
            boosted = audio_chunk * self.pre_gain if self.pre_gain != 1.0 else audio_chunk
            audio      = self.agc.process(boosted)
            rms_future = self.thread_pool.submit(self.rms, audio)

            # Submit every chunk to the VAD worker — never drop.
            # The worker accumulates chunks until it has 512 samples, then
            # runs Silero. Queue size=20 ensures we never block or lose audio.
            try:
                self.vad_queue.put(audio, timeout=0.05)
            except Exception:
                pass   # extremely rare: worker is more than 1s behind

            # Drain result queue to get the freshest VAD probability.
            # With queue size=4, stale results from previous chunks may pile up.
            # We want the most recent one, so drain all but the last.
            chunk_duration_s = len(audio_chunk) / self.sample_rate
            latest_prob = None
            while True:
                try:
                    latest_prob = self.vad_result_queue.get_nowait()
                except Empty:
                    break
            if latest_prob is not None:
                prob = latest_prob
                self.last_vad_prob = prob
            else:
                try:
                    prob = self.vad_result_queue.get(timeout=chunk_duration_s * 1.5)
                    self.last_vad_prob = prob
                except Empty:
                    prob = self.last_vad_prob   # worker hasn't responded yet

            rms_val = rms_future.result(timeout=0.02)

            threshold = self.barge_in_threshold if ai_is_speaking else self.idle_threshold

            if prob > threshold + 0.05:
                is_voice = True
                self.consecutive_voice   += 1
                self.consecutive_silence  = max(0, self.consecutive_silence - 1)
            elif prob < threshold - 0.05:
                is_voice = False
                self.consecutive_silence += 1
                self.consecutive_voice    = max(0, self.consecutive_voice - 1)
            else:
                is_voice = self.consecutive_voice > self.consecutive_silence

            # Remove the RMS gate entirely — AGC normalizes amplitude so raw
            # rms_val is no longer a useful silence indicator. VAD prob is enough.
            # (The old gate was silencing valid speech because rms_val is the
            #  post-AGC value which for min_rms=0.001 always passes, but was
            #  masking the real problem: vad_prob being low due to dropped chunks)

            # Track silence duration for variable-tail use in pipeline
            chunk_ms = len(audio_chunk) / self.sample_rate * 1000
            if not is_voice:
                self._silence_frames += 1
            else:
                self._silence_frames = 0
                self._total_voice_frames += 1

            self._accumulated_silence_ms = self._silence_frames * chunk_ms

            # silence_event fires on the FIRST silent chunk after a voice segment,
            # but only if there were at least 3 consecutive voice frames (≥60ms).
            # This prevents single noisy spikes from triggering a flush mid-word.
            silence_event = (
                self._was_voice and not is_voice
                and self.consecutive_voice >= 3
            )
            if silence_event:
                self._total_voice_frames = 0   # reset for next utterance
            self._was_voice = is_voice

            return audio, is_voice, prob, rms_val, silence_event

        except Exception as e:
            print(f"❌ VAD error: {e}")
            return audio_chunk, False, 0.0, 0.0, False

    def set_partial_text(self, text: str):
        """
        Called by pipeline after each ASR partial result.
        Enables the variable-tail: if the partial ends with sentence-ending
        punctuation, the silence timeout is shortened to sentence_end_silence_ms.
        """
        self._last_partial_text = text

    def get_state(self) -> dict:
        return {
            "prob":          self.last_vad_prob,
            "voice_count":   self.consecutive_voice,
            "silence_count": self.consecutive_silence,
            "was_voice":     self._was_voice,
        }

    def reset(self):
        self.last_vad_prob       = 0.0
        self.consecutive_voice   = 0
        self.consecutive_silence = 0
        self._was_voice          = False
        self._silence_frames     = 0
        self._total_voice_frames = 0
        self._last_partial_text  = ""
        try:
            while True:
                self.vad_queue.get_nowait()
        except Empty:
            pass
        try:
            while True:
                self.vad_result_queue.get_nowait()
        except Empty:
            pass

    def __del__(self):
        self.running = False
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=False)