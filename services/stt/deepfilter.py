"""
deepfilter.py
─────────────
DeepFilterNet noise reducer with:
  • Proper import check (tells you HOW to install if missing)
  • Async GPU processing on a background thread
  • Passthrough mode for zero-latency fallback
  • Adaptive chunk skipping for ultra-low latency

Install DeepFilterNet:
  pip install deepfilternet

If you see "DeepFilterNet not available" run the above command and restart.
"""

import torch
import numpy as np
from threading import Thread
from queue import Queue, Empty

# ─────────────────────────────────────────────────────────────────────────────
#  Availability check — give a clear install message, not a silent warning
# ─────────────────────────────────────────────────────────────────────────────

DEEPFILTER_AVAILABLE = False
_DF_IMPORT_ERROR: str | None = None

try:
    from df.enhance import init_df, enhance
    from df.io import resample as df_resample
    DEEPFILTER_AVAILABLE = True
except ImportError as _e:
    _DF_IMPORT_ERROR = str(_e)
    print(
        "⚠️  DeepFilterNet not available.\n"
        f"    Reason : {_DF_IMPORT_ERROR}\n"
        "    Fix    : pip install deepfilternet\n"
        "    Until then the noise reducer runs in PASSTHROUGH mode."
    )
except Exception as _e:
    _DF_IMPORT_ERROR = str(_e)
    print(f"⚠️  DeepFilterNet failed to load: {_DF_IMPORT_ERROR} — using passthrough")


class DeepFilterNoiseReducer:
    """
    DeepFilterNet noise reducer.

    Modes
    ─────
    PASSTHROUGH  — model not loaded, audio returned unchanged (zero latency).
                   Active when:  deepfilternet not installed
                              OR passthrough_mode=True (explicit bypass)
                              OR model load fails

    ASYNC GPU    — inference runs on a background thread.
                   If enhanced audio isn't ready in time, the original chunk
                   is returned instantly (never blocks the audio pipeline).
    """

    def __init__(
        self,
        sample_rate: int   = 16000,
        device: str        = "cuda" if torch.cuda.is_available() else "cpu",
        chunk_size: int    = 512,
        passthrough_mode: bool = False,   # Set True to skip model entirely
        skip_ratio: float  = 0.0,         # 0.0 = process every chunk
                                          # 0.7 = skip 70% for lower CPU
    ):
        self.input_sample_rate = sample_rate
        self.device            = device
        self.chunk_size        = chunk_size
        self.skip_ratio        = skip_ratio
        self.model             = None
        self.passthrough_mode  = passthrough_mode or not DEEPFILTER_AVAILABLE

        # Prefer GPU if available but caller passed "cpu"
        if torch.cuda.is_available() and self.device == "cpu":
            print("⚡ GPU available — switching DeepFilter to CUDA")
            self.device = "cuda"

        if self.passthrough_mode:
            reason = "explicitly requested" if passthrough_mode else "deepfilternet not installed"
            print(f"⚡ DeepFilter PASSTHROUGH mode ({reason})")
            return

        # ── Load model ──────────────────────────────────────────────────────
        try:
            print(f"🧠 Loading DeepFilterNet on {self.device.upper()}…")
            self.model, self.df_state, _ = init_df(
                post_filter=True,
                log_level="ERROR",
            )
            self.model = self.model.to(self.device).eval()

            if self.device == "cuda":
                self.model = torch.jit.optimize_for_inference(
                    torch.jit.script(self.model)
                )
                # GPU warmup — avoids first-call latency spike
                dummy = torch.randn(1, 1, chunk_size, device=self.device)
                with torch.no_grad():
                    enhance(self.model, self.df_state, dummy)
                torch.cuda.synchronize()
                print("✅ DeepFilter GPU warmed up")

            self.df_sample_rate = self.df_state.sr()
            self.needs_resampling = (self.input_sample_rate != self.df_sample_rate)

            # ── Async worker queues ────────────────────────────────────────
            self.input_queue  = Queue(maxsize=1)   # Only keep the latest chunk
            self.output_queue = Queue(maxsize=2)
            self.running      = True
            self.process_counter = 0

            self._worker_thread = Thread(
                target=self._process_worker, daemon=True, name="DeepFilterWorker"
            )
            self._worker_thread.start()

            print(f"✅ DeepFilterNet loaded ASYNC ({self.df_sample_rate} Hz → {self.input_sample_rate} Hz)")

        except Exception as exc:
            print(f"❌ DeepFilter model load failed: {exc} — falling back to passthrough")
            self.model = None
            self.passthrough_mode = True

    # ─────────────────────────────────────────────────────────────────────────
    #  Background worker
    # ─────────────────────────────────────────────────────────────────────────

    def _process_worker(self):
        """Runs in a daemon thread — pulls chunks from input_queue, enhances them."""
        while self.running:
            try:
                audio_chunk = self.input_queue.get(timeout=0.1)

                # Enforce expected shape
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.flatten()

                n = len(audio_chunk)
                if n < 128:
                    self._put_output(audio_chunk)
                    continue

                # Pad or trim to chunk_size
                if n < self.chunk_size:
                    audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - n))
                else:
                    audio_chunk = audio_chunk[:self.chunk_size]

                audio_t = torch.from_numpy(audio_chunk.astype(np.float32))

                if self.needs_resampling:
                    audio_t = df_resample(audio_t, self.input_sample_rate, self.df_sample_rate)

                audio_t = audio_t.to(self.device)

                with torch.no_grad():
                    enhanced = enhance(
                        self.model,
                        self.df_state,
                        audio_t.unsqueeze(0).unsqueeze(0),
                    ).squeeze(0).squeeze(0)

                if self.needs_resampling:
                    enhanced = df_resample(enhanced.cpu(), self.df_sample_rate, self.input_sample_rate)
                    enhanced_np = enhanced.numpy()
                else:
                    enhanced_np = enhanced.cpu().numpy()

                self._put_output(enhanced_np[:n])   # Trim to original length

            except Empty:
                continue
            except Exception as exc:
                # Log but never crash the worker — just pass through
                print(f"[DeepFilter worker] ⚠️ {exc}")

    def _put_output(self, audio: np.ndarray):
        """Non-blocking put — drops the old result if queue is full."""
        try:
            self.output_queue.get_nowait()
        except Empty:
            pass
        try:
            self.output_queue.put_nowait(audio)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Zero-latency entry point.

        Returns enhanced audio if the worker has it ready, otherwise returns
        the original chunk immediately.  Never blocks.
        """
        if len(audio_chunk) == 0:
            return audio_chunk

        if self.passthrough_mode or self.model is None:
            return audio_chunk

        self.process_counter += 1

        # Adaptive skipping — submit only a fraction of chunks to the worker
        # (reduces CPU/GPU load; unprocessed chunks fall back to original audio)
        if self.skip_ratio > 0.0 and np.random.random() < self.skip_ratio:
            pass  # Don't submit; try to return an already-enhanced chunk below
        else:
            # Submit to worker (non-blocking — drops oldest if busy)
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass
            try:
                self.input_queue.put_nowait(audio_chunk.copy())
            except Exception:
                pass

        # Return enhanced audio if ready, else original (ZERO LATENCY)
        try:
            enhanced = self.output_queue.get_nowait()
            n = len(audio_chunk)
            return enhanced[:n] if len(enhanced) >= n else audio_chunk
        except Empty:
            return audio_chunk

    def flush(self) -> np.ndarray:
        """Clear both queues (call between utterances if needed)."""
        for q in (self.input_queue, self.output_queue):
            try:
                while True:
                    q.get_nowait()
            except Empty:
                pass
        return np.array([], dtype=np.float32)

    def is_available(self) -> bool:
        """True if the model is loaded and running (not passthrough)."""
        return self.model is not None and not self.passthrough_mode

    def __call__(self, audio_chunk: np.ndarray) -> np.ndarray:
        return self.process(audio_chunk)

    def __del__(self):
        if hasattr(self, "running"):
            self.running = False