"""
CAG Architecture - Enhanced Model Loader Module
Centralized model and tokenizer loading with STREAMING support

IMPROVEMENTS v2:
- Flash Attention 2 auto-enabled with eager fallback (RTX 4050 supports FA2)
- torch.cuda.synchronize() / gc.collect() removed from streaming hot-path
  → was causing 10-30 ms stutter per token in the voice stream
- Greedy decoding enforced (do_sample=False, num_beams=1, repetition_penalty=1.0)
  → consistent, fastest generation for a business receptionist
- Async streaming uses proper thread-pool bridge (no O(n²) token loop)
"""

import torch
import gc
import asyncio
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


class ModelLoader:
    """
    Model Loader - Handles model and tokenizer initialization.

    Streaming: Uses TextIteratorStreamer + background Thread so model.generate()
    uses the KV-cache correctly, giving true O(n) streaming with no wasted compute.
    """

    def __init__(self, config):
        self.config    = config
        self.model     = None
        self.tokenizer = None
        self.device    = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public: load
    # ──────────────────────────────────────────────────────────────────────────

    def load_model_and_tokenizer(self, device):
        self.device = device

        print("\n" + "=" * 60)
        print("🚀 LOADING MODEL AND TOKENIZER")
        print("=" * 60)
        print(f"📥 Model: {self.config.model_id}")

        torch.cuda.empty_cache()
        gc.collect()

        free_before = torch.cuda.mem_get_info()[0] // 1024 ** 2
        total_mem   = torch.cuda.mem_get_info()[1] // 1024 ** 2
        print(f"\n📊 GPU Memory: {free_before}MB / {total_mem}MB available")

        self.tokenizer = self._load_tokenizer()
        self.model     = self._load_model()
        self._apply_model_optimizations()

        torch.cuda.empty_cache()
        gc.collect()

        free_after  = torch.cuda.mem_get_info()[0] // 1024 ** 2
        memory_used = free_before - free_after
        print(f"\n✅ Model loaded  |  used ~{memory_used}MB  |  free {free_after}MB")

        return self.model, self.tokenizer

    # ──────────────────────────────────────────────────────────────────────────
    # Private: load helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_tokenizer(self):
        print("\n🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded")
        return tokenizer

    def _load_model(self):
        """
        Load model with 4-bit quantization.

        Tries Flash Attention 2 first (significant speedup on RTX 4050's Ada
        Lovelace architecture).  Falls back to "eager" if FA2 is not installed
        so the service always starts — just slightly slower.
        """
        print("\n🔧 Loading model with quantization...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_quant_type=self.config.quant_type,
            bnb_4bit_use_double_quant=self.config.use_double_quant,
            bnb_4bit_compute_dtype=self._get_compute_dtype(),
            llm_int8_enable_fp32_cpu_offload=False,
        )

        attn_impl = "flash_attention_2" if self.config.use_flash_attention else "eager"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                device_map={"": 0},
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_cache=True,
                attn_implementation=attn_impl,
            )
            print(f"✅ Model loaded  ({attn_impl} attention)")
        except Exception as exc:
            if attn_impl == "flash_attention_2":
                print(f"⚠️  Flash Attention 2 unavailable ({exc}) — falling back to eager")
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    device_map={"": 0},
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_cache=True,
                    attn_implementation="eager",
                )
                print("✅ Model loaded  (eager attention)")
            else:
                raise

        return model

    def _apply_model_optimizations(self):
        print("\n⚡️ Applying optimizations...")

        if self.config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("   ✅ Gradient checkpointing enabled")

        self.model.eval()
        print("   ✅ Model set to eval mode")

        if self.config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("   ✅ TF32 enabled")

        torch.backends.cudnn.benchmark = False
        torch.cuda.set_per_process_memory_fraction(
            self.config.gpu_memory_fraction, device=0
        )
        print(f"   ✅ GPU memory fraction: {self.config.gpu_memory_fraction}")

    def _get_compute_dtype(self):
        if self.config.compute_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16

    # ──────────────────────────────────────────────────────────────────────────
    # Streaming — sync
    # ──────────────────────────────────────────────────────────────────────────

    def stream_response(self, input_ids, attention_mask=None, max_new_tokens=None):
        """
        Stream model response using TextIteratorStreamer.

        IMPORTANT — no torch.cuda.synchronize() or gc.collect() here.
        Both calls block the Python interpreter / GPU pipeline and cause
        audible stutter in the voice stream (~10-30 ms per token).
        Cleanup is deferred to the post-query reset phase.

        Args:
            input_ids:       Input token IDs (torch.Tensor)
            attention_mask:  Attention mask (torch.Tensor, optional)
            max_new_tokens:  Maximum tokens to generate

        Yields:
            Text chunks as they are generated
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=30.0,
        )

        gen_kwargs: dict = {
            "input_ids":          input_ids,
            "max_new_tokens":     max_new_tokens,
            "streamer":           streamer,
            "do_sample":          False,   # greedy: fastest + deterministic
            "pad_token_id":       self.tokenizer.eos_token_id,
            "eos_token_id":       self.tokenizer.eos_token_id,
            "use_cache":          True,
            "num_beams":          1,
            "repetition_penalty": 1.0,     # 1.0 avoids extra per-token math
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        def _generate():
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    self.model.generate(**gen_kwargs)

        thread = Thread(target=_generate, daemon=True)
        thread.start()

        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join(timeout=10.0)

    def stream_text_response(self, text_input, max_new_tokens=None):
        """Convenience: tokenise text then stream."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")

        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_context_tokens,
        )
        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        yield from self.stream_response(input_ids, attention_mask, max_new_tokens)

    # ──────────────────────────────────────────────────────────────────────────
    # Streaming — async
    # ──────────────────────────────────────────────────────────────────────────

    async def stream_response_async(self, input_ids, attention_mask=None, max_new_tokens=None):
        """
        Async wrapper: sync streamer runs in thread-pool so the event loop
        is never blocked.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _producer():
            try:
                for chunk in self.stream_response(input_ids, attention_mask, max_new_tokens):
                    loop.call_soon_threadsafe(q.put_nowait, ("chunk", chunk))
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))

        loop.run_in_executor(None, _producer)

        while True:
            kind, value = await q.get()
            if kind == "chunk":
                yield value
            elif kind == "error":
                raise RuntimeError(value)
            else:
                break

    async def stream_text_response_async(self, text_input, max_new_tokens=None):
        """Async convenience: tokenise then stream."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")

        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_context_tokens,
        )
        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        async for chunk in self.stream_response_async(input_ids, attention_mask, max_new_tokens):
            yield chunk

    # ──────────────────────────────────────────────────────────────────────────
    # Misc
    # ──────────────────────────────────────────────────────────────────────────

    def get_model(self):
        if self.model is None:
            raise ValueError("Model not loaded.")
        return self.model

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        return self.tokenizer

    def unload_model(self):
        """Unload model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ Model unloaded")