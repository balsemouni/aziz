"""
CAG Architecture - Cache Manager Module for Solution Recommendations
Manages KV cache lifecycle with truncation and persistence

IMPROVEMENTS v4 (Transformers 5.x compatibility):
- Legacy cache format (tuple-of-tuples), to_legacy_cache(), and
  from_legacy_cache() are ALL removed in Transformers 5.0.
  This version uses only the stable DynamicCache public API:
    • cache.crop(N)                       ← truncate to N tokens
    • for keys, values in cache: ...      ← iterate layers for serialisation
    • cache.update(k, v, layer_idx)       ← rebuild from saved tensors
- key_cache / value_cache attributes are NOT accessed directly.
- No synchronize() or gc.collect() on the hot-path (per-query truncate).
"""

import torch
import gc
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class CacheState:
    """Represents the state of KV cache"""
    input_ids: torch.Tensor
    token_count: int
    knowledge_token_count: int
    past_key_values: Optional[Any] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_count":           self.token_count,
            "knowledge_token_count": self.knowledge_token_count,
            "timestamp":             self.timestamp,
            "metadata":              self.metadata or {},
        }


class CacheManager:
    """
    Cache Manager — Core of CAG Architecture

    Responsibilities:
    1. Pre-compute KV cache from solution knowledge base (once at startup)
    2. Truncate cache after each query (reset to knowledge_token_count)
    3. Persist and load cache from disk
    4. Handle cache overflow according to policy
    """

    def __init__(self, model, tokenizer, device, config):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.config    = config

        self.cache_state: Optional[CacheState] = None
        self.is_initialized = False

    # ──────────────────────────────────────────────────────────────────────────
    # Pre-compute
    # ──────────────────────────────────────────────────────────────────────────

    def precompute_cache(self, knowledge_text: str) -> CacheState:
        """Pre-compute KV cache from solution knowledge base."""
        print("\n" + "=" * 60)
        print("🎯 PRECOMPUTING KV CACHE")
        print("=" * 60)

        self._cleanup_memory()
        free_before = torch.cuda.mem_get_info()[0] // 1024 ** 2
        print(f"📊 Free memory before cache: {free_before} MB")

        prompt = self._build_cache_prompt(knowledge_text)
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_tokens,
        ).input_ids.to(self.device)

        token_count = input_ids.shape[-1]
        print(f"📝 Knowledge base tokens: {token_count}")

        if token_count > self.config.max_context_tokens:
            print(f"⚠️  Truncating to {self.config.max_context_tokens} tokens")
            input_ids   = input_ids[:, :self.config.max_context_tokens]
            token_count = input_ids.shape[-1]

        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    outputs = self.model(input_ids, use_cache=True, return_dict=True)

            past_key_values = outputs.past_key_values
            # Ensure we have a DynamicCache object (model should return one in v5)
            from transformers import DynamicCache
            if not isinstance(past_key_values, DynamicCache):
                raise RuntimeError(
                    f"Expected DynamicCache from model forward pass, got {type(past_key_values)}. "
                    "Transformers 5.x always returns DynamicCache — check your transformers version."
                )

            cache_state = CacheState(
                input_ids             = input_ids,
                token_count           = token_count,
                knowledge_token_count = token_count,
                past_key_values       = past_key_values,
                timestamp             = None,
                metadata              = {"source": "solution_knowledge_base", "type": "recommendations"},
            )

            del outputs
            self._cleanup_memory()

            free_after  = torch.cuda.mem_get_info()[0] // 1024 ** 2
            memory_used = free_before - free_after
            print(f"✅ KV Cache pre-computed  |  {token_count} tokens  |  ~{memory_used} MB used")

            self.cache_state    = cache_state
            self.is_initialized = True
            return cache_state

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ OUT OF MEMORY with {token_count} tokens!")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Truncate (hot path — called after every query)
    # ──────────────────────────────────────────────────────────────────────────

    def truncate_to_knowledge(self):
        """
        Truncate KV cache back to knowledge-base length (N tokens).

        Uses DynamicCache.crop(N) — the official v5 API for in-place truncation.
        No synchronize() or gc.collect() here (10-30 ms TTS stutter otherwise).
        """
        if not self.is_initialized or self.cache_state is None:
            raise ValueError("Cache not initialized.")

        N = self.cache_state.knowledge_token_count

        # Truncate input_ids
        if self.cache_state.input_ids.shape[-1] > N:
            self.cache_state.input_ids = self.cache_state.input_ids[:, :N]

        # Truncate KV cache using the built-in crop() method
        if self.cache_state.past_key_values is not None:
            self.cache_state.past_key_values.crop(N)

        self.cache_state.token_count = N

    # ──────────────────────────────────────────────────────────────────────────
    # Overflow
    # ──────────────────────────────────────────────────────────────────────────

    def handle_overflow(self, query_tokens: int) -> bool:
        if self.cache_state is None:
            return False
        total     = self.cache_state.knowledge_token_count + query_tokens
        available = self.config.max_context_tokens + self.config.max_new_tokens
        if total <= available:
            return True
        if self.config.cache_overflow_policy == "error":
            raise ValueError(f"Query overflow: {total} > {available}")
        elif self.config.cache_overflow_policy == "truncate":
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save_cache(self, path: Optional[str] = None):
        """
        Serialise the DynamicCache to disk.

        Uses __iter__ on DynamicCache which yields (keys, values) per layer —
        the only stable way to extract tensors in Transformers 5.x without
        accessing private attributes.
        """
        if self.cache_state is None:
            raise ValueError("No cache to save")

        path = path or self.config.cache_file_path

        layers_cpu = None
        if self.cache_state.past_key_values is not None:
            # __iter__ yields (key_tensor, value_tensor) for each layer
            layers_cpu = [
                (k.cpu(), v.cpu())
                for k, v in self.cache_state.past_key_values
            ]

        torch.save(
            {
                "input_ids":    self.cache_state.input_ids.cpu(),
                "cache_layers": layers_cpu,          # list of (k, v) tuples
                "metadata":     self.cache_state.to_dict(),
            },
            path,
        )

        if self.config.verbose:
            print(f"💾 Cache saved → {path}  ({len(layers_cpu) if layers_cpu else 0} layers)")

    def load_cache(self, path: Optional[str] = None) -> bool:
        """
        Load a saved cache from disk and reconstruct a DynamicCache via
        cache.update() — the only stable write API in Transformers 5.x.
        """
        path = path or self.config.cache_file_path

        if not os.path.exists(path):
            return False

        try:
            cache_data = torch.load(path, map_location=self.device, weights_only=False)
            metadata   = cache_data["metadata"]

            # Support both new format ("cache_layers") and old formats
            layers_raw = cache_data.get("cache_layers")

            if layers_raw is None:
                # Try old format keys written by earlier cache_manager versions
                pkv_raw = cache_data.get("past_key_values")
                if pkv_raw is None:
                    print("⚠️  Cache file has no layer data — rebuilding...")
                    return False
                # pkv_raw was a tuple-of-tuples; convert to list of (k,v)
                if isinstance(pkv_raw, dict):
                    # Format written by the v3 fix attempt — unreadable, rebuild
                    print("⚠️  Unreadable old cache format — rebuilding...")
                    return False
                layers_raw = [(layer[0], layer[1]) for layer in pkv_raw]
                print("⚠️  Old cache format detected — converting and saving new format...")

            # Rebuild DynamicCache using update() — stable across v5 versions
            from transformers import DynamicCache
            past_key_values = DynamicCache()
            for layer_idx, (k, v) in enumerate(layers_raw):
                past_key_values.update(
                    k.to(self.device),
                    v.to(self.device),
                    layer_idx,
                )

            self.cache_state = CacheState(
                input_ids             = cache_data["input_ids"].to(self.device),
                token_count           = metadata["token_count"],
                knowledge_token_count = metadata["knowledge_token_count"],
                past_key_values       = past_key_values,
                timestamp             = metadata.get("timestamp"),
                metadata              = metadata.get("metadata"),
            )
            self.is_initialized = True

            layer_count = len(layers_raw)
            print(f"✅ Cache loaded ← {path}  ({layer_count} layers)")
            return True

        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────────────────────────────────

    def get_cache_info(self) -> Dict[str, Any]:
        if self.cache_state is None:
            return {"initialized": False}
        return {
            "initialized":      self.is_initialized,
            "token_count":      self.cache_state.token_count,
            "knowledge_tokens": self.cache_state.knowledge_token_count,
            "metadata":         self.cache_state.metadata,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_cache_prompt(self, knowledge_text: str) -> str:
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are the AI receptionist for Ask Novation, a business solutions company.\n"
            "You are warm, professional, and conversational.\n"
            "Use the knowledge base below to answer user questions accurately.\n"
            "\n"
            "=== KNOWLEDGE BASE ===\n"
            f"{knowledge_text}\n"
            "<|eot_id|>"
        )
        return prompt

    # ──────────────────────────────────────────────────────────────────────────
    # Startup / shutdown cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def _cleanup_memory(self):
        """Heavy cleanup — only at startup or shutdown, never per-query."""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()