"""
ollama_client.py — Lightweight Ollama HTTP wrapper
====================================================
Replaces transformers ModelLoader + BitsAndBytes quantization.

Uses:
  POST /api/chat      — multi-turn streaming (recommended)
  POST /api/generate  — single-turn generate (session summary)
  GET  /api/tags      — health / model presence check

Model: qwen2.5:0.5b-instruct  (~397 MB VRAM, fastest, lowest footprint)
       Set OLLAMA_MODEL env var to override.

keep_alive=-1 keeps the model resident in VRAM between calls.
temperature=0 → greedy / deterministic (mirrors old transformers path).
"""

from __future__ import annotations

import json
import os
from typing import Generator, List

import httpx


class OllamaClient:
    """
    HTTP client for Ollama inference.

    Public API (mirrors the old ModelLoader interface at call sites):
      ping()                       → bool
      stream_chat(messages)        → Generator[str]   (multi-turn, preferred)
      stream_generate(sys, prompt) → Generator[str]   (single-turn)
      generate(sys, prompt)        → str
    """

    def __init__(self, config):
        self.base_url   = getattr(config, "ollama_base_url", "http://127.0.0.1:11434")
        self.model      = getattr(config, "model_id",        "qwen2.5:0.5b-instruct")
        self.max_tokens = getattr(config, "max_new_tokens",  220)
        # Sampling / repetition controls — driven from CAGConfig so prompt
        # quality changes don't require code edits.
        self.temperature        = getattr(config, "temperature",        0.75)
        self.top_p              = getattr(config, "top_p",              0.9)
        self.top_k              = getattr(config, "top_k",              40)
        self.repetition_penalty = getattr(config, "repetition_penalty", 1.18)

    # ── Health ────────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """
        Return True if Ollama is reachable and the configured model is loaded.
        Accepts any tag variant (e.g. 'qwen2.5:0.5b-instruct' matches 'qwen2.5:0.5b-instruct-q4_K_M').
        """
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            names = [m["name"] for m in resp.json().get("models", [])]
            base  = self.model.split(":")[0]
            return any(n.startswith(base) for n in names)
        except Exception:
            return False

    def pull_model_if_missing(self) -> bool:
        """
        If the model is not present, trigger an Ollama pull.
        Returns True when the model becomes available.
        """
        if self.ping():
            return True
        print(f"⬇️  Pulling {self.model} via Ollama (one-time download)…")
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": self.model, "stream": True},
                timeout=None,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if status:
                            print(f"   {status}", end="\r", flush=True)
                        if data.get("done"):
                            break
            print(f"\n✅ {self.model} ready")
            return self.ping()
        except Exception as exc:
            print(f"\n❌ Pull failed: {exc}")
            return False

    # ── Multi-turn streaming (/api/chat) ──────────────────────────────────────

    def stream_chat(
        self,
        messages: List[dict],
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens from Ollama /api/chat.

        ``messages`` is a standard OpenAI-style list:
          [{"role": "system",    "content": "…"},
           {"role": "user",      "content": "…"},
           {"role": "assistant", "content": "…"},
           …]

        keep_alive=-1  → model stays loaded in VRAM between calls.
        """
        url     = f"{self.base_url}/api/chat"
        payload = {
            "model":      self.model,
            "messages":   messages,
            "stream":     True,
            "keep_alive": -1,
            "options": {
                "temperature":    self.temperature if self.temperature is not None else 0.0,
                "top_p":          self.top_p if self.top_p is not None else 1.0,
                "top_k":          self.top_k if self.top_k is not None else 0,
                "num_predict":    max_tokens or self.max_tokens,
                "repeat_penalty": self.repetition_penalty,
                "num_ctx":        8192,   # Limit context window → faster KV-cache fill
            },
        }

        with httpx.stream("POST", url, json=payload, timeout=120.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data  = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    # ── Single-turn streaming (/api/generate) ─────────────────────────────────

    def stream_generate(
        self,
        system_prompt: str,
        prompt: str,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens from Ollama /api/generate (single-turn convenience).
        Prefer stream_chat() for multi-turn conversations.
        """
        url     = f"{self.base_url}/api/generate"
        payload = {
            "model":      self.model,
            "system":     system_prompt,
            "prompt":     prompt,
            "stream":     True,
            "keep_alive": -1,
            "options": {
                "temperature":    0,
                "num_predict":    max_tokens or self.max_tokens,
                "repeat_penalty": 1.0,
            },
        }

        with httpx.stream("POST", url, json=payload, timeout=120.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data  = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    # ── Non-streaming generate ────────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Blocking single-turn generate — returns complete response string."""
        return "".join(self.stream_generate(system_prompt, prompt, max_tokens))
