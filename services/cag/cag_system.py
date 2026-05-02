"""
cag_system.py — CAG System (Ollama backend)
============================================
Replaces the transformers / BitsAndBytes / KV-cache implementation with a
thin Ollama HTTP client.  Public interface is unchanged so gateway, main.py,
and cag_main.py all work without modification.

Removed vs. transformers version:
  * ModelLoader, CacheManager, KV-cache precomputation
  * torch, bitsandbytes, flash-attn imports
  * _aggressive_cleanup() / synchronize() (no GPU state to manage)
  * _build_full_prompt() (replaced by _build_messages())
"""

from __future__ import annotations

import gc
import os
import re as _re
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from cag_config import CAGConfig, COMPRESSED_SYSTEM_PROMPT
from ollama_client import OllamaClient
from knowledge_store import SolutionKnowledgeStore as KnowledgeStore
from conversation_memory import ConversationMemory


# ── Tiny tokenizer shim ────────────────────────────────────────────────────────
# KnowledgeStore.encode() is used only for statistics reporting.
# We provide a char-count approximation (1 token ≈ 4 chars).

class _CharTokenizer:
    """Minimal tokenizer shim: encode returns a list sized by char-count / 4."""
    def encode(self, text: str) -> list:
        return [0] * max(1, len(text) // 4)


# ═══════════════════════════════════════════════════════════════════════════════
# CAGSystemFreshSession  (Ollama-backed)
# ═══════════════════════════════════════════════════════════════════════════════

class CAGSystemFreshSession:
    """
    CAG System — Fresh Session Mode.

    Each voice session starts with a clean conversation.  No history is written
    to disk.  The LLM (guided by the system prompt + injected knowledge base)
    manages the conversation naturally.
    """

    def __init__(self, config: Optional[CAGConfig] = None):
        self.config = config or CAGConfig()
        self.system_prompt: str = getattr(
            self.config, "system_prompt", COMPRESSED_SYSTEM_PROMPT
        )
        # Core components — populated by initialize()
        self.ollama          = None   # OllamaClient
        self.knowledge_store = None   # KnowledgeStore
        self.knowledge_text  = ""     # pre-built knowledge string

        # In-memory conversation — no disk persistence
        self.memory = ConversationMemory(
            config=self.config,
            max_history=self.config.max_conversation_history,
            persist=False,
        )
        self._disable_memory_persistence()

        self.is_initialized     = False
        self.total_queries      = 0
        self.session_start_time = None

    # ── Memory persistence control ────────────────────────────────────────────

    def set_system_prompt(self, prompt: str):
        """Replace the system prompt at runtime."""
        self.system_prompt = prompt

    def _disable_memory_persistence(self):
        """Override save/load so conversation never touches disk, AND wipe
        any in-memory state plus stale on-disk files from previous sessions."""
        self.memory.save_memory = lambda: None
        self.memory.load_memory = lambda: None
        # Hard-clear in-memory history and profile — defense in depth so a
        # fresh session never inherits state from a prior process.
        from conversation_memory import UserProfile as _UserProfile
        self.memory.messages = []
        self.memory.user_profile = _UserProfile()
        for path in (self.memory.conversation_file, self.memory.profile_file):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self, force_cache_rebuild: bool = False):
        """
        Start-up sequence:
          1. Create OllamaClient and verify Ollama is reachable
          2. Auto-pull model if missing (one-time download)
          3. Load knowledge base from JSON
        """
        self.session_start_time = datetime.now()
        print("\n" + "=" * 70)
        print("🚀 CAG SYSTEM — FRESH SESSION MODE  (Ollama backend)")
        print("=" * 70)

        # Phase 1: Ollama
        print("\n📡 PHASE 1: OLLAMA CLIENT")
        self.ollama = OllamaClient(self.config)
        if not self.ollama.pull_model_if_missing():
            raise RuntimeError(
                f"Ollama not reachable at {self.ollama.base_url} or model "
                f"'{self.ollama.model}' could not be pulled.\n"
                "Start Ollama with:  ollama serve"
            )
        print(f"✅ Model {self.ollama.model} ready at {self.ollama.base_url}")

        # Phase 2: Knowledge base
        print("\n📚 PHASE 2: KNOWLEDGE BASE")
        self.knowledge_store = KnowledgeStore(_CharTokenizer(), self.config)
        entry_count          = self.knowledge_store.load_from_sources()
        self.knowledge_text  = self.knowledge_store.build_knowledge_text(use_compact=True)
        print(f"✅ Loaded {entry_count:,} knowledge entries")

        self.is_initialized = True
        print("\n✅ CAG SYSTEM READY")

        # Phase 3: Warmup — pre-fill the model's KV cache with the system prompt
        # so the first real user query sees TTFT of ~1-2s instead of ~10s.
        print("\n⚡ PHASE 3: MODEL WARMUP (pre-filling KV cache)…")
        try:
            msgs = self._build_messages() + [{"role": "user", "content": "Hi"}]
            for _ in self.ollama.stream_chat(msgs, max_tokens=1):
                pass  # drain single-token warmup response
            print("✅ Warmup complete — KV cache primed")
        except Exception as exc:
            print(f"⚠️  Warmup skipped: {exc}")

        self._print_system_status()

    # ── Messages builder ──────────────────────────────────────────────────────

    def _build_messages(self) -> List[dict]:
        """
        Build the Ollama /api/chat messages array.

        Layout:
          [system]  system prompt + user-data block + knowledge base
          [user/assistant turns from memory]
        """
        user_data_lines = []
        if self.memory.user_profile.name:
            user_data_lines.append(f"User name: {self.memory.user_profile.name}")
        if self.memory.user_profile.preferences:
            for k, v in self.memory.user_profile.preferences.items():
                user_data_lines.append(f"User stated {k}: {v}")

        system_content = self.system_prompt
        if user_data_lines:
            system_content += (
                "\n\n══ USER DATA (highest priority — always use this first) ══\n"
                + "\n".join(user_data_lines)
            )
        if self.knowledge_text:
            system_content += (
                "\n\n══ KNOWLEDGE BASE (use when user data doesn't already answer) ══\n"
                + self.knowledge_text
            )

        messages: List[dict] = [{"role": "system", "content": system_content}]
        for msg in self.memory.messages[-(self.config.max_conversation_history * 2):]:
            messages.append({"role": msg.role, "content": msg.content})
        return messages

    # ── Core query path ───────────────────────────────────────────────────────

    def query(self, user_message: str) -> Dict[str, Any]:
        """Non-streaming query — returns complete response dict."""
        if not self.is_initialized:
            raise ValueError("Call initialize() first.")
        self.total_queries += 1
        self._update_user_name(user_message)
        self.memory.add_message("user", user_message)
        try:
            messages = self._build_messages()
            answer   = "".join(self.ollama.stream_chat(messages)).strip()
            self.memory.add_message("assistant", answer)
            return {
                "answer":       answer,
                "query_number": self.total_queries,
                "success":      True,
                "user_name":    self.memory.user_profile.name,
            }
        except Exception as exc:
            return {
                "answer":       f"Error: {exc}",
                "query_number": self.total_queries,
                "success":      False,
                "error":        str(exc),
            }

    def stream_query(self, user_message: str) -> Generator[str, None, None]:
        """Stream response token-by-token."""
        if not self.is_initialized:
            raise ValueError("Call initialize() first.")
        self.total_queries += 1
        self._update_user_name(user_message)
        self.memory.add_message("user", user_message)
        try:
            messages      = self._build_messages()
            response_text = ""
            for token in self.ollama.stream_chat(messages):
                if token:
                    response_text += token
                    yield token
            if response_text:
                self.memory.add_message("assistant", response_text.strip())
        except Exception as exc:
            yield f"\n[Error: {exc}]"

    def stream_chunks(self, user_message: str) -> Generator[str, None, None]:
        """
        Stream complete, TTS-ready sentence chunks.

        Buffers raw tokens from stream_query() and flushes at natural speech
        boundaries.  Each yielded string is a complete utterance the TTS
        engine can speak immediately.

        Chunking rules (mirrors gateway TonalAccumulator):
          1. First chunk fires at FIRST_CHUNK_CHARS (≥30) at a word boundary.
          2. Sentence-ending punctuation (.!?) → flush immediately.
          3. Clause break (,;:—) → flush if chunk ≥ MIN_TTS_CHARS (20).
          4. Hard cap at LOGIC_MAX_CHARS (160) — split at last word boundary.
          5. Tail flush at end of stream.
        """
        _RE_SENTENCE_END = _re.compile(r'(?<=[^\d])([.!?…]+["\'»]?)(?=[\s\u00A0\u202F]|$)')
        _RE_CLAUSE_BREAK = _re.compile(r'([,;:—–])[\s\u00A0\u202F]')
        _RE_STARTS_PUNCT = _re.compile(r'^[\s,\.!?;:\)\]\}\'\"\\u2019\\u2018\\u201c\\u201d\-]')

        FIRST_CHUNK_CHARS = 30
        MIN_TTS_CHARS     = 20
        TONE_MAX_CHARS    = 60
        LOGIC_MAX_CHARS   = 160

        def _try_flush(buf: str, first_sent: bool):
            results = []
            while True:
                if first_sent and len(buf) >= FIRST_CHUNK_CHARS:
                    split = buf.rfind(" ")
                    if split >= MIN_TTS_CHARS:
                        candidate = buf[:split].strip()
                        remainder = buf[split:].lstrip()
                        if candidate:
                            results.append((candidate, remainder, False))
                            buf = remainder; first_sent = False; continue
                elif not first_sent and len(buf) >= TONE_MAX_CHARS:
                    candidate = buf[:TONE_MAX_CHARS].strip()
                    remainder = buf[TONE_MAX_CHARS:].lstrip()
                    if candidate:
                        results.append((candidate, remainder, False))
                        buf = remainder; continue

                m = _RE_SENTENCE_END.search(buf)
                if m:
                    candidate = buf[:m.end()].strip()
                    remainder = buf[m.end():].lstrip()
                    if candidate and len(candidate) >= MIN_TTS_CHARS:
                        results.append((candidate, remainder, False))
                        buf = remainder; first_sent = False; continue

                m = _RE_CLAUSE_BREAK.search(buf)
                if m and not first_sent:
                    candidate = buf[:m.end()].strip()
                    remainder = buf[m.end():].lstrip()
                    if len(candidate) >= MIN_TTS_CHARS and len(remainder) >= 3:
                        results.append((candidate, remainder, False))
                        buf = remainder; continue

                if len(buf) >= LOGIC_MAX_CHARS:
                    split = buf.rfind(" ", 0, LOGIC_MAX_CHARS)
                    if split < MIN_TTS_CHARS:
                        split = LOGIC_MAX_CHARS
                    candidate = buf[:split].strip()
                    remainder = buf[split:].lstrip()
                    if candidate:
                        results.append((candidate, remainder, False))
                        buf = remainder; first_sent = False; continue
                break
            return results, buf, first_sent

        buf        = ""
        first_sent = True

        for raw_token in self.stream_query(user_message):
            if not raw_token:
                continue
            if raw_token.startswith(" "):
                if buf and buf[-1] == " ":
                    raw_token = raw_token.lstrip(" ")
            else:
                if buf and not buf[-1].isspace() and not _RE_STARTS_PUNCT.match(raw_token):
                    raw_token = " " + raw_token
            buf += raw_token
            chunks, buf, first_sent = _try_flush(buf, first_sent)
            for chunk_text, _, _ in chunks:
                if chunk_text:
                    yield chunk_text

        tail = buf.strip()
        if tail:
            yield tail

    # ── Combined reset + query (saves one HTTP round-trip) ───────────────────

    def reset_and_query(self, user_message: str) -> Dict[str, Any]:
        """Clear session then run batch query in one call."""
        self._fast_reset()
        return self.query(user_message)

    def reset_and_stream(self, user_message: str) -> Generator[str, None, None]:
        """Clear session then stream response in one call."""
        self._fast_reset()
        yield from self.stream_query(user_message)

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def _fast_reset(self):
        """Lightweight reset: clears conversation history only."""
        self.memory.messages.clear()
        self.total_queries = 0

    def reset_conversation(self):
        """Full reset: clears history and memory."""
        self.memory.reset_all()
        self.total_queries = 0
        gc.collect()

    def reset_session(self):
        """Alias for reset_conversation() — backward compatibility."""
        self.reset_conversation()

    # ── Stats / summary ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        if not self.is_initialized:
            return {"initialized": False}
        return {
            "initialized":   self.is_initialized,
            "total_queries": self.total_queries,
            "knowledge": {
                "entries": self.knowledge_store.get_entry_count(),
                "tokens":  self.knowledge_store.get_token_count(),
            },
            "model": {
                "name":     self.ollama.model,
                "base_url": self.ollama.base_url,
            },
            "config": {
                "max_context_tokens": self.config.max_context_tokens,
                "max_new_tokens":     self.config.max_new_tokens,
            },
            "session_mode": "fresh_session_no_persistence",
            "memory":       self.memory.get_stats(),
            "session_start": (
                self.session_start_time.isoformat() if self.session_start_time else None
            ),
        }

    def generate_session_summary(self) -> Dict[str, Any]:
        """Ask the LLM to summarise the session (name + one-line summary)."""
        if not self.is_initialized or not self.memory.messages:
            return {"user_name": None, "llm_name": None, "summary": "No conversation."}
        transcript = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in self.memory.messages
        )
        sys_prompt = (
            "You are a precise conversation analyst. Respond with EXACTLY two lines:\n"
            "Line 1: Name: <first name or Unknown>\n"
            "Line 2: Summary: <one sentence>\nNo other text."
        )
        try:
            raw = self.ollama.generate(
                sys_prompt, f"TRANSCRIPT:\n{transcript}", max_tokens=120
            ).strip()
            llm_name = None
            summary  = raw
            for line in raw.splitlines():
                line = line.strip()
                if line.lower().startswith("name:"):
                    c = line[5:].strip()
                    if c.lower() not in {"unknown", "n/a", "none", ""}:
                        llm_name = c
                elif line.lower().startswith("summary:"):
                    summary = line[8:].strip()
            return {
                "user_name": self.memory.user_profile.name,
                "llm_name":  llm_name,
                "summary":   summary,
            }
        except Exception as exc:
            return {
                "user_name": self.memory.user_profile.name,
                "llm_name":  None,
                "summary":   f"[Failed: {exc}]",
            }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self):
        """No-op: Ollama manages its own lifecycle."""
        gc.collect()
        print("\n🧹 CAG system cleaned up (Ollama keeps running)")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_user_name(self, user_message: str):
        if not self.memory.user_profile.name:
            name = self.memory.extract_name_from_response(user_message)
            if name:
                self.memory.set_user_name(name)

    def _print_system_status(self):
        stats = self.get_stats()
        print(f"\n📊 Knowledge entries: {stats['knowledge']['entries']:,}")
        print(f"   Model:             {stats['model']['name']}")
        print(f"   Ollama URL:        {stats['model']['base_url']}")


# ═══════════════════════════════════════════════════════════════════════════════
# CAGSystemWithMemory — persistent profile variant
# ═══════════════════════════════════════════════════════════════════════════════

class CAGSystemWithMemory(CAGSystemFreshSession):
    """
    Same as CAGSystemFreshSession except the user profile and conversation
    history are saved to disk so the user's name is remembered across sessions.
    """

    def __init__(self, config: Optional[CAGConfig] = None):
        super().__init__(config)
        # Re-create memory WITH disk persistence (don't call _disable_memory_persistence)
        self.memory = ConversationMemory(
            config=self.config,
            max_history=self.config.max_conversation_history,
        )

    def reset_all(self):
        """Wipe everything including saved user profile."""
        self.memory.reset_all()
        self.total_queries = 0
        print("🗑️  All memory cleared (including user profile)")
