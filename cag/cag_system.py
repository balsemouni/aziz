"""
CAG Architecture - System (Fresh Session + With Memory variants)

Conversation flow is driven entirely by the LLM system prompt.
No hard-coded stage machine.

IMPROVEMENTS v2:
- _aggressive_cleanup() (contains torch.cuda.synchronize) is NO LONGER called
  inside query() or stream_query() hot-paths.  It caused 10-30 ms stutter per
  response in TTS pipelines.  It is now only called in reset_conversation() and
  cleanup() (cold paths where latency doesn't matter).
- gc.collect() in query() replaced with a no-op comment (already handled by
  cache_manager.truncate_to_knowledge).
- stream_query() thread is marked daemon=True so it can't hang process exit.
- Combined reset+query pattern: reset_and_query() / reset_and_stream() added
  so the gateway can eliminate the separate POST /reset round-trip.
- System prompt sourced from config.system_prompt (defaults to
  COMPRESSED_SYSTEM_PROMPT defined in cag_config.py).

FIX v3:
- Added Event to threading import (was causing NameError: name 'threading'
  is not defined when stream_query() called threading.Event()).
"""

import os
import gc
import torch
from typing import Optional, Dict, Any, Generator
from datetime import datetime

from cag_config import CAGConfig, COMPRESSED_SYSTEM_PROMPT
from gpu import free_gpu_smart, force_gpu, get_gpu_memory_info
from model_loader import ModelLoader
from knowledge_store import SolutionKnowledgeStore as KnowledgeStore
from cache_manager import CacheManager
from conversation_memory import ConversationMemory
from transformers import TextIteratorStreamer
from threading import Thread, Event          # ← FIX: added Event


# ═══════════════════════════════════════════════════════════════════════════════
# CAGSystemFreshSession
# ═══════════════════════════════════════════════════════════════════════════════

class CAGSystemFreshSession:
    """
    CAG System — Fresh Session Mode.

    Each run starts with a clean conversation.  No history is written to disk.
    The LLM (guided by the system prompt in config.system_prompt) manages the
    conversation naturally without any hard-coded stage machine.
    """

    def __init__(self, config: Optional[CAGConfig] = None):
        self.config = config or CAGConfig()
        self.config.enable_cache_persistence = True

        # Use compressed system prompt from config (can be overridden)
        self.system_prompt: str = getattr(
            self.config, "system_prompt", COMPRESSED_SYSTEM_PROMPT
        )

        # Core components populated during initialize()
        self.model_loader    = None
        self.knowledge_store = None
        self.cache_manager   = None

        # In-memory conversation — no disk persistence
        self.memory = ConversationMemory(
            config=self.config,
            max_history=self.config.max_conversation_history,
        )
        self._disable_memory_persistence()

        self.device          = None
        self.model           = None
        self.tokenizer       = None
        self.is_initialized  = False
        self.total_queries   = 0
        self.session_start_time = None

    # ──────────────────────────────────────────────────────────────────────────
    # System prompt
    # ──────────────────────────────────────────────────────────────────────────

    def set_system_prompt(self, prompt: str):
        """Replace the system prompt at runtime (no cache rebuild needed)."""
        self.system_prompt = prompt

    # ──────────────────────────────────────────────────────────────────────────
    # Memory persistence control
    # ──────────────────────────────────────────────────────────────────────────

    def _disable_memory_persistence(self):
        """Override save/load so conversation never touches disk."""
        self.memory.save_memory = lambda: None
        self.memory.load_memory = lambda: None
        for path in (self.memory.conversation_file, self.memory.profile_file):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ──────────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────────

    def initialize(self, force_cache_rebuild: bool = False):
        self.session_start_time = datetime.now()

        print("\n" + "=" * 70)
        print("🚀 CAG SYSTEM — FRESH SESSION MODE")
        print("=" * 70)

        self._initialize_gpu()
        self._load_model()
        self._load_knowledge()
        self._precompute_cache(force_rebuild=force_cache_rebuild)

        self.is_initialized = True
        print("\n✅ CAG SYSTEM READY")
        self._print_system_status()

    def _initialize_gpu(self):
        print("\n📊 PHASE 1: GPU INITIALIZATION")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.config.get_pytorch_alloc_config()
        freed = free_gpu_smart(min_mem_mb=self.config.min_free_memory_mb)
        print(f"✅ GPU cleanup: {freed} processes freed")
        gc.collect()
        torch.cuda.empty_cache()
        self.device = force_gpu()

    def _load_model(self):
        print("\n📦 PHASE 2: MODEL LOADING")
        self.model_loader = ModelLoader(self.config)
        self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer(self.device)
        print("✅ Model and tokenizer loaded")

    def _load_knowledge(self):
        print("\n📚 PHASE 3: KNOWLEDGE BASE LOADING")
        self.knowledge_store = KnowledgeStore(self.tokenizer, self.config)
        entry_count = self.knowledge_store.load_from_sources()
        print(f"✅ Loaded {entry_count:,} knowledge entries")
        if self.config.verbose:
            self.knowledge_store.preview_entries(n=3)

    def _precompute_cache(self, force_rebuild: bool = False):
        print("\n🎯 PHASE 4: CACHE PRECOMPUTATION")
        knowledge_text = self.knowledge_store.build_knowledge_text(use_compact=True)
        self.cache_manager = CacheManager(
            self.model, self.tokenizer, self.device, self.config
        )
        cache_loaded = False
        if not force_rebuild and self.config.enable_cache_persistence:
            cache_loaded = self.cache_manager.load_cache()
        if not cache_loaded:
            self.cache_manager.precompute_cache(knowledge_text)
            if self.config.enable_cache_persistence:
                self.cache_manager.save_cache()
                self.knowledge_store.save_metadata()
        print("✅ Cache ready")

    # ──────────────────────────────────────────────────────────────────────────
    # Core query path
    # ──────────────────────────────────────────────────────────────────────────

    def query(self, user_message: str) -> Dict[str, Any]:
        """
        Process a single query by building the full prompt (knowledge prefix +
        conversation history + current question) and running a standard
        model.generate() call — no past_key_values injection.

        This is robust against quantized-model cache-mutation bugs and avoids
        the index-out-of-bounds errors that arise when DynamicCache is passed
        directly into generate() with 4-bit models.
        """
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize() first.")

        self.total_queries += 1

        if not self.memory.user_profile.name:
            name = self.memory.extract_name_from_response(user_message)
            if name:
                self.memory.set_user_name(name)

        self.memory.add_message("user", user_message)

        try:
            full_prompt = self._build_full_prompt()

            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_tokens,
            )
            input_ids      = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            prompt_len     = input_ids.shape[-1]

            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        repetition_penalty=1.0,
                        # temperature and top_p intentionally omitted:
                        # passing None with do_sample=False triggers a
                        # transformers warning and is a no-op anyway.
                    )

            answer = self.tokenizer.decode(
                output_ids[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()

            self.memory.add_message("assistant", answer)
            del input_ids, attention_mask, output_ids, inputs

            return {
                "answer":       answer,
                "query_number": self.total_queries,
                "input_tokens": prompt_len,
                "success":      True,
                "user_name":    self.memory.user_profile.name,
            }

        except Exception as e:
            return {
                "answer":       f"Error: {e}",
                "query_number": self.total_queries,
                "success":      False,
                "error":        str(e),
            }

    def stream_query(self, user_message: str) -> Generator[str, None, None]:
        """
        Stream response token-by-token.

        Builds the complete prompt (system + knowledge base + conversation
        history + current message) and streams using TextIteratorStreamer +
        a daemon Thread.  No past_key_values injection — avoids the
        DynamicCache mutation / index-out-of-bounds bug with 4-bit models.
        """
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize() first.")

        self.total_queries += 1

        if not self.memory.user_profile.name:
            name = self.memory.extract_name_from_response(user_message)
            if name:
                self.memory.set_user_name(name)

        self.memory.add_message("user", user_message)

        try:
            full_prompt = self._build_full_prompt()

            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_tokens,
            )
            input_ids      = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=None,
            )

            gen_kwargs = {
                "input_ids":          input_ids,
                "attention_mask":     attention_mask,
                "max_new_tokens":     self.config.max_new_tokens,
                "streamer":           streamer,
                "do_sample":          False,
                "pad_token_id":       self.tokenizer.eos_token_id,
                "eos_token_id":       self.tokenizer.eos_token_id,
                "use_cache":          True,
                "num_beams":          1,
                "repetition_penalty": 1.0,
            }

            done_event = Event()

            def _gen_thread():
                try:
                    with torch.no_grad():
                        with torch.amp.autocast("cuda"):
                            self.model.generate(**gen_kwargs)
                except Exception as e:
                    print(f"\n❌ Generation thread error: {e}")
                    try:
                        streamer.end()
                    except Exception:
                        pass
                finally:
                    done_event.set()

            thread = Thread(target=_gen_thread, daemon=True)
            thread.start()

            response_text = ""
            try:
                for chunk in streamer:
                    if chunk:
                        response_text += chunk
                        yield chunk
            except Exception as e:
                print(f"\n❌ Streaming error: {e}")
            finally:
                done_event.wait(timeout=120.0)
                thread.join(timeout=5.0)
                if response_text:
                    self.memory.add_message("assistant", response_text.strip())
                del input_ids, attention_mask, inputs

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._aggressive_cleanup()
                yield "\n[Error: GPU out of memory. Please try a shorter message.]"
            else:
                yield f"\n[Error: {e}]"
        except Exception as e:
            yield f"\n[Error: {e}]"

    def stream_chunks(self, user_message: str) -> Generator[str, None, None]:
        """
        Stream response as complete, TTS-ready sentence chunks.

        Unlike stream_query() which yields raw sub-word tokens, this method
        buffers tokens internally and yields only when a natural speech boundary
        is reached (sentence end, clause break, or max-char cap).

        Each yielded string is a complete utterance the TTS engine can speak
        immediately — no further accumulation needed in the gateway.

        Chunking rules (mirrors gateway TonalAccumulator):
          1. First chunk fires as soon as FIRST_CHUNK_CHARS (≥30) chars are ready
             and a word boundary exists — minimises time-to-first-audio.
          2. Sentence-ending punctuation (.!?) → flush immediately.
          3. Clause break (,;:—) → flush if chunk ≥ MIN_TTS_CHARS (20) and
             remainder ≥ 3 chars.
          4. Hard cap at LOGIC_MAX_CHARS (160) — split at last word boundary.
          5. Tail flush at end of stream (remaining buffer).
        """
        # ── chunking constants (match gateway TonalAccumulator) ──────────────
        import re as _re

        _RE_SENTENCE_END = _re.compile(r'(?<=[^\d])([.!?]+["\']?)(?=\s|$)')
        _RE_CLAUSE_BREAK = _re.compile(r'([,;:—–])\s')
        _RE_STARTS_PUNCT = _re.compile(r'^[\s,\.!?;:\)\]\}\'\"\\u2019\\u2018\\u201c\\u201d\-]')

        FIRST_CHUNK_CHARS = 30   # ~5-6 words — minimum first chunk
        MIN_TTS_CHARS     = 20   # minimum chars to bother sending to TTS
        TONE_MAX_CHARS    = 60   # short = tone chunk (question/exclamation)
        LOGIC_MAX_CHARS   = 160  # hard cap per chunk

        def _try_flush(buf: str, first_sent: bool):
            """
            Try to extract one or more chunks from buf.
            Returns list of (chunk_text, new_buf, new_first_sent).
            """
            results = []
            while True:
                # 1. First-chunk early fire
                if first_sent and len(buf) >= FIRST_CHUNK_CHARS:
                    split = buf.rfind(" ")
                    if split >= MIN_TTS_CHARS:
                        candidate = buf[:split].strip()
                        remainder = buf[split:].lstrip()
                        if candidate:
                            results.append((candidate, remainder, False))
                            buf        = remainder
                            first_sent = False
                            continue
                # 2. Short chunk or hard cap when not first
                elif not first_sent and len(buf) >= TONE_MAX_CHARS:
                    candidate = buf[:TONE_MAX_CHARS].strip()
                    remainder = buf[TONE_MAX_CHARS:].lstrip()
                    if candidate:
                        results.append((candidate, remainder, False))
                        buf = remainder
                        continue

                # 3. Sentence boundary
                m = _RE_SENTENCE_END.search(buf)
                if m:
                    candidate = buf[:m.end()].strip()
                    remainder = buf[m.end():].lstrip()
                    if candidate:
                        results.append((candidate, remainder, False))
                        buf        = remainder
                        first_sent = False
                        continue
                    break

                # 4. Clause break
                m = _RE_CLAUSE_BREAK.search(buf)
                if m:
                    candidate = buf[:m.start() + 1].strip()
                    remainder = buf[m.end():].lstrip()
                    if len(candidate) >= MIN_TTS_CHARS and len(remainder) >= 3:
                        results.append((candidate, remainder, False))
                        buf        = remainder
                        first_sent = False
                        continue
                    break

                # 5. Hard length cap
                max_cap = LOGIC_MAX_CHARS if not first_sent else TONE_MAX_CHARS
                if len(buf) > max_cap:
                    split = buf[:max_cap].rfind(" ")
                    if split <= MIN_TTS_CHARS:
                        split = max_cap
                    candidate = buf[:split].strip()
                    remainder = buf[split:].lstrip()
                    if candidate:
                        results.append((candidate, remainder, False))
                        buf        = remainder
                        first_sent = False
                        continue
                    break
                break

            return results, buf, first_sent

        # ── stream raw tokens and accumulate ─────────────────────────────────
        buf        = ""
        first_sent = True

        for raw_token in self.stream_query(user_message):
            if not raw_token:
                continue

            # normalise spacing
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

        # ── tail flush ────────────────────────────────────────────────────────
        tail = buf.strip()
        if len(tail) >= 1:
            yield tail

    def reset_and_query(self, user_message: str) -> Dict[str, Any]:
        """
        Clear session history then run a batch query in a single call.
        Saves one full HTTP round-trip vs. POST /reset → POST /chat.
        """
        self._fast_reset()
        return self.query(user_message)

    def reset_and_stream(self, user_message: str) -> Generator[str, None, None]:
        """
        Clear session history then stream a response in a single call.
        Saves one full HTTP round-trip vs. POST /reset → POST /chat/stream.
        """
        self._fast_reset()
        yield from self.stream_query(user_message)

    def _fast_reset(self):
        """
        Lightweight reset: clears conversation history only.
        Does NOT run synchronize() — safe to call on every voice turn.
        """
        self.memory.messages.clear()
        self.total_queries = 0
        if self.cache_manager:
            self.cache_manager.truncate_to_knowledge()

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt building
    # ──────────────────────────────────────────────────────────────────────────

    def _build_full_prompt(self) -> str:
        """
        Build the complete prompt for a single inference call:

          <s>  system prompt + knowledge base  </s>
          [conversation history turns]
          <user>  current message  </user>
          <assistant>              ← model generates from here

        This is the safe, straightforward approach: the full knowledge text
        is included in every call so we never need to inject past_key_values
        manually (which is fragile with 4-bit quantized models).
        """
        # ── Decode the knowledge base stored in the pre-computed cache ─────
        cache_state    = self.cache_manager.cache_state
        knowledge_text = self.tokenizer.decode(
            cache_state.input_ids[0], skip_special_tokens=True
        )

        # ── User-provided data block (HIGHEST priority) ────────────────────
        # Summarise everything the user has told us so the LLM never has to
        # guess — this section is placed BEFORE the knowledge base so it wins
        # in any attention competition.
        user_data_lines = []
        if self.memory.user_profile.name:
            user_data_lines.append(f"User name: {self.memory.user_profile.name}")
        if self.memory.user_profile.preferences:
            for k, v in self.memory.user_profile.preferences.items():
                user_data_lines.append(f"User stated {k}: {v}")

        user_data_block = ""
        if user_data_lines:
            user_data_block = (
                "\n\n══ USER DATA (highest priority — always use this first) ══\n"
                + "\n".join(user_data_lines)
            )

        # ── System block ───────────────────────────────────────────────────
        parts = [
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            + self.system_prompt
            + user_data_block
            + "\n\n══ KNOWLEDGE BASE (use when user data doesn't already answer the question) ══\n"
            + knowledge_text
            + "<|eot_id|>"
        ]

        # ── Conversation history (all but the last/current user message) ───
        history = self.memory.messages[:-1]
        for msg in history[-(self.config.max_conversation_history * 2):]:
            if msg.role == "user":
                parts.append(
                    "<|start_header_id|>user<|end_header_id|>\n"
                    f"{msg.content}<|eot_id|>"
                )
            else:
                parts.append(
                    "<|start_header_id|>assistant<|end_header_id|>\n"
                    f"{msg.content}<|eot_id|>"
                )

        # ── Current user message ───────────────────────────────────────────
        current_query = self.memory.messages[-1].content
        user_name     = self.memory.user_profile.name
        display_query = f"[{user_name}] {current_query}" if user_name else current_query

        parts.append(
            "<|start_header_id|>user<|end_header_id|>\n"
            + display_query
            + "<|eot_id|>"
        )

        # ── Assistant header — model generates from here ───────────────────
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n")

        return "\n".join(parts)

    def reset_conversation(self):
        """Full reset: clears history, memory, and runs heavy GPU cleanup."""
        if self.cache_manager:
            self.cache_manager.truncate_to_knowledge()
        self.memory.reset_all()
        self.total_queries = 0
        self._aggressive_cleanup()   # synchronize() OK here — cold path

    def reset_session(self):
        """Alias for reset_conversation() — backward compatibility."""
        self.reset_conversation()

    # ──────────────────────────────────────────────────────────────────────────
    # Stats / summary
    # ──────────────────────────────────────────────────────────────────────────

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
            "cache":   self.cache_manager.get_cache_info(),
            "config": {
                "max_context_tokens": self.config.max_context_tokens,
                "max_new_tokens":     self.config.max_new_tokens,
                "flash_attention":    self.config.use_flash_attention,
            },
            "gpu_memory":   get_gpu_memory_info(),
            "session_mode": "fresh_session_no_persistence",
            "memory":       self.memory.get_stats(),
            "session_start": (
                self.session_start_time.isoformat() if self.session_start_time else None
            ),
        }

    def generate_session_summary(self) -> Dict[str, Any]:
        """Ask the LLM to summarise the session (name + one-line summary)."""
        if not self.is_initialized or not self.memory.messages:
            return {"user_name": None, "llm_name": None, "summary": "No conversation to summarise."}

        transcript = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in self.memory.messages
        )

        summary_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a precise conversation analyst. "
            "Read the transcript and respond with EXACTLY two lines.\n"
            "Line 1: Name: <user's first name, or 'Unknown'>\n"
            "Line 2: Summary: <one concise sentence describing issue and outcome>\n"
            "No other text."
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"TRANSCRIPT:\n{transcript}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

        try:
            inputs = self.tokenizer(
                summary_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_tokens,
            )
            input_ids      = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=120,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                    )

            raw = self.tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
            ).strip()

            del input_ids, output_ids, attention_mask, inputs
            self._aggressive_cleanup()   # cold path — synchronize OK

            llm_name, summary = None, raw
            for line in raw.splitlines():
                line = line.strip()
                if line.lower().startswith("name:"):
                    candidate = line[5:].strip()
                    if candidate.lower() not in {"unknown", "n/a", "none", ""}:
                        llm_name = candidate
                elif line.lower().startswith("summary:"):
                    summary = line[8:].strip()

            return {
                "user_name": self.memory.user_profile.name,
                "llm_name":  llm_name,
                "summary":   summary,
            }

        except Exception as e:
            return {
                "user_name": self.memory.user_profile.name,
                "llm_name":  None,
                "summary":   f"[Summary generation failed: {e}]",
            }

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup helpers
    # ──────────────────────────────────────────────────────────────────────────

    def cleanup(self):
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self._aggressive_cleanup()
        print("\n🧹 Cleaned up CAG system")

    def print_cache_content(self):
        if not self.cache_manager or not self.cache_manager.cache_state:
            print("❌ Cache is empty or not initialized")
            return
        ids  = self.cache_manager.cache_state.input_ids[0]
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        print(text)
        print(f"📏 Total Tokens: {len(ids)}")

    def _generate_thread(self, **kw):
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    self.model.generate(**kw)
        except Exception as e:
            print(f"\n❌ Generation thread error: {e}")

    def _aggressive_cleanup(self):
        """
        Heavy cleanup: gc + empty_cache + synchronize.

        Call only on COLD paths (reset, cleanup, session summary).
        NEVER call inside query() or stream_query() — it blocks the GPU
        pipeline and causes 10-30 ms stutter per response.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _print_system_status(self):
        stats = self.get_stats()
        print(f"\n📊 Knowledge entries: {stats['knowledge']['entries']:,}")
        print(f"   Cache initialized: {stats['cache']['initialized']}")
        print(f"   Flash Attention:   {stats['config']['flash_attention']}")


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
        if self.cache_manager:
            self.cache_manager.truncate_to_knowledge()
        self._aggressive_cleanup()
        print("🗑️  All memory cleared (including user profile)")