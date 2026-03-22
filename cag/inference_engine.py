"""
CAG Architecture - Inference Engine Module
Stateless inference engine that uses pre-computed cache

FIXES:
- _build_query_prompt: UPDATED system prompt to be MORE LENIENT
- System prompt now encourages including benefits and pricing in solutions
"""

import torch
import gc
from typing import Optional, Dict, Any


# ============================================================
# SYSTEM PROMPT  — MORE LENIENT, INCLUDES BENEFITS & PRICING
# ============================================================

SYSTEM_PROMPT = """\
You are a professional, friendly technical support agent for Ask Novation.
Your goal is to help users quickly and efficiently with their technical problems.

══════════════════════════════════════════════════════
GUIDELINES — FOLLOW THESE FLEXIBLY
══════════════════════════════════════════════════════

• Be helpful first. If the user has a clear problem and you know the solution,
  provide it directly. Don't force a clarifying question if it's not needed.

• If you don't know the user's name, you can ask for it naturally, but don't
  let it block helping them. You can ask AND help in the same message.

• When providing solutions, ALWAYS include:
  → Key benefits of the recommended solution
  → Pricing information (if available in the knowledge base)
  → Implementation timeline or estimated time to resolve

• If the user's problem description is vague, you may ask ONE clarifying
  question, but use your judgment. Sometimes it's better to offer common
  solutions and let them specify.

• For follow-up questions, answer concisely and directly. If they ask about
  pricing, benefits, or timeline specifically, emphasize those details.

• If the user mentions a completely new problem, treat it fresh. You can
  either ask for clarification or provide a solution if it's clear.

• Stay calm and professional even if the user is impatient or rude.
• Be concise — do not pad responses with unnecessary explanation.

══════════════════════════════════════════════════════
EXAMPLE OF GOOD RESPONSE (with benefits & pricing)
══════════════════════════════════════════════════════
User: My website keeps getting 503 errors on shared hosting.
Assistant: That's a common resource limit issue on shared hosting. 

Here's the solution:
• Contact your host to check error logs and restart PHP services (usually free)
• If it happens frequently, upgrading to a VPS ($20-50/month) gives dedicated resources
• Implementation: 1-3 hours for log analysis, VPS migration takes 1-2 days

Would you like me to walk you through the steps to check your logs?
"""


# ============================================================
# INFERENCE ENGINE
# ============================================================

class CAGInferenceEngine:
    """
    CAG Inference Engine — Stateless Query Handler

    Key principle: This engine is STATELESS relative to conversations.
    It uses the pre-computed knowledge cache for every query.

    The separation between CacheManager (stateful) and InferenceEngine
    (stateless) is what makes this an "architecture" instead of a script.
    """

    def __init__(self, model, tokenizer, cache_manager, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_manager = cache_manager
        self.device = device
        self.config = config
        self.query_count = 0

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def generate(self, query: str, memory=None) -> Dict[str, Any]:
        """
        Generate an answer for a query using cached knowledge.

        Args:
            query:  User's question
            memory: ConversationMemory instance (optional but recommended).
                    When provided, the stage-aware system prompt and
                    conversation history are injected automatically.

        Returns:
            Dictionary with answer and metadata
        """
        if not self.cache_manager.is_initialized:
            raise ValueError("Cache not initialized. Run pre-loading phase first.")

        self.query_count += 1

        cache_state = self.cache_manager.cache_state
        full_prompt = self._build_query_prompt(cache_state, query, memory=memory)

        input_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_tokens + 200,
        ).input_ids.to(self.device)

        query_tokens = input_ids.shape[-1]
        self.cache_manager.handle_overflow(query_tokens)

        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        # temperature/top_p omitted: passing None with
                        # do_sample=False triggers a transformers warning.
                    )

            answer = self.tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )

            del input_ids, output_ids
            self._cleanup_memory()
            self.cache_manager.truncate_to_knowledge()

            return {
                'answer': answer.strip(),
                'query_number': self.query_count,
                'input_tokens': query_tokens,
                'success': True,
            }

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("❌ OOM during generation")
                self._cleanup_memory()
                self.cache_manager.truncate_to_knowledge()
                return {
                    'answer': "Error: Out of memory. Try a shorter query.",
                    'query_number': self.query_count,
                    'success': False,
                    'error': str(e),
                }
            raise

    def generate_streaming(self, query: str, memory=None):
        """
        Placeholder for streaming generation.
        The actual streaming path lives in CAGSystemFreshSession.stream_query().
        Kept here for interface completeness.
        """
        raise NotImplementedError(
            "Use CAGSystemFreshSession.stream_query() for streaming."
        )

    def batch_generate(self, queries: list, memory=None) -> list:
        """
        Generate answers for multiple queries in batch.

        All queries share the same knowledge cache, making this efficient.

        Args:
            queries: List of query strings
            memory:  Shared ConversationMemory instance (optional)

        Returns:
            List of result dictionaries
        """
        results = []
        for query in queries:
            result = self.generate(query, memory=memory)
            results.append(result)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        cache_info = self.cache_manager.get_cache_info()
        return {
            'total_queries': self.query_count,
            'cache_initialized': cache_info['initialized'],
            'knowledge_tokens': cache_info.get('knowledge_tokens', 0),
            'max_new_tokens': self.config.max_new_tokens,
        }

    def reset_stats(self):
        """Reset query counter"""
        self.query_count = 0

    # ----------------------------------------------------------
    # Prompt building
    # ----------------------------------------------------------

    def _build_query_prompt(self, cache_state, query: str,
                            memory=None) -> str:
        """
        Build the complete prompt for inference.

        Structure:
            <system>
              SYSTEM_PROMPT + stage instruction + knowledge base
            </system>
            [conversation history]
            <user>  current query  </user>
            <assistant>            ← model generates from here

        Args:
            cache_state: Pre-computed KV cache state
            query:       Current user message
            memory:      ConversationMemory instance (optional)
        """
        # Decode the knowledge base from the cached input_ids
        knowledge_text = self.tokenizer.decode(
            cache_state.input_ids[0],
            skip_special_tokens=True,
        )

        # ── User-provided data block (HIGHEST priority) ────────────────────
        user_data_lines = []
        if memory is not None and memory.user_profile.name:
            user_data_lines.append(f"User name: {memory.user_profile.name}")
        if memory is not None and memory.user_profile.preferences:
            for k, v in memory.user_profile.preferences.items():
                user_data_lines.append(f"User stated {k}: {v}")
        user_data_block = ""
        if user_data_lines:
            user_data_block = (
                "\n\n\u2550\u2550 USER DATA (highest priority \u2014 always use this first) \u2550\u2550\n"
                + "\n".join(user_data_lines)
            )

        # ── Stage-specific instruction ─────────────────────────
        stage_instruction = ""
        if memory is not None:
            si = memory.get_stage_instruction()
            if si:
                stage_instruction = "\n\n\u2550\u2550 CURRENT TURN INSTRUCTION \u2550\u2550\n" + si

        # ── System block ───────────────────────────────────────
        system_block = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            + SYSTEM_PROMPT
            + user_data_block
            + stage_instruction
            + "\n\n\u2550\u2550 KNOWLEDGE BASE (use when user data doesn't already answer) \u2550\u2550\n"
            + knowledge_text
            + "<|eot_id|>\n"
        )

        # ── Conversation history ───────────────────────────────
        history_block = ""
        if memory is not None:
            history_block = memory.format_conversation_for_prompt()
            if history_block:
                history_block += "\n"

        # ── Current user turn ──────────────────────────────────
        # Personalise the query with the user's name if we have it
        if memory is not None and memory.user_profile.name:
            display_query = f"[{memory.user_profile.name}] {query}"
        else:
            display_query = query

        user_block = (
            "<|start_header_id|>user<|end_header_id|>\n"
            + display_query
            + "<|eot_id|>"
        )

        # ── Assistant header (model generates from here) ───────
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n"

        return system_block + history_block + user_block + assistant_header

    # ----------------------------------------------------------
    # Memory cleanup
    # ----------------------------------------------------------

    def _cleanup_memory(self):
        """Memory cleanup after generation"""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# ============================================================
# SESSION MANAGER (unchanged logic, updated generate calls)
# ============================================================

class CAGSessionManager:
    """
    Session Manager for multi-user CAG deployment.

    Production pattern:
      - One shared KnowledgeStore
      - One shared CacheManager (pre-computed cache)
      - Multiple InferenceEngines (one per user session)

    This lets thousands of users query the same knowledge base
    without re-encoding it for each user.
    """

    def __init__(self, model, tokenizer, cache_manager, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_manager = cache_manager
        self.device = device
        self.config = config
        self.sessions: Dict[str, CAGInferenceEngine] = {}

    def create_session(self, session_id: str) -> CAGInferenceEngine:
        """Create a new inference session"""
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")

        engine = CAGInferenceEngine(
            self.model,
            self.tokenizer,
            self.cache_manager,
            self.device,
            self.config,
        )
        self.sessions[session_id] = engine
        return engine

    def get_session(self, session_id: str) -> Optional[CAGInferenceEngine]:
        """Get existing session"""
        return self.sessions.get(session_id)

    def close_session(self, session_id: str):
        """Close and cleanup a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_active_sessions(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)