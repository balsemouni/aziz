"""
CAG Architecture - Conversation Memory Module
Manages conversation history and user profile.

Stage/clarification logic has been removed — the LLM governs conversation
flow entirely through its system prompt.

IMPROVEMENTS v2:
- get_stage_instruction() stub added so inference_engine.py references don't
  crash (it was being called but never defined, causing AttributeError).
  Returns an empty string — the LLM prompt handles stage logic natively.
- extract_name_from_response() patterns extended with common variants.
- load_memory() / save_memory() are no-ops when enable_cache_persistence=False,
  preventing file I/O on every message in fresh-session mode.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    """Single message in conversation"""
    role: str                          # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserProfile:
    """User profile information"""
    name: Optional[str]          = None
    preferences: Dict[str, Any]  = field(default_factory=dict)
    first_interaction: str       = field(default_factory=lambda: datetime.now().isoformat())
    last_interaction: str        = field(default_factory=lambda: datetime.now().isoformat())
    total_interactions: int      = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(**data)


# ─────────────────────────────────────────────────────────────────────────────
# ConversationMemory
# ─────────────────────────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Conversation Memory Manager

    Responsibilities:
    1. Store conversation message history
    2. Manage user profile (name, preferences)
    3. Format history for LLM prompt injection
    4. Optionally persist to disk between sessions

    NOTE: All stage-machine / clarification-tracking logic has been removed.
    The LLM system prompt drives conversation flow naturally.
    """

    def __init__(self, config, max_history: int = 10, persist: bool = True):
        self.config      = config
        self.max_history = max_history
        self.persist     = persist

        self.messages: List[Message]    = []
        self.user_profile: UserProfile  = UserProfile()

        # File paths for optional persistence
        self.memory_dir = os.path.join(
            os.path.dirname(config.cache_file_path), "memory"
        )
        os.makedirs(self.memory_dir, exist_ok=True)

        self.conversation_file = os.path.join(self.memory_dir, "conversation_history.json")
        self.profile_file      = os.path.join(self.memory_dir, "user_profile.json")

        # Only load from disk when persistence is enabled. In fresh-session
        # mode this prevents stale conversation_history.json / user_profile.json
        # from leaking into a new voice session.
        if self.persist:
            self.load_memory()

    # ──────────────────────────────────────────────────────────────────────────
    # Message management
    # ──────────────────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None):
        """Add a message to conversation history."""
        self.messages.append(Message(role=role, content=content, metadata=metadata))

        # Keep within window
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]

        if role == "user":
            self.user_profile.total_interactions += 1
            self.user_profile.last_interaction = datetime.now().isoformat()

        if self.config.enable_cache_persistence:
            self.save_memory()

    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Message]:
        if last_n is None:
            return self.messages
        return self.messages[-(last_n * 2):]

    def format_conversation_for_prompt(self) -> str:
        """Format recent history in Llama 3 chat tokens for prompt injection."""
        if not self.messages:
            return ""

        formatted = []
        for msg in self.get_conversation_history(last_n=5):
            if msg.role == "user":
                formatted.append(
                    "<|start_header_id|>user<|end_header_id|>\n"
                    f"{msg.content}<|eot_id|>"
                )
            else:
                formatted.append(
                    "<|start_header_id|>assistant<|end_header_id|>\n"
                    f"{msg.content}<|eot_id|>"
                )
        return "\n".join(formatted)

    def get_stage_instruction(self) -> str:
        """
        Stub for backward compatibility with inference_engine.py.

        Stage logic has been removed — the LLM system prompt handles
        conversation flow natively.  Returns an empty string so existing
        callers don't need to be modified.
        """
        return ""

    # ──────────────────────────────────────────────────────────────────────────
    # Name helpers
    # ──────────────────────────────────────────────────────────────────────────

    def extract_name_from_response(self, user_response: str) -> Optional[str]:
        """
        Extract user's name from their message.

        Handles: "My name is Sarah", "I'm Alex", "Call me Mike",
        "just call me Tom", or a bare first name.
        """
        text  = user_response.strip()
        lower = text.lower()

        patterns = [
            "people usually call me ",
            "everyone calls me ",
            "just call me ",
            "you can call me ",
            "my name is ",
            "name's ",
            "i'm ",
            "i am ",
            "call me ",
            "this is ",
            "it's ",
        ]

        for pattern in patterns:
            if pattern in lower:
                after     = text[lower.index(pattern) + len(pattern):].strip()
                candidate = after.split()[0].strip('.,!?;:"\'-') if after.split() else ""
                if candidate.lower() not in {
                    "and", "but", "or", "so", "the", "a", "an",
                    "yes", "no", "ok", "sure", "yeah", "nope", "",
                    "here", "calling", "there", "good", "great",
                }:
                    return candidate.capitalize()

        # Last resort: very short response (1-2 words) looks like a name
        words = text.split()
        if len(words) <= 2:
            candidate = words[0].strip('.,!?;:"\'-').capitalize()
            if (
                candidate.lower() not in {
                    "yes", "no", "ok", "sure", "yeah", "nope", "hi",
                    "hello", "hey", "fine", "good", "great", "help",
                    "please", "thanks", "thank", "what", "how",
                }
                and len(candidate) >= 2
            ):
                return candidate

        return None

    def set_user_name(self, name: str):
        """Save the user's name to their profile."""
        self.user_profile.name = name
        if self.config.enable_cache_persistence:
            self.save_memory()

    # ──────────────────────────────────────────────────────────────────────────
    # Reset / clear
    # ──────────────────────────────────────────────────────────────────────────

    def clear_conversation(self):
        """Clear message history but keep user profile."""
        self.messages = []
        if self.config.enable_cache_persistence:
            self.save_memory()

    def reset_all(self):
        """Reset everything including user profile."""
        self.messages     = []
        self.user_profile = UserProfile()
        if self.config.enable_cache_persistence:
            self.save_memory()

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save_memory(self):
        """Persist conversation history and user profile to disk."""
        try:
            with open(self.conversation_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"messages": [m.to_dict() for m in self.messages]},
                    f, indent=2, ensure_ascii=False,
                )
            with open(self.profile_file, "w", encoding="utf-8") as f:
                json.dump(self.user_profile.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Failed to save memory: {e}")

    def load_memory(self):
        """Load conversation history and user profile from disk."""
        try:
            if os.path.exists(self.conversation_file):
                with open(self.conversation_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.messages = [Message(**m) for m in data.get("messages", [])]
                if self.config.verbose:
                    print(f"📝 Loaded {len(self.messages)} messages from memory")

            if os.path.exists(self.profile_file):
                with open(self.profile_file, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
                self.user_profile = UserProfile.from_dict(profile_data)
                if self.config.verbose and self.user_profile.name:
                    print(f"👤 Loaded user profile: {self.user_profile.name}")
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Failed to load memory: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_messages":       len(self.messages),
            "user_name":            self.user_profile.name,
            "total_interactions":   self.user_profile.total_interactions,
            "first_interaction":    self.user_profile.first_interaction,
            "last_interaction":     self.user_profile.last_interaction,
        }