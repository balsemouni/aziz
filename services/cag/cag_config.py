"""
CAG Architecture Configuration Module
OPTIMIZED FOR ZERO LATENCY + COMPLETE RESPONSES

IMPROVEMENTS v2:
- use_flash_attention now defaults to True (RTX 4050 Ada supports FA2)
- COMPRESSED_SYSTEM_PROMPT added — ~80 tokens vs 500+ tokens in the old version.
  Every 100 tokens removed from the system prompt saves ~5-10 ms of TTFT.
  Examples/few-shots have been moved to the KnowledgeStore or removed entirely.
- from_env() supports CAG_PRESET env var for one-line deployment changes
- Presets cleaned up (no duplicate "large" / "max_coverage" confusion)
"""

import os

base_dir = os.path.dirname(__file__)

from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Compressed system prompt — used by CAGSystemFreshSession._build_prompt()
# Stripping the "Examples" section saves ~350 tokens → ~15-20 ms off TTFT.
# ═══════════════════════════════════════════════════════════════════════════════

COMPRESSED_SYSTEM_PROMPT = (
    "You are a friendly, natural-sounding AI voice assistant. "
    "You are speaking OUT LOUD over a real-time voice interface. Replies are spoken, not read. "
    "STYLE: be concise — 1 to 2 short sentences per turn, plain conversational English, "
    "contractions welcome ('I'll', 'you're', 'let's'). No bullet points, no markdown, no lists, "
    "no headings, no emoji, no stage directions, no role labels. "
    "Sound like a real person in a casual conversation — not a brochure, not a support agent. "
    "Avoid corporate filler ('I'd be happy to', 'certainly', 'of course', 'as an AI', 'feel free to'). "
    "FLOW: "
    "(1) Match the user's tone. If they greet casually ('hi', 'hello', 'hey'), greet back casually in one short line — do NOT ask qualifying questions, do NOT mention business, infrastructure, products, or any specific domain unless the user brings it up first. "
    "(2) Answer what was actually asked. Don't redirect to unrelated topics. "
    "(3) If the user's message is unclear or you didn't catch it, say so briefly and ask them to repeat — don't lecture them about phrasing or 'introductions'. "
    "(4) If interrupted, stop immediately on the next user turn — do not finish the previous thought. "
    "(5) Never invent facts. If you don't know, say so in one short line. "
    "LANGUAGE: always reply in the SAME language as the user's last message. "
    "If the user speaks French, reply only in natural French (tutoiement par défaut, contractions naturelles). "
    "If the user switches language mid-conversation, switch with them on the very next turn."
)


@dataclass
class CAGConfig:
    """
    Configuration for CAG Architecture

    OPTIMIZED FOR:
    - Zero-latency streaming (Flash Attention 2, greedy decoding, no sync in hot-path)
    - Complete responses (max_new_tokens=512)
    - RTX 4050 6 GB VRAM with 4-bit quantized Llama 3.2-3B
    """

    # ── Model ────────────────────────────────────────────────────────────────
    # Ollama model tag — lightest available locally (~397 MB VRAM)
    # Override via CAG_MODEL_ID env var or OLLAMA_MODEL env var
    model_id: str         = "qwen2.5:0.5b-instruct"
    model_max_tokens: int = 32768   # Qwen2.5 32 K context window

    # ── Ollama endpoint ──────────────────────────────────────────────────────
    ollama_base_url: str  = "http://127.0.0.1:11434"

    # ── Token budgets ────────────────────────────────────────────────────────
    max_context_tokens: int = 8192   # knowledge + history + query budget
    max_new_tokens: int     = 220    # voice replies are short — keeps TTFT + total turn snappy

    # ── Cache persistence (no-op for Ollama — kept for backward compat) ──────
    cache_file_path: str           = "commercial_kv_cache_7500.pt"
    cache_metadata_path: str       = "cache_metadata_7500.json"
    enable_cache_persistence: bool = False  # not used with Ollama

    # ── Knowledge base ───────────────────────────────────────────────────────
    knowledge_jsonl_path: str      = os.path.join(base_dir, ".\\data\\cache_metadata.json")
    max_knowledge_entries: int     = 50_000

    # ── Quantization (4-bit for 6 GB GPU) ────────────────────────────────────
    use_4bit: bool           = True
    quant_type: str          = "nf4"
    use_double_quant: bool   = True
    compute_dtype: str       = "float16"

    # ── Attention / optimizations ─────────────────────────────────────────────
    # CHANGED: use_flash_attention now defaults to True.
    # The RTX 4050 (Ada Lovelace, sm_89) supports FA2.
    # ModelLoader falls back to "eager" automatically if FA2 is not installed.
    use_flash_attention: bool           = True
    enable_tf32: bool                   = True
    enable_gradient_checkpointing: bool = True
    streaming_buffer_size: int          = 1   # lowest latency

    # ── Cache policy ─────────────────────────────────────────────────────────
    cache_overflow_policy: str    = "truncate"
    cache_truncation_buffer: int  = 50

    # ── Generation — sampling for natural-sounding voice replies ─────────────
    # Greedy decoding produces robotic, repetitive corporate-boilerplate output;
    # a small amount of nucleus sampling makes Qwen2.5 sound dramatically more
    # human while still being fast and stable.
    temperature: Optional[float] = 0.75
    top_p: Optional[float]       = 0.9
    top_k: Optional[int]         = 40
    repetition_penalty: float    = 1.18   # discourage corporate boilerplate loops
    length_penalty: float        = 1.0
    no_repeat_ngram_size: int    = 0
    num_beams: int               = 1
    early_stopping: bool         = False

    # ── System config ────────────────────────────────────────────────────────
    cuda_device: int  = 0
    verbose: bool     = True
    debug_mode: bool  = False

    # ── Conversation memory ──────────────────────────────────────────────────
    max_conversation_history: int  = 10
    enable_conversation_memory: bool = True

    # ── System prompt (overridable at runtime) ────────────────────────────────
    system_prompt: str = field(default_factory=lambda: COMPRESSED_SYSTEM_PROMPT)

    # ─────────────────────────────────────────────────────────────────────────

    def __post_init__(self):
        if self.max_context_tokens + self.max_new_tokens > self.model_max_tokens:
            # Warn only — Ollama can handle very large contexts
            print(
                f"⚠️  NOTE: max_context_tokens ({self.max_context_tokens}) + "
                f"max_new_tokens ({self.max_new_tokens}) = "
                f"{self.max_context_tokens + self.max_new_tokens} exceeds "
                f"model_max_tokens ({self.model_max_tokens}) — "
                "Ollama will truncate if needed."
            )

        valid_policies = {"truncate", "error", "compress"}
        if self.cache_overflow_policy not in valid_policies:
            raise ValueError(
                f"Invalid cache_overflow_policy: '{self.cache_overflow_policy}'. "
                f"Must be one of {valid_policies}"
            )

        if self.max_new_tokens < 150:
            print(
                f"⚠️  WARNING: max_new_tokens ({self.max_new_tokens}) is low — "
                "responses may be cut off mid-sentence. Recommended: 256+"
            )

        if self.max_context_tokens > 8000:
            print(
                f"⚠️  WARNING: max_context_tokens ({self.max_context_tokens}) is very high — "
                "risk of OOM on 6 GB GPU. Recommended: ≤7500"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Class-methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "CAGConfig":
        """Load configuration from environment variables."""
        # Allow a preset name as shortcut
        preset = os.getenv("CAG_PRESET", "").strip()
        if preset:
            return get_config_preset(preset)

        return cls(
            model_id           = os.getenv("CAG_MODEL_ID",  os.getenv("OLLAMA_MODEL", cls.model_id)),
            ollama_base_url    = os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            max_context_tokens = int(os.getenv("CAG_MAX_CONTEXT_TOKENS", cls.max_context_tokens)),
            max_new_tokens     = int(os.getenv("CAG_MAX_NEW_TOKENS",     cls.max_new_tokens)),
            verbose            = os.getenv("CAG_VERBOSE", "true").lower() == "true",
            debug_mode         = os.getenv("CAG_DEBUG",   "false").lower() == "true",
        )

    def get_pytorch_alloc_config(self) -> str:
        return "expandable_segments:True"

    def get_bnb_config_dict(self) -> dict:
        return {
            "load_in_4bit":           self.use_4bit,
            "bnb_4bit_quant_type":    self.quant_type,
            "bnb_4bit_use_double_quant": self.use_double_quant,
            "bnb_4bit_compute_dtype": self.compute_dtype,
        }

    def get_generation_config_dict(self) -> dict:
        cfg: dict = {
            "max_new_tokens":     self.max_new_tokens,
            "do_sample":          False,
            "num_beams":          self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty":     self.length_penalty,
            "early_stopping":     self.early_stopping,
            "use_cache":          True,
        }
        if self.no_repeat_ngram_size > 0:
            cfg["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        return cfg

    def print_config_summary(self):
        print("\n" + "=" * 70)
        print("⚙️  CAG CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"   Model:               {self.model_id}")
        print(f"   Max context tokens:  {self.max_context_tokens:,}")
        print(f"   Max new tokens:      {self.max_new_tokens}")
        print(f"   Flash Attention 2:   {self.use_flash_attention}")
        print(f"   4-bit quant:         {self.use_4bit}  ({self.quant_type})")
        print(f"   GPU memory fraction: {self.gpu_memory_fraction}")
        print(f"   Cache persistence:   {self.enable_cache_persistence}")
        print(f"   Cache file:          {self.cache_file_path}")
        print(f"   Conversation history:{self.max_conversation_history} turns")
        print(f"   Max KB entries:      {self.max_knowledge_entries:,}")
        print("=" * 70)

    def print_memory_estimate(self):
        print("\n" + "=" * 70)
        print("📊 MEMORY ESTIMATION  (RTX 4050 — 6 GB)")
        print("=" * 70)
        model_size_mb  = 1_200
        kv_cache_mb    = (self.max_context_tokens * 16 * 28) / (1024 * 1024)
        activation_mb  = 800
        total_mb       = model_size_mb + kv_cache_mb + activation_mb
        vram           = 6_140
        print(f"   Model (4-bit):    ~{model_size_mb} MB")
        print(f"   KV Cache:         ~{kv_cache_mb:.0f} MB  ({self.max_context_tokens:,} tokens)")
        print(f"   Activations:      ~{activation_mb} MB")
        print(f"   ──────────────────────────────")
        print(f"   Total estimated:  ~{total_mb:.0f} MB")
        print(f"   GPU VRAM:          {vram} MB")
        print(f"   Remaining:        ~{vram - total_mb:.0f} MB  ({(total_mb / vram * 100):.1f}% used)")
        risk = "HIGH" if total_mb > 5_800 else "MEDIUM" if total_mb > 5_000 else "LOW"
        print(f"   OOM risk:          {risk}")
        print("=" * 70)

    def validate_for_gpu(self, gpu_memory_mb: int = 6_140):
        model_size  = 1_200
        kv_cache    = (self.max_context_tokens * 16 * 28) / (1024 * 1024)
        activation  = 800
        total       = model_size + kv_cache + activation
        if total > gpu_memory_mb * 0.95:
            raise ValueError(
                f"Configuration requires ~{total:.0f} MB but GPU only has {gpu_memory_mb} MB! "
                "Reduce max_context_tokens or max_new_tokens."
            )
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Preset configurations  (all sized for RTX 4050 6 GB + Llama 3.2-3B 4-bit)
# ═══════════════════════════════════════════════════════════════════════════════

def get_config_preset(preset_name: str) -> CAGConfig:
    """
    Return a named preset configuration.

    Presets:
      default   — balanced  (7 500 ctx, 256 gen)
      large     — max gen   (6 000 ctx, 512 gen)
      fast      — low TTFT  (5 000 ctx, 256 gen)
      safe      — most stable (4 000 ctx, 256 gen)
    """
    presets = {
        "default": CAGConfig(
            max_context_tokens = 7_500,
            max_new_tokens     = 256,
            cache_file_path    = "commercial_kv_cache_7500.pt",
            cache_metadata_path= "cache_metadata_7500.json",
        ),
        "large": CAGConfig(
            max_context_tokens = 6_000,
            max_new_tokens     = 512,
            cache_file_path    = "commercial_kv_cache_6k.pt",
            cache_metadata_path= "cache_metadata_6k.json",
        ),
        "fast": CAGConfig(
            max_context_tokens = 5_000,
            max_new_tokens     = 256,
            cache_file_path    = "commercial_kv_cache_5k.pt",
            cache_metadata_path= "cache_metadata_5k.json",
        ),
        "safe": CAGConfig(
            max_context_tokens = 4_000,
            max_new_tokens     = 256,
            cache_file_path    = "commercial_kv_cache_4k.pt",
            cache_metadata_path= "cache_metadata_4k.json",
        ),
    }

    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset: '{preset_name}'. Available: {list(presets.keys())}"
        )
    return presets[preset_name]


# ═══════════════════════════════════════════════════════════════════════════════
# Global default instance
# ═══════════════════════════════════════════════════════════════════════════════

config = CAGConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CAG Configuration Tool")
    parser.add_argument("--preset",   choices=["default", "large", "fast", "safe"])
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    cfg = get_config_preset(args.preset) if args.preset else CAGConfig()
    cfg.print_config_summary()

    if args.estimate:
        cfg.print_memory_estimate()

    if args.validate:
        try:
            cfg.validate_for_gpu()
            print("\n✅ Configuration is valid for RTX 4050 (6 GB)")
        except ValueError as e:
            print(f"\n❌ Validation failed: {e}")