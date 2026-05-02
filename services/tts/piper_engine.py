"""
piper_engine.py — Piper ONNX TTS engine.

Loads one ONNX voice model per speaker name at first use (lazy) and caches it
for subsequent calls.  All synthesis runs on CPU via ONNX Runtime — no GPU
required.

── Setup ─────────────────────────────────────────────────────────────────────
Download voice models before starting the server:

    python download_voices.py          # default: en_US-lessac-medium (~63 MB)
    python download_voices.py --all    # all voices used by VOICE_MODELS below

Or download a single model manually with the HuggingFace CLI:

    huggingface-cli download rhasspy/piper-voices \\
        en/en_US/lessac/medium/en_US-lessac-medium.onnx \\
        en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \\
        --local-dir services/tts/models --local-dir-use-symlinks False

── GPU (optional) ────────────────────────────────────────────────────────────
Piper uses ONNX Runtime.  To run on GPU, swap the CPU provider for CUDA:

    pip uninstall onnxruntime -y
    pip install onnxruntime-gpu

Then set USE_GPU=1 in your environment.  No code changes needed — piper-tts
reads the ONNX Runtime providers automatically when onnxruntime-gpu is installed.

── Speaker → model file mapping ─────────────────────────────────────────────
Controlled by VOICE_MODELS below.
Set PIPER_MODELS_DIR (env var or default "models/") to the directory that
contains the .onnx and .onnx.json files.

── Emotion → speed ───────────────────────────────────────────────────────────
Emotion → speed multiplier mapping is via EMOTION_SPEED.  Piper has no native
emotion API so we express emotion purely as speech rate (length_scale):
  length_scale > 1.0 → slower (e.g. empathetic)
  length_scale < 1.0 → faster (e.g. excited)
  length_scale = 1.0 → neutral
"""

import io
import logging
import os
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger("piper_engine")

# ── Models directory ──────────────────────────────────────────────────────────
# Override with the PIPER_MODELS_DIR environment variable.
MODELS_DIR = Path(os.getenv("PIPER_MODELS_DIR", Path(__file__).parent / "models"))

# ── Speaker → ONNX model filename mapping ────────────────────────────────────
# Keys are the voice names the gateway sends (TTS_SPEAKER env var).
# Values are filenames (without path) inside MODELS_DIR.
# Download voices from: https://huggingface.co/rhasspy/piper-voices
VOICE_MODELS: Dict[str, str] = {
    # Default / gateway default
    "tara":     "en_US-lessac-medium.onnx",
    # Legacy Qwen speaker names kept for back-compat
    "serena":   "en_US-lessac-medium.onnx",
    "vivian":   "en_US-libritts_r-medium.onnx",
    "ono_anna": "en_US-amy-medium.onnx",
    "sohee":    "en_US-amy-medium.onnx",
    "aiden":    "en_US-ryan-high.onnx",
    "ryan":     "en_US-ryan-high.onnx",
}

# Fallback model if a requested voice is not in VOICE_MODELS
DEFAULT_MODEL = "en_US-lessac-medium.onnx"

# ── Emotion → length_scale multiplier ────────────────────────────────────────
# length_scale = 1 / desired_speed_ratio
# empathetic → 12% slower, excited → 10% faster, calm → 10% slower
EMOTION_SPEED: Dict[str, float] = {
    "empathetic": 1.14,   # ~0.88× speed
    "excited":    0.91,   # ~1.10× speed
    "calm":       1.11,   # ~0.90× speed
    "neutral":    1.07,   # 7% slower than raw = natural human cadence
}

# Default length_scale when no emotion is specified (human-sounding baseline)
DEFAULT_LENGTH_SCALE = 1.07

# ── Voice cache ───────────────────────────────────────────────────────────────
_voice_cache: Dict[str, object] = {}  # model_path → PiperVoice


def _load_voice(model_filename: str):
    """Load (or return cached) PiperVoice for the given model filename."""
    from piper import PiperVoice  # imported here to keep module import fast

    model_path = str(MODELS_DIR / model_filename)
    if model_path not in _voice_cache:
        log.info(f"Loading Piper model: {model_path}")
        _voice_cache[model_path] = PiperVoice.load(model_path)
        log.info(f"Piper model loaded: {model_filename} "
                 f"(sample_rate={_voice_cache[model_path].config.sample_rate})")
    return _voice_cache[model_path]


def _resolve_voice(voice_name: Optional[str]) -> str:
    """Map a speaker name to a model filename, falling back to DEFAULT_MODEL."""
    if not voice_name:
        return DEFAULT_MODEL
    name = voice_name.lower().strip()
    if name in VOICE_MODELS:
        return VOICE_MODELS[name]
    log.warning(f"Unknown voice '{voice_name}', using default model")
    return DEFAULT_MODEL


def _resolve_length_scale(emotion: Optional[str]) -> float:
    """Map an emotion tag to a Piper length_scale value."""
    if not emotion:
        return DEFAULT_LENGTH_SCALE
    return EMOTION_SPEED.get(emotion.lower().strip(), DEFAULT_LENGTH_SCALE)


# ── Public API ────────────────────────────────────────────────────────────────

def sample_rate(voice_name: Optional[str] = None) -> int:
    """Return the sample rate for the given voice (loads the model if needed)."""
    model_filename = _resolve_voice(voice_name)
    voice = _load_voice(model_filename)
    return voice.config.sample_rate


def synthesize(
    text: str,
    voice_name: Optional[str] = None,
    emotion: Optional[str] = None,
    speed: Optional[float] = None,
) -> bytes:
    """
    Synthesize *text* and return raw 16-bit signed-integer PCM bytes.

    Args:
        text:       The text to synthesize.
        voice_name: Speaker/voice name (mapped via VOICE_MODELS).
        emotion:    Optional emotion tag (mapped to speed via EMOTION_SPEED).
        speed:      Optional explicit speed multiplier (e.g. 1.1 = 10% faster).

    Returns:
        Raw PCM bytes (signed 16-bit little-endian, mono).
        Sample rate can be queried with piper_engine.sample_rate(voice_name).
    """
    return b"".join(
        chunk for chunk in stream(text, voice_name=voice_name, emotion=emotion, speed=speed)
    )


def stream(
    text: str,
    voice_name: Optional[str] = None,
    emotion: Optional[str] = None,
    speed: Optional[float] = None,
):
    """
    Generator yielding raw PCM byte chunks as Piper produces them.

    Used by the streaming WebSocket path so first audio reaches the client
    without waiting for the full sentence to finish synthesizing.
    """
    if not text or not text.strip():
        return

    model_filename = _resolve_voice(voice_name)
    voice = _load_voice(model_filename)
    length_scale = _resolve_length_scale(emotion)
    if speed and speed > 0:
        # speed > 1 → faster → smaller length_scale
        length_scale = max(0.5, min(2.0, length_scale / float(speed)))

    from piper.config import SynthesisConfig

    syn_config = SynthesisConfig(length_scale=length_scale)
    for chunk in voice.synthesize(text, syn_config=syn_config):
        data = chunk.audio_int16_bytes
        if data:
            yield data
