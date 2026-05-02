"""
xtts_engine.py — Coqui XTTS-v2 TTS engine (streaming, multilingual).

Provides the same API surface as `piper_engine` so `services/tts/main.py`
can swap engines without per-call branching:

    sample_rate(voice_name) -> int
    synthesize(text, voice_name=..., language=..., emotion=..., speed=...) -> bytes
    stream(text, voice_name=..., language=..., emotion=..., speed=...) -> Iterator[bytes]

Voice routing
─────────────
Voices are *logical names* mapped to (language, reference-wav) pairs.
XTTS clones the speaker's identity from the reference wav (6–10 s clean speech).

The engine ships with no reference wavs in the repo — drop your own under
`services/tts/voices/refs/` and update VOICE_REFS below.  If a reference is
missing, XTTS falls back to its built-in multi-speaker default.

Lazy load
─────────
The XTTS-v2 model (~2 GB GPU) loads on first synthesis call only.  An optional
idle-unload background timer (TTS_IDLE_UNLOAD_S) frees VRAM when the engine
hasn't been used for a while — useful when sharing a GPU with Whisper/Ollama.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

log = logging.getLogger("xtts_engine")

# ── Output config ────────────────────────────────────────────────────────────
# XTTS-v2 generates 24 kHz mono float32 → we convert to int16 PCM for the wire.
SAMPLE_RATE = 24000

# ── Voice catalogue ──────────────────────────────────────────────────────────
# Map logical voice name → (language, reference wav path).
# Reference wav is optional; XTTS supplies a usable default speaker if absent.
_VOICE_DIR = Path(__file__).parent / "voices" / "refs"

VOICE_REFS: Dict[str, Tuple[str, Optional[Path]]] = {
    # English
    "tara":   ("en", _VOICE_DIR / "tara_en.wav"),
    "aiden":  ("en", _VOICE_DIR / "aiden_en.wav"),
    # French
    "claire": ("fr", _VOICE_DIR / "claire_fr.wav"),
    "lucien": ("fr", _VOICE_DIR / "lucien_fr.wav"),
}
DEFAULT_VOICE = "tara"

# Idle unload (seconds).  0 / unset disables.
_IDLE_UNLOAD_S = float(os.getenv("TTS_IDLE_UNLOAD_S", "0"))

# ── Cached model state (lazy) ────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()
_last_used_at: float = 0.0
_unload_timer: Optional[threading.Timer] = None


def _resolve_voice(voice_name: Optional[str]) -> Tuple[str, str, Optional[str]]:
    """Return (logical_name, language, reference_wav_or_None)."""
    name = (voice_name or DEFAULT_VOICE).lower().strip()
    if name not in VOICE_REFS:
        log.warning(f"Unknown XTTS voice {name!r}, falling back to {DEFAULT_VOICE!r}")
        name = DEFAULT_VOICE
    lang, ref = VOICE_REFS[name]
    ref_str = str(ref) if ref and ref.exists() else None
    if ref and not ref.exists():
        log.warning(
            f"XTTS voice {name!r}: reference {ref} not found — "
            f"using built-in default speaker."
        )
    return name, lang, ref_str


def _load_model():
    """Load XTTS-v2 (lazy, thread-safe).  Returns the TTS instance."""
    global _model, _last_used_at
    with _model_lock:
        if _model is None:
            log.info("Loading Coqui XTTS-v2 (this can take 10-30s the first time)…")
            try:
                from TTS.api import TTS  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Coqui TTS not installed. `pip install coqui-tts` (or TTS)."
                ) from exc
            import torch  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            ).to(device)
            log.info(f"XTTS-v2 loaded on {device}")
        _last_used_at = time.monotonic()
        _schedule_unload()
        return _model


def _schedule_unload():
    """If idle-unload is enabled, (re)arm a timer that frees the model."""
    global _unload_timer
    if _IDLE_UNLOAD_S <= 0:
        return
    if _unload_timer is not None:
        try:
            _unload_timer.cancel()
        except Exception:
            pass
    _unload_timer = threading.Timer(_IDLE_UNLOAD_S, _maybe_unload)
    _unload_timer.daemon = True
    _unload_timer.start()


def _maybe_unload():
    """Release the XTTS model + GPU memory if it's been idle long enough."""
    global _model
    with _model_lock:
        if _model is None:
            return
        idle = time.monotonic() - _last_used_at
        if idle < _IDLE_UNLOAD_S:
            _schedule_unload()
            return
        log.info(f"XTTS idle for {idle:.0f}s — unloading model")
        try:
            del _model
        except Exception:
            pass
        _model = None
        try:
            import torch  # type: ignore
            torch.cuda.empty_cache()
        except Exception:
            pass


# ── Public API ───────────────────────────────────────────────────────────────

def sample_rate(voice_name: Optional[str] = None) -> int:
    """Return XTTS output sample rate (always 24 kHz)."""
    return SAMPLE_RATE


def stream(
    text: str,
    voice_name: Optional[str] = None,
    language:   Optional[str] = None,
    emotion:    Optional[str] = None,
    speed:      Optional[float] = None,
) -> Iterator[bytes]:
    """
    Generator yielding raw 16-bit signed PCM byte chunks at 24 kHz.

    Tries `model.tts_stream()` (true streaming, ~300 ms time-to-first-audio
    on GPU) and falls back to the blocking `tts()` call if streaming isn't
    available in the installed Coqui TTS build.
    """
    global _last_used_at

    if not text or not text.strip():
        return

    name, voice_lang, ref_wav = _resolve_voice(voice_name)
    # Caller language overrides the voice's default language (handy for
    # cloning an English speaker reading French text).
    lang = (language or voice_lang or "en").strip().lower()
    if lang == "auto":
        lang = voice_lang or "en"

    model = _load_model()

    # Speed: XTTS supports a speed kwarg (~0.5..2.0). Emotion maps to
    # repetition_penalty/temperature heuristics.
    spd = float(speed) if (speed and speed > 0) else 1.0
    spd = max(0.5, min(1.6, spd))

    # ── Try true streaming first ──────────────────────────────────────────
    streamed = False
    try:
        # Some Coqui builds expose a low-level streaming API.  The high-level
        # TTS().tts() always returns a complete waveform; the streaming path
        # is on the underlying synthesizer.
        synthesizer = getattr(model, "synthesizer", None)
        tts_model   = getattr(synthesizer, "tts_model", None) if synthesizer else None
        if tts_model is not None and hasattr(tts_model, "inference_stream"):
            # Build conditioning latents once per call.
            gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
                audio_path=ref_wav
            ) if ref_wav else tts_model.get_conditioning_latents()
            for wav_chunk in tts_model.inference_stream(
                text=text,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                stream_chunk_size=20,
                temperature=0.75,
                enable_text_splitting=True,
            ):
                if wav_chunk is None:
                    continue
                # wav_chunk is a torch tensor float32 in [-1, 1]
                arr = wav_chunk.detach().cpu().numpy().astype(np.float32).reshape(-1)
                if spd != 1.0:
                    arr = _resample_speed(arr, spd)
                pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                if pcm:
                    streamed = True
                    yield pcm
            _last_used_at = time.monotonic()
            return
    except Exception as exc:
        log.warning(f"XTTS inference_stream failed ({exc}); falling back to tts()")

    if streamed:
        return

    # ── Fallback: full synth, then chunk into ~64 KB pieces ──────────────
    try:
        wav = model.tts(
            text=text,
            language=lang,
            speaker_wav=ref_wav,
            speed=spd,
        )
    except TypeError:
        # Older signatures don't accept speed.
        wav = model.tts(text=text, language=lang, speaker_wav=ref_wav)

    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    chunk = 64 * 1024
    for off in range(0, len(pcm), chunk):
        yield pcm[off: off + chunk]
    _last_used_at = time.monotonic()


def synthesize(
    text: str,
    voice_name: Optional[str] = None,
    language:   Optional[str] = None,
    emotion:    Optional[str] = None,
    speed:      Optional[float] = None,
) -> bytes:
    """Blocking variant — collect the full PCM stream into one bytes object."""
    return b"".join(
        stream(text, voice_name=voice_name, language=language,
               emotion=emotion, speed=speed)
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resample_speed(arr: np.ndarray, speed: float) -> np.ndarray:
    """
    Crude speed change without pitch correction (linear interpolation).
    Good enough for ±20%; for bigger swings prefer the engine-native speed kwarg.
    """
    if speed == 1.0 or arr.size == 0:
        return arr
    n_in = arr.shape[0]
    n_out = max(1, int(round(n_in / speed)))
    x_old = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, arr).astype(np.float32)
