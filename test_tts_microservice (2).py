"""
test_tts_microservice.py
────────────────────────
Two focused tests for tts_microservice.py (v3.0.0 / XTTS-v2):

  TEST 1 – TTS built-in voice
      Sends text using the default speaker ("Claribel Dervla").
      Compares the returned audio against a reference synthesis of the
      same text and asserts  is_same_voice == True.

  TEST 2 – Your custom voice (speaker_wav)
      Sends text cloned from your own WAV file.
      Saves the result and reports basic audio stats.

Usage
─────
  # Run both tests (service must be running on localhost:8765):
  python test_tts_microservice.py

  # Override the base URL:
  TTS_URL=http://myhost:8765 python test_tts_microservice.py

  # Point to your own voice sample:
  MY_VOICE_WAV=/path/to/my_voice.wav python test_tts_microservice.py
"""

import base64
import io
import json
import os
import struct
import sys
import wave

import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
#  Config  (override via environment variables)
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL      = os.getenv("TTS_URL",        "http://localhost:8765")
MY_VOICE_WAV  = os.getenv("MY_VOICE_WAV",  "my_voice.wav")      # path to your .wav reference file
TEST_TEXT     = os.getenv("TTS_TEST_TEXT",  "Hello, this is a test of the TTS microservice.")
LANGUAGE      = os.getenv("TTS_LANGUAGE",   "en")
DEFAULT_SPEAKER = "Claribel Dervla"

OUTPUT_DIR    = os.getenv("TTS_OUTPUT_DIR", ".")   # where to save result WAVs


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Parse raw WAV bytes → (float32 numpy array, sample_rate)."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr        = wf.getframerate()
        n_frames  = wf.getnframes()
        n_ch      = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw       = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)
    pcm   = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    pcm  /= float(np.iinfo(dtype).max)

    if n_ch > 1:                        # mix down to mono
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    return pcm, sr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D arrays (same or different lengths)."""
    min_len = min(len(a), len(b))
    a, b    = a[:min_len], b[:min_len]
    denom   = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def is_same_voice(audio_a: np.ndarray, audio_b: np.ndarray,
                  threshold: float = 0.85) -> bool:
    """
    Very lightweight voice-similarity check.

    Both arrays are for the *same text* synthesised with the same speaker,
    so we expect high cosine similarity in the raw waveform (deterministic
    XTTS output).  For a production system you would compare speaker
    embeddings instead, but this is sufficient for a self-consistency test.
    """
    sim = cosine_similarity(audio_a, audio_b)
    print(f"    Cosine similarity : {sim:.4f}  (threshold ≥ {threshold})")
    return sim >= threshold


def post_tts_full(text: str, speaker: str = None,
                  speaker_wav: str = None) -> bytes:
    """POST /tts/full and return raw WAV bytes."""
    payload = {"text": text, "language": LANGUAGE}
    if speaker:
        payload["speaker"] = speaker
    if speaker_wav:
        payload["speaker_wav"] = speaker_wav

    resp = requests.post(f"{BASE_URL}/tts/full", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.content


def save_wav(path: str, wav_bytes: bytes) -> None:
    with open(path, "wb") as f:
        f.write(wav_bytes)
    print(f"    Saved → {path}  ({len(wav_bytes):,} bytes)")


def audio_stats(audio: np.ndarray, sr: int) -> str:
    duration = len(audio) / sr
    rms      = float(np.sqrt(np.mean(audio ** 2)))
    peak     = float(np.max(np.abs(audio)))
    return f"duration={duration:.2f}s  rms={rms:.4f}  peak={peak:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────────────────────

def check_health() -> dict:
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 1 — TTS built-in voice  →  is_same_voice check
# ─────────────────────────────────────────────────────────────────────────────

def test_tts_voice_same_voice():
    """
    Synthesise the same text twice with the default TTS speaker and assert
    that the two outputs are recognised as the same voice.

    Expected result:  is_same_voice == True
    """
    print("\n" + "═" * 60)
    print("TEST 1 — Built-in TTS voice  (is_same_voice check)")
    print("═" * 60)

    print(f"  Speaker  : {DEFAULT_SPEAKER}")
    print(f"  Text     : {TEST_TEXT!r}")

    print("  [1/2] Synthesising first  pass …")
    wav_a = post_tts_full(TEST_TEXT, speaker=DEFAULT_SPEAKER)
    audio_a, sr_a = wav_bytes_to_float32(wav_a)
    print(f"    pass-1  {audio_stats(audio_a, sr_a)}")
    save_wav(os.path.join(OUTPUT_DIR, "tts_voice_pass1.wav"), wav_a)

    print("  [2/2] Synthesising second pass …")
    wav_b = post_tts_full(TEST_TEXT, speaker=DEFAULT_SPEAKER)
    audio_b, sr_b = wav_bytes_to_float32(wav_b)
    print(f"    pass-2  {audio_stats(audio_b, sr_b)}")
    save_wav(os.path.join(OUTPUT_DIR, "tts_voice_pass2.wav"), wav_b)

    result = is_same_voice(audio_a, audio_b)

    if result:
        print("  ✅ PASS — is_same_voice = True")
    else:
        print("  ❌ FAIL — is_same_voice = False  (waveforms diverged beyond threshold)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 2 — Your custom voice  (speaker_wav cloning)
# ─────────────────────────────────────────────────────────────────────────────

def test_my_voice():
    """
    Clone your own voice by passing speaker_wav to /tts/full.
    Saves the result and prints audio stats.

    Requires MY_VOICE_WAV to point to a valid .wav file (≥ 6 s recommended).
    """
    print("\n" + "═" * 60)
    print("TEST 2 — Your custom voice  (speaker_wav cloning)")
    print("═" * 60)

    if not os.path.isfile(MY_VOICE_WAV):
        print(f"  ⚠️  SKIP — voice file not found: {MY_VOICE_WAV!r}")
        print("       Set MY_VOICE_WAV=/path/to/your_voice.wav and re-run.")
        return None

    print(f"  speaker_wav : {MY_VOICE_WAV}")
    print(f"  Text        : {TEST_TEXT!r}")

    # The microservice expects an *absolute* or *server-accessible* path.
    # If your WAV lives on the same machine as the service, pass the path.
    # For a remote service you would need to host the file or extend the API.
    abs_wav = os.path.abspath(MY_VOICE_WAV)

    print("  Synthesising …")
    wav_bytes = post_tts_full(TEST_TEXT, speaker_wav=abs_wav)
    audio, sr = wav_bytes_to_float32(wav_bytes)
    print(f"    {audio_stats(audio, sr)}")

    out_path = os.path.join(OUTPUT_DIR, "my_voice_output.wav")
    save_wav(out_path, wav_bytes)

    print("  ✅ PASS — voice cloning request completed")
    print(f"           Play {out_path!r} to verify it sounds like you.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\nTTS Microservice Test Suite")
    print(f"Base URL : {BASE_URL}")
    print(f"Text     : {TEST_TEXT!r}\n")

    # ── Health check ──────────────────────────────────────────────────────────
    try:
        health = check_health()
        print(f"Health   : {json.dumps(health, indent=2)}")
    except Exception as exc:
        print(f"❌ Cannot reach service at {BASE_URL}: {exc}")
        sys.exit(1)

    # ── Run tests ─────────────────────────────────────────────────────────────
    results = {}

    try:
        results["test_tts_voice_same_voice"] = test_tts_voice_same_voice()
    except Exception as exc:
        print(f"  ❌ TEST 1 raised an exception: {exc}")
        results["test_tts_voice_same_voice"] = False

    try:
        results["test_my_voice"] = test_my_voice()
    except Exception as exc:
        print(f"  ❌ TEST 2 raised an exception: {exc}")
        results["test_my_voice"] = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    for name, passed in results.items():
        if passed is None:
            icon = "⏭️  SKIP"
        elif passed:
            icon = "✅ PASS"
        else:
            icon = "❌ FAIL"
        print(f"  {icon}  {name}")

    failed = [k for k, v in results.items() if v is False]
    if failed:
        print(f"\n{len(failed)} test(s) failed.")
        sys.exit(1)
    else:
        print("\nAll tests passed (or skipped).")


if __name__ == "__main__":
    main()
