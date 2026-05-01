"""T3 + T4 verification script."""
import sys, os, wave, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "services", "tts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import piper_engine

print("=== T4: Loading ONNX model ===")
sr = piper_engine.sample_rate("tara")
print(f"Sample rate: {sr} Hz  -- ONNX loaded OK")

print()
print("=== T3: Synthesizing 'hello world' ===")
pcm = piper_engine.synthesize("hello world", voice_name="tara")
print(f"PCM bytes: {len(pcm)}  ({len(pcm)//2} samples)")

out = os.path.join(os.path.dirname(__file__), "test_hello.wav")
with wave.open(out, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm)
print(f"Written: {out}  ({os.path.getsize(out)} bytes)")
print("T3 + T4 PASS")
