from orpheus_tts.engine_class import OrpheusModel
import wave
import time
import sys
from gpu import force_gpu

if __name__ == '__main__':
    force_gpu()

    model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

    prompt = """Man, the way social media has completely changed how we interact is wild."""

    start_time = time.monotonic()

    syn_tokens = model.generate_speech(
        prompt=prompt,
        voice="tara",
    )

    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0

        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)

    duration = total_frames / wf.getframerate()
    end_time = time.monotonic()
    print(f"It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio")