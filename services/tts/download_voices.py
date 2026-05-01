"""
download_voices.py — Download Piper ONNX voice models from HuggingFace.

Usage:
    python download_voices.py                  # downloads default voice only
    python download_voices.py --all            # downloads every voice in VOICES

Voice files are saved to services/tts/models/ (next to this script).

For GPU inference, replace onnxruntime with onnxruntime-gpu in requirements.txt:
    pip uninstall onnxruntime -y
    pip install onnxruntime-gpu
"""

import argparse
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise SystemExit("huggingface_hub is not installed. Run: pip install huggingface_hub")

REPO_ID = "rhasspy/piper-voices"

# filename (in the HF repo) -> local filename
# Format: "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
VOICES: dict[str, str] = {
    # Default — best quality/speed balance
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx":        "en_US-lessac-medium.onnx",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json":   "en_US-lessac-medium.onnx.json",
    # libritts_r — vivian
    "en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx":      "en_US-libritts_r-medium.onnx",
    "en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json": "en_US-libritts_r-medium.onnx.json",
    # amy — ono_anna / sohee
    "en/en_US/amy/medium/en_US-amy-medium.onnx":        "en_US-amy-medium.onnx",
    "en/en_US/amy/medium/en_US-amy-medium.onnx.json":   "en_US-amy-medium.onnx.json",
    # ryan — aiden / ryan
    "en/en_US/ryan/high/en_US-ryan-high.onnx":          "en_US-ryan-high.onnx",
    "en/en_US/ryan/high/en_US-ryan-high.onnx.json":     "en_US-ryan-high.onnx.json",
}

DEFAULT_VOICES = {
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
}

MODELS_DIR = Path(__file__).parent / "models"


def download(repo_filename: str, local_name: str) -> None:
    out = MODELS_DIR / local_name
    if out.exists():
        print(f"  already exists: {out.name} — skipping")
        return
    print(f"  downloading {local_name} ...", flush=True)
    cached = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_filename,
        repo_type="model",
    )
    shutil.copy(cached, out)
    size_mb = out.stat().st_size / 1_000_000
    print(f"  saved {out.name} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Piper TTS voice models")
    parser.add_argument("--all", action="store_true", help="Download all voices, not just the default")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = VOICES if args.all else {k: v for k, v in VOICES.items() if k in DEFAULT_VOICES}

    print(f"Downloading {'all' if args.all else 'default'} Piper voice(s) → {MODELS_DIR}")
    for repo_f, local_f in target.items():
        download(repo_f, local_f)
    print("Done.")


if __name__ == "__main__":
    main()
