#!/usr/bin/env python3
"""
orpheus_chunk_benchmark.py
══════════════════════════
GPU latency benchmark for Orpheus TTS.

Synthesizes a configurable set of test chunks, measures per-chunk latency,
and prints a rich terminal report with ASCII waveforms and a summary table.

Usage
-----
    python orpheus_chunk_benchmark.py
    python orpheus_chunk_benchmark.py --voice leo --chunks 8
    python orpheus_chunk_benchmark.py --model canopylabs/orpheus-tts-0.1-finetune-prod
    python orpheus_chunk_benchmark.py --export results.json

Requirements
------------
    pip install orpheus-tts torch numpy

CUDA is used automatically when available.
"""

import argparse
import asyncio
import json
import os
import queue
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
#  ANSI colour helpers
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
MAGENTA= "\033[35m"
WHITE  = "\033[97m"
BG_DARK= "\033[48;5;234m"

def col(text, *codes): return "".join(codes) + str(text) + RESET
def bar(value, max_val, width=30, fill="█", empty="░"):
    filled = int(round(value / max(max_val, 1e-9) * width))
    return fill * filled + empty * (width - filled)

# ─────────────────────────────────────────────────────────────────────────────
#  Test chunk corpus
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHUNKS = [
    ("tone",  "Hello! How can I help you today?"),
    ("logic", "The Orpheus TTS model uses a large language model backbone to generate expressive, natural-sounding speech from text."),
    ("tone",  "Are you sure you want to proceed with this action?"),
    ("logic", "GPU acceleration dramatically reduces synthesis latency by parallelising the attention and feed-forward computations across thousands of CUDA cores."),
    ("tone",  "Great, let's get started right away!"),
    ("logic", "Per-chunk streaming allows the first audio frame to be delivered to the client in under three hundred milliseconds on modern hardware."),
    ("tone",  "Wait — did you just say forty milliseconds?"),
    ("logic", "The benchmark measures wall-clock time from synthesis start to the last PCM byte received, accumulating audio across all chunks to compute a real-time factor."),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Data classes  (self-contained, no microservice import needed)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 24_000
ORPHEUS_VOICES = {"tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"}

@dataclass
class ChunkResult:
    index:          int
    chunk_type:     str
    text:           str
    voice:          str
    audio:          Optional[np.ndarray] = None
    duration_sec:   float = 0.0
    synth_ms:       float = 0.0   # wall clock for synthesis
    first_pcm_ms:   float = 0.0   # time-to-first-PCM-frame
    rtf:            float = 0.0   # real-time factor  (1.0 = real-time)
    error:          Optional[str] = None
    cuda_mem_mb:    float = 0.0   # peak CUDA mem during synthesis


# ─────────────────────────────────────────────────────────────────────────────
#  Minimal Orpheus runner  (mirrors the microservice's worker pattern)
# ─────────────────────────────────────────────────────────────────────────────

class OrpheusRunner:
    """
    Single worker thread with its own asyncio event loop — same architecture
    as the production microservice to give realistic latency numbers.
    """

    def __init__(self, model_name: str, default_voice: str = "tara"):
        self.model_name    = model_name
        self.default_voice = default_voice
        self._pipeline     = None
        self._ready        = threading.Event()
        self._synth_q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="orpheus-bench-worker"
        )

    # ── Load & start ─────────────────────────────────────────────────────────

    def load(self):
        _hdr("Loading model")
        t0 = time.time()
        from orpheus_tts.engine_class import OrpheusModel
        self._pipeline = OrpheusModel(model_name=self.model_name)
        self._ready.set()
        self._thread.start()
        ms = (time.time() - t0) * 1000
        device = "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"  {col('✓', GREEN, BOLD)} Model loaded in {col(f'{ms:.0f} ms', YELLOW)}  "
              f"device={col(device, CYAN)}\n")

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            item = self._synth_q.get()
            if item is None:
                break
            text, voice, pcm_q, cancel_evt = item
            try:
                loop.run_until_complete(self._run(text, voice, pcm_q, cancel_evt))
            except Exception as exc:
                print(col(f"  [worker error] {exc}", RED))
            finally:
                pcm_q.put(None)
                self._synth_q.task_done()
        loop.close()

    async def _run(self, text, voice, pcm_q, cancel_evt):
        self._ready.wait()
        words = text.split()
        padded = text if len(words) >= 8 else text + " Please let me know if you need anything else."
        v = voice if voice in ORPHEUS_VOICES else self.default_voice
        try:
            for raw in self._pipeline.generate_speech(prompt=padded, voice=v):
                if cancel_evt.is_set():
                    break
                if raw:
                    pcm_q.put(raw)
        except Exception as exc:
            print(col(f"  [synth error] {exc}", RED))

    # ── Async interface ───────────────────────────────────────────────────────

    async def synthesize(self, text: str, voice: str) -> ChunkResult:
        result = ChunkResult(index=0, chunk_type="", text=text, voice=voice)
        pcm_q      = queue.Queue()
        cancel_evt = threading.Event()
        parts: list[bytes] = []

        # Reset CUDA memory stats for accurate peak measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.perf_counter()
        first_pcm_abs = None

        self._synth_q.put((text, voice, pcm_q, cancel_evt))
        loop = asyncio.get_event_loop()

        while True:
            try:
                raw = await loop.run_in_executor(
                    None, lambda: pcm_q.get(timeout=120.0)
                )
            except queue.Empty:
                result.error = "timeout"
                cancel_evt.set()
                break

            if raw is None:
                break

            parts.append(raw)
            if first_pcm_abs is None:
                first_pcm_abs = time.perf_counter()

        t_end = time.perf_counter()
        cancel_evt.set()

        result.synth_ms = (t_end - t_start) * 1000
        result.first_pcm_ms = ((first_pcm_abs - t_start) * 1000) if first_pcm_abs else 0.0

        if torch.cuda.is_available():
            result.cuda_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

        if parts:
            all_bytes    = b"".join(parts)
            audio_i16    = np.frombuffer(all_bytes, dtype=np.int16)
            result.audio = audio_i16.astype(np.float32) / 32767.0
            result.duration_sec = len(result.audio) / SAMPLE_RATE
            result.rtf = result.synth_ms / 1000.0 / max(result.duration_sec, 1e-9)
        else:
            result.audio = np.zeros(0, dtype=np.float32)
            if not result.error:
                result.error = "empty audio"

        return result

    def shutdown(self):
        self._synth_q.put(None)
        self._thread.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
#  ASCII waveform
# ─────────────────────────────────────────────────────────────────────────────

def waveform(audio: np.ndarray, width: int = 50, height: int = 5) -> str:
    if audio is None or len(audio) == 0:
        return "  [no audio]\n"
    step = max(1, len(audio) // width)
    cols = [float(np.sqrt(np.mean(audio[i:i+step]**2)))
            for i in range(0, len(audio), step)][:width]
    mx = max(cols) or 1.0
    norm = [v / mx for v in cols]
    rows = []
    for r in range(height, 0, -1):
        thr = r / height
        row = col("│", DIM)
        for v in norm:
            if v >= thr:
                intensity = v
                if intensity > 0.8:
                    c = GREEN
                elif intensity > 0.5:
                    c = CYAN
                else:
                    c = DIM + CYAN
                row += col("█", c)
            else:
                row += " "
        row += col("│", DIM)
        rows.append("  " + row)
    rows.append("  " + col("└" + "─"*width + "┘", DIM))
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Report helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hdr(title: str, width: int = 60):
    print(col("┌" + "─"*(width-2) + "┐", DIM))
    pad = (width - 2 - len(title)) // 2
    print(col("│", DIM) + " "*pad + col(title, BOLD, WHITE) + " "*(width-2-pad-len(title)) + col("│", DIM))
    print(col("└" + "─"*(width-2) + "┘", DIM))

def _divider(width: int = 60):
    print(col("─" * width, DIM))

def _latency_color(ms: float) -> str:
    if ms < 500:   return GREEN
    if ms < 1200:  return YELLOW
    return RED

def _rtf_color(rtf: float) -> str:
    if rtf < 1.0:  return GREEN
    if rtf < 2.0:  return YELLOW
    return RED

def print_chunk_result(r: ChunkResult, idx: int, total: int):
    type_col = CYAN if r.chunk_type == "tone" else MAGENTA
    print(f"\n  {col(f'[{idx}/{total}]', BOLD)} "
          f"{col(r.chunk_type.upper(), type_col, BOLD)}  "
          f"voice={col(r.voice, YELLOW)}")
    print(f"  {col('Text:', DIM)} {r.text[:80]}")
    print()

    if r.error:
        print(f"  {col('✗ Error:', RED, BOLD)} {r.error}\n")
        return

    # Latency bars
    synth_c  = _latency_color(r.synth_ms)
    first_c  = _latency_color(r.first_pcm_ms)
    rtf_c    = _rtf_color(r.rtf)
    max_ms   = max(r.synth_ms, 2000)

    print(f"  {col('Synthesis wall-clock', DIM)} : "
          f"{col(bar(r.synth_ms, max_ms), synth_c)} "
          f"{col(f'{r.synth_ms:>7.0f} ms', synth_c, BOLD)}")

    print(f"  {col('Time to first PCM    ', DIM)} : "
          f"{col(bar(r.first_pcm_ms, max_ms), first_c)} "
          f"{col(f'{r.first_pcm_ms:>7.0f} ms', first_c, BOLD)}")

    print(f"  {col('Audio duration       ', DIM)} : "
          f"{col(bar(r.duration_sec*1000, max_ms), CYAN)} "
          f"{col(f'{r.duration_sec*1000:>7.0f} ms', CYAN)}")

    print(f"  {col('Real-time factor     ', DIM)} : "
          f"{col(f'{r.rtf:.3f}×', rtf_c, BOLD)}  "
          f"{col('(< 1.0 = faster than real-time)', DIM)}")

    if r.cuda_mem_mb:
        print(f"  {col('Peak CUDA memory     ', DIM)} : "
              f"{col(f'{r.cuda_mem_mb:.0f} MB', YELLOW)}")

    print(f"\n  {col('Waveform:', DIM)}")
    print(waveform(r.audio))


def print_summary(results: List[ChunkResult]):
    ok = [r for r in results if not r.error]
    if not ok:
        print(col("\n  No successful chunks to summarise.", RED))
        return

    total_audio = sum(r.duration_sec for r in ok)
    total_synth = sum(r.synth_ms for r in ok) / 1000.0
    avg_rtf     = total_synth / max(total_audio, 1e-9)
    avg_first   = sum(r.first_pcm_ms for r in ok) / len(ok)
    avg_synth   = sum(r.synth_ms for r in ok) / len(ok)
    worst_rtf   = max(r.rtf for r in ok)
    best_rtf    = min(r.rtf for r in ok)

    _hdr("BENCHMARK SUMMARY", 60)
    print()

    rows = [
        ("Chunks synthesised", f"{len(ok)} / {len(results)}"),
        ("Total audio",        f"{total_audio:.2f} s"),
        ("Total synth time",   f"{total_synth*1000:.0f} ms"),
        ("Avg synth / chunk",  f"{avg_synth:.0f} ms"),
        ("Avg time-to-1st-PCM",f"{avg_first:.0f} ms"),
        ("Overall RTF",        f"{avg_rtf:.3f}×"),
        ("Best RTF",           f"{best_rtf:.3f}×"),
        ("Worst RTF",          f"{worst_rtf:.3f}×"),
    ]
    for label, value in rows:
        print(f"  {col(label+' '*(26-len(label)), DIM)}: {col(value, WHITE, BOLD)}")

    # Per-chunk table
    print(f"\n  {col('Per-chunk table:', BOLD)}")
    hdr_cols = ["#", "Type", "Synth ms", "1st PCM ms", "Audio s", "RTF",  "Status"]
    widths   = [3,    6,      9,           11,            8,         6,     7]
    header   = "  " + "  ".join(col(h.ljust(w), BOLD) for h, w in zip(hdr_cols, widths))
    print(header)
    print("  " + col("─"*60, DIM))
    for r in results:
        status = col("OK", GREEN) if not r.error else col("ERR", RED)
        row = [
            str(r.index).ljust(widths[0]),
            r.chunk_type[:widths[1]].ljust(widths[1]),
            (col(f"{r.synth_ms:.0f}", _latency_color(r.synth_ms))).ljust(widths[2]+10),
            (col(f"{r.first_pcm_ms:.0f}", _latency_color(r.first_pcm_ms))).ljust(widths[3]+10),
            f"{r.duration_sec:.3f}".ljust(widths[4]),
            (col(f"{r.rtf:.3f}×", _rtf_color(r.rtf))).ljust(widths[5]+10),
            status,
        ]
        print("  " + "  ".join(row))

    # Verdict
    print()
    if avg_rtf < 0.5:
        verdict = col("🚀 Excellent — well under half real-time!", GREEN, BOLD)
    elif avg_rtf < 1.0:
        verdict = col("✓  Good — faster than real-time", GREEN)
    elif avg_rtf < 1.5:
        verdict = col("⚠  Acceptable — slightly above real-time", YELLOW)
    else:
        verdict = col("✗  Slow — consider a faster GPU or smaller model", RED)
    print(f"  Verdict: {verdict}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-warm
# ─────────────────────────────────────────────────────────────────────────────

async def prewarm(runner: OrpheusRunner):
    _hdr("Pre-warming GPU (CUDA kernel JIT)", 60)
    t0   = time.perf_counter()
    warm = await runner.synthesize(
        "Hello, I am ready to assist you with any questions you may have.",
        runner.default_voice,
    )
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {col('✓', GREEN, BOLD)} Pre-warm done in {col(f'{ms:.0f} ms', YELLOW)} "
          f"— CUDA kernels compiled\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

async def run_benchmark(
    model_name:  str,
    voice:       str,
    chunk_pairs: List[tuple],
    export_path: Optional[str],
    skip_prewarm:bool,
):
    runner = OrpheusRunner(model_name=model_name, default_voice=voice)

    # Load in thread so asyncio loop stays free
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, runner.load)

    if not skip_prewarm:
        await prewarm(runner)

    _hdr(f"Synthesising {len(chunk_pairs)} chunks — voice={voice}", 60)

    results: List[ChunkResult] = []
    bench_start = time.perf_counter()

    for i, (ctype, text) in enumerate(chunk_pairs, 1):
        print(f"\n  {col('→', CYAN)} Chunk {i}/{len(chunk_pairs)}: {col(text[:55]+'…' if len(text)>55 else text, DIM)}")
        r = await runner.synthesize(text, voice)
        r.index      = i
        r.chunk_type = ctype
        results.append(r)
        print_chunk_result(r, i, len(chunk_pairs))
        _divider()

    bench_total_ms = (time.perf_counter() - bench_start) * 1000
    print(f"\n  {col('Benchmark wall-clock:', DIM)} {col(f'{bench_total_ms:.0f} ms', WHITE, BOLD)}"
          f"  (includes pre-warm + all chunks)\n")

    print_summary(results)

    # Optional JSON export
    if export_path:
        export_data = {
            "model":       model_name,
            "voice":       voice,
            "cuda":        torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "chunks": [
                {
                    "index":        r.index,
                    "type":         r.chunk_type,
                    "text":         r.text,
                    "synth_ms":     round(r.synth_ms, 1),
                    "first_pcm_ms": round(r.first_pcm_ms, 1),
                    "duration_sec": round(r.duration_sec, 4),
                    "rtf":          round(r.rtf, 4),
                    "cuda_mem_mb":  round(r.cuda_mem_mb, 1),
                    "error":        r.error,
                }
                for r in results
            ],
        }
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"  {col('✓', GREEN)} Results exported → {col(export_path, CYAN)}\n")

    runner.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Orpheus TTS GPU latency benchmark")
    parser.add_argument("--model", default=os.getenv("ORPHEUS_MODEL",
                        "canopylabs/orpheus-tts-0.1-finetune-prod"),
                        help="HuggingFace model name")
    parser.add_argument("--voice", default=os.getenv("ORPHEUS_VOICE", "tara"),
                        choices=sorted(ORPHEUS_VOICES), help="Orpheus voice")
    parser.add_argument("--chunks", type=int, default=len(DEFAULT_CHUNKS),
                        help="How many chunks to test (uses first N defaults)")
    parser.add_argument("--export", default=None, metavar="FILE",
                        help="Export results as JSON")
    parser.add_argument("--skip-prewarm", action="store_true",
                        help="Skip CUDA pre-warm (first chunk will be slower)")
    parser.add_argument("--custom-texts", nargs="+", metavar="TEXT",
                        help="Replace default chunks with your own texts")
    args = parser.parse_args()

    # Build chunk list
    if args.custom_texts:
        chunk_pairs = [("logic", t) for t in args.custom_texts]
    else:
        chunk_pairs = DEFAULT_CHUNKS[:args.chunks]

    print()
    _hdr("ORPHEUS TTS — GPU LATENCY BENCHMARK", 60)
    print(f"  model  : {col(args.model, CYAN)}")
    print(f"  voice  : {col(args.voice, YELLOW)}")
    print(f"  CUDA   : {col(torch.cuda.is_available(), GREEN if torch.cuda.is_available() else RED)}")
    if torch.cuda.is_available():
        print(f"  device : {col(torch.cuda.get_device_name(0), CYAN)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM   : {col(f'{mem:.1f} GB', YELLOW)}")
    print(f"  chunks : {col(len(chunk_pairs), WHITE)}")
    print()

    asyncio.run(run_benchmark(
        model_name   = args.model,
        voice        = args.voice,
        chunk_pairs  = chunk_pairs,
        export_path  = args.export,
        skip_prewarm = args.skip_prewarm,
    ))


if __name__ == "__main__":
    main()
