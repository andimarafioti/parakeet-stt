#!/usr/bin/env python3
"""Benchmark PyTorch-first Parakeet implementation vs NeMo baseline."""
import argparse
import time

import numpy as np
import soundfile as sf
import torch

from nano_parakeet import from_pretrained
from nano_parakeet.audio import convert_to_wav16k, load_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', help='input audio file')
    parser.add_argument('--model', default='nvidia/parakeet-tdt-0.6b-v3')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--runs', type=int, default=3,
                        help='number of timed runs (first is warm-up)')
    args = parser.parse_args()

    wav_path = convert_to_wav16k(args.audio)
    audio_np = load_audio(wav_path)
    audio_t  = torch.from_numpy(audio_np)
    duration = len(audio_np) / 16_000

    model = from_pretrained(args.model, device=args.device)

    # Warm-up (first inference loads CUDA kernels, etc.)
    print("Warming up…")
    token_ids = model.transcribe_audio(audio_t)
    text = model.sp.DecodeIds(token_ids).strip()

    # Timed runs
    times = []
    for i in range(args.runs):
        if args.device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        token_ids = model.transcribe_audio(audio_t)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    dt  = min(times)
    rtf = duration / dt if dt > 0 else float('inf')
    print(f"audio_s={duration:.2f}  time_s={dt:.4f}  RTF={rtf:.2f}x  text={text!r}")


if __name__ == '__main__':
    main()
