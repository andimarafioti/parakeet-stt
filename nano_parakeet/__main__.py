"""CLI entry point for nano_parakeet.

Usage:
    python -m nano_parakeet audio.wav
    nano-parakeet audio.wav
"""
import argparse

import torch

from nano_parakeet import from_pretrained
from nano_parakeet.audio import convert_to_wav16k, load_audio


def main():
    parser = argparse.ArgumentParser(description='nano-parakeet — PyTorch-first Parakeet TDT transcription')
    parser.add_argument('audio', help='input audio file (ogg/wav/m4a/…)')
    parser.add_argument('--model', default='nvidia/parakeet-tdt-0.6b-v3')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    wav_path = convert_to_wav16k(args.audio)
    audio    = load_audio(wav_path)
    audio_t  = torch.from_numpy(audio)

    model = from_pretrained(args.model, device=args.device)
    token_ids = model.transcribe_audio(audio_t)
    text = model.sp.DecodeIds(token_ids).strip()
    print(text)


if __name__ == '__main__':
    main()
