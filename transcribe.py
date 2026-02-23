#!/usr/bin/env python3
import argparse
import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf


def convert_to_wav16k_mono(path: str) -> str:
    if path.lower().endswith('.wav'):
        return path
    out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    out.close()
    subprocess.check_call([
        'ffmpeg', '-y', '-i', path, '-ar', '16000', '-ac', '1', out.name
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out.name


def load_audio(path: str):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio.astype('float32'), sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', help='input audio file (ogg/wav/m4a)')
    parser.add_argument('--model', default='nvidia/parakeet-tdt-0.6b-v3')
    args = parser.parse_args()

    wav_path = convert_to_wav16k_mono(args.audio)
    audio, sr = load_audio(wav_path)

    import torch
    # Jetson CUDA torch may not expose torch.distributed APIs expected by NeMo
    if not hasattr(torch, "distributed"):
        class _Dist:
            @staticmethod
            def is_initialized():
                return False
        torch.distributed = _Dist()
    elif not hasattr(torch.distributed, "is_initialized"):
        torch.distributed.is_initialized = lambda: False

    import os
    import logging
    import sys

    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")
    logging.getLogger("nemo").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # Silence ALL logs during load/transcribe
    devnull = open(os.devnull, "w")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        import nemo.collections.asr as nemo_asr

        # Parakeet TDT uses EncDecRNNTBPEModel
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=args.model)
        model = model.cuda()
        model.eval()

        out = model.transcribe([audio], batch_size=1, verbose=False)
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        devnull.close()

    # Normalize output
    text = ""
    if isinstance(out, tuple) and len(out) > 0:
        out = out[0]

    if isinstance(out, list) and len(out) > 0:
        first = out[0]
        if isinstance(first, list) and first:
            first = first[0]
        if hasattr(first, 'text'):
            text = first.text.strip()
        elif isinstance(first, str):
            text = first.strip()
        else:
            text = str(first).strip()
    elif hasattr(out, 'text'):
        text = out.text.strip()
    else:
        text = str(out).strip()

    print(text)


if __name__ == '__main__':
    main()
