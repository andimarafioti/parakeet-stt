#!/usr/bin/env python3
import argparse
import sys
import time
import tempfile
import numpy as np
import soundfile as sf


def convert_to_wav16k_mono(path: str) -> str:
    if path.lower().endswith('.wav'):
        return path
    out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    out.close()
    import subprocess
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
    parser.add_argument('--runs', type=int, default=5,
                        help='number of timed runs (first is warm-up)')
    args = parser.parse_args()

    wav_path = convert_to_wav16k_mono(args.audio)
    audio, sr = load_audio(wav_path)
    duration = len(audio) / sr

    import torch
    if not hasattr(torch, "distributed"):
        class _Dist:
            @staticmethod
            def is_initialized():
                return False
        torch.distributed = _Dist()
    elif not hasattr(torch.distributed, "is_initialized"):
        torch.distributed.is_initialized = lambda: False

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=args.model)
    model = model.cuda()
    model.eval()

    def do_transcribe(wav_path, audio):
        """Try NeMo calling conventions in order until one works."""
        attempts = [
            lambda: model.transcribe([audio], batch_size=1, verbose=False),  # NeMo 1.x numpy
            lambda: model.transcribe([wav_path], batch_size=1),              # NeMo 2.x path
            lambda: model.transcribe([wav_path]),                             # NeMo 2.x minimal
            lambda: model.transcribe([wav_path], batch_size=1, verbose=False),
        ]
        for fn in attempts:
            try:
                out = fn()
                if isinstance(out, tuple):
                    out = out[0]
                result = out[0]
                if isinstance(result, list):
                    result = result[0]
                return (result.text if hasattr(result, 'text') else str(result)).strip()
            except Exception:
                continue
        raise RuntimeError("All NeMo transcribe calling conventions failed")

    print("Warming up…")
    try:
        text = do_transcribe(wav_path, audio)
    except Exception as e:
        print(f"ERROR during warmup: {e}", file=sys.stdout)
        raise

    times = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        do_transcribe(wav_path, audio)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    dt = min(times)
    rtf = duration / dt if dt > 0 else 0
    print(f"audio_s={duration:.2f}  time_s={dt:.4f}  RTF={rtf:.2f}  text={text!r}")


if __name__ == '__main__':
    main()
