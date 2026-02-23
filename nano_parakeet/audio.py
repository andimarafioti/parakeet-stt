"""Audio loading and conversion utilities for nano_parakeet."""
import subprocess
import tempfile

import numpy as np
import soundfile as sf


def convert_to_wav16k(path: str) -> str:
    if path.lower().endswith('.wav'):
        return path
    out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    out.close()
    subprocess.check_call(
        ['ffmpeg', '-y', '-i', path, '-ar', '16000', '-ac', '1', out.name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out.name


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio.astype('float32')
