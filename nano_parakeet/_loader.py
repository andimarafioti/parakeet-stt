"""Weight and tokenizer loading utilities for nano_parakeet."""
import os
import shutil
import sys
import tempfile
import zipfile

import torch


def load_nemo_state_dict(nemo_path: str, map_location='cpu') -> dict:
    """Load a state dict directly from a .nemo file without the NeMo framework."""
    # On Linux systems, /tmp is frequently mounted on a RAM-backed tmpfs where
    # unpacking 2.4 GB weights can exhaust memory. We cache the converted PyTorch
    # state dict in ~/.cache/nano_parakeet/ (or NANO_PARAKEET_CACHE) to avoid OOM
    # and provide fast sub-second startup.
    # Non-Linux platforms (e.g. Windows, macOS) retain the standard tempfile extraction.
    if sys.platform.startswith('linux'):
        cache_dir = os.path.expanduser(os.environ.get("NANO_PARAKEET_CACHE", "~/.cache/nano_parakeet"))
        os.makedirs(cache_dir, exist_ok=True)

        nemo_stat = os.stat(nemo_path)
        cache_file = os.path.join(
            cache_dir, f"parakeet_tdt_0.6b_{int(nemo_stat.st_mtime)}_{nemo_stat.st_size}.pt"
        )

        if os.path.isfile(cache_file) and os.path.getsize(cache_file) > 0:
            return torch.load(cache_file, map_location=map_location, weights_only=False)

        z = zipfile.ZipFile(nemo_path)
        entries = [n for n in z.namelist() if n.startswith('model_weights/')]
        tmp_pt = cache_file + ".tmp"
        try:
            writer = torch._C.PyTorchFileWriter(tmp_pt)
            for entry in entries:
                name = entry[len('model_weights/'):]
                if not name:
                    continue
                data = z.read(entry)
                writer.write_record(name, data, len(data))
            writer.write_end_of_file()
            del writer
            os.replace(tmp_pt, cache_file)
            state_dict = torch.load(cache_file, map_location=map_location, weights_only=False)
        finally:
            if os.path.exists(tmp_pt):
                os.remove(tmp_pt)
        return state_dict

    # Default implementation for non-Linux OSes (Windows, macOS, etc.)
    z = zipfile.ZipFile(nemo_path)
    entries = [n for n in z.namelist() if n.startswith('model_weights/')]
    tmpdir = tempfile.mkdtemp()
    tmp_pt = os.path.join(tmpdir, 'model.pt')
    try:
        writer = torch._C.PyTorchFileWriter(tmp_pt)
        for entry in entries:
            name = entry[len('model_weights/'):]
            if not name:
                continue
            data = z.read(entry)
            writer.write_record(name, data, len(data))
        writer.write_end_of_file()
        del writer
        state_dict = torch.load(tmp_pt, map_location=map_location, weights_only=False)
    finally:
        shutil.rmtree(tmpdir)
    return state_dict


def remap_state_dict(sd: dict) -> dict:
    """Map NeMo key names to our simpler module hierarchy."""
    remapped = {}
    for k, v in sd.items():
        if any(k.startswith(p) for p in ('loss.', 'spec_augmentation.', 'joint._loss.', 'joint._wer.')):
            continue
        if k == 'preprocessor.featurizer.window':
            remapped['preprocessor.window'] = v
        elif k == 'preprocessor.featurizer.fb':
            remapped['preprocessor.fb'] = v
        elif k == 'decoder.prediction.embed.weight':
            remapped['decoder.embed.weight'] = v
        elif k.startswith('decoder.prediction.dec_rnn.lstm.'):
            new_k = 'decoder.lstm.' + k[len('decoder.prediction.dec_rnn.lstm.'):]
            remapped[new_k] = v
        else:
            remapped[k] = v
    return remapped


def get_bundled_tokenizer_proto() -> bytes:
    """Return the bundled SentencePiece tokenizer bytes (extracted from NeMo checkpoint)."""
    import importlib.resources as pkg_resources
    ref = pkg_resources.files('nano_parakeet').joinpath('tokenizer.model')
    return ref.read_bytes()
