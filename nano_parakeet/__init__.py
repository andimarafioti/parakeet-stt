"""nano_parakeet — pure-PyTorch Parakeet TDT inference, no NeMo required."""
import torch
import sentencepiece as spm
from huggingface_hub import hf_hub_download

from nano_parakeet._loader import get_bundled_tokenizer_proto, load_nemo_state_dict, remap_state_dict
from nano_parakeet.model import ParakeetTDT, TranscriptionResult

__all__ = ['ParakeetTDT', 'TranscriptionResult', 'from_pretrained']

_MODEL_CACHE: dict = {}


def from_pretrained(
    model_name: str = 'nvidia/parakeet-tdt-0.6b-v3',
    device: str = 'cuda',
    dtype: torch.dtype = None,
) -> ParakeetTDT:
    """Download (or use cached) model and return a ready-to-use ParakeetTDT.

    Args:
        model_name: HuggingFace model repo ID
        device: torch device string ('cuda' or 'cpu')
        dtype: Optional encoder/decoder dtype, e.g. torch.bfloat16 for modern
               GPUs. Default (None) uses float32 weights with fp16 autocast on
               the encoder, which works on all CUDA devices including Jetson.
    Returns:
        ParakeetTDT model with tokenizer loaded, warmed up and ready to use
    """
    cache_key = (model_name, device, dtype)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    nemo_path = hf_hub_download(
        repo_id=model_name,
        filename=f"{model_name.split('/')[-1]}.nemo",
    )

    model = ParakeetTDT()
    sd = load_nemo_state_dict(nemo_path, map_location='cpu')
    sd = remap_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys in state_dict: {missing[:10]}")
    model = model.to(device).eval()

    if dtype is not None:
        # Preprocessor stays float32 (STFT requires it); encoder/decoder/joint
        # are cast to the requested dtype for native low-precision inference.
        model.encoder.to(dtype)
        model.decoder.to(dtype)
        model.joint.to(dtype)

    proto = get_bundled_tokenizer_proto()
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(proto)
    model.sp = sp

    model.warmup()

    _MODEL_CACHE[cache_key] = model
    return model
