# Parakeet STT

PyTorch-first Parakeet TDT inference — no NeMo required. Loads weights directly from the `.nemo` checkpoint. Produces byte-identical transcriptions while being significantly faster than the official NeMo baseline.

## Results

RTF > 1.0 = faster than real-time. Benchmarks use 5 timed runs after a warm-up run; best time is reported.

### Warm throughput (steady-state, both implementations loaded)

| GPU | Audio | NeMo RTF | nano-parakeet RTF | Speedup |
|---|---|---|---|---|
| Jetson AGX Orin 64GB | 12s | ~73× | ~92× | **1.3×** |

`nano_parakeet` uses fp16 autocast on the encoder and a CUDA graph for the per-step decoder+joint computation.

### Cold start (first inference, including framework load)

| GPU | NeMo first inference | PyTorch-first first inference |
|---|---|---|
| Jetson AGX Orin 64GB | ~30s (framework load) | ~3s (weights only) |

The practical advantage of `transcribe_pt.py` is **startup time**: NeMo initialises PyTorch Lightning, Hydra, OmegaConf, distributed training scaffolding and compiles CUDA kernels on the first call. The PyTorch-first implementation loads only the weights and runs immediately.

## Quick Start

```bash
git clone https://github.com/andimarafioti/parakeet-stt
cd parakeet-stt
pip install -e .
./benchmark.sh sample.wav   # runs both implementations and prints comparison
```

Requires: Python 3.10+, NVIDIA GPU with CUDA, ffmpeg.

## Setup

```bash
# create venv
uv venv /home/andi/Documents/parakeet-stt/.venv

# install CUDA torch (JetPack 6)
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python /home/andi/Documents/parakeet-stt/.venv \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# install deps
uv pip install --python /home/andi/Documents/parakeet-stt/.venv \
  soundfile numpy "nemo_toolkit[asr]==1.23.0" "transformers==4.33.3" "huggingface_hub==0.19.4"

# build torchaudio from source (matches torch 2.5 on Jetson)
uv pip install --python /home/andi/Documents/parakeet-stt/.venv wheel setuptools==68.2.2
rm -rf /tmp/torchaudio && git clone --depth 1 --branch v2.5.0 https://github.com/pytorch/audio.git /tmp/torchaudio
# patch for FLT_MAX on Jetson
sed -i 's/#include <algorithm>/#include <algorithm>\n#include <cfloat>/' /tmp/torchaudio/src/libtorchaudio/cuctc/src/ctc_prefix_decoder_kernel_v2.cu
/home/andi/Documents/parakeet-stt/.venv/bin/pip install --no-build-isolation /tmp/torchaudio

# install nano-parakeet and sentencepiece
uv pip install --python /home/andi/Documents/parakeet-stt/.venv sentencepiece
pip install -e .
```

The tokenizer is extracted automatically from the `.nemo` archive at runtime — no separate extraction step needed.

---

## Usage

### nano-parakeet (recommended)

```bash
# CLI
nano-parakeet /path/to/audio.ogg
# or
python -m nano_parakeet /path/to/audio.ogg

# Python API
python - << 'EOF'
from nano_parakeet import from_pretrained
model = from_pretrained()
print(model.transcribe('/path/to/audio.ogg'))
EOF
```

### NeMo-based (original)

```bash
/home/andi/Documents/parakeet-stt/.venv/bin/python transcribe.py /path/to/audio.ogg
```

Both accept OGG, WAV, M4A, or any format that ffmpeg can read.
Non-16 kHz mono WAV inputs are converted automatically with ffmpeg.

---

## How It Works

`nano_parakeet` re-implements the full inference pipeline in pure PyTorch — no NeMo at runtime:

```
Audio (16 kHz, mono)
  │
  ▼  pre-emphasis (α=0.97) → STFT (n_fft=512, hop=160, win=400)
     → Mel filterbank (128 bins) → log → per-feature normalisation
  │
  ▼  FastConformer Encoder  (24 layers, d_model=1024, 8 heads)
     └─ ConvSubsampling (3× stride-2 → 8× time reduction)
     └─ RelPositionalEncoding (Transformer-XL style)
     └─ 24 × FastConformerLayer:
           FF₁ (macaron, ×0.5) → Self-Attn (rel-pos) → Conv (k=9) → FF₂ (×0.5) → LN
  │
  ▼  TDT Decoder
     └─ RNNT Prediction network: Embed(8193, 640) + 2-layer LSTM(640)
     └─ Joint network: Linear(1024→640) + Linear(640→640) → ReLU → Linear(640→8198)
     └─ TDT greedy decode (durations [0,1,2,3,4], blank_id=8192)
  │
  ▼  SentencePiece decode → text
```

Weights are loaded directly from the `.nemo` file (a ZIP archive) via `torch._C.PyTorchFileWriter`
without importing any NeMo module at inference time.

### Why does PyTorch-first start faster?

NeMo's first-inference overhead comes from loading the full training framework:

| Source of overhead | NeMo | PyTorch-first |
|---|---|---|
| Framework import (Lightning, Hydra, OmegaConf, …) | ~30s on Jetson | none |
| CUDA kernel compilation (first run) | yes | minimal |
| Audio round-trip via temp WAV file | disk I/O | in-memory tensor |
| `torch.distributed` init (even for single GPU) | yes | skipped |
| SpecAugment / training hooks in eval mode | conditional checks | not present |

### What makes PyTorch-first faster in warm throughput?

Two optimisations close the gap with NeMo and push past it:

| Optimisation | Encoder | Decoder | Effect |
|---|---|---|---|
| fp16 autocast | ✓ | ✗ (slower on Jetson) | tensor cores for the 1024→4096→1024 FFN matmuls × 24 layers |
| CUDA graph | ✗ | ✓ | ~20 kernel launches per decode step → 1 graph replay (1.7× raw GPU speedup per step) |

Component measurements always look better in isolation than in the combined pipeline (thermal throttling, memory bandwidth contention, audio-content-dependent decoder work). The combined result is what the benchmark reports.
