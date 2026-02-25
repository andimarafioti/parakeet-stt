# nano-parakeet

Pure-PyTorch inference for [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — no NeMo required.

```python
from nano_parakeet import from_pretrained
model = from_pretrained()
print(model.transcribe("audio.wav"))
```

## Why?

The official NeMo inference stack pulls in ~180 packages — PyTorch Lightning, Hydra, OmegaConf, apex, distributed training scaffolding — none of which are needed at inference time. This makes it painful to integrate Parakeet into existing projects: version conflicts, long installs, and a 30-second cold-start on every process launch.

`nano-parakeet` reimplements the full inference pipeline in plain PyTorch. The only dependencies are things you probably already have:

| | nano-parakeet | NeMo |
|---|---|---|
| Dependencies | **5** (torch, numpy, soundfile, sentencepiece, huggingface-hub) | ~180 |
| Cold start | **~3s** (weights only) | ~30s (framework init + CUDA kernel compile) |
| Warm RTF (Jetson AGX Orin) | **93×** | 73× |

Transcriptions are byte-identical to NeMo's output.

## Install

```bash
pip install nano-parakeet
```

Requires Python 3.10+, PyTorch with CUDA, and ffmpeg.

## Usage

### Python API

```python
from nano_parakeet import from_pretrained

model = from_pretrained()                    # downloads ~1.1GB on first run
text = model.transcribe("audio.wav")        # path, numpy array, or tensor
print(text)
```

### CLI

```bash
nano-parakeet audio.wav
# or
python -m nano_parakeet audio.wav
```

Accepts OGG, WAV, M4A, or any format ffmpeg can read.

## Benchmark

RTF > 1.0 = faster than real-time. 5 timed runs after a warm-up; best time reported.

### Warm throughput

| GPU | Audio | NeMo RTF | nano-parakeet RTF | Speedup |
|---|---|---|---|---|
| RTX 4090 | 12s | ~207× | ~519× | **2.5×** |
| Jetson AGX Orin 64GB | 12s | ~84× | ~112× | **1.3×** |

> **Note (RTX 4090):** NeMo is run with `strategy='greedy'` (single-item, not batch).
> The default `greedy_batch` strategy uses TDT label-looping CUDA graphs that fail to compile
> on NeMo 2.6.2 + cuda-python 12.9 (NVRTC is not permitted inside a graph capture context).
> `strategy='greedy'` uses a different CUDA graph path that works fine.

### Cold start (first inference, including framework load)

| GPU | NeMo | nano-parakeet |
|---|---|---|
| RTX 4090 | ~30s | **~3s** |
| Jetson AGX Orin 64GB | ~30s | **~3s** |

Run both yourself:

```bash
git clone https://github.com/andimarafioti/nano-parakeet
cd parakeet-stt
./benchmark.sh sample.wav
```

## How It Works

The full pipeline in plain PyTorch — no NeMo at runtime:

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
           FF₁ (×0.5) → Self-Attn (rel-pos) → Conv (k=9) → FF₂ (×0.5) → LN
  │
  ▼  TDT Decoder
     └─ RNNT Prediction: Embed(8193, 640) + 2-layer LSTM(640)
     └─ Joint: Linear(1024→640) + Linear(640→640) → ReLU → Linear(640→8198)
     └─ TDT greedy decode (durations [0,1,2,3,4], blank_id=8192)
  │
  ▼  SentencePiece decode → text
```

Weights are loaded directly from the `.nemo` file (a ZIP archive) without importing any NeMo module.

### Optimisations

| | Encoder | Decoder | Effect |
|---|---|---|---|
| bfloat16 (auto, Ampere+) | ✓ | ✓ | native low-precision on modern GPUs; fp16 autocast fallback on older devices |
| CUDA graph | ✗ | ✓ | ~20 kernel launches per decode step → 1 graph replay |

## Jetson Setup

The PyPI wheel works on standard x86 CUDA machines. For Jetson (JetPack 6), PyTorch needs to be installed from NVIDIA's distribution first:

```bash
# Install CUDA-enabled PyTorch for JetPack 6
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Then install nano-parakeet (skipping torch since it's already installed)
pip install nano-parakeet --no-deps
pip install numpy soundfile sentencepiece huggingface-hub
```
