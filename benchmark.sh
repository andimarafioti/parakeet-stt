#!/bin/bash
# Benchmark NeMo baseline vs PyTorch-first Parakeet TDT
# Usage: ./benchmark.sh <audio_file>
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PY="$DIR/.venv/bin/python"
AUDIO="${1:-}"

if [ -z "$AUDIO" ]; then
    echo "Usage: ./benchmark.sh <audio_file>"
    echo "Example: ./benchmark.sh sample.wav"
    exit 1
fi

if [ ! -f "$PY" ]; then
    echo "ERROR: venv not found. Set up the environment first."
    exit 1
fi

"$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: PyTorch with CUDA required."
    exit 1
}

GPU=$("$PY" -c 'import torch; print(torch.cuda.get_device_name(0))')

echo "=== Parakeet TDT Benchmark ==="
echo "GPU:     $GPU"
echo "PyTorch: $("$PY" -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $("$PY" -c 'import torch; print(torch.version.cuda)')"
echo "Audio:   $AUDIO"
echo ""

TMPDIR_BENCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BENCH"' EXIT

echo "--- Baseline (NeMo) ---"
echo "(Loading NeMo framework, takes ~30s…)"
"$PY" "$DIR/benchmark.py" "$AUDIO" 2>/dev/null | tee "$TMPDIR_BENCH/nemo.txt"
echo ""

echo "--- PyTorch-first ---"
"$PY" "$DIR/benchmark_pt.py" "$AUDIO" --runs 5 2>/dev/null | tee "$TMPDIR_BENCH/pt.txt"
echo ""

# Summary table
"$PY" - "$TMPDIR_BENCH/nemo.txt" "$TMPDIR_BENCH/pt.txt" "$GPU" << 'PYEOF'
import sys, re

nemo_file, pt_file, gpu = sys.argv[1], sys.argv[2], sys.argv[3]

def parse(text, key):
    m = re.search(rf'{key}=([0-9.]+)', text)
    return float(m.group(1)) if m else None

nemo = open(nemo_file).read()
pt   = open(pt_file).read()

nr  = parse(nemo, 'RTF')
pr  = parse(pt,   'RTF')
nt  = parse(nemo, 'time_s')
pt_ = parse(pt,   'time_s')
a   = parse(nemo, 'audio_s') or parse(pt, 'audio_s')
sp  = pr / nr if (nr and pr) else 0

print(f"=== Results: {gpu} ===")
print(f"{'':20} {'NeMo':>14} {'nano-parakeet':>14} {'Speedup':>9}")
print(f"{'─'*59}")
print(f"{'RTF':20} {nr:>13.1f}x {pr:>13.1f}x {sp:>8.1f}x")
print(f"{'Inference time':20} {nt:>13.2f}s {pt_:>13.4f}s")
print(f"{'Audio duration':20} {a:>13.1f}s")
PYEOF
