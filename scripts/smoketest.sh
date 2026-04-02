#!/bin/bash
# Quick RunPod smoke test: libraries, I/O, memory, disk, model loading.
# Run this first before any experiments.
#
# Usage: bash scripts/smoketest.sh

set -e

echo "=========================================="
echo "RunPod Smoke Test"
echo "=========================================="
echo ""

# 1. System info
echo "=== System ==="
echo "Hostname: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  NO GPU DETECTED"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | awk '/Mem:/ {print $2}') total, $(free -h | awk '/Mem:/ {print $7}') available"
echo "Disk: $(df -h /workspace | awk 'NR==2 {print $4}') free on /workspace"
echo ""

# 2. Python + libraries
echo "=== Python Libraries ==="
cd /workspace/strict-ft-eval
python3 -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers; print(f'Transformers: {transformers.__version__}')
import peft; print(f'PEFT: {peft.__version__}')

try:
    import llguidance; print(f'llguidance: available')
except ImportError:
    print('llguidance: NOT INSTALLED — constrained decoding will fail')
"
echo ""

# 3. Data check
echo "=== Data ==="
for f in data/Restaurants_1_train.jsonl data/Restaurants_1_test.jsonl \
         data/Flights_1_train.jsonl data/Flights_1_test.jsonl \
         data/Restaurants_1_schema.json data/Flights_1_schema.json; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f" 2>/dev/null || echo "?")
        echo "  OK  $f ($lines lines)"
    else
        echo "  MISSING  $f"
    fi
done
echo ""

# 4. Model download check (don't download, just check cache)
echo "=== Model Cache ==="
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
echo "HF_HOME=$HF_HOME"
for model in "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-32B-Instruct"; do
    python3 -c "
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained('$model', local_files_only=True)
    print(f'  CACHED  $model')
except:
    print(f'  NOT CACHED  $model (will download on first use)')
" 2>/dev/null
done
echo ""

# 5. Quick GPU training test (tiny model, 1 step)
echo "=== GPU Training Test ==="
python3 -c "
import torch
if not torch.cuda.is_available():
    print('  SKIP — no GPU')
else:
    # Quick matmul to verify CUDA works
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    c = a @ b
    print(f'  CUDA matmul OK ({c.shape}, device={c.device})')

    # Quick memory check
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f'  VRAM after test: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved')
"
echo ""

# 6. Quick decomposition test (no model, just grammar role assignment)
echo "=== Decomposition Logic Test ==="
python3 -c "
import json, sys
sys.path.insert(0, '.')

# Test grammar role assignment without loading a model
# (import just the role assignment function)
from src.decompose import assign_grammar_roles, GrammarRole

with open('data/Restaurants_1_schema.json') as f:
    schema = json.load(f)

test_json = json.dumps({'city': 'Test', 'cuisine': 'Italian', 'has_live_music': 'True'}, indent=2)
roles = assign_grammar_roles(test_json, schema)

from collections import Counter
counts = Counter(r.name for r in roles)
print(f'  Roles assigned: {dict(counts)}')
print(f'  Total chars: {len(roles)}')
print(f'  OK')
"
echo ""

echo "=========================================="
echo "Smoke test complete."
echo "=========================================="
