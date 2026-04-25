#!/bin/bash
# Run experiments using slotloss for all evaluations.
# Demonstrates the tool on real fine-tuning runs.
#
# Usage:
#   bash scripts/run_with_slotloss.sh all     # all scales (32B first)
#   bash scripts/run_with_slotloss.sh 32b
#   bash scripts/run_with_slotloss.sh 7b
#   bash scripts/run_with_slotloss.sh 05b

set -e

cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf_cache}

EPOCHS=10
DEVICE=cuda

mkdir -p results checkpoints

# ---------------------------------------------------------------------------
# Train standard LoRA, then evaluate with slotloss
# ---------------------------------------------------------------------------

run_scale() {
    local model=$1 scale=$2

    echo ""
    echo "============================================================"
    echo "  $scale — $model"
    echo "============================================================"

    # --- Train on Restaurants_1 ---
    echo ""
    echo ">>> [$scale] Training: Restaurants_1 ($EPOCHS epochs)"
    python src/train.py \
        --model "$model" \
        --data data/Restaurants_1_train.jsonl \
        --epochs $EPOCHS \
        --device $DEVICE \
        --checkpoint-dir checkpoints \
        --checkpoint-prefix "${scale}_restaurants_lora" \
        $([ "$scale" = "32b" ] && echo "--gradient-checkpointing --max-seq-len 1024")

    # --- Evaluate Restaurants_1 with slotloss ---
    echo ""
    echo ">>> [$scale] slotloss: Restaurants_1"
    slotloss \
        --model "$model" \
        --checkpoint "checkpoints/${scale}_restaurants_lora_epoch${EPOCHS}" \
        --schema data/Restaurants_1_schema.json \
        --data data/Restaurants_1_test.jsonl \
        --device $DEVICE \
        --output "results/${scale}_slotloss_restaurants.json"

    # --- Train on Flights_1 ---
    echo ""
    echo ">>> [$scale] Training: Flights_1 ($EPOCHS epochs)"
    python src/train.py \
        --model "$model" \
        --data data/Flights_1_train.jsonl \
        --epochs $EPOCHS \
        --device $DEVICE \
        --checkpoint-dir checkpoints \
        --checkpoint-prefix "${scale}_flights_lora" \
        $([ "$scale" = "32b" ] && echo "--gradient-checkpointing --max-seq-len 1024")

    # --- Evaluate Flights_1 with slotloss ---
    echo ""
    echo ">>> [$scale] slotloss: Flights_1"
    slotloss \
        --model "$model" \
        --checkpoint "checkpoints/${scale}_flights_lora_epoch${EPOCHS}" \
        --schema data/Flights_1_schema.json \
        --data data/Flights_1_test.jsonl \
        --device $DEVICE \
        --output "results/${scale}_slotloss_flights.json"

    echo ""
    echo ">>> [$scale] Complete."
}

MODEL_32B="Qwen/Qwen2.5-32B-Instruct"
MODEL_7B="Qwen/Qwen2.5-7B-Instruct"
MODEL_05B="Qwen/Qwen2.5-0.5B-Instruct"

CMD=${1:-all}

case $CMD in
    32b)  run_scale "$MODEL_32B" "32b" ;;
    7b)   run_scale "$MODEL_7B" "7b" ;;
    05b)  run_scale "$MODEL_05B" "05b" ;;
    all)
        echo "Running all scales: 32B -> 7B -> 0.5B"
        run_scale "$MODEL_32B" "32b"
        run_scale "$MODEL_7B" "7b"
        run_scale "$MODEL_05B" "05b"
        ;;
    *)
        echo "Usage: bash scripts/run_with_slotloss.sh {all|32b|7b|05b}"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Reports:"
ls -la results/*_slotloss_*.json 2>/dev/null
echo ""
echo "View any report:"
echo "  cat results/32b_slotloss_flights.json | python -m json.tool"
