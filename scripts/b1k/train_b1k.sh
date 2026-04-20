#!/bin/bash

# Use defaults (8 GPUs: 0-7)
#./scripts/b1k/train_b1k.sh
# Specify 4 GPUs (7, 6, 5, 4)
# ./scripts/b1k/train_b1k.sh 4
# Specify 4 GPUs on specific devices
# ./scripts/b1k/train_b1k.sh 4 2,3,4,5

# uv run scripts/compute_norm_stats.py --config-name pi05_i3l

set -e

NUM_GPUS=${1:-8}
CUDA_VISIBLE_DEVICES=${2:-}

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    DEFAULT_IDS=(7 6 5 4 3 2 1 0)
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${DEFAULT_IDS[*]:0:$NUM_GPUS}")
fi

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -ne "$NUM_GPUS" ]; then
    echo "Error: CUDA_VISIBLE_DEVICES has $GPU_COUNT GPUs but NUM_GPUS is $NUM_GPUS"
    exit 1
fi

echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Current time: $(date)"
echo "Running with args: $@"

source /home/ubuntu/jiajun-stanford-lab/Research/openpi/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

DATE=$(date +%Y%m%d-%H%M%S)

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/b1k/train_b1k.py pi05_single \
    --exp_name="openpi_$(date +%Y%m%d_%H%M%S)" \
    --overwrite \
    --batch_size=64

echo "Training finished."