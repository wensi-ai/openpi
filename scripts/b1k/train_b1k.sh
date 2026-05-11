#!/bin/bash

# Use defaults (pi05_single, 8 GPUs: 0-7)
# ./scripts/b1k/train_b1k.sh
# Specify config
# ./scripts/b1k/train_b1k.sh pi05_i3l_sim
# Specify config and 4 GPUs (7, 6, 5, 4)
# ./scripts/b1k/train_b1k.sh pi05_i3l_sim 4
# Specify config and 4 GPUs on specific devices
# ./scripts/b1k/train_b1k.sh pi05_i3l_sim 4 2,3,4,5
# Resume an existing run by name
# ./scripts/b1k/train_b1k.sh pi05_i3l_sim 4 2,3,4,5 --resume-run openpi_20250101_120000
# ./scripts/b1k/train_b1k.sh pi05_i3l_sim 4 2,3,4,5 --resume openpi_20250101_120000

# uv run scripts/compute_norm_stats.py --config-name pi05_i3l

set -e

CONFIG_NAME=pi05_single
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    CONFIG_NAME=$1
    shift
fi

NUM_GPUS=8
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    NUM_GPUS=$1
    shift
fi

CUDA_VISIBLE_DEVICES=""
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    CUDA_VISIBLE_DEVICES=$1
    shift
fi

RESUME_RUN_NAME=""
TRAIN_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --resume-run)
            if [ $# -lt 2 ]; then
                echo "Error: --resume-run requires a run name"
                exit 1
            fi
            RESUME_RUN_NAME="$2"
            shift 2
            ;;
        --resume)
            if [ $# -ge 2 ] && [[ "$2" != --* ]]; then
                RESUME_RUN_NAME="$2"
                shift 2
            else
                TRAIN_ARGS+=("$1")
                shift
            fi
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    DEFAULT_IDS=(7 6 5 4 3 2 1 0)
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${DEFAULT_IDS[*]:0:$NUM_GPUS}")
fi

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -ne "$NUM_GPUS" ]; then
    echo "Error: CUDA_VISIBLE_DEVICES has $GPU_COUNT GPUs but NUM_GPUS is $NUM_GPUS"
    exit 1
fi

echo "Config name: $CONFIG_NAME"
echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Current time: $(date)"
if [ -n "$RESUME_RUN_NAME" ]; then
    EXP_NAME="$RESUME_RUN_NAME"
    RUN_MODE_ARGS=(--resume)
    echo "Resuming run: $EXP_NAME"
else
    EXP_NAME="openpi_$(date +%Y%m%d_%H%M%S)"
    RUN_MODE_ARGS=(--overwrite)
    echo "Starting run: $EXP_NAME"
fi
echo "Running with args: ${TRAIN_ARGS[*]}"

source /home/ubuntu/jiajun-stanford-lab/Research/openpi/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/b1k/train_b1k.py "$CONFIG_NAME" \
    --exp_name="$EXP_NAME" \
    "${RUN_MODE_ARGS[@]}" \
    --batch_size=64 \
    "${TRAIN_ARGS[@]}"

echo "Training finished."
