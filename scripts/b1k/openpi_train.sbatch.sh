#!/bin/bash
#SBATCH --job-name="openpi_train"
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/sc/openpi_%j.log
#SBATCH --error=outputs/sc/openpi_%j.log

# Calculate total GPUs across all nodes
NUM_GPUS=$((${SLURM_GPUS_ON_NODE:-1} * ${SLURM_NNODES:-1}))

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_CPU_PER_TASK="$SLURM_CPUS_PER_TASK
echo "SLURM_MEM_PER_NODE="$SLURM_MEM_PER_NODE
echo "Number of nodes: ${SLURM_NNODES:-1}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE:-1}"
echo "Total GPUs: $NUM_GPUS"
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_NTASKS_PER_NODE"=$SLURM_NTASKS_PER_NODE
echo "working directory="$SLURM_SUBMIT_DIR

source /vision/u/$(whoami)/libs/openpi/.venv/bin/activate

DATE=$(date +%Y%m%d-%H%M%S)

echo "Current time: $(date)"
echo "Running with args: $@"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/b1k/train_b1k.py pi0_b1k \
    --exp_name="openpi_$(date +%Y%m%d_%H%M%S)" \
    --overwrite \
    --batch_size=64 \

echo "Job finished."
exit 0
