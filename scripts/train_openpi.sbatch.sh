#!/bin/bash
#SBATCH --job-name="train_openpi"
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=384G
#SBATCH --cpus-per-task=28
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/sc/train_openpi_%j.out
#SBATCH --error=outputs/sc/train_openpi_%j.err

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /vision/u/$(whoami)/libs/openpi/.venv/bin/activate

echo "Current time: $(date)"
echo "Running with args: $@"

uv run scripts/train_val.py pi0_b1k \
    --exp_name="openpi_$(date +%Y%m%d_%H%M%S)" \
    --overwrite \
    --batch_size=64 \
    --num_train_steps=50000 \
    --weight_loader.params_path=gs://openpi-assets/checkpoints/pi0_base/params

echo "Job finished."
exit 0
