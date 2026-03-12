#!/bin/bash
#SBATCH -A NAISS2025-22-1535
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 02:00:00
#SBATCH -J pathvqa_lora
#SBATCH --output=logs/pathvqa_%j.out
#SBATCH --error=logs/pathvqa_%j.err

set -euo pipefail

echo "JobID=${SLURM_JOB_ID}"
date
hostname

# ---- 1) Environment ----
module purge
module load Python/3.10.8-GCCcore-12.2.0
source /mimer/NOBACKUP/groups/bloom/pathvqa-env/bin/activate

echo "Python path:"
which python
python --version

# ---- 2) Project path ----
cd /cephyr/users/huanli/Alvis/vlm-federated-lora-pathvqa
mkdir -p logs

# ---- 3) GPU debug ----
nvidia-smi || true

# ---- 4) Run ----
python train_qwen_vlm_pathvqa_lora_full.py