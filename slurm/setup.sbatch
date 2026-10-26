#!/bin/bash
#SBATCH --job-name=sudoku-setup
#SBATCH --account=p26037
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err
set -euo pipefail
export STUDY_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$STUDY_ROOT"
source configs/cluster.env.sh
apptainer exec --nv --bind /work --pwd "$STUDY_ROOT" "$TRM_CONTAINER" bash scripts/install_arm.sh
