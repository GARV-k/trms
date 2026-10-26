#!/bin/bash
#SBATCH --job-name=sudoku-profile
#SBATCH --account=p26037
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=logs/profile-%j.out
#SBATCH --error=logs/profile-%j.err
set -euo pipefail
export STUDY_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$STUDY_ROOT"
source configs/cluster.env.sh
/usr/bin/python3 scripts/profile_plan.py "$@"
