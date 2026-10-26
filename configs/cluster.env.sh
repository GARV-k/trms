#!/bin/bash
export STUDY_ROOT="${STUDY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TRM_CONTAINER="${TRM_CONTAINER:-/work/conteneurs/shared/AI/nemo_25.04.03_arm.sif}"
# This account owns its own environment; no path to p26037shrm is used.
export TRM_PYTHON="$STUDY_ROOT/.venv-arm/bin/python"
export TRM_EXPECT_GPU=GH200
export PYTHONPATH="$STUDY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$STUDY_ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$STUDY_ROOT/.cache/triton"
export PYTORCH_KERNEL_CACHE_PATH="$STUDY_ROOT/.cache/torch/kernels"
export TORCH_LOGS="${TORCH_LOGS:-graph_breaks,recompiles,perf_hints}"
export NCCL_SOCKET_FAMILY=AF_INET TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"
