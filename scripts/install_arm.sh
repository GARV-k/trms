#!/bin/bash
set -euo pipefail
cd "${STUDY_ROOT:?}"
[[ "$(uname -m)" = aarch64 ]]
python3 -c 'import sys,torch; assert sys.version_info[:2]==(3,12), sys.version; assert torch.cuda.is_available(); print(sys.version,torch.__version__,torch.version.cuda)'
if [[ -f .venv-arm/pyvenv.cfg ]]; then
  if ! grep -qx 'include-system-site-packages = true' .venv-arm/pyvenv.cfg ||
     ! .venv-arm/bin/python -c 'import pip; import pip._vendor.rich.themes'; then
    rm -rf .venv-arm
  fi
fi
if [[ ! -f .venv-arm/pyvenv.cfg ]]; then
  python3 -m venv --system-site-packages .venv-arm
fi
"$TRM_PYTHON" -m pip install --no-index --find-links wheels -r configs/requirements-overlay.txt
command -v nvcc
export MAX_JOBS=4
"$TRM_PYTHON" -m pip install --no-index --no-deps --no-build-isolation sources-deps/adam_atan2-0.0.3.tar.gz
"$TRM_PYTHON" -c 'import torch,adam_atan2,numpy,matplotlib; print(torch.cuda.get_device_name())'
"$TRM_PYTHON" -m pip freeze > manifests/cluster-environment.txt
"$TRM_PYTHON" scripts/verify_vendor.py
if [[ -f data/paper/sudoku-extreme-1k-aug-800/DATA_READY.json ]]; then
  "$TRM_PYTHON" -m scripts.repro_data audit --dataset sudoku
else
  "$TRM_PYTHON" -m scripts.repro_data build-sudoku
fi
