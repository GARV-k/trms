#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p wheels sources-deps
# This environment belongs to the LOGIN host, not the ARM compute runtime.
# Do not activate it: exporting its bin directory could leak an x86 Python
# into sbatch jobs. Every download/build command uses its interpreter explicitly.
download_python="${TRM_DOWNLOAD_PYTHON:-python3}"
"$download_python" -c 'import sys; sys.exit("Login Python >=3.8 is required; set TRM_DOWNLOAD_PYTHON to a suitable login interpreter.") if sys.version_info < (3,8) else None'
if [ ! -x .venv-download/bin/python ]; then
  "$download_python" -m venv .venv-download
fi
download_python="$PWD/.venv-download/bin/python"
"$download_python" -m pip install --upgrade \
  pip==25.0.1 setuptools==75.3.2 wheel==0.45.1
# ANTLR is pure Python. The optimizer is CUDA code and MUST build on ARM/CUDA.
"$download_python" -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir wheels antlr4-python3-runtime==4.9.3
"$download_python" -m pip download --no-deps --no-binary=:all: --dest sources-deps adam-atan2==0.0.3
# Run on the internet-connected login node after the environment probe records
# the container Python minor version. Default 3.12 must agree with that probe.
minor="${TRM_PYTHON_MINOR:-312}"
"$download_python" -m pip download --only-binary=:all: --platform manylinux2014_aarch64 \
  --platform manylinux_2_28_aarch64 --platform manylinux_2_17_aarch64 \
  --python-version "$minor" --implementation cp --abi "cp$minor" \
  --find-links wheels --dest wheels -r configs/requirements-overlay.txt
