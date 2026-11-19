#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$PROJECT_DIR/.cache"
DEPS_DIR="$PROJECT_DIR/.deps"
ENV_NAME="policy-consensus"

if [[ "${OSTYPE:-}" != linux-gnu* ]]; then
  echo "This setup script requires Linux." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Initialize Miniconda, then rerun this script." >&2
  exit 1
fi

ENV_FILE="$PROJECT_DIR/conda_environment.yaml"
case "${1:-}" in
  "") ;;
  --cluster) ENV_FILE="$PROJECT_DIR/conda_environment_cluster.yaml" ;;
  --real) ENV_FILE="$PROJECT_DIR/conda_environment_real.yaml" ;;
  *) echo "Usage: $0 [--cluster|--real]" >&2; exit 2 ;;
esac

# Keep Conda packages, pip wheels, source-build scratch files, and downloads
# inside NEUZZ. The Conda environment itself remains managed by Miniconda.
mkdir -p "$CACHE_DIR/conda/pkgs" "$CACHE_DIR/pip" "$CACHE_DIR/tmp" "$DEPS_DIR"
export CONDA_PKGS_DIRS="$CACHE_DIR/conda/pkgs"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-1800}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
export TMPDIR="$CACHE_DIR/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Creating $ENV_NAME with $ENV_FILE ..."
  conda env create -f "$ENV_FILE"
else
  echo "Updating existing $ENV_NAME from $ENV_FILE ..."
  conda env update -n "$ENV_NAME" -f "$ENV_FILE"
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-4}"

PYTORCH3D_DIR="$DEPS_DIR/pytorch3d-0.7.9"
if [[ ! -d "$PYTORCH3D_DIR/.git" ]]; then
  git clone --depth 1 --branch v0.7.9 \
    https://github.com/facebookresearch/pytorch3d.git "$PYTORCH3D_DIR"
fi

echo "Building PyTorch3D for RTX 5090 (SM 12.0) ..."
python -m pip install --no-build-isolation "$PYTORCH3D_DIR"

echo "Installing policyconsensus in editable mode ..."
python -m pip install --no-build-isolation -e "$PROJECT_DIR"

echo "Checking the installed CUDA stack ..."
python - <<'PY'
import torch
import torchvision
import pytorch3d

print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU capability: {torch.cuda.get_device_capability(0)}")
print(f"PyTorch3D: {pytorch3d.__file__}")
PY

echo "Done. Activate the environment with: conda activate $ENV_NAME"
