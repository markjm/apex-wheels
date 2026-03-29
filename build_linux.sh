#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")

PYTHON_VERSION=$1
TORCH_VERSION=$2
CUDA_VERSION=$3
APEX_COMMIT=${4:-master}

echo "============================================="
echo " Building NVIDIA Apex wheel"
echo "============================================="
echo "  Python:  $PYTHON_VERSION"
echo "  PyTorch: $TORCH_VERSION"
echo "  CUDA:    $CUDA_VERSION"
echo "  Apex:    $APEX_COMMIT"
echo "============================================="

MATRIX_CUDA_VERSION=$(echo "$CUDA_VERSION" | awk -F. '{print $1 $2}')
MATRIX_TORCH_VERSION=$(echo "$TORCH_VERSION" | awk -F. '{print $1 "." $2}')

echo "Derived versions:"
echo "  CUDA matrix tag:  $MATRIX_CUDA_VERSION"
echo "  Torch matrix tag: $MATRIX_TORCH_VERSION"

# ---------------------------------------------------------------------------
# Install CUDA toolkit if nvcc is not already available
# ---------------------------------------------------------------------------
if ! command -v nvcc &>/dev/null; then
  echo "nvcc not found — installing CUDA ${CUDA_VERSION} via setup_cuda.py ..."
  python "$SCRIPT_DIR/scripts/setup_cuda.py" "$CUDA_VERSION"
  # Re-source env vars that setup_cuda.py wrote for GitHub Actions, or
  # fall back to the default CUDA path.
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# ---------------------------------------------------------------------------
# Install PyTorch from the matching CUDA index
# ---------------------------------------------------------------------------
TORCH_CUDA_VERSION=$(python -m scripts.coverage_matrix torch-cuda "$MATRIX_CUDA_VERSION" "$MATRIX_TORCH_VERSION")

echo "Installing PyTorch ${TORCH_VERSION}+cu${TORCH_CUDA_VERSION} ..."
if [[ $TORCH_VERSION == *"dev"* ]]; then
  pip install --force-reinstall --no-cache-dir --pre "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/nightly/cu${TORCH_CUDA_VERSION}"
else
  pip install --force-reinstall --no-cache-dir "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION}"
fi

# ---------------------------------------------------------------------------
# Verify the toolchain
# ---------------------------------------------------------------------------
echo "Verifying installations ..."
nvcc --version
python -V
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.version.cuda)"
python -c "from torch.utils import cpp_extension; print('CUDA_HOME:', cpp_extension.CUDA_HOME)"

# ---------------------------------------------------------------------------
# Clone apex
# ---------------------------------------------------------------------------
if [ ! -d apex ]; then
  echo "Cloning NVIDIA/apex at ${APEX_COMMIT} ..."
  git clone https://github.com/NVIDIA/apex.git apex
fi
git -C apex fetch --tags --force
git -C apex checkout "$APEX_COMMIT"
git -C apex submodule update --init --recursive

# ---------------------------------------------------------------------------
# Determine wheel version from apex's git tag
#
# Apex uses YY.MM tags (e.g. 25.09) matching NGC container releases.
# setup.py is hardcoded to version="0.1" and never changes, so we
# derive a meaningful version from git instead:
#   - On a tagged commit:  25.09+cu128torch2.9
#   - Off a tag:           25.09.dev3+cu128torch2.9  (3 commits past tag)
# ---------------------------------------------------------------------------
LOCAL_VERSION_LABEL="cu${MATRIX_CUDA_VERSION}torch${MATRIX_TORCH_VERSION}"

APEX_TAG=$(git -C apex describe --tags --abbrev=0 2>/dev/null || echo "")
APEX_DISTANCE=$(git -C apex rev-list "${APEX_TAG:+${APEX_TAG}..HEAD}" --count 2>/dev/null || echo "0")
APEX_SHORT_SHA=$(git -C apex rev-parse --short=7 HEAD)

if [[ -n "$APEX_TAG" && "$APEX_DISTANCE" == "0" ]]; then
  APEX_BASE_VERSION="$APEX_TAG"
elif [[ -n "$APEX_TAG" ]]; then
  APEX_BASE_VERSION="${APEX_TAG}.dev${APEX_DISTANCE}"
else
  APEX_BASE_VERSION="0.0.dev$(date +%Y%m%d)"
fi

export APEX_VERSION="${APEX_BASE_VERSION}+${LOCAL_VERSION_LABEL}"
echo "Wheel version: $APEX_VERSION  (apex commit: $APEX_SHORT_SHA)"

# Apply patches from patches/ — makes setup.py read APEX_VERSION env var
for patch in "$SCRIPT_DIR"/patches/*.patch; do
  [ -f "$patch" ] || continue
  echo "Applying patch: $(basename "$patch")"
  git -C apex apply "$patch"
done

# ---------------------------------------------------------------------------
# Determine build parallelism from system resources
#
# Apex recommends (README + PR #1882):
#   NVCC_APPEND_FLAGS="--threads N"  — nvcc-internal thread parallelism
#   APEX_PARALLEL_BUILD=M            — parallel extension builds
#   MAX_JOBS x NVCC_THREADS should stay ≤ nproc and within RAM budget
#   (~2.5 GB per MAX_JOBS × NVCC_THREADS slot)
# ---------------------------------------------------------------------------
NUM_THREADS=$(nproc)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "System resources: ${NUM_THREADS} CPU threads, ${RAM_GB} GB RAM"

if [[ -z "${MAX_JOBS:-}" && -z "${NVCC_THREADS:-}" ]]; then
  MAX_PRODUCT_CPU=$NUM_THREADS
  MAX_PRODUCT_RAM=$(awk -v ram="$RAM_GB" 'BEGIN {print int(ram / 2.5)}')
  MAX_PRODUCT=$((MAX_PRODUCT_CPU < MAX_PRODUCT_RAM ? MAX_PRODUCT_CPU : MAX_PRODUCT_RAM))

  BASE=$(awk -v m="$MAX_PRODUCT" 'BEGIN {print int(sqrt(m))}')

  if (( RAM_GB <= 16 )); then
    NVCC_THREADS=1
    MAX_JOBS=2
  elif (( BASE <= 4 )); then
    NVCC_THREADS=$BASE
    MAX_JOBS=$BASE
  else
    NVCC_THREADS=4
    MAX_JOBS=$((MAX_PRODUCT / NVCC_THREADS))
  fi

  MAX_JOBS=$((MAX_JOBS < 1 ? 1 : MAX_JOBS))
  NVCC_THREADS=$((NVCC_THREADS < 1 ? 1 : NVCC_THREADS))
fi
MAX_JOBS=${MAX_JOBS:-4}
NVCC_THREADS=${NVCC_THREADS:-4}

echo "Build parallelism: MAX_JOBS=$MAX_JOBS  NVCC_THREADS=$NVCC_THREADS"

# ---------------------------------------------------------------------------
# Verify ninja is available (Apex README: "We recommend installing Ninja
# to make compilation faster")
# ---------------------------------------------------------------------------
if command -v ninja &>/dev/null; then
  echo "Ninja found: $(ninja --version)"
else
  echo "WARNING: ninja not found — build will be slower"
fi

# ---------------------------------------------------------------------------
# Build the wheel with all extensions enabled
# ---------------------------------------------------------------------------
cd apex

export APEX_CPP_EXT=1
export APEX_CUDA_EXT=1
export APEX_ALL_CONTRIB_EXT=1
export APEX_PARALLEL_BUILD=${MAX_JOBS}
export MAX_JOBS=${MAX_JOBS}
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+${NVCC_APPEND_FLAGS} }--threads ${NVCC_THREADS}"

echo "Building wheel (all extensions) ..."
echo "  APEX_PARALLEL_BUILD=$APEX_PARALLEL_BUILD"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  NVCC_APPEND_FLAGS=$NVCC_APPEND_FLAGS"
time python setup.py bdist_wheel --dist-dir=dist

WHEEL_NAME=$(basename "$(ls dist/*.whl | head -n 1)")
echo "============================================="
echo " Built wheel: $WHEEL_NAME"
echo "============================================="
