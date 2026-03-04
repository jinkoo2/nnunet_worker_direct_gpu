#!/bin/bash
# nnUNetv2_preprocess wrapper
# Usage: preprocess.sh <DATASET_NUM> <LOG_DIR>
# Env vars expected: NNUNET_DATA_DIR, CONDA_ENV, CONDA_PROFILE, NP

set -e

DATASET_NUM="$1"
LOG_DIR="$2"

if [ -z "$DATASET_NUM" ] || [ -z "$LOG_DIR" ]; then
    echo "Usage: preprocess.sh <DATASET_NUM> <LOG_DIR>"
    exit 1
fi

if [ -z "$NNUNET_DATA_DIR" ]; then
    echo "ERROR: NNUNET_DATA_DIR is not set"
    exit 1
fi

# nnUNet path environment
export nnUNet_raw="${NNUNET_DATA_DIR}/raw"
export nnUNet_preprocessed="${NNUNET_DATA_DIR}/preprocessed"
export nnUNet_results="${NNUNET_DATA_DIR}/results"
export TORCH_COMPILE_DISABLE=1

echo "nnUNet_raw:          $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results:      $nnUNet_results"
echo "Dataset num:         $DATASET_NUM"

# Conda activation (optional)
if [ -n "$CONDA_ENV" ] && [ -f "${CONDA_PROFILE:-/home/jk/miniconda3/etc/profile.d/conda.sh}" ]; then
    . "${CONDA_PROFILE:-/home/jk/miniconda3/etc/profile.d/conda.sh}"
    conda activate "$CONDA_ENV"
    echo "Activated conda env: $CONDA_ENV"
fi

mkdir -p "$LOG_DIR"

echo ""
echo "=== nnUNetv2_preprocess: Dataset $DATASET_NUM ==="
echo ""

nnUNetv2_preprocess \
    -d "$DATASET_NUM" \
    -np "${NP:-8}" \
    --verbose

STATUS=$?
echo ""
echo "Exit code: $STATUS"
exit $STATUS
