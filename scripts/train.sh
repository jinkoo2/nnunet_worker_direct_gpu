#!/bin/bash
# nnUNetv2_train wrapper — single fold
# Usage: train.sh <DATASET_ID> <CONFIGURATION> <FOLD>
# Env vars expected: NNUNET_DATA_DIR, CONDA_ENV, CONDA_PROFILE, DEVICE, NUM_GPUS

set -e

DATASET_ID="$1"
CONFIGURATION="$2"
FOLD="$3"

if [ -z "$DATASET_ID" ] || [ -z "$CONFIGURATION" ] || [ -z "$FOLD" ]; then
    echo "Usage: train.sh <DATASET_ID> <CONFIGURATION> <FOLD>"
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
echo "Dataset:             $DATASET_ID"
echo "Configuration:       $CONFIGURATION"
echo "Fold:                $FOLD"

# Conda activation (optional)
if [ -n "$CONDA_ENV" ] && [ -f "${CONDA_PROFILE:-/opt/miniconda3/etc/profile.d/conda.sh}" ]; then
    . "${CONDA_PROFILE:-/opt/miniconda3/etc/profile.d/conda.sh}"
    conda activate "$CONDA_ENV"
    echo "Activated conda env: $CONDA_ENV"
fi

echo ""
echo "=== nnUNetv2_train: $DATASET_ID $CONFIGURATION fold $FOLD ==="
echo ""

nnUNetv2_train \
    -device "${DEVICE:-cuda}" \
    -num_gpus "${NUM_GPUS:-1}" \
    --c \
    "$DATASET_ID" \
    "$CONFIGURATION" \
    "$FOLD"

STATUS=$?
echo ""
echo "Exit code: $STATUS"
exit $STATUS
