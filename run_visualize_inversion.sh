#!/usr/bin/env bash
set -e

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

conda run --no-capture-output -n uni python -u -m flow_model.visualize_inversion "$@"
