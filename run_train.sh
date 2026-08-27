#!/usr/bin/env bash
set -e

# Workaround for an OpenMP duplicate-library crash on this Mac (conflicting libomp
# copies linked into torch/numpy wheels vs the system Accelerate framework) — not a
# real fix, just unblocks training.
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1
# conda run buffers stdout by default, so print() output (epoch progress etc.) doesn't
# show up until the buffer flushes — force unbuffered output so it prints live.
export PYTHONUNBUFFERED=1

conda run --no-capture-output -n uni python -u -m flow_model.train "$@"
