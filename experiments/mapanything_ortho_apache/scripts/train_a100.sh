#!/usr/bin/env bash
set -euo pipefail
: "${MANIFEST:?set MANIFEST}"
: "${PREFLIGHT_JSON:?set PREFLIGHT_JSON}"
: "${OUT:?set OUT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u train.py --manifest "$MANIFEST" --preflight-json "$PREFLIGHT_JSON" --out "$OUT" "$@"
