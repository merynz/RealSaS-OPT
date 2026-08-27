#!/usr/bin/env bash
set -euo pipefail
UPSTREAM_URL="https://github.com/facebookresearch/map-anything.git"
UPSTREAM_COMMIT="3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
DEST="${1:-third_party/map-anything}"
if [[ -e "$DEST" ]]; then echo "Refusing to overwrite $DEST" >&2; exit 2; fi
git clone "$UPSTREAM_URL" "$DEST"
git -C "$DEST" checkout --detach "$UPSTREAM_COMMIT"
ACTUAL="$(git -C "$DEST" rev-parse HEAD)"
[[ "$ACTUAL" == "$UPSTREAM_COMMIT" ]] || { echo "commit mismatch" >&2; exit 3; }
python -m pip install -e "$DEST"
python - <<'PY'
import mapanything, uniception
from mapanything.models import MapAnything
print("MapAnything import: PASS")
print("UniCeption import: PASS")
PY
echo "Pinned upstream ready at $DEST"
