from __future__ import annotations

"""Shared fail-closed validators for sealed model materialization inputs.

These helpers validate exact bytes and array shape/dtype/finite contracts. They do
not infer family eligibility, split membership, or training readiness. A notebook or
trainer may consume only manifests already projected from the sealed Corpus370
matrix/rosters/split; missing truth is represented by absent family payloads plus an
explicit MASKED family status, never by numeric zero fill.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SHA256_LEN = 64
ALLOWED_SPLITS = {"TRAIN", "VAL", "GLOBAL_UNSEEN"}
ALLOWED_FAMILY_STATES = {"PASS", "MASKED"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "MATERIALIZATION_JSON_OBJECT_REQUIRED")
    return value


def bound_file(ref: dict[str, Any], *, expected_schema: str | None = None, json_required: bool = False) -> Path:
    require(isinstance(ref, dict), "MATERIALIZATION_FILE_REF_REQUIRED")
    path = Path(str(ref.get("path") or "")).expanduser().resolve()
    digest = str(ref.get("sha256") or "")
    require(path.is_file(), "MATERIALIZATION_BOUND_FILE_MISSING:" + str(path))
    require(len(digest) == SHA256_LEN and sha256(path) == digest, "MATERIALIZATION_BOUND_FILE_DRIFT:" + str(path))
    if expected_schema is not None:
        require(str(ref.get("schema") or "") == expected_schema, "MATERIALIZATION_REF_SCHEMA:" + expected_schema)
    if json_required:
        payload = load_json(path)
        if expected_schema is not None:
            require(payload.get("schema") == expected_schema, "MATERIALIZATION_PAYLOAD_SCHEMA:" + expected_schema)
    return path


def validate_subject_identity(manifest: dict[str, Any]) -> None:
    require(bool(str(manifest.get("subject_id") or "")), "MATERIALIZATION_SUBJECT_ID")
    source_sha = str(manifest.get("source_sha256") or "")
    require(len(source_sha) == SHA256_LEN and all(c in "0123456789abcdef" for c in source_sha), "MATERIALIZATION_SOURCE_SHA256")
    require(str(manifest.get("split") or "") in ALLOWED_SPLITS, "MATERIALIZATION_SPLIT")
    require(manifest.get("eligibility_computed_by_notebook") is False, "MATERIALIZATION_NOTEBOOK_ELIGIBILITY_FORBIDDEN")
    require(manifest.get("training_authorized") is False, "MATERIALIZATION_SUBJECT_PREAUTHORIZED")


def validate_family_status(
    manifest: dict[str, Any],
    *,
    required_pass: Iterable[str],
    allowed_optional: Iterable[str] = (),
) -> dict[str, str]:
    raw = manifest.get("family_status")
    require(isinstance(raw, dict), "MATERIALIZATION_FAMILY_STATUS_REQUIRED")
    statuses = {str(k): str(v) for k, v in raw.items()}
    allowed = set(required_pass) | set(allowed_optional)
    require(set(statuses) == allowed, "MATERIALIZATION_FAMILY_STATUS_KEY_SET")
    require(all(value in ALLOWED_FAMILY_STATES for value in statuses.values()), "MATERIALIZATION_FAMILY_STATUS_VALUE")
    for family in required_pass:
        require(statuses[family] == "PASS", "MATERIALIZATION_REQUIRED_FAMILY_NOT_PASS:" + family)
    return statuses


def require_payload_for_pass(
    statuses: dict[str, str],
    payloads: dict[str, Any],
    family: str,
) -> list[dict[str, Any]]:
    rows = list(payloads.get(family) or ())
    if statuses[family] == "PASS":
        require(rows, "MATERIALIZATION_PASS_FAMILY_PAYLOAD_REQUIRED:" + family)
    else:
        require(not rows, "MATERIALIZATION_MASKED_FAMILY_PAYLOAD_FORBIDDEN:" + family)
    return [dict(row) for row in rows]


def load_npz(ref: dict[str, Any], *, expected_schema: str) -> tuple[Path, dict[str, np.ndarray]]:
    path = bound_file(ref, expected_schema=expected_schema)
    require(path.suffix.lower() == ".npz", "MATERIALIZATION_NPZ_REQUIRED:" + expected_schema)
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    for name, array in arrays.items():
        require(array.dtype != object, "MATERIALIZATION_OBJECT_ARRAY_FORBIDDEN:" + name)
        if np.issubdtype(array.dtype, np.floating):
            require(bool(np.isfinite(array).all()), "MATERIALIZATION_NONFINITE_ARRAY:" + name)
    return path, arrays


def require_shape(array: np.ndarray, *, ndim: int, last_dim: int | None = None, code: str) -> None:
    require(array.ndim == ndim, code + "_NDIM")
    if last_dim is not None:
        require(array.shape[-1] == last_dim, code + "_LAST_DIM")


def validate_boolean_mask(mask: np.ndarray, length: int, code: str) -> np.ndarray:
    require(mask.shape == (length,), code + "_SHAPE")
    if mask.dtype != np.bool_:
        unique = np.unique(mask)
        require(set(unique.tolist()).issubset({0, 1, False, True}), code + "_VALUES")
        mask = mask.astype(bool)
    return mask


def validate_simplex_rows(weights: np.ndarray, valid: np.ndarray, *, tolerance: float = 1.0e-6) -> None:
    require(weights.ndim == 2 and len(weights) == len(valid), "MATERIALIZATION_SKIN_WEIGHT_SHAPE")
    require(bool(np.isfinite(weights).all()), "MATERIALIZATION_SKIN_NONFINITE")
    require(bool(np.all(weights[valid] >= -tolerance)), "MATERIALIZATION_SKIN_NEGATIVE")
    if np.any(valid):
        sums = weights[valid].sum(axis=1)
        require(bool(np.all(np.abs(sums - 1.0) <= tolerance)), "MATERIALIZATION_SKIN_SIMPLEX")
    # Invalid rows are absent supervision. Their numeric contents are not targets.
