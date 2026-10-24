from __future__ import annotations

"""Deterministic SourceObservation.v1 materialization from admitted source truth.

This is an ingress codec, not a model, stage qualification, or subject-specific
converter. It reads only the exact hash-bound observation section already used
by the Compiler: eight source rasters, eight foreground masks, and the camera
projection bundle. The resulting payload is suitable for external-input import
and later consumption by product.input/product.mesh.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
    SOURCE_OBSERVATION_SCHEMA_V1,
    decode_source_observation_arrays_v1,
    derive_foreground_masks_v1,
    encode_source_observation_arrays_v1,
)


@dataclass(frozen=True)
class SourceObservationMaterializationV1:
    payload: bytes
    diagnostics: dict[str, Any]
    schema: str = "RealSaS.SourceObservationMaterialization.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_ref(raw: Mapping[str, Any], *, label: str) -> tuple[Path, str]:
    path = Path(str(raw.get("path") or "")).expanduser().resolve()
    expected = str(raw.get("sha256") or raw.get("expected_sha256") or "")
    if not path.is_file() or len(expected) != 64:
        raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_REF_INVALID:{label}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_REF_DRIFT:{label}")
    return path, expected


def _view_rows(observation: Mapping[str, Any], key: str, member: str) -> dict[int, tuple[Path, str]]:
    rows = observation.get(key)
    if not isinstance(rows, list) or len(rows) != 8:
        raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_VIEW_CARDINALITY:{key}")
    output: dict[int, tuple[Path, str]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_VIEW_ROW_INVALID:{key}")
        try:
            view_index = int(raw.get("view_index"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_VIEW_INDEX_INVALID:{key}") from exc
        if view_index in output or view_index not in range(8):
            raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_VIEW_SET_INVALID:{key}")
        ref = raw.get(member)
        if not isinstance(ref, Mapping):
            raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_VIEW_REF_INVALID:{key}:{view_index}")
        output[view_index] = _exact_ref(ref, label=f"{key}:{view_index}")
    if set(output) != set(range(8)):
        raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_VIEW_SET_INVALID:{key}")
    return output


def materialize_source_observation_from_manifest_v1(
    run_manifest: Mapping[str, Any],
) -> SourceObservationMaterializationV1:
    observation = run_manifest.get("observation")
    if not isinstance(observation, Mapping):
        raise RuntimeError("SOURCE_OBSERVATION_MATERIALIZE_OBSERVATION_REQUIRED")

    camera_raw = observation.get("camera_bundle")
    if not isinstance(camera_raw, Mapping):
        raise RuntimeError("SOURCE_OBSERVATION_MATERIALIZE_CAMERA_REF_REQUIRED")
    camera_path, camera_sha = _exact_ref(camera_raw, label="camera_bundle")
    try:
        camera_bundle = json.loads(camera_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("SOURCE_OBSERVATION_MATERIALIZE_CAMERA_JSON_INVALID") from exc
    if not isinstance(camera_bundle, dict) or camera_bundle.get("schema") != "RealSaS.CameraProjectionBundle.v1":
        raise RuntimeError("SOURCE_OBSERVATION_MATERIALIZE_CAMERA_SCHEMA_INVALID")

    rasters = _view_rows(observation, "source_rasters", "image")
    masks = _view_rows(observation, "source_foreground_masks", "mask")

    rgba_views: list[np.ndarray] = []
    expected_mask_sha: dict[int, str] = {}
    raster_sha: dict[int, str] = {}
    for view_index in range(8):
        raster_path, raster_digest = rasters[view_index]
        mask_path, mask_digest = masks[view_index]
        with Image.open(raster_path) as image:
            image.load()
            if tuple(map(int, image.size)) != (1024, 1024):
                raise RuntimeError(
                    f"SOURCE_OBSERVATION_MATERIALIZE_RASTER_RESOLUTION_INVALID:{view_index}"
                )
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        if rgba.shape != (1024, 1024, 4):
            raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_RGBA_SHAPE_INVALID:{view_index}")

        raw_mask = mask_path.read_bytes()
        if len(raw_mask) != 1024 * 1024 or any(value not in (0, 1) for value in raw_mask):
            raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_MASK_INVALID:{view_index}")
        if hashlib.sha256(raw_mask).hexdigest() != mask_digest:
            raise RuntimeError(f"SOURCE_OBSERVATION_MATERIALIZE_MASK_DRIFT:{view_index}")

        rgba_views.append(np.moveaxis(rgba, -1, 0).copy())
        raster_sha[view_index] = raster_digest
        expected_mask_sha[view_index] = mask_digest

    native_rgba = np.stack(rgba_views, axis=0)[None, ...]
    payload = encode_source_observation_arrays_v1(
        native_rgba=native_rgba,
        camera_bundle=camera_bundle,
    )
    decoded = decode_source_observation_arrays_v1(payload)
    derive_foreground_masks_v1(decoded, expected_sha256_by_view=expected_mask_sha)
    payload_sha = hashlib.sha256(payload).hexdigest()

    diagnostics = {
        "schema": "RealSaS.SourceObservationMaterializationDiagnostics.v1",
        "output_schema": SOURCE_OBSERVATION_SCHEMA_V1,
        "output_content_sha256": payload_sha,
        "output_size_bytes": len(payload),
        "camera_bundle_sha256": camera_sha,
        "source_raster_sha256_by_view": {str(key): raster_sha[key] for key in range(8)},
        "foreground_mask_sha256_by_view": {
            str(key): expected_mask_sha[key] for key in range(8)
        },
        "foreground_authority_check": "PASS_ALPHA_GT_ZERO_EQUALS_EXACT_SOURCE_MASK",
        "source_truth_rewritten": False,
        "model_inference_executed": False,
        "qualification_minted": False,
        "product_authority_minted": False,
        "subject_specific_logic": False,
    }
    return SourceObservationMaterializationV1(payload=payload, diagnostics=diagnostics)


__all__ = [
    "SourceObservationMaterializationV1",
    "materialize_source_observation_from_manifest_v1",
]
