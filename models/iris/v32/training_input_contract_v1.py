from __future__ import annotations

"""IRIS V3.2 materialized-input contract for Corpus370.

This module implements the *data-consumer boundary* frozen by
IRIS_TESSA_V32_PRETRUTH_ARCHITECTURE_FREEZE.json. It is not the V3.2 neural model
implementation and does not authorize training. Its job is to make malformed or
false-authority materialization impossible to feed into a future trainer.

In particular:
- source/camera/surface UDF are mandatory PASS families;
- GLOBAL_SOLID_SDF may be MASKED, in which case no signed/occupancy payload exists;
- if GLOBAL_SOLID_SDF is PASS, signed distance and occupancy share one explicit
  validity mask and no invalid row is silently treated as zero supervision;
- oriented normals are consumed only under SURFACE_NORMALS=PASS;
- optional graphics/visibility/compositing/response families require exact bound
  payload bytes when PASS and no payload bytes when MASKED.
"""

from pathlib import Path
from typing import Any

import numpy as np

from models.materialization_contract_v1 import (
    bound_file,
    load_json,
    load_npz,
    require,
    require_payload_for_pass,
    require_shape,
    validate_boolean_mask,
    validate_family_status,
    validate_subject_identity,
)

SCHEMA = "RealSaS.IRISV32MaterializedSubject.v1"
GEOMETRY_SCHEMA = "RealSaS.IRISV32SurfaceGeometryTargets.v1"
GLOBAL_SDF_SCHEMA = "RealSaS.IRISV32GlobalSolidTargets.v1"
CAMERA_SCHEMA = "RealSaS.CameraProjectionBundle.v1"

REQUIRED_PASS = (
    "SOURCE_IMPORT",
    "CAMERA_OBSERVATION",
    "SURFACE_GEOMETRY_UDF",
)
OPTIONAL = (
    "GLOBAL_SOLID_SDF",
    "SURFACE_NORMALS",
    "APPEARANCE_CAA",
    "VISIBILITY_OCCLUSION",
    "RAY_COMPOSITING",
    "CAUSAL_RESPONSE",
)
PAYLOAD_FAMILIES = (
    "APPEARANCE_CAA",
    "VISIBILITY_OCCLUSION",
    "RAY_COMPOSITING",
    "CAUSAL_RESPONSE",
)


def _validate_ref_payload(ref: dict[str, Any], family: str) -> None:
    schema = str(ref.get("schema") or "")
    require(schema, "IRIS_V32_FAMILY_PAYLOAD_SCHEMA:" + family)
    path = bound_file(ref, expected_schema=schema)
    if path.suffix.lower() == ".json":
        payload = load_json(path)
        require(payload.get("schema") == schema, "IRIS_V32_FAMILY_PAYLOAD_JSON_SCHEMA:" + family)


def _validate_observation(manifest: dict[str, Any]) -> dict[str, Any]:
    source = dict(manifest.get("source") or {})
    require(bool(source.get("schema")), "IRIS_V32_SOURCE_REF_SCHEMA")
    bound_file(source, expected_schema=str(source["schema"]))

    observation = dict(manifest.get("observation") or {})
    camera_ref = dict(observation.get("camera_bundle") or {})
    bound_file(camera_ref, expected_schema=CAMERA_SCHEMA, json_required=True)
    rasters = list(observation.get("source_rasters") or ())
    masks = list(observation.get("source_foreground_masks") or ())
    require(len(rasters) == len(masks) == 8, "IRIS_V32_EIGHT_VIEW_OBSERVATION_REQUIRED")
    require([int(row.get("view_index", -1)) for row in rasters] == list(range(8)), "IRIS_V32_RASTER_VIEW_INDEX")
    require([int(row.get("view_index", -1)) for row in masks] == list(range(8)), "IRIS_V32_MASK_VIEW_INDEX")
    for row in rasters:
        ref = dict(row.get("image") or {})
        path = bound_file(ref, expected_schema=str(ref.get("schema") or "RealSaS.SourceRasterRGBA.v1"))
        require(path.suffix.lower() == ".png", "IRIS_V32_SOURCE_RASTER_PNG")
    for row in masks:
        ref = dict(row.get("mask") or {})
        path = bound_file(ref, expected_schema=str(ref.get("schema") or "RealSaS.SourceForegroundMask.v1"))
        require(path.name.endswith(".mask.bin"), "IRIS_V32_SOURCE_MASK_BIN")
    return observation


def _validate_geometry(manifest: dict[str, Any], statuses: dict[str, str]) -> dict[str, int | float]:
    _, arrays = load_npz(dict(manifest.get("surface_geometry") or {}), expected_schema=GEOMETRY_SCHEMA)
    required = {"surface_query_xyz_canonical", "unsigned_surface_distance"}
    require(required.issubset(arrays), "IRIS_V32_SURFACE_GEOMETRY_ARRAYS")
    xyz = np.asarray(arrays["surface_query_xyz_canonical"], dtype=np.float64)
    udf = np.asarray(arrays["unsigned_surface_distance"], dtype=np.float64)
    require_shape(xyz, ndim=2, last_dim=3, code="IRIS_V32_SURFACE_QUERY")
    require(udf.shape == (len(xyz),), "IRIS_V32_SURFACE_UDF_SHAPE")
    require(len(xyz) > 0, "IRIS_V32_SURFACE_UDF_EMPTY")
    require(bool(np.all(udf >= 0.0)), "IRIS_V32_SURFACE_UDF_NEGATIVE")

    normal_keys = {"surface_normal_xyz", "surface_normal_valid"}
    if statuses["SURFACE_NORMALS"] == "PASS":
        require(normal_keys.issubset(arrays), "IRIS_V32_NORMAL_ARRAYS_REQUIRED")
        normals = np.asarray(arrays["surface_normal_xyz"], dtype=np.float64)
        require(normals.shape == (len(xyz), 3), "IRIS_V32_NORMAL_SHAPE")
        valid = validate_boolean_mask(np.asarray(arrays["surface_normal_valid"]), len(xyz), "IRIS_V32_NORMAL_VALID")
        require(bool(np.any(valid)), "IRIS_V32_NORMAL_VALID_EMPTY")
        norms = np.linalg.norm(normals[valid], axis=1)
        require(bool(np.all(np.abs(norms - 1.0) <= 5.0e-3)), "IRIS_V32_NORMAL_NOT_UNIT")
        normal_count = int(np.count_nonzero(valid))
    else:
        require(not (normal_keys & set(arrays)), "IRIS_V32_MASKED_NORMAL_ARRAYS_FORBIDDEN")
        normal_count = 0
    return {"surface_query_count": int(len(xyz)), "normal_valid_count": normal_count}


def _validate_global_sdf(manifest: dict[str, Any], statuses: dict[str, str]) -> dict[str, int]:
    ref = manifest.get("global_solid")
    if statuses["GLOBAL_SOLID_SDF"] == "MASKED":
        require(ref in {None, False}, "IRIS_V32_MASKED_GLOBAL_SDF_PAYLOAD_FORBIDDEN")
        return {"global_sdf_query_count": 0, "global_sdf_valid_count": 0}
    require(isinstance(ref, dict), "IRIS_V32_GLOBAL_SDF_PAYLOAD_REQUIRED")
    _, arrays = load_npz(dict(ref), expected_schema=GLOBAL_SDF_SCHEMA)
    required = {"query_xyz_canonical", "signed_distance", "occupancy", "valid"}
    require(required.issubset(arrays), "IRIS_V32_GLOBAL_SDF_ARRAYS")
    xyz = np.asarray(arrays["query_xyz_canonical"], dtype=np.float64)
    sdf = np.asarray(arrays["signed_distance"], dtype=np.float64)
    occupancy = np.asarray(arrays["occupancy"])
    require_shape(xyz, ndim=2, last_dim=3, code="IRIS_V32_GLOBAL_SDF_QUERY")
    require(sdf.shape == (len(xyz),), "IRIS_V32_GLOBAL_SDF_SHAPE")
    valid = validate_boolean_mask(np.asarray(arrays["valid"]), len(xyz), "IRIS_V32_GLOBAL_SDF_VALID")
    occupancy = validate_boolean_mask(occupancy, len(xyz), "IRIS_V32_GLOBAL_OCCUPANCY")
    require(bool(np.any(valid)), "IRIS_V32_GLOBAL_SDF_VALID_EMPTY")
    # Valid rows must contain real two-class solid supervision. Invalid rows are
    # never targets and may carry arbitrary storage values without affecting loss.
    require(bool(np.any(occupancy[valid])) and bool(np.any(~occupancy[valid])), "IRIS_V32_GLOBAL_OCCUPANCY_CLASSES")
    require(bool(np.all(sdf[valid][occupancy[valid]] <= 1.0e-7)), "IRIS_V32_GLOBAL_SDF_INSIDE_SIGN")
    require(bool(np.all(sdf[valid][~occupancy[valid]] >= -1.0e-7)), "IRIS_V32_GLOBAL_SDF_OUTSIDE_SIGN")
    return {"global_sdf_query_count": int(len(xyz)), "global_sdf_valid_count": int(np.count_nonzero(valid))}


def validate_manifest(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest = load_json(path)
    require(manifest.get("schema") == SCHEMA, "IRIS_V32_MATERIALIZED_SCHEMA")
    validate_subject_identity(manifest)
    statuses = validate_family_status(manifest, required_pass=REQUIRED_PASS, allowed_optional=OPTIONAL)
    _validate_observation(manifest)
    geometry = _validate_geometry(manifest, statuses)
    solid = _validate_global_sdf(manifest, statuses)

    payloads = dict(manifest.get("family_payloads") or {})
    require(set(payloads).issubset(PAYLOAD_FAMILIES), "IRIS_V32_UNKNOWN_FAMILY_PAYLOAD")
    for family in PAYLOAD_FAMILIES:
        rows = require_payload_for_pass(statuses, payloads, family)
        for ref in rows:
            _validate_ref_payload(ref, family)

    authority = dict(manifest.get("authority") or {})
    require(authority.get("matrix_cell_status_copied_not_recomputed") is True, "IRIS_V32_MATRIX_AUTHORITY_BINDING")
    require(authority.get("split_membership_copied_not_recomputed") is True, "IRIS_V32_SPLIT_AUTHORITY_BINDING")
    require(authority.get("missing_family_targets_zero_filled") is False, "IRIS_V32_ZERO_FILL_FORBIDDEN")
    require(authority.get("component_local_sign_used_as_global_sdf") is False, "IRIS_V32_COMPONENT_SIGN_PROMOTION_FORBIDDEN")
    require(authority.get("fstar_used_as_global_sign_authority") is False, "IRIS_V32_FSTAR_SIGN_AUTHORITY_FORBIDDEN")

    return {
        "schema": SCHEMA,
        "subject_id": manifest["subject_id"],
        "split": manifest["split"],
        "family_status": statuses,
        **geometry,
        **solid,
        "consumer_contract": "IRIS_V3_2_STATIC_CORPUS_INPUT_ONLY__MODEL_IMPLEMENTATION_SEPARATE",
        "training_authorized": False,
    }
