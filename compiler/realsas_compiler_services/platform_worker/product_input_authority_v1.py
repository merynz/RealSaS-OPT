from __future__ import annotations

"""Deterministic product-input qualification derived from SourceObservation.v1.

The legacy Compiler DAG materialized camera, observation and normalization
contracts as stages 05/07/08.  Product execution must not require a pre-created
Compiler run merely to obtain those model-independent contracts.  This module
reuses the same canonical IR types and qualification semantics directly from the
immutable SourceObservation bytes.

No learned inference happens here and no source pixels are altered.
"""

from dataclasses import replace
import hashlib
import json
import math
from typing import Any

import numpy as np

from compiler.realsas_compiler_core.camera_authority_v1 import (
    build_qualified_camera_set,
    camera_projection_binding_hash,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.observation_authority_v1 import (
    QualifiedObservationViewIR,
    build_qualified_observation_set,
)
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    NormalizationDomainIR,
    normalization_domain_hash,
)
from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
    SourceObservationArraysV1,
)

PRODUCT_INPUT_AUTHORITY_CONTRACT_V1 = "RealSaS.ProductInputAuthority.v1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _camera_bundle_sha256(camera_bundle: dict[str, Any]) -> str:
    raw = json.dumps(
        camera_bundle,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(raw)


def _require_common_frame(camera_bundle: dict[str, Any]) -> tuple[tuple[float, float, float], float]:
    center = tuple(map(float, camera_bundle.get("common_center_xyz") or ()))
    half = float(camera_bundle.get("common_half_extent") or 0.0)
    if len(center) != 3 or not all(math.isfinite(value) for value in center):
        raise RuntimeError("PRODUCT_INPUT_AUTHORITY_COMMON_CENTER_INVALID")
    if not math.isfinite(half) or half <= 0.0:
        raise RuntimeError("PRODUCT_INPUT_AUTHORITY_COMMON_HALF_EXTENT_INVALID")
    return center, half


def build_product_input_authority_v1(
    source: SourceObservationArraysV1,
    *,
    source_content_sha256: str,
) -> tuple[Any, Any, NormalizationDomainIR]:
    source.validate()
    if len(source_content_sha256) != 64:
        raise RuntimeError("PRODUCT_INPUT_AUTHORITY_SOURCE_SHA256_INVALID")

    camera_bundle = dict(source.camera_bundle)
    center, half_extent = _require_common_frame(camera_bundle)
    camera_rows = tuple(camera_bundle.get("cameras") or ())
    cameras = build_qualified_camera_set(
        camera_rows,
        source_bundle_sha256=_camera_bundle_sha256(camera_bundle),
        metadata={
            "source_path_role": "SOURCE_OBSERVATION_EMBEDDED_CAMERA_BUNDLE",
            "source_observation_content_sha256": source_content_sha256,
            "product_input_authority": PRODUCT_INPUT_AUTHORITY_CONTRACT_V1,
        },
    )
    by_camera = {int(camera.view_index): camera for camera in cameras.cameras}

    rgba = np.asarray(source.native_rgba, dtype=np.uint8)
    rows: list[QualifiedObservationViewIR] = []
    for view_index in range(8):
        view = np.ascontiguousarray(rgba[0, view_index])
        if view.shape != (4, 1024, 1024):
            raise RuntimeError("PRODUCT_INPUT_AUTHORITY_VIEW_SHAPE_INVALID")
        alpha = np.ascontiguousarray(view[3])
        mask = np.ascontiguousarray(alpha > 0, dtype=np.uint8)
        if not bool(mask.any()):
            raise RuntimeError(f"PRODUCT_INPUT_AUTHORITY_FOREGROUND_EMPTY:{view_index}")
        if bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
            raise RuntimeError(f"PRODUCT_INPUT_AUTHORITY_FOREGROUND_TOUCHES_BORDER:{view_index}")

        source_raster_sha256 = _sha256(view.tobytes(order="C"))
        foreground_mask_sha256 = _sha256(mask.tobytes(order="C"))
        camera_hash = camera_projection_binding_hash(by_camera[view_index])
        source_observation_hash = content_sha256(
            {
                "schema": "RealSaS.SourceObservationAuthority.v1",
                "view_index": view_index,
                "source_observation_content_sha256": source_content_sha256,
                "source_raster_sha256": source_raster_sha256,
                "foreground_mask_sha256": foreground_mask_sha256,
                "camera_binding_hash": camera_hash,
            }
        )
        rows.append(
            QualifiedObservationViewIR(
                view_index=view_index,
                width=1024,
                height=1024,
                source_observation_hash=source_observation_hash,
                source_raster_sha256=source_raster_sha256,
                foreground_mask_sha256=foreground_mask_sha256,
                camera_binding_hash=camera_hash,
                qualification_state="PASS",
                evidence_refs=(source_content_sha256, cameras.camera_set_hash),
                metadata={
                    "evidence_semantics": "POSITIVE_FOREGROUND__NEGATIVE_INFRAME_BACKGROUND__UNKNOWN_OUT_OF_FRAME",
                    "foreground_authority": "SOURCE_OBSERVATION_ALPHA_GT_ZERO",
                    "full_subject_border_margin_minimum_px": 1,
                    "product_input_authority": PRODUCT_INPUT_AUTHORITY_CONTRACT_V1,
                },
            )
        )

    observation = build_qualified_observation_set(
        tuple(rows),
        metadata={
            "camera_set_hash": cameras.camera_set_hash,
            "source_observation_content_sha256": source_content_sha256,
            "unknown_cannot_veto_positive": True,
            "product_input_authority": PRODUCT_INPUT_AUTHORITY_CONTRACT_V1,
        },
    )
    normalization = NormalizationDomainIR(
        observation_set_binding_hash=observation.observation_set_hash,
        camera_set_binding_hash=cameras.camera_set_hash,
        center_xyz=center,
        half_extent=half_extent,
        normalized_bounds_min=(-1.0, -1.0, -1.0),
        normalized_bounds_max=(1.0, 1.0, 1.0),
        coordinate_frame="REALSAS_OBJECT_FRAME",
        normalization_hash="",
        metadata={
            "normalization_policy": "SOURCE_OBSERVATION_COMMON_OBJECT_CUBE_V1",
            "camera_refit_performed": False,
            "source_observation_content_sha256": source_content_sha256,
            "product_input_authority": PRODUCT_INPUT_AUTHORITY_CONTRACT_V1,
        },
    )
    normalization = replace(
        normalization,
        normalization_hash=normalization_domain_hash(normalization),
    )
    return cameras, observation, normalization


__all__ = [
    "PRODUCT_INPUT_AUTHORITY_CONTRACT_V1",
    "build_product_input_authority_v1",
]
