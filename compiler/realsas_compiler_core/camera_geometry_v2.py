from __future__ import annotations

"""Pure camera geometry shared by the current RealSaS V2 product path.

This module contains no slot, attachment, donor, appearance-provenance or runtime
presentation ontology. The schema and numeric projection convention remain identical
to the previously qualified FullSurfaceCameraProjection.v3 contract.
"""

from dataclasses import dataclass
from typing import Mapping
import math

import numpy as np

from .types import QualificationError

FULL_SURFACE_CAMERA_PROJECTION_SCHEMA = "RealSaS.FullSurfaceCameraProjection.v3"


@dataclass(frozen=True)
class CameraProjectionV3:
    view_id: str
    view_index: int
    origin: tuple[float, float, float]
    right: tuple[float, float, float]
    screen_up: tuple[float, float, float]
    forward: tuple[float, float, float]
    half_extent: float
    resolution: int
    schema_version: str = FULL_SURFACE_CAMERA_PROJECTION_SCHEMA


def _unit(value, *, label: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise QualificationError(f"FULL_SURFACE_V3_{label}_INVALID")
    n = float(np.linalg.norm(v))
    if n <= 1.0e-12:
        raise QualificationError(f"FULL_SURFACE_V3_{label}_DEGENERATE")
    return v / n


def qualify_camera_v3(
    camera: Mapping,
    *,
    view_id: str,
    view_index: int,
) -> CameraProjectionV3:
    try:
        raw_index = int(camera.get("view_index", view_index))
        origin = np.asarray(camera["origin"], dtype=np.float64)
        right = _unit(camera["right"], label="CAMERA_RIGHT")
        up = _unit(camera["screen_up"], label="CAMERA_UP")
        forward = _unit(camera["forward"], label="CAMERA_FORWARD")
        half = float(camera["half_extent"])
        resolution = int(camera["resolution"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError("FULL_SURFACE_V3_CAMERA_CONTRACT_INCOMPLETE") from exc
    if raw_index != int(view_index):
        raise QualificationError("FULL_SURFACE_V3_CAMERA_VIEW_INDEX_DRIFT")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise QualificationError("FULL_SURFACE_V3_CAMERA_ORIGIN_INVALID")
    if not math.isfinite(half) or half <= 0.0 or resolution <= 0:
        raise QualificationError("FULL_SURFACE_V3_CAMERA_SCALE_INVALID")
    if abs(float(np.dot(right, up))) > 1.0e-6:
        raise QualificationError(
            "FULL_SURFACE_V3_CAMERA_SCREEN_BASIS_NOT_ORTHOGONAL"
        )
    if (
        abs(float(np.dot(right, forward))) > 1.0e-6
        or abs(float(np.dot(up, forward))) > 1.0e-6
    ):
        raise QualificationError("FULL_SURFACE_V3_CAMERA_FORWARD_NOT_ORTHOGONAL")
    return CameraProjectionV3(
        view_id=str(view_id),
        view_index=int(view_index),
        origin=tuple(map(float, origin)),
        right=tuple(map(float, right)),
        screen_up=tuple(map(float, up)),
        forward=tuple(map(float, forward)),
        half_extent=half,
        resolution=resolution,
    )


def project_points_xyz_v3(
    points_xyz,
    camera: CameraProjectionV3,
) -> np.ndarray:
    p = np.asarray(points_xyz, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise QualificationError("FULL_SURFACE_V3_POINTS_INVALID")
    origin = np.asarray(camera.origin, dtype=np.float64)
    right = np.asarray(camera.right, dtype=np.float64)
    up = np.asarray(camera.screen_up, dtype=np.float64)
    forward = np.asarray(camera.forward, dtype=np.float64)
    d = p - origin[None, :]
    gx = (d @ right) / float(camera.half_extent)
    gy = -(d @ up) / float(camera.half_extent)
    x = (gx + 1.0) * 0.5 * float(camera.resolution)
    y = (gy + 1.0) * 0.5 * float(camera.resolution)
    z = d @ forward
    out = np.stack((x, y, z), axis=1)
    if not np.isfinite(out).all():
        raise QualificationError("FULL_SURFACE_V3_PROJECTED_NONFINITE")
    return out
