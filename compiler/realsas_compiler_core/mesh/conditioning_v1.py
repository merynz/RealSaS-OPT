from __future__ import annotations

"""Dimensionless triangle conditioning metrics for canonical product meshes."""

import math
from typing import Iterable

import numpy as np

from compiler.realsas_compiler_core.types import QualificationError


def triangle_rest_metric(points, *, dtype=np.float64) -> dict:
    p = np.asarray(points, dtype=dtype)
    if p.shape != (3, 3) or not np.isfinite(p).all():
        raise QualificationError("MESH_CONDITIONING_TRIANGLE_INVALID")

    edges = (p[1] - p[0], p[2] - p[1], p[0] - p[2])
    lengths = np.asarray([np.linalg.norm(x) for x in edges], dtype=dtype)
    longest = float(lengths.max())
    shortest = float(lengths.min())
    area2 = float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))
    area = 0.5 * area2
    if shortest <= 0.0 or longest <= 0.0 or area2 <= 0.0:
        return {
            "degenerate": True,
            "area": area,
            "local_scale": longest,
            "min_angle_deg": 0.0,
            "aspect_longest_over_min_altitude": float("inf"),
            "relative_double_area": 0.0,
        }

    def angle(a, b):
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        cosine = float(np.dot(a, b)) / (na * nb)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    angles = (
        angle(p[1] - p[0], p[2] - p[0]),
        angle(p[0] - p[1], p[2] - p[1]),
        angle(p[0] - p[2], p[1] - p[2]),
    )
    min_altitude = area2 / longest
    return {
        "degenerate": False,
        "area": area,
        "local_scale": longest,
        "min_angle_deg": float(min(angles)),
        "aspect_longest_over_min_altitude": float(longest / min_altitude),
        "relative_double_area": float(area2 / (longest * longest)),
    }


def triangle_deformation_metric(rest_points, posed_points, *, dtype=np.float64) -> dict:
    """Measure the 2D-material to 3D deformation map using singular values."""
    r = np.asarray(rest_points, dtype=dtype)
    q = np.asarray(posed_points, dtype=dtype)
    if r.shape != (3, 3) or q.shape != (3, 3) or not np.isfinite(r).all() or not np.isfinite(q).all():
        raise QualificationError("MESH_DEFORMATION_TRIANGLE_INVALID")

    e01 = r[1] - r[0]
    e02 = r[2] - r[0]
    l01 = float(np.linalg.norm(e01))
    if l01 <= 0.0:
        raise QualificationError("MESH_DEFORMATION_REST_DEGENERATE")
    basis_x = e01 / l01
    x2 = float(np.dot(e02, basis_x))
    y2_sq = float(np.dot(e02, e02) - x2 * x2)
    local_scale = max(float(np.linalg.norm(e01)), float(np.linalg.norm(e02)), 1.0)
    tol = float(np.finfo(dtype).eps) * local_scale * local_scale * 32.0
    if y2_sq <= tol:
        raise QualificationError("MESH_DEFORMATION_REST_DEGENERATE")
    y2 = math.sqrt(y2_sq)

    dm = np.asarray(((l01, x2), (0.0, y2)), dtype=dtype)
    ds = np.stack((q[1] - q[0], q[2] - q[0]), axis=1).astype(dtype, copy=False)
    try:
        deformation_gradient = ds @ np.linalg.inv(dm)
        singular = np.linalg.svd(deformation_gradient, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise QualificationError("MESH_DEFORMATION_GRADIENT_SINGULAR") from exc

    sigma_max = float(singular[0])
    sigma_min = float(singular[1])
    if not math.isfinite(sigma_max) or not math.isfinite(sigma_min) or sigma_min <= 0.0:
        return {
            "collapsed": True,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "area_ratio": 0.0,
            "condition_number": float("inf"),
            "symmetric_stretch": float("inf"),
        }
    return {
        "collapsed": False,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "area_ratio": float(sigma_max * sigma_min),
        "condition_number": float(sigma_max / sigma_min),
        "symmetric_stretch": float(max(sigma_max, 1.0 / sigma_min)),
    }


def summarize_rest_quality(vertices, faces: Iterable[Iterable[int]]) -> dict:
    p = np.asarray(vertices, dtype=np.float64)
    rows = [triangle_rest_metric(p[np.asarray(tuple(face), dtype=np.int64)]) for face in faces]
    if not rows:
        raise QualificationError("MESH_CONDITIONING_REQUIRES_FACES")
    finite_aspect = [
        x["aspect_longest_over_min_altitude"]
        for x in rows
        if math.isfinite(x["aspect_longest_over_min_altitude"])
    ]
    return {
        "face_count": len(rows),
        "degenerate_faces": sum(bool(x["degenerate"]) for x in rows),
        "min_angle_deg": min(float(x["min_angle_deg"]) for x in rows),
        "max_aspect_longest_over_min_altitude": max(finite_aspect, default=float("inf")),
        "min_relative_double_area": min(float(x["relative_double_area"]) for x in rows),
    }
