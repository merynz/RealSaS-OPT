from __future__ import annotations

"""Continuous-motion certificate for source-backed directional BODY meshes.

Runtime-v4 linearly interpolates canonical XYZ between stored frames. Orthographic
projection is affine, therefore projected vertex XY is linear in interpolation
parameter and each triangle signed area is a quadratic polynomial. We evaluate the
exact minimum of that polynomial on every interval, so inversion/collapse cannot hide
between sampled frames.

Boundary non-collision is certified conservatively from swept AABBs and a Hausdorff
motion bound. If the proof cannot establish positive clearance, it fails closed.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_directional_body_v1 import DirectionalBodyAssetV1
from compiler.realsas_compiler_core.playback_full_surface_v3 import project_points_xyz_v3
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    DEFAULT_VIEWS,
    RuntimeV4Clip,
    RuntimeV4PlaybackContract,
    validate_playback_runtime_v4_contract,
    validate_runtime_v4_clip,
)
from compiler.realsas_compiler_core.types import QualificationError

DIRECTIONAL_BODY_CERTIFICATE_SCHEMA = "RealSaS.DirectionalBodyMotionCertificate.v1"


@dataclass(frozen=True)
class DirectionalBodyMotionCertificateV1:
    clip_id: str
    view_ids: tuple[str, ...]
    continuous_orientation_certified: bool
    boundary_motion_clearance_certified: bool
    topology_manifold_certified: bool
    global_embedding_certified: bool
    min_signed_area2_margin: float
    min_boundary_clearance_lower_bound: float
    source_alpha_recall_floor: float
    precision_inside_alpha_floor: float
    interval_count: int
    certificate_hash: str
    schema_version: str = DIRECTIONAL_BODY_CERTIFICATE_SCHEMA


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _signed_area2(points: np.ndarray, tri: np.ndarray) -> float:
    a, b, c = points[tri]
    return _cross2(b - a, c - a)


def _interval_signed_area2_min(
    p0: np.ndarray,
    p1: np.ndarray,
    tri: np.ndarray,
    orientation_sign: float,
) -> float:
    """Exact minimum signed double-area on t in [0,1] for linear vertex motion."""

    a0, b0, c0 = p0[tri]
    a1, b1, c1 = p1[tri]
    u0 = b0 - a0
    v0 = c0 - a0
    du = (b1 - a1) - u0
    dv = (c1 - a1) - v0
    q0 = _cross2(u0, v0)
    q1 = _cross2(du, v0) + _cross2(u0, dv)
    q2 = _cross2(du, dv)
    values = [
        orientation_sign * q0,
        orientation_sign * (q0 + q1 + q2),
    ]
    if abs(q2) > 1.0e-18:
        stationary = -q1 / (2.0 * q2)
        if 0.0 < stationary < 1.0:
            values.append(
                orientation_sign
                * (q0 + q1 * stationary + q2 * stationary * stationary)
            )
    return float(min(values))


def _boundary_edges(triangles: np.ndarray) -> tuple[tuple[int, int], ...]:
    counts: dict[tuple[int, int], int] = {}
    for tri in np.asarray(triangles, dtype=np.int64):
        for a, b in (
            (int(tri[0]), int(tri[1])),
            (int(tri[1]), int(tri[2])),
            (int(tri[2]), int(tri[0])),
        ):
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    if any(count > 2 for count in counts.values()):
        raise QualificationError("DIRECTIONAL_BODY_NONMANIFOLD_EDGE")
    boundary = tuple(sorted(edge for edge, count in counts.items() if count == 1))
    if not boundary:
        raise QualificationError("DIRECTIONAL_BODY_BOUNDARY_EMPTY")
    degree: dict[int, int] = {}
    for a, b in boundary:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    if any(value != 2 for value in degree.values()):
        raise QualificationError("DIRECTIONAL_BODY_OPEN_OR_BRANCHING_BOUNDARY")
    return boundary


def _point_segment_distance(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1.0e-24:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    return float(np.linalg.norm(p - (a + t * ab)))


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return _cross2(b - a, c - a)


def _on_segment(
    a: np.ndarray,
    b: np.ndarray,
    p: np.ndarray,
    eps: float,
) -> bool:
    if abs(_orient(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    eps: float = 1.0e-9,
) -> bool:
    o1, o2, o3, o4 = (
        _orient(a, b, c),
        _orient(a, b, d),
        _orient(c, d, a),
        _orient(c, d, b),
    )
    if (
        ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
        and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
    ):
        return True
    return (
        _on_segment(a, b, c, eps)
        or _on_segment(a, b, d, eps)
        or _on_segment(c, d, a, eps)
        or _on_segment(c, d, b, eps)
    )


def _segment_distance(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _boundary_interval_clearance(
    p0: np.ndarray,
    p1: np.ndarray,
    boundary: tuple[tuple[int, int], ...],
    *,
    clearance: float,
) -> float:
    """Conservative continuous-time separation certificate for moving boundaries.

    Each segment's Hausdorff displacement is no greater than the larger endpoint
    displacement under linear interpolation. Thus:
      dist(A(t), B(t)) >= dist(A(0), B(0)) - D_A - D_B.
    Swept AABB disjointness certifies the remaining pairs without distance work.
    """

    swept = []
    displacement = []
    for a, b in boundary:
        pts = np.asarray((p0[a], p0[b], p1[a], p1[b]), dtype=np.float64)
        swept.append(
            (
                float(pts[:, 0].min()),
                float(pts[:, 0].max()),
                float(pts[:, 1].min()),
                float(pts[:, 1].max()),
                a,
                b,
            )
        )
        displacement.append(
            max(
                float(np.linalg.norm(p1[a] - p0[a])),
                float(np.linalg.norm(p1[b] - p0[b])),
            )
        )

    order = sorted(range(len(boundary)), key=lambda i: swept[i][0])
    best = 1.0e30
    for oi, i in enumerate(order):
        minx_i, maxx_i, miny_i, maxy_i, a, b = swept[i]
        for j in order[oi + 1 :]:
            minx_j, maxx_j, miny_j, maxy_j, c, d = swept[j]
            if minx_j > maxx_i + clearance:
                break
            if a in (c, d) or b in (c, d):
                continue
            if maxy_i + clearance < miny_j or maxy_j + clearance < miny_i:
                continue
            d0 = _segment_distance(p0[a], p0[b], p0[c], p0[d])
            lower = d0 - displacement[i] - displacement[j]
            best = min(best, lower)
            if lower <= clearance:
                raise QualificationError(
                    f"DIRECTIONAL_BODY_BOUNDARY_INTERVAL_NOT_CERTIFIED:{lower}"
                )
    return float(best)


def certify_directional_body_motion_v1(
    contract: RuntimeV4PlaybackContract,
    clip: RuntimeV4Clip,
    assets: Sequence[DirectionalBodyAssetV1],
    *,
    min_source_alpha_recall: float = 0.98,
    min_precision_inside_alpha: float = 0.995,
    min_signed_area2: float = 1.0e-4,
    min_boundary_clearance_px: float = 1.0e-4,
    required_view_ids: Sequence[str] = DEFAULT_VIEWS,
) -> DirectionalBodyMotionCertificateV1:
    """Certify source coverage + continuous topological safety before rendering."""

    view_ids = tuple(map(str, required_view_ids))
    rows = tuple(assets)
    validate_playback_runtime_v4_contract(contract, required_view_ids=view_ids)
    validate_runtime_v4_clip(contract, clip, required_view_ids=view_ids)
    by_view = {row.view_id: row for row in rows}
    if len(by_view) != len(rows) or set(by_view) != set(view_ids):
        raise QualificationError("DIRECTIONAL_BODY_CERT_ASSET_VIEW_SET_MISMATCH")

    recall_floor = min(row.source_alpha_recall for row in by_view.values())
    precision_floor = min(row.precision_inside_alpha for row in by_view.values())
    if recall_floor < float(min_source_alpha_recall):
        raise QualificationError(
            f"DIRECTIONAL_BODY_SOURCE_RECALL_GATE_FAIL:{recall_floor}"
        )
    if precision_floor < float(min_precision_inside_alpha):
        raise QualificationError(
            f"DIRECTIONAL_BODY_SOURCE_PRECISION_GATE_FAIL:{precision_floor}"
        )

    asset_by_id = {asset.asset_id: asset for asset in contract.assets}
    min_area_margin = 1.0e30
    min_boundary_margin = 1.0e30
    interval_count = 0

    for view_index, view_id in enumerate(view_ids):
        row = by_view[view_id]
        asset = asset_by_id.get(row.runtime_asset.asset_id)
        if asset is None:
            raise QualificationError("DIRECTIONAL_BODY_CERT_RUNTIME_ASSET_MISSING")
        triangles = np.asarray(asset.triangles, dtype=np.int64)
        boundary = _boundary_edges(triangles)
        camera = contract.views[view_index].camera

        projected = [
            project_points_xyz_v3(
                np.asarray(
                    frame.canonical_posed_xyz_by_asset[asset.asset_id],
                    dtype=np.float64,
                ),
                camera,
            )[:, :2]
            for frame in clip.frames
        ]
        rest_projected = project_points_xyz_v3(
            np.asarray(asset.rest_xyz, dtype=np.float64),
            camera,
        )[:, :2]
        rest_area2 = np.asarray(
            [_signed_area2(rest_projected, tri) for tri in triangles],
            dtype=np.float64,
        )
        if np.any(np.abs(rest_area2) <= float(min_signed_area2)):
            raise QualificationError("DIRECTIONAL_BODY_REST_TRIANGLE_DEGENERATE")
        rest_sign = np.where(rest_area2 > 0.0, 1.0, -1.0)

        for frame_index in range(len(projected) - 1):
            p0, p1 = projected[frame_index], projected[frame_index + 1]
            interval_count += 1
            for tri_index, tri in enumerate(triangles):
                margin = _interval_signed_area2_min(
                    p0,
                    p1,
                    tri,
                    float(rest_sign[tri_index]),
                )
                min_area_margin = min(min_area_margin, margin)
                if margin <= float(min_signed_area2):
                    raise QualificationError(
                        "DIRECTIONAL_BODY_TRIANGLE_INTERVAL_INVERSION_OR_COLLAPSE:"
                        f"{view_id}:{frame_index}:{tri_index}:{margin}"
                    )
            boundary_margin = _boundary_interval_clearance(
                p0,
                p1,
                boundary,
                clearance=float(min_boundary_clearance_px),
            )
            min_boundary_margin = min(min_boundary_margin, boundary_margin)

    if interval_count <= 0:
        raise QualificationError("DIRECTIONAL_BODY_CERT_REQUIRES_FRAME_INTERVALS")

    payload = {
        "schema": DIRECTIONAL_BODY_CERTIFICATE_SCHEMA,
        "clip_id": clip.clip_id,
        "view_ids": list(view_ids),
        "continuous_orientation_certified": True,
        "boundary_motion_clearance_certified": True,
        "topology_manifold_certified": True,
        "global_embedding_certified": True,
        "min_signed_area2_margin": float(min_area_margin),
        "min_boundary_clearance_lower_bound": float(min_boundary_margin),
        "source_alpha_recall_floor": float(recall_floor),
        "precision_inside_alpha_floor": float(precision_floor),
        "interval_count": int(interval_count),
        "runtime_interpolation": "LINEAR_CANONICAL_XYZ",
        "projection": "ORTHOGRAPHIC_AFFINE_XY",
        "completion_used": False,
    }
    return DirectionalBodyMotionCertificateV1(
        clip_id=clip.clip_id,
        view_ids=view_ids,
        continuous_orientation_certified=True,
        boundary_motion_clearance_certified=True,
        topology_manifold_certified=True,
        global_embedding_certified=True,
        min_signed_area2_margin=float(min_area_margin),
        min_boundary_clearance_lower_bound=float(min_boundary_margin),
        source_alpha_recall_floor=float(recall_floor),
        precision_inside_alpha_floor=float(precision_floor),
        interval_count=int(interval_count),
        certificate_hash=content_sha256(payload),
    )


__all__ = [
    "DIRECTIONAL_BODY_CERTIFICATE_SCHEMA",
    "DirectionalBodyMotionCertificateV1",
    "certify_directional_body_motion_v1",
]
