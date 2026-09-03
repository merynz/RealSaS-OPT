from __future__ import annotations

"""Compiler-owned directional joint/view binding from admitted surface correspondences.

The current mechanical surface already carries qualified object-frame points P and
per-view raster observations.  This module fits one affine orthographic map per
view from those admitted correspondences and binds canonical joint pivots through
that map.  It never treats mechanical Vec3.xy as raster coordinates and never
consults teacher/source-mesh geometry.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

import numpy as np

from .hashing import content_sha256
from .types import QualificationError, Vec2, Vec3

Json = dict[str, Any]


@dataclass(frozen=True)
class DirectionalBindingPolicyV1:
    min_correspondences: int = 8
    required_affine_rank: int = 4
    max_p95_residual01: float = 0.015
    max_residual01: float = 0.05
    min_raster_span: float = 8.0
    schema_version: str = "RealSaS.DirectionalBindingPolicy.v1"

    def validate(self) -> None:
        if self.min_correspondences < 4:
            raise ValueError("directional binding requires at least four correspondences")
        if self.required_affine_rank not in (3, 4):
            raise ValueError("unsupported affine-rank policy")
        if not (0.0 < self.max_p95_residual01 <= self.max_residual01 < 1.0):
            raise ValueError("invalid normalized residual policy")
        if self.min_raster_span <= 0.0:
            raise ValueError("invalid raster-span policy")

    @property
    def policy_hash(self) -> str:
        self.validate()
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class DirectionalViewProjectionBindingIR:
    source_product_state_hash: str
    view_index: int
    camera_binding_hash: str
    surface_lineage_hash: str
    affine_rows: tuple[tuple[float, float, float, float], tuple[float, float, float, float]]
    source_surface_ids: tuple[str, ...]
    correspondence_count: int
    affine_rank: int
    rms_residual_px: float
    p95_residual_px: float
    max_residual_px: float
    raster_span_px: float
    p95_residual01: float
    max_residual01: float
    policy_hash: str
    qualification_report: Json
    projection_binding_hash: str
    schema_version: str = "RealSaS.DirectionalViewProjectionBindingIR.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class DirectionalJointPivotIR:
    view_index: int
    canonical_joint_id: str
    raster_xy: Vec2
    projection_binding_hash: str
    schema_version: str = "RealSaS.DirectionalJointPivotIR.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class DirectionalJointViewBindingSetIR:
    source_product_state_hash: str
    surface_lineage_hash: str
    skeleton_lineage_hash: str
    directional_visual_state_hash: str
    projections: tuple[DirectionalViewProjectionBindingIR, ...]
    joint_pivots: tuple[DirectionalJointPivotIR, ...]
    binding_set_hash: str
    policy_hash: str
    schema_version: str = "RealSaS.DirectionalJointViewBindingSetIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


def _projection_hash(value: DirectionalViewProjectionBindingIR) -> str:
    payload = value.to_dict()
    payload.pop("projection_binding_hash", None)
    return content_sha256(payload)


def _binding_set_hash(value: DirectionalJointViewBindingSetIR) -> str:
    payload = value.to_dict()
    payload.pop("binding_set_hash", None)
    return content_sha256(payload)


def _raster_xy(node, view_index: int) -> Vec2 | None:
    rows = [tuple(map(float, xy)) for v, xy in node.raster_bindings if int(v) == int(view_index)]
    if len(rows) > 1:
        raise QualificationError(f"DIRECTIONAL_BINDING_DUPLICATE_RASTER_BINDING:{node.surface_id}:{view_index}")
    return None if not rows else rows[0]


def _fit_view_projection(product, view_index: int, camera_binding_hash: str, policy: DirectionalBindingPolicyV1) -> DirectionalViewProjectionBindingIR:
    surface = product.mechanical_state.surface
    rows = []
    for node in sorted(surface.surface_nodes, key=lambda n: n.surface_id):
        xy = _raster_xy(node, view_index)
        if xy is None:
            continue
        p = tuple(map(float, node.P))
        if len(p) != 3 or not all(math.isfinite(v) for v in (*p, *xy)):
            raise QualificationError("DIRECTIONAL_BINDING_NONFINITE_CORRESPONDENCE")
        rows.append((node.surface_id, p, xy))
    if len(rows) < policy.min_correspondences:
        raise QualificationError(f"DIRECTIONAL_BINDING_INSUFFICIENT_CORRESPONDENCE:V{view_index}:{len(rows)}")

    X = np.asarray([[*p, 1.0] for _, p, _ in rows], dtype=np.float64)
    Y = np.asarray([xy for _, _, xy in rows], dtype=np.float64)
    coeff, _, rank, singular = np.linalg.lstsq(X, Y, rcond=None)
    rank = int(rank)
    if rank < int(policy.required_affine_rank):
        raise QualificationError(f"DIRECTIONAL_BINDING_AFFINE_RANK_DEFICIENT:V{view_index}:{rank}")
    prediction = X @ coeff
    residual = np.linalg.norm(prediction - Y, axis=1)
    span = float(np.linalg.norm(Y.max(axis=0) - Y.min(axis=0)))
    if not math.isfinite(span) or span < float(policy.min_raster_span):
        raise QualificationError(f"DIRECTIONAL_BINDING_RASTER_SPAN_DEGENERATE:V{view_index}:{span}")
    rms = float(np.sqrt(np.mean(residual * residual)))
    p95 = float(np.percentile(residual, 95))
    max_residual = float(residual.max(initial=0.0))
    p95_01 = p95 / span
    max_01 = max_residual / span
    if not all(math.isfinite(v) for v in (rms, p95, max_residual, p95_01, max_01)):
        raise QualificationError("DIRECTIONAL_BINDING_NONFINITE_FIT")
    if p95_01 > float(policy.max_p95_residual01) or max_01 > float(policy.max_residual01):
        raise QualificationError(
            f"DIRECTIONAL_BINDING_RESIDUAL_EXCEEDED:V{view_index}:p95={p95_01}:max={max_01}"
        )

    # np.linalg.lstsq returns [4,2]; serialize as two raster-output rows [2,4].
    affine = tuple(tuple(map(float, coeff[:, axis])) for axis in range(2))
    report = {
        "status": "PASS_QUALIFIED_AFFINE_ORTHOGRAPHIC_BINDING",
        "camera_model": "KNOWN_ORTHOGRAPHIC_8VIEW",
        "source_authority": "RIGGING_SURFACE_P_TO_ADMITTED_RASTER_BINDINGS",
        "source_mesh_used": False,
        "teacher_truth_used": False,
        "correspondence_count": len(rows),
        "affine_rank": rank,
        "condition_number": float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else float("inf"),
        "p95_residual01": p95_01,
        "max_residual01": max_01,
    }
    value = DirectionalViewProjectionBindingIR(
        str(product.product_state_hash), int(view_index), str(camera_binding_hash), str(surface.geometry_lineage_hash),
        affine, tuple(sid for sid, _, _ in rows), len(rows), rank, rms, p95, max_residual, span,
        p95_01, max_01, policy.policy_hash, report, "",
    )
    return replace(value, projection_binding_hash=_projection_hash(value))


def project_mechanical_point(binding: DirectionalViewProjectionBindingIR, point: Vec3) -> Vec2:
    p = np.asarray((*map(float, point), 1.0), dtype=np.float64)
    A = np.asarray(binding.affine_rows, dtype=np.float64)
    if A.shape != (2, 4) or p.shape != (4,) or not np.isfinite(A).all() or not np.isfinite(p).all():
        raise QualificationError("DIRECTIONAL_BINDING_INVALID_AFFINE_PAYLOAD")
    xy = A @ p
    if not np.isfinite(xy).all():
        raise QualificationError("DIRECTIONAL_BINDING_NONFINITE_PROJECTED_POINT")
    return (float(xy[0]), float(xy[1]))


def qualify_directional_joint_view_binding(product, *, policy: DirectionalBindingPolicyV1 = DirectionalBindingPolicyV1()) -> DirectionalJointViewBindingSetIR:
    policy.validate()
    surface = product.mechanical_state.surface
    skeleton = product.mechanical_state.skeleton
    directions = tuple(sorted(product.directional_renderables.directions, key=lambda d: int(d.view_index)))
    if tuple(int(d.view_index) for d in directions) != tuple(range(8)):
        raise QualificationError("DIRECTIONAL_BINDING_REQUIRES_EXACT_VIEWS_0_TO_7")

    projections = tuple(
        _fit_view_projection(product, int(direction.view_index), str(direction.camera_binding_hash), policy)
        for direction in directions
    )
    pivots = []
    for projection in projections:
        for joint in sorted(skeleton.joints, key=lambda j: j.canonical_joint_id):
            pivots.append(DirectionalJointPivotIR(
                int(projection.view_index), str(joint.canonical_joint_id),
                project_mechanical_point(projection, joint.position), projection.projection_binding_hash,
            ))
    value = DirectionalJointViewBindingSetIR(
        str(product.product_state_hash), str(surface.geometry_lineage_hash), str(skeleton.skeleton_lineage_hash),
        str(product.directional_renderables.directional_visual_state_hash), projections, tuple(pivots), "", policy.policy_hash,
        metadata={
            "authority": "COMPILER_QUALIFIED_DERIVED_BINDING",
            "mechanical_xy_is_raster_xy": False,
            "raw_camera_reconstruction_used": False,
            "teacher_truth_used": False,
            "source_mesh_used": False,
        },
    )
    value = replace(value, binding_set_hash=_binding_set_hash(value))
    assert_directional_binding_for_product(product, value)
    return value


def assert_directional_binding_for_product(product, binding: DirectionalJointViewBindingSetIR) -> None:
    if binding.source_product_state_hash != str(product.product_state_hash):
        raise QualificationError("STALE_DIRECTIONAL_BINDING_PRODUCT")
    if binding.surface_lineage_hash != str(product.mechanical_state.surface.geometry_lineage_hash):
        raise QualificationError("STALE_DIRECTIONAL_BINDING_SURFACE")
    if binding.skeleton_lineage_hash != str(product.mechanical_state.skeleton.skeleton_lineage_hash):
        raise QualificationError("STALE_DIRECTIONAL_BINDING_SKELETON")
    if binding.directional_visual_state_hash != str(product.directional_renderables.directional_visual_state_hash):
        raise QualificationError("STALE_DIRECTIONAL_BINDING_VISUAL_STATE")
    if binding.binding_set_hash != _binding_set_hash(binding):
        raise QualificationError("DIRECTIONAL_BINDING_SET_HASH_MISMATCH")
    projections = {int(row.view_index): row for row in binding.projections}
    if tuple(sorted(projections)) != tuple(range(8)) or len(projections) != 8:
        raise QualificationError("DIRECTIONAL_BINDING_PROJECTION_SET_INCOMPLETE")
    directions = {int(d.view_index): d for d in product.directional_renderables.directions}
    for view_index, row in projections.items():
        if row.projection_binding_hash != _projection_hash(row):
            raise QualificationError("DIRECTIONAL_PROJECTION_HASH_MISMATCH")
        if row.source_product_state_hash != product.product_state_hash:
            raise QualificationError("STALE_DIRECTIONAL_PROJECTION_PRODUCT")
        if row.camera_binding_hash != directions[view_index].camera_binding_hash:
            raise QualificationError("DIRECTIONAL_PROJECTION_CAMERA_BINDING_MISMATCH")
    joint_ids = {j.canonical_joint_id for j in product.mechanical_state.skeleton.joints}
    pivots = {(int(p.view_index), p.canonical_joint_id): p for p in binding.joint_pivots}
    expected = {(view, jid) for view in range(8) for jid in joint_ids}
    if set(pivots) != expected:
        raise QualificationError("DIRECTIONAL_JOINT_PIVOT_SET_INCOMPLETE")
    joint_by_id = {j.canonical_joint_id: j for j in product.mechanical_state.skeleton.joints}
    for (view, jid), pivot in pivots.items():
        projection = projections[view]
        if pivot.projection_binding_hash != projection.projection_binding_hash:
            raise QualificationError("DIRECTIONAL_JOINT_PIVOT_PROJECTION_MISMATCH")
        expected_xy = project_mechanical_point(projection, joint_by_id[jid].position)
        if max(abs(float(a) - float(b)) for a, b in zip(pivot.raster_xy, expected_xy)) > 1e-8:
            raise QualificationError("DIRECTIONAL_JOINT_PIVOT_VALUE_MISMATCH")


def projection_for_view(binding: DirectionalJointViewBindingSetIR, view_index: int) -> DirectionalViewProjectionBindingIR:
    rows = [row for row in binding.projections if int(row.view_index) == int(view_index)]
    if len(rows) != 1:
        raise QualificationError("DIRECTIONAL_PROJECTION_LOOKUP_FAILED")
    return rows[0]


def joint_pivot(binding: DirectionalJointViewBindingSetIR, view_index: int, canonical_joint_id: str) -> Vec2:
    rows = [row for row in binding.joint_pivots if int(row.view_index) == int(view_index) and row.canonical_joint_id == canonical_joint_id]
    if len(rows) != 1:
        raise QualificationError("DIRECTIONAL_JOINT_PIVOT_LOOKUP_FAILED")
    return rows[0].raster_xy
