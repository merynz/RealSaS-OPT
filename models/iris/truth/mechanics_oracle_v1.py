"""Consumer-neutral Mechanics oracle evidence from finite causal interventions.

The kernel answers a deliberately narrow question for two canonical surface
queries: does their *relative rigid surface-frame transport* stay invariant under
a declared finite intervention set?

It does not infer attachment, bone identity, contact, support, ownership or a
semantic part label.  It also does not assume an action probability law.  The
current calibration explicitly leaves that law unsealed, so this court uses
worst-case residuals over caller-declared discovery and held-out intervention
sets rather than an implicit uniform average.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from models.iris.truth.measurements_v1 import triangle_metric_strain


RelativeTransportStatusV1 = Literal[
    "RIGID_RELATIVE_TRANSPORT_SUPPORTED",
    "NONRIGID_RELATIVE_TRANSPORT_SUPPORTED",
    "ABSTAIN",
]


@dataclass(frozen=True)
class SurfacePairQueryV1:
    face_a: int
    face_b: int

    def __post_init__(self) -> None:
        if type(self.face_a) is not int or type(self.face_b) is not int:
            raise ValueError("MECHANICS_ORACLE_FACE_INDEX_INTEGER_REQUIRED")
        if self.face_a < 0 or self.face_b < 0 or self.face_a == self.face_b:
            raise ValueError("MECHANICS_ORACLE_DISTINCT_NONNEGATIVE_FACES_REQUIRED")


@dataclass(frozen=True)
class RelativeTransportPolicyV1:
    """Pre-registered conservative bands for the V1 teacher-oracle court.

    The gap between rigid_accept_angle_rad and nonrigid_reject_angle_rad is
    intentional: uncertainty must abstain rather than be forced into a binary
    mechanism label. Translation and strain are required for rigid admission,
    but a large relative angular drift is sufficient evidence that rigid
    relative transport is contradicted.
    """

    rigid_accept_angle_rad: float = 1.0e-3
    rigid_accept_translation_fraction: float = 1.0e-4
    rigid_accept_strain_fro: float = 1.0e-3
    nonrigid_reject_angle_rad: float = 5.0e-3
    minimum_discovery_interventions: int = 2
    minimum_validation_interventions: int = 1

    def __post_init__(self) -> None:
        values = (
            self.rigid_accept_angle_rad,
            self.rigid_accept_translation_fraction,
            self.rigid_accept_strain_fro,
            self.nonrigid_reject_angle_rad,
        )
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("MECHANICS_ORACLE_POLICY_POSITIVE_FINITE_REQUIRED")
        if self.rigid_accept_angle_rad >= self.nonrigid_reject_angle_rad:
            raise ValueError("MECHANICS_ORACLE_POLICY_ABSTENTION_BAND_REQUIRED")
        if type(self.minimum_discovery_interventions) is not int or self.minimum_discovery_interventions < 1:
            raise ValueError("MECHANICS_ORACLE_DISCOVERY_COUNT_INVALID")
        if type(self.minimum_validation_interventions) is not int or self.minimum_validation_interventions < 1:
            raise ValueError("MECHANICS_ORACLE_VALIDATION_COUNT_INVALID")


@dataclass(frozen=True)
class RelativeTransportSplitEvidenceV1:
    intervention_count: int
    valid: bool
    max_relative_angle_rad: float
    max_relative_translation_fraction: float
    max_triangle_strain_fro: float
    status: RelativeTransportStatusV1


@dataclass(frozen=True)
class RelativeTransportEvidenceV1:
    query: SurfacePairQueryV1
    subject_scale: float
    discovery: RelativeTransportSplitEvidenceV1
    validation: RelativeTransportSplitEvidenceV1
    heldout_confirmed: bool
    final_status: RelativeTransportStatusV1


def _finite(value, *, ndim: int) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != ndim or not np.isfinite(out).all():
        raise ValueError("MECHANICS_ORACLE_FINITE_ARRAY_REQUIRED")
    return out


def _validate_geometry(rest_vertices, faces, query: SurfacePairQueryV1):
    rest = _finite(rest_vertices, ndim=2)
    tri = np.asarray(faces)
    if rest.shape[1:] != (3,) or tri.ndim != 2 or tri.shape[1:] != (3,) or not np.issubdtype(tri.dtype, np.integer):
        raise ValueError("MECHANICS_ORACLE_TRIANGULATED_XYZ_REQUIRED")
    if tri.size == 0 or np.min(tri) < 0 or np.max(tri) >= len(rest):
        raise ValueError("MECHANICS_ORACLE_FACE_VERTEX_INDEX_INVALID")
    if query.face_a >= len(tri) or query.face_b >= len(tri):
        raise ValueError("MECHANICS_ORACLE_FACE_QUERY_OUT_OF_RANGE")
    return rest, tri.astype(np.int64, copy=False)


def _surface_rigid_frames(positions: np.ndarray, face: np.ndarray, *, degeneracy_tolerance: float):
    """Return [sample,4,4] centroid+tangent frames and a validity mask.

    The orthonormal frame is used only for relative rigid transport. Metric
    deformation is measured independently by ``triangle_metric_strain`` so this
    construction never fabricates a volumetric deformation gradient from a
    surface triangle.
    """
    p = positions[:, face, :]
    edge = p[:, 1] - p[:, 0]
    normal = np.cross(edge, p[:, 2] - p[:, 0])
    edge_norm = np.linalg.norm(edge, axis=1)
    normal_norm = np.linalg.norm(normal, axis=1)
    valid = (edge_norm > degeneracy_tolerance) & (normal_norm > degeneracy_tolerance)

    frames = np.full((len(p), 4, 4), np.nan, dtype=np.float64)
    frames[:, 3, 3] = 1.0
    if not np.any(valid):
        return frames, valid

    x = np.full_like(edge, np.nan)
    z = np.full_like(normal, np.nan)
    x[valid] = edge[valid] / edge_norm[valid, None]
    z[valid] = normal[valid] / normal_norm[valid, None]
    y = np.cross(z, x)
    y_norm = np.linalg.norm(y, axis=1)
    valid &= np.isfinite(y_norm) & (y_norm > degeneracy_tolerance)
    y[valid] /= y_norm[valid, None]
    # Re-orthogonalize x after y normalization; orientation is deterministic
    # from the authored face winding and first edge.
    x[valid] = np.cross(y[valid], z[valid])

    frames[valid, :3, 0] = x[valid]
    frames[valid, :3, 1] = y[valid]
    frames[valid, :3, 2] = z[valid]
    frames[valid, :3, 3] = p[valid].mean(axis=1)
    return frames, valid


def _relative_rigid_transform(frame_a: np.ndarray, frame_b: np.ndarray):
    ra = frame_a[:3, :3]
    rb = frame_b[:3, :3]
    pa = frame_a[:3, 3]
    pb = frame_b[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = ra.T @ rb
    out[:3, 3] = ra.T @ (pb - pa)
    return out


def _rotation_distance_rad(reference: np.ndarray, current: np.ndarray) -> float:
    delta = reference.T @ current
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def _status(
    *,
    valid: bool,
    angle: float,
    translation: float,
    strain: float,
    policy: RelativeTransportPolicyV1,
) -> RelativeTransportStatusV1:
    if not valid:
        return "ABSTAIN"
    if (
        angle <= policy.rigid_accept_angle_rad
        and translation <= policy.rigid_accept_translation_fraction
        and strain <= policy.rigid_accept_strain_fro
    ):
        return "RIGID_RELATIVE_TRANSPORT_SUPPORTED"
    if angle >= policy.nonrigid_reject_angle_rad:
        return "NONRIGID_RELATIVE_TRANSPORT_SUPPORTED"
    return "ABSTAIN"


def _measure_split(
    rest: np.ndarray,
    faces: np.ndarray,
    positions: np.ndarray,
    query: SurfacePairQueryV1,
    *,
    subject_scale: float,
    degeneracy_tolerance: float,
    minimum_interventions: int,
    policy: RelativeTransportPolicyV1,
) -> RelativeTransportSplitEvidenceV1:
    pos = _finite(positions, ndim=3)
    if pos.shape[1:] != rest.shape or len(pos) < minimum_interventions:
        raise ValueError("MECHANICS_ORACLE_INTERVENTION_SHAPE_OR_COUNT_INVALID")

    # Baseline is prepended only to construct the reference frame. It is not an
    # intervention and is never averaged into a statistic.
    samples = np.concatenate((rest[None, ...], pos), axis=0)
    frame_a, valid_a = _surface_rigid_frames(samples, faces[query.face_a], degeneracy_tolerance=degeneracy_tolerance)
    frame_b, valid_b = _surface_rigid_frames(samples, faces[query.face_b], degeneracy_tolerance=degeneracy_tolerance)
    frame_valid = valid_a & valid_b
    if not frame_valid[0] or not np.all(frame_valid[1:]):
        return RelativeTransportSplitEvidenceV1(
            intervention_count=len(pos),
            valid=False,
            max_relative_angle_rad=float("inf"),
            max_relative_translation_fraction=float("inf"),
            max_triangle_strain_fro=float("inf"),
            status="ABSTAIN",
        )

    reference = _relative_rigid_transform(frame_a[0], frame_b[0])
    max_angle = 0.0
    max_translation = 0.0
    for ia, ib in zip(frame_a[1:], frame_b[1:]):
        current = _relative_rigid_transform(ia, ib)
        max_angle = max(max_angle, _rotation_distance_rad(reference[:3, :3], current[:3, :3]))
        max_translation = max(
            max_translation,
            float(np.linalg.norm(current[:3, 3] - reference[:3, 3]) / subject_scale),
        )

    max_strain = 0.0
    for face_index in (query.face_a, query.face_b):
        rest_tri = np.broadcast_to(rest[faces[face_index]], (len(pos), 3, 3))
        posed_tri = pos[:, faces[face_index], :]
        strain, _area_ratio, strain_valid = triangle_metric_strain(
            rest_tri,
            posed_tri,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        if not np.all(strain_valid) or not np.isfinite(strain).all():
            return RelativeTransportSplitEvidenceV1(
                intervention_count=len(pos),
                valid=False,
                max_relative_angle_rad=float("inf"),
                max_relative_translation_fraction=float("inf"),
                max_triangle_strain_fro=float("inf"),
                status="ABSTAIN",
            )
        max_strain = max(max_strain, float(np.max(np.linalg.norm(strain, axis=(-2, -1)))))

    status = _status(
        valid=True,
        angle=max_angle,
        translation=max_translation,
        strain=max_strain,
        policy=policy,
    )
    return RelativeTransportSplitEvidenceV1(
        intervention_count=len(pos),
        valid=True,
        max_relative_angle_rad=max_angle,
        max_relative_translation_fraction=max_translation,
        max_triangle_strain_fro=max_strain,
        status=status,
    )


def measure_relative_transport_v1(
    rest_vertices,
    faces,
    discovery_positions,
    validation_positions,
    query: SurfacePairQueryV1,
    *,
    subject_scale: float,
    degeneracy_tolerance: float,
    policy: RelativeTransportPolicyV1 | None = None,
) -> RelativeTransportEvidenceV1:
    """Measure finite-intervention relative transport with a held-out court.

    No action weights, control names, bone ids, source weights or owner labels are
    accepted by this API. Privileged metadata may be used by an outer audit to
    choose/evaluate queries, but cannot leak into the measured evidence itself.
    """
    rest, tri = _validate_geometry(rest_vertices, faces, query)
    if not np.isfinite(subject_scale) or subject_scale <= 0:
        raise ValueError("MECHANICS_ORACLE_SUBJECT_SCALE_POSITIVE_REQUIRED")
    if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance <= 0:
        raise ValueError("MECHANICS_ORACLE_DEGENERACY_TOLERANCE_POSITIVE_REQUIRED")
    policy = policy or RelativeTransportPolicyV1()

    discovery = _measure_split(
        rest,
        tri,
        discovery_positions,
        query,
        subject_scale=subject_scale,
        degeneracy_tolerance=degeneracy_tolerance,
        minimum_interventions=policy.minimum_discovery_interventions,
        policy=policy,
    )
    validation = _measure_split(
        rest,
        tri,
        validation_positions,
        query,
        subject_scale=subject_scale,
        degeneracy_tolerance=degeneracy_tolerance,
        minimum_interventions=policy.minimum_validation_interventions,
        policy=policy,
    )
    heldout_confirmed = (
        discovery.status != "ABSTAIN"
        and validation.status == discovery.status
    )
    final_status: RelativeTransportStatusV1 = discovery.status if heldout_confirmed else "ABSTAIN"
    return RelativeTransportEvidenceV1(
        query=query,
        subject_scale=float(subject_scale),
        discovery=discovery,
        validation=validation,
        heldout_confirmed=heldout_confirmed,
        final_status=final_status,
    )
