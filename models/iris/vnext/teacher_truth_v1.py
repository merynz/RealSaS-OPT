from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

Json = dict[str, Any]


@dataclass(frozen=True)
class MechanicalDemandTruthPolicyV1:
    rigid_edge_strain_tolerance: float = 0.02
    rigid_area_change_tolerance: float = 0.03
    rigid_bend_radians_tolerance: float = math.radians(3.0)
    epsilon: float = 1e-10

    def validate(self):
        if self.epsilon <= 0 or min(
            self.rigid_edge_strain_tolerance,
            self.rigid_area_change_tolerance,
            self.rigid_bend_radians_tolerance,
        ) <= 0:
            raise ValueError("MECHANICAL_TRUTH_POLICY_INVALID")


@dataclass(frozen=True)
class MechanicalDemandTruthV1:
    stretch_major: tuple[float, ...]
    stretch_minor: tuple[float, ...]
    tangent_direction_2theta: tuple[tuple[float, float], ...]
    bend_demand: tuple[float, ...]
    rigidity_probability: tuple[float, ...]
    area_change_p95: tuple[float, ...]
    condition_number_p95: tuple[float, ...]
    edge_strain_p95: tuple[float, ...]
    contact_persistence: tuple[float, ...]
    contact_persistence_valid: bool
    face_count: int
    frame_count: int
    schema_version: str = "RealSaS.MechanicalDemandTruth.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _validate(rest_vertices, faces, posed_vertices):
    r = np.asarray(rest_vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    p = np.asarray(posed_vertices, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3 or not np.isfinite(r).all():
        raise ValueError("MECHANICAL_TRUTH_REST_INVALID")
    if (
        f.ndim != 2
        or f.shape[1] != 3
        or len(f) == 0
        or np.any(f < 0)
        or np.any(f >= len(r))
    ):
        raise ValueError("MECHANICAL_TRUTH_FACES_INVALID")
    if (
        p.ndim != 3
        or p.shape[1:] != r.shape
        or p.shape[0] < 1
        or not np.isfinite(p).all()
    ):
        raise ValueError("MECHANICAL_TRUTH_POSES_INVALID")
    return r, f, p


def _basis(vertices, faces, eps):
    a, b, c = (
        vertices[faces[:, 0]],
        vertices[faces[:, 1]],
        vertices[faces[:, 2]],
    )
    e1, e2 = b - a, c - a
    l1 = np.linalg.norm(e1, axis=1)
    n = np.cross(e1, e2)
    nn = np.linalg.norm(n, axis=1)
    if np.any(l1 <= eps) or np.any(nn <= eps):
        raise ValueError("MECHANICAL_TRUTH_DEGENERATE_FACE")
    u = e1 / l1[:, None]
    normal = n / nn[:, None]
    v = np.cross(normal, u)
    x2 = np.sum(e2 * u, axis=1)
    y2 = np.sum(e2 * v, axis=1)
    if np.any(np.abs(y2) <= eps):
        raise ValueError("MECHANICAL_TRUTH_SINGULAR_BASIS")
    dm = np.zeros((len(faces), 2, 2), dtype=np.float64)
    dm[:, 0, 0] = l1
    dm[:, 0, 1] = x2
    dm[:, 1, 1] = y2
    return np.linalg.inv(dm), 0.5 * nn, normal


def _frame(posed, faces, inv_dm, rest_area, rest_normals, eps):
    """Evaluate one authored pose without requiring posed triangles to be full rank.

    The rest face owns the 2-D material coordinates. A posed face therefore has
    a 3x2 deformation gradient ``Ds @ Dm^-1``. Its singular values and right
    singular vectors remain well-defined for line/point collapses, unlike the old
    pose-local 2-D basis construction which rejected rank-deficient animation.
    """

    a, b, c = (
        posed[faces[:, 0]],
        posed[faces[:, 1]],
        posed[faces[:, 2]],
    )
    e1, e2 = b - a, c - a
    n = np.cross(e1, e2)
    nn = np.linalg.norm(n, axis=1)

    ds_world = np.stack((e1, e2), axis=2)
    deformation = np.einsum("fij,fjk->fik", ds_world, inv_dm)
    _, singular, vh = np.linalg.svd(deformation, full_matrices=False)
    major, minor = singular[:, 0], singular[:, 1]

    principal = vh[:, 0, :]
    theta = np.arctan2(principal[:, 1], principal[:, 0])
    direction_2theta = np.stack((np.cos(2 * theta), np.sin(2 * theta)), axis=1)

    degenerate = nn <= eps
    normals = np.array(rest_normals, dtype=np.float64, copy=True)
    regular = ~degenerate
    normals[regular] = n[regular] / nn[regular, None]

    condition = np.empty_like(major)
    rank_zero = (major <= eps) & (minor <= eps)
    condition[rank_zero] = 1.0
    condition[~rank_zero] = major[~rank_zero] / np.maximum(minor[~rank_zero], eps)
    direction_valid = major > eps

    return (
        major,
        minor,
        direction_2theta,
        (0.5 * nn) / rest_area,
        condition,
        normals,
        degenerate,
        direction_valid,
    )


def _adjacency(faces):
    edges = {}
    for fi, (a, b, c) in enumerate(faces.tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault(tuple(sorted((int(u), int(v)))), []).append(fi)
    return tuple(tuple(value) for value in edges.values() if len(value) == 2)


def _bend_precompute(rest_normals, faces):
    """Build immutable adjacent-face pairs and rest angles once per subject."""

    pairs = np.asarray(_adjacency(faces), dtype=np.int64)
    if pairs.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    pairs = pairs.reshape(-1, 2)
    edge_a, edge_b = pairs[:, 0], pairs[:, 1]
    rest_dot = np.clip(
        np.einsum("ij,ij->i", rest_normals[edge_a], rest_normals[edge_b]),
        -1.0,
        1.0,
    )
    return edge_a, edge_b, np.arccos(rest_dot)


def _bend_precomputed(
    posed_normals,
    degenerate_mask,
    face_count,
    edge_a,
    edge_b,
    rest_angles,
):
    """Vectorized conservative bend demand with precomputed adjacency."""

    degenerate = np.asarray(degenerate_mask, dtype=bool)
    if degenerate.shape != (int(face_count),):
        raise ValueError("MECHANICAL_TRUTH_DEGENERATE_MASK_SHAPE_INVALID")
    out = np.zeros(int(face_count), dtype=np.float64)
    out[degenerate] = math.pi
    if len(edge_a) == 0:
        return out

    pair_degenerate = degenerate[edge_a] | degenerate[edge_b]
    if np.any(pair_degenerate):
        out[edge_a[pair_degenerate]] = math.pi
        out[edge_b[pair_degenerate]] = math.pi

    regular = ~pair_degenerate
    if np.any(regular):
        a = edge_a[regular]
        b = edge_b[regular]
        posed_dot = np.clip(
            np.einsum("ij,ij->i", posed_normals[a], posed_normals[b]),
            -1.0,
            1.0,
        )
        delta = np.abs(np.arccos(posed_dot) - rest_angles[regular])
        np.maximum.at(out, a, delta)
        np.maximum.at(out, b, delta)
    return out


def _bend(rest_normals, posed_normals, faces, degenerate_mask):
    """Compatibility wrapper for the original bend helper contract."""

    edge_a, edge_b, rest_angles = _bend_precompute(rest_normals, faces)
    return _bend_precomputed(
        posed_normals,
        degenerate_mask,
        len(faces),
        edge_a,
        edge_b,
        rest_angles,
    )


def _edge_strain(rest, posed, faces, eps):
    rest_faces, posed_faces = rest[faces], posed[faces]
    values = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        rest_length = np.linalg.norm(rest_faces[:, b] - rest_faces[:, a], axis=1)
        posed_length = np.linalg.norm(posed_faces[:, b] - posed_faces[:, a], axis=1)
        if np.any(rest_length <= eps):
            raise ValueError("MECHANICAL_TRUTH_DEGENERATE_EDGE")
        values.append(np.abs(posed_length / rest_length - 1))
    return np.max(np.stack(values, axis=1), axis=1)


def build_mechanical_demand_truth_v1(
    rest_vertices,
    faces,
    posed_vertices,
    *,
    contact_face_mask=None,
    policy=MechanicalDemandTruthPolicyV1(),
):
    """Build training-only mechanical labels from authored motion.

    Rest topology remains fail-closed. Rank-deficient *posed* triangles are valid
    deformation evidence: strain remains mathematically defined through the 3x2
    deformation gradient, while undefined bend normals receive conservative pi
    demand and are explicitly censused in metadata.
    """

    policy.validate()
    rest, tri, posed = _validate(rest_vertices, faces, posed_vertices)
    inv_dm, rest_area, rest_normals = _basis(rest, tri, policy.epsilon)
    bend_edge_a, bend_edge_b, bend_rest_angles = _bend_precompute(rest_normals, tri)

    major_rows = []
    minor_rows = []
    direction_rows = []
    direction_valid_rows = []
    area_rows = []
    condition_rows = []
    bend_rows = []
    edge_rows = []
    degenerate_rows = []

    for frame in posed:
        (
            major,
            minor,
            direction,
            area_ratio,
            condition,
            normals,
            degenerate,
            direction_valid,
        ) = _frame(frame, tri, inv_dm, rest_area, rest_normals, policy.epsilon)
        major_rows.append(np.abs(major - 1))
        minor_rows.append(np.abs(minor - 1))
        direction_rows.append(direction)
        direction_valid_rows.append(direction_valid)
        area_rows.append(np.abs(area_ratio - 1))
        condition_rows.append(condition)
        bend_rows.append(
            _bend_precomputed(
                normals,
                degenerate,
                len(tri),
                bend_edge_a,
                bend_edge_b,
                bend_rest_angles,
            )
        )
        edge_rows.append(_edge_strain(rest, frame, tri, policy.epsilon))
        degenerate_rows.append(degenerate)

    major, minor, directions, areas, conditions, bends, edges, degenerate = map(
        lambda value: np.stack(value, axis=0),
        (
            major_rows,
            minor_rows,
            direction_rows,
            area_rows,
            condition_rows,
            bend_rows,
            edge_rows,
            degenerate_rows,
        ),
    )
    direction_valid = np.stack(direction_valid_rows, axis=0)

    direction_weight = (major + minor + 1e-8) * direction_valid
    direction = np.sum(directions * direction_weight[..., None], axis=0)
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.maximum(direction_norm, policy.epsilon)

    rigid = (
        (edges <= policy.rigid_edge_strain_tolerance)
        & (areas <= policy.rigid_area_change_tolerance)
        & (bends <= policy.rigid_bend_radians_tolerance)
        & (~degenerate)
    ).mean(axis=0)

    if contact_face_mask is None:
        contact = np.zeros(len(tri))
        contact_valid = False
    else:
        contact_mask = np.asarray(contact_face_mask, dtype=bool)
        if contact_mask.shape != (len(posed), len(tri)):
            raise ValueError("MECHANICAL_TRUTH_CONTACT_SHAPE_INVALID")
        contact = contact_mask.mean(axis=0)
        contact_valid = True

    quantile95 = lambda value: tuple(map(float, np.quantile(value, 0.95, axis=0)))
    degenerate_face_any = np.any(degenerate, axis=0)
    degenerate_frame_any = np.any(degenerate, axis=1)

    return MechanicalDemandTruthV1(
        quantile95(major),
        quantile95(minor),
        tuple((float(x), float(y)) for x, y in direction),
        quantile95(bends),
        tuple(map(float, rigid)),
        quantile95(areas),
        quantile95(conditions),
        quantile95(edges),
        tuple(map(float, contact)),
        contact_valid,
        len(tri),
        len(posed),
        metadata={
            "teacher_only": True,
            "rig_or_skin_inference_input": False,
            "principal_direction_representation": "COS2THETA_SIN2THETA",
            "aggregation": "P95_DEFORMATION_DEMAND_OVER_AUTHORED_SEQUENCE",
            "posed_rank_deficiency_policy": (
                "VALID_DEFORMATION_EVIDENCE__3X2_DEFORMATION_GRADIENT__"
                "DEGENERATE_BEND_CONSERVATIVE_PI"
            ),
            "bend_execution_policy": "PRECOMPUTED_ADJACENCY__VECTORIZED_EDGE_REDUCTION",
            "pose_degenerate_face_samples": int(degenerate.sum()),
            "pose_degenerate_face_count": int(degenerate_face_any.sum()),
            "pose_degenerate_frame_count": int(degenerate_frame_any.sum()),
            "pose_rank_zero_face_samples": int((~direction_valid).sum()),
            "policy": asdict(policy),
        },
    )
