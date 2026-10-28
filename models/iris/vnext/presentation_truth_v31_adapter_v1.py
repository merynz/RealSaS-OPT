"""Model-facing V3.1 target adapter for PresentationResponseTeacherV1.

The teacher package remains richer than these targets.  This adapter selects only
quantities with an explicit current objective and preserves unsupported heads as
invalid/abstained.  In particular:

* no source owner/material/bone ID enters a student input;
* no semantic cape/hair/etc. class is produced;
* front/occlusion train only on renderer-measured comparable pairs whose two hits
  both make non-negligible pixel contribution;
* same-drawable/cut/attachment remain abstained;
* bend/rigidity/contact remain abstained until their own qualified response law
  exists;
* principal tangent has an independent anisotropy validity mask.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np


PASS_RAY = "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION"


def _normalization(value: Mapping[str, object]) -> tuple[np.ndarray, float]:
    if value.get("schema") != "RealSaS.NormalizationDomainIR.v1":
        raise ValueError("V31_TRUTH_NORMALIZATION_SCHEMA")
    center = np.asarray(value.get("center_xyz"), dtype=np.float64)
    half = float(value.get("half_extent", 0.0))
    if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(half) or half <= 0:
        raise ValueError("V31_TRUTH_NORMALIZATION_VALUE")
    return center, half


def _mesh(rest_vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(rest_vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if (
        v.ndim != 2
        or v.shape[1:] != (3,)
        or not np.isfinite(v).all()
        or f.ndim != 2
        or f.shape[1:] != (3,)
        or not len(f)
        or np.min(f) < 0
        or np.max(f) >= len(v)
    ):
        raise ValueError("V31_TRUTH_MESH")
    return v, f


def _normalized(points: np.ndarray, center: np.ndarray, half: float) -> np.ndarray:
    value = (np.asarray(points, dtype=np.float64) - center) / half
    if not np.isfinite(value).all():
        raise ValueError("V31_TRUTH_NORMALIZED_POINT_NONFINITE")
    return value


def _array(arrays: Mapping[str, np.ndarray], name: str, shape0: int | None = None) -> np.ndarray:
    if name not in arrays:
        raise ValueError("V31_TRUTH_ARRAY_MISSING:" + name)
    value = np.asarray(arrays[name])
    if shape0 is not None and (value.ndim < 1 or value.shape[0] != shape0):
        raise ValueError("V31_TRUTH_ARRAY_CARDINALITY:" + name)
    return value


def build_v31_training_targets_v1(
    *,
    rest_vertices,
    faces,
    normalization: Mapping[str, object],
    presentation_arrays: Mapping[str, np.ndarray],
    packed_measurement: Mapping[str, object],
    contribution_floor: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    vertices, tri = _mesh(rest_vertices, faces)
    center, half = _normalization(normalization)
    if not math.isfinite(float(contribution_floor)) or contribution_floor < 0:
        raise ValueError("V31_TRUTH_CONTRIBUTION_FLOOR")
    face_count = len(tri)
    centroids = vertices[tri].mean(axis=1)
    face_points = _normalized(centroids, center, half)

    face_valid = _array(presentation_arrays, "surface_face_valid", face_count).astype(np.bool_, copy=False)
    tangent_valid = _array(presentation_arrays, "surface_tangent_valid", face_count).astype(np.bool_, copy=False)
    stretch_major = _array(presentation_arrays, "surface_stretch_major_mean", face_count).astype(np.float64, copy=False)
    stretch_minor = _array(presentation_arrays, "surface_stretch_minor_mean", face_count).astype(np.float64, copy=False)
    tangent = _array(presentation_arrays, "surface_tangent_direction_2theta", face_count).astype(np.float64, copy=False)
    if tangent.shape != (face_count, 2):
        raise ValueError("V31_TRUTH_TANGENT_LAYOUT")

    # Current teacher has no authored physical bend/stiffness/contact law.  Keep
    # finite placeholder carriers because tensor collation requires shapes, but
    # masks are false and mechanical_demand_loss_v3 is validity-separated.
    mechanical = {
        "mechanical_point_xyz_normalized": face_points,
        "stretch_major": stretch_major,
        "stretch_minor": stretch_minor,
        "tangent_direction_2theta": tangent,
        "bend_demand": np.zeros(face_count, dtype=np.float64),
        "rigidity_probability": np.zeros(face_count, dtype=np.float64),
        "contact_persistence_probability": np.zeros(face_count, dtype=np.float64),
        "stretch_valid": face_valid,
        "tangent_valid": tangent_valid,
        "bend_valid": np.zeros(face_count, dtype=np.bool_),
        "rigidity_valid": np.zeros(face_count, dtype=np.bool_),
        "contact_valid": np.zeros(face_count, dtype=np.bool_),
    }

    if packed_measurement.get("schema") != "RealSaS.SourceRayAppearanceMeasurementCandidate.v1":
        raise ValueError("V31_TRUTH_PACKED_MEASUREMENT_SCHEMA")

    pair_view: list[int] = []
    pair_face_a: list[int] = []
    pair_face_b: list[int] = []
    pair_point_a: list[np.ndarray] = []
    pair_point_b: list[np.ndarray] = []
    pair_front: list[float] = []
    pair_occlusion: list[float] = []
    pair_valid: list[bool] = []

    ray_view: list[int] = []
    ray_offsets = [0]
    ray_hit_face: list[int] = []
    ray_hit_point: list[np.ndarray] = []
    ray_hit_depth: list[float] = []
    ray_hit_contribution: list[float] = []
    ray_hit_valid: list[bool] = []

    for ray in packed_measurement.get("rays") or ():
        measured = dict(ray.get("measurement") or {})
        if measured.get("status") != PASS_RAY or measured.get("valid") is not True:
            continue
        hits = list(ray.get("canonical_hits") or ())
        alpha = np.asarray(measured.get("effective_alpha"), dtype=np.float64)
        contribution = np.asarray(measured.get("contribution"), dtype=np.float64)
        if alpha.shape != (len(hits),) or contribution.shape != (len(hits),):
            raise ValueError("V31_TRUTH_RAY_MEASUREMENT_LAYOUT")
        points: list[np.ndarray] = []
        for hit in hits:
            face = int(hit["face_index"])
            bary = np.asarray(hit["barycentric"], dtype=np.float64)
            if face < 0 or face >= face_count or bary.shape != (3,) or not np.isfinite(bary).all() or not np.isclose(bary.sum(), 1.0, atol=1.0e-7, rtol=0):
                raise ValueError("V31_TRUTH_HIT_ADDRESS")
            point = np.sum(vertices[tri[face]] * bary[:, None], axis=0)
            points.append(_normalized(point[None, :], center, half)[0])

        ray_view.append(int(ray["view_index"]))
        for index, hit in enumerate(hits):
            ray_hit_face.append(int(hit["face_index"]))
            ray_hit_point.append(points[index])
            ray_hit_depth.append(float(hit["depth"]) / half)
            ray_hit_contribution.append(float(contribution[index]))
            ray_hit_valid.append(True)
        ray_offsets.append(len(ray_hit_face))

        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                observable = bool(contribution[i] > contribution_floor and contribution[j] > contribution_floor)
                for a, b, front, occ in (
                    (i, j, 1.0, float(alpha[i])),
                    (j, i, 0.0, 0.0),
                ):
                    pair_view.append(int(ray["view_index"]))
                    pair_face_a.append(int(hits[a]["face_index"]))
                    pair_face_b.append(int(hits[b]["face_index"]))
                    pair_point_a.append(points[a])
                    pair_point_b.append(points[b])
                    pair_front.append(front if observable else 0.0)
                    pair_occlusion.append(occ if observable else 0.0)
                    pair_valid.append(observable)

    pair_count = len(pair_view)
    static_invalid = np.zeros(pair_count, dtype=np.bool_)
    zero_target = np.zeros(pair_count, dtype=np.float64)
    relation = {
        "relation_view_index": np.asarray(pair_view, dtype=np.int64),
        "relation_face_a": np.asarray(pair_face_a, dtype=np.int64),
        "relation_face_b": np.asarray(pair_face_b, dtype=np.int64),
        "relation_point_a_normalized": np.asarray(pair_point_a, dtype=np.float64).reshape(-1, 3),
        "relation_point_b_normalized": np.asarray(pair_point_b, dtype=np.float64).reshape(-1, 3),
        "same_drawable_affinity": zero_target.copy(),
        "cut_probability": zero_target.copy(),
        "attachment_affinity": zero_target.copy(),
        "front_probability": np.asarray(pair_front, dtype=np.float64),
        "occlusion_probability": np.asarray(pair_occlusion, dtype=np.float64),
        "projected_alpha_boundary_probability": zero_target.copy(),
        "projected_appearance_boundary_probability": zero_target.copy(),
        "same_drawable_affinity_valid": static_invalid.copy(),
        "cut_probability_valid": static_invalid.copy(),
        "attachment_affinity_valid": static_invalid.copy(),
        "front_probability_valid": np.asarray(pair_valid, dtype=np.bool_),
        "occlusion_probability_valid": np.asarray(pair_valid, dtype=np.bool_),
        "projected_alpha_boundary_probability_valid": static_invalid.copy(),
        "projected_appearance_boundary_probability_valid": static_invalid.copy(),
    }

    ray = {
        "ray_view_index": np.asarray(ray_view, dtype=np.int64),
        "ray_offsets": np.asarray(ray_offsets, dtype=np.int64),
        "ray_hit_face": np.asarray(ray_hit_face, dtype=np.int64),
        "ray_hit_point_normalized": np.asarray(ray_hit_point, dtype=np.float64).reshape(-1, 3),
        "ray_hit_depth_normalized": np.asarray(ray_hit_depth, dtype=np.float64),
        # Actual source-over contribution is the current global visibility target.
        # It is continuous and includes alpha; opaque-first-hit is not substituted.
        "ray_hit_visibility_target": np.asarray(ray_hit_contribution, dtype=np.float64),
        "ray_hit_visibility_valid": np.asarray(ray_hit_valid, dtype=np.bool_),
    }
    return {**mechanical, **relation, **ray}
