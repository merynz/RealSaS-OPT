"""Generic presentation/response truth derived from authored source interventions.

This module deliberately avoids semantic part classes.  It consumes exact source
geometry, the sealed isolated-response tensor, admitted camera rows and measured
renderer interventions.  Its outputs describe what surfaces *do* and how they
compose: pose-conditioned ray order, measured alpha/transmittance/contribution,
local metric response and response discontinuity.

No function here mints student supervision, canonical Mesh ownership, attachment,
secondary-motion policy or product authority.  Privileged source face indices are
measurement addresses only and must be rebound by a qualified downstream adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from models.iris.truth.graphics_oracle_v1 import (
    camera_from_mapping,
    face_centroid_candidate_pixels_v1,
    measure_orthographic_ray_stack_v1,
)
from models.iris.truth.measurements_v1 import triangle_metric_strain


PASS_RAY = "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION"


def _mesh(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if (
        v.ndim != 2
        or v.shape[1:] != (3,)
        or not len(v)
        or not np.isfinite(v).all()
        or f.ndim != 2
        or f.shape[1:] != (3,)
        or not len(f)
        or not np.issubdtype(f.dtype, np.integer)
        or np.min(f) < 0
        or np.max(f) >= len(v)
    ):
        raise ValueError("PRESENTATION_RESPONSE_TRIANGULATED_FINITE_MESH_REQUIRED")
    return v, f.astype(np.int64, copy=False)


def _scale(vertices: np.ndarray) -> float:
    value = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("PRESENTATION_RESPONSE_POSITIVE_MESH_SCALE_REQUIRED")
    return value


def _selected_action_indices(count: int, maximum: int) -> np.ndarray:
    if type(count) is not int or count < 0 or type(maximum) is not int or maximum < 0:
        raise ValueError("PRESENTATION_RESPONSE_ACTION_SELECTION_CONTRACT")
    if count == 0 or maximum == 0:
        return np.empty(0, dtype=np.int64)
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    # Deterministic coverage of the sealed action order.  Selection does not read
    # any response outcome and therefore cannot positive-mine the teacher labels.
    return np.unique(np.linspace(0, count - 1, maximum, dtype=np.int64))


def control_balanced_action_weights_v1(actions: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Explicit action law: equal mass per control, then equal mass within control.

    The calibration producer intentionally leaves action mass unsealed.  This
    policy is an explicit downstream intervention measure and must be preregistered
    before outcome inspection.  It is invariant to duplicating rows within one
    control only when the duplicated mass is correspondingly split here.
    """
    rows = tuple(actions)
    if not rows:
        raise ValueError("PRESENTATION_RESPONSE_ACTION_ROWS_REQUIRED")
    by_control: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        control = row.get("control_index_privileged")
        amplitude = row.get("amplitude")
        sign = row.get("sign")
        if type(control) is not int or control < 0:
            raise ValueError("PRESENTATION_RESPONSE_CONTROL_INDEX_REQUIRED")
        if not isinstance(amplitude, (float, int)) or not math.isfinite(float(amplitude)) or float(amplitude) <= 0:
            raise ValueError("PRESENTATION_RESPONSE_ACTION_AMPLITUDE_REQUIRED")
        if sign not in (-1, 1):
            raise ValueError("PRESENTATION_RESPONSE_ACTION_SIGN_REQUIRED")
        by_control.setdefault(control, []).append(index)
    controls = sorted(by_control)
    if controls != list(range(len(controls))):
        raise ValueError("PRESENTATION_RESPONSE_CONTIGUOUS_CONTROL_SET_REQUIRED")
    weights = np.zeros(len(rows), dtype=np.float64)
    control_mass = 1.0 / len(controls)
    for control in controls:
        indices = by_control[control]
        row_mass = control_mass / len(indices)
        weights[indices] = row_mass
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("PRESENTATION_RESPONSE_ACTION_MASS_DRIFT")
    return weights


def _face_adjacency(faces: np.ndarray) -> np.ndarray:
    edge_owner: dict[tuple[int, int], list[int]] = {}
    for face_index, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_owner.setdefault(edge, []).append(face_index)
    pairs = set()
    for owners in edge_owner.values():
        if len(owners) == 2:
            a, b = sorted(owners)
            pairs.add((a, b))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def build_surface_response_arrays_v1(
    rest_vertices,
    faces,
    isolated_position,
    actions: Sequence[Mapping[str, object]],
    *,
    degeneracy_fraction: float = 1.0e-10,
) -> dict[str, np.ndarray]:
    """Measure intervention-aggregated, semantics-free surface response.

    The returned principal stretches are statistics of the measured authored
    kinematic response under the explicit control-balanced action law.  They are
    not physical stiffness.  ``nonrigid_response`` is a continuous response
    magnitude, never a cape/hair/cloth class.
    """
    rest, tri = _mesh(rest_vertices, faces)
    posed = np.asarray(isolated_position, dtype=np.float64)
    if posed.ndim != 3 or posed.shape[1:] != rest.shape or not np.isfinite(posed).all():
        raise ValueError("PRESENTATION_RESPONSE_ISOLATED_POSITION_LAYOUT")
    if posed.shape[0] != len(actions):
        raise ValueError("PRESENTATION_RESPONSE_ACTION_RESPONSE_CARDINALITY")
    weights = control_balanced_action_weights_v1(actions)
    scale = _scale(rest)
    tolerance = float(degeneracy_fraction) * scale * scale
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("PRESENTATION_RESPONSE_DEGENERACY_POLICY")

    rest_tri = rest[tri]
    posed_tri = posed[:, tri, :]
    repeated_rest = np.broadcast_to(rest_tri[None, ...], posed_tri.shape)
    strain, area_ratio, valid = triangle_metric_strain(
        repeated_rest,
        posed_tri,
        degeneracy_tolerance=tolerance,
    )
    # C = 2E + I.  Its eigenvalues are squared principal stretches in the
    # qualified rest tangent metric.
    cauchy_green = 2.0 * strain + np.eye(2, dtype=np.float64)
    eigval, eigvec = np.linalg.eigh(cauchy_green)
    eigval = np.clip(eigval, 0.0, None)
    stretch_minor_action = np.sqrt(eigval[..., 0])
    stretch_major_action = np.sqrt(eigval[..., 1])
    anisotropy_action = np.abs(stretch_major_action - stretch_minor_action)
    # Principal major eigenvector in the rest tangent basis.  2theta removes the
    # sign ambiguity of an unoriented principal axis.
    major_axis = eigvec[..., :, 1]
    theta = np.arctan2(major_axis[..., 1], major_axis[..., 0])
    tangent_2theta_action = np.stack((np.cos(2.0 * theta), np.sin(2.0 * theta)), axis=-1)

    w = weights[:, None]
    valid_f = valid.astype(np.float64)
    mass = np.sum(w * valid_f, axis=0)
    safe_mass = np.where(mass > 0, mass, 1.0)
    major_mean = np.sum(w * np.where(valid, stretch_major_action, 0.0), axis=0) / safe_mass
    minor_mean = np.sum(w * np.where(valid, stretch_minor_action, 0.0), axis=0) / safe_mass
    area_mean = np.sum(w * np.where(valid, area_ratio, 0.0), axis=0) / safe_mass
    anisotropy_mean = np.sum(w * np.where(valid, anisotropy_action, 0.0), axis=0) / safe_mass
    tangent_vector = np.sum(
        (w * valid_f * anisotropy_action)[..., None] * tangent_2theta_action,
        axis=0,
    )
    tangent_norm = np.linalg.norm(tangent_vector, axis=-1)
    tangent_valid = (mass > 0) & (anisotropy_mean > 1.0e-5) & (tangent_norm > 1.0e-8)
    tangent_mean = np.zeros((len(tri), 2), dtype=np.float64)
    tangent_mean[tangent_valid] = tangent_vector[tangent_valid] / tangent_norm[tangent_valid, None]

    strain_delta = np.stack(
        (stretch_major_action - 1.0, stretch_minor_action - 1.0), axis=-1
    )
    nonrigid = np.sqrt(
        np.sum(weights[:, None, None] * np.where(valid[..., None], strain_delta * strain_delta, 0.0), axis=(0, 2))
        / np.maximum(mass, 1.0e-12)
    )

    rest_centroid = rest_tri.mean(axis=1)
    posed_centroid = posed_tri.mean(axis=2)
    response = posed_centroid - rest_centroid[None, :, :]
    response_energy = np.sqrt(np.sum(weights[:, None, None] * response * response, axis=(0, 2)))

    adjacency = _face_adjacency(tri)
    coherence = np.zeros(len(adjacency), dtype=np.float64)
    coherence_valid = np.zeros(len(adjacency), dtype=np.bool_)
    response_jump = np.zeros(len(adjacency), dtype=np.float64)
    if len(adjacency):
        weighted = response * np.sqrt(weights)[:, None, None]
        flat = np.transpose(weighted, (1, 0, 2)).reshape(len(tri), -1)
        norm = np.linalg.norm(flat, axis=1)
        for index, (a, b) in enumerate(adjacency):
            denom = float(norm[a] * norm[b])
            if denom > 1.0e-12:
                value = float(np.clip(np.dot(flat[a], flat[b]) / denom, -1.0, 1.0))
                coherence[index] = value
                response_jump[index] = 0.5 * (1.0 - value)
                coherence_valid[index] = True

    face_valid = mass > 0
    return {
        "action_weight": weights,
        "surface_face_valid": face_valid.astype(np.bool_),
        "surface_stretch_major_mean": np.where(face_valid, major_mean, 0.0),
        "surface_stretch_minor_mean": np.where(face_valid, minor_mean, 0.0),
        "surface_area_ratio_mean": np.where(face_valid, area_mean, 0.0),
        "surface_anisotropy_mean": np.where(face_valid, anisotropy_mean, 0.0),
        "surface_tangent_direction_2theta": tangent_mean,
        "surface_tangent_valid": tangent_valid.astype(np.bool_),
        "surface_nonrigid_response": np.where(face_valid, nonrigid, 0.0),
        "surface_response_energy": response_energy,
        "response_pair_face": adjacency,
        "response_pair_coherence": coherence,
        "response_pair_jump": response_jump,
        "response_pair_valid": coherence_valid,
    }


def build_pose_conditioned_order_arrays_v1(
    rest_vertices,
    faces,
    isolated_position,
    cameras: Sequence[Mapping[str, object]],
    *,
    maximum_actions: int = 16,
    maximum_rays_per_view: int = 32,
    candidate_multiplier: int = 4,
) -> dict[str, np.ndarray]:
    """Build tie-safe order truth at rest and bounded isolated response poses.

    Every valid same-ray unordered face pair is emitted in *both* directions, so
    front-order supervision contains an exact complementary negative without
    fabricating an unrelated non-overlap negative.
    """
    rest, tri = _mesh(rest_vertices, faces)
    posed = np.asarray(isolated_position, dtype=np.float64)
    if posed.ndim != 3 or posed.shape[1:] != rest.shape or not np.isfinite(posed).all():
        raise ValueError("PRESENTATION_ORDER_ISOLATED_POSITION_LAYOUT")
    if type(maximum_rays_per_view) is not int or maximum_rays_per_view < 1:
        raise ValueError("PRESENTATION_ORDER_RAY_BUDGET")
    if type(candidate_multiplier) is not int or candidate_multiplier < 1:
        raise ValueError("PRESENTATION_ORDER_CANDIDATE_MULTIPLIER")
    camera_rows = tuple(cameras)
    if len(camera_rows) != 8:
        raise ValueError("PRESENTATION_ORDER_EXACT_EIGHT_CAMERAS_REQUIRED")
    parsed = tuple(camera_from_mapping(row) for row in camera_rows)
    action_indices = _selected_action_indices(int(posed.shape[0]), maximum_actions)
    pose_action_index = np.concatenate((np.asarray([-1], dtype=np.int64), action_indices))

    ray_pose: list[int] = []
    ray_view: list[int] = []
    ray_x: list[int] = []
    ray_y: list[int] = []
    ray_complete: list[bool] = []
    stack_offsets = [0]
    stack_face: list[int] = []
    stack_depth: list[float] = []
    pair_pose: list[int] = []
    pair_view: list[int] = []
    pair_ray: list[int] = []
    pair_a: list[int] = []
    pair_b: list[int] = []
    pair_front: list[float] = []
    pair_valid: list[bool] = []

    poses = [rest] + [posed[int(index)] for index in action_indices]
    for pose_index, vertices in enumerate(poses):
        tie_epsilon = max(_scale(vertices) * 1.0e-7, 1.0e-9)
        for camera in parsed:
            pixels = face_centroid_candidate_pixels_v1(
                vertices,
                tri,
                camera,
                maximum=maximum_rays_per_view * candidate_multiplier,
            )
            candidates = []
            for x, y in pixels:
                stack = measure_orthographic_ray_stack_v1(
                    vertices, tri, camera, x, y, tie_epsilon=tie_epsilon
                )
                # Prefer overlap-bearing rays but keep deterministic pixel order
                # as the secondary key.  Geometry is the only selection input.
                candidates.append((-len(stack.hits), y, x, stack))
            candidates.sort(key=lambda row: row[:3])
            for _neg_hits, _y, _x, stack in candidates[:maximum_rays_per_view]:
                ray_index = len(ray_pose)
                ray_pose.append(pose_index)
                ray_view.append(int(stack.view_index))
                ray_x.append(int(stack.pixel_x))
                ray_y.append(int(stack.pixel_y))
                ray_complete.append(bool(stack.complete))
                local_start = len(stack_face)
                for hit in stack.hits:
                    stack_face.append(int(hit.face_index))
                    stack_depth.append(float(hit.depth))
                stack_offsets.append(len(stack_face))
                # A tie invalidates only pairs that cross the tied depth.  V1
                # conservatively abstains all pairs on a ray with any tie.
                valid = bool(stack.complete and not stack.has_depth_tie)
                hits = tuple(stack.hits)
                for i in range(len(hits)):
                    for j in range(i + 1, len(hits)):
                        for a, b, target in (
                            (hits[i], hits[j], 1.0),
                            (hits[j], hits[i], 0.0),
                        ):
                            pair_pose.append(pose_index)
                            pair_view.append(int(stack.view_index))
                            pair_ray.append(ray_index)
                            pair_a.append(int(a.face_index))
                            pair_b.append(int(b.face_index))
                            pair_front.append(target if valid else 0.0)
                            pair_valid.append(valid)
                if len(stack_face) < local_start:
                    raise AssertionError("PRESENTATION_ORDER_INTERNAL_OFFSET_REGRESSION")

    return {
        "pose_action_index": pose_action_index,
        "order_ray_pose_index": np.asarray(ray_pose, dtype=np.int64),
        "order_ray_view_index": np.asarray(ray_view, dtype=np.int64),
        "order_ray_pixel_x": np.asarray(ray_x, dtype=np.int64),
        "order_ray_pixel_y": np.asarray(ray_y, dtype=np.int64),
        "order_ray_complete": np.asarray(ray_complete, dtype=np.bool_),
        "order_stack_offsets": np.asarray(stack_offsets, dtype=np.int64),
        "order_stack_face": np.asarray(stack_face, dtype=np.int64),
        "order_stack_depth": np.asarray(stack_depth, dtype=np.float64),
        "order_pair_pose_index": np.asarray(pair_pose, dtype=np.int64),
        "order_pair_view_index": np.asarray(pair_view, dtype=np.int64),
        "order_pair_ray_index": np.asarray(pair_ray, dtype=np.int64),
        "order_pair_face_a": np.asarray(pair_a, dtype=np.int64),
        "order_pair_face_b": np.asarray(pair_b, dtype=np.int64),
        "order_pair_front_probability": np.asarray(pair_front, dtype=np.float64),
        "order_pair_front_valid": np.asarray(pair_valid, dtype=np.bool_),
    }


def build_measured_compositing_arrays_v1(payload: Mapping[str, object], *, contribution_floor: float = 1.0e-6) -> dict[str, np.ndarray]:
    """Turn qualified renderer interventions into balanced pair/composite targets.

    For ordered hits A before B, A attenuates B by ``alpha_A`` under the admitted
    source-over law.  The reverse ordered relation is an exact comparable negative.
    This replaces the historical positive-only ``same ray => occlusion=1`` target.
    Fully hidden/noncontributing pairs remain useful oracle evidence but are marked
    non-observable for the first student target adapter.
    """
    if payload.get("schema") != "RealSaS.SourceRayAppearanceMeasurementCandidate.v1":
        raise ValueError("PRESENTATION_COMPOSITING_MEASUREMENT_SCHEMA")
    if not math.isfinite(float(contribution_floor)) or contribution_floor < 0:
        raise ValueError("PRESENTATION_COMPOSITING_CONTRIBUTION_FLOOR")

    hit_ray: list[int] = []
    hit_face: list[int] = []
    hit_depth: list[float] = []
    hit_alpha: list[float] = []
    hit_transmittance: list[float] = []
    hit_contribution: list[float] = []
    hit_valid: list[bool] = []
    pair_ray: list[int] = []
    pair_a: list[int] = []
    pair_b: list[int] = []
    pair_front: list[float] = []
    pair_occlusion: list[float] = []
    pair_valid: list[bool] = []
    pair_student_observable: list[bool] = []
    removal_hit: list[int] = []
    removal_delta_rgba: list[list[float]] = []

    for ray in payload.get("rays") or ():
        measurement = dict(ray.get("measurement") or {})
        if measurement.get("status") != PASS_RAY or measurement.get("valid") is not True:
            continue
        hits = list(ray.get("canonical_hits") or ())
        alpha = np.asarray(measurement.get("effective_alpha"), dtype=np.float64)
        trans = np.asarray(measurement.get("incoming_transmittance"), dtype=np.float64)
        contribution = np.asarray(measurement.get("contribution"), dtype=np.float64)
        if (
            alpha.shape != (len(hits),)
            or trans.shape != (len(hits),)
            or contribution.shape != (len(hits),)
            or not np.isfinite(alpha).all()
            or not np.isfinite(trans).all()
            or not np.isfinite(contribution).all()
            or np.any((alpha < 0) | (alpha > 1))
            or np.any((trans < 0) | (trans > 1))
            or np.any((contribution < 0) | (contribution > 1))
        ):
            raise ValueError("PRESENTATION_COMPOSITING_MEASUREMENT_LAYOUT")
        local_to_global = []
        for i, hit in enumerate(hits):
            global_hit = len(hit_face)
            local_to_global.append(global_hit)
            hit_ray.append(int(ray["ray_index"]))
            hit_face.append(int(hit["face_index"]))
            hit_depth.append(float(hit["depth"]))
            hit_alpha.append(float(alpha[i]))
            hit_transmittance.append(float(trans[i]))
            hit_contribution.append(float(contribution[i]))
            hit_valid.append(True)
        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                visible_pair = bool(
                    contribution[i] > contribution_floor and contribution[j] > contribution_floor
                )
                for a, b, front, occ in (
                    (i, j, 1.0, float(alpha[i])),
                    (j, i, 0.0, 0.0),
                ):
                    pair_ray.append(int(ray["ray_index"]))
                    pair_a.append(int(hits[a]["face_index"]))
                    pair_b.append(int(hits[b]["face_index"]))
                    pair_front.append(front)
                    pair_occlusion.append(occ)
                    pair_valid.append(True)
                    pair_student_observable.append(visible_pair)

        full = np.asarray(ray.get("reconstructed_full_small_rgba"), dtype=np.float64)
        interventions = list(ray.get("interventions") or ())
        if full.shape == (4,) and np.isfinite(full).all() and len(interventions) == len(hits):
            for i, row in enumerate(interventions):
                removed = np.asarray(row.get("removal_small_rgba"), dtype=np.float64)
                if removed.shape != (4,) or not np.isfinite(removed).all():
                    raise ValueError("PRESENTATION_COMPOSITING_REMOVAL_LAYOUT")
                removal_hit.append(local_to_global[i])
                removal_delta_rgba.append((full - removed).tolist())

    return {
        "composite_hit_ray_index": np.asarray(hit_ray, dtype=np.int64),
        "composite_hit_face": np.asarray(hit_face, dtype=np.int64),
        "composite_hit_depth": np.asarray(hit_depth, dtype=np.float64),
        "composite_hit_effective_alpha": np.asarray(hit_alpha, dtype=np.float64),
        "composite_hit_incoming_transmittance": np.asarray(hit_transmittance, dtype=np.float64),
        "composite_hit_contribution": np.asarray(hit_contribution, dtype=np.float64),
        "composite_hit_valid": np.asarray(hit_valid, dtype=np.bool_),
        "composite_pair_ray_index": np.asarray(pair_ray, dtype=np.int64),
        "composite_pair_face_a": np.asarray(pair_a, dtype=np.int64),
        "composite_pair_face_b": np.asarray(pair_b, dtype=np.int64),
        "composite_pair_front_probability": np.asarray(pair_front, dtype=np.float64),
        "composite_pair_occlusion_probability": np.asarray(pair_occlusion, dtype=np.float64),
        "composite_pair_valid": np.asarray(pair_valid, dtype=np.bool_),
        "composite_pair_student_observable": np.asarray(pair_student_observable, dtype=np.bool_),
        "raster_removal_hit_index": np.asarray(removal_hit, dtype=np.int64),
        "raster_removal_delta_rgba": np.asarray(removal_delta_rgba, dtype=np.float64).reshape(-1, 4),
    }
