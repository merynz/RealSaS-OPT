#!/usr/bin/env python3
from __future__ import annotations

"""Final dynamic presentation/motion court for Structural-Evidence A/B.

This isolates the TESSA A/B question from future Fit8 AXIS/MIRA fitting.  Both
free-running A/B candidates are bound, for independent evaluation only, to the
same authored teacher deformation field by exact closest-point barycentric
surface addresses.  Teacher rest/posed geometry never enters training or either
arm's inference forward and cannot mint product M/G/W authority.

The court measures:
- POSE_DEPENDENT_ORDER_FLIP_ACCURACY from anonymous-owner frontmost flips;
- OWNER_FRAGMENT_STABILITY_OVER_MOTION from persistent candidate components;
- GAP_TEAR_RATE from existing frozen G3 dynamic numerical bounds;
- G3_DYNAMIC_QUALIFICATION under the authored deformation field;
- ATTACHMENT_PERSISTENCE_WHERE_AUTHORITATIVE as ABSTAIN unless explicit contact
  authority exists (the current authored-teacher export does not mint it).

Together with the separately sealed topology/surface and static-presentation
courts this file may produce the final preregistered A/B decision.  A scientific
A/B PASS still does not authorize canonical M, product promotion, or unseen
generalization.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import zipfile
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import _triangle_metrics_batch_exact
from compiler.realsas_compiler_core.rigid_attachment_surface_anchor_v1 import _closest_point_barycentric
from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.tessa.v2.experiment_v2 import sha256_file
from tools.iris_vnext.build_fit8_relation_all_hit_evidence_v1 import (
    _face_edge_components,
    _ray_triangle_all_parallel_grid,
)
from tools.iris_vnext.build_fit8_direct_surface_presentation_queries_v1 import _canonical_rays
from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import ARM_A, ARM_B, ARMS, load_json, validate_preregistration

SCHEMA = "RealSaS.StructuralEvidenceABDynamicMotionCourt.v1"
GRID_SIZE = 32


def _uniform_pixels(resolution: int, grid_size: int = GRID_SIZE) -> np.ndarray:
    if resolution <= 0 or grid_size <= 0:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_GRID_INVALID")
    axis = np.floor((np.arange(grid_size, dtype=np.float64) + 0.5) * resolution / grid_size).astype(np.int64)
    axis = np.clip(axis, 0, resolution - 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)


def _normalize_source_world(vertices: np.ndarray, center: np.ndarray, half: float) -> np.ndarray:
    value = (np.asarray(vertices, dtype=np.float64) - center[None, :]) / float(half)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_NORMALIZATION_INVALID")
    return value


def _exact_surface_binding(points: np.ndarray, rest: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact closest-face binding using centroid/radius branch-and-bound."""
    p = np.asarray(points, dtype=np.float64)
    r = np.asarray(rest, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = r[f]
    if p.ndim != 2 or p.shape[1] != 3 or tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_BINDING_SHAPE_INVALID")
    centroids = tri.mean(axis=1)
    radii = np.max(np.linalg.norm(tri - centroids[:, None, :], axis=2), axis=1)
    max_radius = float(np.max(radii))
    tree = cKDTree(centroids)
    face_index = np.empty(len(p), dtype=np.int64)
    bary = np.empty((len(p), 3), dtype=np.float64)
    distance = np.empty(len(p), dtype=np.float64)
    for i, point in enumerate(p):
        _, seed = tree.query(point, k=1)
        q, b = _closest_point_barycentric(point, tri[int(seed)])
        best = float(np.linalg.norm(point - q)); best_face = int(seed); best_bary = b
        candidate_ids = tree.query_ball_point(point, r=best + max_radius + 1.0e-14)
        for fi in candidate_ids:
            lower = max(0.0, float(np.linalg.norm(point - centroids[fi])) - float(radii[fi]))
            if lower > best + 1.0e-14:
                continue
            q, b = _closest_point_barycentric(point, tri[int(fi)])
            d = float(np.linalg.norm(point - q))
            if d < best - 1.0e-14 or (abs(d - best) <= 1.0e-14 and int(fi) < best_face):
                best, best_face, best_bary = d, int(fi), b
        face_index[i] = best_face; bary[i] = best_bary; distance[i] = best
    if not np.isfinite(bary).all() or np.any(bary < -1.0e-10) or np.any(bary > 1.0 + 1.0e-10):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_BINDING_BARYCENTRIC_INVALID")
    if np.max(np.abs(np.sum(bary, axis=1) - 1.0)) > 1.0e-9:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_BINDING_BARYCENTRIC_SUM_DRIFT")
    return face_index, bary, distance


def _deform_from_teacher(posed_teacher: np.ndarray, teacher_faces: np.ndarray,
                         bound_face: np.ndarray, bary: np.ndarray) -> np.ndarray:
    tri = posed_teacher[teacher_faces[bound_face]]
    return np.einsum("vi,vij->vj", bary, tri)


def _front_face(vertices: np.ndarray, faces: np.ndarray, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    components = _face_edge_components(faces)
    hits = _ray_triangle_all_parallel_grid(origins, directions, vertices, faces, components)
    offsets = np.asarray(hits["offsets"], dtype=np.int64)
    hit_faces = np.asarray(hits["face"], dtype=np.int64)
    out = np.full(len(origins), -1, dtype=np.int64)
    nonempty = np.flatnonzero(np.diff(offsets) > 0)
    for ray in nonempty.tolist():
        out[ray] = int(hit_faces[int(offsets[ray])])
    return out


def _candidate_face_owner(vertex_owner: np.ndarray, faces: np.ndarray) -> np.ndarray:
    own = vertex_owner[np.asarray(faces, dtype=np.int64)]
    same = (own[:, 0] == own[:, 1]) & (own[:, 0] == own[:, 2])
    return np.where(same, own[:, 0], -1).astype(np.int64)


def _load_policy(path: Path) -> dict[str, float]:
    value = load_json(path)
    if value.get("schema") != "RealSaS.QualifiedMeshProductPolicy.v2":
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_G3_POLICY_SCHEMA_INVALID")
    g3 = value.get("g3") or {}
    return {
        "min_area": float(g3["minimum_dynamic_area_ratio"]),
        "max_area": float(g3["maximum_dynamic_area_ratio"]),
        "max_condition": float(g3["maximum_dynamic_condition_number"]),
    }


def _dynamic_metrics(*, subject_id: str, candidate_path: Path, teacher_path: Path,
                     teacher_receipt_path: Path, cameras: dict[str, Any], policy: dict[str, float]) -> dict[str, Any]:
    receipt = load_json(teacher_receipt_path)
    if receipt.get("teacher_only") is not True or receipt.get("teacher_rig_skin_motion_allowed_as_inference_input") is not False:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_TEACHER_AUTHORITY_INVALID:" + subject_id)
    if receipt.get("canonical_M_G_W_minted") is not False:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_TEACHER_PRODUCT_AUTHORITY_VIOLATION:" + subject_id)
    center = np.asarray(cameras["common_center_xyz"], dtype=np.float64).reshape(3)
    half = float(cameras["common_half_extent"])
    if not np.isfinite(half) or half <= 0.0:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_NORMALIZATION_HALF_INVALID")

    with np.load(teacher_path, allow_pickle=False) as t:
        teacher_rest = _normalize_source_world(t["rest_vertices_source_world"], center, half)
        teacher_faces = np.asarray(t["faces"], dtype=np.int64)
        teacher_face_owner = np.asarray(t["face_owner"], dtype=np.int64)
        teacher_poses_world = np.asarray(t["posed_vertices_source_world"], dtype=np.float64)
    if teacher_poses_world.ndim != 3 or teacher_poses_world.shape[1:] != teacher_rest.shape:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_TEACHER_POSE_SHAPE_INVALID")
    teacher_poses = (teacher_poses_world - center[None, None, :]) / half
    if not np.isfinite(teacher_poses).all():
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_TEACHER_POSE_NONFINITE")

    with np.load(candidate_path, allow_pickle=False) as c:
        candidate_rest = np.asarray(c["vertices_normalized"], dtype=np.float64)
        candidate_faces = np.asarray(c["faces"], dtype=np.int64)
    if candidate_rest.ndim != 2 or candidate_rest.shape[1] != 3 or candidate_faces.ndim != 2 or candidate_faces.shape[1] != 3:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_CANDIDATE_SHAPE_INVALID")

    bound_face, bary, bind_distance = _exact_surface_binding(candidate_rest, teacher_rest, teacher_faces)
    vertex_owner = teacher_face_owner[bound_face]
    candidate_owner = _candidate_face_owner(vertex_owner, candidate_faces)
    candidate_components = _face_edge_components(candidate_faces)

    camera_rows: list[tuple[np.ndarray, np.ndarray]] = []
    teacher_rest_front_owner: list[np.ndarray] = []
    candidate_rest_front_owner: list[np.ndarray] = []
    for camera in cameras.get("cameras") or ():
        pixels = _uniform_pixels(int(camera["resolution"]))
        origins, directions, _ = _canonical_rays(camera, pixels, center=center, normalization_half=half)
        camera_rows.append((origins, directions))
        tf = _front_face(teacher_rest, teacher_faces, origins, directions)
        cf = _front_face(candidate_rest, candidate_faces, origins, directions)
        teacher_rest_front_owner.append(np.where(tf >= 0, teacher_face_owner[np.maximum(tf, 0)], -1))
        candidate_rest_front_owner.append(np.where(cf >= 0, candidate_owner[np.maximum(cf, 0)], -1))

    owner_observations: dict[int, int] = defaultdict(int)
    owner_component_votes: dict[int, list[int]] = defaultdict(list)
    flip_total = 0; flip_correct = 0
    gap_tear_events = 0; face_frame_total = 0
    frame_pass_count = 0
    frame_reports: list[dict[str, Any]] = []

    for frame_index, teacher_pose in enumerate(teacher_poses):
        candidate_pose = _deform_from_teacher(teacher_pose, teacher_faces, bound_face, bary)
        area, condition, edge_min, edge_max, _ = _triangle_metrics_batch_exact(candidate_rest, candidate_pose, candidate_faces)
        finite = np.isfinite(area) & np.isfinite(condition) & np.isfinite(edge_min) & np.isfinite(edge_max)
        bad = (~finite) | (area < policy["min_area"]) | (area > policy["max_area"]) | (condition > policy["max_condition"])
        events = int(np.count_nonzero(bad)); gap_tear_events += events; face_frame_total += len(candidate_faces)
        frame_pass = events == 0
        frame_pass_count += int(frame_pass)
        frame_reports.append({
            "frame_index": frame_index,
            "g3_pass": frame_pass,
            "gap_tear_face_events": events,
            "face_count": int(len(candidate_faces)),
            "minimum_area_ratio": float(np.min(area[finite])) if np.any(finite) else None,
            "maximum_area_ratio": float(np.max(area[finite])) if np.any(finite) else None,
            "maximum_condition_number": float(np.max(condition[finite])) if np.any(finite) else None,
            "minimum_edge_ratio": float(np.min(edge_min[finite])) if np.any(finite) else None,
            "maximum_edge_ratio": float(np.max(edge_max[finite])) if np.any(finite) else None,
        })

        for view_index, (origins, directions) in enumerate(camera_rows):
            teacher_face = _front_face(teacher_pose, teacher_faces, origins, directions)
            candidate_face = _front_face(candidate_pose, candidate_faces, origins, directions)
            teacher_owner = np.where(teacher_face >= 0, teacher_face_owner[np.maximum(teacher_face, 0)], -1)
            cand_owner = np.where(candidate_face >= 0, candidate_owner[np.maximum(candidate_face, 0)], -1)
            cand_component = np.where(candidate_face >= 0, candidate_components[np.maximum(candidate_face, 0)], -1)

            valid_teacher = teacher_owner >= 0
            for owner in np.unique(teacher_owner[valid_teacher]).tolist():
                mask = teacher_owner == owner
                owner_observations[int(owner)] += int(np.count_nonzero(mask))
                good = mask & (cand_owner == owner) & (cand_component >= 0)
                owner_component_votes[int(owner)].extend(map(int, cand_component[good].tolist()))

            base_teacher = teacher_rest_front_owner[view_index]
            base_candidate = candidate_rest_front_owner[view_index]
            flips = (base_teacher >= 0) & (teacher_owner >= 0) & (base_teacher != teacher_owner)
            flip_total += int(np.count_nonzero(flips))
            flip_correct += int(np.count_nonzero(flips & (base_candidate == base_teacher) & (cand_owner == teacher_owner)))

    stability_num = 0; stability_den = 0; owner_rows = []
    for owner in sorted(owner_observations):
        total = int(owner_observations[owner]); votes = owner_component_votes.get(owner, [])
        modal = Counter(votes).most_common(1)[0][1] if votes else 0
        stability_num += modal; stability_den += total
        owner_rows.append({
            "anonymous_owner_index": int(owner),
            "authoritative_front_observations": total,
            "modal_candidate_component_consistent_observations": int(modal),
            "stability": float(modal / max(1, total)),
        })

    frame_count = int(len(teacher_poses))
    return {
        "subject_id": subject_id,
        "teacher_sampled_frame_count": frame_count,
        "dynamic_ray_grid_size": GRID_SIZE,
        "dynamic_rays_per_view": GRID_SIZE * GRID_SIZE,
        "surface_binding_rule": "EXACT_CLOSEST_POINT_BARYCENTRIC__EVALUATION_ONLY",
        "binding_distance_mean": float(np.mean(bind_distance)),
        "binding_distance_p95": float(np.percentile(bind_distance, 95)),
        "binding_distance_max": float(np.max(bind_distance)),
        "OWNER_FRAGMENT_STABILITY_OVER_MOTION": float(stability_num / max(1, stability_den)),
        "owner_fragment_authoritative_observations": int(stability_den),
        "owner_fragment_modal_consistent_observations": int(stability_num),
        "owner_rows": owner_rows,
        "POSE_DEPENDENT_ORDER_FLIP_ACCURACY": (float(flip_correct / flip_total) if flip_total else None),
        "pose_dependent_order_flip_valid_count": int(flip_total),
        "pose_dependent_order_flip_correct_count": int(flip_correct),
        "GAP_TEAR_RATE": float(gap_tear_events / max(1, face_frame_total)),
        "gap_tear_event_definition": "FACE_FRAME_VIOLATES_FROZEN_G3_DYNAMIC_AREA_OR_CONDITION_BOUNDS_OR_NONFINITE",
        "gap_tear_face_events": int(gap_tear_events),
        "face_frame_total": int(face_frame_total),
        "G3_DYNAMIC_QUALIFICATION": bool(frame_pass_count == frame_count),
        "g3_dynamic_frame_pass_fraction": float(frame_pass_count / max(1, frame_count)),
        "g3_dynamic_frame_pass_count": int(frame_pass_count),
        "ATTACHMENT_PERSISTENCE_WHERE_AUTHORITATIVE": None,
        "attachment_authoritative_count": 0,
        "attachment_abstain_reason": "AUTHORED_TEACHER_EXPORT_HAS_NO_EXPLICIT_CONTACT_OR_ATTACHMENT_RELATION_AUTHORITY",
        "frames": frame_reports,
    }


def _macro(rows: list[dict[str, Any]], metric: str) -> float | None:
    valid = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return float(np.mean(valid)) if valid else None


def court(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = Path(args.preregistration).resolve(); prereg = load_json(prereg_path); validate_preregistration(prereg)
    presentation_policy = prereg["independent_court"]["presentation_primary"]
    motion_policy = prereg["independent_court"]["motion_primary"]
    g3_policy_path = Path(args.g3_policy).resolve(); g3_policy = _load_policy(g3_policy_path)

    topology_path = Path(args.topology_surface_court).resolve(); topology = load_json(topology_path)
    static_path = Path(args.static_presentation_court).resolve(); static = load_json(static_path)
    if topology.get("schema") != "RealSaS.StructuralEvidenceABTopologySurfaceCourt.v1":
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_TOPOLOGY_COURT_INVALID")
    if static.get("schema") != "RealSaS.StructuralEvidenceABStaticPresentationCourt.v1":
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_STATIC_PRESENTATION_COURT_INVALID")

    binding_path = Path(args.input_binding).resolve(); binding = load_json(binding_path)
    teacher_spec = ((binding.get("source_inputs") or {}).get("authored_teachers") or {})
    teacher_rows = list(teacher_spec.get("subjects") or ())
    if tuple(row.get("subject_id") for row in teacher_rows) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_TEACHER_BINDING_ORDER_DRIFT")
    teacher_by_subject = {str(row["subject_id"]): row for row in teacher_rows}

    free_root = Path(args.free_running_root).resolve()
    free_path = free_root / "STRUCTURAL_EVIDENCE_AB_FREE_RUNNING_RECEIPT.json"; free = load_json(free_path)
    if tuple(free.get("arm_order") or ()) != ARMS or free.get("teacher_prefix_used") is not False:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_FREE_RUN_INVALID")
    arm_receipts = {str(row["arm_id"]): row for row in free["arms"]}
    teacher_root = Path(args.teacher_root).resolve(); stage01 = Path(args.stage01_bundle).resolve()

    per_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    with zipfile.ZipFile(stage01, "r") as archive:
        for subject_id in SUBJECT_IDS:
            spec = teacher_by_subject[subject_id]; legacy_id = str(spec["legacy_subject_id"])
            teacher_path = teacher_root / f"{legacy_id}.teacher.npz"
            teacher_receipt = teacher_path.with_suffix(teacher_path.suffix + ".json")
            if sha256_file(teacher_path) != str(spec["teacher_sha256"]):
                raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_TEACHER_SHA_DRIFT:" + subject_id)
            if sha256_file(teacher_receipt) != str(spec["receipt_sha256"]):
                raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_TEACHER_RECEIPT_SHA_DRIFT:" + subject_id)
            cameras = json.loads(archive.read(f"observations/{legacy_id}/camera_bundle.json"))
            for arm in ARMS:
                subject_rows = {str(row["subject_id"]): row for row in arm_receipts[arm]["subjects"]}
                row = subject_rows[subject_id]
                if row.get("completed") is not True or row.get("stop_reason") != "EOS":
                    per_arm[arm].append({"subject_id": subject_id, "incomplete": True})
                    continue
                candidate = free_root / str(row["candidate_stage12_file"])
                if sha256_file(candidate) != row.get("candidate_stage12_sha256"):
                    raise ValueError("STRUCTURAL_EVIDENCE_AB_DYNAMIC_CANDIDATE_SHA_DRIFT:" + arm + ":" + subject_id)
                per_arm[arm].append(_dynamic_metrics(
                    subject_id=subject_id,
                    candidate_path=candidate,
                    teacher_path=teacher_path,
                    teacher_receipt_path=teacher_receipt,
                    cameras=cameras,
                    policy=g3_policy,
                ))

    blockers: list[str] = []
    for arm in ARMS:
        if any(row.get("incomplete") for row in per_arm[arm]):
            blockers.append(arm + ":FREE_RUNNING_INCOMPLETE")

    macro = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for metric in (
            "OWNER_FRAGMENT_STABILITY_OVER_MOTION",
            "GAP_TEAR_RATE",
            "POSE_DEPENDENT_ORDER_FLIP_ACCURACY",
            "g3_dynamic_frame_pass_fraction",
        ):
            macro[arm][metric] = _macro(per_arm[arm], metric)

    a_rows = {row["subject_id"]: row for row in per_arm[ARM_A] if not row.get("incomplete")}
    b_rows = {row["subject_id"]: row for row in per_arm[ARM_B] if not row.get("incomplete")}
    owner_gain = None; gap_reduction = None; pose_gain = None
    motion_pass = False; presentation_pass = False
    subject_owner_regressions: list[str] = []; subject_pose_regressions: list[str] = []
    if not blockers:
        owner_gain = float(macro[ARM_B]["OWNER_FRAGMENT_STABILITY_OVER_MOTION"] - macro[ARM_A]["OWNER_FRAGMENT_STABILITY_OVER_MOTION"])
        a_gap = float(macro[ARM_A]["GAP_TEAR_RATE"]); b_gap = float(macro[ARM_B]["GAP_TEAR_RATE"])
        gap_reduction = float((a_gap - b_gap) / a_gap) if a_gap > 0.0 else (0.0 if b_gap <= 0.0 else float("-inf"))
        if macro[ARM_A]["POSE_DEPENDENT_ORDER_FLIP_ACCURACY"] is not None and macro[ARM_B]["POSE_DEPENDENT_ORDER_FLIP_ACCURACY"] is not None:
            pose_gain = float(macro[ARM_B]["POSE_DEPENDENT_ORDER_FLIP_ACCURACY"] - macro[ARM_A]["POSE_DEPENDENT_ORDER_FLIP_ACCURACY"])
        max_owner_reg = float(motion_policy["maximum_single_subject_owner_stability_regression"])
        max_pose_reg = float(presentation_policy["maximum_single_subject_absolute_regression"])
        for subject_id in SUBJECT_IDS:
            owner_delta = float(b_rows[subject_id]["OWNER_FRAGMENT_STABILITY_OVER_MOTION"]) - float(a_rows[subject_id]["OWNER_FRAGMENT_STABILITY_OVER_MOTION"])
            if owner_delta < -max_owner_reg:
                subject_owner_regressions.append(f"{subject_id}:{owner_delta:.9f}")
            av = a_rows[subject_id]["POSE_DEPENDENT_ORDER_FLIP_ACCURACY"]; bv = b_rows[subject_id]["POSE_DEPENDENT_ORDER_FLIP_ACCURACY"]
            if av is not None and bv is not None and float(bv) - float(av) < -max_pose_reg:
                subject_pose_regressions.append(f"{subject_id}:{float(bv)-float(av):.9f}")
        g3_no_regression = float(macro[ARM_B]["g3_dynamic_frame_pass_fraction"]) >= float(macro[ARM_A]["g3_dynamic_frame_pass_fraction"]) - 1.0e-12
        motion_pass = (
            owner_gain >= float(motion_policy["minimum_owner_fragment_stability_absolute_gain_for_b"])
            and gap_reduction >= float(motion_policy["minimum_gap_tear_rate_relative_reduction_for_b"])
            and not subject_owner_regressions
            and g3_no_regression
        )

        static_macro = static.get("macro") or {}
        metric_names = (
            "GLOBAL_VISIBILITY_ACCURACY",
            "LOCAL_PARTIAL_ORDER_ACCURACY",
            "FRAGMENT_IDENTITY_STABILITY",
            "SOURCE_ART_SUPPORT_PRESERVATION",
        )
        a_values = [float(static_macro[ARM_A][name]) for name in metric_names]
        b_values = [float(static_macro[ARM_B][name]) for name in metric_names]
        if macro[ARM_A]["POSE_DEPENDENT_ORDER_FLIP_ACCURACY"] is not None and macro[ARM_B]["POSE_DEPENDENT_ORDER_FLIP_ACCURACY"] is not None:
            a_values.append(float(macro[ARM_A]["POSE_DEPENDENT_ORDER_FLIP_ACCURACY"]))
            b_values.append(float(macro[ARM_B]["POSE_DEPENDENT_ORDER_FLIP_ACCURACY"]))
        presentation_macro_gain = float(np.mean(b_values) - np.mean(a_values)) if len(a_values) == 5 else None
        presentation_pass = (
            presentation_macro_gain is not None
            and presentation_macro_gain >= float(presentation_policy["minimum_macro_absolute_gain_for_b"])
            and not static.get("blockers")
            and not subject_pose_regressions
        )
    else:
        g3_no_regression = False; presentation_macro_gain = None

    blockers.extend("OWNER_REGRESSION:" + row for row in subject_owner_regressions)
    blockers.extend("POSE_ORDER_REGRESSION:" + row for row in subject_pose_regressions)
    topology_pass = bool(topology.get("topology_surface_no_regression_pass"))
    scientific_pass = bool(topology_pass and presentation_pass and motion_pass and not blockers)
    result = {
        "schema": SCHEMA,
        "status": "PASS__STRUCTURAL_EVIDENCE_AB__FIT8_SCOPE_ONLY" if scientific_pass else "FAIL__STRUCTURAL_EVIDENCE_AB__LOCKED_COURT",
        "preregistration_sha256": sha256_file(prereg_path),
        "topology_surface_court_sha256": sha256_file(topology_path),
        "static_presentation_court_sha256": sha256_file(static_path),
        "g3_policy_sha256": sha256_file(g3_policy_path),
        "free_running_receipt_sha256": sha256_file(free_path),
        "subject_order": list(SUBJECT_IDS),
        "dynamic_grid_size": GRID_SIZE,
        "teacher_motion_role": "INDEPENDENT_EVALUATION_DEFORMATION_FIELD_ONLY",
        "teacher_rig_skin_motion_used_in_inference_forward": False,
        "teacher_topology_used_in_inference_forward": False,
        "product_axis_or_mira_fit_required_for_this_isolated_tessa_court": False,
        "arms": per_arm,
        "macro": macro,
        "b_minus_a_owner_fragment_stability_gain": owner_gain,
        "b_gap_tear_relative_reduction": gap_reduction,
        "b_minus_a_pose_dependent_order_flip_gain": pose_gain,
        "presentation_macro_gain_all_5_metrics": presentation_macro_gain,
        "topology_surface_no_regression_pass": topology_pass,
        "presentation_primary_pass": presentation_pass,
        "motion_primary_pass": motion_pass,
        "g3_dynamic_no_regression": g3_no_regression,
        "attachment_persistence_authority_available": False,
        "subject_owner_regressions": subject_owner_regressions,
        "subject_pose_order_regressions": subject_pose_regressions,
        "blockers": blockers,
        "scientific_ab_pass_claimed": scientific_pass,
        "fit8_shared_system_contract_qualification_only": True,
        "unseen_generalization_claimed": False,
        "canonical_m_authorized": False,
        "product_promotion_authorized": False,
    }
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preregistration", default="canonical/STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json")
    p.add_argument("--input-binding", default="canonical/STRUCTURAL_EVIDENCE_INPUT_BINDING_V2.json")
    p.add_argument("--g3-policy", default="canonical/QUALIFIED_MESH_PRODUCT_POLICY_V2_20260919.json")
    p.add_argument("--stage01-bundle", required=True)
    p.add_argument("--teacher-root", required=True)
    p.add_argument("--free-running-root", required=True)
    p.add_argument("--topology-surface-court", required=True)
    p.add_argument("--static-presentation-court", required=True)
    p.add_argument("--output", required=True)
    result = court(p.parse_args())
    return 0 if result["scientific_ab_pass_claimed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
