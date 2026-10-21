#!/usr/bin/env python3
from __future__ import annotations

"""Independent static presentation court for Structural-Evidence A/B.

Measures four preregistered presentation metrics directly on strict free-running
A/B geometry using the same deterministic source foreground rays as the admitted
V2 tie-safe evidence. V2 truth is evaluation-only and never enters inference.
POSE_DEPENDENT_ORDER_FLIP_ACCURACY remains a dynamic-court metric; therefore this
source cannot mint the final presentation-family or scientific A/B PASS.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import zipfile
from typing import Any

import numpy as np

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.tessa.v2.experiment_v2 import sha256_file
from tools.iris_vnext.build_fit8_relation_all_hit_evidence_v1 import (
    _face_edge_components,
    _ray_triangle_all_parallel_grid,
)
from tools.iris_vnext.build_fit8_relation_all_hit_evidence_v2 import stack_visibility_evidence
from tools.iris_vnext.build_fit8_direct_surface_presentation_queries_v1 import (
    _canonical_rays,
    _depth_tie_epsilon,
)
from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import ARM_A, ARM_B, ARMS, load_json, validate_preregistration

SCHEMA = "RealSaS.StructuralEvidenceABStaticPresentationCourt.v1"
METRICS = (
    "GLOBAL_VISIBILITY_ACCURACY",
    "LOCAL_PARTIAL_ORDER_ACCURACY",
    "FRAGMENT_IDENTITY_STABILITY",
    "SOURCE_ART_SUPPORT_PRESERVATION",
)


def _nearest_hit(offsets: np.ndarray, points: np.ndarray, ray: int, target: np.ndarray) -> int:
    start, end = int(offsets[ray]), int(offsets[ray + 1])
    if end <= start:
        return -1
    block = points[start:end]
    return start + int(np.argmin(np.sum((block - target[None, :]) ** 2, axis=1)))


def _candidate_view(*, vertices: np.ndarray, faces: np.ndarray, camera: dict[str, Any],
                    pixels: np.ndarray, center: np.ndarray, half: float) -> dict[str, np.ndarray]:
    origins, directions, _ = _canonical_rays(
        camera,
        pixels,
        center=center,
        normalization_half=half,
    )
    components = _face_edge_components(faces)
    hits = _ray_triangle_all_parallel_grid(origins, directions, vertices, faces, components)
    offsets = np.asarray(hits["offsets"], dtype=np.int64)
    depths = np.asarray(hits["depth"], dtype=np.float64)
    points = np.asarray(hits["point"], dtype=np.float64)
    hit_faces = np.asarray(hits["face"], dtype=np.int64)
    eps = _depth_tie_epsilon(vertices)
    visibility = np.zeros(len(depths), dtype=np.float32)
    visibility_valid = np.zeros(len(depths), dtype=np.bool_)
    for ray in range(len(pixels)):
        start, end = int(offsets[ray]), int(offsets[ray + 1])
        p, v, _ = stack_visibility_evidence(depths[start:end], eps)
        visibility[start:end] = p
        visibility_valid[start:end] = v
    return {
        "offsets": offsets,
        "depths": depths,
        "points": points,
        "faces": hit_faces,
        "components": components,
        "visibility": visibility,
        "visibility_valid": visibility_valid,
        "tie_epsilon": np.asarray(eps, dtype=np.float64),
    }


def _truth_hit_slots(data: Any, prefix: str) -> dict[tuple[int, int], dict[str, Any]]:
    slots: dict[tuple[int, int], dict[str, Any]] = {}
    ray = np.asarray(data[prefix + "ray_index"], dtype=np.int64)
    for suffix in ("a", "b"):
        face = np.asarray(data[prefix + "face_" + suffix], dtype=np.int64)
        point = np.asarray(data[prefix + "point_" + suffix], dtype=np.float64)
        prob = np.asarray(data[prefix + "visibility_probability_" + suffix], dtype=np.float32)
        valid = np.asarray(data[prefix + "visibility_valid_" + suffix], dtype=np.bool_)
        for i in np.flatnonzero(valid):
            key = (int(ray[i]), int(face[i]))
            row = {"point": point[i], "visible": bool(prob[i] >= 0.5)}
            previous = slots.get(key)
            if previous is not None:
                if previous["visible"] != row["visible"] or not np.allclose(previous["point"], row["point"], atol=1e-6, rtol=0.0):
                    raise ValueError("STRUCTURAL_EVIDENCE_AB_STATIC_PRESENTATION_TRUTH_SLOT_DRIFT")
            else:
                slots[key] = row
    return slots


def _view_metrics(*, truth: Any, prefix: str, candidate: dict[str, np.ndarray],
                  fragment_votes: dict[int, list[int | None]]) -> dict[str, Any]:
    pixels = np.asarray(truth[prefix + "pixels_xy"], dtype=np.int64)
    offsets = candidate["offsets"]; points = candidate["points"]; depths = candidate["depths"]
    hit_faces = candidate["faces"]; face_components = candidate["components"]
    tie_eps = float(candidate["tie_epsilon"])

    hit_rays = int(np.count_nonzero(np.diff(offsets) >= 1))
    support = float(hit_rays / max(1, len(pixels)))

    visibility_correct = 0
    visibility_total = 0
    for (ray, truth_face), slot in _truth_hit_slots(truth, prefix).items():
        visibility_total += 1
        hit = _nearest_hit(offsets, points, ray, np.asarray(slot["point"], dtype=np.float64))
        if hit < 0:
            fragment_votes[truth_face].append(None)
            continue
        candidate_face = int(hit_faces[hit])
        fragment_votes[truth_face].append(int(face_components[candidate_face]))
        pred_valid = bool(candidate["visibility_valid"][hit])
        pred_visible = bool(candidate["visibility"][hit] >= 0.5)
        if pred_valid and pred_visible == bool(slot["visible"]):
            visibility_correct += 1

    ray = np.asarray(truth[prefix + "ray_index"], dtype=np.int64)
    base = np.asarray(truth[prefix + "base_pair_id"], dtype=np.int64)
    valid = np.asarray(truth[prefix + "front_valid"], dtype=np.bool_)
    truth_front = np.asarray(truth[prefix + "front_probability"], dtype=np.float32) >= 0.5
    point_a = np.asarray(truth[prefix + "point_a"], dtype=np.float64)
    point_b = np.asarray(truth[prefix + "point_b"], dtype=np.float64)
    _, first = np.unique(base, return_index=True)
    order_correct = 0
    order_total = 0
    for i in first.tolist():
        if not bool(valid[i]):
            continue
        order_total += 1
        ha = _nearest_hit(offsets, points, int(ray[i]), point_a[i])
        hb = _nearest_hit(offsets, points, int(ray[i]), point_b[i])
        if ha < 0 or hb < 0 or ha == hb:
            continue
        da, db = float(depths[ha]), float(depths[hb])
        if abs(da - db) <= tie_eps:
            continue
        if bool(da < db) == bool(truth_front[i]):
            order_correct += 1

    return {
        "source_foreground_ray_count": int(len(pixels)),
        "candidate_hit_ray_count": hit_rays,
        "SOURCE_ART_SUPPORT_PRESERVATION": support,
        "global_visibility_authoritative_slots": visibility_total,
        "global_visibility_correct_slots": visibility_correct,
        "GLOBAL_VISIBILITY_ACCURACY": float(visibility_correct / max(1, visibility_total)),
        "local_partial_order_authoritative_pairs": order_total,
        "local_partial_order_correct_pairs": order_correct,
        "LOCAL_PARTIAL_ORDER_ACCURACY": float(order_correct / max(1, order_total)),
    }


def _subject_metrics(*, subject_id: str, candidate_stage: Path, truth_path: Path,
                     cameras: dict[str, Any]) -> dict[str, Any]:
    with np.load(candidate_stage, allow_pickle=False) as c:
        vertices = np.asarray(c["vertices_normalized"], dtype=np.float64)
        faces = np.asarray(c["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_STATIC_PRESENTATION_CANDIDATE_SHAPE_INVALID:" + subject_id)
    center = np.asarray(cameras["common_center_xyz"], dtype=np.float64)
    half = float(cameras["common_half_extent"])
    fragment_votes: dict[int, list[int | None]] = defaultdict(list)
    views: list[dict[str, Any]] = []
    with np.load(truth_path, allow_pickle=False) as truth:
        for camera in cameras.get("cameras") or ():
            view_index = int(camera["view_index"])
            prefix = f"v{view_index}_"
            pixels = np.asarray(truth[prefix + "pixels_xy"], dtype=np.int64)
            candidate = _candidate_view(
                vertices=vertices, faces=faces, camera=camera, pixels=pixels,
                center=center, half=half,
            )
            views.append({"view_index": view_index, **_view_metrics(
                truth=truth, prefix=prefix, candidate=candidate, fragment_votes=fragment_votes
            )})
    if len(views) != 8:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_STATIC_PRESENTATION_VIEW_COUNT_DRIFT:" + subject_id)

    fragment_den = sum(len(rows) for rows in fragment_votes.values())
    fragment_num = 0
    for rows in fragment_votes.values():
        matched = [value for value in rows if value is not None]
        if matched:
            fragment_num += Counter(matched).most_common(1)[0][1]
    fragment_stability = float(fragment_num / max(1, fragment_den))

    def weighted(name: str, weight: str) -> float:
        numerator = sum(float(row[name]) * int(row[weight]) for row in views)
        denominator = sum(int(row[weight]) for row in views)
        return float(numerator / max(1, denominator))

    return {
        "subject_id": subject_id,
        "GLOBAL_VISIBILITY_ACCURACY": weighted("GLOBAL_VISIBILITY_ACCURACY", "global_visibility_authoritative_slots"),
        "LOCAL_PARTIAL_ORDER_ACCURACY": weighted("LOCAL_PARTIAL_ORDER_ACCURACY", "local_partial_order_authoritative_pairs"),
        "FRAGMENT_IDENTITY_STABILITY": fragment_stability,
        "SOURCE_ART_SUPPORT_PRESERVATION": weighted("SOURCE_ART_SUPPORT_PRESERVATION", "source_foreground_ray_count"),
        "fragment_identity_authoritative_observations": int(fragment_den),
        "fragment_identity_modal_consistent_observations": int(fragment_num),
        "views": views,
    }


def court(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = Path(args.preregistration).resolve(); prereg = load_json(prereg_path); validate_preregistration(prereg)
    policy = prereg["independent_court"]["presentation_primary"]
    evidence_root = Path(args.evidence_root).resolve()
    manifest_path = evidence_root / "ALL_HIT_RELATION_EVIDENCE_V2_MANIFEST.json"
    manifest = load_json(manifest_path)
    expected_manifest_sha = prereg["evidence_input_authority"]["manifest_sha256"]
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_STATIC_PRESENTATION_MANIFEST_SHA_DRIFT")
    if manifest.get("schema") != "RealSaS.IRISVNextFit8AllHitRelationEvidenceManifest.v2" or tuple(manifest.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_STATIC_PRESENTATION_MANIFEST_INVALID")
    evidence_rows = {str(row["subject_id"]): row for row in manifest["subjects"]}

    binding = load_json(Path(args.input_binding).resolve())
    legacy_rows = list(((binding.get("source_inputs") or {}).get("legacy_fit8") or {}).get("subjects") or ())
    if tuple(row.get("subject_id") for row in legacy_rows) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_STATIC_PRESENTATION_BINDING_ORDER_DRIFT")
    legacy = {str(row["subject_id"]): str(row["legacy_subject_id"]) for row in legacy_rows}

    free_root = Path(args.free_running_root).resolve()
    free_path = free_root / "STRUCTURAL_EVIDENCE_AB_FREE_RUNNING_RECEIPT.json"
    free = load_json(free_path)
    if tuple(free.get("arm_order") or ()) != ARMS or free.get("teacher_prefix_used") is not False:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_STATIC_PRESENTATION_FREE_RUN_INVALID")
    arm_receipts = {str(row["arm_id"]): row for row in free["arms"]}

    stage01 = Path(args.stage01_bundle).resolve()
    per_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    with zipfile.ZipFile(stage01, "r") as archive:
        for subject_id in SUBJECT_IDS:
            truth_row = evidence_rows[subject_id]
            truth_path = evidence_root / str(truth_row["file"])
            if sha256_file(truth_path) != str(truth_row["sha256"]):
                raise ValueError("STRUCTURAL_EVIDENCE_AB_STATIC_PRESENTATION_SUBJECT_SHA_DRIFT:" + subject_id)
            prefix = f"observations/{legacy[subject_id]}/"
            cameras = json.loads(archive.read(prefix + "camera_bundle.json"))
            for arm in ARMS:
                subject_rows = {str(row["subject_id"]): row for row in arm_receipts[arm]["subjects"]}
                row = subject_rows[subject_id]
                if row.get("completed") is not True or row.get("stop_reason") != "EOS":
                    per_arm[arm].append({"subject_id": subject_id, "incomplete": True})
                    continue
                stage = free_root / str(row["candidate_stage12_file"])
                if sha256_file(stage) != row.get("candidate_stage12_sha256"):
                    raise ValueError("STRUCTURAL_EVIDENCE_AB_STATIC_PRESENTATION_CANDIDATE_SHA_DRIFT:" + arm + ":" + subject_id)
                per_arm[arm].append(_subject_metrics(
                    subject_id=subject_id,
                    candidate_stage=stage,
                    truth_path=truth_path,
                    cameras=cameras,
                ))

    blockers: list[str] = []
    for arm in ARMS:
        if any(row.get("incomplete") for row in per_arm[arm]):
            blockers.append(arm + ":FREE_RUNNING_INCOMPLETE")
    macro: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    gains: dict[str, float] = {}
    single_subject_regressions: list[str] = []
    if not blockers:
        for arm in ARMS:
            for metric in METRICS:
                macro[arm][metric] = float(np.mean([float(row[metric]) for row in per_arm[arm]]))
        a_rows = {row["subject_id"]: row for row in per_arm[ARM_A]}
        b_rows = {row["subject_id"]: row for row in per_arm[ARM_B]}
        max_reg = float(policy["maximum_single_subject_absolute_regression"])
        for metric in METRICS:
            gains[metric] = float(macro[ARM_B][metric] - macro[ARM_A][metric])
            for subject_id in SUBJECT_IDS:
                delta = float(b_rows[subject_id][metric]) - float(a_rows[subject_id][metric])
                if delta < -max_reg:
                    single_subject_regressions.append(f"{subject_id}:{metric}:{delta:.9f}")
    blockers.extend(single_subject_regressions)

    result = {
        "schema": SCHEMA,
        "status": "STATIC_PRESENTATION_4_OF_5_MEASURED__DYNAMIC_ORDER_FLIP_REQUIRED" if not blockers else "STATIC_PRESENTATION_MEASURED_WITH_BLOCKERS__DYNAMIC_ORDER_FLIP_REQUIRED",
        "preregistration_sha256": sha256_file(prereg_path),
        "evidence_manifest_sha256": sha256_file(manifest_path),
        "free_running_receipt_sha256": sha256_file(free_path),
        "subject_order": list(SUBJECT_IDS),
        "measured_metrics": list(METRICS),
        "pending_metric": "POSE_DEPENDENT_ORDER_FLIP_ACCURACY",
        "arms": per_arm,
        "macro": macro,
        "b_minus_a_macro_gain": gains,
        "single_subject_regression_blockers": single_subject_regressions,
        "blockers": blockers,
        "v2_all_hit_truth_used_as_inference_conditioning": False,
        "v2_all_hit_truth_role": "INDEPENDENT_EVALUATION_ONLY",
        "presentation_primary_final_pass_claimed": False,
        "scientific_ab_pass_claimed": False,
        "next_action": "RUN_DYNAMIC_POSE_ORDER_AND_MOTION_COURT",
    }
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preregistration", default="canonical/STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json")
    p.add_argument("--input-binding", default="canonical/STRUCTURAL_EVIDENCE_INPUT_BINDING_V2.json")
    p.add_argument("--stage01-bundle", required=True)
    p.add_argument("--evidence-root", required=True)
    p.add_argument("--free-running-root", required=True)
    p.add_argument("--output", required=True)
    court(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
