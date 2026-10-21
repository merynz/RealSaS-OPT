#!/usr/bin/env python3
from __future__ import annotations

"""Preregistered first A/B court: geometry, source masks, topology, face economy.

A = exact V8 direct T512->T1024 surface.
B = strict free-running TESSA V2 candidate from the same model-produced evidence.
This court cannot promote TESSA to mandatory/product authority because Compiler,
G3/G5 and depth/render-normal evidence are explicitly outside its scope.
"""

import argparse
import json
from pathlib import Path
import zipfile
from typing import Any

import numpy as np

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.tessa.v2.experiment_v2 import load_preregistration_v1, sha256_file
from tools.corpus.iris_observation_admission_v1 import project_mask
from tools.iris_vnext.audit_fit8_direct_surface_geometry_v1 import measure_surface_against_shard
from tools.iris_vnext.audit_fit8_direct_surface_silhouette_v1 import _legacy_subject_map, _mask_metrics, sha256_bytes

SCHEMA = "RealSaS.TESSAV2Fit8PartialStructuralizationABCourt.v1"
PASS_STATUS = "SCIENTIFIC_STRUCTURALIZATION_CANDIDATE__PRODUCT_PROMOTION_INCOMPLETE"
FAIL_STATUS = "TESSA_V2_NOT_SUPPORTED_BY_FIRST_STRUCTURALIZATION_COURT"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("TESSA_V2_AB_JSON_OBJECT_REQUIRED")
    return value


def bidirectional(row: dict[str, Any], key: str) -> float:
    distance = row["surface_distance"]
    return max(float(distance["candidate_to_teacher"][key]), float(distance["teacher_to_candidate"][key]))


def open_boundary_chain_count(faces: np.ndarray) -> int:
    f = np.asarray(faces, dtype=np.int64)
    incidence: dict[tuple[int, int], int] = {}
    for a, b, c in f.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(u), int(v))))
            incidence[edge] = incidence.get(edge, 0) + 1
    boundary = [edge for edge, count in incidence.items() if count == 1]
    graph: dict[int, set[int]] = {}
    for a, b in boundary:
        graph.setdefault(a, set()).add(b); graph.setdefault(b, set()).add(a)
    remaining = set(graph); chains = 0
    while remaining:
        seed = remaining.pop(); stack = [seed]; component = [seed]
        while stack:
            vertex = stack.pop()
            for neighbor in graph[vertex]:
                if neighbor in remaining:
                    remaining.remove(neighbor); component.append(neighbor); stack.append(neighbor)
        if not all(len(graph[v]) == 2 for v in component):
            chains += 1
    return chains


def b_silhouette(
    *,
    subject_id: str,
    candidate_path: Path,
    archive: zipfile.ZipFile,
    stage_row: dict[str, Any],
) -> dict[str, Any]:
    legacy_id = str(stage_row["legacy_subject_id"])
    prefix = f"observations/{legacy_id}/"
    admission_bytes = archive.read(prefix + "IRIS_OBSERVATION_ADMISSION.json")
    camera_bytes = archive.read(prefix + "camera_bundle.json")
    if sha256_bytes(admission_bytes) != stage_row["observation_admission"]:
        raise ValueError("TESSA_V2_AB_STAGE01_ADMISSION_SHA_DRIFT:" + subject_id)
    if sha256_bytes(camera_bytes) != stage_row["camera_bundle"]:
        raise ValueError("TESSA_V2_AB_STAGE01_CAMERA_SHA_DRIFT:" + subject_id)
    admission = json.loads(admission_bytes); cameras = json.loads(camera_bytes)
    with np.load(candidate_path, allow_pickle=False) as data:
        vertices = np.asarray(data["vertices_normalized"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
    center = np.asarray(cameras["common_center_xyz"], dtype=np.float64)
    half = float(cameras["common_half_extent"])
    vertices_world = center[None] + half * vertices
    admitted = {int(row["view_index"]): row for row in admission["views"]}
    views: list[dict[str, Any]] = []
    for camera in cameras["cameras"]:
        index = int(camera["view_index"]); resolution = int(camera["resolution"])
        mask_bytes = archive.read(prefix + f"V{index}.mask.bin")
        if sha256_bytes(mask_bytes) != admitted[index]["mask_sha256"]:
            raise ValueError(f"TESSA_V2_AB_STAGE01_MASK_SHA_DRIFT:{subject_id}:V{index}")
        truth = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(resolution, resolution) > 0
        predicted = project_mask(vertices_world, faces, camera)
        views.append({"view_index": index, **_mask_metrics(predicted, truth)})
    return {
        "mean_iou": float(np.mean([v["iou"] for v in views])),
        "minimum_iou": float(min(v["iou"] for v in views)),
        "maximum_mask_gap_px": float(max(v["symmetric_maximum_mask_gap_px"] for v in views)),
        "views": views,
    }


def court(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = Path(args.preregistration).resolve(); prereg = load_preregistration_v1(prereg_path)
    policy = prereg["partial_scientific_ab_court"]
    prepared_root = Path(args.prepared_root).resolve()
    prepared_manifest_path = Path(args.prepared_manifest).resolve(); prepared = load_json(prepared_manifest_path)
    prep_receipt_path = Path(args.preparation_receipt).resolve(); prep_receipt = load_json(prep_receipt_path)
    stage01_path = Path(args.stage01_bundle).resolve()
    direct_root = Path(args.direct_surface_root).resolve()
    a_geometry_path = Path(args.direct_geometry_court).resolve(); a_geometry = load_json(a_geometry_path)
    a_silhouette_path = Path(args.direct_silhouette_court).resolve(); a_silhouette = load_json(a_silhouette_path)
    free_root = Path(args.free_running_root).resolve()
    free_receipt_path = free_root / "TESSA_V2_FREE_RUNNING_RECEIPT.json"; free = load_json(free_receipt_path)

    if tuple(prepared.get("subject_order") or ()) != SUBJECT_IDS:
        # Manifest V4 uses subject_order; keep exact order binding.
        raise ValueError("TESSA_V2_AB_PREPARED_SUBJECT_ORDER_DRIFT")
    if tuple(a_geometry.get("subject_order") or ()) != SUBJECT_IDS or tuple(a_silhouette.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("TESSA_V2_AB_A_ARM_SUBJECT_ORDER_DRIFT")
    if tuple(free.get("subject_order") or ()) != SUBJECT_IDS or free.get("teacher_prefix_used") is not False:
        raise ValueError("TESSA_V2_AB_FREE_RUNNING_AUTHORITY_DRIFT")

    prepared_rows = {str(row["subject_id"]): row for row in prepared["subjects"]}
    a_g = {str(row["subject_id"]): row for row in a_geometry["subjects"]}
    a_s = {str(row["subject_id"]): row for row in a_silhouette["subjects"]}
    b_receipts = {str(row["subject_id"]): row for row in free["subjects"]}
    stage_map = _legacy_subject_map(prep_receipt)

    blockers: list[str] = []; subjects: list[dict[str, Any]] = []
    with zipfile.ZipFile(stage01_path, "r") as archive:
        for index, subject_id in enumerate(SUBJECT_IDS):
            br = b_receipts[subject_id]
            sb: list[str] = []
            if br.get("completed") is not True or br.get("stop_reason") != "EOS" or not br.get("candidate_stage12_file"):
                sb.append("FREE_RUNNING_EOS_INCOMPLETE")
                subjects.append({"subject_id": subject_id, "blockers": sb})
                blockers.extend(f"{subject_id}:{x}" for x in sb)
                continue
            topo = br.get("topology_court") or {}
            if topo.get("grammar_valid") is not True or topo.get("finite_geometry") is not True or topo.get("within_scale_policy") is not True:
                sb.append("FREE_RUNNING_TOPOLOGY_OR_SCALE_INVALID")
            b_stage = free_root / str(br["candidate_stage12_file"])
            if sha256_file(b_stage) != br["candidate_stage12_sha256"]:
                raise ValueError("TESSA_V2_AB_B_CANDIDATE_SHA_DRIFT:" + subject_id)
            manifest_row = prepared_rows[subject_id]
            shard = prepared_root / str(manifest_row["shard_file"])
            if sha256_file(shard) != manifest_row["shard_sha256"]:
                raise ValueError("TESSA_V2_AB_PREPARED_SHARD_SHA_DRIFT:" + subject_id)
            teacher_components = ((manifest_row.get("metadata") or {}).get("topology_teacher") or {}).get("component_count")
            b_geometry = measure_surface_against_shard(
                subject_id=subject_id,
                stage12_path=b_stage,
                prepared_shard_path=shard,
                teacher_component_count=int(teacher_components) if teacher_components is not None else None,
                seed=26091991 + index,
            )
            b_mask = b_silhouette(subject_id=subject_id, candidate_path=b_stage, archive=archive, stage_row=stage_map[subject_id])
            ag = a_g[subject_id]; ass = a_s[subject_id]
            a_topo = ag["candidate_topology"]; b_topo = b_geometry["candidate_topology"]
            a_faces = int(a_topo["face_count"]); b_faces = int(b_topo["face_count"])
            reduction = 1.0 - (float(b_faces) / float(a_faces)) if a_faces else float("-inf")
            a_p95, b_p95 = bidirectional(ag, "p95"), bidirectional(b_geometry, "p95")
            a_max, b_max = bidirectional(ag, "max"), bidirectional(b_geometry, "max")
            if reduction < float(policy["minimum_face_reduction_fraction"]): sb.append("FACE_REDUCTION")
            if b_p95 > a_p95 * (1.0 + float(policy["maximum_relative_surface_p95_increase"])): sb.append("SURFACE_P95_REGRESSION")
            if b_max > a_max * (1.0 + float(policy["maximum_relative_surface_max_increase"])): sb.append("SURFACE_MAX_REGRESSION")
            if int(b_topo["nonmanifold_edge_count"]) > int(a_topo["nonmanifold_edge_count"]): sb.append("NEW_NONMANIFOLD_EDGES")
            if int(b_topo["degenerate_face_count"]) > int(a_topo["degenerate_face_count"]): sb.append("NEW_ZERO_AREA_FACES")
            if int(b_topo["component_count"]) != int(a_topo["component_count"]): sb.append("GEOMETRIC_COMPONENT_COUNT_CHANGED")
            with np.load(direct_root / "subjects" / subject_id / "DIRECT_IRIS_SURFACE.stage12.npz", allow_pickle=False) as data:
                a_open = open_boundary_chain_count(data["faces"])
            with np.load(b_stage, allow_pickle=False) as data:
                b_open = open_boundary_chain_count(data["faces"])
            if b_open > a_open: sb.append("NEW_OPEN_BOUNDARY_CHAINS")
            a_views = {int(v["view_index"]): v for v in ass["views"]}
            b_views = {int(v["view_index"]): v for v in b_mask["views"]}
            max_iou_drop = max(float(a_views[i]["iou"]) - float(b_views[i]["iou"]) for i in sorted(a_views))
            if max_iou_drop > float(policy["maximum_mask_iou_drop"]): sb.append("SOURCE_MASK_IOU_REGRESSION")
            subjects.append({
                "subject_id": subject_id,
                "a_face_count": a_faces,
                "b_face_count": b_faces,
                "face_reduction_fraction": reduction,
                "a_surface_p95": a_p95,
                "b_surface_p95": b_p95,
                "a_surface_max": a_max,
                "b_surface_max": b_max,
                "a_mean_mask_iou": float(ass["mean_iou"]),
                "b_mean_mask_iou": float(b_mask["mean_iou"]),
                "maximum_per_view_iou_drop": max_iou_drop,
                "a_open_boundary_chain_count": a_open,
                "b_open_boundary_chain_count": b_open,
                "a_component_count": int(a_topo["component_count"]),
                "b_component_count": int(b_topo["component_count"]),
                "b_topology_court": topo,
                "blockers": sb,
            })
            blockers.extend(f"{subject_id}:{x}" for x in sb)

    passed = not blockers
    result = {
        "schema": SCHEMA,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "preregistration_sha256": sha256_file(prereg_path),
        "prepared_manifest_sha256": sha256_file(prepared_manifest_path),
        "stage01_bundle_sha256": sha256_file(stage01_path),
        "direct_geometry_court_sha256": sha256_file(a_geometry_path),
        "direct_silhouette_court_sha256": sha256_file(a_silhouette_path),
        "free_running_receipt_sha256": sha256_file(free_receipt_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": subjects,
        "blockers": blockers,
        "partial_structuralization_court_pass": passed,
        "scientific_pass_scope": "SOURCE_GEOMETRY_MASK_TOPOLOGY_AND_FACE_ECONOMY_ONLY" if passed else None,
        "tessa_mandatory": False,
        "product_promotion_complete": False,
        "compiler_qualified": False,
        "canonical_m_authorized": False,
        "unseen_generalization_claimed": False,
        "missing_promotion_courts": ["DEPTH", "RENDER_NORMAL", "COMPILER_STATIC", "COMPILER_SUPPORT_BINDING", "G3", "G5"],
        "next_action": "BUILD_FULL_COMPILER_DYNAMIC_PROMOTION_COURT" if passed else "DO_NOT_PROMOTE_TESSA__INSPECT_FAILED_DIMENSIONS",
    }
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preregistration", default="canonical/TESSA_V2_FIT8_PREREGISTRATION_V1.json")
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--preparation-receipt", required=True)
    p.add_argument("--stage01-bundle", required=True)
    p.add_argument("--direct-surface-root", required=True)
    p.add_argument("--direct-geometry-court", required=True)
    p.add_argument("--direct-silhouette-court", required=True)
    p.add_argument("--free-running-root", required=True)
    p.add_argument("--output", required=True)
    result = court(p.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["partial_structuralization_court_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
