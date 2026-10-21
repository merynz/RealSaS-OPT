#!/usr/bin/env python3
from __future__ import annotations

"""Preregistered topology/surface no-regression court for Structural-Evidence A/B.

This is one required family of the locked court, not the final scientific
A/B decision. Presentation and motion primary families are evaluated separately.
"""

import argparse
import json
from pathlib import Path
import zipfile
from typing import Any

import numpy as np

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.tessa.v2.experiment_v2 import sha256_file
from tools.iris_vnext.audit_fit8_direct_surface_geometry_v1 import measure_surface_against_shard
from tools.iris_vnext.audit_fit8_direct_surface_silhouette_v1 import _legacy_subject_map
from tools.iris_vnext.court_fit8_tessa_v2_partial_ab_v1 import (
    b_silhouette,
    bidirectional,
    load_json,
    open_boundary_chain_count,
)
from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import ARM_A, ARM_B, ARMS, validate_preregistration

SCHEMA = "RealSaS.StructuralEvidenceABTopologySurfaceCourt.v1"
PASS_STATUS = "PASS__TOPOLOGY_SURFACE_NO_REGRESSION__PRESENTATION_MOTION_COURTS_REQUIRED"
FAIL_STATUS = "FAIL__TOPOLOGY_SURFACE_NO_REGRESSION"


def _arm_metrics(*, arm_id: str, arm_receipt: dict[str, Any], free_root: Path,
                 prepared_root: Path, prepared_rows: dict[str, Any], stage_map: dict[str, Any],
                 archive: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    rows = {str(row["subject_id"]): row for row in arm_receipt.get("subjects") or ()}
    if tuple(rows) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_COURT_ARM_SUBJECT_ORDER_DRIFT:" + arm_id)
    out: dict[str, dict[str, Any]] = {}
    for index, subject_id in enumerate(SUBJECT_IDS):
        row = rows[subject_id]
        if row.get("completed") is not True or row.get("stop_reason") != "EOS" or not row.get("candidate_stage12_file"):
            out[subject_id] = {"completed": False, "blocker": "FREE_RUNNING_EOS_INCOMPLETE"}
            continue
        topo = row.get("topology_court") or {}
        stage = free_root / str(row["candidate_stage12_file"])
        if sha256_file(stage) != row.get("candidate_stage12_sha256"):
            raise ValueError("STRUCTURAL_EVIDENCE_AB_COURT_CANDIDATE_SHA_DRIFT:" + arm_id + ":" + subject_id)
        manifest_row = prepared_rows[subject_id]
        shard = prepared_root / str(manifest_row["shard_file"])
        if sha256_file(shard) != manifest_row["shard_sha256"]:
            raise ValueError("STRUCTURAL_EVIDENCE_AB_COURT_PREPARED_SHARD_SHA_DRIFT:" + subject_id)
        teacher_components = ((manifest_row.get("metadata") or {}).get("topology_teacher") or {}).get("component_count")
        geometry = measure_surface_against_shard(
            subject_id=subject_id,
            stage12_path=stage,
            prepared_shard_path=shard,
            teacher_component_count=int(teacher_components) if teacher_components is not None else None,
            seed=26091991 + index,
        )
        mask = b_silhouette(
            subject_id=subject_id,
            candidate_path=stage,
            archive=archive,
            stage_row=stage_map[subject_id],
        )
        with np.load(stage, allow_pickle=False) as data:
            open_chains = open_boundary_chain_count(data["faces"])
        out[subject_id] = {
            "completed": True,
            "topology_court": topo,
            "geometry": geometry,
            "silhouette": mask,
            "open_boundary_chain_count": int(open_chains),
        }
    return out


def court(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = Path(args.preregistration).resolve()
    prereg = load_json(prereg_path); validate_preregistration(prereg)
    policy = prereg["independent_court"]["topology_surface_no_regression"]
    prepared_root = Path(args.prepared_root).resolve()
    prepared_manifest_path = Path(args.prepared_manifest).resolve(); prepared = load_json(prepared_manifest_path)
    preparation_receipt_path = Path(args.preparation_receipt).resolve(); preparation_receipt = load_json(preparation_receipt_path)
    stage01_path = Path(args.stage01_bundle).resolve()
    direct_root = Path(args.direct_surface_root).resolve()
    direct_geometry_path = Path(args.direct_geometry_court).resolve(); direct_geometry = load_json(direct_geometry_path)
    direct_silhouette_path = Path(args.direct_silhouette_court).resolve(); direct_silhouette = load_json(direct_silhouette_path)
    free_root = Path(args.free_running_root).resolve()
    free_receipt_path = free_root / "STRUCTURAL_EVIDENCE_AB_FREE_RUNNING_RECEIPT.json"
    free = load_json(free_receipt_path)

    if tuple(prepared.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_COURT_PREPARED_SUBJECT_ORDER_DRIFT")
    if tuple(direct_geometry.get("subject_order") or ()) != SUBJECT_IDS or tuple(direct_silhouette.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_COURT_DIRECT_AUTHORITY_ORDER_DRIFT")
    if tuple(free.get("arm_order") or ()) != ARMS or free.get("teacher_prefix_used") is not False:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_COURT_FREE_RUNNING_AUTHORITY_DRIFT")

    prepared_rows = {str(row["subject_id"]): row for row in prepared["subjects"]}
    direct_g = {str(row["subject_id"]): row for row in direct_geometry["subjects"]}
    direct_s = {str(row["subject_id"]): row for row in direct_silhouette["subjects"]}
    arm_receipts = {str(row["arm_id"]): row for row in free["arms"]}
    if set(arm_receipts) != set(ARMS):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_COURT_ARM_SET_DRIFT")
    stage_map = _legacy_subject_map(preparation_receipt)

    with zipfile.ZipFile(stage01_path, "r") as archive:
        arm_metrics = {
            arm: _arm_metrics(
                arm_id=arm,
                arm_receipt=arm_receipts[arm],
                free_root=free_root,
                prepared_root=prepared_root,
                prepared_rows=prepared_rows,
                stage_map=stage_map,
                archive=archive,
            )
            for arm in ARMS
        }

    blockers: list[str] = []
    subjects: list[dict[str, Any]] = []
    for subject_id in SUBJECT_IDS:
        a = arm_metrics[ARM_A][subject_id]; b = arm_metrics[ARM_B][subject_id]
        sb: list[str] = []
        if not a.get("completed") or not b.get("completed"):
            sb.append("ALL_SUBJECTS_COMPLETE_EOS_REQUIRED")
            subjects.append({"subject_id": subject_id, "blockers": sb})
            blockers.extend(f"{subject_id}:{x}" for x in sb)
            continue
        for arm_id, metric in ((ARM_A, a), (ARM_B, b)):
            topo = metric["topology_court"]
            if topo.get("grammar_valid") is not True or topo.get("finite_geometry") is not True or topo.get("within_scale_policy") is not True:
                sb.append(arm_id + ":FREE_RUNNING_TOPOLOGY_OR_SCALE_INVALID")

        dg = direct_g[subject_id]; ds = direct_s[subject_id]
        a_geometry = a["geometry"]; b_geometry = b["geometry"]
        direct_topo = dg["candidate_topology"]
        a_topo = a_geometry["candidate_topology"]; b_topo = b_geometry["candidate_topology"]
        direct_faces = int(direct_topo["face_count"]); b_faces = int(b_topo["face_count"])
        face_reduction = 1.0 - float(b_faces) / float(direct_faces) if direct_faces else float("-inf")
        if face_reduction < float(policy["minimum_face_reduction_fraction_vs_qualified_surface"]):
            sb.append("B_FACE_REDUCTION_VS_QUALIFIED_SURFACE")

        a_p95, b_p95 = bidirectional(a_geometry, "p95"), bidirectional(b_geometry, "p95")
        a_max, b_max = bidirectional(a_geometry, "max"), bidirectional(b_geometry, "max")
        if b_p95 > a_p95 * (1.0 + float(policy["maximum_relative_surface_p95_increase"])):
            sb.append("B_SURFACE_P95_REGRESSION_VS_A")
        if b_max > a_max * (1.0 + float(policy["maximum_relative_surface_max_increase"])):
            sb.append("B_SURFACE_MAX_REGRESSION_VS_A")
        if int(b_topo["nonmanifold_edge_count"]) - int(a_topo["nonmanifold_edge_count"]) > int(policy["maximum_new_nonmanifold_edges"]):
            sb.append("B_NEW_NONMANIFOLD_EDGES")
        if int(b_topo["degenerate_face_count"]) - int(a_topo["degenerate_face_count"]) > int(policy["maximum_new_zero_area_faces"]):
            sb.append("B_NEW_ZERO_AREA_FACES")
        if int(b["open_boundary_chain_count"]) - int(a["open_boundary_chain_count"]) > int(policy["maximum_new_open_boundary_chains"]):
            sb.append("B_NEW_OPEN_BOUNDARY_CHAINS")
        if policy["same_geometric_component_count_required"] and int(b_topo["component_count"]) != int(a_topo["component_count"]):
            sb.append("B_GEOMETRIC_COMPONENT_COUNT_CHANGED")

        a_views = {int(v["view_index"]): v for v in a["silhouette"]["views"]}
        b_views = {int(v["view_index"]): v for v in b["silhouette"]["views"]}
        max_iou_drop = max(float(a_views[i]["iou"]) - float(b_views[i]["iou"]) for i in sorted(a_views))
        if max_iou_drop > float(policy["maximum_mask_iou_drop"]):
            sb.append("B_SOURCE_MASK_IOU_REGRESSION_VS_A")

        subjects.append({
            "subject_id": subject_id,
            "direct_face_count": direct_faces,
            "a_face_count": int(a_topo["face_count"]),
            "b_face_count": b_faces,
            "b_face_reduction_fraction_vs_qualified_surface": face_reduction,
            "a_surface_p95": a_p95,
            "b_surface_p95": b_p95,
            "a_surface_max": a_max,
            "b_surface_max": b_max,
            "a_mean_mask_iou": float(a["silhouette"]["mean_iou"]),
            "b_mean_mask_iou": float(b["silhouette"]["mean_iou"]),
            "maximum_b_per_view_iou_drop_vs_a": max_iou_drop,
            "a_open_boundary_chain_count": int(a["open_boundary_chain_count"]),
            "b_open_boundary_chain_count": int(b["open_boundary_chain_count"]),
            "a_component_count": int(a_topo["component_count"]),
            "b_component_count": int(b_topo["component_count"]),
            "blockers": sb,
        })
        blockers.extend(f"{subject_id}:{x}" for x in sb)

    passed = not blockers
    result = {
        "schema": SCHEMA,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "preregistration_sha256": sha256_file(prereg_path),
        "prepared_manifest_sha256": sha256_file(prepared_manifest_path),
        "preparation_receipt_sha256": sha256_file(preparation_receipt_path),
        "stage01_bundle_sha256": sha256_file(stage01_path),
        "direct_geometry_court_sha256": sha256_file(direct_geometry_path),
        "direct_silhouette_court_sha256": sha256_file(direct_silhouette_path),
        "free_running_receipt_sha256": sha256_file(free_receipt_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": subjects,
        "blockers": blockers,
        "topology_surface_no_regression_pass": passed,
        "scientific_ab_pass_claimed": False,
        "presentation_primary_court_complete": False,
        "motion_primary_court_complete": False,
        "canonical_m_authorized": False,
        "next_action": "RUN_PRESENTATION_AND_MOTION_PRIMARY_COURTS" if passed else "AB_FAILS_LOCKED_TOPOLOGY_SURFACE_GATE",
    }
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preregistration", default="canonical/STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json")
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
    return 0 if result["topology_surface_no_regression_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
