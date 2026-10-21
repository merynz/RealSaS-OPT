#!/usr/bin/env python3
from __future__ import annotations

"""Authorize optimizer step 1 only after all preregistered A-arm gates pass."""

import argparse
import json
from pathlib import Path
from typing import Any

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.tessa.v2.experiment_v2 import load_preregistration_v1, sha256_file

SCHEMA = "RealSaS.TESSAV2PretrainingAdmission.v1"
PASS_STATUS = "PASS__OPTIMIZER_STEP_1_AUTHORIZED_FOR_LOCKED_TESSA_V2_FIT"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("TESSA_V2_ADMISSION_JSON_OBJECT_REQUIRED")
    return value


def _bidirectional_p95(row: dict[str, Any]) -> float:
    distance = row["surface_distance"]
    return max(
        float(distance["candidate_to_teacher"]["p95"]),
        float(distance["teacher_to_candidate"]["p95"]),
    )


def _bidirectional_max(row: dict[str, Any]) -> float:
    distance = row["surface_distance"]
    return max(
        float(distance["candidate_to_teacher"]["max"]),
        float(distance["teacher_to_candidate"]["max"]),
    )


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = Path(args.preregistration).resolve()
    prereg = load_preregistration_v1(prereg_path)
    geometry_path = Path(args.geometry_court).resolve()
    silhouette_path = Path(args.silhouette_court).resolve()
    corpus_manifest_path = Path(args.corpus_manifest).resolve()
    preflight_path = Path(args.tessa_preflight).resolve()
    direct_batch_path = Path(args.direct_surface_batch).resolve()

    geometry = load_json(geometry_path)
    silhouette = load_json(silhouette_path)
    corpus = load_json(corpus_manifest_path)
    preflight = load_json(preflight_path)
    direct_batch = load_json(direct_batch_path)

    if geometry.get("schema") != "RealSaS.Fit8DirectIRISSurfaceGeometryCourt.v1":
        raise ValueError("TESSA_V2_ADMISSION_GEOMETRY_SCHEMA_INVALID")
    if silhouette.get("schema") != "RealSaS.Fit8DirectIRISSurfaceSilhouetteCourt.v1":
        raise ValueError("TESSA_V2_ADMISSION_SILHOUETTE_SCHEMA_INVALID")
    if corpus.get("schema") != "RealSaS.Fit8TESSAV2TrainingCorpus.v1":
        raise ValueError("TESSA_V2_ADMISSION_CORPUS_SCHEMA_INVALID")
    if preflight.get("schema") != "RealSaS.TESSAV2A100TrainingPreflight.v1":
        raise ValueError("TESSA_V2_ADMISSION_PREFLIGHT_SCHEMA_INVALID")
    if direct_batch.get("schema") != "RealSaS.Fit8DirectIRISSurfaceMaterialization.v1":
        raise ValueError("TESSA_V2_ADMISSION_DIRECT_BATCH_SCHEMA_INVALID")

    if tuple(geometry.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("TESSA_V2_ADMISSION_GEOMETRY_SUBJECT_ORDER_DRIFT")
    if tuple(silhouette.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("TESSA_V2_ADMISSION_SILHOUETTE_SUBJECT_ORDER_DRIFT")
    if tuple(corpus.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("TESSA_V2_ADMISSION_CORPUS_SUBJECT_ORDER_DRIFT")
    if tuple(direct_batch.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("TESSA_V2_ADMISSION_DIRECT_SUBJECT_ORDER_DRIFT")

    if preflight.get("status") != "PASS__ENGINEERING_ONLY__ZERO_OPTIMIZER_STEPS":
        raise ValueError("TESSA_V2_ADMISSION_ENGINEERING_PREFLIGHT_NOT_PASS")
    if preflight.get("optimizer_steps_executed") != 0 or preflight.get("parameters_updated") is not False:
        raise ValueError("TESSA_V2_ADMISSION_PREFLIGHT_MUTATED_PARAMETERS")
    if preflight.get("preregistration_sha256") != sha256_file(prereg_path):
        raise ValueError("TESSA_V2_ADMISSION_PREREGISTRATION_SHA_DRIFT")
    if preflight.get("corpus_manifest_sha256") != sha256_file(corpus_manifest_path):
        raise ValueError("TESSA_V2_ADMISSION_CORPUS_PREFLIGHT_SHA_DRIFT")
    if corpus.get("direct_surface_batch_sha256") != sha256_file(direct_batch_path):
        raise ValueError("TESSA_V2_ADMISSION_CORPUS_DIRECT_BATCH_DRIFT")
    if geometry.get("surface_batch_receipt_sha256") != sha256_file(direct_batch_path):
        raise ValueError("TESSA_V2_ADMISSION_GEOMETRY_DIRECT_BATCH_DRIFT")
    if silhouette.get("direct_surface_batch_receipt_sha256") != sha256_file(direct_batch_path):
        raise ValueError("TESSA_V2_ADMISSION_SILHOUETTE_DIRECT_BATCH_DRIFT")

    policy = prereg["direct_surface_admission"]
    blockers: list[str] = []
    geometry_rows = {str(row["subject_id"]): row for row in geometry["subjects"]}
    silhouette_rows = {str(row["subject_id"]): row for row in silhouette["subjects"]}
    corpus_rows = {str(row["subject_id"]): row for row in corpus["subjects"]}
    if tuple(geometry_rows) != SUBJECT_IDS or tuple(silhouette_rows) != SUBJECT_IDS or tuple(corpus_rows) != SUBJECT_IDS:
        raise ValueError("TESSA_V2_ADMISSION_SUBJECT_MAP_DRIFT")

    per_subject: list[dict[str, Any]] = []
    for subject_id in SUBJECT_IDS:
        g = geometry_rows[subject_id]
        s = silhouette_rows[subject_id]
        topo = g["candidate_topology"]
        p95 = _bidirectional_p95(g)
        dmax = _bidirectional_max(g)
        subject_blockers: list[str] = []
        if topo.get("finite_vertices") is not True:
            subject_blockers.append("NONFINITE_GEOMETRY")
        if int(topo.get("nonmanifold_edge_count", -1)) > int(policy["maximum_nonmanifold_edges_per_subject"]):
            subject_blockers.append("NONMANIFOLD_EDGES")
        if int(topo.get("degenerate_face_count", -1)) > int(policy["maximum_zero_area_faces_per_subject"]):
            subject_blockers.append("ZERO_AREA_FACES")
        if p95 > float(policy["maximum_surface_p95_distance_iris"]):
            subject_blockers.append("SURFACE_P95_DISTANCE")
        if dmax > float(policy["maximum_surface_max_distance_iris"]):
            subject_blockers.append("SURFACE_MAX_DISTANCE")
        if float(s["mean_iou"]) < float(policy["minimum_subject_mean_mask_iou"]):
            subject_blockers.append("SUBJECT_MEAN_MASK_IOU")
        if float(s["minimum_iou"]) < float(policy["minimum_view_mask_iou"]):
            subject_blockers.append("VIEW_MASK_IOU")
        if float(s["maximum_mask_gap_px"]) > float(policy["maximum_global_mask_gap_px"]):
            subject_blockers.append("MASK_GAP")
        blockers.extend(f"{subject_id}:{code}" for code in subject_blockers)
        per_subject.append({
            "subject_id": subject_id,
            "surface_p95_distance_iris": p95,
            "surface_max_distance_iris": dmax,
            "mean_mask_iou": float(s["mean_iou"]),
            "minimum_view_mask_iou": float(s["minimum_iou"]),
            "maximum_mask_gap_px": float(s["maximum_mask_gap_px"]),
            "nonmanifold_edge_count": int(topo["nonmanifold_edge_count"]),
            "degenerate_face_count": int(topo["degenerate_face_count"]),
            "component_count": int(topo["component_count"]),
            "training_window_count": int(corpus_rows[subject_id]["training_window_count"]),
            "blockers": subject_blockers,
        })

    global_mean = float(silhouette["global"]["mean_subject_iou"])
    global_gap = float(silhouette["global"]["maximum_mask_gap_px"])
    if global_mean < float(policy["minimum_global_mean_subject_mask_iou"]):
        blockers.append("GLOBAL:MEAN_SUBJECT_MASK_IOU")
    if global_gap > float(policy["maximum_global_mask_gap_px"]):
        blockers.append("GLOBAL:MASK_GAP")

    status = PASS_STATUS if not blockers else "BLOCKED__PRETRAINING_ADMISSION_FAILED"
    result = {
        "schema": SCHEMA,
        "status": status,
        "preregistration_sha256": sha256_file(prereg_path),
        "direct_surface_batch_sha256": sha256_file(direct_batch_path),
        "geometry_court_sha256": sha256_file(geometry_path),
        "silhouette_court_sha256": sha256_file(silhouette_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "tessa_a100_preflight_sha256": sha256_file(preflight_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": per_subject,
        "global_mean_subject_mask_iou": global_mean,
        "global_maximum_mask_gap_px": global_gap,
        "blockers": blockers,
        "optimizer_step_1_authorized": not blockers,
        "training_authorized": not blockers,
        "scientific_pass_claimed": False,
        "tessa_mandatory": False,
        "canonical_m_authorized": False,
        "next_action": "RUN_LOCKED_TESSA_V2_TRAINER" if not blockers else "DO_NOT_TRAIN__INSPECT_A_ARM",
    }
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preregistration", default="canonical/TESSA_V2_FIT8_PREREGISTRATION_V1.json")
    p.add_argument("--direct-surface-batch", required=True)
    p.add_argument("--geometry-court", required=True)
    p.add_argument("--silhouette-court", required=True)
    p.add_argument("--corpus-manifest", required=True)
    p.add_argument("--tessa-preflight", required=True)
    p.add_argument("--output", required=True)
    result = qualify(p.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
