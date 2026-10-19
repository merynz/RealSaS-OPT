#!/usr/bin/env python3
from __future__ import annotations

"""Independent source-mask court for the TESSA-free direct IRIS surface baseline.

The court reuses the exact Stage01 admission raster semantics. Candidate vertices
are mapped from IRIS canonical coordinates back to source world with the already-
reviewed normalization inverse: ``world = center + half_extent * canonical``.
No PASS thresholds are encoded here; this is measurement authority only.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.corpus.iris_observation_admission_v1 import project_mask


SCHEMA = "RealSaS.Fit8DirectIRISSurfaceSilhouetteCourt.v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DIRECT_SURFACE_SILHOUETTE_JSON_OBJECT_REQUIRED")
    return value


def _mask_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | int | bool]:
    pred = np.asarray(predicted, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("DIRECT_SURFACE_SILHOUETTE_MASK_SHAPE_DRIFT")
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    union = tp + fp + fn
    pred_count = tp + fp
    truth_count = tp + fn
    iou = float(tp / union) if union else 1.0
    precision = float(tp / pred_count) if pred_count else 0.0
    recall = float(tp / truth_count) if truth_count else 0.0
    if pred_count and truth_count:
        gap = float(max(distance_transform_edt(~target)[pred].max(), distance_transform_edt(~pred)[target].max()))
    elif not pred_count and not truth_count:
        gap = 0.0
    else:
        gap = float(math.hypot(*pred.shape))
    return {
        "predicted_foreground_pixels": pred_count,
        "truth_foreground_pixels": truth_count,
        "intersection_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "symmetric_maximum_mask_gap_px": gap,
        "predicted_mask_empty": pred_count == 0,
        "truth_mask_empty": truth_count == 0,
    }


def _legacy_subject_map(preparation_receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stage01 = ((preparation_receipt.get("verified_inputs") or {}).get("stage01") or {})
    rows = list(stage01.get("subjects") or ())
    result = {str(row["subject_id"]): row for row in rows}
    if tuple(result) != SUBJECT_IDS:
        raise ValueError("DIRECT_SURFACE_SILHOUETTE_STAGE01_SUBJECT_ORDER_DRIFT")
    return result


def audit(
    *,
    binding_path: Path,
    preparation_receipt_path: Path,
    stage01_bundle_path: Path,
    direct_surface_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    binding = load_json(binding_path.resolve())
    prep = load_json(preparation_receipt_path.resolve())
    stage_spec = binding.get("stage01") or {}
    expected_stage_sha = str(stage_spec.get("sha256") or "").lower()
    actual_stage_sha = sha256_file(stage01_bundle_path.resolve())
    if actual_stage_sha != expected_stage_sha:
        raise ValueError("DIRECT_SURFACE_SILHOUETTE_STAGE01_SHA_DRIFT")
    verified_stage = ((prep.get("verified_inputs") or {}).get("stage01") or {})
    if verified_stage.get("stage01_bundle_sha256") != expected_stage_sha:
        raise ValueError("DIRECT_SURFACE_SILHOUETTE_PREPARATION_STAGE01_DRIFT")
    subject_map = _legacy_subject_map(prep)

    batch_path = direct_surface_root / "FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json"
    batch = load_json(batch_path)
    if batch.get("schema") != "RealSaS.Fit8DirectIRISSurfaceMaterialization.v1":
        raise ValueError("DIRECT_SURFACE_SILHOUETTE_BATCH_SCHEMA_INVALID")
    if tuple(batch.get("subject_order") or ()) != SUBJECT_IDS or batch.get("tessa_invoked") is not False:
        raise ValueError("DIRECT_SURFACE_SILHOUETTE_BATCH_AUTHORITY_DRIFT")
    batch_rows = {str(row["subject_id"]): row for row in batch.get("subjects") or ()}

    subject_results: list[dict[str, Any]] = []
    with zipfile.ZipFile(stage01_bundle_path, "r") as archive:
        for subject_id in SUBJECT_IDS:
            stage_row = subject_map[subject_id]
            legacy_id = str(stage_row["legacy_subject_id"])
            prefix = f"observations/{legacy_id}/"
            admission_bytes = archive.read(prefix + "IRIS_OBSERVATION_ADMISSION.json")
            camera_bytes = archive.read(prefix + "camera_bundle.json")
            if sha256_bytes(admission_bytes) != stage_row["observation_admission"]:
                raise ValueError("DIRECT_SURFACE_SILHOUETTE_ADMISSION_SHA_DRIFT:" + subject_id)
            if sha256_bytes(camera_bytes) != stage_row["camera_bundle"]:
                raise ValueError("DIRECT_SURFACE_SILHOUETTE_CAMERA_SHA_DRIFT:" + subject_id)
            admission = json.loads(admission_bytes)
            cameras = json.loads(camera_bytes)
            if admission.get("status") != "PASS_IRIS_BIND_OBSERVATIONS" or admission.get("subject_id") != legacy_id:
                raise ValueError("DIRECT_SURFACE_SILHOUETTE_ADMISSION_STATUS_DRIFT:" + subject_id)
            if cameras.get("per_view_pose_change") or cameras.get("per_view_recenter"):
                raise ValueError("DIRECT_SURFACE_SILHOUETTE_CAMERA_MUTATION_DRIFT:" + subject_id)

            subject_root = direct_surface_root / "subjects" / subject_id
            receipt_path = subject_root / "DIRECT_IRIS_SURFACE_RECEIPT.json"
            candidate_path = subject_root / "DIRECT_IRIS_SURFACE.stage12.npz"
            receipt = load_json(receipt_path)
            if receipt.get("subject_id") != subject_id or receipt.get("compiler_qualified") is not False:
                raise ValueError("DIRECT_SURFACE_SILHOUETTE_RECEIPT_SCOPE_DRIFT:" + subject_id)
            if sha256_file(candidate_path) != receipt.get("stage12_npz_sha256"):
                raise ValueError("DIRECT_SURFACE_SILHOUETTE_CANDIDATE_SHA_DRIFT:" + subject_id)
            if batch_rows.get(subject_id, {}).get("stage12_npz_sha256") != receipt.get("stage12_npz_sha256"):
                raise ValueError("DIRECT_SURFACE_SILHOUETTE_BATCH_CANDIDATE_DRIFT:" + subject_id)

            with np.load(candidate_path, allow_pickle=False) as data:
                vertices = np.asarray(data["vertices_normalized"], dtype=np.float64)
                faces = np.asarray(data["faces"], dtype=np.int64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError("DIRECT_SURFACE_SILHOUETTE_CANDIDATE_SHAPE_INVALID:" + subject_id)
            center = np.asarray(cameras["common_center_xyz"], dtype=np.float64)
            half = float(cameras["common_half_extent"])
            vertices_world = center[None, :] + half * vertices

            views: list[dict[str, Any]] = []
            admission_views = {int(row["view_index"]): row for row in admission.get("views") or ()}
            for camera in cameras.get("cameras") or ():
                view_index = int(camera["view_index"])
                resolution = int(camera["resolution"])
                mask_name = prefix + f"V{view_index}.mask.bin"
                mask_bytes = archive.read(mask_name)
                admitted = admission_views[view_index]
                if sha256_bytes(mask_bytes) != admitted["mask_sha256"]:
                    raise ValueError(f"DIRECT_SURFACE_SILHOUETTE_MASK_SHA_DRIFT:{subject_id}:V{view_index}")
                truth = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(resolution, resolution) > 0
                predicted = project_mask(vertices_world, faces, camera)
                views.append({"view_index": view_index, **_mask_metrics(predicted, truth)})
            if len(views) != 8:
                raise ValueError("DIRECT_SURFACE_SILHOUETTE_VIEW_COUNT_DRIFT:" + subject_id)
            subject_results.append(
                {
                    "subject_id": subject_id,
                    "legacy_subject_id": legacy_id,
                    "vertex_count": int(len(vertices)),
                    "face_count": int(len(faces)),
                    "mean_iou": float(np.mean([v["iou"] for v in views])),
                    "minimum_iou": float(min(v["iou"] for v in views)),
                    "mean_recall": float(np.mean([v["recall"] for v in views])),
                    "maximum_mask_gap_px": float(max(v["symmetric_maximum_mask_gap_px"] for v in views)),
                    "views": views,
                }
            )

    result = {
        "schema": SCHEMA,
        "status": "MEASURED__NO_PASS_THRESHOLD_PREREGISTERED",
        "stage01_bundle_sha256": actual_stage_sha,
        "direct_surface_batch_receipt_sha256": sha256_file(batch_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": subject_results,
        "global": {
            "mean_subject_iou": float(np.mean([row["mean_iou"] for row in subject_results])),
            "minimum_subject_iou": float(min(row["mean_iou"] for row in subject_results)),
            "minimum_view_iou": float(min(v["iou"] for row in subject_results for v in row["views"])),
            "maximum_mask_gap_px": float(max(row["maximum_mask_gap_px"] for row in subject_results)),
        },
        "raster_authority": "EXACT_STAGE01_TRIANGLE_UNIT_PIXEL_SQUARE_INTERSECTION",
        "coordinate_inverse": "WORLD_EQUALS_COMMON_CENTER_PLUS_COMMON_HALF_EXTENT_TIMES_IRIS_CANONICAL",
        "teacher_topology_used": False,
        "tessa_invoked": False,
        "compiler_qualified": False,
        "training_authorized": False,
        "scientific_pass_claimed": False,
        "next_action": "COMBINE_WITH_DIRECT_SURFACE_GEOMETRY_COURT_THEN_PREREGISTER_A_ARM_THRESHOLDS",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", default="canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json")
    p.add_argument("--preparation-receipt", required=True)
    p.add_argument("--stage01-bundle", required=True)
    p.add_argument("--direct-surface-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = audit(
        binding_path=Path(args.binding),
        preparation_receipt_path=Path(args.preparation_receipt),
        stage01_bundle_path=Path(args.stage01_bundle),
        direct_surface_root=Path(args.direct_surface_root),
        output_path=Path(args.output),
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
