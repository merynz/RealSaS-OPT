#!/usr/bin/env python3
"""Validate Truth139 K-hit V2 peel against complete CPU orthographic ray stacks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from models.iris.truth.graphics_oracle_v1 import camera_from_mapping, measure_orthographic_ray_stack_v1
from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256

SCHEMA = "RealSaS.Truth139GraphicsLayerPeelCourt.v2"
PARENT_SCHEMA = "RealSaS.Truth139GraphicsLayerPeelTeacher.v1"
PARENT_STATUS = "PASS_K_HIT_GEOMETRIC_LAYER_PEEL_CANDIDATE__CPU_PREFIX_COURT_REQUIRED__RGBA_COMPLETION_SEPARATE"
PARENT_REVISION = "V2_DENSE_GBUFFER_PARENT__FLOAT64_PIXEL_CENTER_BARY"
PASS_STATUS = "PASS_LAYER_PEEL_V2_GPU_FACE_SELECTION_MATCHES_COMPLETE_CPU_RAY_STACK"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _bound_path(ref: Mapping[str, object], code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _sample_rows(count: int, maximum: int) -> np.ndarray:
    if count <= 0 or maximum <= 0:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_SAMPLE_POLICY")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def validate(manifest: dict, *, maximum_samples_per_view: int = 128, minimum_validated_per_view: int = 16, barycentric_abs_tolerance: float = 7.5e-4, depth_relative_tolerance: float = 3.0e-5) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_PARENT_NOT_CANDIDATE")
    if manifest.get("producer_revision") != PARENT_REVISION:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_REVISION")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_SUBJECT")
    k = int(manifest.get("k", 0))
    if k < 2:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_K")

    parents = dict(manifest.get("parents") or {})
    geometry_path = _bound_path(dict(parents.get("geometry_npz") or {}), "TRUTH139_LAYER_PEEL_V2_COURT_GEOMETRY_DRIFT")
    camera_path = _bound_path(dict(parents.get("camera_bundle") or {}), "TRUTH139_LAYER_PEEL_V2_COURT_CAMERA_DRIFT")
    with np.load(geometry_path, allow_pickle=False) as z:
        vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_GEOMETRY_LAYOUT")
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_SCALE")
    tie_epsilon = float(manifest.get("tie_epsilon", 0.0))
    if not math.isfinite(tie_epsilon) or tie_epsilon <= 0:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_TIE_EPSILON")

    camera_payload = json.loads(camera_path.read_text())
    cameras = sorted(list(camera_payload.get("cameras") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(cameras) != 8 or [int(row.get("view_index", -1)) for row in cameras] != list(range(8)):
        raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_CAMERA_COUNT")

    view_meta = sorted(list(manifest.get("views") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(view_meta) != 8 or [int(row.get("view_index", -1)) for row in view_meta] != list(range(8)):
        raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_VIEW_COUNT")

    view_results = []
    total_validated = total_skipped_tie = total_saturated = total_proven_overflow = 0
    total_saturated_exact_or_filtered = 0
    max_bary_error = max_depth_error = 0.0

    for view_index, row in enumerate(view_meta):
        path = _bound_path({"path": row.get("path"), "sha256": row.get("sha256")}, "TRUTH139_LAYER_PEEL_V2_COURT_VIEW_DRIFT")
        with np.load(path, allow_pickle=False) as z:
            pixels = np.asarray(z["pixel_linear_index"], dtype=np.int64).reshape(-1)
            hit_count = np.asarray(z["hit_count_lower_bound"], dtype=np.int64).reshape(-1)
            raw_saturated = np.asarray(z["raw_stack_saturated"], dtype=np.bool_).reshape(-1)
            stack_complete = np.asarray(z["stack_complete"], dtype=np.bool_).reshape(-1)
            truncated = np.asarray(z["truncated_or_overflow_unknown"], dtype=np.bool_).reshape(-1)
            has_tie = np.asarray(z["has_depth_tie"], dtype=np.bool_).reshape(-1)
            valid = np.asarray(z["hit_valid"], dtype=np.bool_)
            triangle = np.asarray(z["hit_triangle_id"], dtype=np.int64)
            bary = np.asarray(z["hit_barycentric"], dtype=np.float64)
            depth = np.asarray(z["hit_depth"], dtype=np.float64)
        n = len(pixels)
        if not (hit_count.shape == raw_saturated.shape == stack_complete.shape == truncated.shape == has_tie.shape == (n,) and valid.shape == triangle.shape == depth.shape == (n, k) and bary.shape == (n, k, 3)):
            raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_ARRAY_LAYOUT")
        if np.any(stack_complete != ~raw_saturated) or np.any(truncated != raw_saturated):
            raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_COMPLETENESS_TYPING")
        if np.any(hit_count != np.sum(valid, axis=1)) or np.any(hit_count > k):
            raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_HIT_COUNT_DRIFT")
        if n < minimum_validated_per_view:
            raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_TOO_FEW_ROWS")

        picked = _sample_rows(n, int(maximum_samples_per_view))
        cam = camera_from_mapping(cameras[view_index])
        depth_tol = max(float(cam.half_extent) * float(depth_relative_tolerance), 1.0e-6)
        validated = skipped_tie = saturated = proven_overflow = saturated_exact = 0
        view_bary = view_depth = 0.0

        for row_index in picked:
            if bool(has_tie[row_index]):
                skipped_tie += 1
                continue
            pixel = int(pixels[row_index])
            py, px = divmod(pixel, cam.resolution)
            stack = measure_orthographic_ray_stack_v1(vertices, faces, cam, int(px), int(py), tie_epsilon=tie_epsilon)
            if not stack.complete:
                raise ValueError(f"TRUTH139_LAYER_PEEL_V2_COURT_CPU_STACK_INCOMPLETE:V{view_index}:P{pixel}")
            if stack.has_depth_tie:
                skipped_tie += 1
                continue
            cpu_count = len(stack.hits)
            stored_count = int(hit_count[row_index])
            if bool(stack_complete[row_index]):
                if cpu_count != stored_count:
                    raise ValueError(f"TRUTH139_LAYER_PEEL_V2_COURT_COMPLETE_COUNT_MISMATCH:V{view_index}:P{pixel}:{stored_count}!={cpu_count}")
            else:
                saturated += 1
                if cpu_count < stored_count:
                    raise ValueError(f"TRUTH139_LAYER_PEEL_V2_COURT_SATURATED_PREFIX_COUNT:V{view_index}:P{pixel}:{stored_count}>{cpu_count}")
                if cpu_count > stored_count:
                    proven_overflow += 1
                else:
                    saturated_exact += 1
            prefix = min(stored_count, cpu_count)
            for slot in range(prefix):
                if not bool(valid[row_index, slot]):
                    raise ValueError("TRUTH139_LAYER_PEEL_V2_COURT_PREFIX_VALIDITY_HOLE")
                hit = stack.hits[slot]
                if int(triangle[row_index, slot]) != int(hit.face_index):
                    raise ValueError(f"TRUTH139_LAYER_PEEL_V2_COURT_FACE_MISMATCH:V{view_index}:P{pixel}:S{slot}:gpu={int(triangle[row_index, slot])}:cpu={int(hit.face_index)}")
                be = float(np.max(np.abs(bary[row_index, slot] - np.asarray(hit.barycentric, dtype=np.float64))))
                de = abs(float(depth[row_index, slot]) - float(hit.depth))
                if be > barycentric_abs_tolerance:
                    raise ValueError(f"TRUTH139_LAYER_PEEL_V2_COURT_BARY_MISMATCH:V{view_index}:P{pixel}:S{slot}:{be}")
                if de > depth_tol:
                    raise ValueError(f"TRUTH139_LAYER_PEEL_V2_COURT_DEPTH_MISMATCH:V{view_index}:P{pixel}:S{slot}:{de}")
                view_bary = max(view_bary, be)
                view_depth = max(view_depth, de)
            validated += 1

        if validated < minimum_validated_per_view:
            raise ValueError(f"TRUTH139_LAYER_PEEL_V2_COURT_TOO_FEW_VALIDATED:V{view_index}:{validated}<{minimum_validated_per_view}")
        total_validated += validated
        total_skipped_tie += skipped_tie
        total_saturated += saturated
        total_proven_overflow += proven_overflow
        total_saturated_exact_or_filtered += saturated_exact
        max_bary_error = max(max_bary_error, view_bary)
        max_depth_error = max(max_depth_error, view_depth)
        view_results.append({"view_index": view_index, "sampled_row_count": int(len(picked)), "validated_row_count": int(validated), "depth_tie_skipped": int(skipped_tie), "raw_capacity_saturated_validated": int(saturated), "overflow_beyond_stored_prefix_proven": int(proven_overflow), "saturated_exact_or_filtered_no_overflow_proven": int(saturated_exact), "maximum_barycentric_abs_error": float(view_bary), "maximum_depth_abs_error": float(view_depth), "depth_abs_tolerance": float(depth_tol)})

    payload = {"schema": SCHEMA, "date": "2026-10-29", "status": PASS_STATUS, "subject_id": subject_id, "layer_peel_teacher_identity_sha256": str(manifest.get("layer_peel_teacher_identity_sha256") or ""), "producer_revision": str(manifest.get("producer_revision") or ""), "k": int(k), "policy": {"cpu_oracle": "COMPLETE_ORTHOGRAPHIC_RAY_STACK", "raw_gpu_capacity_saturation_preserves_unknown": True, "float64_pixel_center_surface_address_validated": True, "depth_ties_skipped_not_force_ordered": True, "material_rgb_owner_or_semantic_evidence_used": False, "maximum_samples_per_view": int(maximum_samples_per_view), "minimum_validated_per_view": int(minimum_validated_per_view)}, "view_courts": view_results, "total_validated_row_count": int(total_validated), "total_depth_tie_skipped": int(total_skipped_tie), "total_raw_capacity_saturated_validated": int(total_saturated), "total_overflow_beyond_stored_prefix_proven": int(total_proven_overflow), "total_saturated_exact_or_filtered_no_overflow_proven": int(total_saturated_exact_or_filtered), "maximum_barycentric_abs_error": float(max_bary_error), "maximum_depth_abs_error": float(max_depth_error), "training_performed": False, "qualification_minted": False}
    payload["court_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-peel-teacher-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-samples-per-view", type=int, default=128)
    parser.add_argument("--minimum-validated-per-view", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(args.layer_peel_teacher_manifest.resolve().read_text())
    result = validate(manifest, maximum_samples_per_view=args.maximum_samples_per_view, minimum_validated_per_view=args.minimum_validated_per_view)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"], "total_validated_row_count": result["total_validated_row_count"], "court_identity_sha256": result["court_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
