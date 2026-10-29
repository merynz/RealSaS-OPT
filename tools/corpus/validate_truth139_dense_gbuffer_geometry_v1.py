#!/usr/bin/env python3
"""Independently validate Truth139 dense appearance raster addresses against the CPU ray oracle.

The dense G-buffer is high-volume teacher authority.  A GPU rasterizer bug or camera-convention
mistake could therefore poison millions of labels while still producing plausible masks.  This
court samples deterministic interior pixels from every view and re-measures the exact orthographic
ray against the source rest mesh using the independent NumPy Graphics oracle.

No material, colour, semantic part, owner or source image participates in the geometric verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from models.iris.truth.graphics_oracle_v1 import camera_from_mapping, measure_orthographic_ray_stack_v1
from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256

SCHEMA = "RealSaS.Truth139DenseAppearanceGeometryCourt.v1"
GBUFFER_SCHEMA = "RealSaS.Truth139DenseAppearanceGBuffer.v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _bound_path(ref: dict, code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _sample_rows(subject_id: str, view_index: int, pixels: np.ndarray, eligible: np.ndarray, maximum: int) -> np.ndarray:
    ids = np.flatnonzero(eligible)
    if not len(ids):
        return ids
    ranked = sorted(
        ids.tolist(),
        key=lambda row: hashlib.sha256(
            f"{subject_id}|{view_index}|{int(pixels[row])}".encode()
        ).hexdigest(),
    )
    return np.asarray(ranked[:maximum], dtype=np.int64)


def validate(
    manifest: dict,
    *,
    maximum_samples_per_view: int = 64,
    minimum_validated_per_view: int = 16,
    interior_barycentric_floor: float = 1.0e-4,
    barycentric_abs_tolerance: float = 5.0e-4,
    depth_relative_tolerance: float = 2.0e-5,
) -> dict:
    if manifest.get("schema") != GBUFFER_SCHEMA:
        raise ValueError("TRUTH139_DENSE_GBUFFER_COURT_SCHEMA")
    if manifest.get("status") != "PASS_DENSE_SURFACE_ADDRESSED_APPEARANCE_GBUFFER__STAGE48_INTERVENTION_WITNESS_STILL_REQUIRED":
        raise ValueError("TRUTH139_DENSE_GBUFFER_COURT_PARENT_NOT_PASS")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_DENSE_GBUFFER_COURT_SUBJECT")
    if maximum_samples_per_view < 1 or minimum_validated_per_view < 1:
        raise ValueError("TRUTH139_DENSE_GBUFFER_COURT_SAMPLE_POLICY")
    if not 0.0 < interior_barycentric_floor < 1.0 / 3.0:
        raise ValueError("TRUTH139_DENSE_GBUFFER_COURT_INTERIOR_POLICY")

    geometry_path = _bound_path(dict(manifest.get("geometry_npz") or {}), "TRUTH139_DENSE_GBUFFER_COURT_GEOMETRY_DRIFT")
    camera_path = _bound_path(dict(manifest.get("camera_bundle") or {}), "TRUTH139_DENSE_GBUFFER_COURT_CAMERA_DRIFT")
    with np.load(geometry_path, allow_pickle=False) as z:
        vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("TRUTH139_DENSE_GBUFFER_COURT_GEOMETRY_LAYOUT")
    cameras_payload = json.loads(camera_path.read_text())
    cameras = sorted(list(cameras_payload.get("cameras") or ()), key=lambda row: int(row["view_index"]))
    if len(cameras) != 8:
        raise ValueError("TRUTH139_DENSE_GBUFFER_COURT_CAMERA_COUNT")

    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("TRUTH139_DENSE_GBUFFER_COURT_GEOMETRY_SCALE")
    tie_epsilon = max(scale * 1.0e-7, 1.0e-9)

    rows = []
    total_validated = 0
    maximum_barycentric_error = 0.0
    maximum_depth_error = 0.0
    maximum_surface_error = 0.0
    for view in list(manifest.get("views") or ()):
        view_index = int(view["view_index"])
        camera_row = cameras[view_index]
        camera = camera_from_mapping(camera_row)
        gbuffer = _bound_path(
            {"path": view["gbuffer_path"], "sha256": view["gbuffer_sha256"]},
            "TRUTH139_DENSE_GBUFFER_COURT_VIEW_DRIFT",
        )
        with np.load(gbuffer, allow_pickle=False) as z:
            pixels = np.asarray(z["pixel_linear_index"], dtype=np.int64).reshape(-1)
            triangle = np.asarray(z["triangle_id"], dtype=np.int64).reshape(-1)
            uv = np.asarray(z["barycentric_uv"], dtype=np.float64)
            depth = np.asarray(z["depth_source_camera"], dtype=np.float64).reshape(-1)
        if uv.shape != (len(pixels), 2) or triangle.shape != pixels.shape or depth.shape != pixels.shape:
            raise ValueError("TRUTH139_DENSE_GBUFFER_COURT_VIEW_LAYOUT")
        weights = np.stack((uv[:, 0], uv[:, 1], 1.0 - uv[:, 0] - uv[:, 1]), axis=1)
        eligible = np.all(weights > float(interior_barycentric_floor), axis=1) & np.isfinite(depth)
        picked = _sample_rows(subject_id, view_index, pixels, eligible, int(maximum_samples_per_view))
        if len(picked) < min(int(minimum_validated_per_view), int(np.count_nonzero(eligible))):
            raise ValueError("TRUTH139_DENSE_GBUFFER_COURT_SAMPLE_SHORTFALL")

        validated = 0
        tie_skipped = 0
        no_hit = 0
        view_bary_max = 0.0
        view_depth_max = 0.0
        view_surface_max = 0.0
        depth_tolerance = max(1.0e-6, float(camera.half_extent) * float(depth_relative_tolerance))
        for row_index in picked:
            pixel = int(pixels[row_index])
            pixel_y, pixel_x = divmod(pixel, camera.resolution)
            stack = measure_orthographic_ray_stack_v1(
                vertices,
                faces,
                camera,
                int(pixel_x),
                int(pixel_y),
                tie_epsilon=tie_epsilon,
            )
            if not stack.complete:
                raise ValueError(f"TRUTH139_DENSE_GBUFFER_COURT_CAMERA_NOT_BEFORE_MESH:V{view_index}")
            if stack.has_depth_tie:
                tie_skipped += 1
                continue
            if not stack.hits:
                no_hit += 1
                continue
            hit = stack.hits[0]
            if int(hit.face_index) != int(triangle[row_index]):
                raise ValueError(
                    f"TRUTH139_DENSE_GBUFFER_COURT_FACE_MISMATCH:V{view_index}:P{pixel}:"
                    f"gpu={int(triangle[row_index])}:cpu={int(hit.face_index)}"
                )
            raster_weight = weights[row_index]
            oracle_weight = np.asarray(hit.barycentric, dtype=np.float64)
            bary_error = float(np.max(np.abs(raster_weight - oracle_weight)))
            if bary_error > float(barycentric_abs_tolerance):
                raise ValueError(
                    f"TRUTH139_DENSE_GBUFFER_COURT_BARYCENTRIC_MISMATCH:V{view_index}:P{pixel}:{bary_error}"
                )
            depth_error = abs(float(depth[row_index]) - float(hit.depth))
            if depth_error > depth_tolerance:
                raise ValueError(
                    f"TRUTH139_DENSE_GBUFFER_COURT_DEPTH_MISMATCH:V{view_index}:P{pixel}:{depth_error}"
                )
            face = faces[int(hit.face_index)]
            p_raster = np.sum(vertices[face] * raster_weight[:, None], axis=0)
            p_oracle = np.sum(vertices[face] * oracle_weight[:, None], axis=0)
            surface_error = float(np.linalg.norm(p_raster - p_oracle))
            if surface_error > max(scale * 1.0e-5, 1.0e-7):
                raise ValueError(
                    f"TRUTH139_DENSE_GBUFFER_COURT_SURFACE_MISMATCH:V{view_index}:P{pixel}:{surface_error}"
                )
            validated += 1
            view_bary_max = max(view_bary_max, bary_error)
            view_depth_max = max(view_depth_max, depth_error)
            view_surface_max = max(view_surface_max, surface_error)

        required = min(int(minimum_validated_per_view), len(picked))
        if validated < required:
            raise ValueError(
                f"TRUTH139_DENSE_GBUFFER_COURT_TOO_FEW_INDEPENDENT_WITNESSES:V{view_index}:{validated}<{required}"
            )
        total_validated += validated
        maximum_barycentric_error = max(maximum_barycentric_error, view_bary_max)
        maximum_depth_error = max(maximum_depth_error, view_depth_max)
        maximum_surface_error = max(maximum_surface_error, view_surface_max)
        rows.append(
            {
                "view_index": view_index,
                "eligible_interior_pixel_count": int(np.count_nonzero(eligible)),
                "sampled_pixel_count": int(len(picked)),
                "validated_pixel_count": int(validated),
                "depth_tie_skipped": int(tie_skipped),
                "no_hit_count": int(no_hit),
                "maximum_barycentric_abs_error": view_bary_max,
                "maximum_depth_abs_error": view_depth_max,
                "maximum_surface_l2_error": view_surface_max,
                "depth_abs_tolerance": depth_tolerance,
            }
        )

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_DENSE_GPU_GBUFFER",
        "subject_id": subject_id,
        "dense_gbuffer_authority_identity_sha256": str(manifest.get("authority_identity_sha256") or ""),
        "policy": {
            "selection": "DETERMINISTIC_HASHED_INTERIOR_PIXELS_PER_VIEW",
            "maximum_samples_per_view": int(maximum_samples_per_view),
            "minimum_validated_per_view": int(minimum_validated_per_view),
            "interior_barycentric_floor": float(interior_barycentric_floor),
            "barycentric_abs_tolerance": float(barycentric_abs_tolerance),
            "depth_relative_tolerance": float(depth_relative_tolerance),
            "tie_epsilon": tie_epsilon,
            "material_or_semantic_evidence_used": False,
        },
        "view_courts": rows,
        "total_validated_pixel_count": int(total_validated),
        "maximum_barycentric_abs_error": maximum_barycentric_error,
        "maximum_depth_abs_error": maximum_depth_error,
        "maximum_surface_l2_error": maximum_surface_error,
        "training_performed": False,
        "qualification_minted": False,
    }
    payload["court_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-gbuffer-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-samples-per-view", type=int, default=64)
    parser.add_argument("--minimum-validated-per-view", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(parser.parse_args().dense_gbuffer_manifest.resolve().read_text())
    # Reparse only once for ordinary CLI callers while keeping argparse's type validation above.
    args = parser.parse_args()
    result = validate(
        manifest,
        maximum_samples_per_view=args.maximum_samples_per_view,
        minimum_validated_per_view=args.minimum_validated_per_view,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"], "total_validated_pixel_count": result["total_validated_pixel_count"], "court_identity_sha256": result["court_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
