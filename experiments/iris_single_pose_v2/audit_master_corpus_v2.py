from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from geometry import geometric_vertex_normals

VIEWS = 8
STYLES = ("cel_clean", "ink_cel")
SEALED = {"CAL", "DEV", "EXTERNAL_HOLDOUT"}


def atomic_json(path: str | Path, obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _percentiles(values):
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "min": None, "p01": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(len(a)),
        "min": float(np.min(a)),
        "p01": float(np.percentile(a, 1)),
        "p05": float(np.percentile(a, 5)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def _load_selection(root: Path):
    path = root / "metadata" / "CANONICAL_VARIANT_SELECTION.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = json.load(open(path, encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("CANONICAL_VARIANT_SELECTION.json must be an asset-id mapping")
    return path, obj


def _face_degenerate_fraction(vertices, faces, eps=1e-12):
    if len(faces) == 0:
        return 1.0
    tri = vertices[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return float(np.mean(area2 <= eps))


def _bbox_fraction(pixel_linear_index, resolution):
    pix = np.asarray(pixel_linear_index, np.int64)
    if not len(pix):
        return 0.0
    y = pix // resolution
    x = pix % resolution
    area = (int(x.max()) - int(x.min()) + 1) * (int(y.max()) - int(y.min()) + 1)
    return float(area / float(resolution * resolution))


def _check_alpha(path: Path, expected_pix: np.ndarray, resolution: int):
    with Image.open(path) as im:
        if im.mode != "RGBA" or im.size != (resolution, resolution):
            return False, f"bad_image_mode_or_size:{im.mode}:{im.size}", None
        alpha = np.asarray(im.getchannel("A"), np.uint8)
    got = np.flatnonzero(alpha.reshape(-1) > 0).astype(np.int64)
    exact = len(got) == len(expected_pix) and np.array_equal(got, expected_pix)
    return exact, None if exact else "alpha_authority_pixel_set_mismatch", int(len(got))


def inspect_asset(root: Path, asset_id: str, selection: dict, verify_alpha=True):
    record = selection[asset_id]
    split = str(record.get("split", "UNKNOWN")).upper()
    provider = str(record.get("source_registry_id", "UNKNOWN"))
    ar = root / "master" / "assets" / asset_id
    issues = []
    warnings = []
    coverage = []
    bbox_fraction = []
    style_alpha_counts = {s: [] for s in STYLES}
    geometry = {}

    if not ar.is_dir():
        return {
            "asset_id": asset_id,
            "split": split,
            "source_registry_id": provider,
            "fatal": ["missing_asset_directory"],
            "warnings": [],
        }
    if not (ar / "RENDER_COMPLETE.json").is_file():
        issues.append("missing_RENDER_COMPLETE")

    gp = ar / "primary_geometry.npz"
    if not gp.is_file():
        issues.append("missing_primary_geometry")
        vertices = np.zeros((0, 3), np.float32)
        faces = np.zeros((0, 3), np.int64)
    else:
        try:
            with np.load(gp, allow_pickle=False) as z:
                if "vertices" not in z.files or "faces" not in z.files:
                    raise RuntimeError(f"missing vertices/faces fields: {z.files}")
                vertices = np.asarray(z["vertices"], np.float32)
                faces = np.asarray(z["faces"], np.int64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                issues.append("invalid_vertices_shape_or_empty")
            if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
                issues.append("invalid_faces_shape_or_empty")
            if not np.isfinite(vertices).all():
                issues.append("nonfinite_vertices")
            if len(faces) and (faces.min(initial=0) < 0 or faces.max(initial=-1) >= len(vertices)):
                issues.append("face_index_out_of_bounds")
            if len(vertices) and len(faces) and not issues:
                deg = _face_degenerate_fraction(vertices, faces)
                vn = geometric_vertex_normals(vertices, faces)
                nlen = np.linalg.norm(vn, axis=1)
                zero_n = float(np.mean(nlen <= 1e-6))
                geometry = {
                    "vertices": int(len(vertices)),
                    "faces": int(len(faces)),
                    "degenerate_face_fraction": deg,
                    "zero_recomputed_vertex_normal_fraction": zero_n,
                    "p_max_abs": float(np.max(np.abs(vertices))),
                    "p_min": vertices.min(axis=0).astype(float).tolist(),
                    "p_max": vertices.max(axis=0).astype(float).tolist(),
                }
                if deg > 0:
                    warnings.append("has_degenerate_faces")
                if zero_n > 0:
                    warnings.append("has_zero_recomputed_vertex_normals")
        except Exception as e:
            issues.append(f"geometry_load_or_validation_error:{type(e).__name__}:{e}")
            vertices = np.zeros((0, 3), np.float32)
            faces = np.zeros((0, 3), np.int64)

    expected_yaws = np.arange(VIEWS, dtype=np.float32) * 45.0
    seen_yaws = []
    for v in range(VIEWS):
        vd = ar / "renders" / f"V{v}"
        if not vd.is_dir():
            issues.append(f"V{v}:missing_view_directory")
            continue
        cp = vd / "camera.json"
        rp = vd / "raster_authority.npz"
        if not cp.is_file():
            issues.append(f"V{v}:missing_camera")
            continue
        if not rp.is_file():
            issues.append(f"V{v}:missing_raster_authority")
            continue
        try:
            cam = json.load(open(cp, encoding="utf-8"))
            yaw = float(cam["yaw_deg"])
            seen_yaws.append(yaw)
            for key in ("right", "up", "half_extent"):
                if key not in cam:
                    issues.append(f"V{v}:camera_missing_{key}")
            if not math.isfinite(float(cam.get("half_extent", float("nan")))) or float(cam.get("half_extent", 0.0)) <= 0:
                issues.append(f"V{v}:bad_half_extent")
        except Exception as e:
            issues.append(f"V{v}:camera_error:{type(e).__name__}:{e}")
            continue

        try:
            with np.load(rp, allow_pickle=False) as ra:
                required = {"resolution", "pixel_linear_index", "triangle_id", "barycentric_uv"}
                missing = required - set(ra.files)
                if missing:
                    raise RuntimeError(f"missing fields {sorted(missing)}")
                res = int(np.asarray(ra["resolution"]).reshape(-1)[0])
                pix = np.asarray(ra["pixel_linear_index"], np.int64)
                tri = np.asarray(ra["triangle_id"], np.int64)
                uv = np.asarray(ra["barycentric_uv"], np.float32)
            if res != 1024:
                issues.append(f"V{v}:authority_resolution_{res}")
            if len(pix) == 0:
                issues.append(f"V{v}:blank_raster")
            if len(pix) != len(tri) or uv.shape != (len(pix), 2):
                issues.append(f"V{v}:raster_row_shape_mismatch")
            if len(pix) and (pix.min() < 0 or pix.max() >= res * res):
                issues.append(f"V{v}:pixel_index_out_of_bounds")
            if len(pix) and (np.any(pix[1:] <= pix[:-1])):
                issues.append(f"V{v}:pixel_index_not_strictly_sorted_unique")
            if len(tri) and (tri.min() < 0 or tri.max() >= len(faces)):
                issues.append(f"V{v}:triangle_id_out_of_bounds")
            if not np.isfinite(uv).all():
                issues.append(f"V{v}:nonfinite_barycentric")
            if len(uv):
                w2 = 1.0 - uv[:, 0] - uv[:, 1]
                if min(float(uv.min()), float(w2.min())) < -1e-5:
                    issues.append(f"V{v}:barycentric_outside_tolerance")
            coverage.append(int(len(pix)))
            bbox_fraction.append(_bbox_fraction(pix, res))

            for style in STYLES:
                sp = vd / f"{style}.png"
                if not sp.is_file():
                    issues.append(f"V{v}:missing_{style}.png")
                    continue
                if verify_alpha:
                    exact, err, count = _check_alpha(sp, pix, res)
                    if err:
                        issues.append(f"V{v}:{style}:{err}")
                    if count is not None:
                        style_alpha_counts[style].append(count)
                else:
                    with Image.open(sp) as im:
                        if im.mode != "RGBA" or im.size != (1024, 1024):
                            issues.append(f"V{v}:{style}:bad_image_mode_or_size:{im.mode}:{im.size}")
        except Exception as e:
            issues.append(f"V{v}:raster_error:{type(e).__name__}:{e}")

    if len(seen_yaws) != VIEWS or not np.allclose(np.asarray(seen_yaws), expected_yaws, atol=1e-4):
        issues.append(f"yaw_order_mismatch:{seen_yaws}")

    min_coverage = min(coverage) if coverage else 0
    min_bbox = min(bbox_fraction) if bbox_fraction else 0.0
    return {
        "asset_id": asset_id,
        "split": split,
        "source_registry_id": provider,
        "candidate_id": record.get("candidate_id"),
        "coverage_pixels_min": int(min_coverage),
        "coverage_pixels_max": int(max(coverage)) if coverage else 0,
        "coverage_fraction_min": float(min_coverage / (1024.0 * 1024.0)),
        "bbox_fraction_min": float(min_bbox),
        "geometry": geometry,
        "fatal": issues,
        "warnings": sorted(set(warnings)),
    }


def main():
    ap = argparse.ArgumentParser(description="Read-only optimizer=0 master-corpus apparatus census")
    ap.add_argument("--root", required=True, help="RealSaS_MASTER_CORPUS_1024_V3 root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug only; 0 means all selected assets")
    ap.add_argument("--no-alpha-pixel-parity", action="store_true")
    ap.add_argument("--progress-every", type=int, default=25)
    a = ap.parse_args()

    root = Path(a.root)
    selection_path, selection = _load_selection(root)
    asset_ids = sorted(selection)
    if a.limit:
        asset_ids = asset_ids[: a.limit]
    rows = []
    for i, aid in enumerate(asset_ids, 1):
        rows.append(inspect_asset(root, aid, selection, verify_alpha=not a.no_alpha_pixel_parity))
        if i % a.progress_every == 0 or i == len(asset_ids):
            fatal_n = sum(bool(r["fatal"]) for r in rows)
            print(f"[master-audit-v2] {i}/{len(asset_ids)} fatal_assets={fatal_n}", flush=True)

    coverage = [r["coverage_fraction_min"] for r in rows]
    bbox = [r["bbox_fraction_min"] for r in rows]
    pmax = [r.get("geometry", {}).get("p_max_abs") for r in rows]
    pmax = [x for x in pmax if x is not None]
    deg = [r.get("geometry", {}).get("degenerate_face_fraction") for r in rows]
    deg = [x for x in deg if x is not None]
    zero_n = [r.get("geometry", {}).get("zero_recomputed_vertex_normal_fraction") for r in rows]
    zero_n = [x for x in zero_n if x is not None]
    blank = [r["asset_id"] for r in rows if r["coverage_pixels_min"] == 0]
    fatal = [r for r in rows if r["fatal"]]
    low = sorted(rows, key=lambda r: (r["coverage_fraction_min"], r["bbox_fraction_min"], r["asset_id"]))[:50]

    report = {
        "schema": "RealSaS.IRISSinglePoseV2.MasterApparatusAudit.v1",
        "optimizer_steps": 0,
        "scope": "all canonical selected assets" if not a.limit else f"DEBUG_PREFIX_{a.limit}",
        "selection_path": str(selection_path),
        "selected_count": len(asset_ids),
        "split_counts": dict(sorted(Counter(r["split"] for r in rows).items())),
        "source_registry_counts": dict(sorted(Counter(r["source_registry_id"] for r in rows).items())),
        "alpha_pixel_parity_checked": not a.no_alpha_pixel_parity,
        "fatal_asset_count": len(fatal),
        "blank_asset_count": len(blank),
        "blank_assets": blank,
        "coverage_fraction_min_across_views": _percentiles(coverage),
        "bbox_fraction_min_across_views": _percentiles(bbox),
        "canonical_vertex_abs_envelope": _percentiles(pmax),
        "degenerate_face_fraction": _percentiles(deg),
        "zero_recomputed_vertex_normal_fraction": _percentiles(zero_n),
        "lowest_coverage_witnesses": [
            {
                "asset_id": r["asset_id"],
                "split": r["split"],
                "source_registry_id": r["source_registry_id"],
                "coverage_pixels_min": r["coverage_pixels_min"],
                "coverage_fraction_min": r["coverage_fraction_min"],
                "bbox_fraction_min": r["bbox_fraction_min"],
                "fatal": r["fatal"],
            }
            for r in low
        ],
        "fatal_assets": fatal,
        "policy": {
            "blank": "fatal/quarantine candidate",
            "tiny": "REPORT_ONLY_UNTIL_THRESHOLD_FROZEN_FROM_CENSUS; no arbitrary auto-quarantine threshold in code",
            "mesh_authority": "This census checks published geometry/raster integrity only. It does NOT close evaluated-depsgraph, concave-n-gon, or authored custom-normal parity.",
            "appearance": "cel_clean/ink_cel are geometry-control observations only; this census does NOT close appearance Gate 3.",
            "sealed": "CAL/DEV/EXTERNAL are inspected only as read-only apparatus/file authority, never used for model selection or learning metrics.",
        },
        "status": "PASS_APPARATUS_CENSUS" if len(fatal) == 0 and not a.limit else ("FAIL_APPARATUS_CENSUS" if fatal else "DEBUG_ONLY"),
    }
    atomic_json(a.out, report)
    print(json.dumps({k: report[k] for k in ("status", "selected_count", "fatal_asset_count", "blank_asset_count", "split_counts", "source_registry_counts")}, indent=2))


if __name__ == "__main__":
    main()
