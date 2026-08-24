from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from geometry import geometric_vertex_normals, reconstruct_surface, pixel_linear_to_grid
from p_formulation_corpus_audit_v1 import load_legal_primary_geometry, load_panel, stable_rows, summary, expected_basis

CAMERA_CONTRACT = "realsas.level_orthographic_z_orbit.v1"
AUTHORITY_RESOLUTION = 1024
VIEWS = 8
STYLES = ("cel_clean", "ink_cel")
RESOLUTIONS = (256, 512, 1024)
ALPHA_THRESHOLD_U8 = 128
P_TARGET_P95 = 0.005


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _rgba_for(asset: Path, view: int, style: str, resolution: int) -> np.ndarray:
    vd = asset / "renders" / f"V{view}"
    if resolution == 1024:
        path = vd / f"{style}.png"
        with Image.open(path) as im:
            if im.mode != "RGBA" or im.size != (1024, 1024):
                raise RuntimeError(f"bad native RGBA {path}: {im.mode} {im.size}")
            return np.asarray(im, np.uint8)
    path = vd / f"{style}_512.png"
    with Image.open(path) as im:
        if im.mode != "RGBA" or im.size != (512, 512):
            raise RuntimeError(f"bad 512 RGBA {path}: {im.mode} {im.size}")
        if resolution == 512:
            return np.asarray(im, np.uint8)
        im = im.resize((256, 256), resample=Image.Resampling.BILINEAR)
        return np.asarray(im, np.uint8)


def _bbox_span(mask: np.ndarray, axis: int) -> int:
    if mask.ndim != 2:
        raise ValueError(mask.shape)
    occupied = mask.any(axis=axis)
    ids = np.flatnonzero(occupied)
    if not len(ids):
        raise RuntimeError("blank alpha support")
    return int(ids[-1] - ids[0] + 1)


def estimate_sheet_half_extent(asset: Path, style: str, resolution: int) -> tuple[float, dict]:
    masks = []
    for v in range(VIEWS):
        rgba = _rgba_for(asset, v, style, resolution)
        masks.append(rgba[..., 3] >= ALPHA_THRESHOLD_U8)
    w0 = _bbox_span(masks[0], axis=0)
    w2 = _bbox_span(masks[2], axis=0)
    heights = [_bbox_span(m, axis=1) for m in masks]
    hz = max(heights)
    occ = max(w0 / resolution, w2 / resolution, hz / resolution)
    if not np.isfinite(occ) or occ <= 0:
        raise RuntimeError(f"invalid alpha occupancy {style} R{resolution}: {occ}")
    h = 0.5 / occ
    return float(h), {
        "w0_px": int(w0),
        "w2_px": int(w2),
        "max_height_px": int(hz),
        "max_occupancy": float(occ),
        "half_extent": float(h),
    }


def inspect_asset(root: Path, role: str, aid: str, samples_per_view: int):
    asset = root / "master" / "assets" / aid
    fatal = []
    gp = asset / "primary_geometry.npz"
    if not gp.is_file():
        return {"asset_id": aid, "role": role, "fatal": ["missing_primary_geometry"]}
    try:
        vertices, faces, geometry_io = load_legal_primary_geometry(gp)
    except Exception as e:
        return {"asset_id": aid, "role": role, "fatal": [f"geometry:{type(e).__name__}:{e}"]}
    if not len(vertices) or not len(faces):
        fatal.append("empty_geometry")
    if not np.isfinite(vertices).all():
        fatal.append("nonfinite_vertices")
    lo = vertices.min(0); hi = vertices.max(0); center = 0.5 * (lo + hi); extent = hi - lo
    gauge = {
        "bbox_center_inf": float(np.max(np.abs(center))),
        "largest_extent": float(np.max(extent)),
        "max_abs_coordinate": float(np.max(np.abs(vertices))),
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
    }
    if gauge["max_abs_coordinate"] > 0.55:
        fatal.append(f"canonical_coordinate_envelope_exceeded:{gauge['max_abs_coordinate']}")
    if gauge["bbox_center_inf"] > 0.01:
        fatal.append(f"canonical_bbox_not_centered:{gauge['bbox_center_inf']}")
    if not (0.98 <= gauge["largest_extent"] <= 1.02):
        fatal.append(f"canonical_largest_extent_not_unit:{gauge['largest_extent']}")

    cams = []
    half_extents = []
    for v in range(VIEWS):
        vd = asset / "renders" / f"V{v}"
        try:
            cam = json.load(open(vd / "camera.json", encoding="utf-8"))
        except Exception as e:
            fatal.append(f"V{v}:camera:{type(e).__name__}:{e}")
            cams.append(None)
            continue
        cams.append(cam)
        yaw = float(cam.get("yaw_deg", np.nan))
        if cam.get("contract") != CAMERA_CONTRACT:
            fatal.append(f"V{v}:camera_contract:{cam.get('contract')}")
        if not np.isfinite(yaw) or abs(yaw - 45.0 * v) > 1e-4:
            fatal.append(f"V{v}:yaw:{yaw}")
        h = float(cam.get("half_extent", np.nan))
        if not np.isfinite(h) or h <= 0:
            fatal.append(f"V{v}:invalid_half_extent:{h}")
        else:
            half_extents.append(h)
        if cam.get("image_origin") != "TOP_LEFT" or cam.get("image_y_direction") != "DOWN" or cam.get("ndc_y_direction") != "UP":
            fatal.append(f"V{v}:image_coordinate_contract")
        er, eu, ef = expected_basis(yaw)
        for name, expected in (("right", er), ("up", eu), ("forward", ef)):
            aa = np.asarray(cam.get(name), np.float32) if cam.get(name) is not None else np.zeros(0, np.float32)
            if aa.shape != (3,) or not np.allclose(aa, expected, atol=1e-5, rtol=0):
                fatal.append(f"V{v}:{name}_basis_mismatch")
    camera_half_extent = float(np.median(half_extents)) if half_extents else None
    if len(half_extents) != VIEWS:
        fatal.append("camera_half_extent_missing")
    elif float(np.max(np.abs(np.asarray(half_extents) - camera_half_extent))) > 1e-6:
        fatal.append(f"camera_half_extent_not_constant:{half_extents}")

    observable_scales = {}
    for style in STYLES:
        observable_scales[style] = {}
        for resolution in RESOLUTIONS:
            try:
                h_obs, diag = estimate_sheet_half_extent(asset, style, resolution)
                diag["camera_half_extent_abs_error"] = (
                    abs(h_obs - camera_half_extent) if camera_half_extent is not None else None
                )
                observable_scales[style][str(resolution)] = diag
            except Exception as e:
                fatal.append(f"alpha_scale:{style}:R{resolution}:{type(e).__name__}:{e}")

    vn = geometric_vertex_normals(vertices, faces)
    projection_errors = []
    recon_errors = {(style, resolution): [] for style in STYLES for resolution in RESOLUTIONS}
    depth_values = []
    view_rows = []
    for v in range(VIEWS):
        cam = cams[v]
        if cam is None:
            continue
        vd = asset / "renders" / f"V{v}"
        try:
            with np.load(vd / "raster_authority.npz", allow_pickle=False) as ra:
                res = int(np.asarray(ra["resolution"]).reshape(-1)[0])
                pix = np.asarray(ra["pixel_linear_index"], np.int64)
                tri = np.asarray(ra["triangle_id"], np.int64)
                uv = np.asarray(ra["barycentric_uv"], np.float32)
        except Exception as e:
            fatal.append(f"V{v}:raster:{type(e).__name__}:{e}")
            continue
        if res != AUTHORITY_RESOLUTION:
            fatal.append(f"V{v}:raster_resolution:{res}")
        if len(pix) == 0:
            fatal.append(f"V{v}:blank_raster")
            continue
        if np.any(pix[1:] <= pix[:-1]):
            fatal.append(f"V{v}:raster_not_sorted_unique")
        if tri.min(initial=0) < 0 or tri.max(initial=-1) >= len(faces):
            fatal.append(f"V{v}:triangle_out_of_bounds")
            continue
        ids = stable_rows(len(pix), samples_per_view)
        p, _ = reconstruct_surface(vertices, faces, vn, tri[ids], uv[ids])
        grid = pixel_linear_to_grid(pix[ids], res)
        right = np.asarray(cam["right"], np.float32)
        up = np.asarray(cam["up"], np.float32)
        forward = np.asarray(cam["forward"], np.float32)
        hcam = float(cam["half_extent"])
        proj = np.stack([(p @ right) / hcam, -(p @ up) / hcam], axis=-1).astype(np.float32)
        perr = np.linalg.norm(proj - grid, axis=1).astype(np.float64)
        projection_errors.extend(perr.tolist())
        depth = (p @ forward).astype(np.float32)
        depth_values.extend(depth.astype(np.float64).tolist())
        for style in STYLES:
            for resolution in RESOLUTIONS:
                diag = observable_scales.get(style, {}).get(str(resolution))
                if not diag:
                    continue
                hobs = float(diag["half_extent"])
                pv = (
                    hobs * grid[:, 0, None] * right[None]
                    - hobs * grid[:, 1, None] * up[None]
                    + depth[:, None] * forward[None]
                ).astype(np.float32)
                err = np.linalg.norm(pv - p, axis=1).astype(np.float64)
                recon_errors[(style, resolution)].extend(err.tolist())
        view_rows.append({
            "view": v,
            "camera_half_extent": hcam,
            "raster_pixels": int(len(pix)),
            "sample_count": int(len(ids)),
            "projection_error": summary(perr),
        })

    ps = summary(projection_errors)
    frac_gt_1e3 = float(np.mean(np.asarray(projection_errors) > 1e-3)) if projection_errors else None
    if ps["p95"] is None or ps["p95"] > 1e-4:
        fatal.append(f"projection_p95_gate:{ps['p95']}")
    if frac_gt_1e3 is None or frac_gt_1e3 > 0.005:
        fatal.append(f"projection_outlier_fraction_gate:{frac_gt_1e3}")

    observable_p_errors = {}
    for style in STYLES:
        observable_p_errors[style] = {}
        for resolution in RESOLUTIONS:
            s = summary(recon_errors[(style, resolution)])
            observable_p_errors[style][str(resolution)] = s
            if s["p95"] is None or s["p95"] > P_TARGET_P95:
                fatal.append(f"observable_P_p95_gate:{style}:R{resolution}:{s['p95']}")

    return {
        "asset_id": aid,
        "role": role,
        "fatal": fatal,
        "geometry_io": geometry_io,
        "gauge": gauge,
        "camera_half_extent": camera_half_extent,
        "camera_half_extent_by_view": half_extents,
        "observable_scales": observable_scales,
        "teacher_projection_error": ps,
        "teacher_projection_fraction_gt_1e3": frac_gt_1e3,
        "observable_P_error": observable_p_errors,
        "depth": summary(depth_values),
        "views": view_rows,
    }


def main():
    ap = argparse.ArgumentParser(description="Optimizer-zero real-corpus closure for image-derived sheet-scale P V4")
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", default=str(Path(__file__).with_name("P_FORMULATION_V3_PANEL_V1.json")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-per-view", type=int, default=4096)
    a = ap.parse_args()
    root = Path(a.root)
    panel, rows = load_panel(a.panel)
    result = []
    for i, (role, aid) in enumerate(rows, 1):
        row = inspect_asset(root, role, aid, a.samples_per_view)
        result.append(row)
        print(f"[P-V4-corpus] {i}/{len(rows)} {aid} fatal={len(row['fatal'])}", flush=True)
    fatal = [x for x in result if x["fatal"]]

    all_recon_p95 = []
    scale_abs_err = []
    for row in result:
        for style in STYLES:
            for resolution in RESOLUTIONS:
                p95 = row.get("observable_P_error", {}).get(style, {}).get(str(resolution), {}).get("p95")
                if p95 is not None:
                    all_recon_p95.append(p95)
                se = row.get("observable_scales", {}).get(style, {}).get(str(resolution), {}).get("camera_half_extent_abs_error")
                if se is not None:
                    scale_abs_err.append(se)

    report = {
        "schema": "RealSaS.IRISSinglePoseV2.PFormulationV4ObservableScaleClosure.v1",
        "status": "P_V4_OBSERVABLE_SCALE_GEOMETRY_CLOSED" if not fatal else "P_V4_OBSERVABLE_SCALE_GEOMETRY_FAIL",
        "optimizer_steps": 0,
        "training_authorized": False,
        "model_camera_half_extent_input": False,
        "alpha_consumed": True,
        "rgb_color_consumed": False,
        "tune_consumed": False,
        "sealed_splits_opened": False,
        "observable_scale_authority": "RGBA alpha occupancy + canonical max bbox extent=1",
        "alpha_threshold": 0.5,
        "P_p95_gate": P_TARGET_P95,
        "asset_count": len(result),
        "fatal_asset_count": len(fatal),
        "samples_per_view": int(a.samples_per_view),
        "camera_half_extent_across_assets": summary([x["camera_half_extent"] for x in result if x.get("camera_half_extent") is not None]),
        "observable_scale_abs_error_vs_camera": summary(scale_abs_err),
        "observable_P_p95_across_asset_style_resolution": summary(all_recon_p95),
        "canonical_bbox_center_inf": summary([x["gauge"]["bbox_center_inf"] for x in result if x.get("gauge")]),
        "canonical_largest_extent": summary([x["gauge"]["largest_extent"] for x in result if x.get("gauge")]),
        "canonical_max_abs_coordinate": summary([x["gauge"]["max_abs_coordinate"] for x in result if x.get("gauge")]),
        "fatal_assets": fatal,
        "assets": result,
    }
    atomic_json(a.out, report)
    print(json.dumps({
        "status": report["status"],
        "asset_count": report["asset_count"],
        "fatal_asset_count": report["fatal_asset_count"],
        "camera_half_extent_across_assets": report["camera_half_extent_across_assets"],
        "observable_scale_abs_error_vs_camera": report["observable_scale_abs_error_vs_camera"],
        "observable_P_p95_across_asset_style_resolution": report["observable_P_p95_across_asset_style_resolution"],
    }, indent=2), flush=True)
    if fatal:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
