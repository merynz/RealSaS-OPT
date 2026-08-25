from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from geometry import sha256_file, project_grid

OPEN_SPLITS = {"FIT", "TUNE"}
SEALED_SPLITS = {"CAL", "DEV", "EXTERNAL_HOLDOUT"}
VIEWS = 8
STYLES = ("cel_clean", "ink_cel")
CAMERA_CONTRACT = "realsas.level_orthographic_z_orbit.v1"
AUTHORITY_RESOLUTION = 1024
P_TARGET_P95 = 0.005
ALPHA_THRESHOLD_U8 = 128


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def percentiles(values):
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "min": None, "p10": None, "median": None, "p90": None, "max": None}
    return {
        "n": int(len(a)),
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def load_seed_map(path):
    if not path:
        return {}
    obj = json.load(open(path, encoding="utf-8"))
    return {r["asset_id"]: r for r in obj["records"]}


def expected_camera_basis(yaw_deg: float):
    t = np.deg2rad(float(yaw_deg))
    s, c = np.sin(t), np.cos(t)
    right = np.asarray([c, -s, 0.0], np.float32)
    up = np.asarray([0.0, 0.0, 1.0], np.float32)
    forward = np.asarray([s, c, 0.0], np.float32)
    return right, up, forward


def geometry_gauge(vertices: np.ndarray):
    if not len(vertices):
        return None
    lo = vertices.min(0)
    hi = vertices.max(0)
    center = 0.5 * (lo + hi)
    extent = hi - lo
    return {
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "bbox_center": center.tolist(),
        "bbox_extent": extent.tolist(),
        "bbox_center_inf": float(np.max(np.abs(center))),
        "largest_extent": float(np.max(extent)),
        "max_abs_coordinate": float(np.max(np.abs(vertices))),
    }


def _bbox_span(mask: np.ndarray, coordinate: str) -> int:
    if mask.ndim != 2:
        raise ValueError(mask.shape)
    if coordinate == "x":
        occupied = mask.any(axis=0)
    elif coordinate == "y":
        occupied = mask.any(axis=1)
    else:
        raise ValueError(coordinate)
    ids = np.flatnonzero(occupied)
    if not len(ids):
        raise RuntimeError("blank alpha support")
    return int(ids[-1] - ids[0] + 1)


def observable_half_extent(alpha_masks: list[np.ndarray], resolution: int) -> float:
    if len(alpha_masks) != VIEWS or resolution <= 0:
        raise RuntimeError("incomplete staged alpha sheet")
    w0 = _bbox_span(alpha_masks[0], "x") / float(resolution)
    w2 = _bbox_span(alpha_masks[2], "x") / float(resolution)
    hz = max(_bbox_span(m, "y") for m in alpha_masks) / float(resolution)
    occ = max(w0, w2, hz)
    if not np.isfinite(occ) or occ <= 0:
        raise RuntimeError(f"invalid staged alpha occupancy: {occ}")
    return float(0.5 / occ)


def inspect_one(stage_root: Path, stage_record: dict, cache_record: dict, seed_record: dict | None):
    aid = stage_record["asset_id"]
    split = str(stage_record["split"]).upper()
    ar = stage_root / "assets" / aid
    fatal = []
    warn = []

    if split not in OPEN_SPLITS:
        fatal.append(f"non_open_split_in_staging:{split}")
    marker_path = ar / "STAGE.json"
    if not marker_path.is_file():
        fatal.append("missing_STAGE.json")
        marker = {}
    else:
        marker = json.load(open(marker_path, encoding="utf-8"))
        if marker.get("schema") != "RealSaS.IRISSinglePoseV2.Stage.v2":
            fatal.append(f"bad_stage_schema:{marker.get('schema')}")
        if marker.get("physical_firewall") is not True:
            fatal.append("physical_firewall_false")
        if marker.get("geometry_fields") != ["vertices", "faces"]:
            fatal.append(f"bad_geometry_allowlist:{marker.get('geometry_fields')}")
        if int(marker.get("authority_resolution", 0) or 0) != AUTHORITY_RESOLUTION:
            fatal.append(f"bad_stage_authority_resolution:{marker.get('authority_resolution')}")
        expected = marker.get("staged_sha256", {})
        if not expected:
            fatal.append("missing_staged_sha256")
        else:
            for rel, digest in expected.items():
                p = ar / rel
                if not p.is_file() or sha256_file(p) != digest:
                    fatal.append(f"staged_hash_mismatch:{rel}")

    gp = ar / "primary_geometry.npz"
    vertices = np.zeros((0, 3), np.float32)
    faces = np.zeros((0, 3), np.int64)
    gauge = None
    if gp.is_file():
        with np.load(gp, allow_pickle=False) as z:
            if set(z.files) != {"vertices", "faces"}:
                fatal.append(f"geometry_firewall_fields:{sorted(z.files)}")
            else:
                vertices = z["vertices"].astype(np.float32)
                faces = z["faces"].astype(np.int64)
        if not np.isfinite(vertices).all():
            fatal.append("nonfinite_vertices")
        if len(vertices):
            gauge = geometry_gauge(vertices)
            if gauge["max_abs_coordinate"] > 0.55:
                fatal.append(f"canonical_coordinate_envelope_exceeded:{gauge['max_abs_coordinate']}")
            if gauge["bbox_center_inf"] > 0.01:
                fatal.append(f"canonical_bbox_not_centered:{gauge['bbox_center_inf']}")
            if not (0.98 <= gauge["largest_extent"] <= 1.02):
                fatal.append(f"canonical_largest_extent_not_unit:{gauge['largest_extent']}")
    else:
        fatal.append("missing_staged_geometry")

    cams = []
    camera_half_extents = []
    raster_counts = []
    input_resolution = int(marker.get("input_resolution", 0) or 0)
    alpha_masks = {style: [] for style in STYLES}
    for v in range(VIEWS):
        vd = ar / "renders" / f"V{v}"
        try:
            cam = json.load(open(vd / "camera.json", encoding="utf-8"))
            cams.append(cam)
            yaw = float(cam["yaw_deg"])
            if abs(yaw - 45.0 * v) > 1e-4:
                fatal.append(f"V{v}:yaw_mismatch")
            if cam.get("contract") != CAMERA_CONTRACT:
                fatal.append(f"V{v}:camera_contract:{cam.get('contract')}")
            he = float(cam.get("half_extent", np.nan))
            if not np.isfinite(he) or he <= 0:
                fatal.append(f"V{v}:invalid_half_extent:{he}")
            else:
                camera_half_extents.append(he)
            if cam.get("image_origin") != "TOP_LEFT" or cam.get("image_y_direction") != "DOWN" or cam.get("ndc_y_direction") != "UP":
                fatal.append(f"V{v}:image_coordinate_contract")
            er, eu, ef = expected_camera_basis(yaw)
            for name, actual, expected_basis in (
                ("right", cam.get("right"), er),
                ("up", cam.get("up"), eu),
                ("forward", cam.get("forward"), ef),
            ):
                aa = np.asarray(actual, np.float32) if actual is not None else np.zeros(0, np.float32)
                if aa.shape != (3,) or not np.allclose(aa, expected_basis, atol=1e-5, rtol=0):
                    fatal.append(f"V{v}:{name}_basis_mismatch")
        except Exception as e:
            fatal.append(f"V{v}:camera_error:{type(e).__name__}:{e}")
            cams.append(None)
        try:
            with np.load(vd / "raster_authority.npz", allow_pickle=False) as ra:
                pix = np.asarray(ra["pixel_linear_index"], np.int64)
                tri = np.asarray(ra["triangle_id"], np.int64)
                rr = int(np.asarray(ra["resolution"]).reshape(-1)[0])
                raster_counts.append(int(len(pix)))
                if rr != AUTHORITY_RESOLUTION:
                    fatal.append(f"V{v}:raster_authority_resolution:{rr}")
                if len(pix) == 0:
                    fatal.append(f"V{v}:blank_raster")
                if len(pix) and np.any(pix[1:] <= pix[:-1]):
                    fatal.append(f"V{v}:raster_not_sorted_unique")
                if len(tri) and (tri.min() < 0 or tri.max() >= len(faces)):
                    fatal.append(f"V{v}:triangle_out_of_bounds")
        except Exception as e:
            fatal.append(f"V{v}:raster_error:{type(e).__name__}:{e}")
        for style in STYLES:
            ip = vd / f"{style}_input.png"
            if not ip.is_file():
                fatal.append(f"V{v}:missing_{style}_input")
            else:
                with Image.open(ip) as im:
                    if im.mode != "RGBA" or im.size != (input_resolution, input_resolution):
                        fatal.append(f"V{v}:{style}:bad_input:{im.mode}:{im.size}")
                    else:
                        alpha_masks[style].append(np.asarray(im.getchannel("A"), np.uint8) >= ALPHA_THRESHOLD_U8)

    camera_half_extent = float(np.median(camera_half_extents)) if camera_half_extents else None
    if len(camera_half_extents) != VIEWS:
        fatal.append("camera_half_extent_missing")
    elif float(np.max(np.abs(np.asarray(camera_half_extents) - camera_half_extent))) > 1e-6:
        fatal.append(f"camera_half_extent_not_constant:{camera_half_extents}")

    observable_half_extents = {}
    for style in STYLES:
        try:
            observable_half_extents[style] = observable_half_extent(alpha_masks[style], input_resolution)
        except Exception as e:
            fatal.append(f"observable_scale:{style}:{type(e).__name__}:{e}")

    if cache_record.get("schema") != "RealSaS.IRISSinglePoseV2.CacheAsset.v2":
        fatal.append(f"bad_cache_schema:{cache_record.get('schema')}")
    if cache_record.get("split") != split:
        fatal.append(f"cache_split_mismatch:{cache_record.get('split')}:{split}")
    truth = Path(cache_record.get("truth_path", ""))
    if not truth.is_file():
        fatal.append("missing_truth_npz")
        truth_data = None
    elif sha256_file(truth) != cache_record.get("truth_sha256"):
        fatal.append("truth_sha256_mismatch")
        truth_data = None
    else:
        truth_data = np.load(truth, allow_pickle=False)

    track_count = 0
    geom_valid = 0
    exact_projection_max_grid_error = None
    geom_projection_p95_grid_error = None
    geom_projection_fraction_gt_1e3 = None
    observable_P_p95 = {}
    max_surface_error = None
    support_min = None
    support_median = None
    if truth_data is not None:
        try:
            required = {
                "geom_xy", "geom_p", "geom_n", "geom_mask", "track_p", "track_xy",
                "track_visible", "track_support", "track_surface_error", "track_n_view", "yaw_deg",
            }
            missing = required - set(truth_data.files)
            if missing:
                fatal.append(f"truth_missing_fields:{sorted(missing)}")
            else:
                gm = truth_data["geom_mask"].astype(bool)
                gxy = truth_data["geom_xy"].astype(np.float32)
                gp_ = truth_data["geom_p"].astype(np.float32)
                gn = truth_data["geom_n"].astype(np.float32)
                geom_valid = int(gm.sum())
                if not np.isfinite(gp_[gm]).all() or not np.isfinite(gn[gm]).all():
                    fatal.append("nonfinite_geometry_truth")
                if gm.any():
                    nlen = np.linalg.norm(gn[gm], axis=1)
                    if np.max(np.abs(nlen - 1.0)) > 1e-3:
                        fatal.append("geometry_normal_not_unit")
                    if float(np.max(np.abs(gp_[gm]))) > 0.55:
                        fatal.append("geom_p_outside_canonical_envelope")

                proj_err = []
                for v, cam in enumerate(cams):
                    mv = gm[v]
                    if cam is None or not np.any(mv):
                        continue
                    exact = project_grid(gp_[v, mv], cam)
                    ee = np.linalg.norm(exact - gxy[v, mv], axis=1)
                    proj_err.append(ee)
                if proj_err:
                    pe = np.concatenate(proj_err).astype(np.float64)
                    geom_projection_p95_grid_error = float(np.percentile(pe, 95))
                    geom_projection_fraction_gt_1e3 = float(np.mean(pe > 1e-3))
                    if geom_projection_p95_grid_error > 1e-4:
                        fatal.append(f"geom_p_projection_p95_mismatch:{geom_projection_p95_grid_error}")
                    if geom_projection_fraction_gt_1e3 > 0.005:
                        fatal.append(f"geom_p_projection_outlier_fraction:{geom_projection_fraction_gt_1e3}")
                else:
                    fatal.append("no_geom_projection_samples")

                # P-V4 learner-side geometry gate: screen-plane scale must be recoverable from
                # staged RGBA alpha alone. Camera half_extent is diagnostic and is never used here.
                for style, hobs in observable_half_extents.items():
                    errs = []
                    for v, cam in enumerate(cams):
                        mv = gm[v]
                        if cam is None or not np.any(mv):
                            continue
                        right, up, forward = expected_camera_basis(float(cam["yaw_deg"]))
                        q = gp_[v, mv]
                        grid = gxy[v, mv]
                        depth = q @ forward
                        recon = (
                            hobs * grid[:, 0, None] * right[None]
                            - hobs * grid[:, 1, None] * up[None]
                            + depth[:, None] * forward[None]
                        )
                        errs.append(np.linalg.norm(recon - q, axis=1))
                    if errs:
                        e = np.concatenate(errs).astype(np.float64)
                        p95 = float(np.percentile(e, 95))
                        observable_P_p95[style] = p95
                        if p95 > P_TARGET_P95:
                            fatal.append(f"observable_P_p95_gate:{style}:{p95}")
                    else:
                        fatal.append(f"no_observable_P_samples:{style}")

                tp = truth_data["track_p"].astype(np.float32)
                xy = truth_data["track_xy"].astype(np.float32)
                vis = truth_data["track_visible"].astype(bool)
                support = truth_data["track_support"].astype(np.int64)
                serr = truth_data["track_surface_error"].astype(np.float32)
                ytruth = truth_data["yaw_deg"].astype(np.float32)
                if ytruth.shape != (VIEWS,) or not np.allclose(ytruth, np.arange(VIEWS, dtype=np.float32) * 45.0, atol=1e-4):
                    fatal.append("truth_yaw_contract_mismatch")
                track_count = int(len(tp))
                if track_count == 0:
                    fatal.append("zero_tracks")
                if len(tp) and float(np.max(np.abs(tp))) > 0.55:
                    fatal.append("track_p_outside_canonical_envelope")
                if len(support):
                    support_min = int(support.min())
                    support_median = float(np.median(support))
                    if support_min < 2:
                        fatal.append("track_support_below_2")
                finite_serr = serr[np.isfinite(serr)]
                max_surface_error = float(finite_serr.max(initial=0.0)) if finite_serr.size else None
                declared = float(cache_record.get("settings", {}).get("max_surface_error", np.inf))
                if finite_serr.size and float(finite_serr.max()) > declared + 1e-7:
                    fatal.append("track_surface_error_exceeds_declared_gate")
                per_view = []
                for v, cam in enumerate(cams):
                    if cam is None or not np.any(vis[:, v]):
                        continue
                    exact = project_grid(tp, cam)
                    per_view.append(float(np.linalg.norm(xy[vis[:, v], v] - exact[vis[:, v]], axis=1).max(initial=0.0)))
                exact_projection_max_grid_error = max(per_view) if per_view else None
                if exact_projection_max_grid_error is None or exact_projection_max_grid_error > 1e-6:
                    fatal.append(f"track_xy_not_exact_continuous_projection:{exact_projection_max_grid_error}")
        finally:
            truth_data.close()

    geom_target = int(cache_record.get("settings", {}).get("geom_samples", 0))
    expected_geom_valid = sum(min(n, geom_target) for n in raster_counts) if geom_target else None
    if expected_geom_valid is not None and geom_valid != expected_geom_valid:
        fatal.append(f"geom_sampling_count_mismatch:{geom_valid}:{expected_geom_valid}")

    provider = (seed_record or {}).get("source_registry_id", "UNKNOWN")
    return {
        "asset_id": aid,
        "split": split,
        "source_registry_id": provider,
        "fatal": fatal,
        "warnings": warn,
        "input_resolution": input_resolution,
        "geom_valid": geom_valid,
        "track_count": track_count,
        "track_support_min": support_min,
        "track_support_median": support_median,
        "max_surface_error": max_surface_error,
        "exact_projection_max_grid_error": exact_projection_max_grid_error,
        "geom_projection_p95_grid_error": geom_projection_p95_grid_error,
        "geom_projection_fraction_gt_1e3": geom_projection_fraction_gt_1e3,
        "camera_half_extent": camera_half_extent,
        "observable_half_extent": observable_half_extents,
        "observable_P_p95": observable_P_p95,
        "canonical_bbox_center_inf": gauge["bbox_center_inf"] if gauge else None,
        "canonical_largest_extent": gauge["largest_extent"] if gauge else None,
        "canonical_max_abs_coordinate": gauge["max_abs_coordinate"] if gauge else None,
        "raster_pixels_min": min(raster_counts) if raster_counts else 0,
    }


def main():
    ap = argparse.ArgumentParser(description="Read-only optimizer=0 audit of IRIS V2 open staging/cache")
    ap.add_argument("--stage-manifest", required=True)
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--seed-manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress-every", type=int, default=25)
    a = ap.parse_args()

    stage_manifest = json.load(open(a.stage_manifest, encoding="utf-8"))
    cache_manifest = json.load(open(a.cache_manifest, encoding="utf-8"))
    stage_root = Path(a.stage_manifest).parent
    seed = load_seed_map(a.seed_manifest)
    srows = {r["asset_id"]: r for r in stage_manifest["records"]}
    crows = {r["asset_id"]: r for r in cache_manifest["records"]}
    if len(srows) != len(stage_manifest["records"]):
        raise RuntimeError("duplicate asset_id in stage manifest")
    if len(crows) != len(cache_manifest["records"]):
        raise RuntimeError("duplicate asset_id in cache manifest")
    if set(srows) != set(crows):
        missing_cache = sorted(set(srows) - set(crows))
        extra_cache = sorted(set(crows) - set(srows))
        raise RuntimeError(f"stage/cache membership mismatch missing_cache={missing_cache[:10]} extra_cache={extra_cache[:10]}")

    rows = []
    for i, aid in enumerate(sorted(srows), 1):
        rows.append(inspect_one(stage_root, srows[aid], crows[aid], seed.get(aid)))
        if i % a.progress_every == 0 or i == len(srows):
            print(f"[stage-cache-audit] {i}/{len(srows)} fatal_assets={sum(bool(r['fatal']) for r in rows)}", flush=True)

    fatal = [r for r in rows if r["fatal"]]
    observable_p = [
        p95 for r in rows for p95 in r["observable_P_p95"].values() if p95 is not None
    ]
    camera_h = [r["camera_half_extent"] for r in rows if r["camera_half_extent"] is not None]
    observable_h = [
        h for r in rows for h in r["observable_half_extent"].values() if h is not None
    ]
    report = {
        "schema": "RealSaS.IRISSinglePoseV2.StageCacheAudit.v3",
        "optimizer_steps": 0,
        "stage_manifest": str(Path(a.stage_manifest).resolve()),
        "cache_manifest": str(Path(a.cache_manifest).resolve()),
        "asset_count": len(rows),
        "split_counts": dict(sorted(Counter(r["split"] for r in rows).items())),
        "source_registry_counts": dict(sorted(Counter(r["source_registry_id"] for r in rows).items())),
        "fatal_asset_count": len(fatal),
        "geom_valid": percentiles([r["geom_valid"] for r in rows]),
        "track_count": percentiles([r["track_count"] for r in rows]),
        "track_support_median": percentiles([r["track_support_median"] for r in rows if r["track_support_median"] is not None]),
        "raster_pixels_min": percentiles([r["raster_pixels_min"] for r in rows]),
        "surface_error_max": percentiles([r["max_surface_error"] for r in rows if r["max_surface_error"] is not None]),
        "continuous_projection_grid_error_max": percentiles([r["exact_projection_max_grid_error"] for r in rows if r["exact_projection_max_grid_error"] is not None]),
        "geom_p_projection_p95_grid_error": percentiles([r["geom_projection_p95_grid_error"] for r in rows if r["geom_projection_p95_grid_error"] is not None]),
        "geom_p_projection_fraction_gt_1e3": percentiles([r["geom_projection_fraction_gt_1e3"] for r in rows if r["geom_projection_fraction_gt_1e3"] is not None]),
        "camera_half_extent_diagnostic": percentiles(camera_h),
        "observable_half_extent_from_staged_alpha": percentiles(observable_h),
        "observable_P_p95": percentiles(observable_p),
        "observable_P_p95_gate": P_TARGET_P95,
        "canonical_bbox_center_inf": percentiles([r["canonical_bbox_center_inf"] for r in rows if r["canonical_bbox_center_inf"] is not None]),
        "canonical_largest_extent": percentiles([r["canonical_largest_extent"] for r in rows if r["canonical_largest_extent"] is not None]),
        "canonical_max_abs_coordinate": percentiles([r["canonical_max_abs_coordinate"] for r in rows if r["canonical_max_abs_coordinate"] is not None]),
        "fatal_assets": fatal,
        "camera_contract": CAMERA_CONTRACT,
        "camera_half_extent_policy": "finite positive constant per asset; diagnostic only; never model input",
        "P_screen_plane_scale_authority": "staged RGBA alpha occupancy + canonical max bbox extent=1",
        "authority_resolution": AUTHORITY_RESOLUTION,
        "density_policy": "4096-class targets are audited as achieved-when-coverage-permits; no asset is failed merely for having fewer persistent physical loci than the cap.",
        "status": "PASS" if not fatal and set(r["split"] for r in rows) <= OPEN_SPLITS else "FAIL",
    }
    atomic_json(a.out, report)
    print(json.dumps({k: report[k] for k in (
        "status", "asset_count", "fatal_asset_count", "split_counts", "source_registry_counts",
        "track_count", "canonical_largest_extent", "geom_p_projection_p95_grid_error", "observable_P_p95"
    )}, indent=2))


if __name__ == "__main__":
    main()
