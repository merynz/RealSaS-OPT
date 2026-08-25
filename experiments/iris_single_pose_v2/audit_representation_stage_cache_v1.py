from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from geometry import project_grid, sha256_file

OPEN_SPLITS = {"FIT", "TUNE"}
VIEWS = 8
EXPECTED_STAGE_SCHEMA = "RealSaS.IRISSinglePoseV2.RepresentationStage.v1"
EXPECTED_STAGE_MANIFEST_SCHEMA = "RealSaS.IRISSinglePoseV2.RepresentationStageManifest.v1"
EXPECTED_STAGE_PROFILE = "representation_authority_geometry_only"
EXPECTED_CACHE_SCHEMA = "RealSaS.IRISSinglePoseV2.CacheManifest.v2"
EXPECTED_CACHE_ASSET_SCHEMA = "RealSaS.IRISSinglePoseV2.CacheAsset.v2"


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
    obj = json.load(open(path, encoding="utf-8"))
    return {r["asset_id"]: r for r in obj["records"]}


def inspect_one(stage_root: Path, stage_record: dict, cache_record: dict, seed_record: dict | None):
    aid = stage_record["asset_id"]
    split = str(stage_record["split"]).upper()
    ar = stage_root / "assets" / aid
    fatal = []

    if split not in OPEN_SPLITS:
        fatal.append(f"non_open_split:{split}")

    marker_path = ar / "STAGE.json"
    marker = {}
    if not marker_path.is_file():
        fatal.append("missing_STAGE.json")
    else:
        marker = json.load(marker_path.open(encoding="utf-8"))
        if marker.get("schema") != EXPECTED_STAGE_SCHEMA:
            fatal.append(f"bad_stage_schema:{marker.get('schema')}")
        if marker.get("stage_profile") != EXPECTED_STAGE_PROFILE:
            fatal.append(f"bad_stage_profile:{marker.get('stage_profile')}")
        if marker.get("physical_firewall") is not True:
            fatal.append("physical_firewall_false")
        if marker.get("geometry_fields") != ["vertices", "faces"]:
            fatal.append(f"bad_geometry_allowlist:{marker.get('geometry_fields')}")
        if marker.get("images_staged") is not False:
            fatal.append("images_staged_not_false")
        expected = marker.get("staged_sha256", {})
        if not expected:
            fatal.append("missing_staged_sha256")
        for rel, digest in expected.items():
            p = ar / rel
            if not p.is_file() or sha256_file(p) != digest:
                fatal.append(f"staged_hash_mismatch:{rel}")

    gp = ar / "primary_geometry.npz"
    faces = np.zeros((0, 3), np.int64)
    if not gp.is_file():
        fatal.append("missing_staged_geometry")
    else:
        with np.load(gp, allow_pickle=False) as z:
            if set(z.files) != {"vertices", "faces"}:
                fatal.append(f"geometry_firewall_fields:{sorted(z.files)}")
            else:
                vertices = np.asarray(z["vertices"], np.float32)
                faces = np.asarray(z["faces"], np.int64)
                if not np.isfinite(vertices).all():
                    fatal.append("nonfinite_vertices")
                if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
                    fatal.append("face_index_out_of_bounds")

    cams = []
    raster_counts = []
    for v in range(VIEWS):
        vd = ar / "renders" / f"V{v}"
        allowed = {"camera.json", "raster_authority.npz"}
        if not vd.is_dir():
            fatal.append(f"V{v}:missing_render_dir")
            cams.append(None)
            continue
        extras = sorted(p.name for p in vd.iterdir() if p.is_file() and p.name not in allowed)
        if extras:
            fatal.append(f"V{v}:unexpected_files:{extras}")
        missing = sorted(name for name in allowed if not (vd / name).is_file())
        if missing:
            fatal.append(f"V{v}:missing_files:{missing}")

        try:
            cam = json.load((vd / "camera.json").open(encoding="utf-8"))
            cams.append(cam)
            if abs(float(cam["yaw_deg"]) - 45.0 * v) > 1e-4:
                fatal.append(f"V{v}:yaw_mismatch")
        except Exception as e:
            fatal.append(f"V{v}:camera_error:{type(e).__name__}:{e}")
            cams.append(None)

        try:
            with np.load(vd / "raster_authority.npz", allow_pickle=False) as ra:
                required = {"resolution", "pixel_linear_index", "triangle_id", "barycentric_uv"}
                miss = required - set(ra.files)
                if miss:
                    fatal.append(f"V{v}:raster_missing:{sorted(miss)}")
                    continue
                res = int(np.asarray(ra["resolution"]).reshape(-1)[0])
                if res != 1024:
                    fatal.append(f"V{v}:authority_resolution:{res}")
                pix = np.asarray(ra["pixel_linear_index"], np.int64)
                tri = np.asarray(ra["triangle_id"], np.int64)
                raster_counts.append(int(len(pix)))
                if len(pix) == 0:
                    fatal.append(f"V{v}:blank_raster")
                if len(pix) and np.any(pix[1:] <= pix[:-1]):
                    fatal.append(f"V{v}:raster_not_sorted_unique")
                if len(tri) and (tri.min() < 0 or tri.max() >= len(faces)):
                    fatal.append(f"V{v}:triangle_out_of_bounds")
        except Exception as e:
            fatal.append(f"V{v}:raster_error:{type(e).__name__}:{e}")

    if cache_record.get("schema") != EXPECTED_CACHE_ASSET_SCHEMA:
        fatal.append(f"bad_cache_schema:{cache_record.get('schema')}")
    if cache_record.get("stage_profile") != EXPECTED_STAGE_PROFILE:
        fatal.append(f"bad_cache_stage_profile:{cache_record.get('stage_profile')}")
    if cache_record.get("rgb_consumed") is not False:
        fatal.append("cache_rgb_consumed_not_false")
    if str(cache_record.get("split", "")).upper() != split:
        fatal.append(f"cache_split_mismatch:{cache_record.get('split')}:{split}")

    truth = Path(cache_record.get("truth_path", ""))
    truth_data = None
    if not truth.is_file():
        fatal.append("missing_truth_npz")
    elif sha256_file(truth) != cache_record.get("truth_sha256"):
        fatal.append("truth_sha256_mismatch")
    else:
        truth_data = np.load(truth, allow_pickle=False)

    track_count = 0
    geom_valid = 0
    exact_projection_max_grid_error = None
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
                gp_ = truth_data["geom_p"].astype(np.float32)
                gn = truth_data["geom_n"].astype(np.float32)
                geom_valid = int(gm.sum())
                if not np.isfinite(gp_[gm]).all() or not np.isfinite(gn[gm]).all():
                    fatal.append("nonfinite_geometry_truth")
                if gm.any():
                    nlen = np.linalg.norm(gn[gm], axis=1)
                    if np.max(np.abs(nlen - 1.0)) > 1e-3:
                        fatal.append("geometry_normal_not_unit")

                tp = truth_data["track_p"].astype(np.float32)
                xy = truth_data["track_xy"].astype(np.float32)
                vis = truth_data["track_visible"].astype(bool)
                support = truth_data["track_support"].astype(np.int64)
                serr = truth_data["track_surface_error"].astype(np.float32)
                track_count = int(len(tp))
                if track_count == 0:
                    fatal.append("zero_tracks")
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
        "geom_valid": geom_valid,
        "track_count": track_count,
        "track_support_min": support_min,
        "track_support_median": support_median,
        "max_surface_error": max_surface_error,
        "exact_projection_max_grid_error": exact_projection_max_grid_error,
        "raster_pixels_min": min(raster_counts) if raster_counts else 0,
    }


def main():
    ap = argparse.ArgumentParser(description="Read-only audit for representation-only geometry/raster/camera staging and cache")
    ap.add_argument("--stage-manifest", required=True)
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--seed-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress-every", type=int, default=25)
    a = ap.parse_args()

    sm = json.load(open(a.stage_manifest, encoding="utf-8"))
    cm = json.load(open(a.cache_manifest, encoding="utf-8"))
    if sm.get("schema") != EXPECTED_STAGE_MANIFEST_SCHEMA or sm.get("stage_profile") != EXPECTED_STAGE_PROFILE:
        raise RuntimeError(f"wrong representation stage manifest: {sm.get('schema')} {sm.get('stage_profile')}")
    if sm.get("images_staged") is not False:
        raise RuntimeError("representation stage manifest claims images are staged")
    if cm.get("schema") != EXPECTED_CACHE_SCHEMA or cm.get("stage_profile") != EXPECTED_STAGE_PROFILE:
        raise RuntimeError(f"wrong representation cache manifest: {cm.get('schema')} {cm.get('stage_profile')}")
    if cm.get("rgb_consumed") is not False:
        raise RuntimeError("representation cache claims RGB was consumed")

    stage_root = Path(a.stage_manifest).parent
    seed = load_seed_map(a.seed_manifest)
    srows = {r["asset_id"]: r for r in sm["records"]}
    crows = {r["asset_id"]: r for r in cm["records"]}
    if len(srows) != len(sm["records"]) or len(crows) != len(cm["records"]):
        raise RuntimeError("duplicate asset_id in stage/cache manifest")
    if set(srows) != set(crows):
        raise RuntimeError("stage/cache membership mismatch")
    if set(srows) != set(seed):
        raise RuntimeError("stage/cache membership differs from frozen seed")

    rows = []
    for i, aid in enumerate(sorted(srows), 1):
        rows.append(inspect_one(stage_root, srows[aid], crows[aid], seed.get(aid)))
        if i % a.progress_every == 0 or i == len(srows):
            print(f"[repr-stage-cache-audit] {i}/{len(srows)} fatal_assets={sum(bool(r['fatal']) for r in rows)}", flush=True)

    fatal = [r for r in rows if r["fatal"]]
    report = {
        "schema": "RealSaS.IRISSinglePoseV2.RepresentationStageCacheAudit.v1",
        "optimizer_steps": 0,
        "stage_profile": EXPECTED_STAGE_PROFILE,
        "rgb_consumed": False,
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
        "fatal_assets": fatal,
        "images_required": False,
        "density_policy": "4096-class targets are achieved-when-coverage-permits; no replacement or hidden image dependency is allowed.",
        "status": "PASS" if not fatal and set(r["split"] for r in rows) <= OPEN_SPLITS else "FAIL",
    }
    atomic_json(a.out, report)
    print(json.dumps({k: report[k] for k in ("status", "asset_count", "fatal_asset_count", "split_counts", "track_count", "images_required")}, indent=2))


if __name__ == "__main__":
    main()
