from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from geometry import geometric_vertex_normals, reconstruct_surface, pixel_linear_to_grid

CAMERA_CONTRACT = "realsas.level_orthographic_z_orbit.v1"
ORTHO_HALF_EXTENT = 0.54
AUTHORITY_RESOLUTION = 1024
VIEWS = 8
LEGAL_GEOMETRY_FIELDS = ("vertices", "faces")


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def summary(values):
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "min": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(len(a)),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def expected_basis(yaw_deg):
    t = np.deg2rad(float(yaw_deg))
    s, c = np.sin(t), np.cos(t)
    return (
        np.asarray([c, -s, 0.0], np.float32),
        np.asarray([0.0, 0.0, 1.0], np.float32),
        np.asarray([s, c, 0.0], np.float32),
    )


def stable_rows(n: int, take: int):
    n = int(n)
    take = min(int(take), n)
    if take <= 0:
        return np.zeros(0, np.int64)
    if take == n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, take).round().astype(np.int64))


def load_panel(path):
    obj = json.load(open(path, encoding="utf-8"))
    if obj.get("schema") != "RealSaS.IRISSinglePoseV2.PFormulationV3Panel.v1":
        raise RuntimeError(f"bad panel schema: {obj.get('schema')}")
    if int(obj.get("tune_assets", -1)) != 0 or int(obj.get("sealed_assets", -1)) != 0:
        raise RuntimeError("P V3 panel leaks TUNE/sealed assets")
    rows = []
    for role in ("FIT_SELECT_SENTINEL", "FIT_TRAIN_SENTINEL"):
        for aid in obj["roles"][role]:
            rows.append((role, aid))
    if len(rows) != int(obj["asset_count"]) or len({x[1] for x in rows}) != len(rows):
        raise RuntimeError("panel membership/count drift")
    return obj, rows


def load_legal_primary_geometry(path: Path):
    """Consume only vertices/faces from a master NPZ that may legally contain extra fields.

    The master corpus is a superset authority. This optimizer-zero audit must not interpret,
    deserialize or depend on rig/mechanics metadata. With allow_pickle=False, an object-dtype
    hidden field acts as an executable bomb if future code accidentally consumes it.
    """
    with np.load(path, allow_pickle=False) as z:
        available = tuple(z.files)
        missing = [k for k in LEGAL_GEOMETRY_FIELDS if k not in available]
        if missing:
            raise RuntimeError(f"missing_legal_geometry_fields:{missing}")
        vertices = z["vertices"].astype(np.float32)
        faces = z["faces"].astype(np.int64)
    return vertices, faces, {
        "available_field_names": list(available),
        "consumed_fields": list(LEGAL_GEOMETRY_FIELDS),
        "ignored_field_names": sorted(set(available) - set(LEGAL_GEOMETRY_FIELDS)),
        "hidden_field_values_consumed": False,
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
        return {"asset_id": aid, "role": role, "fatal": [f"primary_geometry:{type(e).__name__}:{e}"]}

    if not len(vertices) or not len(faces):
        fatal.append("empty_geometry")
    if not np.isfinite(vertices).all():
        fatal.append("nonfinite_vertices")
    lo = vertices.min(0) if len(vertices) else np.zeros(3, np.float32)
    hi = vertices.max(0) if len(vertices) else np.zeros(3, np.float32)
    center = 0.5 * (lo + hi)
    extent = hi - lo
    gauge = {
        "bbox_center_inf": float(np.max(np.abs(center))),
        "largest_extent": float(np.max(extent)),
        "max_abs_coordinate": float(np.max(np.abs(vertices))) if len(vertices) else None,
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
    }
    if gauge["max_abs_coordinate"] is not None and gauge["max_abs_coordinate"] > 0.55:
        fatal.append(f"canonical_coordinate_envelope_exceeded:{gauge['max_abs_coordinate']}")
    if gauge["bbox_center_inf"] > 0.01:
        fatal.append(f"canonical_bbox_not_centered:{gauge['bbox_center_inf']}")
    if not (0.98 <= gauge["largest_extent"] <= 1.02):
        fatal.append(f"canonical_largest_extent_not_unit:{gauge['largest_extent']}")

    vn = geometric_vertex_normals(vertices, faces)
    projection_errors = []
    analytic_reconstruction_errors = []
    depth_values = []
    basis_errors = []
    view_rows = []

    for v in range(VIEWS):
        vd = asset / "renders" / f"V{v}"
        try:
            cam = json.load(open(vd / "camera.json", encoding="utf-8"))
        except Exception as e:
            fatal.append(f"V{v}:camera:{type(e).__name__}:{e}")
            continue
        yaw = float(cam.get("yaw_deg", np.nan))
        if cam.get("contract") != CAMERA_CONTRACT:
            fatal.append(f"V{v}:camera_contract:{cam.get('contract')}")
        if not np.isfinite(yaw) or abs(yaw - 45.0 * v) > 1e-4:
            fatal.append(f"V{v}:yaw:{yaw}")
        if abs(float(cam.get("half_extent", np.nan)) - ORTHO_HALF_EXTENT) > 1e-6:
            fatal.append(f"V{v}:half_extent:{cam.get('half_extent')}")
        if cam.get("image_origin") != "TOP_LEFT" or cam.get("image_y_direction") != "DOWN" or cam.get("ndc_y_direction") != "UP":
            fatal.append(f"V{v}:image_coordinate_contract")
        er, eu, ef = expected_basis(yaw)
        actual_basis = []
        for name, expected in (("right", er), ("up", eu), ("forward", ef)):
            aa = np.asarray(cam.get(name), np.float32) if cam.get(name) is not None else np.zeros(0, np.float32)
            be = float(np.max(np.abs(aa - expected))) if aa.shape == (3,) else np.inf
            basis_errors.append(be)
            actual_basis.append(aa)
            if not np.isfinite(be) or be > 1e-5:
                fatal.append(f"V{v}:{name}_basis_error:{be}")
        right, up, forward = actual_basis

        rp = vd / "raster_authority.npz"
        try:
            with np.load(rp, allow_pickle=False) as ra:
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
        he = float(cam["half_extent"])
        proj = np.stack([(p @ right) / he, -(p @ up) / he], axis=-1).astype(np.float32)
        perr = np.linalg.norm(proj - grid, axis=1).astype(np.float64)
        projection_errors.extend(perr.tolist())
        depth = (p @ forward).astype(np.float32)
        p_v3 = (
            he * grid[:, 0, None] * right[None]
            - he * grid[:, 1, None] * up[None]
            + depth[:, None] * forward[None]
        ).astype(np.float32)
        aerr = np.linalg.norm(p_v3 - p, axis=1).astype(np.float64)
        analytic_reconstruction_errors.extend(aerr.tolist())
        depth_values.extend(depth.astype(np.float64).tolist())
        view_rows.append({
            "view": v,
            "yaw_deg": yaw,
            "raster_pixels": int(len(pix)),
            "sample_count": int(len(ids)),
            "projection_error": summary(perr),
            "analytic_reconstruction_error": summary(aerr),
            "depth": summary(depth),
        })

    ps = summary(projection_errors)
    ars = summary(analytic_reconstruction_errors)
    frac_gt_1e3 = float(np.mean(np.asarray(projection_errors) > 1e-3)) if projection_errors else None
    if ps["p95"] is None or ps["p95"] > 1e-4:
        fatal.append(f"projection_p95_gate:{ps['p95']}")
    if frac_gt_1e3 is None or frac_gt_1e3 > 0.005:
        fatal.append(f"projection_outlier_fraction_gate:{frac_gt_1e3}")
    if ars["p95"] is None or ars["p95"] > 1e-5:
        fatal.append(f"analytic_reconstruction_p95_gate:{ars['p95']}")

    return {
        "asset_id": aid,
        "role": role,
        "fatal": fatal,
        "geometry_io": geometry_io,
        "gauge": gauge,
        "camera_basis_error": summary(basis_errors),
        "projection_error": ps,
        "projection_fraction_gt_1e3": frac_gt_1e3,
        "analytic_reconstruction_error": ars,
        "depth": summary(depth_values),
        "views": view_rows,
    }


def main():
    ap = argparse.ArgumentParser(description="Optimizer-zero real-corpus closure for camera-conditioned P V3")
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", default=str(Path(__file__).with_name("P_FORMULATION_V3_PANEL_V1.json")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-per-view", type=int, default=4096)
    a = ap.parse_args()
    root = Path(a.root)
    panel, rows = load_panel(a.panel)
    result = []
    for i, (role, aid) in enumerate(rows, 1):
        r = inspect_asset(root, role, aid, a.samples_per_view)
        result.append(r)
        print(f"[P-V3-corpus] {i}/{len(rows)} {aid} fatal={len(r['fatal'])}", flush=True)
    fatal = [r for r in result if r["fatal"]]
    report = {
        "schema": "RealSaS.IRISSinglePoseV2.PFormulationV3CorpusClosure.v2",
        "status": "P_V3_FORMULATION_GEOMETRY_CLOSED" if not fatal else "P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL",
        "optimizer_steps": 0,
        "training_authorized": False,
        "rgb_consumed": False,
        "tune_consumed": False,
        "sealed_splits_opened": False,
        "master_geometry_policy": "primary_geometry may be a superset; consume vertices/faces only; hidden field values are not loaded",
        "camera_contract": CAMERA_CONTRACT,
        "orthographic_half_extent": ORTHO_HALF_EXTENT,
        "authority_resolution": AUTHORITY_RESOLUTION,
        "panel_schema": panel["schema"],
        "asset_count": len(result),
        "fatal_asset_count": len(fatal),
        "samples_per_view": int(a.samples_per_view),
        "canonical_bbox_center_inf": summary([r["gauge"]["bbox_center_inf"] for r in result if r.get("gauge")]),
        "canonical_largest_extent": summary([r["gauge"]["largest_extent"] for r in result if r.get("gauge")]),
        "canonical_max_abs_coordinate": summary([r["gauge"]["max_abs_coordinate"] for r in result if r.get("gauge")]),
        "projection_p95_across_assets": summary([r["projection_error"]["p95"] for r in result if r.get("projection_error")]),
        "projection_fraction_gt_1e3_across_assets": summary([r["projection_fraction_gt_1e3"] for r in result if r.get("projection_fraction_gt_1e3") is not None]),
        "analytic_reconstruction_p95_across_assets": summary([r["analytic_reconstruction_error"]["p95"] for r in result if r.get("analytic_reconstruction_error")]),
        "fatal_assets": fatal,
        "assets": result,
    }
    atomic_json(a.out, report)
    print(json.dumps({k: report[k] for k in ("status", "asset_count", "fatal_asset_count", "canonical_largest_extent", "projection_p95_across_assets", "analytic_reconstruction_p95_across_assets")}, indent=2), flush=True)
    if fatal:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
