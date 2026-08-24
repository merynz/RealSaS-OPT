from __future__ import annotations
import argparse, hashlib, json, math, os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import lsmr

VIEWS = 8
STYLES = ("cel_clean", "ink_cel")
YAWS = np.arange(8, dtype=np.float64) * 45.0
CANDIDATE_HW = (64, 128, 256)
SAMPLES_PER_VIEW = 4096
TARGET_P95 = 0.005


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def seed_for(aid, view):
    return int(hashlib.sha256(f"{aid}|pv5-depth|{view}".encode()).hexdigest()[:16], 16) & 0x7fffffff


def bbox_span(mask, axis):
    occ = mask.any(axis=0 if axis == "x" else 1)
    idx = np.flatnonzero(occ)
    if not len(idx):
        raise RuntimeError("blank alpha support")
    return int(idx[-1] - idx[0] + 1)


def native_h(asset, style):
    alpha = []
    for v in range(VIEWS):
        p = asset / "renders" / f"V{v}" / f"{style}.png"
        with Image.open(p) as im:
            if im.mode != "RGBA" or im.size != (1024, 1024):
                raise RuntimeError(f"bad native RGBA: {p}: {im.mode} {im.size}")
            alpha.append(np.asarray(im.getchannel("A"), np.uint8))
    fg = [a >= 128 for a in alpha]
    w0 = bbox_span(fg[0], "x") / 1024.0
    w2 = bbox_span(fg[2], "x") / 1024.0
    hz = max(bbox_span(m, "y") / 1024.0 for m in fg)
    return float(0.5 / max(w0, w2, hz))


def vertex_normals(vertices, faces):
    v = np.asarray(vertices, np.float64); f = np.asarray(faces, np.int64)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(v)
    for j in range(3):
        np.add.at(out, f[:, j], fn)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def reconstruct_surface(vertices, faces, tri_id, bary_uv):
    uv = np.asarray(bary_uv, np.float64)
    w = np.stack([uv[:, 0], uv[:, 1], 1.0 - uv[:, 0] - uv[:, 1]], axis=-1)
    return (vertices[faces[np.asarray(tri_id, np.int64)]] * w[..., None]).sum(axis=-2)


def pixel_linear_to_grid(pix, resolution):
    pix = np.asarray(pix, np.int64)
    y = pix // resolution; x = pix % resolution
    return np.stack([2.0 * (x + 0.5) / resolution - 1.0,
                     2.0 * (y + 0.5) / resolution - 1.0], axis=-1).astype(np.float64)


def basis(yaw_deg):
    t = math.radians(float(yaw_deg)); s, c = math.sin(t), math.cos(t)
    right = np.array([c, -s, 0.0], np.float64)
    up = np.array([0.0, 0.0, 1.0], np.float64)
    forward = np.array([s, c, 0.0], np.float64)
    return right, up, forward


def bilinear_matrix(xy, h, w):
    """Sparse matrix matching grid_sample bilinear/border/align_corners=False."""
    xy = np.asarray(xy, np.float64)
    ux = ((xy[:, 0] + 1.0) * w - 1.0) * 0.5
    uy = ((xy[:, 1] + 1.0) * h - 1.0) * 0.5
    ux = np.clip(ux, 0.0, w - 1.0); uy = np.clip(uy, 0.0, h - 1.0)
    x0 = np.floor(ux).astype(np.int64); y0 = np.floor(uy).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    wx1 = ux - x0; wy1 = uy - y0; wx0 = 1.0 - wx1; wy0 = 1.0 - wy1
    rows = np.repeat(np.arange(len(xy), dtype=np.int64), 4)
    cols = np.stack([y0*w+x0, y0*w+x1, y1*w+x0, y1*w+x1], axis=1).reshape(-1)
    data = np.stack([wy0*wx0, wy0*wx1, wy1*wx0, wy1*wx1], axis=1).reshape(-1)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(len(xy), h*w), dtype=np.float64).tocsr()
    A.sum_duplicates()
    return A


def stats(x):
    x = np.asarray(x, np.float64)
    return {
        "n": int(x.size),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def load_asset_truth(root, aid, samples_per_view):
    asset = Path(root) / "master" / "assets" / aid
    if not asset.is_dir():
        raise FileNotFoundError(asset)
    with np.load(asset / "primary_geometry.npz", allow_pickle=False) as z:
        vertices = np.asarray(z["vertices"], np.float64)
        faces = np.asarray(z["faces"], np.int64)
    per_view = []
    for v in range(VIEWS):
        rp = asset / "renders" / f"V{v}" / "raster_authority.npz"
        with np.load(rp, allow_pickle=False) as ra:
            pix = np.asarray(ra["pixel_linear_index"], np.int64)
            tri = np.asarray(ra["triangle_id"], np.int64)
            uv = np.asarray(ra["barycentric_uv"], np.float64)
            res = int(np.asarray(ra["resolution"]).reshape(-1)[0])
        take = min(int(samples_per_view), len(pix))
        rng = np.random.default_rng(seed_for(aid, v))
        ids = np.sort(rng.choice(len(pix), size=take, replace=False))
        p = reconstruct_surface(vertices, faces, tri[ids], uv[ids])
        xy = pixel_linear_to_grid(pix[ids], res)
        _, _, fwd = basis(YAWS[v])
        depth = p @ fwd
        per_view.append({"xy": xy, "p": p, "depth": depth})
    hs = {style: native_h(asset, style) for style in STYLES}
    return asset, per_view, hs


def solve_view(xy, depth, hw):
    A = bilinear_matrix(xy, hw, hw)
    sol = lsmr(A, np.asarray(depth, np.float64), atol=1e-11, btol=1e-11, conlim=1e12, maxiter={64:5000,128:10000,256:5000}.get(hw,5000))
    field = sol[0]
    pred = A @ field
    return pred, {
        "istop": int(sol[1]), "iterations": int(sol[2]),
        "normr": float(sol[3]), "normar": float(sol[4]),
        "conda": float(sol[6]), "normx": float(sol[7]),
    }


def run(root, membership, out, samples_per_view=SAMPLES_PER_VIEW):
    mem = json.load(open(membership, encoding="utf-8"))
    rows = mem["records"]
    if len(rows) != 8 or any(r.get("split") != "FIT" for r in rows):
        raise RuntimeError("exact 8 FIT membership required")
    all_results = []
    aggregate = {hw: {style: [] for style in STYLES} for hw in CANDIDATE_HW}
    cells = {hw: [] for hw in CANDIDATE_HW}

    for ai, rec in enumerate(rows, 1):
        aid = rec["asset_id"]
        asset, views, hs = load_asset_truth(root, aid, samples_per_view)
        row = {"asset_id": aid, "h_native_by_style": hs, "resolutions": {}}
        for hw in CANDIDATE_HW:
            style_err = {style: [] for style in STYLES}
            view_rows = []
            for v, truth in enumerate(views):
                d_pred, srep = solve_view(truth["xy"], truth["depth"], hw)
                right, up, fwd = basis(YAWS[v])
                vr = {"view": v, "solver": srep, "depth_abs": stats(np.abs(d_pred - truth["depth"]))}
                for style in STYLES:
                    h = hs[style]; xy = truth["xy"]
                    pp = (h * xy[:, 0, None] * right[None]
                          - h * xy[:, 1, None] * up[None]
                          + d_pred[:, None] * fwd[None])
                    err = np.linalg.norm(pp - truth["p"], axis=1)
                    style_err[style].append(err)
                view_rows.append(vr)
            rrow = {"views": view_rows, "styles": {}}
            for style in STYLES:
                err = np.concatenate(style_err[style])
                st = stats(err)
                rrow["styles"][style] = st
                aggregate[hw][style].append(err)
                cells[hw].append({"asset_id": aid, "style": style, **st})
            row["resolutions"][str(hw)] = rrow
        all_results.append(row)
        print(f"[field-closure] {ai}/8 {aid}", flush=True)

    resolution_summary = {}
    certified = []
    for hw in CANDIDATE_HW:
        cell_rows = cells[hw]
        ok = all(c["p95"] <= TARGET_P95 for c in cell_rows)
        if ok: certified.append(hw)
        resolution_summary[str(hw)] = {
            "field_hw": hw,
            "relative_to_R256": {64:"R/4",128:"R/2",256:"R"}[hw],
            "certified_all_16_cells": bool(ok),
            "worst_cell_P_p95": float(max(c["p95"] for c in cell_rows)),
            "best_cell_P_p95": float(min(c["p95"] for c in cell_rows)),
            "cells": cell_rows,
            "global_by_style": {style: stats(np.concatenate(aggregate[hw][style])) for style in STYLES},
        }
    smallest = min(certified) if certified else None
    report = {
        "schema": "RealSaS.IRISSinglePoseV2.PV5FieldRepresentationClosure.v1",
        "status": "P_V5_FIELD_REPRESENTATION_CLOSED" if certified else "P_V5_FIELD_REPRESENTATION_NOT_CLOSED",
        "parent_status": "P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED",
        "historical_depth_overfit_status": "P_V5_DEPTH_OPTIMIZATION_INSUFFICIENT",
        "neural_optimizer_steps": 0,
        "oracle_solver": "scipy.sparse.linalg.lsmr__bilinear_grid_sample_matrix",
        "oracle_semantics": "sufficiency certificate only; a failed resolution is NOT_CERTIFIED, not an impossibility proof",
        "camera_json_consumed": False,
        "teacher_camera_half_extent_consumed": False,
        "tune_consumed": False,
        "sealed_splits_opened": False,
        "asset_count": 8,
        "asset_style_cells": 16,
        "samples_per_view": int(samples_per_view),
        "target_P_p95": TARGET_P95,
        "candidate_field_hw": list(CANDIDATE_HW),
        "certified_field_hw": certified,
        "smallest_certified_field_hw": smallest,
        "current_R2_certified": 128 in certified,
        "full_R_certified": 256 in certified,
        "next_policy": (
            "preregister R256 one-asset one-style learner overfit" if (256 in certified and 128 not in certified)
            else "localize learner/optimizer on current R128 before changing resolution" if 128 in certified
            else "reopen output field representation only; do not reopen P ontology/V5 geometry"
        ),
        "resolution_summary": resolution_summary,
        "assets": all_results,
    }
    atomic_json(out, report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--membership", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-per-view", type=int, default=SAMPLES_PER_VIEW)
    a = ap.parse_args()
    rep = run(a.root, a.membership, a.out, a.samples_per_view)
    print(json.dumps({
        "status": rep["status"],
        "certified_field_hw": rep["certified_field_hw"],
        "smallest_certified_field_hw": rep["smallest_certified_field_hw"],
        "current_R2_certified": rep["current_R2_certified"],
        "full_R_certified": rep["full_R_certified"],
        "next_policy": rep["next_policy"],
        "worst_cell_P_p95_by_hw": {k:v["worst_cell_P_p95"] for k,v in rep["resolution_summary"].items()},
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
