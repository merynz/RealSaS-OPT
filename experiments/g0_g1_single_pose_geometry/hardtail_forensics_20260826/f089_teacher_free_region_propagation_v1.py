#!/usr/bin/env python3
"""
RealSaS exploratory CPU oracle for asset_f089abadcd071194617d640b.

Inference-side evidence:
- 8 x cel_clean 512 RGBA images
- known orthographic 8-yaw camera contract (half_extent=0.54)

Forbidden from inference:
- teacher triangle IDs
- teacher visibility
- teacher geometry

Teacher raster/geometry are used only for evaluation.
This script intentionally preserves an exploratory, not preregistered treatment.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

H = W = 512
HALF_EXTENT = 0.54
DEPTHS = np.linspace(-HALF_EXTENT, HALF_EXTENT, 321)
QUAD_FACE_IDS = np.array([776, 777], dtype=np.int64)


def camera_bases():
    yaws = np.arange(8) * 45.0
    rights, forwards, ups = [], [], []
    for yaw in yaws:
        th = np.deg2rad(yaw)
        rights.append([np.cos(th), -np.sin(th), 0.0])
        forwards.append([np.sin(th), np.cos(th), 0.0])
        ups.append([0.0, 0.0, 1.0])
    return np.asarray(rights), np.asarray(forwards), np.asarray(ups)


def load_inputs(root: Path):
    imgs = []
    for v in range(8):
        p = root / f"f089_V{v}_cel.png"
        imgs.append(np.asarray(Image.open(p)).astype(np.float32) / 255.0)
    geom = np.load(root / "primary_geometry.npz")
    return np.stack(imgs), geom


def raster_truth(root: Path, view: int, vertices, faces):
    r = np.load(root / f"f089_V{view}_raster.npz")
    inds = r["pixel_linear_index"].astype(np.int64)
    tri = r["triangle_id"].astype(np.int64)
    uv = r["barycentric_uv"].astype(np.float64)
    f = faces[tri]
    # raster_authority convention: weights=(u, v, 1-u-v) for face vertices (v0,v1,v2)
    w = np.stack([uv[:, 0], uv[:, 1], 1.0 - uv[:, 0] - uv[:, 1]], axis=1)
    P = (w[:, :, None] * vertices[f]).sum(axis=1)
    return inds, tri, P


def discover_uniform_region(img: np.ndarray, tol: int = 6):
    arr = np.rint(img * 255.0).astype(np.int16)
    rgb = arr[..., :3]
    fg = arr[..., 3] > 0
    vals = rgb[fg]
    uniq, cnt = np.unique(vals.reshape(-1, 3), axis=0, return_counts=True)
    mode = uniq[cnt.argmax()]
    dist = np.max(np.abs(rgb - mode), axis=2)
    cand = fg & (dist <= tol)
    cand = ndi.binary_closing(cand, structure=np.ones((3, 3)), iterations=1)
    lab, n = ndi.label(cand, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(fg), mode, {"rectangularity": 0.0, "fg_frac": 0.0}
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    comp = lab == sizes.argmax()
    ys, xs = np.where(comp)
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return comp, mode, {
        "rectangularity": float(comp.sum() / bbox_area),
        "fg_frac": float(comp.sum() / max(1, fg.sum())),
    }


def bilinear_sample(im: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    h, w, c = im.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    out = np.zeros((len(xs), c), dtype=np.float64)
    if valid.any():
        xv, yv = xs[valid], ys[valid]
        xa, xb, ya, yb = x0[valid], x1[valid], y0[valid], y1[valid]
        wx, wy = xv - xa, yv - ya
        out[valid] = (
            im[ya, xa] * (1 - wx)[:, None] * (1 - wy)[:, None]
            + im[ya, xb] * wx[:, None] * (1 - wy)[:, None]
            + im[yb, xa] * (1 - wx)[:, None] * wy[:, None]
            + im[yb, xb] * wx[:, None] * wy[:, None]
        )
    return out, valid


def per_view_costs(imgs, rights, forwards, ups, ref_view, x512, y512, depths):
    n, D = len(x512), len(depths)
    sr = HALF_EXTENT * ((x512 + 0.5) / W * 2.0 - 1.0)
    su = HALF_EXTENT * (1.0 - (y512 + 0.5) / H * 2.0)
    ref_rgb = imgs[ref_view, y512, x512, :3].astype(np.float64)
    P = (
        sr[:, None, None] * rights[ref_view][None, None, :]
        + su[:, None, None] * ups[ref_view][None, None, :]
        + depths[None, :, None] * forwards[ref_view][None, None, :]
    )
    per = []
    for tv in range(8):
        if tv == ref_view:
            continue
        tr = P @ rights[tv]
        tu = P[:, :, 2]
        xs = (tr / HALF_EXTENT + 1.0) * 0.5 * W - 0.5
        ys = (1.0 - tu / HALF_EXTENT) * 0.5 * H - 0.5
        samp, valid = bilinear_sample(imgs[tv], xs.ravel(), ys.ravel())
        samp = samp.reshape(n, D, 4)
        valid = valid.reshape(n, D)
        rgb = np.mean(np.abs(samp[:, :, :3] - ref_rgb[:, None, :]), axis=2)
        alpha = samp[:, :, 3]
        c = rgb + 0.75 * np.maximum(0.0, 0.5 - alpha) * 2.0
        c[~valid] = 1.5
        per.append(c)
    return np.stack(per, axis=2)


def ransac_plane(sr, su, d, conf, seed, thr=0.008, iters=2500):
    rng = np.random.default_rng(seed)
    X = np.stack([sr, su, np.ones_like(sr)], axis=1)
    probs = np.maximum(conf, 1e-8)
    probs = probs / probs.sum()
    best = None
    for _ in range(iters):
        idx = rng.choice(len(d), 3, replace=False, p=probs)
        try:
            beta = np.linalg.solve(X[idx], d[idx])
        except np.linalg.LinAlgError:
            continue
        resid = np.abs(X @ beta - d)
        inl = resid < thr
        score = (int(inl.sum()), float(conf[inl].sum()), -float(np.median(resid[inl])) if inl.any() else -1e9)
        if best is None or score > best[0]:
            best = (score, beta, inl)
    beta, inl = best[1], best[2]
    Xi, yi = X[inl], d[inl]
    wi = np.maximum(conf[inl], 1e-6)
    beta = np.linalg.lstsq(Xi * wi[:, None] ** 0.5, yi * wi ** 0.5, rcond=None)[0]
    return beta, inl


def teacher_quad_mask_512(root: Path, view: int):
    r = np.load(root / f"f089_V{view}_raster.npz")
    inds = r["pixel_linear_index"].astype(np.int64)
    tri = r["triangle_id"].astype(np.int64)
    mask1024 = np.zeros((1024, 1024), dtype=bool)
    rows, cols = inds // 1024, inds % 1024
    q = np.isin(tri, QUAD_FACE_IDS)
    mask1024[rows[q], cols[q]] = True
    return mask1024.reshape(512, 2, 512, 2).any(axis=(1, 3))


def q(a, p):
    return float(np.quantile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("F089_TEACHER_FREE_REGION_PROPAGATION_V1_METRICS.json"))
    args = ap.parse_args()

    imgs, geom = load_inputs(args.root)
    vertices = geom["vertices"].astype(np.float64)
    faces = geom["faces"].astype(np.int64)
    rights, forwards, ups = camera_bases()

    region_iou = {}
    discovered = {}
    for v in range(8):
        mask, mode, meta = discover_uniform_region(imgs[v])
        gt = teacher_quad_mask_512(args.root, v)
        inter = int((mask & gt).sum())
        union = int((mask | gt).sum())
        region_iou[f"V{v}"] = float(inter / union) if union else 1.0
        discovered[v] = (mask, mode, meta)

    results = []
    for v in [1, 2, 3, 5, 6, 7]:
        mask, mode, meta = discovered[v]
        inds, tri, Ptrue = raster_truth(args.root, v, vertices, faces)
        row, col = inds // 1024, inds % 1024
        xs, ys = (col // 2).astype(np.int64), (row // 2).astype(np.int64)
        cand = np.where(mask[ys, xs])[0]
        rng = np.random.default_rng(20260826 + v)
        si = rng.choice(cand, min(1400, len(cand)), replace=False)
        sx, sy = xs[si], ys[si]

        per = per_view_costs(imgs, rights, forwards, ups, v, sx, sy, DEPTHS)
        C = per.mean(axis=2)  # all seven views: no teacher visibility selection
        bi = C.argmin(axis=1)
        bd = DEPTHS[bi]
        conf = np.empty(len(si), dtype=np.float64)
        for j, b in enumerate(bi):
            far = np.abs(DEPTHS - DEPTHS[b]) >= 0.012
            conf[j] = C[j, far].min() - C[j, b]

        order = np.argsort(-conf)
        sel = []
        for oi in order:
            if conf[oi] <= 0:
                break
            if all((sx[oi] - sx[s]) ** 2 + (sy[oi] - sy[s]) ** 2 >= 10 ** 2 for s in sel):
                sel.append(int(oi))
            if len(sel) >= 80:
                break
        sel = np.asarray(sel, dtype=np.int64)

        sr = HALF_EXTENT * ((sx + 0.5) / 512.0 * 2.0 - 1.0)
        su = HALF_EXTENT * (1.0 - (sy + 0.5) / 512.0 * 2.0)
        beta, inl = ransac_plane(sr[sel], su[sel], bd[sel], conf[sel], seed=303000 + v)
        pred = np.stack([sr, su, np.ones_like(sr)], axis=1) @ beta
        td = Ptrue[si] @ forwards[v]
        ep = np.abs(bd - td)
        epl = np.abs(pred - td)
        gt = np.isin(tri[si], QUAD_FACE_IDS)

        results.append({
            "view": v,
            "modal_rgb": mode.tolist(),
            "region_rectangularity": meta["rectangularity"],
            "region_foreground_fraction": meta["fg_frac"],
            "discovered_visible_rows": int(len(cand)),
            "sample_n": int(len(si)),
            "teacher_plane_fraction_in_discovered_sample": float(gt.mean()),
            "seed_n": int(len(sel)),
            "seed_good_lt005_eval": float((np.abs(bd[sel] - td[sel]) < 0.005).mean()),
            "ransac_seed_inlier_rate": float(inl.mean()),
            "point_p50": q(ep, 0.50),
            "point_p90": q(ep, 0.90),
            "point_p95": q(ep, 0.95),
            "prop_p50": q(epl, 0.50),
            "prop_p90": q(epl, 0.90),
            "prop_p95": q(epl, 0.95),
            "prop_gtplane_p90_eval": q(epl[gt], 0.90),
            "plane_beta": beta.tolist(),
        })

    qf = faces[QUAD_FACE_IDS]
    qverts = np.unique(qf)
    skin = geom["skin"]
    areas = []
    normals = []
    for f in qf:
        a, b, c = vertices[f]
        cr = np.cross(b - a, c - a)
        areas.append(float(np.linalg.norm(cr) / 2.0))
        normals.append((cr / np.linalg.norm(cr)).tolist())

    out = {
        "schema": "RealSaS.IRIS.PV5R256.F089TeacherFreeRegionPropagation.v1",
        "status": "EXPLORATORY_POSITIVE__NOT_FORMAL_GATE",
        "asset_id": "asset_f089abadcd071194617d640b",
        "optimizer_steps": 0,
        "teacher_plane_membership_used_for_inference": False,
        "teacher_visibility_used_for_inference": False,
        "region_iou_vs_teacher_eval": region_iou,
        "all_view_mean_results": results,
        "detached_quad_component": {
            "vertex_indices": qverts.tolist(),
            "face_indices": QUAD_FACE_IDS.tolist(),
            "skin_weight_sum_per_vertex": [float(skin[i].sum()) for i in qverts],
            "face_area": areas,
            "unit_normals": normals,
            "semantic_status": "UNKNOWN__ANCILLARY_STATIC_CARD_OR_VALID_STATIC_GEOMETRY_NOT_RESOLVED",
        },
    }
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
