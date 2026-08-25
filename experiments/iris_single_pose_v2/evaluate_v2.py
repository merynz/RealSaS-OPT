from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from coords import grid_to_cell_index, cell_index_to_grid
from dataset import IRISV2Dataset
from losses import sample_field
from matcher import MatcherConfig, match_query, _sample_one, _alpha_mask_at, _norm
from metrics import native_pixel_error
from model import IRISSinglePoseV2, IRISV2Config, count_parameters
from qualification import exact_cell_rank, reciprocal_support, cycle_support_views

AUTHORITY_RESOLUTION = 1024
PAIR_CATEGORIES = {1: "adjacent", 2: "skip_one", 4: "opposite"}


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_seed(path):
    obj = json.load(open(path, encoding="utf-8"))
    rows = {r["asset_id"]: r for r in obj["records"]}
    return obj, rows


def _rankdata(x):
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return None
    rx = _rankdata(x[m])
    ry = _rankdata(y[m])
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def risk_coverage_error_auc(risk, error):
    risk = np.asarray(risk, np.float64)
    error = np.asarray(error, np.float64)
    m = np.isfinite(risk) & np.isfinite(error)
    if int(m.sum()) < 2:
        return None
    order = np.argsort(risk[m], kind="stable")
    e = error[m][order]
    cumulative = np.cumsum(e) / np.arange(1, len(e) + 1)
    return float(np.mean(cumulative))


def dist_summary(x):
    a = np.asarray(x, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
    }


def fraction(xs):
    xs = [bool(x) for x in xs]
    return float(np.mean(xs)) if xs else None


def circular_view_distance(a, b):
    d = abs(int(a) - int(b))
    return min(d, 8 - d)


def select_queries(track_visible: np.ndarray, per_asset: int):
    groups = {name: [] for name in PAIR_CATEGORIES.values()}
    tracks, views = track_visible.shape
    for t in range(tracks):
        vv = np.flatnonzero(track_visible[t])
        for i, s in enumerate(vv):
            for target in vv[i + 1 :]:
                cat = PAIR_CATEGORIES.get(circular_view_distance(s, target))
                if cat is not None:
                    groups[cat].append((t, int(s), int(target), cat))
    if per_asset <= 0:
        return [q for cat in ("adjacent", "skip_one", "opposite") for q in groups[cat]]
    selected = []
    quota = max(1, per_asset // 3)
    leftovers = []
    for cat in ("adjacent", "skip_one", "opposite"):
        g = groups[cat]
        if len(g) <= quota:
            selected.extend(g)
        elif g:
            ids = np.linspace(0, len(g) - 1, quota).round().astype(np.int64)
            selected.extend([g[i] for i in ids])
        leftovers.extend([q for q in g if q not in selected])
    if len(selected) < per_asset and leftovers:
        need = min(per_asset - len(selected), len(leftovers))
        ids = np.linspace(0, len(leftovers) - 1, need).round().astype(np.int64)
        selected.extend([leftovers[i] for i in ids])
    return selected[:per_asset]


def truth_basin_ranks(result, truth_xy):
    hc, wc = result.coarse_field_hw
    tx = int(grid_to_cell_index(float(truth_xy[0]), wc))
    ty = int(grid_to_cell_index(float(truth_xy[1]), hc))
    zrank = None
    prank = None
    fused = None
    for i, b in enumerate(result.basins, 1):
        bx = int(grid_to_cell_index(float(b.coarse_coord[0]), wc))
        by = int(grid_to_cell_index(float(b.coarse_coord[1]), hc))
        if bx == tx and by == ty:
            zrank = b.coarse_rank if zrank is None else min(zrank, b.coarse_rank)
            prank = b.p_rank if prank is None else min(prank, b.p_rank)
            fused = i if fused is None else min(fused, i)
    return zrank, prank, fused


def oracle_local_refine(outputs, images, source_view, target_view, query_xy, truth_xy, cfg: MatcherConfig, topk=8):
    """Evaluator-only local Zf test with the true coarse basin supplied as oracle authority."""
    zf = outputs["Z_fine"]
    hf, wf = zf.shape[-2:]
    hc, wc = outputs["Z_coarse"].shape[-2:]
    truth_xy = np.asarray(truth_xy, np.float32)
    cx = int(grid_to_cell_index(float(truth_xy[0]), wc))
    cy = int(grid_to_cell_index(float(truth_xy[1]), hc))
    coarse_center = np.asarray([float(cell_index_to_grid(cx, wc)), float(cell_index_to_grid(cy, hc))], np.float32)
    center_x = int(grid_to_cell_index(float(coarse_center[0]), wf))
    center_y = int(grid_to_cell_index(float(coarse_center[1]), hf))
    alpha_t = images[0, target_view, 3:4][None]
    fine_mask = _alpha_mask_at(alpha_t, (hf, wf), cfg.alpha_threshold)
    cells = []
    for y in range(max(0, center_y - cfg.fine_radius_cells), min(hf, center_y + cfg.fine_radius_cells + 1)):
        for x in range(max(0, center_x - cfg.fine_radius_cells), min(wf, center_x + cfg.fine_radius_cells + 1)):
            if fine_mask[y, x]:
                cells.append((y, x))
    if not cells:
        return np.zeros((0, 2), np.float32)
    coords = np.asarray(
        [[float(cell_index_to_grid(x, wf)), float(cell_index_to_grid(y, hf))] for y, x in cells],
        np.float32,
    )
    q = _norm(_sample_one(zf, source_view, np.asarray(query_xy, np.float32).reshape(1, 2)))[0]
    target = _norm(_sample_one(zf, target_view, coords))
    score = target @ q
    order = np.argsort(-score, kind="stable")[: min(topk, len(coords))]
    return coords[order]


def model_batch(sample, device):
    out = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0).to(device)
        else:
            out[k] = v
    return out


def geometry_metrics(outputs, batch):
    grid = batch["geom_xy"]
    mask = batch["geom_mask"].bool()
    gt_p = batch["geom_p"]
    gt_n = batch["geom_n"]
    pp = sample_field(outputs["P"], grid)
    pn = sample_field(outputs["N"], grid)
    u = sample_field(outputs["U_geo"], grid)[..., 0]
    perr = torch.linalg.norm(pp - gt_p, dim=-1)
    cos = (pn * gt_n).sum(-1).clamp(-1, 1)
    nang = torch.rad2deg(torch.acos(cos))
    pe = perr[mask].detach().float().cpu().numpy()
    na = nang[mask].detach().float().cpu().numpy()
    risk = u[mask].detach().float().cpu().numpy()
    return {
        "P": dist_summary(pe),
        "N_deg": dist_summary(na),
        "U_error_spearman": spearman(risk, pe),
        "U_risk_coverage_error_auc": risk_coverage_error_auc(risk, pe),
        "_p_error": pe,
        "_n_error": na,
        "_risk": risk,
    }


def evaluate_correspondence(outputs, batch, cfg, query_limit, qualification_limit):
    xy = batch["track_xy"][0].detach().cpu().numpy()
    vis = batch["track_visible"][0].detach().cpu().numpy().astype(bool)
    images = batch["images"]
    queries = select_queries(vis, query_limit)
    rows = []
    for qi, (track, source, target, category) in enumerate(queries):
        qxy = xy[track, source]
        truth = xy[track, target]
        res = match_query(outputs, images, source, target, qxy, cfg)
        zr, pr, fr = truth_basin_ranks(res, truth)
        top = res.top_coords
        errors = native_pixel_error(top, truth, AUTHORITY_RESOLUTION) if len(top) else np.zeros(0, np.float32)
        oracle = oracle_local_refine(outputs, images, source, target, qxy, truth, cfg, topk=8)
        oracle_err = native_pixel_error(oracle, truth, AUTHORITY_RESOLUTION) if len(oracle) else np.zeros(0, np.float32)
        fine_truth_rank = exact_cell_rank(top, truth, res.fine_field_hw, topk=8)
        reciprocal = None
        cycle_count = None
        if qi < qualification_limit and fine_truth_rank is not None:
            truth_candidate = top[fine_truth_rank - 1]
            reciprocal, _ = reciprocal_support(outputs, images, source, target, qxy, truth_candidate, cfg, reverse_topk=8)
            cycle_count = len(
                cycle_support_views(
                    outputs,
                    images,
                    source,
                    target,
                    qxy,
                    truth_candidate,
                    cfg,
                    cycle_branch_k=2,
                    return_topk=8,
                )
            )
        row = {
            "track": int(track),
            "source_view": int(source),
            "target_view": int(target),
            "pair_category": category,
            "Zc_truth_rank": zr,
            "P_truth_rank": pr,
            "fused_basin_truth_rank": fr,
            "end_to_end_top1_error_native_px": float(errors[0]) if len(errors) else None,
            "end_to_end_top4_hit_2px": bool(np.any(errors[:4] <= 2.0)) if len(errors) else False,
            "end_to_end_top4_hit_4px": bool(np.any(errors[:4] <= 4.0)) if len(errors) else False,
            "end_to_end_top8_hit_2px": bool(np.any(errors[:8] <= 2.0)) if len(errors) else False,
            "end_to_end_top8_hit_4px": bool(np.any(errors[:8] <= 4.0)) if len(errors) else False,
            "end_to_end_top8_hit_8px": bool(np.any(errors[:8] <= 8.0)) if len(errors) else False,
            "end_to_end_top8_hit_16px": bool(np.any(errors[:8] <= 16.0)) if len(errors) else False,
            "oracle_Zf_top1_error_native_px": float(oracle_err[0]) if len(oracle_err) else None,
            "oracle_Zf_top4_hit_2px": bool(np.any(oracle_err[:4] <= 2.0)) if len(oracle_err) else False,
            "oracle_Zf_top4_hit_4px": bool(np.any(oracle_err[:4] <= 4.0)) if len(oracle_err) else False,
            "truth_exact_fine_cell_rank": fine_truth_rank,
            "truth_candidate_reciprocal_supported": reciprocal,
            "truth_candidate_cycle_support_view_count": cycle_count,
        }
        for k in (1, 4, 8, 16, 32):
            row[f"Zc_top{k}_exact_cell"] = zr is not None and zr <= k
            row[f"fused_basin_top{k}_exact_cell"] = fr is not None and fr <= k
        for k in (1, 4, 8):
            row[f"P_top{k}_exact_cell"] = pr is not None and pr <= k
        rows.append(row)
    return rows


def aggregate_asset_queries(rows):
    if not rows:
        return {"query_count": 0}
    out = {"query_count": len(rows)}
    bool_keys = [k for k, v in rows[0].items() if isinstance(v, bool)]
    for k in bool_keys:
        out[k] = fraction(r[k] for r in rows)
    out["end_to_end_top1_error_native_px"] = dist_summary([r["end_to_end_top1_error_native_px"] for r in rows if r["end_to_end_top1_error_native_px"] is not None])
    out["oracle_Zf_top1_error_native_px"] = dist_summary([r["oracle_Zf_top1_error_native_px"] for r in rows if r["oracle_Zf_top1_error_native_px"] is not None])
    qrows = [r for r in rows if r["truth_candidate_reciprocal_supported"] is not None]
    out["qualification_query_count"] = len(qrows)
    out["truth_candidate_reciprocal_support_rate"] = fraction(r["truth_candidate_reciprocal_supported"] for r in qrows) if qrows else None
    out["truth_candidate_cycle_support_views"] = dist_summary([r["truth_candidate_cycle_support_view_count"] for r in qrows if r["truth_candidate_cycle_support_view_count"] is not None])
    by_pair = {}
    for cat in PAIR_CATEGORIES.values():
        rr = [r for r in rows if r["pair_category"] == cat]
        if rr:
            by_pair[cat] = {
                "n": len(rr),
                "Zc_top8_exact_cell": fraction(r["Zc_top8_exact_cell"] for r in rr),
                "end_to_end_top8_hit_4px": fraction(r["end_to_end_top8_hit_4px"] for r in rr),
                "oracle_Zf_top4_hit_4px": fraction(r["oracle_Zf_top4_hit_4px"] for r in rr),
            }
    out["by_pair_category"] = by_pair
    return out


def family_aggregate(asset_rows):
    if not asset_rows:
        return {"asset_count": 0}
    def vals(path):
        out = []
        for r in asset_rows:
            x = r
            for p in path:
                x = x.get(p) if isinstance(x, dict) else None
                if x is None:
                    break
            if isinstance(x, (int, float)) and math.isfinite(float(x)):
                out.append(float(x))
        return out
    return {
        "asset_count": len(asset_rows),
        "P_p95_family": dist_summary(vals(("geometry", "P", "p95"))),
        "N_p95_deg_family": dist_summary(vals(("geometry", "N_deg", "p95"))),
        "U_error_spearman_family": dist_summary(vals(("geometry", "U_error_spearman"))),
        "Zc_top8_exact_cell_family": dist_summary(vals(("correspondence", "Zc_top8_exact_cell"))),
        "end_to_end_top8_hit_4px_family": dist_summary(vals(("correspondence", "end_to_end_top8_hit_4px"))),
        "oracle_Zf_top4_hit_4px_family": dist_summary(vals(("correspondence", "oracle_Zf_top4_hit_4px"))),
        "end_to_end_top1_p95_native_px_family": dist_summary(vals(("correspondence", "end_to_end_top1_error_native_px", "p95"))),
        "oracle_Zf_top1_p95_native_px_family": dist_summary(vals(("correspondence", "oracle_Zf_top1_error_native_px", "p95"))),
    }


def load_model(checkpoint_path, device):
    ck = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ck.get("model_config", {}) if isinstance(ck, dict) else {}
    cfg = IRISV2Config(**cfg_dict) if cfg_dict else IRISV2Config()
    model = IRISSinglePoseV2(cfg).to(device)
    state = ck.get("model", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def evaluate_style(model, cache_manifest, seed_rows, split, style, device, track_samples, query_limit, qualification_limit, cfg):
    ds = IRISV2Dataset(cache_manifest, split=split, track_samples=track_samples, style_mode=style)
    ds.set_epoch(0)
    assets = []
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            batch = model_batch(sample, device)
            outputs = model(batch["images"], batch["yaw_deg"])
            gm = geometry_metrics(outputs, batch)
            qrows = evaluate_correspondence(outputs, batch, cfg, query_limit, qualification_limit)
            corr = aggregate_asset_queries(qrows)
            seed = seed_rows.get(sample["asset_id"], {})
            assets.append(
                {
                    "asset_id": sample["asset_id"],
                    "source_registry_id": seed.get("source_registry_id", "UNKNOWN"),
                    "candidate_id": seed.get("candidate_id"),
                    "geometry": {k: v for k, v in gm.items() if not k.startswith("_")},
                    "correspondence": corr,
                    "query_rows": qrows,
                }
            )
            if (i + 1) % 5 == 0 or i + 1 == len(ds):
                partial = family_aggregate(assets)
                print(f"[eval-v2:{style}] {i+1}/{len(ds)} Zc8_family_median={partial['Zc_top8_exact_cell_family']['median']}", flush=True)
    by_source = {}
    groups = defaultdict(list)
    for r in assets:
        groups[r["source_registry_id"]].append(r)
    for source, rows in sorted(groups.items()):
        by_source[source] = family_aggregate(rows)
    return {"style": style, "family": family_aggregate(assets), "by_source": by_source, "assets": assets}


def main():
    ap = argparse.ArgumentParser(description="Observable/native-unit IRIS Single-Pose V2 evaluator")
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--seed-manifest", required=True, help="required for source-stratified metrics")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="TUNE", choices=("FIT", "TUNE"))
    ap.add_argument("--styles", default="cel_clean,ink_cel")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--track-samples", type=int, default=256)
    ap.add_argument("--queries-per-asset", type=int, default=48)
    ap.add_argument("--qualification-queries-per-asset", type=int, default=6)
    a = ap.parse_args()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    seed_obj, seed_rows = load_seed(a.seed_manifest)
    model, model_cfg = load_model(a.checkpoint, device)
    matcher_cfg = MatcherConfig()
    styles = [s.strip() for s in a.styles.split(",") if s.strip()]
    if set(styles) - {"cel_clean", "ink_cel"}:
        raise ValueError(styles)
    result = []
    for style in styles:
        result.append(
            evaluate_style(
                model,
                a.cache_manifest,
                seed_rows,
                a.split,
                style,
                device,
                a.track_samples,
                a.queries_per_asset,
                a.qualification_queries_per_asset,
                matcher_cfg,
            )
        )
    report = {
        "schema": "RealSaS.IRISSinglePoseV2.ObservableEval.v1",
        "evaluation_mode": "GT_QUERY_CONDITIONED__OBSERVATION_DERIVED_DENSE_CANDIDATE_UNIVERSE",
        "split": a.split,
        "sealed_opened": False,
        "authority_resolution": AUTHORITY_RESOLUTION,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(a.checkpoint),
        "cache_manifest_sha256": sha256_file(a.cache_manifest),
        "seed_manifest_sha256": sha256_file(a.seed_manifest),
        "model_config": model_cfg.__dict__,
        "parameters": count_parameters(model),
        "matcher_config": matcher_cfg.__dict__,
        "queries_per_asset": a.queries_per_asset,
        "qualification_queries_per_asset": a.qualification_queries_per_asset,
        "styles": result,
        "selection_policy": "NOT_DEFINED_HERE__LOSS_SCALAR_FORBIDDEN__FREEZE_IN_PREREG_AFTER_GATE_CLOSURE",
        "interpretation": {
            "Zc": "global high-recall basin containment",
            "P": "independent common-frame rescue/address evidence",
            "Zf": "oracle-basin local precision and end-to-end within admitted basins only",
            "reciprocal_cycle": "deterministic support evidence; no hard candidate deletion or singleton authorization",
            "U_geo": "geometry risk only; not correspondence confidence",
        },
    }
    atomic_json(a.out, report)
    print(json.dumps({"schema": report["schema"], "split": a.split, "styles": [{"style": r["style"], "family": r["family"]} for r in result]}, indent=2))


if __name__ == "__main__":
    main()
