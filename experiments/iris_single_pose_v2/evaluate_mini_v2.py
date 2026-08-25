from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from dataset import IRISV2Dataset
from evaluate_v2 import aggregate_asset_queries, evaluate_correspondence, geometry_metrics, model_batch, spearman, risk_coverage_error_auc
from matcher import MatcherConfig
from model import IRISSinglePoseV2, IRISV2Config


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def summary(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], np.float64)
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "p95": None}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
    }


def bool_mean(rows, key):
    vals = [bool(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def error_summary(rows, key):
    return summary([float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))])


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = IRISSinglePoseV2(IRISV2Config())
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return model, ckpt


def _evaluation_forward(model, batch, device):
    """Run the network with AMP, then leave the evaluator/matcher on explicit FP32 tensors.

    CUDA autocast may return FP16 feature fields. The observable evaluator uses grid_sample,
    geometric distances and deterministic matcher arithmetic outside the autocast region, where
    mixing FP16 fields with FP32 query grids is not a legal/portable contract. Casting once here
    makes evaluation semantics explicit and stable while leaving the production forward AMP path intact.
    """
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
    ):
        raw = model(batch["images"], batch["yaw_deg"])
    out = {k: v.float() for k, v in raw.items()}
    if any(v.dtype != torch.float32 for v in out.values()):
        raise RuntimeError("mini evaluator failed to normalize model outputs to FP32")
    return out


def evaluate_style(model, cache_manifest, split, style, device, query_limit, qualification_limit, track_samples):
    ds = IRISV2Dataset(cache_manifest, split=split, track_samples=track_samples, style_mode=style)
    asset_rows, all_q, p_all, n_all, risk_all, p_for_risk = [], [], [], [], [], []
    cfg = MatcherConfig()
    for i in range(len(ds)):
        sample = ds[i]
        batch = model_batch(sample, device)
        outputs = _evaluation_forward(model, batch, device)
        gm = geometry_metrics(outputs, batch)
        qrows = evaluate_correspondence(outputs, batch, cfg, query_limit, qualification_limit)
        qa = aggregate_asset_queries(qrows)
        asset_rows.append(
            {
                "asset_id": sample["asset_id"],
                "style": style,
                "P_p95": gm["P"]["p95"],
                "N_p95_deg": gm["N_deg"]["p95"],
                "Zc_top8": qa.get("Zc_top8_exact_cell"),
                "P_basin_top8": qa.get("P_top8_exact_cell"),
                "oracle_Zf_top1_p95_native_px": qa.get("oracle_Zf_top1_error_native_px", {}).get("p95"),
                "end_to_end_top8_hit_16px": qa.get("end_to_end_top8_hit_16px"),
                "reciprocal_support_rate": qa.get("truth_candidate_reciprocal_support_rate"),
                "query_count": qa.get("query_count", 0),
            }
        )
        all_q.extend(qrows)
        p_all.extend(gm["_p_error"].tolist())
        n_all.extend(gm["_n_error"].tolist())
        risk_all.extend(gm["_risk"].tolist())
        p_for_risk.extend(gm["_p_error"].tolist())
        if (i + 1) % 8 == 0 or i + 1 == len(ds):
            print(f"[mini-eval:{style}] {i+1}/{len(ds)}", flush=True)
    pooled = {
        "asset_count": len(asset_rows),
        "P": summary(p_all),
        "N_deg": summary(n_all),
        "U_error_spearman": spearman(risk_all, p_for_risk),
        "U_risk_coverage_error_auc": risk_coverage_error_auc(risk_all, p_for_risk),
        "queries": len(all_q),
        "Zc_top1": bool_mean(all_q, "Zc_top1_exact_cell"),
        "Zc_top4": bool_mean(all_q, "Zc_top4_exact_cell"),
        "Zc_top8": bool_mean(all_q, "Zc_top8_exact_cell"),
        "Zc_top16": bool_mean(all_q, "Zc_top16_exact_cell"),
        "Zc_top32": bool_mean(all_q, "Zc_top32_exact_cell"),
        "P_basin_top8": bool_mean(all_q, "P_top8_exact_cell"),
        "fused_basin_top8": bool_mean(all_q, "fused_basin_top8_exact_cell"),
        "oracle_Zf_top1_error_native_px": error_summary(all_q, "oracle_Zf_top1_error_native_px"),
        "end_to_end_top1_error_native_px": error_summary(all_q, "end_to_end_top1_error_native_px"),
        "end_to_end_top8_hit_16px": bool_mean(all_q, "end_to_end_top8_hit_16px"),
        "reciprocal_support_rate": bool_mean(
            [r for r in all_q if r.get("truth_candidate_reciprocal_supported") is not None],
            "truth_candidate_reciprocal_supported",
        ),
    }
    family = {
        "P_p95_across_assets": summary([r["P_p95"] for r in asset_rows]),
        "N_p95_across_assets": summary([r["N_p95_deg"] for r in asset_rows]),
        "Zc_top8_across_assets": summary([r["Zc_top8"] for r in asset_rows]),
        "P_basin_top8_across_assets": summary([r["P_basin_top8"] for r in asset_rows]),
        "oracle_Zf_top1_p95_across_assets": summary([r["oracle_Zf_top1_p95_native_px"] for r in asset_rows]),
    }
    return {"pooled": pooled, "family": family, "assets": asset_rows}


def macro_metric(styles, getter):
    vals = []
    for s in styles.values():
        x = getter(s)
        if x is not None and math.isfinite(float(x)):
            vals.append(float(x))
    return float(np.mean(vals)) if vals else None


def build_report(model, ckpt_meta, cache_manifest, split, device, query_limit, qualification_limit, track_samples):
    styles = {
        style: evaluate_style(
            model,
            cache_manifest,
            split,
            style,
            device,
            query_limit,
            qualification_limit,
            track_samples,
        )
        for style in ("cel_clean", "ink_cel")
    }
    selection = {
        "P_p95": macro_metric(styles, lambda s: s["pooled"]["P"]["p95"]),
        "N_p95_deg": macro_metric(styles, lambda s: s["pooled"]["N_deg"]["p95"]),
        "Zc_top8": macro_metric(styles, lambda s: s["pooled"]["Zc_top8"]),
        "P_basin_top8": macro_metric(styles, lambda s: s["pooled"]["P_basin_top8"]),
        "oracle_Zf_top1_p95_native_px": macro_metric(
            styles, lambda s: s["pooled"]["oracle_Zf_top1_error_native_px"]["p95"]
        ),
        "end_to_end_top8_hit_16px": macro_metric(styles, lambda s: s["pooled"]["end_to_end_top8_hit_16px"]),
        "family_Zc_top8_p10": macro_metric(styles, lambda s: s["family"]["Zc_top8_across_assets"]["p10"]),
        "family_P_p95_p90": macro_metric(styles, lambda s: s["family"]["P_p95_across_assets"]["p90"]),
        "min_style_Zc_top8": min(styles[x]["pooled"]["Zc_top8"] for x in styles),
        "max_style_P_p95": max(styles[x]["pooled"]["P"]["p95"] for x in styles),
    }
    return {
        "schema": "RealSaS.IRISSinglePoseV2.MiniEvaluator.v1",
        "checkpoint_epoch": int(ckpt_meta.get("epoch", -1)),
        "checkpoint_optimizer_steps": int(ckpt_meta.get("optimizer_steps", -1)),
        "split": split,
        "style_policy": "cel_clean + ink_cel; equal macro",
        "query_limit_per_asset_per_style": query_limit,
        "qualification_limit_per_asset_per_style": qualification_limit,
        "track_samples": track_samples,
        "styles": styles,
        "selection_metrics": selection,
        "evaluation_precision_policy": "model forward under CUDA FP16 autocast; all exported evidence fields cast to FP32 before geometry/matcher evaluation",
        "normal_semantics": "N is local raster-observation surface orientation only; never a correspondence admission/ranking feature.",
    }


def main():
    ap = argparse.ArgumentParser(description="Frozen mini learner evaluator; no checkpoint selection side effects.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--split", choices=("FIT", "TUNE"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--query-limit", type=int, default=24)
    ap.add_argument("--qualification-limit", type=int, default=8)
    ap.add_argument("--track-samples", type=int, default=512)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for mini evaluation")
    device = torch.device("cuda")
    model, ckpt = load_checkpoint(a.checkpoint, device)
    out = build_report(
        model,
        ckpt,
        a.cache_manifest,
        a.split,
        device,
        a.query_limit,
        a.qualification_limit,
        a.track_samples,
    )
    out["checkpoint"] = str(Path(a.checkpoint).resolve())
    atomic_json(a.out, out)
    print(json.dumps({"split": a.split, "epoch": out["checkpoint_epoch"], **out["selection_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
