#!/usr/bin/env python3
"""Execute DINOV2_V31_FUSION_FORM_PREREG_V1 exactly."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from models.iris.v2.dinov2_foundation_v2 import load_exact_dinov2_s_v2
from experiments.iris_tessa_v3_apparatus_audit_20261025.dinov2_alpha_residual_probe_v1 import (
    SEEDS, STEPS, LR, WIDTHS, ALPHA_BITS, ALPHA_CELLS, build_features,
)

SCHEMA = "RealSaS.DINOv2V31FusionFormCourt.v1"


class LearnedConcat32012864(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_branch = nn.Linear(384, 320)
        self.detail_branch = nn.Linear(384, 128)
        self.alpha_branch = nn.Linear(ALPHA_CELLS, 64)
    def forward(self, l2, final, alpha_raw, alpha_mean):
        del alpha_mean
        return torch.cat((self.final_branch(final), self.detail_branch(l2), self.alpha_branch(alpha_raw)), dim=-1)


class DirectFinal384L264Alpha64(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail_branch = nn.Linear(384, 64)
        self.alpha_branch = nn.Linear(ALPHA_CELLS, 64)
    def forward(self, l2, final, alpha_raw, alpha_mean):
        del alpha_mean
        return torch.cat((final, self.detail_branch(l2), self.alpha_branch(alpha_raw)), dim=-1)


class CourtModel(nn.Module):
    def __init__(self, arm: nn.Module):
        super().__init__(); self.arm = arm
        self.local = nn.Linear(512, 4)
        self.recon = nn.Linear(512, 384)
        self.alpha = nn.Linear(512, ALPHA_BITS)
    def forward(self, l2, final, alpha_raw, alpha_mean):
        z = self.arm(l2, final, alpha_raw, alpha_mean)
        return self.local(z), self.recon(z), self.alpha(z)


def adapter_params(arm: nn.Module) -> int:
    return sum(p.numel() for p in arm.parameters() if p.requires_grad)


def train_eval(seed: int, arm_name: str, arm_template: nn.Module, data):
    torch.manual_seed(seed)
    model = CourtModel(copy.deepcopy(arm_template))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    keys = ("train_l2", "train_final", "train_local_y", "train_alpha_raw", "train_alpha_mean", "train_alpha_y")
    merged = {key: torch.cat([data[w][key] for w in WIDTHS]) for key in keys}
    for _ in range(STEPS):
        local_logits, final_pred, alpha_logits = model(
            merged["train_l2"], merged["train_final"], merged["train_alpha_raw"], merged["train_alpha_mean"]
        )
        loss = (
            F.binary_cross_entropy_with_logits(local_logits, merged["train_local_y"])
            + (1.0 - F.cosine_similarity(final_pred, merged["train_final"], dim=-1).mean())
            + F.binary_cross_entropy_with_logits(alpha_logits, merged["train_alpha_y"])
        )
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval(); rows = []
    with torch.no_grad():
        for width in WIDTHS:
            d = data[width]
            local_logits, final_pred, alpha_logits = model(d["eval_l2"], d["eval_final"], d["eval_alpha_raw"], d["eval_alpha_mean"])
            rows.append({
                "seed": seed, "arm": arm_name, "width": width,
                "local_accuracy": float(((local_logits >= 0) == (d["eval_local_y"] >= 0.5)).float().mean()),
                "final_cosine": float(F.cosine_similarity(final_pred, d["eval_final"], dim=-1).mean()),
                "alpha_accuracy": float(((alpha_logits >= 0) == (d["eval_alpha_y"] >= 0.5)).float().mean()),
            })
    return rows


def run(source: Path, weights: Path) -> dict:
    torch.set_num_threads(1)
    foundation, seal = load_exact_dinov2_s_v2(source, weights, device="cpu")
    data, identity = build_features(foundation)
    arms = {"LEARNED_CONCAT_320_128_64": LearnedConcat32012864(), "DIRECT_FINAL_384_L2_64_ALPHA64": DirectFinal384L264Alpha64()}
    counts = {name: adapter_params(arm) for name, arm in arms.items()}
    rows = []
    for seed in SEEDS:
        for name, arm in arms.items(): rows += train_eval(seed, name, arm, data)
    def vals(arm, key): return [float(r[key]) for r in rows if r["arm"] == arm]
    summary = {}
    for arm in arms:
        summary[arm] = {
            "mean_local": sum(vals(arm, "local_accuracy")) / len(vals(arm, "local_accuracy")),
            "min_local": min(vals(arm, "local_accuracy")),
            "mean_final_cosine": sum(vals(arm, "final_cosine")) / len(vals(arm, "final_cosine")),
            "min_final_cosine": min(vals(arm, "final_cosine")),
            "mean_alpha": sum(vals(arm, "alpha_accuracy")) / len(vals(arm, "alpha_accuracy")),
            "min_alpha": min(vals(arm, "alpha_accuracy")),
        }
    c = summary["LEARNED_CONCAT_320_128_64"]; d = summary["DIRECT_FINAL_384_L2_64_ALPHA64"]
    concat_valid = identity <= 1e-6 and c["mean_local"] >= .995 and c["min_local"] >= .98 and c["mean_alpha"] >= .995 and c["min_alpha"] >= .98 and c["mean_final_cosine"] >= .99
    direct_valid = (
        d["mean_local"] >= .995 and d["min_local"] >= .98
        and d["mean_alpha"] >= .995 and d["min_alpha"] >= .98
        and d["mean_final_cosine"] >= .995 and d["min_final_cosine"] >= .99
        and d["mean_final_cosine"] >= c["mean_final_cosine"] - .002
        and counts["DIRECT_FINAL_384_L2_64_ALPHA64"] < counts["LEARNED_CONCAT_320_128_64"]
    )
    if not concat_valid:
        status = "DINO_V31_FUSION_COURT_INVALID__REFERENCE_DRIFT"; selected = None
    elif direct_valid:
        status = "DIRECT_FINAL_384_L2_64_ALPHA64_SELECTED_TINY_V1"; selected = "DIRECT_FINAL_384_L2_64_ALPHA64"
    else:
        status = "LEARNED_CONCAT_320_128_64_RETAINED_TINY_V1"; selected = "LEARNED_CONCAT_320_128_64"
    return {
        "schema": SCHEMA, "status": status, "selected_arm": selected,
        "authority": {"source_head": seal.source_head, "weight_sha256": seal.weight_sha256},
        "identity_error": identity, "adapter_parameter_counts": counts, "summary": summary, "results": rows,
        "gates": {"concat_reference_valid": concat_valid, "direct_final_valid": direct_valid},
        "execution": {"device": "cpu", "steps": STEPS, "lr": LR},
        "claims": {"product_authority_changed": False, "dinov3_adoption_authorized": False, "truth_builder_training_authorized": False, "fit8_training_authorized": False},
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dinov2-source", type=Path, required=True); p.add_argument("--dinov2-weights", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    r = run(a.dinov2_source, a.dinov2_weights); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": r["status"], "selected_arm": r["selected_arm"], "summary": r["summary"], "adapter_parameter_counts": r["adapter_parameter_counts"], "out": str(a.out)}, sort_keys=True))


if __name__ == "__main__": main()
