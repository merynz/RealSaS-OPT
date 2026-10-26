#!/usr/bin/env python3
"""Execute DINOV2_ALPHA_RESIDUAL_PREREG_V1 exactly."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from models.iris.v2.dinov2_foundation_v2 import load_exact_dinov2_s_v2
from experiments.iris_tessa_v3_apparatus_audit_20261025.dinov2_feature_access_probe_v1 import (
    WIDTHS,
    TRAIN_CANVASES,
    EVAL_CANVASES,
    MASTER_SEED,
    make_canvases,
    extract_all_layers,
)

SCHEMA = "RealSaS.DINOv2AlphaResidualCourt.v1"
SEEDS = (731, 947, 1217, 1597, 2017)
STEPS = 1200
LR = 1.0e-3
CURRENT_ADAPTER_PARAMS = 197120
ALPHA_BITS = 32
ALPHA_CELLS = 14 * 14
ALPHA_PAIR_SEED = 139240


class DualNoAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_branch = nn.Linear(384, 384)
        self.detail_branch = nn.Linear(384, 128)

    def forward(self, l2, final, alpha_raw, alpha_mean):
        del alpha_raw, alpha_mean
        return torch.cat((self.final_branch(final), self.detail_branch(l2)), dim=-1)


class DualMeanAlpha64(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_branch = nn.Linear(384, 320)
        self.detail_branch = nn.Linear(384, 128)
        self.alpha_branch = nn.Linear(1, 64)

    def forward(self, l2, final, alpha_raw, alpha_mean):
        del alpha_raw
        return torch.cat(
            (
                self.final_branch(final),
                self.detail_branch(l2),
                self.alpha_branch(alpha_mean),
            ),
            dim=-1,
        )


class DualRawAlpha64(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_branch = nn.Linear(384, 320)
        self.detail_branch = nn.Linear(384, 128)
        self.alpha_branch = nn.Linear(ALPHA_CELLS, 64)

    def forward(self, l2, final, alpha_raw, alpha_mean):
        del alpha_mean
        return torch.cat(
            (
                self.final_branch(final),
                self.detail_branch(l2),
                self.alpha_branch(alpha_raw),
            ),
            dim=-1,
        )


class DualRawAlpha96(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_branch = nn.Linear(384, 288)
        self.detail_branch = nn.Linear(384, 128)
        self.alpha_branch = nn.Linear(ALPHA_CELLS, 96)

    def forward(self, l2, final, alpha_raw, alpha_mean):
        del alpha_mean
        return torch.cat(
            (
                self.final_branch(final),
                self.detail_branch(l2),
                self.alpha_branch(alpha_raw),
            ),
            dim=-1,
        )


class CourtModel(nn.Module):
    def __init__(self, arm: nn.Module):
        super().__init__()
        self.arm = arm
        self.local = nn.Linear(512, 4)
        self.recon = nn.Linear(512, 384)
        self.alpha = nn.Linear(512, ALPHA_BITS)

    def forward(self, l2, final, alpha_raw, alpha_mean):
        z = self.arm(l2, final, alpha_raw, alpha_mean)
        return self.local(z), self.recon(z), self.alpha(z)


def adapter_params(model: CourtModel) -> int:
    return sum(p.numel() for p in model.arm.parameters() if p.requires_grad)


def alpha_pairs() -> torch.Tensor:
    g = torch.Generator().manual_seed(ALPHA_PAIR_SEED)
    return torch.randperm(ALPHA_CELLS, generator=g)[: 2 * ALPHA_BITS].reshape(
        ALPHA_BITS, 2
    )


def make_alpha_rows(count: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("ALPHA_ROW_COUNT_NONPOSITIVE")
    g = torch.Generator().manual_seed(int(seed))
    target = torch.randint(0, 2, (count, ALPHA_BITS), generator=g)
    raw = torch.zeros(count, ALPHA_CELLS, dtype=torch.float32)
    pairs = alpha_pairs()
    rows = torch.arange(count)[:, None].expand(-1, ALPHA_BITS)
    chosen = torch.where(target.bool(), pairs[:, 1][None, :], pairs[:, 0][None, :])
    raw[rows, chosen] = 1.0
    mean = raw.mean(dim=-1, keepdim=True)
    return raw, mean, target.float()


def build_features(model):
    per_width = {}
    identity = 0.0
    for width in WIDTHS:
        train_images, train_labels = make_canvases(
            TRAIN_CANVASES, width=width, seed=MASTER_SEED + width * 100 + 1
        )
        eval_images, eval_labels = make_canvases(
            EVAL_CANVASES, width=width, seed=MASTER_SEED + width * 100 + 2
        )
        train_features, train_identity = extract_all_layers(model, train_images)
        eval_features, eval_identity = extract_all_layers(model, eval_images)
        identity = max(identity, train_identity, eval_identity)

        train_l2 = train_features["L2"].reshape(-1, 384).float()
        train_final = train_features["FINAL_CONTRACT"].reshape(-1, 384).float()
        eval_l2 = eval_features["L2"].reshape(-1, 384).float()
        eval_final = eval_features["FINAL_CONTRACT"].reshape(-1, 384).float()
        train_alpha = make_alpha_rows(
            train_l2.shape[0], seed=MASTER_SEED + width * 1000 + 31
        )
        eval_alpha = make_alpha_rows(
            eval_l2.shape[0], seed=MASTER_SEED + width * 1000 + 47
        )
        per_width[width] = {
            "train_l2": train_l2,
            "train_final": train_final,
            "train_local_y": train_labels.reshape(-1, 4).float(),
            "train_alpha_raw": train_alpha[0],
            "train_alpha_mean": train_alpha[1],
            "train_alpha_y": train_alpha[2],
            "eval_l2": eval_l2,
            "eval_final": eval_final,
            "eval_local_y": eval_labels.reshape(-1, 4).float(),
            "eval_alpha_raw": eval_alpha[0],
            "eval_alpha_mean": eval_alpha[1],
            "eval_alpha_y": eval_alpha[2],
        }
    return per_width, identity


def train_eval(seed: int, arm_name: str, arm_template: nn.Module, data):
    torch.manual_seed(seed)
    model = CourtModel(copy.deepcopy(arm_template))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0
    )
    keys = (
        "train_l2",
        "train_final",
        "train_local_y",
        "train_alpha_raw",
        "train_alpha_mean",
        "train_alpha_y",
    )
    merged = {key: torch.cat([data[w][key] for w in WIDTHS]) for key in keys}
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        local_logits, final_pred, alpha_logits = model(
            merged["train_l2"],
            merged["train_final"],
            merged["train_alpha_raw"],
            merged["train_alpha_mean"],
        )
        local_loss = F.binary_cross_entropy_with_logits(
            local_logits, merged["train_local_y"]
        )
        final_loss = 1.0 - F.cosine_similarity(
            final_pred, merged["train_final"], dim=-1
        ).mean()
        alpha_loss = F.binary_cross_entropy_with_logits(
            alpha_logits, merged["train_alpha_y"]
        )
        loss = local_loss + final_loss + alpha_loss
        loss.backward()
        optimizer.step()

    rows = []
    model.eval()
    with torch.no_grad():
        for width in WIDTHS:
            d = data[width]
            local_logits, final_pred, alpha_logits = model(
                d["eval_l2"],
                d["eval_final"],
                d["eval_alpha_raw"],
                d["eval_alpha_mean"],
            )
            rows.append(
                {
                    "seed": seed,
                    "arm": arm_name,
                    "width": width,
                    "local_accuracy": float(
                        ((local_logits >= 0) == (d["eval_local_y"] >= 0.5))
                        .float()
                        .mean()
                    ),
                    "final_cosine": float(
                        F.cosine_similarity(final_pred, d["eval_final"], dim=-1).mean()
                    ),
                    "alpha_accuracy": float(
                        ((alpha_logits >= 0) == (d["eval_alpha_y"] >= 0.5))
                        .float()
                        .mean()
                    ),
                }
            )
    return rows


def run(source: Path, weights: Path):
    torch.set_num_threads(1)
    foundation, seal = load_exact_dinov2_s_v2(source, weights, device="cpu")
    data, identity = build_features(foundation)
    arms = {
        "DUAL_NO_ALPHA": DualNoAlpha(),
        "DUAL_MEAN_ALPHA64": DualMeanAlpha64(),
        "DUAL_RAW_ALPHA64": DualRawAlpha64(),
        "DUAL_RAW_ALPHA96": DualRawAlpha96(),
    }
    counts = {
        name: adapter_params(CourtModel(copy.deepcopy(arm)))
        for name, arm in arms.items()
    }

    rows = []
    for seed in SEEDS:
        for name, arm in arms.items():
            rows += train_eval(seed, name, arm, data)

    def values(arm, key):
        return [row[key] for row in rows if row["arm"] == arm]

    summary = {
        arm: {
            "mean_local": sum(values(arm, "local_accuracy")) / len(values(arm, "local_accuracy")),
            "min_local": min(values(arm, "local_accuracy")),
            "mean_final_cosine": sum(values(arm, "final_cosine")) / len(values(arm, "final_cosine")),
            "min_final_cosine": min(values(arm, "final_cosine")),
            "mean_alpha": sum(values(arm, "alpha_accuracy")) / len(values(arm, "alpha_accuracy")),
            "min_alpha": min(values(arm, "alpha_accuracy")),
        }
        for arm in arms
    }

    alpha_means = torch.cat(
        [data[w]["train_alpha_mean"].reshape(-1) for w in WIDTHS]
        + [data[w]["eval_alpha_mean"].reshape(-1) for w in WIDTHS]
    )
    alpha_targets = torch.cat(
        [data[w]["train_alpha_y"].reshape(-1) for w in WIDTHS]
        + [data[w]["eval_alpha_y"].reshape(-1) for w in WIDTHS]
    )
    mean_variance = float(alpha_means.var(unbiased=False))
    target_fraction = float(alpha_targets.mean())

    base = summary["DUAL_NO_ALPHA"]
    mean_control = summary["DUAL_MEAN_ALPHA64"]
    validity = (
        identity <= 1.0e-6
        and all(count <= CURRENT_ADAPTER_PARAMS for count in counts.values())
        and mean_variance <= 1.0e-12
        and 0.45 <= target_fraction <= 0.55
        and base["mean_local"] >= 0.98
        and base["min_local"] >= 0.95
        and base["mean_final_cosine"] >= 0.98
        and mean_control["mean_alpha"] <= 0.55
    )

    def eligible(name: str) -> bool:
        s = summary[name]
        return (
            s["mean_local"] >= 0.995
            and s["min_local"] >= 0.98
            and s["mean_final_cosine"] >= 0.99
            and s["min_final_cosine"] >= 0.98
            and s["mean_alpha"] >= 0.995
            and s["min_alpha"] >= 0.98
            and s["mean_alpha"] - base["mean_alpha"] >= 0.30
            and s["mean_alpha"] - mean_control["mean_alpha"] >= 0.30
            and counts[name] <= CURRENT_ADAPTER_PARAMS
        )

    raw64 = eligible("DUAL_RAW_ALPHA64")
    raw96 = eligible("DUAL_RAW_ALPHA96")
    if not validity:
        status = "ALPHA_RESIDUAL_COURT_INVALID"
        selected = None
    elif raw64:
        status = "PATCH_ALIGNED_RAW_ALPHA64_SUPPORTED_TINY_V1"
        selected = "DUAL_RAW_ALPHA64"
    elif raw96:
        status = "PATCH_ALIGNED_RAW_ALPHA96_SUPPORTED_TINY_V1"
        selected = "DUAL_RAW_ALPHA96"
    else:
        status = "PATCH_ALIGNED_ALPHA_RESIDUAL_NOT_ESTABLISHED_TINY_V1"
        selected = None

    return {
        "schema": SCHEMA,
        "status": status,
        "selected_arm": selected,
        "authority": {
            "source_head": seal.source_head,
            "weight_sha256": seal.weight_sha256,
        },
        "identity_error": identity,
        "adapter_parameter_counts": counts,
        "alpha_construction": {
            "bits": ALPHA_BITS,
            "cells": ALPHA_CELLS,
            "pair_seed": ALPHA_PAIR_SEED,
            "mean_variance": mean_variance,
            "target_positive_fraction": target_fraction,
        },
        "results": rows,
        "summary": summary,
        "gates": {
            "validity": validity,
            "raw64_eligible": raw64,
            "raw96_eligible": raw96,
            "mean_alpha_negative_control": mean_control["mean_alpha"] <= 0.55,
        },
        "execution": {
            "device": "cpu",
            "gpu_required": False,
            "foundation_frozen": True,
            "steps": STEPS,
            "lr": LR,
        },
        "claims": {
            "product_authority_changed": False,
            "truth139_training_authorized": False,
            "fit8_training_authorized": False,
            "dinov3_adoption_authorized": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2-source", type=Path, required=True)
    parser.add_argument("--dinov2-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dinov2_source, args.dinov2_weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_arm": result["selected_arm"],
                "summary": result["summary"],
                "adapter_parameter_counts": result["adapter_parameter_counts"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
