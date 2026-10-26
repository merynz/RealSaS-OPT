#!/usr/bin/env python3
"""Execute DINOV2_FEATURE_ACCESS_PREREG_V1 exactly."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from models.iris.v2.dinov2_foundation_v2 import (
    DINOv2FoundationAuthorityV2,
    load_exact_dinov2_s_v2,
    preprocess_dinov2_rgb_flat_v2,
)

SCHEMA = "RealSaS.DINOv2FeatureAccessCourt.v1"
WIDTHS = (2, 4, 8, 12)
LAYERS = (2, 5, 8, 11)
TRAIN_CANVASES = 12
EVAL_CANVASES = 6
ACTIVE = tuple(range(15, 22))  # central 7x7 logical cells
MASTER_SEED = 240139
PROBE_SEED = 731
PROBE_STEPS = 800
PROBE_LR = 3.0e-3


def _partition_bounds(index: int) -> tuple[int, int]:
    # Integer-rounded 37-bin partition of native [0,1024).
    lo = int(round(index * 1024.0 / 37.0))
    hi = int(round((index + 1) * 1024.0 / 37.0))
    return lo, hi


def _square_bounds(center: float, width: int, lo: int, hi: int) -> tuple[int, int]:
    start = int(round(center - width / 2.0))
    start = max(lo, min(start, hi - width))
    return start, start + width


def make_canvases(count: int, *, width: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.full((count, 3, 1024, 1024), 127, dtype=torch.uint8)
    labels = torch.randint(0, 2, (count, len(ACTIVE), len(ACTIVE), 4), generator=g, dtype=torch.int64)
    for r_i, r in enumerate(ACTIVE):
        y0, y1 = _partition_bounds(r)
        h = y1 - y0
        cy_top = y0 + 0.25 * h
        cy_bottom = y0 + 0.75 * h
        for c_i, c in enumerate(ACTIVE):
            x0, x1 = _partition_bounds(c)
            w = x1 - x0
            cx_left = x0 + 0.25 * w
            cx_right = x0 + 0.75 * w
            centers = (
                (cy_top, cx_left),
                (cy_top, cx_right),
                (cy_bottom, cx_left),
                (cy_bottom, cx_right),
            )
            for bit_index, (cy, cx) in enumerate(centers):
                ys, ye = _square_bounds(cy, width, y0, y1)
                xs, xe = _square_bounds(cx, width, x0, x1)
                values = torch.where(
                    labels[:, r_i, c_i, bit_index] > 0,
                    torch.tensor(240, dtype=torch.uint8),
                    torch.tensor(15, dtype=torch.uint8),
                )
                images[:, :, ys:ye, xs:xe] = values[:, None, None, None]
    return images, labels


def _active_patch_indices() -> torch.Tensor:
    return torch.tensor([r * 37 + c for r in ACTIVE for c in ACTIVE], dtype=torch.long)


@torch.no_grad()
def extract_all_layers(
    model: nn.Module,
    native_rgb: torch.Tensor,
    *,
    batch_size: int = 2,
) -> tuple[dict[str, torch.Tensor], float]:
    selected = _active_patch_indices()
    collected: dict[str, list[torch.Tensor]] = {f"L{i}": [] for i in LAYERS}
    collected["FINAL_CONTRACT"] = []
    equality_max = 0.0

    for start in range(0, native_rgb.shape[0], batch_size):
        batch_native = native_rgb[start : start + batch_size]
        x = preprocess_dinov2_rgb_flat_v2(batch_native)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for idx in LAYERS:
            def hook(_module, _inputs, output, idx=idx):
                captured[idx] = output
            handles.append(model.blocks[idx].register_forward_hook(hook))
        try:
            features = model.forward_features(x)
        finally:
            for handle in handles:
                handle.remove()
        final = features["x_norm_patchtokens"].float()
        if tuple(final.shape[1:]) != (37 * 37, 384):
            raise RuntimeError(f"FINAL_FEATURE_SHAPE_DRIFT:{tuple(final.shape)}")
        final_active = final[:, selected]
        collected["FINAL_CONTRACT"].append(final_active.cpu())
        for idx in LAYERS:
            raw = captured[idx]
            normed = model.norm(raw).float()[:, 1:]
            if tuple(normed.shape[1:]) != (37 * 37, 384):
                raise RuntimeError(f"LAYER_FEATURE_SHAPE_DRIFT:{idx}:{tuple(normed.shape)}")
            active = normed[:, selected]
            collected[f"L{idx}"].append(active.cpu())
            if idx == 11:
                equality_max = max(equality_max, float((active - final_active).abs().max().cpu()))
    return {key: torch.cat(value, dim=0) for key, value in collected.items()}, equality_max


def flatten_samples(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # features [canvas,49,384], labels [canvas,7,7,4]
    return features.reshape(-1, 384).float(), labels.reshape(-1, 4).float()


def fit_probe(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor, eval_y: torch.Tensor) -> dict:
    torch.manual_seed(PROBE_SEED)
    head = nn.Linear(384, 4)
    optimizer = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, betas=(0.9, 0.999), weight_decay=0.0)
    for _ in range(PROBE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_x)
        loss = F.binary_cross_entropy_with_logits(logits, train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = head(eval_x)
        bce = float(F.binary_cross_entropy_with_logits(logits, eval_y))
        pred = logits >= 0
        truth = eval_y >= 0.5
        bit_acc = (pred == truth).float().mean(dim=0)
        overall = float(bit_acc.mean())
    return {
        "heldout_bit_accuracy": overall,
        "heldout_bce": bce,
        "per_bit_accuracy": [float(x) for x in bit_acc],
        "train_feature_variance_mean": float(train_x.var(dim=0, unbiased=False).mean()),
        "eval_feature_variance_mean": float(eval_x.var(dim=0, unbiased=False).mean()),
    }


def _positive_fractions(labels: torch.Tensor) -> list[float]:
    return [float(x) for x in labels.float().reshape(-1, 4).mean(dim=0)]


def _fractions_valid(values: Iterable[float]) -> bool:
    return all(0.40 <= float(x) <= 0.60 for x in values)


def run(source: Path, weights: Path) -> dict:
    torch.set_num_threads(1)
    model, seal = load_exact_dinov2_s_v2(source, weights, device="cpu")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("FOUNDATION_GRADIENT_AUTHORITY_DRIFT")

    rows: list[dict] = []
    max_identity_error = 0.0
    label_checks = []
    finite_ok = True

    for width in WIDTHS:
        train_images, train_labels = make_canvases(TRAIN_CANVASES, width=width, seed=MASTER_SEED + width * 100 + 1)
        eval_images, eval_labels = make_canvases(EVAL_CANVASES, width=width, seed=MASTER_SEED + width * 100 + 2)
        train_pos = _positive_fractions(train_labels)
        eval_pos = _positive_fractions(eval_labels)
        label_checks.append({"width": width, "train_positive_fraction": train_pos, "eval_positive_fraction": eval_pos})

        train_features, err_a = extract_all_layers(model, train_images)
        eval_features, err_b = extract_all_layers(model, eval_images)
        max_identity_error = max(max_identity_error, err_a, err_b)

        for arm in ("L2", "L5", "L8", "L11", "FINAL_CONTRACT"):
            tx, ty = flatten_samples(train_features[arm], train_labels)
            ex, ey = flatten_samples(eval_features[arm], eval_labels)
            finite_ok = finite_ok and bool(torch.isfinite(tx).all()) and bool(torch.isfinite(ex).all())
            metrics = fit_probe(tx, ty, ex, ey)
            rows.append({"width_native_px": width, "arm": arm, **metrics})

    label_balance_ok = all(
        _fractions_valid(row["train_positive_fraction"]) and _fractions_valid(row["eval_positive_fraction"])
        for row in label_checks
    )
    identity_ok = max_identity_error <= 1.0e-6
    authority_ok = (
        seal.source_head == DINOv2FoundationAuthorityV2().source_revision
        and seal.weight_sha256 == DINOv2FoundationAuthorityV2().weight_sha256
    )
    validity_ok = label_balance_ok and identity_ok and finite_ok and authority_ok

    by = {(r["width_native_px"], r["arm"]): r for r in rows}
    final_acc = {w: by[(w, "FINAL_CONTRACT")]["heldout_bit_accuracy"] for w in WIDTHS}
    earlier = {}
    for arm in ("L2", "L5", "L8"):
        gains = {w: by[(w, arm)]["heldout_bit_accuracy"] - final_acc[w] for w in WIDTHS}
        earlier[arm] = {
            "per_width_gain_vs_final": gains,
            "mean_gain_vs_final": sum(gains.values()) / len(gains),
            "count_gain_ge_0p05": sum(v >= 0.05 for v in gains.values()),
        }

    final_sufficient = all(final_acc[w] >= 0.95 for w in WIDTHS) and not any(
        earlier[arm]["per_width_gain_vs_final"][w] >= 0.03
        for arm in earlier for w in WIDTHS
    )
    materially_better_layers = [
        arm for arm, data in earlier.items()
        if data["count_gain_ge_0p05"] >= 2 and data["mean_gain_vs_final"] >= 0.03
    ]
    no_current_access = any(final_acc[w] < 0.90 for w in WIDTHS) and not any(
        all(by[(w, arm)]["heldout_bit_accuracy"] >= 0.95 for w in WIDTHS)
        for arm in ("L2", "L5", "L8")
    )

    if not validity_ok:
        status = "DINOV2_FEATURE_ACCESS_COURT_INVALID"
    elif final_sufficient:
        status = "CURRENT_FINAL_ACCESS_SUFFICIENT_TINY_V1"
    elif materially_better_layers:
        status = "EARLIER_FEATURE_ACCESS_MATERIALLY_BETTER_TINY_V1"
    elif no_current_access:
        status = "CURRENT_DINOV2_LOCAL_RGB_ACCESS_NOT_ESTABLISHED_TINY_V1"
    else:
        status = "DINOV2_FEATURE_ACCESS_INCONCLUSIVE_TINY_V1"

    return {
        "schema": SCHEMA,
        "status": status,
        "preregistration": "DINOV2_FEATURE_ACCESS_PREREG_V1.md",
        "authority": {
            "source_head": seal.source_head,
            "weight_sha256": seal.weight_sha256,
            "feature_contract_hash": seal.feature_contract_hash,
            "verified": authority_ok,
        },
        "dataset": {
            "widths_native_px": list(WIDTHS),
            "train_canvases_per_width": TRAIN_CANVASES,
            "eval_canvases_per_width": EVAL_CANVASES,
            "active_patch_cells_per_canvas": len(ACTIVE) ** 2,
            "master_seed": MASTER_SEED,
            "label_checks": label_checks,
        },
        "probe": {
            "readout": "Linear(384,4)",
            "seed": PROBE_SEED,
            "steps": PROBE_STEPS,
            "lr": PROBE_LR,
            "optimizer": "AdamW",
        },
        "validity": {
            "label_balance_ok": label_balance_ok,
            "l11_final_identity_max_abs_error": max_identity_error,
            "l11_final_identity_ok": identity_ok,
            "features_finite": finite_ok,
            "authority_ok": authority_ok,
        },
        "results": rows,
        "analysis": {
            "final_accuracy_by_width": final_acc,
            "earlier_layer_gains": earlier,
            "materially_better_layers": materially_better_layers,
        },
        "execution": {"device": "cpu", "foundation_frozen": True, "gpu_required": False},
        "claims": {
            "truth139_training_authorized": False,
            "fit8_training_authorized": False,
            "dinov3_adoption_authorized": False,
            "product_authority_changed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov2-source", type=Path, required=True)
    parser.add_argument("--dinov2-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.dinov2_source, args.dinov2_weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": report["schema"],
        "status": report["status"],
        "identity_error": report["validity"]["l11_final_identity_max_abs_error"],
        "final_accuracy": report["analysis"]["final_accuracy_by_width"],
        "materially_better_layers": report["analysis"]["materially_better_layers"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
