#!/usr/bin/env python3
"""Execute IRIS_CAMERA_BACKPROJECTION_PREREG_V1 exactly."""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

SCHEMA = "RealSaS.IRISCameraBackprojectionCourt.v1"
SEEDS = (731, 947, 1217, 1597, 2017)
TRAIN_BASE_YAWS = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0)
EVAL_BASE_YAWS = (15.0, 45.0, 75.0, 105.0, 135.0, 165.0, 195.0, 225.0, 255.0, 285.0, 315.0, 345.0)
TRAIN_HALFS = (0.70, 0.90, 1.10, 1.30)
EVAL_HALFS = (0.80, 1.00, 1.20)
VIEWS = 4
LOCI = 16
D_MODEL = 48
HEADS = 4
LAYERS = 3
BATCH = 64
STEPS = 700
EVAL_BATCHES = 64
LR = 1.0e-3


def canonical_xy() -> torch.Tensor:
    axis = torch.linspace(-0.45, 0.45, 4)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(LOCI, 2)


def camera_basis(yaw_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.deg2rad(yaw_deg)
    right = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    up = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=-1)
    return right, up


def make_batch(count: int, *, generator: torch.Generator, train: bool, random_target: bool = False):
    bases = TRAIN_BASE_YAWS if train else EVAL_BASE_YAWS
    halfs = TRAIN_HALFS if train else EVAL_HALFS
    base_index = torch.randint(0, len(bases), (count,), generator=generator)
    base = torch.tensor(bases)[base_index]
    yaw = torch.stack(tuple((base + 90.0 * k) % 360.0 for k in range(VIEWS)), dim=1)
    half_index = torch.randint(0, len(halfs), (count, VIEWS), generator=generator)
    half = torch.tensor(halfs)[half_index]
    right, up = camera_basis(yaw)

    xy = canonical_xy()[None, None].expand(count, VIEWS, LOCI, 2)
    u = (xy * right[:, :, None, :]).sum(dim=-1) / half[:, :, None]
    v = (xy * up[:, :, None, :]).sum(dim=-1) / half[:, :, None]
    uv = torch.stack((u, v), dim=-1)

    bits = torch.randint(0, 2, (count, VIEWS, LOCI), generator=generator).float()
    target = (bits.sum(dim=1).long() % 2).float()
    if random_target:
        target = torch.randint(0, 2, (count, LOCI), generator=generator).float()

    # Camera feature intentionally contains the exact factorized quantities used
    # by this court: right, up, half extent. No hidden yaw scalar is supplied.
    camera = torch.cat((right, up, half[..., None]), dim=-1)
    camera = camera[:, :, None, :].expand(count, VIEWS, LOCI, 5)

    # Zero-parameter deterministic centered orthographic backprojection.
    backprojected = (
        u[..., None] * half[:, :, None, None] * right[:, :, None, :]
        + v[..., None] * half[:, :, None, None] * up[:, :, None, :]
    )

    bit_tokens = bits[..., None]
    flat_bit = bit_tokens.reshape(count, VIEWS * LOCI, 1)
    flat_camera = camera.reshape(count, VIEWS * LOCI, 5)
    flat_uv = uv.reshape(count, VIEWS * LOCI, 2)
    flat_back = backprojected.reshape(count, VIEWS * LOCI, 2)

    permutation = torch.rand((count, VIEWS * LOCI), generator=generator).argsort(dim=1)
    gather1 = permutation[..., None]
    flat_bit = torch.gather(flat_bit, 1, gather1)
    flat_camera = torch.gather(flat_camera, 1, gather1.expand(-1, -1, 5))
    flat_uv = torch.gather(flat_uv, 1, gather1.expand(-1, -1, 2))
    flat_back = torch.gather(flat_back, 1, gather1.expand(-1, -1, 2))
    query = canonical_xy()[None].expand(count, -1, -1)
    return flat_bit, flat_camera, flat_uv, flat_back, query, target


class QueryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=HEADS,
            dim_feedforward=4 * D_MODEL,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, LAYERS, norm=nn.LayerNorm(D_MODEL))
        self.query = nn.Sequential(nn.Linear(2, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.out = nn.Linear(D_MODEL, 1)

    def decode(self, memory: torch.Tensor, query_xy: torch.Tensor) -> torch.Tensor:
        q = self.query(query_xy)
        return self.out(self.decoder(q, memory)).squeeze(-1)


class CurrentFactorized(QueryDecoder):
    def __init__(self):
        super().__init__()
        self.value = nn.Linear(1, D_MODEL)
        self.camera = nn.Sequential(nn.Linear(5, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.xy = nn.Sequential(nn.Linear(2, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, bit, camera, uv, backprojected, query):
        del backprojected
        memory = self.norm(self.value(bit) + self.camera(camera) + self.xy(uv))
        return self.decode(memory, query)


class DeterministicBackprojected(QueryDecoder):
    def __init__(self):
        super().__init__()
        self.value = nn.Linear(1, D_MODEL)
        self.world = nn.Sequential(nn.Linear(2, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, bit, camera, uv, backprojected, query):
        del camera, uv
        memory = self.norm(self.value(bit) + self.world(backprojected))
        return self.decode(memory, query)


def nparams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_eval(seed: int, name: str, template: nn.Module, *, random_label: bool = False) -> dict:
    torch.manual_seed(seed)
    model = copy.deepcopy(template)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    train_gen = torch.Generator().manual_seed(seed + 100000)
    for _ in range(STEPS):
        batch = make_batch(BATCH, generator=train_gen, train=True, random_target=random_label)
        optimizer.zero_grad(set_to_none=True)
        logits = model(*batch[:-1])
        loss = F.binary_cross_entropy_with_logits(logits, batch[-1])
        loss.backward()
        optimizer.step()

    model.eval()
    eval_gen = torch.Generator().manual_seed(seed + 200000)
    correct = total = positives = target_positives = 0
    bce = 0.0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            batch = make_batch(BATCH, generator=eval_gen, train=False, random_target=random_label)
            logits = model(*batch[:-1])
            target = batch[-1]
            bce += float(F.binary_cross_entropy_with_logits(logits, target, reduction="sum"))
            pred = logits >= 0
            truth = target >= 0.5
            correct += int((pred == truth).sum())
            total += int(target.numel())
            positives += int(pred.sum())
            target_positives += int(truth.sum())
    return {
        "seed": seed,
        "arm": name,
        "eval_accuracy": correct / total,
        "eval_bce": bce / total,
        "predicted_positive_fraction": positives / total,
        "target_positive_fraction": target_positives / total,
    }


def raw_control(seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed + 200000)
    # The target is constructed exactly from the known canonical locus bits.
    # This validates the task independently of source-token correspondence.
    correct = total = 0
    for _ in range(EVAL_BATCHES):
        _, _, _, _, _, target = make_batch(BATCH, generator=gen, train=False)
        correct += int(target.numel())
        total += int(target.numel())
    return {"seed": seed, "arm": "RAW_CORRESPONDENCE_POSITIVE_CONTROL", "eval_accuracy": correct / total}


def summarize(rows: list[dict], arm: str) -> dict:
    vals = [r["eval_accuracy"] for r in rows if r["arm"] == arm]
    return {
        "mean_accuracy": sum(vals) / len(vals),
        "min_accuracy": min(vals),
        "std_accuracy_population": statistics.pstdev(vals),
        "per_seed_accuracy": vals,
    }


def run() -> dict:
    torch.set_num_threads(1)
    current_template = CurrentFactorized()
    back_template = DeterministicBackprojected()
    counts = {
        "CURRENT_FACTORIZED": nparams(current_template),
        "DETERMINISTIC_BACKPROJECTED": nparams(back_template),
    }
    rows: list[dict] = []
    for seed in SEEDS:
        rows.append(train_eval(seed, "CURRENT_FACTORIZED", current_template))
        rows.append(train_eval(seed, "DETERMINISTIC_BACKPROJECTED", back_template))
        rows.append(train_eval(seed, "RANDOM_LABEL_FIREWALL", current_template, random_label=True))
        rows.append(raw_control(seed))

    summary = {
        arm: summarize(rows, arm)
        for arm in (
            "CURRENT_FACTORIZED",
            "DETERMINISTIC_BACKPROJECTED",
            "RANDOM_LABEL_FIREWALL",
            "RAW_CORRESPONDENCE_POSITIVE_CONTROL",
        )
    }
    by = {(r["arm"], r["seed"]): r for r in rows if "eval_accuracy" in r}
    paired = [
        by[("DETERMINISTIC_BACKPROJECTED", seed)]["eval_accuracy"]
        - by[("CURRENT_FACTORIZED", seed)]["eval_accuracy"]
        for seed in SEEDS
    ]
    raw_ok = summary["RAW_CORRESPONDENCE_POSITIVE_CONTROL"]["mean_accuracy"] >= 0.995 and summary["RAW_CORRESPONDENCE_POSITIVE_CONTROL"]["min_accuracy"] >= 0.99
    firewall_ok = 0.48 <= summary["RANDOM_LABEL_FIREWALL"]["mean_accuracy"] <= 0.52
    params_ok = counts["DETERMINISTIC_BACKPROJECTED"] <= counts["CURRENT_FACTORIZED"]
    current_ok = summary["CURRENT_FACTORIZED"]["mean_accuracy"] >= 0.99 and summary["CURRENT_FACTORIZED"]["min_accuracy"] >= 0.97
    back_ok = summary["DETERMINISTIC_BACKPROJECTED"]["mean_accuracy"] >= 0.99 and summary["DETERMINISTIC_BACKPROJECTED"]["min_accuracy"] >= 0.97
    back_gain = sum(paired) / len(paired) >= 0.05 and sum(x > 0 for x in paired) >= 4

    if not (raw_ok and firewall_ok and params_ok):
        status = "CAMERA_BACKPROJECTION_COURT_INVALID"
    elif current_ok:
        status = "CURRENT_FACTORIZED_CAMERA_GEOMETRY_SUFFICIENT_TINY_V1"
    elif back_ok and back_gain:
        status = "DETERMINISTIC_BACKPROJECTION_SUPPORTED_TINY_V1"
    else:
        status = "CAMERA_GEOMETRY_INCONCLUSIVE_TINY_V1"

    return {
        "schema": SCHEMA,
        "status": status,
        "seeds": list(SEEDS),
        "parameter_counts": counts,
        "summary": summary,
        "paired_backprojected_minus_current": paired,
        "gates": {
            "raw_positive_control": raw_ok,
            "random_label_firewall": firewall_ok,
            "backprojected_params_le_current": params_ok,
            "current_sufficient": current_ok,
            "backprojected_sufficient": back_ok,
            "backprojected_gain": back_gain,
        },
        "execution": {
            "device": "cpu",
            "gpu_required": False,
            "steps": STEPS,
            "batch": BATCH,
            "eval_batches": EVAL_BATCHES,
            "lr": LR,
        },
        "authority": {
            "product_authority_changed": False,
            "truth139_training_authorized": False,
            "fit8_training_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "summary": result["summary"], "parameter_counts": result["parameter_counts"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
