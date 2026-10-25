#!/usr/bin/env python3
"""Execute SYNTHETIC_IRIS_SUBCELL_ROUTING_PREREG_V1 exactly."""
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

SCHEMA = "RealSaS.IRISV3PostFoundationSubcellRoutingCourt.v1"
SEEDS = (731, 947, 1217, 1597, 2017)
TRAIN_STEPS = 1200
BATCH_SIZE = 64
EVAL_COUNT = 4096
LR = 1.0e-3
GRID = 4
OUT_GRID = 8
D_MODEL = 64
HEADS = 4
LAYERS = 2


def grid_xy() -> torch.Tensor:
    # align_corners=False-style cell centers for a 4x4 logical source grid.
    axis = -1.0 + (torch.arange(GRID, dtype=torch.float32) + 0.5) * (2.0 / GRID)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(GRID * GRID, 2)


def input_batch(count: int, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    coords = grid_xy()
    bits01 = torch.randint(0, 2, (count, GRID * GRID, 4), generator=generator, dtype=torch.int64)
    bits = bits01.float().mul(2.0).sub(1.0)
    source = torch.cat((coords[None].expand(count, -1, -1), bits), dim=-1)
    permutation = torch.rand((count, GRID * GRID), generator=generator).argsort(dim=1)
    source = torch.gather(source, 1, permutation[..., None].expand(-1, -1, 6))

    target = torch.empty(count, OUT_GRID, OUT_GRID, dtype=torch.float32)
    canonical = bits01.reshape(count, GRID, GRID, 4)
    target[:, 0::2, 0::2] = canonical[..., 0].float()
    target[:, 0::2, 1::2] = canonical[..., 1].float()
    target[:, 1::2, 0::2] = canonical[..., 2].float()
    target[:, 1::2, 1::2] = canonical[..., 3].float()
    return source, target


class CurrentLikeQueryUpsample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.payload_projection = nn.Linear(4, D_MODEL)
        self.xy_projection = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, D_MODEL))
        self.source_norm = nn.LayerNorm(D_MODEL)
        self.scene_queries = nn.Parameter(torch.randn(GRID, GRID, D_MODEL) / math.sqrt(D_MODEL))
        self.query_position = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, D_MODEL))
        layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=LAYERS, norm=nn.LayerNorm(D_MODEL))
        self.upsample = nn.ConvTranspose2d(D_MODEL, 1, kernel_size=2, stride=2)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        xy = source[..., :2]
        payload = source[..., 2:]
        memory = self.source_norm(self.payload_projection(payload) + self.xy_projection(xy))
        qxy = grid_xy().to(source).reshape(GRID, GRID, 2)
        q = self.scene_queries.to(source) + self.query_position(qxy)
        q = q.reshape(1, GRID * GRID, D_MODEL).expand(source.shape[0], -1, -1)
        scene = self.decoder(q, memory).reshape(source.shape[0], GRID, GRID, D_MODEL)
        scene = scene.permute(0, 3, 1, 2).contiguous()
        return self.upsample(scene).squeeze(1)


class RawBlockPositiveControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 4))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        block_logits = self.net(source)
        xy = source[..., :2]
        # Exact deterministic decoding of the known synthetic lattice coordinate.
        ix = torch.round((xy[..., 0] + 0.75) / 0.5).to(torch.long).clamp(0, GRID - 1)
        iy = torch.round((xy[..., 1] + 0.75) / 0.5).to(torch.long).clamp(0, GRID - 1)
        batch = source.shape[0]
        out = torch.empty(batch, OUT_GRID, OUT_GRID, device=source.device, dtype=source.dtype)
        b = torch.arange(batch, device=source.device)[:, None].expand_as(ix)
        out[b, iy * 2, ix * 2] = block_logits[..., 0]
        out[b, iy * 2, ix * 2 + 1] = block_logits[..., 1]
        out[b, iy * 2 + 1, ix * 2] = block_logits[..., 2]
        out[b, iy * 2 + 1, ix * 2 + 1] = block_logits[..., 3]
        return out


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_eval(*, seed: int, arm: str, template: nn.Module, random_label: bool = False) -> dict:
    model = copy.deepcopy(template)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    input_gen = torch.Generator(device="cpu").manual_seed(seed + 100_000)
    label_gen = torch.Generator(device="cpu").manual_seed(seed + 300_000)
    last_loss = None
    for _ in range(TRAIN_STEPS):
        source, true_target = input_batch(BATCH_SIZE, generator=input_gen)
        if random_label:
            target = torch.randint(0, 2, true_target.shape, generator=label_gen, dtype=torch.int64).float()
        else:
            target = true_target
        optimizer.zero_grad(set_to_none=True)
        logits = model(source)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())

    model.eval()
    input_gen = torch.Generator(device="cpu").manual_seed(seed + 200_000)
    label_gen = torch.Generator(device="cpu").manual_seed(seed + 400_000)
    correct = 0
    total = 0
    bce_sum = 0.0
    seen = 0
    with torch.no_grad():
        while seen < EVAL_COUNT:
            n = min(BATCH_SIZE, EVAL_COUNT - seen)
            source, true_target = input_batch(n, generator=input_gen)
            if random_label:
                target = torch.randint(0, 2, true_target.shape, generator=label_gen, dtype=torch.int64).float()
            else:
                target = true_target
            logits = model(source)
            bce_sum += float(F.binary_cross_entropy_with_logits(logits, target, reduction="sum"))
            pred = logits >= 0
            truth = target >= 0.5
            correct += int((pred == truth).sum())
            total += int(truth.numel())
            seen += n
    return {
        "seed": seed,
        "arm": arm,
        "train_final_bce": last_loss,
        "eval_bit_accuracy": correct / float(total),
        "eval_bce_per_bit": bce_sum / float(total),
    }


def summary(rows: list[dict], arm: str) -> dict:
    vals = [r["eval_bit_accuracy"] for r in rows if r["arm"] == arm]
    return {
        "mean_bit_accuracy": sum(vals) / len(vals),
        "min_bit_accuracy": min(vals),
        "std_bit_accuracy_population": statistics.pstdev(vals),
        "per_seed_bit_accuracy": vals,
    }


def run() -> dict:
    torch.set_num_threads(1)
    rows: list[dict] = []
    param_counts = None
    for seed in SEEDS:
        torch.manual_seed(seed)
        current = CurrentLikeQueryUpsample()
        torch.manual_seed(seed)
        raw = RawBlockPositiveControl()
        if param_counts is None:
            param_counts = {
                "CURRENT_LIKE_QUERY_UPSAMPLE": parameter_count(current),
                "RAW_BLOCK_POSITIVE_CONTROL": parameter_count(raw),
            }
        rows.append(train_eval(seed=seed, arm="CURRENT_LIKE_QUERY_UPSAMPLE", template=current))
        rows.append(train_eval(seed=seed, arm="RAW_BLOCK_POSITIVE_CONTROL", template=raw))
        rows.append(train_eval(seed=seed, arm="RANDOM_LABEL_FIREWALL", template=current, random_label=True))

    current_s = summary(rows, "CURRENT_LIKE_QUERY_UPSAMPLE")
    raw_s = summary(rows, "RAW_BLOCK_POSITIVE_CONTROL")
    firewall_s = summary(rows, "RANDOM_LABEL_FIREWALL")
    firewall_ok = all(0.48 <= v <= 0.52 for v in firewall_s["per_seed_bit_accuracy"])
    raw_ok = raw_s["min_bit_accuracy"] >= 0.98 and raw_s["mean_bit_accuracy"] >= 0.995
    current_ok = current_s["min_bit_accuracy"] >= 0.95 and current_s["mean_bit_accuracy"] >= 0.98
    if not firewall_ok:
        status = "IRIS_SUBCELL_FIREWALL_INVALID"
    elif not raw_ok:
        status = "IRIS_SUBCELL_TASK_OR_CONTROL_INVALID"
    elif current_ok:
        status = "POSTFOUNDATION_SUBCELL_ROUTING_ESTABLISHED_TINY_V1"
    else:
        status = "POSTFOUNDATION_SUBCELL_ROUTING_NOT_ESTABLISHED_TINY_V1"

    return {
        "schema": SCHEMA,
        "status": status,
        "preregistration": "SYNTHETIC_IRIS_SUBCELL_ROUTING_PREREG_V1.md",
        "synthetic_scale": {"source_grid": [GRID, GRID], "carrier_grid": [OUT_GRID, OUT_GRID], "subcell_states_per_source_token": 4},
        "seeds": list(SEEDS),
        "train_steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "evaluation_count": EVAL_COUNT,
        "optimizer": {"name": "AdamW", "lr": LR, "betas": [0.9, 0.999], "weight_decay": 0.0},
        "parameter_counts": param_counts,
        "results": rows,
        "summary": {"current_like": current_s, "raw_positive_control": raw_s, "random_label_firewall": firewall_s},
        "gates": {"firewall_ok": firewall_ok, "positive_control_ok": raw_ok, "current_like_ok": current_ok},
        "authority": {
            "reference_iris_pass_claimed": False,
            "dinov2_access_pass_claimed": False,
            "truth139_training_authorized": False,
            "fit8_training_authorized": False,
            "product_authority_changed": False,
        },
        "execution": {"device": "cpu", "gpu_required": False, "foundation_model_executed": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": report["schema"],
        "status": report["status"],
        "mean_current": report["summary"]["current_like"]["mean_bit_accuracy"],
        "mean_raw": report["summary"]["raw_positive_control"]["mean_bit_accuracy"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
