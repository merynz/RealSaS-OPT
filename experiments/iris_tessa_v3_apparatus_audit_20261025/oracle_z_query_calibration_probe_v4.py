#!/usr/bin/env python3
"""Execute ORACLE_Z_QUERY_CALIBRATION_PREREG_V4 exactly."""
from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path

import torch
from torch import nn

SCHEMA = "RealSaS.OracleZQueryCalibrationCourt.v4"
LRS = (3.0e-3, 1.0e-3)
STEPS = 5000
ARMS = {
    "Q96_D2_PLAIN": {"width": 96, "depth": 2, "fourier_bands": 0},
    "Q256_D4_PLAIN": {"width": 256, "depth": 4, "fourier_bands": 0},
    "Q128_D3_FOURIER4": {"width": 128, "depth": 3, "fourier_bands": 4},
    "Q192_D3_FOURIER4": {"width": 192, "depth": 3, "fourier_bands": 4},
}

BASE = runpy.run_path(str(Path(__file__).with_name("oracle_z_surface_calibration_probe_v3.py")))
bank = BASE["bank"]
cal_scenes = BASE["cal_scenes"]
sample_batch = BASE["sample_batch"]
summary = BASE["summary"]
seed_all = BASE["seed_all"]
gen = BASE["gen"]
OUT = int(BASE["OUT"])


class QueryNet(nn.Module):
    def __init__(self, width: int, depth: int, fourier_bands: int):
        super().__init__()
        self.fourier_bands = int(fourier_bands)
        inp = 3 * (1 + 2 * self.fourier_bands)
        layers = []
        for i in range(int(depth)):
            layers.append(nn.Linear(inp if i == 0 else width, width))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(width, OUT))
        self.net = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier_bands <= 0:
            return x
        feats = [x]
        for band in range(self.fourier_bands):
            freq = (2.0 ** band) * math.pi
            feats.extend((torch.sin(freq * x), torch.cos(freq * x)))
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(x))


class PerSceneQuery(nn.Module):
    def __init__(self, scene_count: int, spec: dict):
        super().__init__()
        self.nets = nn.ModuleList(
            QueryNet(spec["width"], spec["depth"], spec["fourier_bands"])
            for _ in range(scene_count)
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return torch.stack([net(p[i]) for i, net in enumerate(self.nets)], dim=0)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_arm(arm: str, B: dict, lr: float, *, steps: int, seed: int) -> nn.Module:
    seed_all(seed)
    model = PerSceneQuery(len(B["scenes"]), ARMS[arm])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
    g = gen(seed * 7919 + 17)
    scale = B["scale"][:, None]
    for _ in range(int(steps)):
        p, y = sample_batch(B, g)
        z = model(p)
        loss = (((z - y) / scale) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def score(model: nn.Module, B: dict) -> dict:
    S = summary(model, B)
    rows = S["rows"]
    vals = [float(x["nrmse"]) for x in rows]
    local = [float(x["local_nrmse"]) for x in rows if x["local_nrmse"] is not None]
    return {
        "mean_nrmse": sum(vals) / len(vals),
        "worst_nrmse": max(vals),
        "mean_local_nrmse": sum(local) / len(local),
        "worst_local_nrmse": max(local),
        "detail": S,
    }


def firewall(arm: str, lr: float) -> dict:
    B = bank(cal_scenes(), tp=1024, ep=1024)
    g1 = gen(44101)
    g2 = gen(44102)
    B["y"] = torch.randn(B["y"].shape, generator=g1)
    B["ey"] = torch.randn(B["ey"].shape, generator=g2)
    B["scale"] = B["y"].std(1).clamp_min(0.05)
    model = train_arm(arm, B, lr, steps=STEPS, seed=882211)
    s = score(model, B)
    return {
        "mean_nrmse": s["mean_nrmse"],
        "worst_nrmse": s["worst_nrmse"],
        "valid": s["mean_nrmse"] >= 0.90,
    }


def run() -> dict:
    torch.set_num_threads(1)
    B = bank(cal_scenes())
    all_results = {}
    arm_winners = {}
    parameter_counts = {}

    for arm_index, arm in enumerate(ARMS):
        parameter_counts[arm] = parameter_count(PerSceneQuery(len(B["scenes"]), ARMS[arm]))
        rows = []
        for lr_index, lr in enumerate(LRS):
            model = train_arm(arm, B, lr, steps=STEPS, seed=550000 + arm_index * 1000 + lr_index * 101)
            s = score(model, B)
            rows.append({"lr": lr, **s})
        winner = min(rows, key=lambda x: (round(float(x["mean_nrmse"]), 12), float(x["lr"])))
        valid = (
            float(winner["mean_nrmse"]) <= 0.10
            and float(winner["worst_nrmse"]) <= 0.14
            and float(winner["mean_local_nrmse"]) <= 0.16
        )
        arm_winners[arm] = {
            "lr": winner["lr"],
            "mean_nrmse": winner["mean_nrmse"],
            "worst_nrmse": winner["worst_nrmse"],
            "mean_local_nrmse": winner["mean_local_nrmse"],
            "worst_local_nrmse": winner["worst_local_nrmse"],
            "parameter_count": parameter_counts[arm],
            "surface_calibrated": valid,
        }
        all_results[arm] = rows

    eligible = [arm for arm, row in arm_winners.items() if row["surface_calibrated"]]
    selected = None
    fw = None
    if eligible:
        min_params = min(parameter_counts[a] for a in eligible)
        near = [a for a in eligible if parameter_counts[a] <= min_params * 1.05]
        selected = min(near, key=lambda a: (arm_winners[a]["mean_nrmse"], a))
        fw = firewall(selected, float(arm_winners[selected]["lr"]))

    if not eligible:
        status = "QUERY_CALIBRATION_FAIL__TARGET_LADDER_REQUIRED_V4"
    elif fw is None or not fw["valid"]:
        status = "QUERY_CALIBRATION_INVALID__FIREWALL_FAIL_V4"
    else:
        status = "QUERY_CLASS_SELECTED_V4"

    return {
        "schema": SCHEMA,
        "status": status,
        "configuration": {
            "learning_rates": list(LRS),
            "steps": STEPS,
            "arms": ARMS,
            "target_and_sampling": "EXACT_V3_CALIBRATION_REUSE",
        },
        "parameter_counts": parameter_counts,
        "arm_winners": arm_winners,
        "selected_query_class": selected,
        "firewall": fw,
        "results": all_results,
        "next": (
            "FREEZE_SELECTED_Q_AND_REOPEN_MATCHED_CARRIER_COURT"
            if status == "QUERY_CLASS_SELECTED_V4"
            else "BUILD_PREREGISTERED_SURFACE_TARGET_EASY_TO_HARD_LADDER"
        ),
        "authority": {
            "product_authority_changed": False,
            "truth_builder_training_authorized": False,
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
    print(json.dumps({"status": result["status"], "arm_winners": result["arm_winners"], "selected": result["selected_query_class"], "firewall": result["firewall"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
