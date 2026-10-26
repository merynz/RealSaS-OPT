#!/usr/bin/env python3
"""Execute ORACLE_Z_MATCHED_CARRIER_PREREG_V5 exactly."""
from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

SCHEMA = "RealSaS.OracleZMatchedCarrierCourt.v5"
LR = 3.0e-3
DIRECT_STEPS = 5000
CONTROL_STEPS = 5000
CARRIER_STEPS = 8000
WIDTH = 192
DEPTH = 3
FOURIER_BANDS = 4

BASE = runpy.run_path(str(Path(__file__).with_name("oracle_z_surface_calibration_probe_v3.py")))
SEEDS = tuple(BASE["SEEDS"])
FACTORS = tuple(BASE["FACTORS"])
H_REF = float(BASE["H_REF"])
OUT = int(BASE["OUT"])
bank = BASE["bank"]
scenes_for = BASE["scenes_for"]
sample_batch = BASE["sample_batch"]
seed_all = BASE["seed_all"]
gen = BASE["gen"]


def fourier_xyz(x: torch.Tensor) -> torch.Tensor:
    feats = [x]
    for band in range(FOURIER_BANDS):
        freq = (2.0 ** band) * math.pi
        feats.extend((torch.sin(freq * x), torch.cos(freq * x)))
    return torch.cat(feats, dim=-1)


class SharedQuery(nn.Module):
    def __init__(self, carrier_dim: int = 48):
        super().__init__()
        inp = int(carrier_dim) + 3 * (1 + 2 * FOURIER_BANDS)
        layers = []
        for i in range(DEPTH):
            layers.extend((nn.Linear(inp if i == 0 else WIDTH, WIDTH), nn.SiLU()))
        layers.append(nn.Linear(WIDTH, OUT))
        self.net = nn.Sequential(*layers)

    def forward(self, carrier: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((carrier, fourier_xyz(p)), dim=-1))


class PerSceneDirectQ(nn.Module):
    """Vectorized per-scene direct upper bound using the V4-selected Q class."""
    def __init__(self, scene_count: int):
        super().__init__()
        inp = 3 * (1 + 2 * FOURIER_BANDS)
        S = int(scene_count)
        self.w1 = nn.Parameter(torch.empty(S, WIDTH, inp)); self.b1 = nn.Parameter(torch.zeros(S, WIDTH))
        self.w2 = nn.Parameter(torch.empty(S, WIDTH, WIDTH)); self.b2 = nn.Parameter(torch.zeros(S, WIDTH))
        self.w3 = nn.Parameter(torch.empty(S, WIDTH, WIDTH)); self.b3 = nn.Parameter(torch.zeros(S, WIDTH))
        self.w4 = nn.Parameter(torch.empty(S, OUT, WIDTH)); self.b4 = nn.Parameter(torch.zeros(S, OUT))
        for w in (self.w1, self.w2, self.w3, self.w4):
            for s in range(S):
                nn.init.kaiming_uniform_(w[s], a=math.sqrt(5))

    @staticmethod
    def lin(x, w, b):
        return torch.einsum("sni,soi->sno", x, w) + b[:, None]

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        h = fourier_xyz(p)
        h = F.silu(self.lin(h, self.w1, self.b1))
        h = F.silu(self.lin(h, self.w2, self.b2))
        h = F.silu(self.lin(h, self.w3, self.b3))
        return self.lin(h, self.w4, self.b4)


class CoordinateOnly(nn.Module):
    def __init__(self):
        super().__init__(); self.q = SharedQuery(48)
    def forward(self, p: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros((*p.shape[:-1], 48), device=p.device, dtype=p.dtype)
        return self.q(zeros, p)


def sample_triplane(planes: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    # planes [S,3,16,24,24], p [S,N,3] -> [S,N,48]
    x, y, z = p.unbind(-1)
    grids = (
        torch.stack((x, y), dim=-1),
        torch.stack((x, z), dim=-1),
        torch.stack((y, z), dim=-1),
    )
    out = []
    for i, grid in enumerate(grids):
        v = F.grid_sample(
            planes[:, i],
            grid.unsqueeze(2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(-1).transpose(1, 2)
        out.append(v)
    return torch.cat(out, dim=-1)


def sample_dense(grid: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    v = F.grid_sample(
        grid,
        p.unsqueeze(2).unsqueeze(2),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(-1).squeeze(-1).transpose(1, 2)
    # Fixed non-trainable expansion to the same 48-D shared-Q input width.
    return v.repeat_interleave(3, dim=-1)


class CarrierSharedQ(nn.Module):
    def __init__(self, arm: str, scene_count: int):
        super().__init__(); self.arm = arm
        S = int(scene_count)
        if arm == "TRIPLANE_SHARED_Q":
            self.carrier = nn.Parameter(torch.randn(S, 3, 16, 24, 24) * 0.02)
        elif arm == "DENSE3D_SHARED_Q":
            self.carrier = nn.Parameter(torch.randn(S, 16, 12, 12, 12) * 0.02)
        else:
            raise ValueError(arm)
        self.q = SharedQuery(48)

    def features(self, p: torch.Tensor, carrier: torch.Tensor | None = None) -> torch.Tensor:
        c = self.carrier if carrier is None else carrier
        if self.arm == "TRIPLANE_SHARED_Q":
            return sample_triplane(c, p)
        return sample_dense(c, p)

    def forward(self, p: torch.Tensor, *, carrier: torch.Tensor | None = None) -> torch.Tensor:
        return self.q(self.features(p, carrier), p)

    def shuffled_forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.forward(p, carrier=torch.roll(self.carrier, shifts=1, dims=0))


def nparams(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train(model: nn.Module, B: dict, *, steps: int, seed: int) -> nn.Module:
    seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    g = gen(seed * 7919 + 17)
    scale = B["scale"][:, None]
    model.train()
    for _ in range(int(steps)):
        p, y = sample_batch(B, g)
        z = model(p)
        loss = (((z - y) / scale) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def nrmse(pred: torch.Tensor, truth: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (torch.sqrt(((pred - truth) ** 2).mean(1)) / scale).mean(-1)


def summarize(model: nn.Module, B: dict, *, shuffled: bool = False) -> dict:
    model.eval(); rows = []
    with torch.no_grad():
        pred = model.shuffled_forward(B["ep"]) if shuffled else model(B["ep"])
        vals = nrmse(pred, B["ey"], B["scale"])
        for i, scene in enumerate(B["scenes"]):
            mask = B["mask"][i]
            local = None
            if int(mask.sum()) >= 16:
                local = float(nrmse(pred[i][mask][None], B["ey"][i][mask][None], B["scale"][i][None])[0])
            rows.append({
                "seed": scene["rep"],
                "family": scene["family"],
                "radius": float(scene["radius"]),
                "nrmse": float(vals[i]),
                "local_nrmse": local,
                "local_count": int(mask.sum()),
            })
    groups = {}
    for family, radius in sorted({(x["family"], x["radius"]) for x in rows}):
        q = [x for x in rows if x["family"] == family and x["radius"] == radius]
        loc = [float(x["local_nrmse"]) for x in q if x["local_nrmse"] is not None]
        groups[f"{family}:{radius:g}"] = {
            "family": family,
            "radius": radius,
            "mean_nrmse": sum(float(x["nrmse"]) for x in q) / len(q),
            "worst_nrmse": max(float(x["nrmse"]) for x in q),
            "mean_local_nrmse": sum(loc) / len(loc),
            "worst_local_nrmse": max(loc),
            "per_seed": [float(x["nrmse"]) for x in q],
        }
    overall = sum(float(x["nrmse"]) for x in rows) / len(rows)
    return {"rows": rows, "groups": groups, "overall_mean_nrmse": overall}


def support(row: dict) -> bool:
    return (
        float(row["mean_nrmse"]) <= 0.12
        and float(row["worst_nrmse"]) <= 0.18
        and float(row["mean_local_nrmse"]) <= 0.18
    )


def run() -> dict:
    torch.set_num_threads(1)
    B = bank(scenes_for(SEEDS))
    S = len(B["scenes"])

    seed_all(610001)
    U = train(PerSceneDirectQ(S), B, steps=DIRECT_STEPS, seed=610001)
    direct = summarize(U, B)

    seed_all(610101)
    N = train(CoordinateOnly(), B, steps=CONTROL_STEPS, seed=610101)
    coord_only = summarize(N, B)

    carrier_results = {}
    carrier_models = {}
    for arm, seed in (("TRIPLANE_SHARED_Q", 610201), ("DENSE3D_SHARED_Q", 610301)):
        seed_all(seed)
        model = train(CarrierSharedQ(arm, S), B, steps=CARRIER_STEPS, seed=seed)
        carrier_models[arm] = model
        ordinary = summarize(model, B)
        shuffled = summarize(model, B, shuffled=True)
        gain = shuffled["overall_mean_nrmse"] / max(ordinary["overall_mean_nrmse"], 1.0e-12) - 1.0
        carrier_results[arm] = {
            "ordinary": ordinary,
            "scene_carrier_shuffled": shuffled,
            "shuffle_relative_error_increase": gain,
            "shuffle_dependence_pass": gain >= 0.25,
        }

    carrier_param_counts = {
        arm: int(carrier_models[arm].carrier.numel())
        for arm in carrier_models
    }
    shared_q_param_counts = {
        arm: nparams(carrier_models[arm].q)
        for arm in carrier_models
    }
    matched = len(set(carrier_param_counts.values())) == 1 and len(set(shared_q_param_counts.values())) == 1

    points = {}
    dense_concerns = []
    eligible_count = 0
    coord_support_count = 0
    for family in ("SURFACE_STRUCTURED", "THIN_SURFACE"):
        ordered = []
        for factor in FACTORS:
            radius = factor * H_REF
            key = f"{family}:{radius:g}"
            u = direct["groups"][key]
            n = coord_only["groups"][key]
            t = carrier_results["TRIPLANE_SHARED_Q"]["ordinary"]["groups"][key]
            d = carrier_results["DENSE3D_SHARED_Q"]["ordinary"]["groups"][key]
            target_valid = support(u)
            t_ok = target_valid and support(t)
            d_ok = target_valid and support(d)
            n_ok = target_valid and support(n)
            if target_valid: eligible_count += 1
            if n_ok: coord_support_count += 1
            gain = (float(t["mean_nrmse"]) - float(d["mean_nrmse"])) / max(float(t["mean_nrmse"]), 1.0e-12)
            concern = target_valid and d_ok and (not t_ok) and gain >= 0.25
            points[key] = {
                "target_valid": target_valid,
                "coordinate_only_supported": n_ok,
                "triplane_supported": t_ok,
                "dense_supported": d_ok,
                "dense_relative_mean_gain": gain,
                "credible_dense_concern": concern,
            }
            ordered.append((factor, concern))
        for a, b in zip(ordered, ordered[1:]):
            if a[1] and b[1]:
                dense_concerns.append({"family": family, "factor_a": a[0], "factor_b": b[0]})

    easy_keys = [
        f"SURFACE_STRUCTURED:{4.0 * H_REF:g}",
        f"THIN_SURFACE:{4.0 * H_REF:g}",
    ]
    easy_target_valid = all(points[k]["target_valid"] for k in easy_keys)
    at_least_one_carrier_easy = (
        all(points[k]["triplane_supported"] for k in easy_keys)
        or all(points[k]["dense_supported"] for k in easy_keys)
    )
    coord_negative_ok = eligible_count > 0 and coord_support_count < eligible_count
    successful_carriers = []
    for arm in ("TRIPLANE_SHARED_Q", "DENSE3D_SHARED_Q"):
        label = "triplane_supported" if arm.startswith("TRIPLANE") else "dense_supported"
        if any(v[label] for v in points.values() if v["target_valid"]):
            successful_carriers.append(arm)
    shuffle_ok = all(carrier_results[a]["shuffle_dependence_pass"] for a in successful_carriers)
    apparatus_valid = matched and easy_target_valid and at_least_one_carrier_easy and coord_negative_ok and shuffle_ok

    if not apparatus_valid:
        status = "MATCHED_CARRIER_COURT_INCONCLUSIVE_V5"
        selected = None
    elif dense_concerns:
        status = "DENSE3D_SELECTED_TINY_V5"
        selected = "DENSE3D"
    else:
        status = "TRIPLANE_RETAINED_PROVISIONAL_TINY_V5"
        selected = "TRIPLANE"

    return {
        "schema": SCHEMA,
        "status": status,
        "selected_carrier": selected,
        "configuration": {
            "seeds": list(SEEDS),
            "factors": list(FACTORS),
            "h_ref": H_REF,
            "lr": LR,
            "direct_steps": DIRECT_STEPS,
            "control_steps": CONTROL_STEPS,
            "carrier_steps": CARRIER_STEPS,
            "query": {"width": WIDTH, "depth": DEPTH, "fourier_bands": FOURIER_BANDS},
        },
        "parameter_counts": {
            "carrier": carrier_param_counts,
            "shared_query": shared_q_param_counts,
            "carrier_exactly_matched": matched,
        },
        "direct_upper_bound": direct,
        "coordinate_only_negative_control": coord_only,
        "carriers": carrier_results,
        "points": points,
        "dense_adjacent_material_concerns": dense_concerns,
        "gates": {
            "easy_target_valid": easy_target_valid,
            "at_least_one_carrier_supports_both_easy_points": at_least_one_carrier_easy,
            "coordinate_only_negative_control": coord_negative_ok,
            "successful_carrier_scene_shuffle_dependence": shuffle_ok,
            "apparatus_valid": apparatus_valid,
        },
        "authority": {
            "product_authority_changed": False,
            "truth_builder_training_authorized": False,
            "fit8_training_authorized": False,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    r = run(); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": r["status"],
        "selected_carrier": r["selected_carrier"],
        "parameter_counts": r["parameter_counts"],
        "gates": r["gates"],
        "dense_adjacent_material_concerns": r["dense_adjacent_material_concerns"],
        "out": str(a.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
