#!/usr/bin/env python3
"""Execute CONTINUOUS_JOINT_SYSTEM_PREREG_V1 exactly."""
from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

SCHEMA = "RealSaS.ContinuousJointSystemCourt.v1"
LR = 3.0e-3
STEPS = 6000
WIDTH = 192
DEPTH = 3
FOURIER_BANDS = 4
BATCH = 32
PAIRS = 12
EVAL_PAIRS = 256

BASE = runpy.run_path(str(Path(__file__).with_name("oracle_z_surface_calibration_probe_v3.py")))
SEEDS = tuple(BASE["SEEDS"])
FACTORS = (4.0, 2.0, 1.0)
H_REF = float(BASE["H_REF"])
make_scene = BASE["make_scene"]
bank_base = BASE["bank"]
gen = BASE["gen"]
seed_all = BASE["seed_all"]


def scenes():
    out = []
    for seed in SEEDS:
        for j, factor in enumerate(FACTORS):
            r = factor * H_REF
            out.append(make_scene(seed * 10000 + 100 + j, seed, "SURFACE_STRUCTURED", r))
            out.append(make_scene(seed * 10000 + 200 + j, seed, "THIN_SURFACE", r))
    return out


def support_weights(scene: dict, p: torch.Tensor) -> torch.Tensor:
    q = (p[:, None, :] - scene["centers"][None]) / scene["scales"][None]
    d = torch.linalg.vector_norm(q, dim=-1) - 1.0
    return torch.exp(-12.0 * d.square())


def gp_coefficients(scene: dict) -> torch.Tensor:
    g = gen(int(scene["seed"]) * 97 + 440011)
    return torch.empty(4, 4).uniform_(-0.9, 0.9, generator=g)


def gp_targets(scene: dict, p: torch.Tensor) -> torch.Tensor:
    w = support_weights(scene, p)
    coeff = gp_coefficients(scene)
    return torch.einsum("nk,kc->nc", w, coeff) / 4.0


def build_bank() -> dict:
    B = bank_base(scenes())
    gp_y = []
    gp_ey = []
    for scene, p, ep in zip(B["scenes"], B["p"], B["ep"]):
        gp_y.append(gp_targets(scene, p))
        gp_ey.append(gp_targets(scene, ep))
    B["gp_y"] = torch.stack(gp_y)
    B["gp_ey"] = torch.stack(gp_ey)
    B["gp_scale"] = B["gp_y"].std(1).clamp_min(0.05)
    return B


def fourier_xyz(x: torch.Tensor) -> torch.Tensor:
    feats = [x]
    for band in range(FOURIER_BANDS):
        freq = (2.0 ** band) * math.pi
        feats.extend((torch.sin(freq * x), torch.cos(freq * x)))
    return torch.cat(feats, dim=-1)


def sample_triplane(planes: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    x, y, z = p.unbind(-1)
    grids = (torch.stack((x, y), -1), torch.stack((x, z), -1), torch.stack((y, z), -1))
    out = []
    for i, grid in enumerate(grids):
        v = F.grid_sample(planes[:, i], grid.unsqueeze(2), mode="bilinear", padding_mode="border", align_corners=False).squeeze(-1).transpose(1, 2)
        out.append(v)
    return torch.cat(out, dim=-1)


def sample_dense(grid: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    v = F.grid_sample(grid, p.unsqueeze(2).unsqueeze(2), mode="bilinear", padding_mode="border", align_corners=False).squeeze(-1).squeeze(-1).transpose(1, 2)
    return v.repeat_interleave(3, dim=-1)


class SharedFeatureQ(nn.Module):
    def __init__(self):
        super().__init__()
        inp = 48 + 3 * (1 + 2 * FOURIER_BANDS)
        layers = []
        for i in range(DEPTH):
            layers.extend((nn.Linear(inp if i == 0 else WIDTH, WIDTH), nn.SiLU()))
        self.net = nn.Sequential(*layers)

    def forward(self, carrier_features: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((carrier_features, fourier_xyz(p)), dim=-1))


class PrivateAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.fc1 = nn.Linear(WIDTH, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x):
        return (x + self.fc2(F.silu(self.fc1(self.norm(x))))) * self.scale


class JointModel(nn.Module):
    def __init__(self, carrier_kind: str, scene_count: int, *, private: bool, bound: bool):
        super().__init__()
        self.carrier_kind = carrier_kind
        self.bound = bool(bound)
        S = int(scene_count)
        if carrier_kind == "TRIPLANE":
            self.carrier = nn.Parameter(torch.randn(S, 3, 16, 24, 24) * 0.02)
        elif carrier_kind == "DENSE3D":
            self.carrier = nn.Parameter(torch.randn(S, 16, 12, 12, 12) * 0.02)
        else:
            raise ValueError(f"UNKNOWN_CARRIER:{carrier_kind}")
        self.q = SharedFeatureQ()
        self.gm_adapter = PrivateAdapter() if private else nn.Identity()
        self.gp_adapter = PrivateAdapter() if private else nn.Identity()
        self.gm_head = nn.Linear(WIDTH, 4)
        self.gp_head = nn.Linear(WIDTH, 4)
        self.relation = nn.Sequential(nn.Linear(4 * WIDTH, 2 * WIDTH), nn.SiLU(), nn.Linear(2 * WIDTH, 1))

    def carrier_features(self, p: torch.Tensor, carrier: torch.Tensor | None = None) -> torch.Tensor:
        c = self.carrier if carrier is None else carrier
        return sample_triplane(c, p) if self.carrier_kind == "TRIPLANE" else sample_dense(c, p)

    def features(self, p: torch.Tensor, carrier: torch.Tensor | None = None):
        h0 = self.q(self.carrier_features(p, carrier), p)
        return h0, self.gm_adapter(h0), self.gp_adapter(h0)

    def local(self, p: torch.Tensor, carrier: torch.Tensor | None = None):
        h0, gm, gp = self.features(p, carrier)
        return h0, gm, gp, self.gm_head(gm), self.gp_head(gp)

    def relation_logits(self, gp: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        S, _, D = gp.shape
        ga = torch.gather(gp, 1, a[..., None].expand(-1, -1, D))
        gb = torch.gather(gp, 1, b[..., None].expand(-1, -1, D))
        if not self.bound:
            mean = gp.mean(1, keepdim=True).expand(-1, a.shape[1], -1)
            ga = mean; gb = mean
        return self.relation(torch.cat((ga, gb, ga - gb, ga * gb), dim=-1)).squeeze(-1)


def nparams(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def training_batch(B: dict, g: torch.Generator):
    S, N, _ = B["p"].shape
    idx = torch.randint(0, N, (S, BATCH), generator=g)
    def gather(x):
        return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
    p = gather(B["p"]); gm = gather(B["y"]); gp = gather(B["gp_y"])
    a = torch.randint(0, BATCH, (S, PAIRS), generator=g)
    b = torch.randint(0, BATCH - 1, (S, PAIRS), generator=g); b = b + (b >= a).long()
    bi = torch.arange(S)[:, None]
    rel = (gp[bi, a, 1] > gp[bi, b, 1]).float()
    return p, gm, gp, a, b, rel


def train_arm(carrier: str, B: dict, arm: str, seed: int):
    private = arm not in ("FULL_SHARED_JOINT_BOUND",)
    bound = arm not in ("S2_JOINT_UNBOUND",)
    model = JointModel(carrier, len(B["scenes"]), private=private, bound=bound)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    g = gen(seed * 7919 + 17)
    gm_scale = B["scale"][:, None]
    gp_scale = B["gp_scale"][:, None]
    mode = "joint"
    if arm == "GM_SINGLE_ORACLE": mode = "gm"
    elif arm == "GP_SINGLE_ORACLE": mode = "gp"
    random_relation = arm == "S2_JOINT_BOUND_RANDOM_RELATION"
    for _ in range(STEPS):
        p, gm_t, gp_t, a, b, rel_t = training_batch(B, g)
        _, gm_h, gp_h, gm_p, gp_p = model.local(p)
        losses = []
        if mode in ("joint", "gm"):
            losses.append((((gm_p - gm_t) / gm_scale) ** 2).mean())
        if mode in ("joint", "gp"):
            losses.append((((gp_p - gp_t) / gp_scale) ** 2).mean())
        if mode == "joint":
            logits = model.relation_logits(gp_h, a, b)
            if random_relation:
                rel_t = torch.randint(0, 2, rel_t.shape, generator=g).float()
            losses.append(F.binary_cross_entropy_with_logits(logits, rel_t))
        loss = sum(losses)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model


def nrmse(pred, truth, scale):
    return (torch.sqrt(((pred - truth) ** 2).mean(1)) / scale).mean(-1)


def evaluate(model: JointModel, B: dict, *, random_relation: bool = False, shuffled_carrier: bool = False):
    model.eval(); carrier = torch.roll(model.carrier, 1, 0) if shuffled_carrier else None
    with torch.no_grad():
        _, gm_h, gp_h, gm_p, gp_p = model.local(B["ep"], carrier)
        gm_vals = nrmse(gm_p, B["ey"], B["scale"])
        gp_vals = nrmse(gp_p, B["gp_ey"], B["gp_scale"])
        S, N, _ = B["ep"].shape
        rg = gen(990123 if not random_relation else 990987)
        a = torch.randint(0, N, (S, EVAL_PAIRS), generator=rg)
        b = torch.randint(0, N - 1, (S, EVAL_PAIRS), generator=rg); b = b + (b >= a).long()
        bi = torch.arange(S)[:, None]
        truth = (B["gp_ey"][bi, a, 1] > B["gp_ey"][bi, b, 1]).float()
        if random_relation:
            truth = torch.randint(0, 2, truth.shape, generator=rg).float()
        logits = model.relation_logits(gp_h, a, b)
        acc = float(((logits >= 0) == (truth >= 0.5)).float().mean())
        pos = float(truth.mean())
    return {
        "gm_mean_nrmse": float(gm_vals.mean()),
        "gm_worst_nrmse": float(gm_vals.max()),
        "gp_mean_nrmse": float(gp_vals.mean()),
        "gp_worst_nrmse": float(gp_vals.max()),
        "relation_accuracy": acc,
        "relation_positive_fraction": pos,
    }


def run(selection_path: Path) -> dict:
    torch.set_num_threads(1)
    selection = json.loads(selection_path.read_text())
    carrier = selection.get("selected_carrier")
    if carrier not in ("TRIPLANE", "DENSE3D"):
        return {"schema": SCHEMA, "status": "CONTINUOUS_JOINT_SYSTEM_BLOCKED__CARRIER_UNRESOLVED", "selection": selection}
    B = build_bank()
    arms = (
        "GM_SINGLE_ORACLE", "GP_SINGLE_ORACLE", "S2_JOINT_BOUND",
        "FULL_SHARED_JOINT_BOUND", "S2_JOINT_UNBOUND", "S2_JOINT_BOUND_RANDOM_RELATION",
    )
    results = {}; models = {}
    for i, arm in enumerate(arms):
        seed_all(720000 + i * 101)
        model = train_arm(carrier, B, arm, 720000 + i * 101)
        models[arm] = model
        results[arm] = {
            "parameter_count": nparams(model),
            "metrics": evaluate(model, B, random_relation=(arm == "S2_JOINT_BOUND_RANDOM_RELATION")),
        }

    selected = results["S2_JOINT_BOUND"]["metrics"]
    gm_single = results["GM_SINGLE_ORACLE"]["metrics"]
    gp_single = results["GP_SINGLE_ORACLE"]["metrics"]
    full = results["FULL_SHARED_JOINT_BOUND"]["metrics"]
    unbound = results["S2_JOINT_UNBOUND"]["metrics"]
    firewall = results["S2_JOINT_BOUND_RANDOM_RELATION"]["metrics"]

    gm_interference = selected["gm_mean_nrmse"] / max(gm_single["gm_mean_nrmse"], 1e-12) - 1.0
    gp_interference = selected["gp_mean_nrmse"] / max(gp_single["gp_mean_nrmse"], 1e-12) - 1.0
    shuffled = evaluate(models["S2_JOINT_BOUND"], B, shuffled_carrier=True)
    ordinary_local = 0.5 * (selected["gm_mean_nrmse"] + selected["gp_mean_nrmse"])
    shuffled_local = 0.5 * (shuffled["gm_mean_nrmse"] + shuffled["gp_mean_nrmse"])
    shuffle_gain = shuffled_local / max(ordinary_local, 1e-12) - 1.0

    selected_ok = selected["gm_mean_nrmse"] <= 0.15 and selected["gp_mean_nrmse"] <= 0.15 and selected["relation_accuracy"] >= 0.95 and 0.35 <= selected["relation_positive_fraction"] <= 0.65
    interference_ok = gm_interference <= 0.15 and gp_interference <= 0.15
    unbound_ok = unbound["relation_accuracy"] <= 0.60 and unbound["gm_mean_nrmse"] <= selected["gm_mean_nrmse"] * 1.15 and unbound["gp_mean_nrmse"] <= selected["gp_mean_nrmse"] * 1.15
    firewall_ok = 0.48 <= firewall["relation_accuracy"] <= 0.52
    shuffle_ok = shuffle_gain >= 0.25
    full_not_better_both = not (full["gm_mean_nrmse"] <= selected["gm_mean_nrmse"] * 0.90 and full["gp_mean_nrmse"] <= selected["gp_mean_nrmse"] * 0.90)

    gates = {
        "selected_joint": selected_ok,
        "joint_negative_transfer_le_15pct": interference_ok,
        "endpoint_binding_owner": unbound_ok,
        "random_relation_firewall": firewall_ok,
        "scene_carrier_shuffle_dependence": shuffle_ok,
        "full_shared_not_10pct_better_on_both": full_not_better_both,
    }
    status = "CONTINUOUS_JOINT_SYSTEM_CLOSED_TINY_V1" if all(gates.values()) else "CONTINUOUS_JOINT_SYSTEM_NOT_CLOSED_TINY_V1"
    return {
        "schema": SCHEMA,
        "status": status,
        "selected_carrier": carrier,
        "configuration": {"lr": LR, "steps": STEPS, "width": WIDTH, "depth": DEPTH, "fourier_bands": FOURIER_BANDS, "factors": list(FACTORS), "seeds": list(SEEDS)},
        "results": results,
        "interference": {"gm_relative": gm_interference, "gp_relative": gp_interference},
        "selected_carrier_shuffled_metrics": shuffled,
        "selected_carrier_shuffle_relative_local_error_increase": shuffle_gain,
        "gates": gates,
        "authority": {"product_authority_changed": False, "truth_builder_training_authorized": False, "fit8_training_authorized": False},
    }


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--selection", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    r = run(a.selection); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": r["status"], "selected_carrier": r.get("selected_carrier"), "gates": r.get("gates"), "interference": r.get("interference"), "out": str(a.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
