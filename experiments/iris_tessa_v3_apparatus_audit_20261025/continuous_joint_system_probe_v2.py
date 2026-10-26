#!/usr/bin/env python3
"""Execute CONTINUOUS_JOINT_SYSTEM_PREREG_V2: only relation-gradient ownership changes."""
from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

SCHEMA = "RealSaS.ContinuousJointSystemCourt.v2"
LR = 3.0e-3
STEPS = 6000
BATCH = 32
TRAIN_POOL = 4096
EVAL_POINTS = 2048
WIDTH = 192
DEPTH = 3
FOURIER_BANDS = 4
ADAPTER_BOTTLENECK = 64
RELATION_HIDDEN = 128
ARM_SEEDS = {
    "GM_SINGLE_ORACLE": 720101,
    "GP_SINGLE_ORACLE": 720211,
    "S2_JOINT_BOUND": 720307,
    "FULL_SHARED_JOINT_BOUND": 720401,
    "S2_JOINT_UNBOUND": 720503,
    "S2_JOINT_BOUND_RANDOM_RELATION": 720601,
}

HERE = Path(__file__).resolve().parent
V5 = runpy.run_path(str(HERE / "oracle_z_matched_carrier_probe_v5.py"))
V3 = V5["BASE"]
SEEDS = tuple(V5["SEEDS"])
H_REF = float(V5["H_REF"])
seed_all = V5["seed_all"]
gen = V5["gen"]
scenes_for = V5["scenes_for"]
sample = V3["sample"]
gm_targets = V3["targets"]
sample_triplane = V5["sample_triplane"]
sample_dense = V5["sample_dense"]


def fourier_xyz(x: torch.Tensor) -> torch.Tensor:
    feats = [x]
    for band in range(FOURIER_BANDS):
        freq = (2.0 ** band) * math.pi
        feats.extend((torch.sin(freq * x), torch.cos(freq * x)))
    return torch.cat(feats, dim=-1)


def load_selection(selection_path: Path) -> dict:
    if not selection_path.exists():
        raise RuntimeError("ORACLE_Z_FINAL_CARRIER_SELECTION_V6_MISSING")
    row = json.loads(selection_path.read_text())
    if row.get("schema") != "RealSaS.OracleZFinalCarrierSelection.v6":
        raise RuntimeError("ORACLE_Z_FINAL_CARRIER_SELECTION_V6_SCHEMA_INVALID")
    if row.get("status") != "FINAL_PRETRUTH_CARRIER_SELECTION_V6":
        raise RuntimeError("ORACLE_Z_FINAL_CARRIER_SELECTION_V6_STATUS_INVALID")
    selected = row.get("selected_carrier")
    if selected not in {"DENSE3D", "TRIPLANE"}:
        raise RuntimeError("ORACLE_Z_FINAL_CARRIER_SELECTION_V6_VALUE_INVALID")
    gates = row.get("evidence", {}).get("gates", {})
    if not gates or not all(bool(v) for v in gates.values()):
        raise RuntimeError("ORACLE_Z_FINAL_CARRIER_SELECTION_V6_GATES_INVALID")
    if not bool(row.get("authority", {}).get("continuous_joint_system_dependency_authorized")):
        raise RuntimeError("ORACLE_Z_FINAL_CARRIER_SELECTION_V6_AUTHORITY_INVALID")
    return row


def valid_scenes() -> list[dict]:
    out = []
    for scene in scenes_for(SEEDS):
        factor = round(float(scene["radius"]) / H_REF, 6)
        if factor in (1.0, 2.0, 4.0):
            out.append(scene)
    if len(out) != len(SEEDS) * 3 * 2:
        raise RuntimeError("CONTINUOUS_JOINT_SCENE_BANK_INVALID")
    return out


def gp_coefficients(scene: dict) -> torch.Tensor:
    g = gen(13_900_000 + int(scene["seed"]) * 37 + int(scene["rep"]) * 101)
    return torch.empty(4, 4).uniform_(-0.85, 0.85, generator=g)


def gp_targets(scene: dict, p: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    q = (p[:, None, :] - scene["centers"][None]) / scene["scales"][None]
    d = torch.linalg.vector_norm(q, dim=-1) - 1.0
    w = torch.exp(-12.0 * d.square())
    return torch.einsum("nk,kc->nc", w, coeff) / 4.0


def build_bank() -> dict:
    scenes = valid_scenes()
    ps, gm_y, gp_y, eps, gm_ey, gp_ey, masks = [], [], [], [], [], [], []
    for i, scene in enumerate(scenes):
        coeff = gp_coefficients(scene)
        p = sample(scene, TRAIN_POOL, gen(7_300_000 + int(scene["seed"]) * 13 + i))
        ep = sample(scene, EVAL_POINTS, gen(9_300_000 + int(scene["seed"]) * 17 + i))
        gy = gm_targets(scene, p)
        py = gp_targets(scene, p, coeff)
        gey = gm_targets(scene, ep)
        pey = gp_targets(scene, ep, coeff)
        ps.append(p); gm_y.append(gy); gp_y.append(py)
        eps.append(ep); gm_ey.append(gey); gp_ey.append(pey)
        masks.append(gey[:, 0] >= 0.05)
    P = torch.stack(ps); GMY = torch.stack(gm_y); GPY = torch.stack(gp_y)
    EP = torch.stack(eps); GMEY = torch.stack(gm_ey); GPEY = torch.stack(gp_ey)
    return {
        "scenes": scenes,
        "p": P,
        "gm_y": GMY,
        "gp_y": GPY,
        "gm_scale": GMY.std(1).clamp_min(0.05),
        "gp_scale": GPY.std(1).clamp_min(0.05),
        "ep": EP,
        "gm_ey": GMEY,
        "gp_ey": GPEY,
        "mask": torch.stack(masks),
    }


def gather_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))


def training_pair(B: dict, g: torch.Generator) -> dict:
    S, N, _ = B["p"].shape
    ia = torch.randint(0, N, (S, BATCH), generator=g)
    offset = torch.randint(1, N, (S, BATCH), generator=g)
    ib = (ia + offset) % N
    a = gather_points(B["p"], ia)
    b = gather_points(B["p"], ib)
    gm_a = gather_points(B["gm_y"], ia)
    gp_a = gather_points(B["gp_y"], ia)
    gp_b = gather_points(B["gp_y"], ib)
    relation = (gp_a[..., 1] > gp_b[..., 1]).float()
    random_relation = torch.randint(0, 2, relation.shape, generator=g).float()
    return {"a": a, "b": b, "gm_a": gm_a, "gp_a": gp_a, "relation": relation, "random_relation": random_relation}


class SharedCarrierQuery(nn.Module):
    def __init__(self, carrier_type: str, scene_count: int):
        super().__init__()
        self.carrier_type = carrier_type
        S = int(scene_count)
        if carrier_type == "DENSE3D":
            self.carrier = nn.Parameter(torch.randn(S, 16, 12, 12, 12) * 0.02)
        elif carrier_type == "TRIPLANE":
            self.carrier = nn.Parameter(torch.randn(S, 3, 16, 24, 24) * 0.02)
        else:
            raise ValueError(carrier_type)
        inp = 48 + 3 * (1 + 2 * FOURIER_BANDS)
        layers = []
        for i in range(DEPTH):
            layers.extend((nn.Linear(inp if i == 0 else WIDTH, WIDTH), nn.SiLU()))
        self.query = nn.Sequential(*layers)

    def carrier_features(self, p: torch.Tensor, carrier: torch.Tensor | None = None) -> torch.Tensor:
        c = self.carrier if carrier is None else carrier
        if self.carrier_type == "DENSE3D":
            return sample_dense(c, p)
        return sample_triplane(c, p)

    def encode(self, p: torch.Tensor, carrier: torch.Tensor | None = None) -> torch.Tensor:
        return self.query(torch.cat((self.carrier_features(p, carrier), fourier_xyz(p)), dim=-1))

    def global_features(self, carrier: torch.Tensor | None = None) -> torch.Tensor:
        c = self.carrier if carrier is None else carrier
        if self.carrier_type == "DENSE3D":
            return c.mean(dim=(-1, -2, -3)).repeat_interleave(3, dim=-1)
        return c.mean(dim=(-1, -2)).reshape(c.shape[0], -1)


class ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(WIDTH, ADAPTER_BOTTLENECK)
        self.up = nn.Linear(ADAPTER_BOTTLENECK, WIDTH)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(F.silu(self.down(h)))


class RelationHead(nn.Module):
    def __init__(self, inp: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, RELATION_HIDDEN), nn.SiLU(), nn.Linear(RELATION_HIDDEN, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CourtModel(nn.Module):
    def __init__(self, arm: str, carrier_type: str, scene_count: int):
        super().__init__()
        self.arm = arm
        self.base = SharedCarrierQuery(carrier_type, scene_count)
        self.has_gm = arm != "GP_SINGLE_ORACLE"
        self.has_gp = arm != "GM_SINGLE_ORACLE"
        self.full_shared = arm == "FULL_SHARED_JOINT_BOUND"
        self.unbound = arm == "S2_JOINT_UNBOUND"
        self.has_relation = arm not in {"GM_SINGLE_ORACLE", "GP_SINGLE_ORACLE"}
        if self.has_gm and not self.full_shared:
            self.gm_adapter = ResidualAdapter()
        if self.has_gp and not self.full_shared:
            self.gp_adapter = ResidualAdapter()
        if self.has_gm:
            self.gm_head = nn.Linear(WIDTH, 4)
        if self.has_gp:
            self.gp_head = nn.Linear(WIDTH, 4)
        if self.has_relation:
            if self.unbound:
                self.global_gp = nn.Sequential(nn.Linear(48, WIDTH), nn.SiLU())
                self.relation = RelationHead(WIDTH)
            else:
                self.relation = RelationHead(WIDTH * 2)

    def private_states(self, h: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        gm_h = None; gp_h = None
        if self.has_gm:
            gm_h = h if self.full_shared else self.gm_adapter(h)
        if self.has_gp:
            gp_h = h if self.full_shared else self.gp_adapter(h)
        return gm_h, gp_h

    def local(self, p: torch.Tensor, carrier: torch.Tensor | None = None) -> dict:
        h = self.base.encode(p, carrier)
        gm_h, gp_h = self.private_states(h)
        out = {}
        if gm_h is not None: out["gm"] = self.gm_head(gm_h)
        if gp_h is not None: out["gp"] = self.gp_head(gp_h)
        return out

    def pair(self, a: torch.Tensor, b: torch.Tensor, carrier: torch.Tensor | None = None) -> dict:
        ha = self.base.encode(a, carrier)
        gm_a, gp_a = self.private_states(ha)
        out = {}
        if gm_a is not None: out["gm"] = self.gm_head(gm_a)
        if gp_a is not None: out["gp"] = self.gp_head(gp_a)
        if self.has_relation:
            if self.unbound:
                global_h = self.global_gp(self.base.global_features(carrier).detach())
                logits = self.relation(global_h)[:, None].expand(-1, a.shape[1])
            else:
                hb = self.base.encode(b, carrier)
                _, gp_b = self.private_states(hb)
                if gp_a is None or gp_b is None:
                    raise RuntimeError("BOUND_RELATION_REQUIRES_GP_STATE")
                logits = self.relation(torch.cat((gp_a.detach(), gp_b.detach()), dim=-1))
            out["relation_logits"] = logits
        return out


def nparams(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train_arm(arm: str, carrier_type: str, B: dict) -> CourtModel:
    seed = ARM_SEEDS[arm]
    seed_all(seed)
    model = CourtModel(arm, carrier_type, len(B["scenes"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    g = gen(seed * 7919 + 17)
    gm_scale = B["gm_scale"][:, None]
    gp_scale = B["gp_scale"][:, None]
    model.train()
    for step in range(STEPS):
        batch = training_pair(B, g)
        out = model.pair(batch["a"], batch["b"])
        losses = []
        if "gm" in out:
            losses.append((((out["gm"] - batch["gm_a"]) / gm_scale) ** 2).mean())
        if "gp" in out:
            losses.append((((out["gp"] - batch["gp_a"]) / gp_scale) ** 2).mean())
        if "relation_logits" in out:
            label = batch["random_relation"] if arm == "S2_JOINT_BOUND_RANDOM_RELATION" else batch["relation"]
            losses.append(F.binary_cross_entropy_with_logits(out["relation_logits"], label))
        loss = sum(losses)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if (step + 1) % 1000 == 0:
            print(json.dumps({"arm": arm, "step": step + 1, "steps": STEPS}), flush=True)
    return model


def nrmse(pred: torch.Tensor, truth: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (torch.sqrt(((pred - truth) ** 2).mean(1)) / scale).mean(-1)


def balanced_random_eval_labels(scene_count: int, n: int) -> torch.Tensor:
    rows = []
    for s in range(scene_count):
        base = torch.cat((torch.zeros(n // 2), torch.ones(n - n // 2)))
        perm = torch.randperm(n, generator=gen(24_100_000 + s * 313))
        rows.append(base[perm])
    return torch.stack(rows)


def summarize(model: CourtModel, B: dict, *, carrier: torch.Tensor | None = None) -> dict:
    model.eval(); out = {}
    with torch.no_grad():
        local = model.local(B["ep"], carrier)
        local_errors = []
        if "gm" in local:
            gm_scene = nrmse(local["gm"], B["gm_ey"], B["gm_scale"])
            gm_local = []
            for i in range(len(B["scenes"])):
                mask = B["mask"][i]
                if int(mask.sum()) >= 16:
                    gm_local.append(float(nrmse(local["gm"][i][mask][None], B["gm_ey"][i][mask][None], B["gm_scale"][i][None])[0]))
            out["gm_overall_mean_nrmse"] = float(gm_scene.mean())
            out["gm_surface_local_mean_nrmse"] = sum(gm_local) / len(gm_local)
            local_errors.extend(gm_local)
        if "gp" in local:
            gp_scene = nrmse(local["gp"], B["gp_ey"], B["gp_scale"])
            gp_local = []
            for i in range(len(B["scenes"])):
                mask = B["mask"][i]
                if int(mask.sum()) >= 16:
                    gp_local.append(float(nrmse(local["gp"][i][mask][None], B["gp_ey"][i][mask][None], B["gp_scale"][i][None])[0]))
            out["gp_overall_mean_nrmse"] = float(gp_scene.mean())
            out["gp_surface_local_mean_nrmse"] = sum(gp_local) / len(gp_local)
            local_errors.extend(gp_local)
        if local_errors:
            out["mean_surface_local_nrmse"] = sum(local_errors) / len(local_errors)
        if model.has_relation:
            a = B["ep"]
            b = torch.roll(B["ep"], shifts=EVAL_POINTS // 2, dims=1)
            gp_a = B["gp_ey"]
            gp_b = torch.roll(B["gp_ey"], shifts=EVAL_POINTS // 2, dims=1)
            target = (gp_a[..., 1] > gp_b[..., 1]).float()
            logits = model.pair(a, b, carrier)["relation_logits"]
            if model.arm == "S2_JOINT_BOUND_RANDOM_RELATION":
                target = balanced_random_eval_labels(len(B["scenes"]), EVAL_POINTS)
            pred = (logits >= 0.0).float()
            out["relation_accuracy"] = float((pred == target).float().mean())
            out["relation_target_positive_fraction"] = float(target.mean())
    return out


def run(selection_path: Path) -> dict:
    torch.set_num_threads(1)
    selection = load_selection(selection_path)
    carrier_type = selection["selected_carrier"]
    B = build_bank()
    arms = (
        "GM_SINGLE_ORACLE", "GP_SINGLE_ORACLE", "S2_JOINT_BOUND",
        "FULL_SHARED_JOINT_BOUND", "S2_JOINT_UNBOUND", "S2_JOINT_BOUND_RANDOM_RELATION",
    )
    models = {}; summaries = {}; parameter_counts = {}
    for arm in arms:
        print(json.dumps({"starting_arm": arm}), flush=True)
        model = train_arm(arm, carrier_type, B)
        models[arm] = model
        summaries[arm] = summarize(model, B)
        parameter_counts[arm] = nparams(model)

    s2 = summaries["S2_JOINT_BOUND"]
    gm_single = summaries["GM_SINGLE_ORACLE"]["gm_overall_mean_nrmse"]
    gp_single = summaries["GP_SINGLE_ORACLE"]["gp_overall_mean_nrmse"]
    gm_transfer = s2["gm_overall_mean_nrmse"] / max(gm_single, 1.0e-12) - 1.0
    gp_transfer = s2["gp_overall_mean_nrmse"] / max(gp_single, 1.0e-12) - 1.0
    unbound = summaries["S2_JOINT_UNBOUND"]
    full = summaries["FULL_SHARED_JOINT_BOUND"]
    random_control = summaries["S2_JOINT_BOUND_RANDOM_RELATION"]

    rolled_carrier = torch.roll(models["S2_JOINT_BOUND"].base.carrier.detach(), shifts=1, dims=0)
    rolled = summarize(models["S2_JOINT_BOUND"], B, carrier=rolled_carrier)
    rolling_increase = rolled["mean_surface_local_nrmse"] / max(s2["mean_surface_local_nrmse"], 1.0e-12) - 1.0

    full_shared_catastrophic_contradiction = (
        full["gm_overall_mean_nrmse"] <= 0.90 * s2["gm_overall_mean_nrmse"]
        and full["gp_overall_mean_nrmse"] <= 0.90 * s2["gp_overall_mean_nrmse"]
    )
    gates = {
        "s2_gm_nrmse_le_0p15": s2["gm_overall_mean_nrmse"] <= 0.15,
        "s2_gp_nrmse_le_0p15": s2["gp_overall_mean_nrmse"] <= 0.15,
        "s2_relation_accuracy_ge_0p95": s2["relation_accuracy"] >= 0.95,
        "relation_positive_fraction_in_range": 0.35 <= s2["relation_target_positive_fraction"] <= 0.65,
        "gm_negative_transfer_le_15pct": gm_transfer <= 0.15,
        "gp_negative_transfer_le_15pct": gp_transfer <= 0.15,
        "unbound_relation_accuracy_le_0p60": unbound["relation_accuracy"] <= 0.60,
        "unbound_gm_local_within_15pct": unbound["gm_surface_local_mean_nrmse"] <= 1.15 * s2["gm_surface_local_mean_nrmse"],
        "unbound_gp_local_within_15pct": unbound["gp_surface_local_mean_nrmse"] <= 1.15 * s2["gp_surface_local_mean_nrmse"],
        "random_relation_firewall_in_0p48_0p52": 0.48 <= random_control["relation_accuracy"] <= 0.52,
        "rolling_carrier_local_error_increase_ge_25pct": rolling_increase >= 0.25,
        "full_shared_no_catastrophic_10pct_both_advantage": not full_shared_catastrophic_contradiction,
    }
    closed = all(gates.values())
    status = "CONTINUOUS_JOINT_SYSTEM_CLOSED_TINY_V2" if closed else "CONTINUOUS_JOINT_SYSTEM_OPEN_TINY_V2"
    return {
        "schema": SCHEMA,
        "status": status,
        "dependency": selection,
        "configuration": {
            "seeds": list(SEEDS), "factors": [4.0, 2.0, 1.0],
            "families": ["SURFACE_STRUCTURED", "THIN_SURFACE"],
            "scene_count": len(B["scenes"]), "carrier_type": carrier_type,
            "query": {"width": WIDTH, "depth": DEPTH, "fourier_bands": FOURIER_BANDS},
            "adapter_bottleneck": ADAPTER_BOTTLENECK, "relation_hidden": RELATION_HIDDEN,
            "lr": LR, "steps_per_arm": STEPS, "batch": BATCH,
            "train_pool": TRAIN_POOL, "eval_points": EVAL_POINTS,
            "arm_seeds": ARM_SEEDS,
            "loss_policy": "equal_unweighted_sum_of_normalized_local_task_losses_and_relation_bce",
            "relation_gradient_ownership": "STOPGRAD_RELATION",
        },
        "parameter_counts": parameter_counts,
        "summaries": summaries,
        "negative_transfer": {"gm": gm_transfer, "gp": gp_transfer},
        "rolling_carrier": {
            "ordinary_mean_surface_local_nrmse": s2["mean_surface_local_nrmse"],
            "rolled_mean_surface_local_nrmse": rolled["mean_surface_local_nrmse"],
            "relative_error_increase": rolling_increase,
        },
        "full_shared_catastrophic_contradiction": full_shared_catastrophic_contradiction,
        "gates": gates,
        "authority": {
            "continuous_joint_dependency_closed": closed,
            "v3_1_architecture_freeze_authorized": False,
            "product_authority_changed": False,
            "truth_builder_training_authorized": False,
            "fit8_training_authorized": False,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = run(args.selection)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "carrier_type": result["configuration"]["carrier_type"], "gates": result["gates"], "negative_transfer": result["negative_transfer"], "rolling_carrier": result["rolling_carrier"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
