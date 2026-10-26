#!/usr/bin/env python3
"""Execute GRADIENT_OWNERSHIP_PREREG_V1 exactly.

This probe reuses the sealed continuous-joint V1 scene bank, truth functions,
carrier/query, GM/GP adapters, optimizer constants and evaluation metrics.  The
only scientific factors are relation-gradient ownership and relation-private
low-rank capacity.
"""
from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BASE = runpy.run_path(str(HERE / "continuous_joint_system_probe_v1.py"))

SCHEMA = "RealSaS.GradientOwnershipCourt.v1"
RANK = 16
LORA_ALPHA = 16.0
SCIENTIFIC_ARMS = (
    "CURRENT_FULL_GRAD",
    "STOPGRAD_RELATION",
    "PRIVATE_REL_ADAPTER_FULL_GRAD",
    "PRIVATE_REL_ADAPTER_STOPGRAD",
)
ARM_SEED = 730307
FIREWALL_SEED = 730601

# Exact frozen apparatus imported from the sealed joint V1 implementation.
LR = float(BASE["LR"])
STEPS = int(BASE["STEPS"])
BATCH = int(BASE["BATCH"])
WIDTH = int(BASE["WIDTH"])
TRAIN_POOL = int(BASE["TRAIN_POOL"])
EVAL_POINTS = int(BASE["EVAL_POINTS"])
seed_all = BASE["seed_all"]
gen = BASE["gen"]
build_bank = BASE["build_bank"]
training_pair = BASE["training_pair"]
load_selection = BASE["load_selection"]
BaseCourtModel = BASE["CourtModel"]
base_train_arm = BASE["train_arm"]
base_summarize = BASE["summarize"]
nparams = BASE["nparams"]


class RelationLowRankAdapter(nn.Module):
    """Identity-initialized rank-16 linear residual on endpoint GP state."""

    def __init__(self, width: int = WIDTH, rank: int = RANK, alpha: float = LORA_ALPHA):
        super().__init__()
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.A = nn.Linear(width, rank, bias=False)
        self.B = nn.Linear(rank, width, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.B(self.A(x)) * self.scale


class OwnershipModel(nn.Module):
    """Exact S2 joint model with only relation-path ownership/capacity varied."""

    def __init__(self, policy: str, carrier_type: str, scene_count: int, *, random_firewall: bool = False):
        super().__init__()
        if policy not in SCIENTIFIC_ARMS:
            raise ValueError(policy)
        self.policy = policy
        self.random_firewall = bool(random_firewall)
        # Important: construct the unchanged S2 core first. With the same seed,
        # all scientific arms therefore begin with identical shared/core weights.
        self.core = BaseCourtModel("S2_JOINT_BOUND", carrier_type, scene_count)
        self.use_stopgrad = policy in {"STOPGRAD_RELATION", "PRIVATE_REL_ADAPTER_STOPGRAD"}
        self.use_private_relation_adapter = policy in {
            "PRIVATE_REL_ADAPTER_FULL_GRAD", "PRIVATE_REL_ADAPTER_STOPGRAD"
        }
        if self.use_private_relation_adapter:
            # Added only after the unchanged core so its existence cannot perturb
            # shared/core initialization through RNG ordering.
            self.rel_adapter = RelationLowRankAdapter()

        # Compatibility with sealed summarize().
        self.arm = "S2_JOINT_BOUND_RANDOM_RELATION" if self.random_firewall else "S2_JOINT_BOUND"
        self.has_relation = True
        self.base = self.core.base

    def _relation_state(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_stopgrad:
            x = x.detach()
        if self.use_private_relation_adapter:
            x = self.rel_adapter(x)
        return x

    def local(self, p: torch.Tensor, carrier: torch.Tensor | None = None) -> dict:
        return self.core.local(p, carrier)

    def pair(self, a: torch.Tensor, b: torch.Tensor, carrier: torch.Tensor | None = None) -> dict:
        ha = self.core.base.encode(a, carrier)
        gm_a, gp_a = self.core.private_states(ha)
        if gm_a is None or gp_a is None:
            raise RuntimeError("OWNERSHIP_COURT_REQUIRES_GM_GP")
        hb = self.core.base.encode(b, carrier)
        _, gp_b = self.core.private_states(hb)
        if gp_b is None:
            raise RuntimeError("OWNERSHIP_COURT_REQUIRES_BOUND_GP_ENDPOINTS")
        out = {
            "gm": self.core.gm_head(gm_a),
            "gp": self.core.gp_head(gp_a),
        }
        ra = self._relation_state(gp_a)
        rb = self._relation_state(gp_b)
        out["relation_logits"] = self.core.relation(torch.cat((ra, rb), dim=-1))
        return out


def train_ownership(policy: str, carrier_type: str, B: dict, *, random_firewall: bool = False) -> OwnershipModel:
    seed = FIREWALL_SEED if random_firewall else ARM_SEED
    seed_all(seed)
    model = OwnershipModel(policy, carrier_type, len(B["scenes"]), random_firewall=random_firewall)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    g = gen(seed * 7919 + 17)
    gm_scale = B["gm_scale"][:, None]
    gp_scale = B["gp_scale"][:, None]
    model.train()
    for _ in range(STEPS):
        batch = training_pair(B, g)
        out = model.pair(batch["a"], batch["b"])
        gm_loss = ((((out["gm"] - batch["gm_a"]) / gm_scale) ** 2).mean())
        gp_loss = ((((out["gp"] - batch["gp_a"]) / gp_scale) ** 2).mean())
        label = batch["random_relation"] if random_firewall else batch["relation"]
        relation_loss = F.binary_cross_entropy_with_logits(out["relation_logits"], label)
        loss = gm_loss + gp_loss + relation_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def scientific_summary(model: OwnershipModel, B: dict, gm_single: float, gp_single: float) -> dict:
    s = base_summarize(model, B)
    gm_transfer = s["gm_overall_mean_nrmse"] / max(gm_single, 1.0e-12) - 1.0
    gp_transfer = s["gp_overall_mean_nrmse"] / max(gp_single, 1.0e-12) - 1.0
    gates = {
        "gm_nrmse_le_0p15": s["gm_overall_mean_nrmse"] <= 0.15,
        "gp_nrmse_le_0p15": s["gp_overall_mean_nrmse"] <= 0.15,
        "relation_accuracy_ge_0p95": s["relation_accuracy"] >= 0.95,
        "relation_positive_fraction_in_range": 0.35 <= s["relation_target_positive_fraction"] <= 0.65,
        "gm_negative_transfer_le_15pct": gm_transfer <= 0.15,
        "gp_negative_transfer_le_15pct": gp_transfer <= 0.15,
    }
    return {
        **s,
        "negative_transfer": {"gm": gm_transfer, "gp": gp_transfer},
        "joint_gates": gates,
        "joint_valid": all(gates.values()),
    }


def choose_arm(rows: dict[str, dict], params: dict[str, int]) -> str | None:
    current = rows["CURRENT_FULL_GRAD"]
    valid = [a for a in SCIENTIFIC_ARMS if rows[a]["joint_valid"]]
    if not valid:
        return None

    if current["joint_valid"]:
        challengers = []
        for arm in valid:
            if arm == "CURRENT_FULL_GRAD":
                continue
            gm_better = rows[arm]["gm_overall_mean_nrmse"] <= 0.90 * current["gm_overall_mean_nrmse"]
            gp_better = rows[arm]["gp_overall_mean_nrmse"] <= 0.90 * current["gp_overall_mean_nrmse"]
            rel_ok = rows[arm]["relation_accuracy"] >= current["relation_accuracy"] - 0.01
            if gm_better and gp_better and rel_ok:
                challengers.append(arm)
        if not challengers:
            return "CURRENT_FULL_GRAD"
        valid = challengers

    scored = []
    for arm in valid:
        tr = rows[arm]["negative_transfer"]
        worst = max(float(tr["gm"]), float(tr["gp"]))
        scored.append((arm, worst, params[arm]))
    best_worst = min(x[1] for x in scored)
    near = [x for x in scored if x[1] <= best_worst + 0.01]
    min_params = min(x[2] for x in near)
    near = [x for x in near if x[2] == min_params]
    if len(near) == 1:
        return near[0][0]
    # Final prereg tie break: less restrictive full-gradient policy.
    full_grad = [x[0] for x in near if "STOPGRAD" not in x[0]]
    return sorted(full_grad if full_grad else [x[0] for x in near])[0]


def classify(rows: dict[str, dict], selected: str | None) -> str:
    if selected is None:
        return "DEEPER_REPRESENTATION_INTERFERENCE"
    if selected == "CURRENT_FULL_GRAD":
        return "CURRENT_COUPLING_NOT_CAUSAL"
    if selected == "STOPGRAD_RELATION" and not rows["CURRENT_FULL_GRAD"]["joint_valid"]:
        return "OPTIMIZATION_COUPLING"
    if selected == "PRIVATE_REL_ADAPTER_FULL_GRAD":
        return "PRIVATE_RELATION_CAPACITY"
    if selected == "PRIVATE_REL_ADAPTER_STOPGRAD":
        no_adapter_stop = rows["STOPGRAD_RELATION"]["joint_valid"]
        adapter_full = rows["PRIVATE_REL_ADAPTER_FULL_GRAD"]["joint_valid"]
        if not no_adapter_stop and not adapter_full:
            return "GRADIENT_OWNERSHIP_AND_PRIVATE_CAPACITY"
        if no_adapter_stop:
            return "OPTIMIZATION_COUPLING"
        return "PRIVATE_RELATION_CAPACITY"
    return "CURRENT_COUPLING_NOT_CAUSAL"


def run(selection_path: Path) -> dict:
    torch.set_num_threads(1)
    selection = load_selection(selection_path)
    if selection["selected_carrier"] != "DENSE3D":
        raise RuntimeError("GRADIENT_OWNERSHIP_V1_REQUIRES_V6_DENSE3D")
    B = build_bank()
    carrier_type = selection["selected_carrier"]

    # Frozen single-family references from joint V1.
    gm_oracle = base_train_arm("GM_SINGLE_ORACLE", carrier_type, B)
    gp_oracle = base_train_arm("GP_SINGLE_ORACLE", carrier_type, B)
    gm_oracle_summary = base_summarize(gm_oracle, B)
    gp_oracle_summary = base_summarize(gp_oracle, B)
    gm_single = gm_oracle_summary["gm_overall_mean_nrmse"]
    gp_single = gp_oracle_summary["gp_overall_mean_nrmse"]

    models: dict[str, OwnershipModel] = {}
    rows: dict[str, dict] = {}
    params: dict[str, int] = {
        "GM_SINGLE_ORACLE": nparams(gm_oracle),
        "GP_SINGLE_ORACLE": nparams(gp_oracle),
    }
    for arm in SCIENTIFIC_ARMS:
        model = train_ownership(arm, carrier_type, B)
        models[arm] = model
        params[arm] = nparams(model)
        rows[arm] = scientific_summary(model, B, gm_single, gp_single)

    current_params = params["CURRENT_FULL_GRAD"]
    adapter_overhead = {}
    for arm in SCIENTIFIC_ARMS:
        overhead = (params[arm] - current_params) / max(current_params, 1)
        adapter_overhead[arm] = overhead
        if "PRIVATE_REL_ADAPTER" in arm:
            rows[arm]["adapter_parameter_overhead_fraction"] = overhead
            rows[arm]["adapter_overhead_le_5pct"] = overhead <= 0.05
            rows[arm]["joint_valid"] = bool(rows[arm]["joint_valid"] and overhead <= 0.05)
        else:
            rows[arm]["adapter_parameter_overhead_fraction"] = 0.0
            rows[arm]["adapter_overhead_le_5pct"] = True

    selected = choose_arm(rows, params)
    interpretation = classify(rows, selected)

    firewall = None
    controls = None
    final_closed = False
    if selected is not None:
        firewall_model = train_ownership(selected, carrier_type, B, random_firewall=True)
        firewall = base_summarize(firewall_model, B)
        rolled_carrier = torch.roll(models[selected].base.carrier.detach(), shifts=1, dims=0)
        rolled = base_summarize(models[selected], B, carrier=rolled_carrier)
        ordinary_local = rows[selected]["mean_surface_local_nrmse"]
        rolling_increase = rolled["mean_surface_local_nrmse"] / max(ordinary_local, 1.0e-12) - 1.0
        controls = {
            "random_relation_firewall_accuracy": firewall["relation_accuracy"],
            "random_relation_firewall_in_0p48_0p52": 0.48 <= firewall["relation_accuracy"] <= 0.52,
            "ordinary_mean_surface_local_nrmse": ordinary_local,
            "rolled_mean_surface_local_nrmse": rolled["mean_surface_local_nrmse"],
            "rolling_carrier_relative_error_increase": rolling_increase,
            "rolling_carrier_error_increase_ge_25pct": rolling_increase >= 0.25,
            "selected_adapter_overhead_le_5pct": rows[selected]["adapter_overhead_le_5pct"],
        }
        final_closed = bool(rows[selected]["joint_valid"] and all([
            controls["random_relation_firewall_in_0p48_0p52"],
            controls["rolling_carrier_error_increase_ge_25pct"],
            controls["selected_adapter_overhead_le_5pct"],
        ]))

    status = "GRADIENT_OWNERSHIP_CLOSED_TINY_V1" if final_closed else "GRADIENT_OWNERSHIP_OPEN_TINY_V1"
    return {
        "schema": SCHEMA,
        "status": status,
        "dependency": selection,
        "configuration": {
            "carrier_type": carrier_type,
            "scientific_arms": list(SCIENTIFIC_ARMS),
            "shared_scientific_arm_seed": ARM_SEED,
            "firewall_seed": FIREWALL_SEED,
            "rank": RANK,
            "lora_alpha": LORA_ALPHA,
            "lr": LR,
            "steps_per_arm": STEPS,
            "batch": BATCH,
            "train_pool": TRAIN_POOL,
            "eval_points": EVAL_POINTS,
            "loss_policy": "unchanged_equal_unweighted_normalized_gm_gp_plus_relation_bce",
        },
        "single_family_references": {
            "GM_SINGLE_ORACLE": gm_oracle_summary,
            "GP_SINGLE_ORACLE": gp_oracle_summary,
        },
        "parameter_counts": params,
        "adapter_overhead_fraction": adapter_overhead,
        "scientific_results": rows,
        "provisional_selected_arm": selected,
        "interpretation": interpretation,
        "selected_arm_controls": controls,
        "random_firewall_summary": firewall,
        "authority": {
            "gradient_ownership_treatment_selected": final_closed,
            "selected_treatment": selected if final_closed else None,
            "continuous_joint_v2_authorized": final_closed,
            "v3_1_architecture_freeze_authorized": False,
            "truth_builder_training_authorized": False,
            "fit8_training_authorized": False,
            "product_authority_changed": False,
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
    compact = {
        "status": result["status"],
        "selected": result["provisional_selected_arm"],
        "interpretation": result["interpretation"],
        "controls": result["selected_arm_controls"],
        "authority": result["authority"],
        "out": str(args.out),
    }
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
