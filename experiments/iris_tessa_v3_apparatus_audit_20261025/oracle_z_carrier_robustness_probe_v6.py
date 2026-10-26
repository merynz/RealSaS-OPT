#!/usr/bin/env python3
"""Execute ORACLE_Z_CARRIER_ROBUSTNESS_PREREG_V6 exactly."""
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import torch

SCHEMA = "RealSaS.OracleZCarrierRobustnessCourt.v6"
OPT_SEEDS = (8301, 9109, 10243)
STEPS = 8000

V5 = runpy.run_path(str(Path(__file__).with_name("oracle_z_matched_carrier_probe_v5.py")))
SEEDS = tuple(V5["SEEDS"])
FACTORS = tuple(V5["FACTORS"])
H_REF = float(V5["H_REF"])
bank = V5["bank"]
scenes_for = V5["scenes_for"]
seed_all = V5["seed_all"]
CarrierSharedQ = V5["CarrierSharedQ"]
train_base = V5["train"]
summarize = V5["summarize"]
support = V5["support"]


def valid_scenes():
    out = []
    for scene in scenes_for(SEEDS):
        factor = round(float(scene["radius"]) / H_REF, 6)
        if scene["family"] == "SURFACE_STRUCTURED":
            out.append(scene)
        elif scene["family"] == "THIN_SURFACE" and factor in (1.0, 4.0):
            out.append(scene)
    return out


def run() -> dict:
    torch.set_num_threads(1)
    B = bank(valid_scenes())
    arms = ("TRIPLANE_SHARED_Q", "DENSE3D_SHARED_Q")
    replications = []
    group_accumulator = {a: {} for a in arms}

    for opt_seed in OPT_SEEDS:
        seed_row = {"optimizer_seed": opt_seed, "arms": {}}
        for arm in arms:
            # Same optimizer/data seed for paired arms; only representation differs.
            seed_all(opt_seed)
            model = CarrierSharedQ(arm, len(B["scenes"]))
            model = train_base(model, B, steps=STEPS, seed=opt_seed)
            S = summarize(model, B)
            pass_count = sum(1 for row in S["groups"].values() if support(row))
            seed_row["arms"][arm] = {
                "overall_mean_nrmse": S["overall_mean_nrmse"],
                "support_group_count": pass_count,
                "groups": S["groups"],
            }
            for key, row in S["groups"].items():
                group_accumulator[arm].setdefault(key, []).append(float(row["mean_nrmse"]))
        t = seed_row["arms"]["TRIPLANE_SHARED_Q"]["overall_mean_nrmse"]
        d = seed_row["arms"]["DENSE3D_SHARED_Q"]["overall_mean_nrmse"]
        seed_row["dense_relative_gain"] = (t - d) / max(t, 1.0e-12)
        replications.append(seed_row)

    gains = [float(r["dense_relative_gain"]) for r in replications]
    dense_lower_all = all(g > 0.0 for g in gains)
    mean_gain = sum(gains) / len(gains)
    dense_support_not_worse = all(
        r["arms"]["DENSE3D_SHARED_Q"]["support_group_count"]
        >= r["arms"]["TRIPLANE_SHARED_Q"]["support_group_count"]
        for r in replications
    )

    group_average = {}
    dense_no_group_regression = True
    for key in sorted(group_accumulator["TRIPLANE_SHARED_Q"]):
        t = sum(group_accumulator["TRIPLANE_SHARED_Q"][key]) / len(OPT_SEEDS)
        d = sum(group_accumulator["DENSE3D_SHARED_Q"][key]) / len(OPT_SEEDS)
        regression = d / max(t, 1.0e-12) - 1.0
        group_average[key] = {
            "triplane_mean_nrmse": t,
            "dense_mean_nrmse": d,
            "dense_relative_change": regression,
        }
        if regression > 0.10:
            dense_no_group_regression = False

    dense_selected = dense_lower_all and mean_gain >= 0.05 and dense_support_not_worse and dense_no_group_regression
    selected = "DENSE3D" if dense_selected else "TRIPLANE"
    status = "DENSE3D_SELECTED_PRETRUTH_V6" if dense_selected else "TRIPLANE_SELECTED_PRETRUTH_V6"

    return {
        "schema": SCHEMA,
        "status": status,
        "selected_carrier": selected,
        "optimizer_seeds": list(OPT_SEEDS),
        "steps": STEPS,
        "replications": replications,
        "paired_dense_relative_gains": gains,
        "mean_dense_relative_gain": mean_gain,
        "group_average": group_average,
        "gates": {
            "dense_lower_all_optimizer_seeds": dense_lower_all,
            "mean_dense_relative_gain_ge_5pct": mean_gain >= 0.05,
            "dense_support_count_not_worse_each_seed": dense_support_not_worse,
            "dense_no_group_worse_gt_10pct": dense_no_group_regression,
        },
        "authority": {
            "product_authority_changed": False,
            "truth_builder_training_authorized": False,
            "fit8_training_authorized": False,
        },
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    r = run(); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": r["status"], "selected_carrier": r["selected_carrier"], "mean_dense_relative_gain": r["mean_dense_relative_gain"], "gates": r["gates"], "out": str(a.out)}, sort_keys=True))


if __name__ == "__main__": main()
