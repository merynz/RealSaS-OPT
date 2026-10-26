#!/usr/bin/env python3
"""Execute OBSERVATION_ENDPOINT_INTERACTION_PREREG_V2: selected relation stop-gradient."""
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

SCHEMA = "RealSaS.ObservationEndpointInteractionCourt.v2"
SEEDS = (731, 947, 1217, 1597, 2017)
D = 64
HEADS = 4
LAYERS = 3
BATCH = 64
STEPS = 1200
PAIRS = 8
EVAL_BATCHES = 32
LR = 1.0e-3
VIEWS = 4
LOCI = 16


def lattice() -> torch.Tensor:
    axis = torch.linspace(-0.75, 0.75, 4)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(LOCI, 2)


def rotate(xy: torch.Tensor, k: int) -> torch.Tensor:
    x, y = xy[..., 0], xy[..., 1]
    if k == 0: return xy
    if k == 1: return torch.stack((-y, x), dim=-1)
    if k == 2: return torch.stack((-x, -y), dim=-1)
    return torch.stack((y, -x), dim=-1)


def make_batch(n: int, *, generator: torch.Generator, alpha_mode: str, random_relation: bool):
    xy = lattice()
    gm_bits = torch.randint(0, 2, (n, LOCI, VIEWS), generator=generator).float()
    alpha_bits = torch.randint(0, 2, (n, LOCI, VIEWS), generator=generator).float()
    final_bit = torch.randint(0, 2, (n, 1), generator=generator).float()

    token_values = []
    token_xy = []
    token_camera = []
    for v in range(VIEWS):
        p = rotate(xy, v)[None].expand(n, -1, -1)
        theta = v * math.pi / 2.0
        camera = torch.tensor([math.cos(theta), math.sin(theta)]).float()[None, None].expand(n, LOCI, -1)
        gm = gm_bits[:, :, v:v+1] * 2.0 - 1.0
        if alpha_mode == "raw":
            alpha = alpha_bits[:, :, v:v+1] * 2.0 - 1.0
        elif alpha_mode == "mean":
            alpha = (alpha_bits[:, :, v].mean(dim=1, keepdim=True) * 2.0 - 1.0)[:, None].expand(-1, LOCI, -1)
        elif alpha_mode == "none":
            alpha = torch.zeros(n, LOCI, 1)
        else:
            raise ValueError(alpha_mode)
        fin = (final_bit * 2.0 - 1.0)[:, None].expand(-1, LOCI, -1)
        token_values.append(torch.cat((gm, alpha, fin), dim=-1))
        token_xy.append(p)
        token_camera.append(camera)

    value = torch.cat(token_values, dim=1)
    uv = torch.cat(token_xy, dim=1)
    camera = torch.cat(token_camera, dim=1)
    perm = torch.rand((n, VIEWS * LOCI), generator=generator).argsort(dim=1)
    value = torch.gather(value, 1, perm[..., None].expand(-1, -1, 3))
    uv = torch.gather(uv, 1, perm[..., None].expand(-1, -1, 2))
    camera = torch.gather(camera, 1, perm[..., None].expand(-1, -1, 2))

    a = torch.randint(0, LOCI, (n, PAIRS), generator=generator)
    b = torch.randint(0, LOCI - 1, (n, PAIRS), generator=generator)
    b = b + (b >= a).long()
    batch_index = torch.arange(n)[:, None]
    relation = torch.logical_xor(
        alpha_bits[batch_index, a, 0] >= 0.5,
        alpha_bits[batch_index, b, 0] >= 0.5,
    ).float()
    if random_relation:
        relation = torch.randint(0, 2, relation.shape, generator=generator).float()

    return {
        "value": value,
        "uv": uv,
        "camera": camera,
        "query": xy[None].expand(n, -1, -1),
        "gm_target": gm_bits,
        "alpha_target": alpha_bits,
        "final_target": final_bit[:, None].expand(-1, LOCI, -1),
        "endpoint_a": a,
        "endpoint_b": b,
        "relation_target": relation,
    }


class CourtModel(nn.Module):
    def __init__(self, *, private_subspaces: bool, bound_relation: bool):
        super().__init__()
        self.private_subspaces = bool(private_subspaces)
        self.bound_relation = bool(bound_relation)
        self.value = nn.Sequential(nn.Linear(3, D), nn.GELU(), nn.Linear(D, D))
        self.camera = nn.Sequential(nn.Linear(2, D), nn.GELU(), nn.Linear(D, D))
        self.xy = nn.Sequential(nn.Linear(2, D), nn.GELU(), nn.Linear(D, D))
        self.source_norm = nn.LayerNorm(D)
        self.query = nn.Sequential(nn.Linear(2, D), nn.GELU(), nn.Linear(D, D))
        layer = nn.TransformerDecoderLayer(
            d_model=D, nhead=HEADS, dim_feedforward=4 * D, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, LAYERS, norm=nn.LayerNorm(D))
        if private_subspaces:
            self.gm_private = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
            self.gp_private = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        else:
            self.gm_private = nn.Identity()
            self.gp_private = nn.Identity()
        self.gm_head = nn.Linear(D, VIEWS)
        self.alpha_head = nn.Linear(D, VIEWS)
        self.final_head = nn.Linear(D, 1)
        self.relation = nn.Sequential(
            nn.Linear(4 * D + 4, 2 * D), nn.GELU(), nn.Linear(2 * D, 1)
        )

    def forward(self, batch: dict) -> dict:
        memory = self.source_norm(
            self.value(batch["value"]) + self.camera(batch["camera"]) + self.xy(batch["uv"])
        )
        q = self.query(batch["query"])
        shared = self.decoder(q, memory)
        gm = self.gm_private(shared)
        gp = self.gp_private(shared)

        n = shared.shape[0]
        ai = batch["endpoint_a"]
        bi = batch["endpoint_b"]
        gather_a = ai[..., None].expand(-1, -1, D)
        gather_b = bi[..., None].expand(-1, -1, D)
        if self.bound_relation:
            fa = torch.gather(gp, 1, gather_a)
            fb = torch.gather(gp, 1, gather_b)
        else:
            compressed = gp.mean(dim=1, keepdim=True).expand(-1, PAIRS, -1)
            fa = compressed
            fb = compressed
        xy = batch["query"]
        xya = torch.gather(xy, 1, ai[..., None].expand(-1, -1, 2))
        xyb = torch.gather(xy, 1, bi[..., None].expand(-1, -1, 2))
        fa, fb = fa.detach(), fb.detach()
        rel_features = torch.cat((fa, fb, fa - fb, fa * fb, xya, xyb), dim=-1)
        return {
            "gm": self.gm_head(gm),
            "alpha": self.alpha_head(gp),
            "final": self.final_head(shared),
            "relation": self.relation(rel_features).squeeze(-1),
        }


def nparams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


ARMS = {
    "S2_RAW_ALPHA_BOUND": {"private": True, "bound": True, "alpha_mode": "raw", "random_relation": False},
    "FULL_SHARED_RAW_ALPHA_BOUND": {"private": False, "bound": True, "alpha_mode": "raw", "random_relation": False},
    "S2_MEAN_ALPHA_BOUND": {"private": True, "bound": True, "alpha_mode": "mean", "random_relation": False},
    "S2_RAW_ALPHA_UNBOUND": {"private": True, "bound": False, "alpha_mode": "raw", "random_relation": False},
    "S2_RAW_ALPHA_BOUND_RANDOM_RELATION": {"private": True, "bound": True, "alpha_mode": "raw", "random_relation": True},
}


def train_eval(seed: int, arm_name: str) -> dict:
    spec = ARMS[arm_name]
    torch.manual_seed(seed)
    model = CourtModel(private_subspaces=spec["private"], bound_relation=spec["bound"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    train_gen = torch.Generator().manual_seed(seed + 100000)
    for _ in range(STEPS):
        batch = make_batch(BATCH, generator=train_gen, alpha_mode=spec["alpha_mode"], random_relation=spec["random_relation"])
        pred = model(batch)
        loss = (
            F.binary_cross_entropy_with_logits(pred["gm"], batch["gm_target"])
            + F.binary_cross_entropy_with_logits(pred["alpha"], batch["alpha_target"])
            + F.binary_cross_entropy_with_logits(pred["final"], batch["final_target"])
            + F.binary_cross_entropy_with_logits(pred["relation"], batch["relation_target"])
        )
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

    model.eval(); eval_gen = torch.Generator().manual_seed(seed + 200000)
    counts = {"gm": [0, 0], "alpha": [0, 0], "final": [0, 0], "relation": [0, 0]}
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            batch = make_batch(BATCH, generator=eval_gen, alpha_mode=spec["alpha_mode"], random_relation=spec["random_relation"])
            pred = model(batch)
            for key, target_key in (("gm", "gm_target"), ("alpha", "alpha_target"), ("final", "final_target"), ("relation", "relation_target")):
                truth = batch[target_key] >= 0.5
                guess = pred[key] >= 0
                counts[key][0] += int((truth == guess).sum())
                counts[key][1] += int(truth.numel())
    return {
        "seed": seed,
        "arm": arm_name,
        "parameter_count": nparams(model),
        **{f"{k}_accuracy": v[0] / v[1] for k, v in counts.items()},
    }


def summarize(rows: list[dict], arm: str) -> dict:
    q = [r for r in rows if r["arm"] == arm]
    out = {"parameter_count": q[0]["parameter_count"]}
    for key in ("gm_accuracy", "alpha_accuracy", "final_accuracy", "relation_accuracy"):
        vals = [float(r[key]) for r in q]
        out[f"mean_{key}"] = sum(vals) / len(vals)
        out[f"min_{key}"] = min(vals)
        out[f"std_{key}"] = statistics.pstdev(vals)
        out[f"per_seed_{key}"] = vals
    return out


def assemble(rows: list[dict]) -> dict:
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    actual = [(r["seed"], r["arm"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError("INCOMPLETE_OR_DUPLICATE_ENDPOINT_COURT")
    rows = sorted(rows, key=lambda r: (SEEDS.index(r["seed"]), list(ARMS).index(r["arm"])))
    S = {arm: summarize(rows, arm) for arm in ARMS}
    sel = S["S2_RAW_ALPHA_BOUND"]
    shared = S["FULL_SHARED_RAW_ALPHA_BOUND"]
    mean_alpha = S["S2_MEAN_ALPHA_BOUND"]
    unbound = S["S2_RAW_ALPHA_UNBOUND"]
    firewall = S["S2_RAW_ALPHA_BOUND_RANDOM_RELATION"]

    selected_ok = (
        sel["mean_gm_accuracy"] >= 0.98 and sel["min_gm_accuracy"] >= 0.95
        and sel["mean_alpha_accuracy"] >= 0.98 and sel["min_alpha_accuracy"] >= 0.95
        and sel["mean_final_accuracy"] >= 0.98 and sel["min_final_accuracy"] >= 0.95
        and sel["mean_relation_accuracy"] >= 0.98 and sel["min_relation_accuracy"] >= 0.95
    )
    alpha_owner = mean_alpha["mean_alpha_accuracy"] <= 0.60 and mean_alpha["mean_gm_accuracy"] >= 0.95 and mean_alpha["mean_final_accuracy"] >= 0.95
    endpoint_owner = unbound["mean_relation_accuracy"] <= 0.60 and unbound["mean_gm_accuracy"] >= 0.95 and unbound["mean_alpha_accuracy"] >= 0.95
    firewall_ok = 0.48 <= firewall["mean_relation_accuracy"] <= 0.52
    shared_not_universally_superior = not all(
        shared[f"mean_{key}_accuracy"] - sel[f"mean_{key}_accuracy"] >= 0.05
        for key in ("gm", "alpha", "final", "relation")
    )

    if not firewall_ok:
        status = "OBSERVATION_ENDPOINT_INTERACTION_INVALID__FIREWALL"
    elif selected_ok and alpha_owner and endpoint_owner and shared_not_universally_superior:
        status = "OBSERVATION_ENDPOINT_INTERACTION_COMPATIBLE_TINY_V2"
    else:
        status = "OBSERVATION_ENDPOINT_INTERACTION_NOT_CLOSED_TINY_V2"

    return {
        "schema": SCHEMA,
        "status": status,
        "seeds": list(SEEDS),
        "configuration": {"relation_gradient_ownership": "STOPGRAD_RELATION", "d_model": D, "layers": LAYERS, "steps": STEPS, "batch": BATCH, "pairs": PAIRS, "lr": LR},
        "summary": S,
        "results": rows,
        "gates": {
            "selected_assembly": selected_ok,
            "raw_alpha_information_owner": alpha_owner,
            "endpoint_binding_information_owner": endpoint_owner,
            "random_relation_firewall": firewall_ok,
            "full_shared_not_universally_superior_by_5pt": shared_not_universally_superior,
        },
        "authority": {"product_authority_changed": False, "truth_builder_training_authorized": False, "fit8_training_authorized": False},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--aggregate", type=Path)
    a = p.parse_args()
    if a.seed is not None and a.aggregate is not None:
        p.error("--seed and --aggregate are mutually exclusive")
    torch.set_num_threads(1)
    if a.aggregate is not None:
        files = sorted(a.aggregate.rglob("endpoint_seed_*.json"))
        rows = [row for f in files for row in json.loads(f.read_text())["results"]]
        r = assemble(rows)
    else:
        seeds = (a.seed,) if a.seed is not None else SEEDS
        rows = []
        for seed in seeds:
            for arm in ARMS:
                print(json.dumps({"seed": seed, "starting_arm": arm}), flush=True)
                rows.append(train_eval(seed, arm))
                print(json.dumps(rows[-1], sort_keys=True), flush=True)
        r = {"schema": SCHEMA, "shard_seed": a.seed, "results": rows} if a.seed is not None else assemble(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": r.get("status", "SHARD_COMPLETE"), "gates": r.get("gates"), "out": str(a.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
