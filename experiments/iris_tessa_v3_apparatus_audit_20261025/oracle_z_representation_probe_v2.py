#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

SCIENTIFIC_SEEDS = (731, 947, 1217, 1601, 1999)
CALIBRATION_SEEDS = (101, 211, 307)
CALIBRATION_LRS = (1e-2, 3e-3, 1e-3)
CALIBRATION_STEPS = 1200
COMPARISON_STEPS = 1600
TRAIN_POOL = 4096
EVAL_POINTS = 2048
BATCH_PER_SCENE = 32
H_REF = 2.0 / 24.0
SURFACE_FACTORS = (4.0, 2.0, 1.0, 0.5)
VOLUME_FREQS = (1, 2, 4, 6)
ARMS = ("TRIPLANE", "TRIPLANE_XYZ", "MULTIRES_TRIPLANE", "DENSE3D")
REGIMES = ("PER_SCENE_Q", "SHARED_Q")
OUTPUT_DIM = 4


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SceneDef:
    seed: int
    replicate_seed: int
    family: str
    difficulty: float
    centers: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    attrs: torch.Tensor | None = None
    fourier_w: torch.Tensor | None = None
    fourier_a: torch.Tensor | None = None
    fourier_b: torch.Tensor | None = None


@dataclass
class DatasetBank:
    scenes: list[SceneDef]
    train_points: torch.Tensor
    train_targets: torch.Tensor
    train_scale: torch.Tensor
    eval_points: torch.Tensor
    eval_targets: torch.Tensor
    eval_boundary: torch.Tensor


def _scene_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_surface_scene(seed: int, replicate_seed: int, family: str, radius: float) -> SceneDef:
    g = _scene_generator(seed)
    k = 4
    centers = torch.empty(k, 3).uniform_(-0.42, 0.42, generator=g)
    attrs = torch.empty(k, 3).uniform_(-0.85, 0.85, generator=g)
    if family == "SURFACE_STRUCTURED":
        scales = torch.empty(k, 3).uniform_(0.16, 0.34, generator=g)
        scales[0].fill_(float(radius))
    elif family == "THIN_SURFACE":
        scales = torch.empty(k, 3).uniform_(0.18, 0.36, generator=g)
        for i in range(k):
            scales[i, i % 3] = float(radius)
    else:
        raise ValueError(family)
    scales.clamp_(min=0.025, max=0.36)
    return SceneDef(
        seed=seed,
        replicate_seed=replicate_seed,
        family=family,
        difficulty=float(radius),
        centers=centers,
        scales=scales,
        attrs=attrs,
    )


def make_volumetric_scene(seed: int, replicate_seed: int, freq: int) -> SceneDef:
    g = _scene_generator(seed)
    terms = 10
    w = torch.randint(1, freq + 1, (OUTPUT_DIM, terms, 3), generator=g).float()
    signs = torch.randint(0, 2, (OUTPUT_DIM, terms, 3), generator=g).float() * 2.0 - 1.0
    w = w * signs
    a = torch.randn(OUTPUT_DIM, terms, generator=g) / math.sqrt(terms)
    b = torch.rand(OUTPUT_DIM, terms, generator=g) * (2.0 * math.pi)
    return SceneDef(
        seed=seed,
        replicate_seed=replicate_seed,
        family="VOLUMETRIC_3WAY",
        difficulty=float(freq),
        fourier_w=w,
        fourier_a=a,
        fourier_b=b,
    )


def make_comparison_scenes(seeds: Iterable[int]) -> list[SceneDef]:
    scenes: list[SceneDef] = []
    for seed in seeds:
        for idx, factor in enumerate(SURFACE_FACTORS):
            radius = factor * H_REF
            scenes.append(make_surface_scene(seed * 10000 + 100 + idx, seed, "SURFACE_STRUCTURED", radius))
            scenes.append(make_surface_scene(seed * 10000 + 200 + idx, seed, "THIN_SURFACE", radius))
        for idx, freq in enumerate(VOLUME_FREQS):
            scenes.append(make_volumetric_scene(seed * 10000 + 300 + idx, seed, freq))
    return scenes


def scene_targets(scene: SceneDef, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scene.family in ("SURFACE_STRUCTURED", "THIN_SURFACE"):
        assert scene.centers is not None and scene.scales is not None and scene.attrs is not None
        q = (points[:, None, :] - scene.centers[None]) / scene.scales[None]
        d = torch.linalg.vector_norm(q, dim=-1) - 1.0
        sdf, idx = d.min(dim=1)
        attr = scene.attrs[idx]
        y = torch.cat((sdf.clamp(-1.0, 1.0)[:, None], attr), dim=1)
        boundary = sdf.abs() <= 0.08
        return y, boundary

    assert scene.fourier_w is not None and scene.fourier_a is not None and scene.fourier_b is not None
    phase = math.pi * torch.einsum("nd,ktd->nkt", points, scene.fourier_w) + scene.fourier_b[None]
    y = torch.einsum("nkt,kt->nk", torch.sin(phase), scene.fourier_a)
    y = torch.tanh(y)
    return y, torch.ones(points.shape[0], dtype=torch.bool)


def sample_points(scene: SceneDef, n: int, generator: torch.Generator) -> torch.Tensor:
    if scene.family == "VOLUMETRIC_3WAY":
        return torch.empty(n, 3).uniform_(-0.95, 0.95, generator=generator)
    assert scene.centers is not None and scene.scales is not None
    nu = n // 2
    uniform = torch.empty(nu, 3).uniform_(-0.95, 0.95, generator=generator)
    ns = n - nu
    comp = torch.randint(0, scene.centers.shape[0], (ns,), generator=generator)
    direction = F.normalize(torch.randn(ns, 3, generator=generator), dim=-1)
    radial = 1.0 + 0.10 * torch.randn(ns, 1, generator=generator)
    near = scene.centers[comp] + direction * scene.scales[comp] * radial
    near = near.clamp(-0.98, 0.98)
    return torch.cat((uniform, near), dim=0)


def build_bank(scenes: list[SceneDef], train_pool: int, eval_points: int) -> DatasetBank:
    train_p, train_y, eval_p, eval_y, eval_b = [], [], [], [], []
    for i, scene in enumerate(scenes):
        g_train = _scene_generator(700000 + scene.seed * 13 + i)
        g_eval = _scene_generator(900000 + scene.seed * 17 + i)
        p = sample_points(scene, train_pool, g_train)
        y, _ = scene_targets(scene, p)
        ep = sample_points(scene, eval_points, g_eval)
        ey, eb = scene_targets(scene, ep)
        train_p.append(p)
        train_y.append(y)
        eval_p.append(ep)
        eval_y.append(ey)
        eval_b.append(eb)
    train_points_t = torch.stack(train_p)
    train_targets_t = torch.stack(train_y)
    scale = train_targets_t.std(dim=1).clamp_min(0.10)
    return DatasetBank(
        scenes=scenes,
        train_points=train_points_t,
        train_targets=train_targets_t,
        train_scale=scale,
        eval_points=torch.stack(eval_p),
        eval_targets=torch.stack(eval_y),
        eval_boundary=torch.stack(eval_b),
    )


def _batched_grid_sample_2d(planes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    x, y, z = points.unbind(-1)
    grids = (torch.stack((x, y), -1), torch.stack((x, z), -1), torch.stack((y, z), -1))
    out = []
    for plane_i, grid in enumerate(grids):
        sampled = F.grid_sample(
            planes[:, plane_i],
            grid.unsqueeze(2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        out.append(sampled.squeeze(-1).transpose(1, 2))
    return torch.cat(out, dim=-1)


def _batched_grid_sample_3d(grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    sampled = F.grid_sample(
        grid,
        points.unsqueeze(2).unsqueeze(2),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.squeeze(-1).squeeze(-1).transpose(1, 2)


class CarrierBank(nn.Module):
    def __init__(self, kind: str, scene_count: int):
        super().__init__()
        self.kind = kind
        if kind in ("TRIPLANE", "TRIPLANE_XYZ"):
            self.base = nn.Parameter(torch.randn(scene_count, 3, 16, 24, 24) * 0.02)
            self.feature_dim = 48 + (3 if kind == "TRIPLANE_XYZ" else 0)
        elif kind == "MULTIRES_TRIPLANE":
            self.low = nn.Parameter(torch.randn(scene_count, 3, 16, 12, 12) * 0.02)
            self.high = nn.Parameter(torch.randn(scene_count, 3, 12, 24, 24) * 0.02)
            self.feature_dim = 84
        elif kind == "DENSE3D":
            self.base = nn.Parameter(torch.randn(scene_count, 16, 12, 12, 12) * 0.02)
            self.feature_dim = 16
        else:
            raise ValueError(kind)

    def features(self, points: torch.Tensor) -> torch.Tensor:
        if self.kind in ("TRIPLANE", "TRIPLANE_XYZ"):
            f = _batched_grid_sample_2d(self.base, points)
            if self.kind == "TRIPLANE_XYZ":
                f = torch.cat((f, points), dim=-1)
            return f
        if self.kind == "MULTIRES_TRIPLANE":
            return torch.cat(
                (_batched_grid_sample_2d(self.low, points), _batched_grid_sample_2d(self.high, points)),
                dim=-1,
            )
        return _batched_grid_sample_3d(self.base, points)

    def scalars_per_scene(self) -> int:
        if self.kind == "MULTIRES_TRIPLANE":
            return self.low[0].numel() + self.high[0].numel()
        return self.base[0].numel()


QUERY_HIDDEN = {
    "TRIPLANE": 96,
    "TRIPLANE_XYZ": 95,
    "MULTIRES_TRIPLANE": 83,
    "DENSE3D": 109,
}


class SharedQuery(nn.Module):
    def __init__(self, inp: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, n, d = x.shape
        return self.net(x.reshape(s * n, d)).reshape(s, n, OUTPUT_DIM)


class PerSceneQuery(nn.Module):
    def __init__(self, scene_count: int, inp: int, hidden: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(scene_count, hidden, inp))
        self.b1 = nn.Parameter(torch.zeros(scene_count, hidden))
        self.w2 = nn.Parameter(torch.empty(scene_count, hidden, hidden))
        self.b2 = nn.Parameter(torch.zeros(scene_count, hidden))
        self.w3 = nn.Parameter(torch.empty(scene_count, OUTPUT_DIM, hidden))
        self.b3 = nn.Parameter(torch.zeros(scene_count, OUTPUT_DIM))
        for s in range(scene_count):
            nn.init.kaiming_uniform_(self.w1[s], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[s], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w3[s], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("sni,shi->snh", x, self.w1) + self.b1[:, None, :]
        h = F.silu(h)
        h = torch.einsum("sni,shi->snh", h, self.w2) + self.b2[:, None, :]
        h = F.silu(h)
        return torch.einsum("snh,soh->sno", h, self.w3) + self.b3[:, None, :]


class OracleModel(nn.Module):
    def __init__(self, arm: str, regime: str, scene_count: int):
        super().__init__()
        self.arm = arm
        self.regime = regime
        self.carrier = CarrierBank(arm, scene_count)
        hidden = QUERY_HIDDEN[arm]
        if regime == "PER_SCENE_Q":
            self.query = PerSceneQuery(scene_count, self.carrier.feature_dim, hidden)
        elif regime == "SHARED_Q":
            self.query = SharedQuery(self.carrier.feature_dim, hidden)
        else:
            raise ValueError(regime)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.query(self.carrier.features(points))

    def query_parameters_per_scene_equivalent(self) -> float:
        q = sum(p.numel() for p in self.query.parameters())
        if self.regime == "PER_SCENE_Q":
            scene_count = self.carrier.base.shape[0] if hasattr(self.carrier, "base") else self.carrier.low.shape[0]
            return q / scene_count
        return float(q)


def normalized_rmse(pred: torch.Tensor, truth: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rmse = torch.sqrt(torch.mean((pred - truth) ** 2, dim=1))
    return torch.mean(rmse / scale, dim=1)


def _sample_batch(bank: DatasetBank, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    s, p, _ = bank.train_points.shape
    idx = torch.randint(0, p, (s, batch_size), generator=generator)
    gather_p = idx[..., None].expand(-1, -1, 3)
    gather_y = idx[..., None].expand(-1, -1, OUTPUT_DIM)
    return torch.gather(bank.train_points, 1, gather_p), torch.gather(bank.train_targets, 1, gather_y)


def train_model(
    arm: str,
    regime: str,
    bank: DatasetBank,
    lr: float,
    steps: int,
    seed: int,
) -> tuple[OracleModel, list[float]]:
    seed_all(seed)
    model = OracleModel(arm, regime, len(bank.scenes))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = _scene_generator(seed * 7919 + 17)
    trace: list[float] = []
    scale = bank.train_scale[:, None, :]
    for step in range(steps):
        p, y = _sample_batch(bank, BATCH_PER_SCENE, g)
        pred = model(p)
        loss = torch.mean(((pred - y) / scale) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1):
            trace.append(float(loss.detach()))
    return model, trace


def summarize_model(model: OracleModel, bank: DatasetBank) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(bank.eval_points)
        per_scene = normalized_rmse(pred, bank.eval_targets, bank.train_scale)
        rows = []
        for i, scene in enumerate(bank.scenes):
            mask = bank.eval_boundary[i]
            boundary_nrmse = None
            if scene.family != "VOLUMETRIC_3WAY":
                if int(mask.sum()) < 16:
                    raise RuntimeError("BOUNDARY_SUBSET_TOO_SMALL")
                bpred = pred[i][mask]
                btruth = bank.eval_targets[i][mask]
                scale = bank.train_scale[i]
                boundary_nrmse = float(normalized_rmse(bpred[None], btruth[None], scale[None])[0])
            rows.append(
                {
                    "seed": scene.replicate_seed,
                    "scene_seed": scene.seed,
                    "family": scene.family,
                    "difficulty": scene.difficulty,
                    "nrmse": float(per_scene[i]),
                    "boundary_nrmse": boundary_nrmse,
                }
            )

    groups: dict[str, dict] = {}
    keys = sorted({(r["family"], r["difficulty"]) for r in rows})
    for family, difficulty in keys:
        rr = [r for r in rows if r["family"] == family and r["difficulty"] == difficulty]
        key = f"{family}:{difficulty:g}"
        groups[key] = {
            "family": family,
            "difficulty": difficulty,
            "mean_nrmse": sum(r["nrmse"] for r in rr) / len(rr),
            "worst_nrmse": max(r["nrmse"] for r in rr),
            "per_seed_nrmse": [r["nrmse"] for r in rr],
        }
        bvals = [r["boundary_nrmse"] for r in rr if r["boundary_nrmse"] is not None]
        if bvals:
            groups[key]["mean_boundary_nrmse"] = sum(bvals) / len(bvals)
            groups[key]["worst_boundary_nrmse"] = max(bvals)
    return {"rows": rows, "groups": groups}


def calibration_bank(train_pool: int, eval_points: int) -> DatasetBank:
    scenes = [make_volumetric_scene(880000 + seed, seed, 1) for seed in CALIBRATION_SEEDS]
    return build_bank(scenes, train_pool, eval_points)


def calibrate(train_pool: int, eval_points: int, steps: int) -> dict:
    bank = calibration_bank(train_pool, eval_points)
    rows = []
    for lr in CALIBRATION_LRS:
        model, trace = train_model("DENSE3D", "PER_SCENE_Q", bank, lr, steps, seed=424242)
        summary = summarize_model(model, bank)
        vals = [r["nrmse"] for r in summary["rows"]]
        rows.append(
            {
                "lr": lr,
                "mean_nrmse": sum(vals) / len(vals),
                "worst_nrmse": max(vals),
                "trace": trace,
            }
        )
    winner = min(rows, key=lambda r: (round(r["mean_nrmse"], 12), r["lr"]))
    valid = winner["mean_nrmse"] <= 0.08 and winner["worst_nrmse"] <= 0.12
    return {"rows": rows, "selected_lr": winner["lr"], "valid": valid, "winner": winner}


def firewall(lr: float, steps: int, train_pool: int, eval_points: int) -> dict:
    scenes = [make_volumetric_scene(990001, 990001, 1), make_volumetric_scene(990002, 990002, 1)]
    bank = build_bank(scenes, train_pool, eval_points)
    g1 = _scene_generator(123456)
    g2 = _scene_generator(654321)
    bank.train_targets = torch.randn(bank.train_targets.shape, generator=g1)
    bank.eval_targets = torch.randn(bank.eval_targets.shape, generator=g2)
    bank.train_scale = bank.train_targets.std(dim=1).clamp_min(0.10)
    model, trace = train_model("DENSE3D", "SHARED_Q", bank, lr, steps, seed=777)
    summary = summarize_model(model, bank)
    vals = [r["nrmse"] for r in summary["rows"]]
    mean = sum(vals) / len(vals)
    return {"mean_nrmse": mean, "worst_nrmse": max(vals), "valid": mean >= 0.90, "trace": trace}


def surface_support(groups: dict[str, dict]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for factor in SURFACE_FACTORS:
        radius = factor * H_REF
        s = groups[f"SURFACE_STRUCTURED:{radius:g}"]
        t = groups[f"THIN_SURFACE:{radius:g}"]
        out[f"radius_{radius:g}"] = (
            s["mean_nrmse"] <= 0.08
            and t["mean_nrmse"] <= 0.12
            and s["worst_nrmse"] <= 0.18
            and t["worst_nrmse"] <= 0.18
            and s["mean_boundary_nrmse"] <= 0.16
            and t["mean_boundary_nrmse"] <= 0.16
        )
    return out


def select_diagnostics(results: dict[str, dict]) -> dict:
    base_upper = results["TRIPLANE/PER_SCENE_Q"]["groups"]
    dense_upper = results["DENSE3D/PER_SCENE_Q"]["groups"]
    base_shared = results["TRIPLANE/SHARED_Q"]["groups"]
    xyz_shared = results["TRIPLANE_XYZ/SHARED_Q"]["groups"]
    multi_shared = results["MULTIRES_TRIPLANE/SHARED_Q"]["groups"]

    support = surface_support(base_upper)
    credible_concern = {}
    for factor in SURFACE_FACTORS:
        radius = factor * H_REF
        key = f"THIN_SURFACE:{radius:g}"
        tri = base_upper[key]
        dense = dense_upper[key]
        dense_solves = dense["mean_nrmse"] <= 0.12 and dense["worst_nrmse"] <= 0.18
        tri_fails = tri["mean_nrmse"] > 0.12 or tri["worst_nrmse"] > 0.18
        credible_concern[f"radius_{radius:g}"] = bool(dense_solves and tri_fails)

    xyz_gains = []
    xyz_degradation_ok = True
    multi_gains = []
    multi_struct_ok = True
    for factor in SURFACE_FACTORS:
        radius = factor * H_REF
        thin_key = f"THIN_SURFACE:{radius:g}"
        struct_key = f"SURFACE_STRUCTURED:{radius:g}"
        bthin = base_shared[thin_key]["worst_nrmse"]
        xyz = xyz_shared[thin_key]["worst_nrmse"]
        multi = multi_shared[thin_key]["worst_nrmse"]
        xyz_gains.append((bthin - xyz) / max(bthin, 1e-9))
        multi_gains.append((bthin - multi) / max(bthin, 1e-9))
        if xyz_shared[struct_key]["mean_nrmse"] > base_shared[struct_key]["mean_nrmse"] + 0.01:
            xyz_degradation_ok = False
        if multi_shared[struct_key]["mean_nrmse"] > base_shared[struct_key]["mean_nrmse"] + 0.01:
            multi_struct_ok = False

    return {
        "triplane_surface_support_curve": support,
        "credible_triplane_concern_curve": credible_concern,
        "explicit_xyz_preferred": max(xyz_gains) >= 0.20 and xyz_degradation_ok,
        "explicit_xyz_best_worst_seed_gain": max(xyz_gains),
        "multires_preferred": max(multi_gains) >= 0.20 and multi_struct_ok,
        "multires_best_thin_worst_seed_gain": max(multi_gains),
    }


def platform_context() -> dict[str, str]:
    required = (
        "REALSAS_EXPERIMENT_ID",
        "REALSAS_ATTEMPT_ID",
        "REALSAS_ENGINE_RELEASE_ID",
        "REALSAS_CANONICAL_CODE_SHA",
    )
    out = {key: os.environ.get(key, "").strip() for key in required}
    missing = [key for key, value in out.items() if not value]
    if missing:
        raise RuntimeError("PLATFORM_CONTEXT_REQUIRED:" + ",".join(missing))
    return out


def run_scientific() -> dict:
    ctx = platform_context()
    calibration = calibrate(TRAIN_POOL, EVAL_POINTS, CALIBRATION_STEPS)
    if not calibration["valid"]:
        return {
            "schema": "RealSaS.OracleZRepresentationCourt.v2",
            "mode": "PLATFORM_SCIENTIFIC",
            "status": "APPARATUS_INCONCLUSIVE__CALIBRATION_FAIL",
            "platform_context": ctx,
            "calibration": calibration,
            "authority": {"product_authority_changed": False, "model_promotion_authorized": False},
        }

    selected_lr = float(calibration["selected_lr"])
    fw = firewall(selected_lr, min(COMPARISON_STEPS, 1200), TRAIN_POOL, EVAL_POINTS)
    if not fw["valid"]:
        return {
            "schema": "RealSaS.OracleZRepresentationCourt.v2",
            "mode": "PLATFORM_SCIENTIFIC",
            "status": "APPARATUS_INCONCLUSIVE__FIREWALL_FAIL",
            "platform_context": ctx,
            "calibration": calibration,
            "firewall": fw,
            "authority": {"product_authority_changed": False, "model_promotion_authorized": False},
        }

    scenes = make_comparison_scenes(SCIENTIFIC_SEEDS)
    bank = build_bank(scenes, TRAIN_POOL, EVAL_POINTS)
    results: dict[str, dict] = {}
    for arm_i, arm in enumerate(ARMS):
        for regime_i, regime in enumerate(REGIMES):
            model, trace = train_model(
                arm,
                regime,
                bank,
                selected_lr,
                COMPARISON_STEPS,
                seed=555000 + arm_i * 101 + regime_i * 13,
            )
            summary = summarize_model(model, bank)
            key = f"{arm}/{regime}"
            results[key] = {
                "carrier_scalars_per_scene": model.carrier.scalars_per_scene(),
                "query_parameters_total": sum(p.numel() for p in model.query.parameters()),
                "query_parameters_per_scene_equivalent": model.query_parameters_per_scene_equivalent(),
                "trace": trace,
                **summary,
            }

    diagnostics = select_diagnostics(results)
    return {
        "schema": "RealSaS.OracleZRepresentationCourt.v2",
        "mode": "PLATFORM_SCIENTIFIC",
        "status": "VALID_REPRESENTATION_CAPACITY_CURVES_V2",
        "platform_context": ctx,
        "configuration": {
            "scientific_seeds": list(SCIENTIFIC_SEEDS),
            "calibration_seeds": list(CALIBRATION_SEEDS),
            "calibration_lrs": list(CALIBRATION_LRS),
            "calibration_steps": CALIBRATION_STEPS,
            "comparison_steps": COMPARISON_STEPS,
            "train_pool": TRAIN_POOL,
            "eval_points": EVAL_POINTS,
            "batch_per_scene": BATCH_PER_SCENE,
            "h_ref": H_REF,
            "surface_factors": list(SURFACE_FACTORS),
            "volume_freqs": list(VOLUME_FREQS),
            "loss": "PER_SCENE_PER_CHANNEL_STD_NORMALIZED_MSE",
        },
        "calibration": calibration,
        "firewall": fw,
        "diagnostics": diagnostics,
        "results": results,
        "authority": {
            "product_authority_changed": False,
            "truth139_training_authorized": False,
            "fit8_training_authorized": False,
            "model_promotion_authorized": False,
        },
    }


def run_smoke() -> dict:
    cal = calibrate(train_pool=128, eval_points=64, steps=2)
    scenes = make_comparison_scenes(SCIENTIFIC_SEEDS[:1])
    bank = build_bank(scenes, train_pool=128, eval_points=512)
    rows = {}
    for arm in ARMS:
        for regime in REGIMES:
            model, trace = train_model(arm, regime, bank, lr=3e-3, steps=2, seed=42)
            rows[f"{arm}/{regime}"] = {
                "carrier_scalars_per_scene": model.carrier.scalars_per_scene(),
                "query_parameters_total": sum(p.numel() for p in model.query.parameters()),
                "trace": trace,
                "summary": summarize_model(model, bank),
            }
    return {
        "schema": "RealSaS.OracleZRepresentationCourt.v2",
        "mode": "ENGINEERING_SMOKE_ONLY",
        "status": "ENGINEERING_SMOKE_PASS",
        "scientific_authority": False,
        "calibration_smoke": cal,
        "arms": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "scientific"), default="smoke")
    args = parser.parse_args()
    result = run_smoke() if args.mode == "smoke" else run_scientific()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"mode": result["mode"], "status": result["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
