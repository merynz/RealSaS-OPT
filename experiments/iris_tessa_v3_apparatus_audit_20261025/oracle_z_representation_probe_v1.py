#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

SEEDS = (731, 947, 1217)
SCENES_PER_FAMILY = 3
STEPS = 500
BATCH_PER_SCENE = 64
EVAL_POINTS = 4096


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SceneDef:
    family: str
    centers: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    attrs: torch.Tensor | None = None
    fourier_w: torch.Tensor | None = None
    fourier_a: torch.Tensor | None = None
    fourier_b: torch.Tensor | None = None


def make_scene(family: str, seed: int) -> SceneDef:
    g = torch.Generator().manual_seed(seed)
    if family in ("SURFACE_STRUCTURED", "THIN_SURFACE"):
        k = 4
        centers = torch.empty(k, 3).uniform_(-0.55, 0.55, generator=g)
        scales = torch.empty(k, 3).uniform_(0.18, 0.42, generator=g)
        if family == "THIN_SURFACE":
            # Deliberately sub-carrier-ish local thickness on one axis.
            for i in range(k):
                axis = i % 3
                scales[i, axis] = 0.055 + 0.015 * torch.rand((), generator=g)
        attrs = torch.empty(k, 3).uniform_(0.05, 0.95, generator=g)
        return SceneDef(family=family, centers=centers, scales=scales, attrs=attrs)

    # Smooth scene-specific genuine xyz-interaction control.
    terms = 24
    fourier_w = torch.randint(1, 8, (4, terms, 3), generator=g).float()
    signs = torch.randint(0, 2, (4, terms, 3), generator=g).float() * 2.0 - 1.0
    fourier_w = fourier_w * signs
    fourier_a = torch.randn(4, terms, generator=g) / math.sqrt(terms)
    fourier_b = torch.rand(4, terms, generator=g) * (2.0 * math.pi)
    return SceneDef(family=family, fourier_w=fourier_w, fourier_a=fourier_a, fourier_b=fourier_b)


def scene_targets(scene: SceneDef, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scene.family in ("SURFACE_STRUCTURED", "THIN_SURFACE"):
        q = (points[:, None, :] - scene.centers[None]) / scene.scales[None]
        d = torch.linalg.vector_norm(q, dim=-1) - 1.0
        sdf, idx = d.min(dim=1)
        attr = scene.attrs[idx]
        # Geometry plus three component-local authored quantities.
        y = torch.cat((sdf.clamp(-1.0, 1.0)[:, None], attr), dim=1)
        boundary = sdf.abs() <= (0.12 if scene.family == "SURFACE_STRUCTURED" else 0.08)
        return y, boundary

    # sin(2pi * <w,x> + b) with genuine 3-way frequency vectors.
    phase = 2.0 * math.pi * torch.einsum("nd,ktd->nkt", points, scene.fourier_w) + scene.fourier_b[None]
    y = torch.einsum("nkt,kt->nk", torch.sin(phase), scene.fourier_a)
    y = torch.tanh(y)
    return y, torch.ones(points.shape[0], dtype=torch.bool)


def sample_points(scene: SceneDef, n: int, generator: torch.Generator) -> torch.Tensor:
    # Mix uniform volume and surface-near points for surface families.
    if scene.family == "VOLUMETRIC_3WAY_CONTROL":
        return torch.empty(n, 3).uniform_(-0.95, 0.95, generator=generator)
    nu = n // 2
    uniform = torch.empty(nu, 3).uniform_(-0.95, 0.95, generator=generator)
    ns = n - nu
    comp = torch.randint(0, scene.centers.shape[0], (ns,), generator=generator)
    direction = torch.randn(ns, 3, generator=generator)
    direction = F.normalize(direction, dim=-1)
    radial = 1.0 + 0.12 * torch.randn(ns, 1, generator=generator)
    near = scene.centers[comp] + direction * scene.scales[comp] * radial
    near = near.clamp(-0.98, 0.98)
    return torch.cat((uniform, near), dim=0)


def sample_triplane(planes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    # planes [3,C,R,R], points [N,3]. Exact current semantic ordering.
    x, y, z = points.unbind(-1)
    grids = (torch.stack((x, y), -1), torch.stack((x, z), -1), torch.stack((y, z), -1))
    out = []
    for i, grid in enumerate(grids):
        s = F.grid_sample(planes[i:i+1], grid.view(1, -1, 1, 2), mode="bilinear", padding_mode="border", align_corners=False)
        out.append(s.squeeze(0).squeeze(-1).transpose(0, 1))
    return torch.cat(out, dim=-1)


def sample_dense(grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    # grid [C,D,H,W], grid_sample expects xyz order in final coordinate axis.
    s = F.grid_sample(grid[None], points.view(1, -1, 1, 1, 3), mode="bilinear", padding_mode="border", align_corners=False)
    return s.squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)


class QueryMLP(nn.Module):
    def __init__(self, inp: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Arm(nn.Module):
    def __init__(self, kind: str, scene_count: int):
        super().__init__()
        self.kind = kind
        if kind in ("TRIPLANE_Q", "TRIPLANE_Q_XYZ"):
            self.carrier = nn.Parameter(torch.randn(scene_count, 3, 16, 24, 24) * 0.02)
            inp = 48 + (3 if kind.endswith("XYZ") else 0)
        elif kind == "MULTIRES_TRIPLANE_Q":
            # Carrier scalars: 3*(12^2*16 + 24^2*12) = 27648, exactly base budget.
            self.low = nn.Parameter(torch.randn(scene_count, 3, 16, 12, 12) * 0.02)
            self.high = nn.Parameter(torch.randn(scene_count, 3, 12, 24, 24) * 0.02)
            inp = 3 * (16 + 12)
        elif kind == "DENSE3D_Q":
            # 12^3*16 = 27648, exactly base carrier budget.
            self.carrier = nn.Parameter(torch.randn(scene_count, 16, 12, 12, 12) * 0.02)
            inp = 16
        else:
            raise ValueError(kind)
        self.query = QueryMLP(inp)

    def features(self, scene_idx: int, points: torch.Tensor) -> torch.Tensor:
        if self.kind in ("TRIPLANE_Q", "TRIPLANE_Q_XYZ"):
            f = sample_triplane(self.carrier[scene_idx], points)
            return torch.cat((f, points), -1) if self.kind.endswith("XYZ") else f
        if self.kind == "MULTIRES_TRIPLANE_Q":
            return torch.cat((sample_triplane(self.low[scene_idx], points), sample_triplane(self.high[scene_idx], points)), -1)
        return sample_dense(self.carrier[scene_idx], points)

    def forward_scene(self, scene_idx: int, points: torch.Tensor) -> torch.Tensor:
        return self.query(self.features(scene_idx, points))

    def carrier_scalar_count_per_scene(self) -> int:
        if self.kind == "MULTIRES_TRIPLANE_Q":
            return self.low[0].numel() + self.high[0].numel()
        return self.carrier[0].numel()


def normalized_rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    rmse = torch.sqrt(torch.mean((pred - truth) ** 2, dim=0))
    scale = truth.std(dim=0).clamp_min(0.10)
    return float(torch.mean(rmse / scale))


def train_arm(kind: str, scenes: list[SceneDef], seed: int) -> dict:
    seed_all(seed)
    arm = Arm(kind, len(scenes))
    opt = torch.optim.AdamW(arm.parameters(), lr=3e-3, weight_decay=0.0)
    gens = [torch.Generator().manual_seed(seed * 1000 + i * 17) for i in range(len(scenes))]

    for _ in range(STEPS):
        losses = []
        for i, scene in enumerate(scenes):
            p = sample_points(scene, BATCH_PER_SCENE, gens[i])
            y, _ = scene_targets(scene, p)
            losses.append(F.smooth_l1_loss(arm.forward_scene(i, p), y))
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    per_family: dict[str, list[dict]] = {}
    with torch.no_grad():
        for i, scene in enumerate(scenes):
            g = torch.Generator().manual_seed(900000 + seed * 37 + i)
            p = sample_points(scene, EVAL_POINTS, g)
            y, boundary = scene_targets(scene, p)
            pred = arm.forward_scene(i, p)
            row = {"nrmse": normalized_rmse(pred, y)}
            if scene.family != "VOLUMETRIC_3WAY_CONTROL":
                if int(boundary.sum()) < 32:
                    raise RuntimeError("BOUNDARY_SUBSET_TOO_SMALL")
                row["boundary_nrmse"] = normalized_rmse(pred[boundary], y[boundary])
            per_family.setdefault(scene.family, []).append(row)

    summary = {}
    for family, rows in per_family.items():
        summary[family] = {
            "mean_nrmse": sum(r["nrmse"] for r in rows) / len(rows),
            "worst_nrmse": max(r["nrmse"] for r in rows),
        }
        if "boundary_nrmse" in rows[0]:
            summary[family]["mean_boundary_nrmse"] = sum(r["boundary_nrmse"] for r in rows) / len(rows)
            summary[family]["worst_boundary_nrmse"] = max(r["boundary_nrmse"] for r in rows)

    return {
        "seed": seed,
        "arm": kind,
        "carrier_scalars_per_scene": arm.carrier_scalar_count_per_scene(),
        "shared_parameters": sum(p.numel() for n, p in arm.named_parameters() if not ("carrier" in n or ".low" in n or ".high" in n or n.startswith("low") or n.startswith("high"))),
        "summary": summary,
    }


def run() -> dict:
    scene_defs: list[SceneDef] = []
    for family_i, family in enumerate(("SURFACE_STRUCTURED", "THIN_SURFACE", "VOLUMETRIC_3WAY_CONTROL")):
        for i in range(SCENES_PER_FAMILY):
            scene_defs.append(make_scene(family, 139000 + family_i * 1000 + i * 97))

    arms = ("TRIPLANE_Q", "TRIPLANE_Q_XYZ", "MULTIRES_TRIPLANE_Q", "DENSE3D_Q")
    rows = [train_arm(arm, scene_defs, seed) for arm in arms for seed in SEEDS]

    def arm_rows(name: str) -> list[dict]:
        return [r for r in rows if r["arm"] == name]

    def fam_values(name: str, family: str, key: str) -> list[float]:
        return [r["summary"][family][key] for r in arm_rows(name)]

    base_struct = fam_values("TRIPLANE_Q", "SURFACE_STRUCTURED", "mean_nrmse")
    base_thin = fam_values("TRIPLANE_Q", "THIN_SURFACE", "mean_nrmse")
    base_worst = max(
        max(fam_values("TRIPLANE_Q", "SURFACE_STRUCTURED", "worst_nrmse")),
        max(fam_values("TRIPLANE_Q", "THIN_SURFACE", "worst_nrmse")),
    )
    xyz_worst = max(
        max(fam_values("TRIPLANE_Q_XYZ", "SURFACE_STRUCTURED", "worst_nrmse")),
        max(fam_values("TRIPLANE_Q_XYZ", "THIN_SURFACE", "worst_nrmse")),
    )
    multires_thin_worst = max(fam_values("MULTIRES_TRIPLANE_Q", "THIN_SURFACE", "worst_nrmse"))
    base_thin_worst = max(fam_values("TRIPLANE_Q", "THIN_SURFACE", "worst_nrmse"))
    multires_struct_mean = sum(fam_values("MULTIRES_TRIPLANE_Q", "SURFACE_STRUCTURED", "mean_nrmse")) / len(SEEDS)
    base_struct_mean = sum(base_struct) / len(SEEDS)

    gates = {
        "base_surface_structured": (sum(base_struct) / len(SEEDS)) <= 0.06,
        "base_thin_surface": (sum(base_thin) / len(SEEDS)) <= 0.10,
        "base_worst_seed": base_worst <= 0.15,
    }
    xyz_improvement = (base_worst - xyz_worst) / max(base_worst, 1e-9)
    multires_improvement = (base_thin_worst - multires_thin_worst) / max(base_thin_worst, 1e-9)
    selection = {
        "explicit_xyz_supported": xyz_improvement >= 0.20,
        "explicit_xyz_relative_worst_improvement": xyz_improvement,
        "multires_supported": multires_improvement >= 0.20 and multires_struct_mean <= base_struct_mean + 0.01,
        "multires_thin_worst_improvement": multires_improvement,
    }

    return {
        "schema": "RealSaS.OracleZRepresentationCourt.v1",
        "status": "ORACLE_Z_TINY_SURFACE_CAPACITY_SUPPORTED" if all(gates.values()) else "ORACLE_Z_TINY_SURFACE_CAPACITY_NOT_ESTABLISHED",
        "configuration": {"steps": STEPS, "batch_per_scene": BATCH_PER_SCENE, "eval_points": EVAL_POINTS, "seeds": list(SEEDS), "scenes_per_family": SCENES_PER_FAMILY},
        "gates": gates,
        "selection": selection,
        "rows": rows,
        "authority": {"product_authority_changed": False, "truth139_training_authorized": False, "fit8_training_authorized": False, "gpu_required": False},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "gates": result["gates"], "selection": result["selection"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
