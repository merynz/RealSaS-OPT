#!/usr/bin/env python3
"""Execute SURFACE_WITNESS_SAMPLER_PREREG_V1 deterministically."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

SCHEMA = "RealSaS.SurfaceWitnessSamplerCourt.v1"
BUDGET = 64
PERMUTATIONS = 16
LAMBDAS = (0.0, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class StressSurface:
    name: str
    xyz: torch.Tensor
    normal: torch.Tensor
    group: torch.Tensor
    rare_groups: tuple[int, ...]


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1.0e-8)


def _fibonacci_sphere(count: int, *, center=(0.0, 0.0, 0.0), radius=1.0) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(count, dtype=torch.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / float(count)
    r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    theta = golden * i
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    n = torch.stack((x, y, z), dim=-1).float()
    c = torch.tensor(center, dtype=torch.float32)
    return c + float(radius) * n, _unit(n)


def _grid_sheet(nx: int, ny: int, *, z: float, normal: tuple[float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 1.0, nx)
    y = torch.linspace(-1.0, 1.0, ny)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xyz = torch.stack((xx, yy, torch.full_like(xx, float(z))), dim=-1).reshape(-1, 3)
    n = torch.tensor(normal, dtype=torch.float32)[None].expand(xyz.shape[0], -1).clone()
    return xyz, _unit(n)


def stress_surfaces() -> tuple[StressSurface, ...]:
    # 1. Dense body with a low-area thin appendage and tiny terminal group.
    body, body_n = _fibonacci_sphere(512, radius=0.58)
    t = torch.linspace(0.0, 1.0, 48)
    append = torch.stack((0.55 + 0.75 * t, 0.04 * torch.sin(4.0 * math.pi * t), torch.zeros_like(t)), dim=-1)
    tangent = torch.stack((torch.ones_like(t), 0.16 * math.pi * torch.cos(4.0 * math.pi * t), torch.zeros_like(t)), dim=-1)
    append_n = _unit(torch.cross(tangent, torch.tensor([0.0, 0.0, 1.0]).expand_as(tangent), dim=-1))
    a = torch.linspace(0.0, 2.0 * math.pi, 9)[:-1]
    terminal = torch.stack((torch.full_like(a, 1.34), 0.035 * torch.cos(a), 0.035 * torch.sin(a)), dim=-1)
    terminal_n = _unit(torch.stack((torch.zeros_like(a), torch.cos(a), torch.sin(a)), dim=-1))
    xyz = torch.cat((body, append, terminal), dim=0)
    normal = torch.cat((body_n, append_n, terminal_n), dim=0)
    group = torch.cat((torch.zeros(512, dtype=torch.long), torch.ones(48, dtype=torch.long), torch.full((8,), 2, dtype=torch.long)))
    s1 = StressSurface("DENSE_BODY_PLUS_TINY_DISTANT_APPENDAGE", xyz, normal, group, (1, 2))

    # 2. Spatially close sheets with opposite normals.
    s_top, n_top = _grid_sheet(18, 18, z=0.018, normal=(0.0, 0.0, 1.0))
    s_bottom, n_bottom = _grid_sheet(18, 18, z=-0.018, normal=(0.0, 0.0, -1.0))
    xyz = torch.cat((s_top, s_bottom), dim=0)
    normal = torch.cat((n_top, n_bottom), dim=0)
    group = torch.cat((torch.zeros(s_top.shape[0], dtype=torch.long), torch.ones(s_bottom.shape[0], dtype=torch.long)))
    s2 = StressSurface("NEARBY_OPPOSITE_NORMAL_SHEETS", xyz, normal, group, (0, 1))

    # 3. Dense body plus a long low-area curved support.
    body, body_n = _fibonacci_sphere(448, center=(-0.35, 0.0, 0.0), radius=0.52)
    t = torch.linspace(0.0, 1.0, 72)
    curve = torch.stack((0.15 + 1.25 * t, 0.22 * torch.sin(math.pi * t), 0.13 * torch.sin(2.0 * math.pi * t)), dim=-1)
    d = torch.gradient(curve, spacing=(t,), dim=0)[0]
    ref = torch.tensor([0.0, 0.0, 1.0]).expand_as(d)
    curve_n = _unit(torch.cross(d, ref, dim=-1))
    xyz = torch.cat((body, curve), dim=0)
    normal = torch.cat((body_n, curve_n), dim=0)
    group = torch.cat((torch.zeros(body.shape[0], dtype=torch.long), torch.ones(curve.shape[0], dtype=torch.long)))
    s3 = StressSurface("LONG_THIN_CURVE_PLUS_BODY", xyz, normal, group, (1,))

    # 4. One large component and sparse components at different scales/separations.
    large, large_n = _fibonacci_sphere(480, center=(-0.25, 0.0, 0.0), radius=0.52)
    c1, n1 = _fibonacci_sphere(24, center=(0.90, 0.65, 0.10), radius=0.10)
    c2, n2 = _fibonacci_sphere(14, center=(1.10, -0.55, -0.05), radius=0.065)
    c3, n3 = _fibonacci_sphere(8, center=(0.62, 0.10, 0.72), radius=0.045)
    xyz = torch.cat((large, c1, c2, c3), dim=0)
    normal = torch.cat((large_n, n1, n2, n3), dim=0)
    group = torch.cat((torch.zeros(480, dtype=torch.long), torch.ones(24, dtype=torch.long), torch.full((14,), 2, dtype=torch.long), torch.full((8,), 3, dtype=torch.long)))
    s4 = StressSurface("MULTISCALE_COMPONENTS", xyz, normal, group, (1, 2, 3))

    # 5. Approximately uniform surface, used as a global-coverage guard.
    uniform, uniform_n = _fibonacci_sphere(640, radius=1.0)
    s5 = StressSurface("UNIFORM_SURFACE", uniform, uniform_n, torch.zeros(640, dtype=torch.long), ())
    return (s1, s2, s3, s4, s5)


def _canonical_order(xyz: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    rows = torch.cat((xyz.double(), normal.double()), dim=-1).tolist()
    order = sorted(range(len(rows)), key=lambda i: tuple(float(v) for v in rows[i]))
    return torch.tensor(order, dtype=torch.long)


def current_index_indices(count: int, budget: int) -> torch.Tensor:
    return torch.floor(torch.arange(budget, dtype=torch.float64) * (float(count) / float(budget))).long().clamp_max(count - 1)


def fps_indices(xyz: torch.Tensor, normal: torch.Tensor, budget: int, lam: float) -> torch.Tensor:
    if xyz.shape[0] <= budget:
        return torch.arange(xyz.shape[0], dtype=torch.long)
    order = _canonical_order(xyz, normal)
    xyz_c = xyz[order].double()
    normal_c = _unit(normal[order]).double()
    diag = float(torch.linalg.vector_norm(xyz_c.max(dim=0).values - xyz_c.min(dim=0).values))
    if not math.isfinite(diag) or diag <= 0.0:
        raise ValueError("SAMPLER_STRESS_BBOX_DEGENERATE")
    xyz_c = xyz_c / diag
    selected = [0]
    min_d = torch.full((xyz_c.shape[0],), float("inf"), dtype=torch.float64)
    for _ in range(1, budget):
        j = selected[-1]
        dxyz = ((xyz_c - xyz_c[j]) ** 2).sum(dim=-1)
        if lam > 0.0:
            dn = ((normal_c - normal_c[j]) ** 2).sum(dim=-1)
            d = dxyz + (float(lam) ** 2) * dn
        else:
            d = dxyz
        min_d = torch.minimum(min_d, d)
        min_d[torch.tensor(selected, dtype=torch.long)] = -1.0
        best = float(min_d.max())
        # canonical order is already lexicographic, so first max is deterministic tie-break.
        next_idx = int(torch.nonzero(min_d == best, as_tuple=False)[0, 0])
        selected.append(next_idx)
    return order[torch.tensor(selected, dtype=torch.long)]


def metrics(surface: StressSurface, selected: torch.Tensor) -> dict[str, float | dict[str, bool]]:
    xyz = surface.xyz.double()
    normal = _unit(surface.normal).double()
    sel_xyz = xyz[selected]
    sel_n = normal[selected]
    diag = float(torch.linalg.vector_norm(xyz.max(dim=0).values - xyz.min(dim=0).values))
    dist = torch.cdist(xyz, sel_xyz)
    nearest_dist, nearest = dist.min(dim=1)
    cover = float(nearest_dist.max()) / diag
    dots = (normal * sel_n[nearest]).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(dots))
    normal_p95 = float(torch.quantile(angle.float(), 0.95))
    rare = {}
    for gid in surface.rare_groups:
        source_mask = surface.group == int(gid)
        selected_hit = bool((surface.group[selected] == int(gid)).any())
        if not bool(source_mask.any()):
            raise AssertionError("RARE_GROUP_EMPTY")
        rare[str(gid)] = selected_hit
    return {"spatial_cover_radius": cover, "normal_mismatch_p95_deg": normal_p95, "rare_group_hits": rare}


def run() -> dict:
    surfaces = stress_surfaces()
    arms = {
        "CURRENT_INDEX_UNIFORM": None,
        "XYZ_FPS": 0.0,
        "XYZ_NORMAL_FPS_0p05": 0.05,
        "XYZ_NORMAL_FPS_0p10": 0.10,
        "XYZ_NORMAL_FPS_0p20": 0.20,
    }
    records = []
    selected_sets: dict[tuple[str, str], list[tuple[int, ...]]] = {}
    for surface_index, surface in enumerate(surfaces):
        n = surface.xyz.shape[0]
        if n <= BUDGET:
            raise ValueError("STRESS_SURFACE_TOO_SMALL")
        for permutation_index in range(PERMUTATIONS):
            g = torch.Generator().manual_seed(731000 + surface_index * 1000 + permutation_index)
            permutation = torch.randperm(n, generator=g)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(n)
            xyz_p = surface.xyz[permutation]
            normal_p = surface.normal[permutation]
            group_p = surface.group[permutation]
            perm_surface = StressSurface(surface.name, xyz_p, normal_p, group_p, surface.rare_groups)
            for arm, lam in arms.items():
                if lam is None:
                    selected_p = current_index_indices(n, BUDGET)
                else:
                    selected_p = fps_indices(xyz_p, normal_p, BUDGET, lam)
                selected_canonical = permutation[selected_p]
                canonical_tuple = tuple(sorted(int(v) for v in selected_canonical.tolist()))
                selected_sets.setdefault((surface.name, arm), []).append(canonical_tuple)
                m = metrics(perm_surface, selected_p)
                records.append({
                    "surface": surface.name,
                    "permutation": permutation_index,
                    "arm": arm,
                    **m,
                })

    summary = {}
    current_worst = max(float(r["spatial_cover_radius"]) for r in records if r["arm"] == "CURRENT_INDEX_UNIFORM")
    current_uniform = max(float(r["spatial_cover_radius"]) for r in records if r["arm"] == "CURRENT_INDEX_UNIFORM" and r["surface"] == "UNIFORM_SURFACE")
    for arm in arms:
        arm_rows = [r for r in records if r["arm"] == arm]
        invariant_by_surface = {
            surface.name: len(set(selected_sets[(surface.name, arm)])) == 1
            for surface in surfaces
        }
        all_rare = []
        for row in arm_rows:
            all_rare.extend(bool(v) for v in row["rare_group_hits"].values())
        rare_recall = (sum(all_rare) / len(all_rare)) if all_rare else 1.0
        cover_values = [float(r["spatial_cover_radius"]) for r in arm_rows]
        normal_values = [float(r["normal_mismatch_p95_deg"]) for r in arm_rows]
        uniform_cover = max(float(r["spatial_cover_radius"]) for r in arm_rows if r["surface"] == "UNIFORM_SURFACE")
        summary[arm] = {
            "permutation_invariant": all(invariant_by_surface.values()),
            "permutation_invariant_by_surface": invariant_by_surface,
            "rare_group_hit_recall": rare_recall,
            "mean_spatial_cover_radius": sum(cover_values) / len(cover_values),
            "worst_spatial_cover_radius": max(cover_values),
            "uniform_surface_worst_cover_radius": uniform_cover,
            "mean_normal_mismatch_p95_deg": sum(normal_values) / len(normal_values),
            "worst_normal_mismatch_p95_deg": max(normal_values),
        }

    current = summary["CURRENT_INDEX_UNIFORM"]
    current_eligible = bool(current["permutation_invariant"]) and float(current["worst_spatial_cover_radius"]) <= 1.05 * min(float(summary[a]["worst_spatial_cover_radius"]) for a in arms)

    eligible = []
    for arm in ("XYZ_FPS", "XYZ_NORMAL_FPS_0p05", "XYZ_NORMAL_FPS_0p10", "XYZ_NORMAL_FPS_0p20"):
        s = summary[arm]
        ok = (
            bool(s["permutation_invariant"])
            and abs(float(s["rare_group_hit_recall"]) - 1.0) <= 1e-12
            and float(s["worst_spatial_cover_radius"]) <= current_worst * 0.80
            and float(s["uniform_surface_worst_cover_radius"]) <= current_uniform * 1.05
        )
        s["eligible"] = ok
        if ok:
            eligible.append(arm)
    current["eligible"] = current_eligible

    selected = None
    if eligible:
        best_cover = min(float(summary[a]["worst_spatial_cover_radius"]) for a in eligible)
        near_cover = [a for a in eligible if float(summary[a]["worst_spatial_cover_radius"]) <= best_cover * 1.02]
        if len(near_cover) == 1:
            selected = near_cover[0]
        else:
            best_normal = min(float(summary[a]["worst_normal_mismatch_p95_deg"]) for a in near_cover)
            near_normal = [a for a in near_cover if float(summary[a]["worst_normal_mismatch_p95_deg"]) <= best_normal * 1.02 + 1e-9]
            simplicity = ("XYZ_FPS", "XYZ_NORMAL_FPS_0p05", "XYZ_NORMAL_FPS_0p10", "XYZ_NORMAL_FPS_0p20")
            selected = next(a for a in simplicity if a in near_normal)

    if selected is None:
        status = "GEOMETRY_NATIVE_SAMPLER_NOT_ESTABLISHED_TINY_V1"
    else:
        status = "GEOMETRY_NATIVE_SAMPLER_SELECTED_TINY_V1"

    return {
        "schema": SCHEMA,
        "status": status,
        "selected_arm": selected,
        "budget": BUDGET,
        "permutations": PERMUTATIONS,
        "summary": summary,
        "records": records,
        "gates": {
            "current_retained": current_eligible,
            "candidate_count_eligible": len(eligible),
        },
        "authority": {
            "product_sampler_replacement_authorized": False,
            "full_1024_real_surface_validation_required": True,
            "truth139_training_authorized": False,
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
    print(json.dumps({"status": result["status"], "selected_arm": result["selected_arm"], "summary": result["summary"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
