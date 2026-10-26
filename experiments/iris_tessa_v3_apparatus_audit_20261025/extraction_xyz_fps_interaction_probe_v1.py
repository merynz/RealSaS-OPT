#!/usr/bin/env python3
"""Execute EXTRACTION_XYZ_FPS_INTERACTION_PREREG_V1 deterministically."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

SCHEMA = "RealSaS.ExtractionXYZFPSInteractionCourt.v1"
RESOLUTIONS = (24, 32, 48, 64)
BUDGETS = (64, 128)
SCENARIOS = 27
PERMUTATIONS = 4
FEATURES = (
    ("sphere", 0.030),
    ("sphere", 0.045),
    ("cylinder", 0.022),
    ("cylinder", 0.035),
    ("gap", 0.018),
    ("gap", 0.028),
)
CENTERS = (
    (0.75, 0.0, 0.0), (-0.75, 0.0, 0.0),
    (0.0, 0.75, 0.0), (0.0, -0.75, 0.0),
    (0.0, 0.0, 0.75), (0.0, 0.0, -0.75),
)


def nearest_lattice_distance_1d(phase: float) -> float:
    return min(float(phase), 1.0 - float(phase))


def required_scale_tessa(family: str, phase: tuple[float, float, float], resolution: int) -> float:
    h = 1.0 / float(resolution)  # TESSA domain is half the IRIS [-1,1] domain.
    d = tuple(nearest_lattice_distance_1d(x) for x in phase)
    if family == "sphere":
        return 2.0 * math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) * h
    if family == "cylinder":
        return 2.0 * math.sqrt(d[0] ** 2 + d[1] ** 2) * h
    if family == "gap":
        return 2.0 * d[0] * h
    raise ValueError(f"UNKNOWN_FAMILY:{family}")


def fibonacci_sphere(count: int, *, center=(0.0, 0.0, 0.0), radius=1.0):
    pts = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        y = 1.0 - 2.0 * (i + 0.5) / float(count)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pts.append((center[0] + radius * x, center[1] + radius * y, center[2] + radius * z))
    return pts


def build_scene(admitted: tuple[bool, ...]):
    points = fibonacci_sphere(512, radius=0.35)
    groups = [0] * len(points)
    for i, ((_, scale), center, keep) in enumerate(zip(FEATURES, CENTERS, admitted), start=1):
        if not keep:
            continue
        feature = fibonacci_sphere(16, center=center, radius=max(float(scale) / 2.0, 0.006))
        points.extend(feature)
        groups.extend([i] * len(feature))
    return points, groups


def canonical_fps(points, budget: int):
    if len(points) <= budget:
        return list(range(len(points)))
    order = sorted(range(len(points)), key=lambda i: tuple(float(v) for v in points[i]))
    p = [points[i] for i in order]
    selected = [0]
    min_d = [float("inf")] * len(p)
    for _ in range(1, budget):
        anchor = p[selected[-1]]
        for i, q in enumerate(p):
            d = sum((anchor[k] - q[k]) ** 2 for k in range(3))
            if d < min_d[i]:
                min_d[i] = d
        for j in selected:
            min_d[j] = -1.0
        best = max(min_d)
        selected.append(next(i for i, d in enumerate(min_d) if d == best))
    return [order[i] for i in selected]


def bbox_diag(points) -> float:
    mins = [min(p[k] for p in points) for k in range(3)]
    maxs = [max(p[k] for p in points) for k in range(3)]
    return math.sqrt(sum((maxs[k] - mins[k]) ** 2 for k in range(3)))


def coverage(points, selected) -> float:
    diag = bbox_diag(points)
    worst = 0.0
    for p in points:
        nearest = min(math.sqrt(sum((p[k] - points[j][k]) ** 2 for k in range(3))) for j in selected)
        worst = max(worst, nearest)
    return worst / diag if diag > 0.0 else 0.0


def scenario_admission(resolution: int, scenario: int) -> tuple[bool, ...]:
    out = []
    for i, (family, scale) in enumerate(FEATURES):
        phase = tuple(((scenario + i * 3 + axis * 7) % SCENARIOS) / float(SCENARIOS) for axis in range(3))
        out.append(float(scale) + 1.0e-12 >= required_scale_tessa(family, phase, resolution))
    return tuple(out)


def run() -> dict:
    records = []
    for resolution in RESOLUTIONS:
        for scenario in range(SCENARIOS):
            admitted = scenario_admission(resolution, scenario)
            base_points, base_groups = build_scene(admitted)
            true_groups = set(range(1, len(FEATURES) + 1))
            admitted_groups = {i + 1 for i, keep in enumerate(admitted) if keep}
            for budget in BUDGETS:
                canonical_selected_sets = []
                permutation_rows = []
                for permutation in range(PERMUTATIONS):
                    rng = random.Random(731000 + resolution * 10000 + scenario * 100 + budget + permutation)
                    order = list(range(len(base_points)))
                    rng.shuffle(order)
                    points = [base_points[i] for i in order]
                    groups = [base_groups[i] for i in order]
                    selected = canonical_fps(points, budget)
                    selected_points = tuple(sorted(tuple(round(float(v), 12) for v in points[j]) for j in selected))
                    canonical_selected_sets.append(selected_points)
                    hit = {groups[j] for j in selected if groups[j] != 0}
                    conditional = (len(hit & admitted_groups) / len(admitted_groups)) if admitted_groups else 1.0
                    end_to_end = len(hit & true_groups) / len(true_groups)
                    permutation_rows.append((conditional, end_to_end, coverage(points, selected)))
                records.append({
                    "resolution": resolution,
                    "scenario": scenario,
                    "budget": budget,
                    "admitted_count": len(admitted_groups),
                    "extraction_admission_recall": len(admitted_groups) / len(true_groups),
                    "conditional_sampler_recall": min(x[0] for x in permutation_rows),
                    "end_to_end_recall": min(x[1] for x in permutation_rows),
                    "worst_cover_radius": max(x[2] for x in permutation_rows),
                    "permutation_invariant": len(set(canonical_selected_sets)) == 1,
                })

    summary = {}
    for resolution in RESOLUTIONS:
        summary[str(resolution)] = {}
        for budget in BUDGETS:
            rows = [r for r in records if r["resolution"] == resolution and r["budget"] == budget]
            summary[str(resolution)][str(budget)] = {
                "mean_extraction_admission_recall": sum(r["extraction_admission_recall"] for r in rows) / len(rows),
                "worst_extraction_admission_recall": min(r["extraction_admission_recall"] for r in rows),
                "mean_conditional_sampler_recall": sum(r["conditional_sampler_recall"] for r in rows) / len(rows),
                "worst_conditional_sampler_recall": min(r["conditional_sampler_recall"] for r in rows),
                "mean_end_to_end_recall": sum(r["end_to_end_recall"] for r in rows) / len(rows),
                "worst_end_to_end_recall": min(r["end_to_end_recall"] for r in rows),
                "worst_cover_radius": max(r["worst_cover_radius"] for r in rows),
                "permutation_invariant": all(r["permutation_invariant"] for r in rows),
            }

    sampler_valid = all(
        summary[str(r)]["64"]["permutation_invariant"]
        and summary[str(r)]["64"]["worst_conditional_sampler_recall"] >= 0.99
        for r in RESOLUTIONS
    )
    admission_means = [summary[str(r)]["64"]["mean_extraction_admission_recall"] for r in RESOLUTIONS]
    monotonic = all(b + 1e-12 >= a for a, b in zip(admission_means, admission_means[1:]))
    res24_misses = summary["24"]["64"]["worst_extraction_admission_recall"] < 0.999
    budget_gain = max(
        summary[str(r)]["128"]["mean_end_to_end_recall"] - summary[str(r)]["64"]["mean_end_to_end_recall"]
        for r in RESOLUTIONS
    )
    policy_selected = sampler_valid and res24_misses and monotonic and budget_gain <= 1e-12

    if not sampler_valid:
        status = "EXTRACTION_XYZ_FPS_INTERACTION_INVALID__SAMPLER_GATE_FAIL"
    elif policy_selected:
        status = "OBSERVABILITY_GATED_BOUNDED_RESOLUTION_ESCALATION_SELECTED_TINY_V1"
    else:
        status = "EXTRACTION_XYZ_FPS_INTERACTION_INCONCLUSIVE_TINY_V1"

    return {
        "schema": SCHEMA,
        "status": status,
        "features": [{"family": f, "scale_tessa": s} for f, s in FEATURES],
        "resolutions": list(RESOLUTIONS),
        "budgets": list(BUDGETS),
        "scenarios": SCENARIOS,
        "permutations": PERMUTATIONS,
        "summary": summary,
        "gates": {
            "sampler_valid": sampler_valid,
            "res24_misses_qualified_stress": res24_misses,
            "admission_monotonic_with_resolution": monotonic,
            "max_budget_128_minus_64_end_to_end_gain": budget_gain,
            "policy_form_selected": policy_selected,
        },
        "selection": {
            "sampler": "XYZ_FPS",
            "extraction_policy_form": "OBSERVABILITY_GATED_BOUNDED_RESOLUTION_ESCALATION" if policy_selected else None,
            "production_resolution_selected": False,
            "production_resolution_reason": "REAL_NORMALIZED_MORPHOLOGY_CENSUS_REQUIRED",
        },
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
    print(json.dumps({"status": result["status"], "summary": result["summary"], "gates": result["gates"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
