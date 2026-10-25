#!/usr/bin/env python3
"""Execute EXTRACTION_TOPOLOGY_OBSERVABILITY_PREREG_V1 deterministically."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RESOLUTIONS = (24, 32, 48, 64)
PHASE_COUNT = 33


def phases() -> list[float]:
    # Periodic offsets in [0,1); endpoint 1 is intentionally excluded.
    return [i / PHASE_COUNT for i in range(PHASE_COUNT)]


def nearest_lattice_distance_1d(phase: float) -> float:
    # Distance in units of grid spacing to the nearest integer lattice site.
    return min(phase, 1.0 - phase)


def percentile_threshold(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("EMPTY_THRESHOLD_DISTRIBUTION")
    ordered = sorted(values)
    # Smallest threshold for which >= probability of finite phase samples pass.
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def family_thresholds_h() -> dict[str, dict[str, float]]:
    p = phases()

    # Required scale per phase, expressed in units of h.
    # Sphere: diameter = 2 * distance to nearest 3-D lattice vertex.
    sphere = []
    for x in p:
        dx = nearest_lattice_distance_1d(x)
        for y in p:
            dy = nearest_lattice_distance_1d(y)
            for z in p:
                dz = nearest_lattice_distance_1d(z)
                sphere.append(2.0 * math.sqrt(dx * dx + dy * dy + dz * dz))

    # Infinite axis-aligned cylinder: diameter = 2 * nearest 2-D lattice distance.
    cylinder = []
    for x in p:
        dx = nearest_lattice_distance_1d(x)
        for y in p:
            dy = nearest_lattice_distance_1d(y)
            cylinder.append(2.0 * math.sqrt(dx * dx + dy * dy))

    # Two parallel sheets bounding a negative slab: gap must contain a lattice
    # sample. Required gap = 2 * nearest 1-D lattice distance.
    gap = [2.0 * nearest_lattice_distance_1d(x) for x in p]

    def summarize(values: list[float], analytic_guaranteed: float) -> dict[str, float]:
        return {
            "p50_h": percentile_threshold(values, 0.50),
            "p95_h": percentile_threshold(values, 0.95),
            "finite_phase_guaranteed_h": max(values),
            "analytic_phase_independent_guaranteed_h": analytic_guaranteed,
            "phase_samples": len(values),
        }

    return {
        "sphere_diameter": summarize(sphere, math.sqrt(3.0)),
        "cylinder_diameter": summarize(cylinder, math.sqrt(2.0)),
        "parallel_sheet_gap": summarize(gap, 1.0),
    }


def run() -> dict:
    normalized = family_thresholds_h()
    table: dict[str, dict[str, dict[str, float]]] = {}
    for resolution in RESOLUTIONS:
        h_iris = 2.0 / resolution
        table[str(resolution)] = {}
        for family, row in normalized.items():
            table[str(resolution)][family] = {
                **row,
                "grid_spacing_iris": h_iris,
                "grid_spacing_tessa": 0.5 * h_iris,
                "p50_iris": row["p50_h"] * h_iris,
                "p95_iris": row["p95_h"] * h_iris,
                "finite_phase_guaranteed_iris": row["finite_phase_guaranteed_h"] * h_iris,
                "analytic_guaranteed_iris": row["analytic_phase_independent_guaranteed_h"] * h_iris,
                "p50_tessa": 0.5 * row["p50_h"] * h_iris,
                "p95_tessa": 0.5 * row["p95_h"] * h_iris,
                "finite_phase_guaranteed_tessa": 0.5 * row["finite_phase_guaranteed_h"] * h_iris,
                "analytic_guaranteed_tessa": 0.5 * row["analytic_phase_independent_guaranteed_h"] * h_iris,
            }

    sanity = {
        "sphere_finite_le_analytic": normalized["sphere_diameter"]["finite_phase_guaranteed_h"] <= math.sqrt(3.0) + 1e-12,
        "cylinder_finite_le_analytic": normalized["cylinder_diameter"]["finite_phase_guaranteed_h"] <= math.sqrt(2.0) + 1e-12,
        "gap_finite_le_analytic": normalized["parallel_sheet_gap"]["finite_phase_guaranteed_h"] <= 1.0 + 1e-12,
    }
    if not all(sanity.values()):
        status = "EXTRACTION_OBSERVABILITY_COURT_INVALID"
    else:
        status = "EXTRACTION_OBSERVABILITY_TABLE_ESTABLISHED_V1"

    return {
        "schema": "RealSaS.ExtractionTopologyObservabilityCourt.v1",
        "status": status,
        "resolutions": list(RESOLUTIONS),
        "phase_count_per_axis": PHASE_COUNT,
        "normalized_thresholds": normalized,
        "table": table,
        "sanity": sanity,
        "selection": {
            "production_resolution_selected": False,
            "reason": "REAL_MORPHOLOGY_SCALE_CENSUS_REQUIRED",
        },
        "authority": {
            "product_authority_changed": False,
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
    print(json.dumps({"status": result["status"], "table": result["table"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
