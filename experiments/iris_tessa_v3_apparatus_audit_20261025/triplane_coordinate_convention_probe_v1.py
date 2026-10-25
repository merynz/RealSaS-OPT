#!/usr/bin/env python3
"""Measure endpoint-grid vs align_corners=False triplane coordinate semantics.

The current scene/TESSA positional grids use linspace(-1,1,N), while local H
sampling uses grid_sample(..., align_corners=False). This probe quantifies the
coordinate convention difference without claiming empirical harm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "RealSaS.IRISTESSAV3TriplaneCoordinateConventionProbe.v1"


def endpoint_coordinate(i: int, n: int) -> float:
    return -1.0 + 2.0 * i / float(n - 1)


def align_false_center_coordinate(i: int, n: int) -> float:
    return -1.0 + (2.0 * i + 1.0) / float(n)


def run(resolution: int = 64) -> dict:
    if resolution < 2:
        raise ValueError("TRIPLANE_COORD_PROBE_RESOLUTION_INVALID")
    deltas = [
        endpoint_coordinate(i, resolution) - align_false_center_coordinate(i, resolution)
        for i in range(resolution)
    ]
    max_abs = max(abs(x) for x in deltas)
    expected = 1.0 / resolution
    if abs(max_abs - expected) > 1.0e-12:
        raise AssertionError("TRIPLANE_COORD_PROBE_MATH_DRIFT")

    return {
        "schema": SCHEMA,
        "status": "COORDINATE_CONVENTION_DIFFERENCE_MEASURED__IMPACT_OPEN",
        "resolution": resolution,
        "endpoint_grid": "linspace(-1,1,N)",
        "sampler_grid": "grid_sample_align_corners_false_pixel_centers",
        "first_endpoint_coordinate": endpoint_coordinate(0, resolution),
        "first_sampler_center_coordinate": align_false_center_coordinate(0, resolution),
        "last_endpoint_coordinate": endpoint_coordinate(resolution - 1, resolution),
        "last_sampler_center_coordinate": align_false_center_coordinate(resolution - 1, resolution),
        "maximum_absolute_normalized_coordinate_difference": max_abs,
        "normalized_full_span_fraction": max_abs / 2.0,
        "interpretation": [
            "TESSA scene-position features label plane samples using endpoint coordinates while H sampling interprets normalized coordinates with align_corners=False semantics",
            "the two conventions are not numerically identical at finite resolution",
            "learned layers may compensate, so this probe does not classify the difference as a model failure",
        ],
        "required_before_readiness": [
            "add a coordinate-contract unit test that states the intended physical meaning of triplane cell centers",
            "measure whether a convention-matched sampler/position arm changes controlled spatial-access error before modifying production architecture",
        ],
        "training_executed": False,
        "model_inference_executed": False,
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.resolution)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "status": report["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
