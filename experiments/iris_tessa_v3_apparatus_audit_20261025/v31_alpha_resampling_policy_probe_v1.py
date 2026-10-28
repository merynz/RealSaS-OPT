#!/usr/bin/env python3
"""Bounded numerical court for the V3.1 source-alpha 1024->518 policy.

This court selects only a frozen-feature preprocessing rule. It does not admit
observations, train a model, qualify V3.1, activate product.shape, or change any
product authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


SCHEMA = "RealSaS.V31AlphaResamplingPolicyCourt.v1"
INPUT_RESOLUTION = 1024
OUTPUT_RESOLUTION = 518
CANDIDATES = (
    "NEAREST",
    "BILINEAR_ANTIALIAS",
    "BICUBIC_ANTIALIAS",
    "AREA",
)
MAX_WORST_COVERAGE_ABS_ERROR = 2.5e-4
RANGE_EPSILON = 1.0e-7
SUPPORT_EPSILON = 1.0e-8


def _resize(alpha: torch.Tensor, candidate: str) -> torch.Tensor:
    if candidate == "NEAREST":
        return F.interpolate(alpha, size=(OUTPUT_RESOLUTION, OUTPUT_RESOLUTION), mode="nearest")
    if candidate == "BILINEAR_ANTIALIAS":
        return F.interpolate(
            alpha,
            size=(OUTPUT_RESOLUTION, OUTPUT_RESOLUTION),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    if candidate == "BICUBIC_ANTIALIAS":
        return F.interpolate(
            alpha,
            size=(OUTPUT_RESOLUTION, OUTPUT_RESOLUTION),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    if candidate == "AREA":
        return F.interpolate(alpha, size=(OUTPUT_RESOLUTION, OUTPUT_RESOLUTION), mode="area")
    raise ValueError("UNKNOWN_ALPHA_RESAMPLING_CANDIDATE")


def _vertical(width: int, offset: int) -> torch.Tensor:
    value = torch.zeros(1, 1, INPUT_RESOLUTION, INPUT_RESOLUTION, dtype=torch.float32)
    value[:, :, :, offset : offset + width] = 1.0
    return value


def _horizontal(width: int, offset: int) -> torch.Tensor:
    value = torch.zeros(1, 1, INPUT_RESOLUTION, INPUT_RESOLUTION, dtype=torch.float32)
    value[:, :, offset : offset + width, :] = 1.0
    return value


def _diagonal(width: int, offset: int) -> torch.Tensor:
    axis = torch.arange(INPUT_RESOLUTION)
    yy = axis[:, None]
    xx = axis[None, :]
    value = (torch.abs((xx - yy) - int(offset)) < int(width)).to(torch.float32)
    return value[None, None]


def _disk(radius: int, center_x: int, center_y: int) -> torch.Tensor:
    axis = torch.arange(INPUT_RESOLUTION)
    yy = axis[:, None]
    xx = axis[None, :]
    value = (
        (xx - int(center_x)).square() + (yy - int(center_y)).square()
        <= int(radius) * int(radius)
    ).to(torch.float32)
    return value[None, None]


def _scenarios():
    phases = (0, 1, 2, 3, 17, 31, 127, 255, 511, 700, 1000)
    for width in (1, 2, 3, 4, 8):
        for offset in phases:
            if offset + width <= INPUT_RESOLUTION:
                yield f"vertical_w{width}_p{offset}", f"vertical_w{width}", _vertical(width, offset)
                yield f"horizontal_w{width}_p{offset}", f"horizontal_w{width}", _horizontal(width, offset)
    for width in (1, 2, 3):
        for offset in (-400, -127, -31, -3, 0, 3, 31, 127, 400):
            yield f"diagonal_w{width}_p{offset}", f"diagonal_w{width}", _diagonal(width, offset)
    centers = ((100, 100), (101, 100), (255, 257), (511, 511), (700, 333), (900, 900))
    for radius in (1, 2, 3, 4, 8, 16):
        for center_x, center_y in centers:
            yield (
                f"disk_r{radius}_{center_x}_{center_y}",
                f"disk_r{radius}",
                _disk(radius, center_x, center_y),
            )


def run() -> dict:
    torch.set_num_threads(1)
    rows = []
    grouped_coverage: dict[str, dict[str, list[float]]] = {
        candidate: {} for candidate in CANDIDATES
    }
    scenario_count = 0

    for name, family, source in _scenarios():
        scenario_count += 1
        source_coverage = float(source.mean())
        if not (0.0 <= source_coverage <= 1.0) or float(source.max()) <= 0.0:
            raise RuntimeError("ALPHA_COURT_SOURCE_INVALID")
        for candidate in CANDIDATES:
            target = _resize(source, candidate)
            target_coverage = float(target.mean())
            target_min = float(target.min())
            target_max = float(target.max())
            row = {
                "scenario": name,
                "family": family,
                "candidate": candidate,
                "source_coverage": source_coverage,
                "target_coverage": target_coverage,
                "coverage_abs_error": abs(target_coverage - source_coverage),
                "target_min": target_min,
                "target_max": target_max,
                "support_survived": bool(target_max > SUPPORT_EPSILON),
            }
            rows.append(row)
            grouped_coverage[candidate].setdefault(family, []).append(target_coverage)

    summary = {}
    for candidate in CANDIDATES:
        current = [row for row in rows if row["candidate"] == candidate]
        coverage_errors = [float(row["coverage_abs_error"]) for row in current]
        minimum = min(float(row["target_min"]) for row in current)
        maximum = max(float(row["target_max"]) for row in current)
        support_recall = sum(bool(row["support_survived"]) for row in current) / len(current)
        phase_span = max(
            max(values) - min(values)
            for values in grouped_coverage[candidate].values()
        )
        bounded_without_clamp = minimum >= -RANGE_EPSILON and maximum <= 1.0 + RANGE_EPSILON
        worst_coverage = max(coverage_errors)
        eligible = (
            bounded_without_clamp
            and support_recall == 1.0
            and worst_coverage <= MAX_WORST_COVERAGE_ABS_ERROR
        )
        summary[candidate] = {
            "worst_coverage_abs_error": worst_coverage,
            "mean_coverage_abs_error": sum(coverage_errors) / len(coverage_errors),
            "minimum_output_alpha": minimum,
            "maximum_output_alpha": maximum,
            "support_recall": support_recall,
            "worst_family_phase_coverage_span": phase_span,
            "bounded_without_clamp": bounded_without_clamp,
            "eligible": eligible,
        }

    eligible = [candidate for candidate in CANDIDATES if summary[candidate]["eligible"]]
    eligible.sort(key=lambda name: (summary[name]["worst_coverage_abs_error"], name))
    if len(eligible) == 1:
        selected = eligible[0]
        status = "V31_ALPHA_RESAMPLING_POLICY_SELECTED_BOUNDED_V1"
    elif len(eligible) > 1:
        best, runner_up = eligible[:2]
        best_error = summary[best]["worst_coverage_abs_error"]
        runner_error = summary[runner_up]["worst_coverage_abs_error"]
        if best_error <= runner_error / 4.0:
            selected = best
            status = "V31_ALPHA_RESAMPLING_POLICY_SELECTED_BOUNDED_V1"
        else:
            selected = None
            status = "V31_ALPHA_RESAMPLING_POLICY_INCONCLUSIVE_BOUNDED_V1"
    else:
        selected = None
        status = "V31_ALPHA_RESAMPLING_POLICY_NO_ELIGIBLE_CANDIDATE_V1"

    return {
        "schema": SCHEMA,
        "status": status,
        "selected_policy": selected,
        "input_resolution": INPUT_RESOLUTION,
        "output_resolution": OUTPUT_RESOLUTION,
        "scenario_count": scenario_count,
        "candidates": list(CANDIDATES),
        "gates": {
            "maximum_worst_coverage_abs_error": MAX_WORST_COVERAGE_ABS_ERROR,
            "range_epsilon": RANGE_EPSILON,
            "support_epsilon": SUPPORT_EPSILON,
            "selection_margin_if_multiple_eligible": "BEST_WORST_COVERAGE_ERROR_AT_LEAST_4X_LOWER",
            "clamping_allowed": False,
        },
        "summary": summary,
        "records": rows,
        "claims": {
            "observation_admission_minted": False,
            "training_authorized": False,
            "model_release_minted": False,
            "model_promotion_performed": False,
            "product_shape_activated": False,
            "shape_proposal_minted": False,
            "mesh_authority_changed": False,
            "product_revision_minted": False,
            "fit8_claimed": False,
            "unseen_claimed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_policy": result["selected_policy"],
                "scenario_count": result["scenario_count"],
                "summary": result["summary"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
