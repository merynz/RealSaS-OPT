#!/usr/bin/env python3
"""Deterministic proof that index-spaced surface witnesses have no component guarantee.

This mirrors the index formula in deterministic_surface_witness_v1 without torch.
It does not claim that a real FIT8 subject currently exhibits the constructed
ordering; it proves that the sampling contract itself does not guarantee small
component/thin-support coverage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCHEMA = "RealSaS.IRISTESSAV3SurfaceWitnessCoverageProbe.v1"


def selected_indices(vertex_count: int, witness_count: int) -> tuple[int, ...]:
    if vertex_count <= 0 or witness_count <= 0:
        raise ValueError("SURFACE_WITNESS_PROBE_COUNT_INVALID")
    out = []
    for i in range(witness_count):
        index = math.floor(i * (vertex_count / witness_count))
        out.append(min(index, vertex_count - 1))
    return tuple(out)


def find_contiguous_omitted_block(vertex_count: int, witness_count: int) -> tuple[int, ...]:
    chosen = set(selected_indices(vertex_count, witness_count))
    # Find the largest contiguous gap between selected indices.
    best: tuple[int, ...] = ()
    current = []
    for index in range(vertex_count):
        if index in chosen:
            if len(current) > len(best):
                best = tuple(current)
            current = []
        else:
            current.append(index)
    if len(current) > len(best):
        best = tuple(current)
    return best


def run(vertex_count: int = 4096, witness_count: int = 1024) -> dict:
    chosen = selected_indices(vertex_count, witness_count)
    omitted = find_contiguous_omitted_block(vertex_count, witness_count)
    if not omitted:
        raise RuntimeError("SURFACE_WITNESS_PROBE_NO_OMITTED_BLOCK")
    if set(omitted).intersection(chosen):
        raise AssertionError("SURFACE_WITNESS_PROBE_INTERNAL_DRIFT")

    return {
        "schema": SCHEMA,
        "status": "NO_COMPONENT_COVERAGE_GUARANTEE_PROVEN",
        "reference": {
            "vertex_count": vertex_count,
            "witness_count": witness_count,
            "unique_selected_count": len(set(chosen)),
            "selection_rule": "floor(i * vertex_count / witness_count)",
            "example_contiguous_omitted_vertex_block": list(omitted),
            "example_omitted_block_size": len(omitted),
        },
        "proof": {
            "observation": "a non-empty contiguous vertex subset can receive zero witnesses under the current index-spaced rule",
            "consequence": "the sampling contract alone cannot guarantee coverage of every connected component, thin appendage, attachment support, boundary stratum or low-area structure",
            "scope": "sampling guarantee only; no claim is made that current real subjects have this exact vertex ordering",
        },
        "required_before_causal_truth_training": [
            "measure real witness coverage by connected component, surface area, thinness, curvature, boundary cause, attachment/contact stratum and occlusion/grazing stratum",
            "if coverage gaps are material, preregister a deterministic coverage-aware witness policy before model outcomes are inspected",
        ],
        "training_executed": False,
        "model_inference_executed": False,
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vertex-count", type=int, default=4096)
    parser.add_argument("--witness-count", type=int, default=1024)
    args = parser.parse_args()
    report = run(args.vertex_count, args.witness_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "status": report["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
