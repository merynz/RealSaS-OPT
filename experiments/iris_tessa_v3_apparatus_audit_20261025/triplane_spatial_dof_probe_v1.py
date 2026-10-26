#!/usr/bin/env python3
"""Measure spatial-site structure of the current 32->64 triplane upsampler.

ConvTranspose2d(kernel=2,stride=2,padding=0) has non-overlapping spatial support:
each input scene-query site maps to one 2x2 output block. This probe distinguishes
carrier raster size from the count of independently positioned scene-query sites.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "RealSaS.IRISTESSAV3TriplaneSpatialDOFProbe.v1"


def contributors(output_index: int, *, input_size: int, kernel: int, stride: int) -> tuple[int, ...]:
    rows = []
    for i in range(input_size):
        start = i * stride
        if start <= output_index < start + kernel:
            rows.append(i)
    return tuple(rows)


def run(input_size: int = 32, kernel: int = 2, stride: int = 2) -> dict:
    if min(input_size, kernel, stride) <= 0:
        raise ValueError("TRIPLANE_DOF_PROBE_CONFIG_INVALID")
    output_size = (input_size - 1) * stride + kernel
    support_counts = [len(contributors(o, input_size=input_size, kernel=kernel, stride=stride)) for o in range(output_size)]
    if kernel == stride and any(count != 1 for count in support_counts):
        raise AssertionError("TRIPLANE_DOF_PROBE_NONOVERLAP_MATH_DRIFT")

    independent_sites_per_plane = input_size * input_size
    carrier_sites_per_plane = output_size * output_size
    return {
        "schema": SCHEMA,
        "status": "CARRIER_RESOLUTION_DISTINGUISHED_FROM_SCENE_QUERY_SITE_COUNT",
        "reference": {
            "scene_query_grid": [input_size, input_size],
            "transpose_conv_kernel": kernel,
            "transpose_conv_stride": stride,
            "triplane_carrier_grid": [output_size, output_size],
            "independent_scene_query_sites_per_plane": independent_sites_per_plane,
            "carrier_samples_per_plane": carrier_sites_per_plane,
            "carrier_samples_per_scene_query_site": carrier_sites_per_plane / float(independent_sites_per_plane),
            "minimum_input_sites_contributing_to_one_output_sample": min(support_counts),
            "maximum_input_sites_contributing_to_one_output_sample": max(support_counts),
            "upsample_spatial_support_overlaps_between_neighbor_input_sites": max(support_counts) > 1,
        },
        "interpretation": [
            "the 64x64 carrier should not be described as 64x64 independently learned scene-query positions",
            "each 32x32 query site generates its own non-overlapping 2x2 carrier block through a shared learned channel transform",
            "global information can already be present in each input site because the scene-query Transformer attended to all views; this probe does not define an absolute recoverable-detail limit",
            "bilinear H sampling can interpolate carrier values, but interpolation does not create additional independently positioned upstream scene-query sites",
        ],
        "required_before_readiness": [
            "spatial-access experiments must report both scene-query grid and carrier grid instead of calling either one simply the model resolution",
            "if spatial capacity is changed, separate treatment of query-site density from carrier upsampling/interpolation so the causal variable is known",
        ],
        "training_executed": False,
        "model_inference_executed": False,
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--kernel", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.input_size, args.kernel, args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "status": report["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
