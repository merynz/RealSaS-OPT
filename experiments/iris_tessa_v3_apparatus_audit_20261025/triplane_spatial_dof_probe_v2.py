#!/usr/bin/env python3
"""Rank-aware structural audit of the current 32->64 triplane upsampler.

V1 correctly observed non-overlapping 2x2 support, but its language could be read
as implying only one independent output value per 2x2 block. That is too coarse:
with ConvTranspose2d(in=512,out=96,kernel=2,stride=2), one 512-D scene-query
vector maps linearly to 4*96=384 output scalars. The architecture can therefore
be full row-rank locally. V2 seals that distinction without running a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "RealSaS.IRISTESSAV3TriplaneSpatialDOFProbe.v2"


def run(
    *,
    input_grid: int = 32,
    in_channels: int = 512,
    out_channels: int = 96,
    kernel: int = 2,
    stride: int = 2,
) -> dict:
    if min(input_grid, in_channels, out_channels, kernel, stride) <= 0:
        raise ValueError("TRIPLANE_DOF_V2_CONFIG_INVALID")
    if kernel != stride:
        raise ValueError("TRIPLANE_DOF_V2_ONLY_CURRENT_NONOVERLAP_CASE_SUPPORTED")
    output_grid = (input_grid - 1) * stride + kernel
    block_output_scalars = out_channels * kernel * kernel
    local_max_rank = min(in_channels, block_output_scalars)
    locally_surjective_possible = local_max_rank == block_output_scalars
    return {
        "schema": SCHEMA,
        "status": "LOCAL_CHANNEL_RANK_BOUND_SEALED__SPATIAL_ACCESS_IMPACT_OPEN",
        "supersedes_interpretation_of": "RealSaS.IRISTESSAV3TriplaneSpatialDOFProbe.v1",
        "reference": {
            "scene_query_grid": [input_grid, input_grid],
            "scene_query_channels": in_channels,
            "transpose_conv_kernel": [kernel, kernel],
            "transpose_conv_stride": [stride, stride],
            "triplane_carrier_grid": [output_grid, output_grid],
            "triplane_carrier_channels": out_channels,
            "scene_query_sites_per_plane": input_grid * input_grid,
            "carrier_sites_per_plane": output_grid * output_grid,
            "carrier_sites_generated_per_query_site": kernel * kernel,
            "local_input_scalar_dof_per_query_site": in_channels,
            "local_output_scalars_per_2x2_block": block_output_scalars,
            "architectural_max_local_linear_rank": local_max_rank,
            "full_row_rank_locally_possible": locally_surjective_possible,
        },
        "interpretation": [
            "non-overlapping spatial support does not imply one shared feature value across each 2x2 output block",
            "because 512 input channels exceed 4*96=384 output scalars, the learned transposed-convolution map can in principle produce four independently varying 96-D carrier vectors from one scene-query vector",
            "there are still only 32x32 attention query positions per plane before upsampling; whether one query vector can reliably retrieve and encode four subcell-specific source states is an information-routing question, not a simple carrier-DOF impossibility",
            "therefore 32x32 query density versus 64x64 carrier density must be tested with a controlled source-to-H spatial-access court rather than declared insufficient from support geometry alone",
        ],
        "required_before_readiness": [
            "run a controlled spatial-frequency/addressability court that compares query density and carrier density as separate treatment variables",
            "do not classify current 32->64 upsampling as a proven resolution failure solely from non-overlap support",
        ],
        "training_executed": False,
        "model_inference_executed": False,
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "status": report["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
