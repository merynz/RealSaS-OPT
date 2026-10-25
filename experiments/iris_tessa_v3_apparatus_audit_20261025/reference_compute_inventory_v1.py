#!/usr/bin/env python3
"""Source-derived compute inventory for reference IRIS/TESSA V3.

No model is instantiated and no GPU is used. The parameter arithmetic mirrors
current module definitions so future execution planning does not accidentally
turn hardware limits into a scientific treatment variable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "RealSaS.IRISTESSAV3ReferenceComputeInventory.v1"


def mha_same_dim(d: int) -> int:
    # in_proj_weight 3d*d + in_proj_bias 3d + out_proj d*d + bias d
    return 4 * d * d + 4 * d


def mha_cross_kv_dim(d: int, kv: int) -> int:
    # separate q/k/v weights, shared 3d bias, out projection + bias
    return d * d + d * kv + d * kv + 3 * d + d * d + d


def encoder_layer(d: int, ff: int) -> int:
    attention = mha_same_dim(d)
    mlp = d * ff + ff + ff * d + d
    norms = 4 * d  # two LayerNorms, each weight+bias
    return attention + mlp + norms


def typed_stream(input_dim: int, token_count: int, *, d: int, source_width: int, refinement_layers: int, ff: int) -> int:
    source_projection = input_dim * source_width + source_width + 2 * source_width
    queries = token_count * d
    type_embedding = d
    query_norm = 2 * d
    source_norm = 2 * source_width
    cross = mha_cross_kv_dim(d, source_width)
    refinement = refinement_layers * encoder_layer(d, ff)
    out_norm = 2 * d
    return source_projection + queries + type_embedding + query_norm + source_norm + cross + refinement + out_norm


def run() -> dict:
    # Reference TESSA V3 config/source dimensions.
    d = 1024
    heads = 16
    layers = 24
    ff = 4096
    source_width = 512
    refinement_layers = 4
    scene_input = 101       # plane_channels 96 + xy2 + plane-id3
    surface_input = 398     # xyz3 + normal3 + H384 + mechanical6 + contact value/valid2
    presentation_input = 45
    scene_tokens = 384
    surface_tokens = 384
    presentation_tokens = 256
    vocab = 1032            # 1024 coordinate bins + 8 special
    position_period = 4096

    memory_scene = typed_stream(scene_input, scene_tokens, d=d, source_width=source_width, refinement_layers=refinement_layers, ff=ff)
    memory_surface = typed_stream(surface_input, surface_tokens, d=d, source_width=source_width, refinement_layers=refinement_layers, ff=ff)
    memory_presentation = typed_stream(presentation_input, presentation_tokens, d=d, source_width=source_width, refinement_layers=refinement_layers, ff=ff)
    fusion_norm = 2 * d
    memory_total = memory_scene + memory_surface + memory_presentation + fusion_norm

    # TESSADecoderBlockV2: custom bias-free qkv/out self-attn + standard cross MHA + MLP + 3 norms.
    self_attn = 4 * d * d
    cross_attn = mha_same_dim(d)
    mlp = d * ff + ff + ff * d + d
    three_norms = 6 * d
    decoder_block = self_attn + cross_attn + mlp + three_norms
    token_embedding = vocab * d
    position_embedding = position_period * d
    final_norm = 2 * d
    lm_head = d * vocab  # bias=False
    decoder_total = layers * decoder_block + token_embedding + position_embedding + final_norm + lm_head
    tessa_total = memory_total + decoder_total

    fp32_param_bytes = tessa_total * 4
    fp32_param_grad_adam_bytes = tessa_total * 4 * 4

    # IRIS scene attention geometry (not a memory estimate; optimized kernels may avoid score materialization).
    dino_tokens = 8 * 37 * 37
    scene_queries = 3 * 32 * 32
    iris_heads = 8
    iris_layers = 6
    cross_pairs_per_layer = scene_queries * dino_tokens
    query_self_pairs_per_layer = scene_queries * scene_queries

    return {
        "schema": SCHEMA,
        "status": "REFERENCE_COMPUTE_GEOMETRY_INVENTORIED__NO_EXECUTION_AUTHORIZED",
        "tessa_v3_reference": {
            "parameter_count": tessa_total,
            "memory_parameter_count": memory_total,
            "decoder_parameter_count": decoder_total,
            "decoder_block_parameter_count": decoder_block,
            "fp32_parameter_bytes": fp32_param_bytes,
            "fp32_parameter_gib": fp32_param_bytes / (1024.0 ** 3),
            "fp32_parameter_plus_gradient_plus_two_adam_moments_bytes": fp32_param_grad_adam_bytes,
            "fp32_parameter_plus_gradient_plus_two_adam_moments_gib": fp32_param_grad_adam_bytes / (1024.0 ** 3),
            "note": "optimizer/training-state figure excludes activations, attention workspace, dataloader tensors, checkpointing overhead and allocator fragmentation; actual policy may change storage precision and therefore must be measured",
        },
        "iris_v3_scene_attention_geometry": {
            "dino_source_tokens": dino_tokens,
            "scene_query_tokens": scene_queries,
            "heads": iris_heads,
            "decoder_layers": iris_layers,
            "cross_query_key_pairs_per_head_per_layer": cross_pairs_per_layer,
            "cross_score_elements_all_heads_per_layer_if_materialized": cross_pairs_per_layer * iris_heads,
            "scene_query_self_pairs_per_head_per_layer": query_self_pairs_per_layer,
            "scene_query_self_score_elements_all_heads_per_layer_if_materialized": query_self_pairs_per_layer * iris_heads,
            "note": "pair counts describe attention geometry, not guaranteed allocated memory; optimized attention kernels can avoid materializing full score matrices",
        },
        "execution_decision": {
            "current_static_and_tiny_cpu_audits_require_a100": False,
            "reference_tessa_training_should_be_hardware_preregistered": True,
            "gtx_1660ti_6gb_should_not_be_assumed_capable_of_reference_tessa_training": True,
            "a100_run_started": False,
        },
        "gpu_required_by_this_probe": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "status": report["status"], "tessa_parameter_count": report["tessa_v3_reference"]["parameter_count"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
