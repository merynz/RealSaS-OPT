#!/usr/bin/env python3
"""Prove an alpha-aware visibility ambiguity under the current V3 observation/ray contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "RealSaS.IRISTESSAV3RayVisibilityAlphaIdentifiabilityProbe.v1"


def composite_u8(rgb: int, alpha: int) -> int:
    # Mirrors round(RGB*a + white*(1-a)) at uint8 precision for one gray channel.
    return round((rgb * alpha + 255 * (255 - alpha)) / 255.0)


def contribution(alpha_front: float, alpha_back: float) -> tuple[float, float]:
    # Standard front-to-back alpha compositing contribution weights.
    w_front = alpha_front
    w_back = (1.0 - alpha_front) * alpha_back
    return w_front, w_back


def run() -> dict:
    # Reuse an exact collision class: two distinct source RGBA states map to the
    # same white-composited RGB before DINO.
    a = {"rgb": 0, "alpha": 159}
    b = {"rgb": 73, "alpha": 223}
    ca, cb = composite_u8(a["rgb"], a["alpha"]), composite_u8(b["rgb"], b["alpha"])
    if ca != cb:
        raise RuntimeError("RAY_ALPHA_PROBE_COLLISION_DRIFT")

    # Put the ambiguous hit in front of one fixed opaque-ish back hit. Geometry,
    # camera, depth, rank and downstream H inputs are identical across worlds if
    # current DINO observations collide; true compositing contributions differ.
    back_alpha = 0.8
    wa = contribution(a["alpha"] / 255.0, back_alpha)
    wb = contribution(b["alpha"] / 255.0, back_alpha)
    if wa == wb:
        raise RuntimeError("RAY_ALPHA_PROBE_VISIBILITY_NOT_DIFFERENT")

    return {
        "schema": SCHEMA,
        "status": "ALPHA_AWARE_RAY_VISIBILITY_NONIDENTIFIABLE_UNDER_CURRENT_OBSERVATION_CONTRACT",
        "collision": {
            "world_a_front_rgba_gray": [a["rgb"], a["alpha"]],
            "world_b_front_rgba_gray": [b["rgb"], b["alpha"]],
            "white_composited_rgb_u8_both": ca,
            "fixed_back_alpha": back_alpha,
        },
        "true_compositing_contribution": {
            "world_a": {"front": wa[0], "back": wa[1]},
            "world_b": {"front": wb[0], "back": wb[1]},
            "back_contribution_absolute_delta": abs(wa[1] - wb[1]),
        },
        "current_ray_head_explicit_inputs": ["H", "xyz", "depth", "normalized_rank", "camera"],
        "current_ray_head_explicit_alpha_or_transmittance": False,
        "proof": [
            "distinct source RGBA states can map to identical white-composited RGB before the frozen foundation",
            "with identical camera/geometry this yields identical deterministic downstream shared-scene observation for that collision class",
            "RayStackVisibilityHeadV3 receives no explicit alpha/transmittance channel that could break the collision",
            "the two worlds nevertheless have different front-to-back compositing contribution weights",
            "therefore alpha-aware global visibility/compositing contribution is not identifiable from the current ray-head input contract for this equivalence class",
        ],
        "required_before_readiness": [
            "define whether observed source alpha/support is deterministic evidence versus a shared-scene input channel",
            "define ray truth as complete compositing-aware contribution/transmittance with validity rather than pairwise visibility",
            "if learned visibility must infer alpha-dependent hidden state, supply an observation contract capable of distinguishing the required alpha/support state",
            "do not wire the existing RayStackVisibilityHeadV3 into training as if the alpha-aware target were already identifiable",
        ],
        "claims": {
            "ray_stack_transformer_architecture_falsified": False,
            "deterministic_raster_authority_selected": False,
            "truth139_training_authorized": False,
            "product_authority_changed": False,
        },
        "training_executed": False,
        "model_inference_executed": False,
        "gpu_required": False,
    }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); r=run(); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({"schema":r["schema"],"status":r["status"],"out":str(a.out)},sort_keys=True))

if __name__=="__main__": main()
