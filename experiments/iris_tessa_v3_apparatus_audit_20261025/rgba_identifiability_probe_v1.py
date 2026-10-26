#!/usr/bin/env python3
"""Exact uint8 proof that the current RGBA->white-RGB frontend is non-injective.

No model is loaded.  The probe mirrors the current uint8 white compositing
formula channel-wise and finds authored RGBA pairs with different alpha/source
support that become the same RGB observation before frozen DINO.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "RealSaS.IRISTESSAV3RGBAIdentifiabilityProbe.v1"


def composite_channel_u8(source: int, alpha: int) -> int:
    if not (0 <= source <= 255 and 0 <= alpha <= 255):
        raise ValueError("RGBA_PROBE_UINT8_RANGE")
    # Exact algebra of current float01 -> white composite -> torch.round(*255)
    # for uint8 inputs. Python round uses ties-to-even like torch.round.
    return int(round((source * alpha) / 255.0 + (255 - alpha)))


def find_collision(*, minimum_alpha_delta: int = 64) -> dict:
    by_output: dict[int, list[tuple[int, int]]] = {}
    for alpha in range(1, 256):
        for source in range(256):
            output = composite_channel_u8(source, alpha)
            by_output.setdefault(output, []).append((source, alpha))

    # Prefer a non-extreme visible output so the witness is not a nearly-white
    # rounding curiosity.
    for output in range(96, 193):
        rows = by_output.get(output, [])
        for i, (source_a, alpha_a) in enumerate(rows):
            for source_b, alpha_b in rows[i + 1 :]:
                if abs(alpha_a - alpha_b) < minimum_alpha_delta:
                    continue
                if source_a == source_b:
                    continue
                rgba_a = [source_a, source_a, source_a, alpha_a]
                rgba_b = [source_b, source_b, source_b, alpha_b]
                rgb_a = [composite_channel_u8(source_a, alpha_a)] * 3
                rgb_b = [composite_channel_u8(source_b, alpha_b)] * 3
                if rgb_a != rgb_b:
                    raise AssertionError("RGBA_PROBE_COLLISION_INTERNAL_DRIFT")
                return {
                    "rgba_a": rgba_a,
                    "rgba_b": rgba_b,
                    "composited_rgb_a": rgb_a,
                    "composited_rgb_b": rgb_b,
                    "alpha_delta": abs(alpha_a - alpha_b),
                }
    raise RuntimeError("RGBA_PROBE_NO_NONTRIVIAL_COLLISION_FOUND")


def run() -> dict:
    collision = find_collision()
    if collision["rgba_a"] == collision["rgba_b"]:
        raise AssertionError("RGBA_PROBE_DISTINCT_SOURCE_REQUIRED")
    if collision["composited_rgb_a"] != collision["composited_rgb_b"]:
        raise AssertionError("RGBA_PROBE_EQUAL_RGB_REQUIRED")
    if collision["rgba_a"][3] == collision["rgba_b"][3]:
        raise AssertionError("RGBA_PROBE_ALPHA_DIFFERENCE_REQUIRED")

    return {
        "schema": SCHEMA,
        "status": "PROVEN_NON_INJECTIVE_BEFORE_DINO",
        "witness": collision,
        "proof": {
            "frontend_map": "RGBA_UINT8 -> WHITE_COMPOSITED_RGB_UINT8 -> FROZEN_DINOV2_S",
            "observation": "two distinct RGBA observations with different alpha map to identical RGB before DINO",
            "consequence": "for identical cameras and otherwise identical canvases, all deterministic downstream functions of the current DINO RGB tokens plus cameras receive identical observation for the witness pair",
            "identifiability_limit": "a target whose distinction depends on the lost alpha/source-support state cannot be identified from current shared-scene H alone for this equivalence class",
        },
        "does_not_claim": [
            "that RGB scene information is generally insufficient",
            "that V3 is globally invalid",
            "that an RGBA-native repair is already selected",
            "that alpha/source-support must be learned rather than supplied deterministically",
            "that the shared-scene scientific hypothesis is false",
        ],
        "required_decision_before_causal_truth_training": [
            "either narrow the scientific hypothesis so alpha/source-support is explicitly an external deterministic/presentation authority rather than shared-scene evidence",
            "or preregister a matched shared-scene input arm that preserves independent RGBA/native support and test whether it is required",
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
