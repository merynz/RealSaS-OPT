#!/usr/bin/env python3
"""Deterministic audit of TESSA topology scale policy vs decode-token budget.

No model is loaded. The probe uses the canonical adjacent-face serialization
bounds and the current product decode cap to identify the part of the declared
mesh scale that is not automatically covered by token budget alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "RealSaS.TESSAScaleBudgetProbe.v1"
REFERENCE_MAX_FACES = 32768
REFERENCE_MAX_NEW_TOKENS = 131072
REFERENCE_COORDINATE_BINS = 1024


def worst_case_tokens(face_count: int) -> int:
    # Existing tokenizer contract: specials + every face restarts => 6 + 10F.
    return 6 + 10 * int(face_count)


def one_component_tokens(face_count: int, restart_count: int) -> int:
    """Exact token count for one component under current adjacent serialization.

    First/restarted face costs FACE_BREAK+9 coords =10. Continuation face costs
    3 coordinates. Specials are BOS, COMPONENT_BEGIN, CHART_BEGIN, CHART_END,
    COMPONENT_END, EOS => 6 total for a one-component asset.
    Thus 6 + 3F + 7R where R is number of strip starts/restarts (>=1 for F>0).
    """
    if face_count < 0 or restart_count < 0 or restart_count > face_count:
        raise ValueError("TESSA_SCALE_PROBE_INPUT_INVALID")
    if face_count and restart_count < 1:
        raise ValueError("TESSA_SCALE_PROBE_NONEMPTY_REQUIRES_RESTART")
    return 6 + 3 * int(face_count) + 7 * int(restart_count)


def max_restarts_within_budget(face_count: int, max_total_tokens: int) -> int:
    # Product generation begins with BOS and maximum_new_tokens counts generated
    # tokens after it, so total representable sequence length is cap + 1.
    total_budget = int(max_total_tokens) + 1
    base = 6 + 3 * int(face_count)
    if base > total_budget:
        return -1
    return min(int(face_count), (total_budget - base) // 7)


def run(
    max_faces: int = REFERENCE_MAX_FACES,
    maximum_new_tokens: int = REFERENCE_MAX_NEW_TOKENS,
    coordinate_bins: int = REFERENCE_COORDINATE_BINS,
) -> dict:
    if min(max_faces, maximum_new_tokens, coordinate_bins) <= 0:
        raise ValueError("TESSA_SCALE_PROBE_CONFIG_INVALID")

    total_budget = maximum_new_tokens + 1
    worst = worst_case_tokens(max_faces)
    allowed_restarts = max_restarts_within_budget(max_faces, maximum_new_tokens)
    if allowed_restarts < 0:
        restart_fraction = None
    else:
        restart_fraction = allowed_restarts / float(max_faces)

    # Minimum face count whose worst-case serialization exceeds product budget.
    first_worst_case_over = ((total_budget - 6) // 10) + 1

    return {
        "schema": SCHEMA,
        "status": "DECLARED_FACE_SCALE_NOT_UNCONDITIONALLY_COVERED_BY_DECODE_BUDGET",
        "reference": {
            "max_faces": max_faces,
            "maximum_new_tokens": maximum_new_tokens,
            "maximum_total_sequence_tokens_including_initial_bos": total_budget,
            "coordinate_bins": coordinate_bins,
            "coordinate_bin_width_normalized_axis": 1.0 / coordinate_bins,
            "worst_case_tokens_at_max_faces": worst,
            "first_face_count_whose_worst_case_serialization_exceeds_budget": first_worst_case_over,
            "max_restarts_at_max_faces_within_budget": allowed_restarts,
            "max_restart_fraction_at_max_faces_within_budget": restart_fraction,
        },
        "interpretation": [
            "max_faces is a mesh legality/scale policy, not a proof that every legal mesh serialization fits the product autoregressive decode cap",
            "at 32768 faces, the worst-case adjacent serialization exceeds 131072 generated-token budget",
            "for one component at max_faces, fitting the current cap requires restart_count <= 4680 (about 14.28 percent of faces)",
            "typical well-connected meshes may serialize much more compactly; no real-corpus failure is claimed by this arithmetic alone",
            "1024 coordinate bins imply approximately 9.765625e-4 normalized-axis quantization width; current teacher code separately fail-closes within-component quantized vertex collisions",
        ],
        "required_before_readiness": [
            "measure teacher topology token-length distribution on Knight plus intended FIT8/Truth corpus",
            "measure restart_count/face_count and component-count distributions",
            "measure quantized-vertex collision rejection frequency and morphology correlation",
            "report fraction of otherwise legal teacher meshes that exceed current product decode cap",
            "only then decide whether tokenization, scale policy, decode cap or topology representation needs revision",
        ],
        "training_executed": False,
        "model_inference_executed": False,
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-faces", type=int, default=REFERENCE_MAX_FACES)
    parser.add_argument("--maximum-new-tokens", type=int, default=REFERENCE_MAX_NEW_TOKENS)
    parser.add_argument("--coordinate-bins", type=int, default=REFERENCE_COORDINATE_BINS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.max_faces, args.maximum_new_tokens, args.coordinate_bins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "status": report["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
