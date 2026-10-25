#!/usr/bin/env python3
"""Deterministic audit of TESSA V2/V3 chunk-anchored causal-window semantics.

No torch/model execution is required. The probe evaluates the indexing rule used
by WindowedCausalSelfAttentionV2: key_start=max(0, chunk_end-window).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "RealSaS.TESSAWindowSemanticsProbe.v1"
REFERENCE_WINDOW = 4096
REFERENCE_CHUNK = 256


def direct_history_count(query_position: int, chunk_start: int, chunk_end: int, window: int) -> int:
    if not (chunk_start <= query_position < chunk_end):
        raise ValueError("TESSA_WINDOW_PROBE_QUERY_OUTSIDE_CHUNK")
    key_start = max(0, chunk_end - window)
    return query_position - key_start + 1


def run(window: int = REFERENCE_WINDOW, chunk: int = REFERENCE_CHUNK) -> dict:
    if window <= 0 or chunk <= 0 or chunk > window:
        raise ValueError("TESSA_WINDOW_PROBE_CONFIG_INVALID")

    # Examine a full chunk entirely beyond the warmup region.
    chunk_end = window + chunk
    chunk_start = chunk_end - chunk
    counts = [
        direct_history_count(pos, chunk_start, chunk_end, window)
        for pos in range(chunk_start, chunk_end)
    ]
    expected_min = window - chunk + 1
    if min(counts) != expected_min or max(counts) != window:
        raise AssertionError("TESSA_WINDOW_PROBE_REFERENCE_MATH_DRIFT")

    return {
        "schema": SCHEMA,
        "status": "REFERENCE_WINDOW_SEMANTICS_MEASURED",
        "reference": {
            "local_attention_window": window,
            "query_chunk_size": chunk,
            "post_window_chunk_start": chunk_start,
            "post_window_chunk_end_exclusive": chunk_end,
            "first_query_direct_history_tokens": counts[0],
            "last_query_direct_history_tokens": counts[-1],
            "minimum_direct_history_tokens_in_full_post_window_chunk": min(counts),
            "maximum_direct_history_tokens_in_full_post_window_chunk": max(counts),
            "history_span_delta_within_chunk": max(counts) - min(counts),
            "position_embedding_period": window,
        },
        "interpretation": [
            "the shipped semantics are chunk-end anchored rather than a per-query fixed-width lower bound",
            "a full post-window 256-token chunk therefore gives direct self-attention history from 3841 through 4096 tokens across query positions",
            "this is not classified as an implementation bug because incremental generation explicitly reproduces the same reference semantics",
            "24 decoder layers and repeated cross-attention to conditioning memory may extend effective information flow beyond one layer; this probe does not infer adequacy or inadequacy",
        ],
        "required_before_readiness": [
            "measure real teacher topology token-length and dependency-span distributions",
            "stratify structural errors by dependency span before changing the window/chunk design",
            "if a learned ablation is required, preregister it before inspecting candidate outcomes",
        ],
        "training_executed": False,
        "model_inference_executed": False,
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=REFERENCE_WINDOW)
    parser.add_argument("--chunk", type=int, default=REFERENCE_CHUNK)
    args = parser.parse_args()
    report = run(args.window, args.chunk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "status": report["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
