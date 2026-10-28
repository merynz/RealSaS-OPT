#!/usr/bin/env python3
"""Stable-identity wrapper for IRIS V3.1 LOFO input preparation.

V1 reports whether a frozen-feature cache entry happened to be a cache hit in the
current process.  That is useful runtime telemetry but is not scientific input
identity: it flips from false to true after the first successful preparation and
would therefore break durable checkpoint resume binding for identical bytes.

V2 calls the exact V1 producer, removes only that operational diagnostic from the
sealed manifest, and rewrites the manifest atomically.  Feature receipt/content
hashes remain mandatory and unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.training.prepare_iris_v31_lofo_inputs_v1 import atomic_json, prepare as prepare_v1
from models.training_runtime.durable_state_v1 import file_sha256


def prepare(**kwargs):
    manifest = prepare_v1(**kwargs)
    for row in manifest.get("subjects") or ():
        cache = row.get("frozen_feature_cache") or {}
        cache.pop("cache_hit", None)
    manifest["runtime_cache_hit_excluded_from_scientific_identity"] = True
    output = Path(kwargs["output"]).resolve()
    path = output / "IRIS_V31_LOFO_INPUT_MANIFEST_V1.json"
    atomic_json(path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "sha256": file_sha256(path),
                "cache_hit_in_scientific_identity": False,
                "folds": len(manifest.get("folds") or ()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input-root", "legacy-manifest", "legacy-shard-root", "presentation-bundle",
        "presentation-root", "source-dir", "weight-path", "cache-root", "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view-batch-size", type=int, default=8)
    args = parser.parse_args()
    prepare(
        input_root=args.input_root,
        legacy_manifest=args.legacy_manifest,
        legacy_shard_root=args.legacy_shard_root,
        presentation_bundle=args.presentation_bundle,
        presentation_root=args.presentation_root,
        source_dir=args.source_dir,
        weight_path=args.weight_path,
        cache_root=args.cache_root,
        output=args.output,
        device=args.device,
        view_batch_size=args.view_batch_size,
    )
