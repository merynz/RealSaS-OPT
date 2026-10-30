#!/usr/bin/env python3
"""Validate the guarded Truth139 K-hit rescue producer with the unchanged CPU oracle court.

This wrapper changes only the admitted producer revision.  The actual court remains
V2's complete CPU orthographic ray-stack comparison with the same face, barycentric,
depth and tie rules; no tolerance is relaxed by the rescue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.corpus import validate_truth139_graphics_layer_peel_v2 as v2
from tools.corpus.build_truth139_graphics_layer_peel_v3_rescue import PRODUCER_REVISION

PASS_STATUS = v2.PASS_STATUS


def validate(manifest: dict, **kwargs) -> dict:
    v2.PARENT_REVISION = PRODUCER_REVISION
    return v2.validate(manifest, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-peel-teacher-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-samples-per-view", type=int, default=128)
    parser.add_argument("--minimum-validated-per-view", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(args.layer_peel_teacher_manifest.resolve().read_text())
    result = validate(
        manifest,
        maximum_samples_per_view=args.maximum_samples_per_view,
        minimum_validated_per_view=args.minimum_validated_per_view,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "subject_id": result["subject_id"],
                "producer_revision": result["producer_revision"],
                "total_validated_row_count": result["total_validated_row_count"],
                "court_identity_sha256": result["court_identity_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
