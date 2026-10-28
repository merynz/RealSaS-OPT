#!/usr/bin/env python3
"""Compatibility entrypoint for the current objective-exact Stage48 FIT8 bundle.

The workflow keeps this historical V2 filename so deployment plumbing remains
stable.  The current implementation is V4: V3 provides exact Platform UUID to
logical FIT8 binding, then V4 admits or explicitly abstains the pair family from
its measured cross-fold observability census.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.truth_foundry.build_fit8_presentation_truth_bundle_v4 import build


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observation", "artifact-root", "repo-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    build(
        json.loads(Path(args.observation).read_text()),
        api_root=args.api_root,
        artifact_root=args.artifact_root,
        repo_root=args.repo_root,
        output=args.output,
    )
