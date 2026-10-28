#!/usr/bin/env python3
"""Compatibility entrypoint for the objective-exact Stage48 FIT8 bundle.

The V3 implementation preserves the V2 student-observability contract while
correctly separating Platform subject UUIDs from scientific logical subject keys.
The workflow keeps this V2 filename so existing deployment plumbing does not
change identity for unrelated steps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.truth_foundry.build_fit8_presentation_truth_bundle_v3 import build


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
