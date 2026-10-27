#!/usr/bin/env python3
from __future__ import annotations

"""Diagnostic wrapper for the source-ray appearance reference producer.

The wrapped producer remains the authority under test. This file only writes a
small durable failure receipt when Blender exits before the normal reference
receipt exists, so a self-hosted court can be diagnosed without relying on the
GitHub log redirect endpoint.
"""

import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _output_from_argv() -> Path | None:
    if "--" not in sys.argv:
        return None
    raw = sys.argv[sys.argv.index("--") + 1 :]
    try:
        index = raw.index("--output")
    except ValueError:
        return None
    if index + 1 >= len(raw):
        return None
    return Path(raw[index + 1]).expanduser().resolve()


def main() -> None:
    output = _output_from_argv()
    try:
        from tools.truth_foundry import blender_source_ray_appearance_reference_v1 as reference

        reference.main()
    except BaseException as exc:
        if output is not None:
            failure = output.with_name(output.stem + "_FAILURE.json")
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(
                json.dumps(
                    {
                        "schema": "RealSaS.SourceRayAppearanceReferenceFailure.v1",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc().splitlines(),
                        "authority": "DIAGNOSTIC_ONLY__NO_SCIENTIFIC_CLAIM",
                        "qualification_minted": False,
                        "training_allowed": False,
                    },
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
