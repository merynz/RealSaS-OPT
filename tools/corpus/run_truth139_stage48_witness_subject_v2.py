#!/usr/bin/env python3
"""Stage48 V2 runner.

Keeps the canonical V1 Stage48 policy/qualification logic unchanged while routing
only the independent source-reference Blender subprocess through the
attribute-complete V2 adapter.  Packed measurement remains on the already-fixed
V2 presentation wrapper selected by the V1 runner.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.corpus import run_truth139_stage48_witness_subject_v1 as base


_ORIGINAL_RUN_BLENDER = base.run_blender


def _route_reference_v2(blender, program, args, log, maximum, blend=None):
    program = Path(program)
    if program.name == "blender_source_ray_appearance_reference_v1.py":
        program = ROOT / "tools/truth_foundry/blender_source_ray_appearance_reference_v2.py"
    return _ORIGINAL_RUN_BLENDER(blender, program, args, log, maximum, blend=blend)


def main() -> None:
    base.run_blender = _route_reference_v2
    base.main()


if __name__ == "__main__":
    main()
