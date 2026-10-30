#!/usr/bin/env python3
"""Truth139 V3.2 full finalizer V4 routing shim.

Delegates all V3 provenance/runtime/SVD logic unchanged.  The only additional
route replaces the Stage48 V1 executable with the Stage48 V2 executable, whose
sole semantic change is the independent source-reference attribute transport.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.corpus import run_truth139_v32_finalize_remaining_full139_v3 as base


_ORIGINAL_IMPL_RUN = base._ORIGINAL_RUN


def _route_stage48_v2(cmd, log=None):
    routed = []
    for item in cmd:
        text = str(item)
        if text.endswith("tools/corpus/run_truth139_stage48_witness_subject_v1.py"):
            text = str(ROOT / "tools/corpus/run_truth139_stage48_witness_subject_v2.py")
        routed.append(text)
    return _ORIGINAL_IMPL_RUN(routed, log)


def main() -> None:
    base._ORIGINAL_RUN = _route_stage48_v2
    base.main()


if __name__ == "__main__":
    main()
