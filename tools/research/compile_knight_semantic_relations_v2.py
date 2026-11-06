#!/usr/bin/env python3
from __future__ import annotations

"""Compile sealed semantic masks with conservative M-fusion v2.

This is a downstream-only research diagnostic. It reuses the existing V1
relation derivation unchanged after replacing only the mask->M semantic fusion
step with label-specific multiview support and calibrated confidence.
"""

import json
from pathlib import Path
import sys

from compiler.realsas_compiler_core.semantic_relation_evidence_v2 import (
    SCHEMA as FUSION_SCHEMA,
    lift_mask_evidence_to_m_v2,
)
from tools.research import compile_knight_semantic_relations_v1 as v1

MINIMUM_LABEL_SUPPORT_VIEWS = 2


def _output_path(argv: list[str]) -> Path:
    if "--output" not in argv:
        raise RuntimeError("--output is required")
    i = argv.index("--output")
    if i + 1 >= len(argv):
        raise RuntimeError("--output value missing")
    return Path(argv[i + 1])


def _fusion_adapter(**kwargs):
    return lift_mask_evidence_to_m_v2(
        **kwargs,
        minimum_label_support_views=MINIMUM_LABEL_SUPPORT_VIEWS,
    )


def main() -> int:
    # Keep all relation consumers exactly V1; replace only semantic fusion.
    v1.lift_mask_evidence_to_m_v1 = _fusion_adapter
    rc = v1.main()
    if rc != 0:
        return rc

    out = _output_path(sys.argv[1:])
    report = json.loads(out.read_text(encoding="utf-8"))
    report["schema"] = "RealSaS.KnightFrozenSemanticRelationDiagnostic.v2"
    report["semantic_fusion_policy"] = {
        "schema": FUSION_SCHEMA,
        "minimum_winning_label_support_views": MINIMUM_LABEL_SUPPORT_VIEWS,
        "same_label_same_view_aggregation": "MAX",
        "confidence": "LABEL_DOMINANCE_X_SUPPORTED_VIEW_MEAN_SCORE",
        "geometry_authority": "M_UNCHANGED",
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
