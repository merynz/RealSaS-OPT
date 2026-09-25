from __future__ import annotations

import hashlib
import json

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.iris_geometry_v2 import (
    _DEMO_STAGE14_CLOSEST_RULE,
    _demo_stage14_fallback_prereg,
)


def _prereg(tmp_path):
    payload = {
        "schema": "RealSaS.KnightDemoStage14FallbackPreregistration.v1",
        "status": "FROZEN_BEFORE_KNIGHT_STAGE14_CANDIDATE_METRICS_INSPECTION",
        "run_id": "RUN",
        "subject_id": "SUBJECT",
        "execution_class": "DEMO_WITNESS",
        "eligibility": {
            "stage13_ledger_status_required": "PASS_DEMO_ONLY",
            "stage13_scientific_pass_required": False,
            "stage14_frozen_product_adequacy_status_required": "FAIL",
            "product_authority_claimed_required": False,
            "teacher_truth_allowed": False,
            "appearance_authority_allowed": False,
        },
        "frozen_closest_candidate_rule": {
            "rule_id": _DEMO_STAGE14_CLOSEST_RULE,
        },
        "demo_admission_semantics": {
            "product_pass_forbidden": True,
        },
    }
    path = tmp_path / "fallback.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _ctx(ref, *, execution_class="DEMO_WITNESS", stage13="PASS_DEMO_ONLY"):
    return {
        "ledger": {
            "run_id": "RUN",
            "subject_id": "SUBJECT",
            "execution_class": execution_class,
            "stages": [
                {
                    "id": "13_GEOMETRY_SUBSTRATE_QUALIFIED",
                    "status": stage13,
                }
            ],
        },
        "run_manifest": {
            "demo_execution": {
                "allow_stage14_scientific_fail_for_demo": True,
                "stage13_scientific_pass": False,
                "product_authority_claimed": False,
                "stage14_fallback_preregistration": ref,
            }
        },
    }


def test_stage14_demo_fallback_requires_demo_witness(tmp_path):
    ref = _prereg(tmp_path)
    assert _demo_stage14_fallback_prereg(
        _ctx(ref, execution_class="WITNESS")
    ) is None


def test_stage14_demo_fallback_accepts_exact_prereg(tmp_path):
    ref = _prereg(tmp_path)
    payload, digest = _demo_stage14_fallback_prereg(_ctx(ref))
    assert payload["status"].startswith("FROZEN_BEFORE_")
    assert digest == ref["sha256"]


def test_stage14_demo_fallback_rejects_non_demo_stage13(tmp_path):
    ref = _prereg(tmp_path)
    with pytest.raises(
        QualificationError,
        match="DEMO_STAGE14_FALLBACK_STAGE13_STATUS_DRIFT",
    ):
        _demo_stage14_fallback_prereg(_ctx(ref, stage13="PASS"))
