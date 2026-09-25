from __future__ import annotations

import hashlib
import json

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.mesh_v2 import (
    _DEMO_STAGE18_FALLBACK_RULE,
    _demo_stage18_fallback_prereg,
)


def _prereg(tmp_path):
    payload = {
        "schema": "RealSaS.KnightDemoStage18MeshFallbackPreregistration.v1",
        "status": "FROZEN_BEFORE_KNIGHT_STAGE18_PARENT_QUALITY_METRICS_INSPECTION",
        "run_id": "RUN",
        "subject_id": "SUBJECT",
        "execution_class": "DEMO_WITNESS",
        "eligibility": {
            "stage15_ledger_status_required": "PASS_DEMO_ONLY",
            "product_authority_claimed_required": False,
            "requested_backend_required": "CANONICAL_CDT_LOCAL_CHART_V1",
            "only_admissible_precondition_failure": "RELATION_PARENT_MIN_ANGLE_BELOW_FROZEN_G3_TARGET",
            "boundary_split_must_remain_forbidden": True,
            "teacher_truth_allowed": False,
            "source_rig_labels_allowed": False,
            "source_skin_labels_allowed": False,
            "appearance_quality_allowed_for_selection": False,
            "future_geppetto_result_allowed_for_selection": False,
            "future_arachne_result_allowed_for_selection": False,
        },
        "fallback_rule": {
            "rule_id": _DEMO_STAGE18_FALLBACK_RULE,
            "producer": "CANONICAL_RELATION_BASELINE_V1",
            "candidate_choice": "EXACT_CURRENT_RELATION_BASELINE",
            "no_candidate_ranking": True,
            "no_threshold_relaxation": True,
            "stage19_remeasurement_required": True,
        },
        "demo_admission_semantics": {
            "stage19_actual_candidate_source_fidelity_replay_required": True,
            "product_pass_forbidden": True,
        },
    }
    path = tmp_path / "fallback.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _ctx(ref, *, execution_class="DEMO_WITNESS", stage15="PASS_DEMO_ONLY"):
    return {
        "ledger": {
            "run_id": "RUN",
            "subject_id": "SUBJECT",
            "execution_class": execution_class,
            "stages": [
                {
                    "id": "15_RIGGING_SURFACE_QUALIFIED",
                    "status": stage15,
                }
            ],
        },
        "run_manifest": {
            "demo_execution": {
                "product_authority_claimed": False,
                "stage18_mesh_fallback_preregistration": ref,
            }
        },
    }


def test_stage18_demo_fallback_requires_demo_witness(tmp_path):
    ref = _prereg(tmp_path)
    assert _demo_stage18_fallback_prereg(
        _ctx(ref, execution_class="WITNESS")
    ) is None


def test_stage18_demo_fallback_accepts_exact_prereg(tmp_path):
    ref = _prereg(tmp_path)
    payload, digest = _demo_stage18_fallback_prereg(_ctx(ref))
    assert payload["fallback_rule"]["rule_id"] == _DEMO_STAGE18_FALLBACK_RULE
    assert digest == ref["sha256"]


def test_stage18_demo_fallback_rejects_non_demo_stage15(tmp_path):
    ref = _prereg(tmp_path)
    with pytest.raises(
        QualificationError,
        match="DEMO_STAGE18_FALLBACK_STAGE15_STATUS_DRIFT",
    ):
        _demo_stage18_fallback_prereg(_ctx(ref, stage15="PASS"))
