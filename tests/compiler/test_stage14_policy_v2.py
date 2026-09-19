from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.iris_geometry_v1 import (
    _validate_gsa_policy_document_v2,
)

POLICY_V2=Path("canonical/STAGE14_SUBSTRATE_ADEQUACY_POLICY_V2_20260920.json")
POLICY_V1=Path("canonical/STAGE14_SUBSTRATE_ADEQUACY_POLICY_V1_20260919.json")


def _doc():
    return json.loads(POLICY_V2.read_text(encoding="utf-8"))


def _inline_from_doc(doc):
    gsa=dict(doc["gsa"])
    return {
        "policy_document":{"path":str(POLICY_V2),"sha256":"0"*64},
        "normal_k":gsa["normal_k"],
        "visibility_depth_tolerance_norm":gsa["visibility_depth_tolerance_norm"],
        "adequacy_policy":copy.deepcopy(gsa["adequacy_policy"]),
    }


def test_stage14_v2_canonical_policy_is_frozen_12k_without_quality_relaxation():
    doc=_doc()
    p=doc["gsa"]["adequacy_policy"]

    assert doc["schema"]=="RealSaS.Stage14SubstrateAdequacyPolicy.v2"
    assert doc["status"].startswith("FROZEN_")
    assert p["max_candidate_nodes"]==12288
    assert p["visible_component_always_eligible"] is True
    assert p["component_aware_voxel_compaction"] is True

    # V2 extends compute/search capacity; it does not relax the calibrated V1 quality limits.
    assert p["max_dense_to_surface_p95_norm"]==pytest.approx(0.03)
    assert p["max_dense_to_surface_max_norm"]==pytest.approx(0.08)
    assert p["max_normal_p95_deg"]==pytest.approx(30.0)
    assert p["max_projected_p95_px"]==pytest.approx(6.0)
    assert p["max_projected_max_px"]==pytest.approx(16.0)
    assert p["min_nodes_per_component"]==8
    assert p["max_component_alias_nodes"]==0


def test_stage14_v2_inline_manifest_must_exactly_match_frozen_document():
    doc=_doc()
    cfg=_inline_from_doc(doc)
    normal_k,tolerance,adequacy=_validate_gsa_policy_document_v2(cfg,doc)
    assert normal_k==64
    assert tolerance==pytest.approx(0.02)
    assert adequacy["max_candidate_nodes"]==12288

    bad=copy.deepcopy(cfg)
    bad["adequacy_policy"]["max_projected_p95_px"]=6.0001
    with pytest.raises(QualificationError,match="GSA_POLICY_DOCUMENT_DRIFT:adequacy_policy"):
        _validate_gsa_policy_document_v2(bad,doc)


def test_stage14_v2_rejects_non_frozen_or_wrong_schema_policy_document():
    doc=_doc()
    cfg=_inline_from_doc(doc)

    unfrozen=copy.deepcopy(doc)
    unfrozen["status"]="DRAFT"
    with pytest.raises(QualificationError,match="GSA_POLICY_DOCUMENT_NOT_FROZEN"):
        _validate_gsa_policy_document_v2(cfg,unfrozen)

    wrong=copy.deepcopy(doc)
    wrong["schema"]="RealSaS.Stage14SubstrateAdequacyPolicy.v1"
    with pytest.raises(QualificationError,match="GSA_POLICY_DOCUMENT_SCHEMA_INVALID"):
        _validate_gsa_policy_document_v2(cfg,wrong)


def test_stage14_v1_is_historical_only():
    old=json.loads(POLICY_V1.read_text(encoding="utf-8"))
    assert old["status"]=="SUPERSEDED__NO_NEW_STAGE14_PASS_AUTHORITY"
    assert old["superseded_by"]==str(POLICY_V2)
