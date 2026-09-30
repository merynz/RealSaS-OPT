from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    ZERO_SURFACE_NORMAL_OPERATOR_V2_ID,
    zero_surface_normal_operator_identity_v2,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.iris_geometry_v2 import (
    _validate_gsa_policy_document,
)

V3=Path("canonical/STAGE14_SUBSTRATE_ADEQUACY_POLICY_V3_20260929.json")
V2=Path("canonical/STAGE14_SUBSTRATE_ADEQUACY_POLICY_V2_20260920.json")


def _doc(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cfg(doc):
    gsa=dict(doc["gsa"])
    return {
        "normal_k":gsa["normal_k"],
        "visibility_depth_tolerance_norm":gsa["visibility_depth_tolerance_norm"],
        "adequacy_policy":copy.deepcopy(gsa["adequacy_policy"]),
    }


def test_v3_preserves_all_v2_numeric_thresholds_and_binds_topology_normal():
    old=_doc(V2)
    new=_doc(V3)
    assert new["schema"]=="RealSaS.Stage14SubstrateAdequacyPolicy.v3"
    assert new["status"].startswith("FROZEN_")
    assert new["gsa"]["adequacy_policy"]==old["gsa"]["adequacy_policy"]
    assert new["gsa"]["visibility_depth_tolerance_norm"]==old["gsa"]["visibility_depth_tolerance_norm"]
    assert new["gsa"]["normal_k"]==old["gsa"]["normal_k"]
    assert new["gsa"]["normal_operator"]==zero_surface_normal_operator_identity_v2()
    assert new["gsa"]["normal_operator"]["operator_id"]==ZERO_SURFACE_NORMAL_OPERATOR_V2_ID
    assert new["gsa"]["normal_operator"]["euclidean_cross_sheet_neighbors_forbidden"] is True


def test_v3_validator_rejects_operator_identity_drift():
    doc=_doc(V3)
    cfg=_cfg(doc)
    normal_k,tolerance,adequacy=_validate_gsa_policy_document(cfg,doc)
    assert normal_k==64
    assert tolerance==pytest.approx(0.02)
    assert adequacy==doc["gsa"]["adequacy_policy"]

    bad=copy.deepcopy(doc)
    bad["gsa"]["normal_operator"]["neighborhood_authority"]="EUCLIDEAN_KNN"
    with pytest.raises(QualificationError,match="GSA_POLICY_DOCUMENT_DRIFT:normal_operator"):
        _validate_gsa_policy_document(cfg,bad)


def test_v2_document_cannot_mint_new_v3_stage14_authority():
    old=_doc(V2)
    cfg={
        "normal_k":old["gsa"]["normal_k"],
        "visibility_depth_tolerance_norm":old["gsa"]["visibility_depth_tolerance_norm"],
        "adequacy_policy":old["gsa"]["adequacy_policy"],
    }
    with pytest.raises(QualificationError,match="GSA_POLICY_DOCUMENT_SCHEMA_INVALID"):
        _validate_gsa_policy_document(cfg,old)
