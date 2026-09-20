from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_presentation_graph_from_dict,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.visibility_v2 import VISIBILITY_CONTRACT_V2_HASH


def _payload():
    payload = {
        "schema_version": "RealSaS.QualifiedPresentationGraphIR.v2",
        "slots": [
            {
                "slot_id": "S0",
                "bone_id": "J0",
                "setup_order": 0,
                "default_attachment_id": None,
                "keyable_channels": ["ATTACHMENT", "TINT"],
                "metadata": {"role": "test"},
            }
        ],
        "attachments": [],
        "view_overlays": [
            {
                "view_index": 0,
                "camera_binding_hash": "camera",
                "appearance_binding_hash": "caa-asset",
                "composition_binding_hash": "composition-policy",
                "metadata": {
                    "appearance_authority": "CAA_V2",
                    "visibility_contract_hash": VISIBILITY_CONTRACT_V2_HASH,
                },
            }
        ],
        "decisions": [],
        "skeleton_binding_hash": "skeleton",
        "mesh_binding_hash": "mesh",
        "partition_binding_hash": "partition",
        "carrier_policy_binding_hash": "carrier",
        "mechanical_state_binding_hash": "mechanical",
        "presentation_structure_binding_hash": "structure",
        "complete_appearance_asset_binding_hash": "caa-asset",
        "complete_appearance_qualification_binding_hash": "caa-qualification",
        "composition_policy_binding_hash": "composition-policy",
        "qualification_report": {
            "status": "PASS_CAA_BOUND_PRESENTATION_V2",
            "visibility_contract_hash": VISIBILITY_CONTRACT_V2_HASH,
        },
        "presentation_lineage_hash": "",
        "metadata": {"legacy_appearance_set_semantics_used": False},
    }
    hash_payload = dict(payload)
    hash_payload.pop("presentation_lineage_hash")
    payload["presentation_lineage_hash"] = content_sha256(hash_payload)
    return payload


def test_v2_presentation_graph_codec_uses_explicit_caa_authority_fields():
    graph = qualified_presentation_graph_from_dict(_payload())
    assert graph.schema_version == "RealSaS.QualifiedPresentationGraphIR.v2"
    assert graph.mechanical_state_binding_hash == "mechanical"
    assert graph.complete_appearance_asset_binding_hash == "caa-asset"
    assert (
        graph.complete_appearance_qualification_binding_hash
        == "caa-qualification"
    )
    assert graph.composition_policy_binding_hash == "composition-policy"
    assert not hasattr(graph, "appearance_set_binding_hash")
    assert not hasattr(graph, "composition_set_binding_hash")
    assert not hasattr(graph, "product_state_binding_hash")


def test_v2_codec_rejects_historical_presentation_graph_schema():
    payload = _payload()
    payload["schema_version"] = "RealSaS.QualifiedPresentationGraphIR.v1"
    with pytest.raises(ValueError, match="V2_ARTIFACT_SCHEMA_MISMATCH"):
        qualified_presentation_graph_from_dict(payload)
