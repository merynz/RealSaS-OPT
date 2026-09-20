from __future__ import annotations

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_presentation_graph_from_dict as v2_graph,
)
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    qualified_presentation_graph_from_dict as historical_graph,
)


def test_v2_presentation_graph_codec_preserves_historical_schema_semantics():
    payload = {
        "schema_version": "RealSaS.QualifiedPresentationGraphIR.v1",
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
                "appearance_binding_hash": "caa",
                "composition_binding_hash": "depth",
                "metadata": {},
            }
        ],
        "decisions": [],
        "skeleton_binding_hash": "skeleton",
        "mesh_binding_hash": "mesh",
        "partition_binding_hash": "partition",
        "carrier_policy_binding_hash": "carrier",
        "product_state_binding_hash": "mechanical",
        "presentation_structure_binding_hash": "structure",
        "appearance_set_binding_hash": "appearance-qualification",
        "composition_set_binding_hash": "canonical-depth",
        "qualification_report": {"status": "PASS"},
        "presentation_lineage_hash": "presentation",
        "metadata": {"appearance_authority": "CAA_V2"},
    }
    current = v2_graph(payload)
    historical = historical_graph(payload)
    assert current.to_dict() == historical.to_dict()
    assert current.slots[0].default_attachment_id is None
