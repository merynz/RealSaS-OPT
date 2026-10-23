from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMOTION = ROOT / "canonical/KNIGHT_AXIS541_MIRA55_CARRIER_NATIVE_PROMOTION_20261007.json"
CAPABILITIES = ROOT / "platform/internal/capability/contracts/product_capabilities_v1.json"
RESPONSIBILITIES = ROOT / "platform/internal/capability/contracts/product_responsibilities_v1.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_product_rig_preserves_fit1_axis_authority_without_stealing_post_bind_mechanics():
    baseline = _load(PROMOTION)
    capabilities = _load(CAPABILITIES)["capabilities"]
    ledger = _load(RESPONSIBILITIES)["responsibilities"]
    rig = capabilities["product.rig"]
    metadata = rig["metadata"]
    architecture = baseline["architecture"]

    assert baseline["claims"]["fit1_knight_mechanical_relative_source_envelope_closed"] is True
    assert baseline["claims"]["product_authority"] is False
    assert architecture["axis_parent_authority"] == "HARD_CAUSAL_REQUIRED_TREE"
    assert architecture["compiler_parent_reselection_allowed"] is False
    assert architecture["axis_diffusion_runtime_authority"] is False
    assert architecture["axis_geometry_authority"] == "ORDERED_256_BIN_XYZ_ARGMAX"

    assert metadata["hard_parent_authority_preserved"] is True
    assert metadata["compiler_parent_reselection_allowed"] is False
    assert metadata["frame_observability_claimed"] is False
    assert metadata["post_bind_frame_observability_required"] is True
    assert metadata["mesh_is_canonical_parent"] is True
    assert metadata["feature_evidence_is_canonical_parent"] is False
    assert metadata["fit_allowed_during_product_execution"] is False
    assert metadata["required_models"] == [
        {
            "family": "AXIS",
            "architecture_id": "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1",
        }
    ]

    by_id = {row["id"]: row for row in ledger}
    assert "rig.frame_observability_qualification" not in by_id
    post_bind = by_id["mechanics.post_bind_frame_observability_qualification"]
    assert post_bind["role_id"] == "mechanics"
    assert post_bind["execution_owner_class"] == "COMPILER"
    assert post_bind["authority_kind"] == "QUALIFICATION"
    assert baseline["generic_compiler_rule"]["coincident_control_policy"] == "POST_BIND_SUBTREE_MECHANICAL_OBSERVABILITY"
