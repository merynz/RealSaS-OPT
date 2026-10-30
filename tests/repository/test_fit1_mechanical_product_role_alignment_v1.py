from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMOTION = ROOT / "canonical/KNIGHT_AXIS541_MIRA55_CARRIER_NATIVE_PROMOTION_20261007.json"
CAPABILITIES = ROOT / "platform/internal/capability/contracts/product_capabilities_v1.json"
RESPONSIBILITIES = ROOT / "platform/internal/capability/contracts/product_responsibilities_v1.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_fit1_axis541_remains_historical_regression_while_product_rig_moves_to_v6():
    baseline = _load(PROMOTION)
    capabilities = _load(CAPABILITIES)["capabilities"]
    ledger = _load(RESPONSIBILITIES)["responsibilities"]
    rig = capabilities["product.rig"]
    metadata = rig["metadata"]
    architecture = baseline["architecture"]

    # The sealed Knight/FIT1 witness is immutable historical evidence.
    assert baseline["claims"]["fit1_knight_mechanical_relative_source_envelope_closed"] is True
    assert baseline["claims"]["product_authority"] is False
    assert architecture["axis_parent_authority"] == "HARD_CAUSAL_REQUIRED_TREE"
    assert architecture["compiler_parent_reselection_allowed"] is False
    assert architecture["axis_diffusion_runtime_authority"] is False
    assert architecture["axis_geometry_authority"] == "ORDERED_256_BIN_XYZ_ARGMAX"

    # Active product Rig is now the V6 control-ready program, not the historical
    # AXIS541 skeleton-only implementation.
    assert metadata["executor_contract"] == "RealSaS.ProductRigInference.v2"
    assert metadata["output_schema_version"] == "v2"
    assert metadata["compiler_parent_reselection_allowed"] is False
    assert metadata["compiler_control_synthesis_allowed"] is False
    assert metadata["compiler_constraint_synthesis_allowed"] is False
    assert metadata["compiler_constraint_solver_execution_allowed"] is False
    assert metadata["mesh_is_canonical_parent"] is True
    assert metadata["feature_evidence_is_canonical_parent"] is False
    assert metadata["fit_allowed_during_product_execution"] is False
    assert metadata["required_models"] == [
        {
            "family": "AXIS",
            "architecture_id": "RealSaS.AXIS.RigProgram.TypedCausal.ControlSkin.v6_0",
        }
    ]

    # Post-bind observable behavior remains Mechanics responsibility. Moving Rig
    # to a richer control program does not move deformation/observability proof
    # upstream into AXIS or the Rig Compiler court.
    by_id = {row["id"]: row for row in ledger}
    assert "rig.frame_observability_qualification" not in by_id
    post_bind = by_id["mechanics.post_bind_frame_observability_qualification"]
    assert post_bind["role_id"] == "mechanics"
    assert post_bind["execution_owner_class"] == "COMPILER"
    assert post_bind["authority_kind"] == "QUALIFICATION"
    assert baseline["generic_compiler_rule"]["coincident_control_policy"] == "POST_BIND_SUBTREE_MECHANICAL_OBSERVABILITY"
