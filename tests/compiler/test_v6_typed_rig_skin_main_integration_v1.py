from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2, CanonicalSkinRowIRV2,
    canonical_skin_from_dict_v2, canonical_skin_lineage_hash_v2,
)
from compiler.realsas_compiler_core.rig_constraint_schema_v1 import (
    RIG_CONSTRAINT_OPERATOR_IDS_V1, validate_rig_constraint_operator_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def _sealed_skin() -> CanonicalSkinIRV2:
    draft = CanonicalSkinIRV2(
        rows=(CanonicalSkinRowIRV2(
            "V:0", (("J:0", 0.75), ("J:1", 0.25)), "CV:0"
        ),),
        mesh_lineage_hash="1" * 64,
        deform_structure_hash="2" * 64,
        mesh_carrier_evidence_hash="3" * 64,
        mesh_carrier_topology_hash="4" * 64,
        mesh_carrier_geometry_hash="5" * 64,
        carrier_skin_qualification_hash="6" * 64,
        skin_code_proposal_hash="7" * 64,
        skin_codec_architecture_id="research-fixture-not-a-release",
        skin_codec_checkpoint_hash="8" * 64,
        skin_lineage_hash="",
    )
    return replace(
        draft, skin_lineage_hash=canonical_skin_lineage_hash_v2(draft)
    )


def test_v2_skin_exact_roundtrip_without_model_or_mechanics_runtime():
    sealed = _sealed_skin()
    assert canonical_skin_from_dict_v2(sealed.to_dict()) == sealed


def test_v2_skin_rejects_modified_deform_structure_without_lineage_resealing():
    original = _sealed_skin().to_dict()
    original["deform_structure_hash"] = "f" * 64
    with pytest.raises(QualificationError, match="CANONICAL_SKIN_V2_LINEAGE_HASH_DRIFT"):
        canonical_skin_from_dict_v2(original)


def test_v2_skin_rejects_modified_weights_without_lineage_resealing():
    original = _sealed_skin().to_dict()
    original["rows"][0]["influences"] = (("J:0", 0.5), ("J:1", 0.25))
    with pytest.raises(QualificationError, match="CANONICAL_SKIN_V2_LINEAGE_HASH_DRIFT"):
        canonical_skin_from_dict_v2(original)


def test_v6_operator_schema_is_exact_executable_subset_not_a_full_axis_language():
    assert RIG_CONSTRAINT_OPERATOR_IDS_V1 == (
        "IK", "AIM", "FOLLOW", "COPY_TRANSFORM", "STRETCH"
    )
    validate_rig_constraint_operator_v1(
        operator_id="IK", input_control_count=2,
        output_joint_count=3, has_chain=True,
        parameters={"p0": 1., "p1": 0.8},
    )
    for unsupported in ("LIMIT_ROTATION", "LIMIT_TRANSLATION", "PATH"):
        with pytest.raises(QualificationError, match="RIG_CONSTRAINT_OPERATOR_UNSUPPORTED"):
            validate_rig_constraint_operator_v1(
                operator_id=unsupported, input_control_count=1,
                output_joint_count=1, has_chain=False, parameters={},
            )


def test_v6_operator_schema_rejects_unproven_and_illformed_runtime_behavior():
    with pytest.raises(QualificationError, match="RIG_CONSTRAINT_CHAIN_REQUIRED"):
        validate_rig_constraint_operator_v1(
            operator_id="IK", input_control_count=1,
            output_joint_count=2, has_chain=False, parameters={}
        )
    with pytest.raises(QualificationError, match="RIG_CONSTRAINT_PARAMETER_NONFINITE"):
        validate_rig_constraint_operator_v1(
            operator_id="IK", input_control_count=1,
            output_joint_count=2, has_chain=True, parameters={"p0": float("nan")}
        )
