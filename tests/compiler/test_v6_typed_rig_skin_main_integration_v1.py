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


def _qualified_synthetic_carrier_skin(mesh, rig):
    from compiler.realsas_compiler_core.carrier_skin_v1 import (
        QualifiedCarrierSkinIR, QualifiedCarrierSkinRow
    )
    from compiler.realsas_compiler_core.hashing import content_sha256

    root_joint_id=rig.joints[0].canonical_joint_id
    rows=tuple(
        QualifiedCarrierSkinRow(
            carrier_vertex_id=v.source_candidate_vertex_id,
            influences=((root_joint_id,1.0),),
            simplex_residual_before=0.0,
            correction_l1=0.0,
        ) for v in mesh.vertices
    )
    draft=QualifiedCarrierSkinIR(
        rows=rows,
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        carrier_topology_hash=mesh.carrier_topology_hash,
        carrier_geometry_hash=mesh.carrier_geometry_hash,
        skeleton_binding_hash=rig.deform_structure_hash,
        qualification_report={"status":"PASS_DIRECT_CARRIER_SKIN",
                              "test_fixture_only_not_teacher_evidence":True},
        skin_lineage_hash="",
    )
    sealed_hash=content_sha256({
        "carrier_evidence_hash":draft.carrier_evidence_hash,
        "carrier_topology_hash":draft.carrier_topology_hash,
        "carrier_geometry_hash":draft.carrier_geometry_hash,
        "skeleton_binding_hash":draft.skeleton_binding_hash,
        "rows":[row.to_dict() for row in draft.rows],
        "qualification_report":draft.qualification_report,
        "metadata":draft.metadata,
    })
    return replace(draft,skin_lineage_hash=sealed_hash)


def _real_rig_fixture():
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import mint_canonical_rig_v2
    from compiler.realsas_compiler_core.rig_program_ir_v1 import qualify_rig_program_v1
    from tests.compiler.test_control_ready_rig_v2 import _mesh,_proposal
    mesh=_mesh()
    rig=mint_canonical_rig_v2(
        mesh,qualified_program=qualify_rig_program_v1(_proposal(mesh))
    )
    return mesh,rig


def test_main_v6_rig_to_skin_mint_preserves_mesh_and_deform_authority():
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import mint_canonical_skin_v2
    mesh,rig=_real_rig_fixture()
    qualified=_qualified_synthetic_carrier_skin(mesh,rig)
    skin=mint_canonical_skin_v2(
        mesh,rig,qualified_skin=qualified,
        skin_code_proposal_hash="a"*64,
        skin_codec_architecture_id="fixture-codec-no-model-release",
        skin_codec_checkpoint_hash="b"*64,
    )
    assert skin.mesh_lineage_hash==mesh.mesh_lineage_hash
    assert skin.deform_structure_hash==rig.deform_structure_hash
    assert skin.carrier_skin_qualification_hash==qualified.skin_lineage_hash
    assert all(len(row.influences)==1 for row in skin.rows)
    assert skin.metadata["skin_code_decode_is_model_owned"] is True
    assert skin.metadata["control_program_mutation_invalidates_skin"] is False
    assert canonical_skin_from_dict_v2(skin.to_dict())==skin


def test_main_skin_mint_rejects_valid_hash_but_wrong_skeleton_binding():
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import mint_canonical_skin_v2
    mesh,rig=_real_rig_fixture()
    qualified=_qualified_synthetic_carrier_skin(mesh,rig)
    # A re-sealed and internally consistent source receipt still MUST NOT be
    # allowed to bind a different deform-structure identity.
    drifted=replace(qualified,skeleton_binding_hash="f"*64)
    from compiler.realsas_compiler_core.hashing import content_sha256
    drifted=replace(drifted,skin_lineage_hash=content_sha256({
        "carrier_evidence_hash":drifted.carrier_evidence_hash,
        "carrier_topology_hash":drifted.carrier_topology_hash,
        "carrier_geometry_hash":drifted.carrier_geometry_hash,
        "skeleton_binding_hash":drifted.skeleton_binding_hash,
        "rows":[row.to_dict() for row in drifted.rows],
        "qualification_report":drifted.qualification_report,
        "metadata":drifted.metadata,
    }))
    with pytest.raises(QualificationError,match="DEFORM_STRUCTURE_BINDING_DRIFT"):
        mint_canonical_skin_v2(
            mesh,rig,qualified_skin=drifted,
            skin_code_proposal_hash="a"*64,
            skin_codec_architecture_id="fixture-codec-no-model-release",
            skin_codec_checkpoint_hash="b"*64,
        )


def test_control_only_edit_reuses_exact_qualified_skin_without_rebinding():
    """Exercise real minting courts, not just matching string hashes."""
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import mint_canonical_rig_v2
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import mint_canonical_skin_v2
    from compiler.realsas_compiler_core.rig_program_ir_v1 import (
        qualify_rig_program_v1, seal_rig_program_proposal_v1,
    )
    from tests.compiler.test_control_ready_rig_v2 import _mesh, _proposal

    mesh = _mesh()
    proposal = _proposal(mesh)
    original_rig = mint_canonical_rig_v2(
        mesh, qualified_program=qualify_rig_program_v1(proposal),
    )
    existing_skin_qualification = _qualified_synthetic_carrier_skin(mesh, original_rig)
    original_skin = mint_canonical_skin_v2(
        mesh, original_rig, qualified_skin=existing_skin_qualification,
        skin_code_proposal_hash="a" * 64,
        skin_codec_architecture_id="fixture-codec-no-model-release",
        skin_codec_checkpoint_hash="b" * 64,
    )
    altered = seal_rig_program_proposal_v1(replace(
        proposal,
        controls=(replace(proposal.controls[0], rest_P=(0.5, 0.0, 0.0)),)
        + proposal.controls[1:],
        proposal_lineage_hash="",
    ))
    changed_rig = mint_canonical_rig_v2(
        mesh, qualified_program=qualify_rig_program_v1(altered),
    )
    reused = mint_canonical_skin_v2(
        mesh, changed_rig, qualified_skin=existing_skin_qualification,
        skin_code_proposal_hash="a" * 64,
        skin_codec_architecture_id="fixture-codec-no-model-release",
        skin_codec_checkpoint_hash="b" * 64,
    )
    assert changed_rig.control_program_hash != original_rig.control_program_hash
    assert changed_rig.deform_structure_hash == original_rig.deform_structure_hash
    assert reused.skin_lineage_hash == original_skin.skin_lineage_hash
    assert reused.rows == original_skin.rows


@pytest.mark.parametrize(
    "influences,reason",
    [
        ((("J:0", 0.75), ("J:1", 0.10)), "INFLUENCE_SIMPLEX_DRIFT"),
        ((("J:0", -0.25), ("J:1", 1.25)), "INFLUENCE_INVALID"),
        ((("J:0", 0.5), ("J:0", 0.5)), "JOINT_INFLUENCE_DUPLICATE_OR_EMPTY"),
        ((), "INFLUENCES_EMPTY"),
    ],
)
def test_rehashed_canonical_skin_still_rejects_semantically_illegal_weights(influences, reason):
    original = _sealed_skin()
    rehashed = replace(
        original,
        rows=(replace(original.rows[0], influences=influences),),
        skin_lineage_hash="",
    )
    rehashed = replace(
        rehashed, skin_lineage_hash=canonical_skin_lineage_hash_v2(rehashed),
    )
    with pytest.raises(QualificationError, match=reason):
        canonical_skin_from_dict_v2(rehashed.to_dict())


def test_rehashed_canonical_skin_rejects_duplicate_vertex_address():
    original = _sealed_skin()
    duplicate_row = replace(
        original.rows[0], source_carrier_vertex_id="CV:another",
    )
    rehashed = replace(
        original, rows=(original.rows[0], duplicate_row),
        skin_lineage_hash="",
    )
    rehashed = replace(
        rehashed, skin_lineage_hash=canonical_skin_lineage_hash_v2(rehashed),
    )
    with pytest.raises(QualificationError, match="ADDRESS_DUPLICATE"):
        canonical_skin_from_dict_v2(rehashed.to_dict())
