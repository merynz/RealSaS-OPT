from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2,
    CanonicalJointFrameIRV2,
    CanonicalRigControlIRV2,
    CanonicalRigIRV2,
    _hash_control_program,
    _hash_deform_structure,
    canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
    solve_control_pose_v1,
)
from compiler.realsas_compiler_core.studio_rig_control_patch_v1 import (
    apply_qualified_studio_rig_control_patch_v1,
    qualify_studio_rig_control_patch_v1,
    studio_rig_control_patch_from_authoring_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def _base_rig() -> CanonicalRigIRV2:
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    value = CanonicalRigIRV2(
        joints=(
            CanonicalDeformJointIRV2("J0", (0.0, 0.0, 0.0), None, "PJ0"),
            CanonicalDeformJointIRV2("J1", (1.0, 0.0, 0.0), "J0", "PJ1"),
            CanonicalDeformJointIRV2("J2", (2.0, 0.0, 0.0), "J1", "PJ2"),
        ),
        local_frames=(
            CanonicalJointFrameIRV2("J0", identity, ("RX", "RY", "RZ")),
            CanonicalJointFrameIRV2("J1", identity, ("RX", "RY", "RZ")),
            CanonicalJointFrameIRV2("J2", identity, ("RX", "RY", "RZ")),
        ),
        controls=(
            CanonicalRigControlIRV2(
                "C_TARGET",
                "IK_TARGET",
                (),
                None,
                (2.0, 0.0, 0.0),
                "PC_TARGET",
            ),
        ),
        chains=(),
        constraints=(),
        attachment_anchors=(),
        deform_root_joint_ids=("J0",),
        mesh_lineage_hash="1" * 64,
        feature_evidence_binding_hash="2" * 64,
        proposal_evidence_hash="3" * 64,
        rig_program_qualification_hash="4" * 64,
        deform_structure_hash="",
        control_program_hash="",
        rig_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_RIG",
            "constraint_execution_authority": "MECHANICS",
            "skin_binding_target": "DEFORM_STRUCTURE_HASH",
            "motion_editing_target": "CONTROL_PROGRAM",
        },
    )
    value = replace(value, deform_structure_hash=_hash_deform_structure(value))
    value = replace(value, control_program_hash=_hash_control_program(value))
    return replace(value, rig_lineage_hash=canonical_rig_lineage_hash_v2(value))


def _authoring(**ik_override):
    ik = {
        "constraint_id": "IK:ARM",
        "start_bone_id": "J0",
        "end_bone_id": "J2",
        "target_control_id": "C_TARGET",
        "mix": 1.0,
        "max_stretch": 1.25,
        "enabled": True,
    }
    ik.update(ik_override)
    return {
        "schema_version": "RealSaS.UserPuppetEditLayer.v3",
        "source_product_state_hash": "a" * 64,
        "authority_class": "OPTIONAL_USER_AUTHORING_LAYER",
        "can_promote_product_directly": False,
        "requires_compiler_requalification": True,
        "rig_pose_edits": [],
        "mesh_vertex_edits": [],
        "mesh_topology_edits": [],
        "weight_edits": [],
        "binding_edits": [],
        "presentation_override": None,
        "rig_topology_edits": [],
        "ik_constraints": [ik],
        "animation_edits": [],
        "metadata": {"canonical_bundle_mutated": False},
    }


def _patch(authoring=None):
    rig = _base_rig()
    proposal = studio_rig_control_patch_from_authoring_v1(
        authoring or _authoring(),
        base_rig=rig,
        authoring_payload_sha256="b" * 64,
    )
    qualified = qualify_studio_rig_control_patch_v1(proposal, base_rig=rig)
    return rig, qualified, apply_qualified_studio_rig_control_patch_v1(rig, qualified)


def test_studio_ik_patch_preserves_deform_structure_and_changes_control_program():
    base, qualified, patched = _patch()
    assert qualified.qualification_report["status"] == "PASS_STUDIO_RIG_CONTROL_PATCH_VALIDATED"
    assert patched.deform_structure_hash == base.deform_structure_hash
    assert patched.control_program_hash != base.control_program_hash
    assert patched.rig_lineage_hash != base.rig_lineage_hash
    assert len(patched.chains) == 1
    assert patched.chains[0].canonical_joint_ids == ("J0", "J1", "J2")
    assert patched.chains[0].chain_kind == "STUDIO_IK_CHAIN"
    assert len(patched.constraints) == 1
    constraint = patched.constraints[0]
    assert constraint.operator_id == "IK"
    assert constraint.input_canonical_control_ids == ("C_TARGET",)
    assert constraint.output_canonical_joint_ids == ("J0", "J1", "J2")
    assert constraint.parameters == {"p0": 1.0, "p2": 1.25}
    assert patched.metadata["deform_structure_mutated_by_studio_patch"] is False
    assert patched.metadata["control_synthesis_performed_by_studio_patch"] is False


def test_studio_ik_patch_executes_through_existing_mechanics_solver():
    _, _, patched = _patch()
    rest = solve_control_pose_v1(patched, ())
    assert max(value for _, value in rest.constraint_residuals) <= 1e-8

    moved = solve_control_pose_v1(
        patched,
        (ControlTransformIRV1("C_TARGET", translation=(0.0, 1.0, 0.0)),),
    )
    by_id = {joint.canonical_joint_id: np.asarray(joint.solved_P) for joint in moved.joints}
    assert np.linalg.norm(by_id["J2"] - np.asarray((2.0, 1.0, 0.0))) <= 1e-5
    assert by_id["J1"][1] > 0.0


def test_studio_ik_patch_rejects_legacy_target_bone_semantics():
    authoring = _authoring()
    row = authoring["ik_constraints"][0]
    row.pop("target_control_id")
    row["target_bone_id"] = "J2"
    with pytest.raises(QualificationError, match="STUDIO_RIG_PATCH_IK_TARGET_CONTROL_REQUIRED"):
        studio_rig_control_patch_from_authoring_v1(
            authoring,
            base_rig=_base_rig(),
            authoring_payload_sha256="b" * 64,
        )


def test_studio_ik_patch_rejects_non_ancestor_chain():
    rig = _base_rig()
    proposal = studio_rig_control_patch_from_authoring_v1(
        _authoring(start_bone_id="J1", end_bone_id="J0"),
        base_rig=rig,
        authoring_payload_sha256="b" * 64,
    )
    with pytest.raises(QualificationError, match="STUDIO_RIG_PATCH_IK_START_NOT_ANCESTOR"):
        qualify_studio_rig_control_patch_v1(proposal, base_rig=rig)


def test_studio_ik_patch_rejects_mixed_deform_edit_domains():
    authoring = _authoring()
    authoring["rig_pose_edits"] = [
        {"control_id": "J1", "translation_xy": [1.0, 0.0], "rotation_degrees": 0.0}
    ]
    with pytest.raises(QualificationError, match="STUDIO_RIG_PATCH_UNSUPPORTED_EDIT_DOMAIN:rig_pose_edits"):
        studio_rig_control_patch_from_authoring_v1(
            authoring,
            base_rig=_base_rig(),
            authoring_payload_sha256="b" * 64,
        )
