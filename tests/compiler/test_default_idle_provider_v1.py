from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2,
    CanonicalRigControlIRV2,
    CanonicalRigIRV2,
)
from compiler.realsas_compiler_core.control_motion_ir_v1 import qualify_control_motion_v1
from compiler.realsas_compiler_core.default_idle_provider_v1 import (
    DEFAULT_IDLE_PROVIDER_V1,
    build_default_idle_proposal_v1,
)


def _rig() -> CanonicalRigIRV2:
    joints = (
        CanonicalDeformJointIRV2("J:core", (0.0, 0.0, 0.0), None, "source:core"),
        CanonicalDeformJointIRV2("J:limb-a", (-0.7, -0.2, 0.0), "J:core", "source:limb-a"),
        CanonicalDeformJointIRV2("J:limb-b", (0.8, -0.1, 0.0), "J:core", "source:limb-b"),
    )
    controls = (
        CanonicalRigControlIRV2(
            canonical_control_id="C:arachnid-core",
            control_kind="GENERIC_ROOT",
            target_canonical_joint_ids=("J:core", "J:limb-a", "J:limb-b"),
            parent_canonical_control_id=None,
            rest_P=(0.0, 0.0, 0.0),
            source_proposal_control_id="source:control-core",
        ),
        CanonicalRigControlIRV2(
            canonical_control_id="C:limb-a",
            control_kind="GENERIC_LOCAL",
            target_canonical_joint_ids=("J:limb-a",),
            parent_canonical_control_id="C:arachnid-core",
            rest_P=(-0.7, -0.2, 0.0),
            source_proposal_control_id="source:control-limb-a",
        ),
    )
    return CanonicalRigIRV2(
        joints=joints,
        local_frames=(),
        controls=controls,
        chains=(),
        constraints=(),
        attachment_anchors=(),
        deform_root_joint_ids=("J:core",),
        mesh_lineage_hash="mesh-lineage",
        feature_evidence_binding_hash="feature-evidence",
        proposal_evidence_hash="proposal-evidence",
        rig_program_qualification_hash="rig-program-qualification",
        deform_structure_hash="deform-structure",
        control_program_hash="control-program",
        rig_lineage_hash="rig-lineage",
        metadata={"authority": "COMPILER_CANONICAL_RIG"},
    )


def _mechanics(rig: CanonicalRigIRV2) -> CanonicalMechanicsIRV1:
    return CanonicalMechanicsIRV1(
        mesh_lineage_hash=rig.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        skin_lineage_hash="skin-lineage",
        constraint_schedule=(),
        executable_operator_ids=(),
        rest_pose_hash="rest-pose",
        qualification_hash="mechanics-qualification",
        mechanics_lineage_hash="mechanics-lineage",
        metadata={"authority": "COMPILER_CANONICAL_MECHANICS"},
    )


def test_default_idle_is_morphology_neutral_and_control_space_legal():
    rig = _rig()
    mechanics = _mechanics(rig)
    proposal = build_default_idle_proposal_v1(rig, mechanics)

    assert proposal.intent == "idle"
    assert proposal.loop is True
    assert proposal.metadata["provider"] == DEFAULT_IDLE_PROVIDER_V1
    assert proposal.metadata["morphology_semantics_used"] is False
    assert proposal.metadata["compiler_retargeting_performed"] is False
    assert len(proposal.tracks) == 1
    assert proposal.tracks[0].canonical_control_id == "C:arachnid-core"

    keys = proposal.tracks[0].keys
    assert len(keys) == 5
    assert keys[0].translation == (0.0, 0.0, 0.0)
    assert keys[-1].translation == (0.0, 0.0, 0.0)
    assert keys[1].translation[1] > 0.0
    assert keys[3].translation[1] < 0.0

    qualified = qualify_control_motion_v1(proposal, rig=rig, mechanics=mechanics)
    assert qualified.qualification_report["status"] == "PASS_CONTROL_MOTION_VALIDATED"
    assert qualified.qualification_report["track_retargeting_performed"] is False


def test_default_idle_selection_is_replay_stable():
    rig = _rig()
    mechanics = _mechanics(rig)
    left = build_default_idle_proposal_v1(rig, mechanics)
    right = build_default_idle_proposal_v1(rig, mechanics)
    assert left == right
    assert left.proposal_lineage_hash == right.proposal_lineage_hash
