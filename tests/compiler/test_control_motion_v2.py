from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from compiler.realsas_compiler_core.canonical_motion_ir_v2 import (
    canonical_motion_from_dict_v2,
    mint_canonical_motion_v2,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2,
    CanonicalJointFrameIRV2,
    CanonicalRigControlIRV2,
    CanonicalRigIRV2,
)
from compiler.realsas_compiler_core.control_motion_ir_v1 import (
    ControlMotionClipProposalIRV1,
    ControlMotionKeyProposalIRV1,
    ControlMotionTrackProposalIRV1,
    qualify_control_motion_v1,
    seal_control_motion_proposal_v1,
)
from compiler.realsas_compiler_core.control_motion_runtime_v2 import (
    sample_control_transforms_v2,
    solve_motion_pose_v2,
)
from compiler.realsas_compiler_core.types import QualificationError


def _rig():
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return CanonicalRigIRV2(
        joints=(CanonicalDeformJointIRV2("J0", (0.0, 0.0, 0.0), None, "PJ0"),),
        local_frames=(CanonicalJointFrameIRV2("J0", identity, ("RX", "RY", "RZ")),),
        controls=(CanonicalRigControlIRV2("C0", "ROOT", ("J0",), None, (0.0, 0.0, 0.0), "PC0"),),
        chains=(), constraints=(), attachment_anchors=(), deform_root_joint_ids=("J0",),
        mesh_lineage_hash="1" * 64,
        feature_evidence_binding_hash="2" * 64,
        proposal_evidence_hash="3" * 64,
        rig_program_qualification_hash="4" * 64,
        deform_structure_hash="5" * 64,
        control_program_hash="6" * 64,
        rig_lineage_hash="7" * 64,
        metadata={"authority": "COMPILER_CANONICAL_RIG"},
    )


def _mechanics(rig):
    return CanonicalMechanicsIRV1(
        mesh_lineage_hash=rig.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        skin_lineage_hash="8" * 64,
        constraint_schedule=(),
        executable_operator_ids=(),
        rest_pose_hash="9" * 64,
        qualification_hash="a" * 64,
        mechanics_lineage_hash="b" * 64,
        metadata={"authority": "COMPILER_CANONICAL_MECHANICS"},
    )


def _proposal(rig, mechanics):
    return seal_control_motion_proposal_v1(ControlMotionClipProposalIRV1(
        proposal_clip_id="run",
        display_name="Run",
        intent="RUN",
        duration_seconds=1.0,
        loop=True,
        tracks=(ControlMotionTrackProposalIRV1(
            proposal_track_id="root-track",
            canonical_control_id="C0",
            keys=(
                ControlMotionKeyProposalIRV1(0.0, translation=(0.0, 0.0, 0.0)),
                ControlMotionKeyProposalIRV1(1.0, translation=(2.0, 0.0, 0.0)),
            ),
        ),),
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
    ))


def test_control_motion_is_canonicalized_without_joint_track_authority():
    rig = _rig()
    mechanics = _mechanics(rig)
    qualified = qualify_control_motion_v1(_proposal(rig, mechanics), rig=rig, mechanics=mechanics)
    motion = mint_canonical_motion_v2(qualified)
    assert motion.metadata["editable_motion_space"] == "ANIMATOR_CONTROLS"
    assert motion.metadata["direct_joint_track_authority"] is False
    assert motion.metadata["constraint_execution_owner"] == "MECHANICS"
    assert canonical_motion_from_dict_v2(motion.to_dict()) == motion


def test_control_motion_sampling_drives_mechanics_controls_not_joint_tracks():
    rig = _rig()
    mechanics = _mechanics(rig)
    motion = mint_canonical_motion_v2(
        qualify_control_motion_v1(_proposal(rig, mechanics), rig=rig, mechanics=mechanics)
    )
    controls = sample_control_transforms_v2(motion, rig=rig, mechanics=mechanics, time_seconds=0.5)
    assert len(controls) == 1
    assert controls[0].canonical_control_id == "C0"
    assert np.allclose(controls[0].translation, (1.0, 0.0, 0.0))
    pose = solve_motion_pose_v2(motion, rig=rig, mechanics=mechanics, time_seconds=0.5)
    assert np.allclose(pose.joints[0].solved_P, (1.0, 0.0, 0.0))


def test_motion_rejects_unknown_control_without_retargeting():
    rig = _rig()
    mechanics = _mechanics(rig)
    proposal = _proposal(rig, mechanics)
    bad_track = replace(proposal.tracks[0], canonical_control_id="C_MISSING")
    bad = seal_control_motion_proposal_v1(replace(proposal, tracks=(bad_track,), proposal_lineage_hash=""))
    with pytest.raises(QualificationError, match="MOTION_V2_CONTROL_UNKNOWN"):
        qualify_control_motion_v1(bad, rig=rig, mechanics=mechanics)


def test_motion_rejects_non_unit_quaternion_instead_of_repairing_it():
    rig = _rig()
    mechanics = _mechanics(rig)
    proposal = _proposal(rig, mechanics)
    keys = list(proposal.tracks[0].keys)
    keys[0] = replace(keys[0], rotation_quat_wxyz=(2.0, 0.0, 0.0, 0.0))
    bad_track = replace(proposal.tracks[0], keys=tuple(keys))
    bad = seal_control_motion_proposal_v1(replace(proposal, tracks=(bad_track,), proposal_lineage_hash=""))
    with pytest.raises(QualificationError, match="MOTION_V2_ROTATION_NOT_UNIT"):
        qualify_control_motion_v1(bad, rig=rig, mechanics=mechanics)
