from __future__ import annotations

"""Pre-training secondary Mechanics runtime contract regressions.

These fixtures are explicitly declared kinematic chains, NOT PartVerse teacher
labels, dynamic physical truth, or synthetic proof of learned behavior.
"""

from dataclasses import replace
import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalAttachmentAnchorIRV2,
    CanonicalDeformJointIRV2,
    CanonicalJointFrameIRV2,
    CanonicalRigChainIRV2,
    CanonicalRigIRV2,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    MechanicsPoseIRV1,
    SolvedJointTransformIRV1,
)
from compiler.realsas_compiler_core.mechanics_secondary_motion_v1 import (
    SecondaryChainPolicyV1,
    SecondaryMotionPolicyV1,
    step_secondary_motion_v1,
)
from compiler.realsas_compiler_core.types import QualificationError

I = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def _rig() -> CanonicalRigIRV2:
    ids = ("BASE", "T1", "T2")
    return CanonicalRigIRV2(
        joints=(
            CanonicalDeformJointIRV2("BASE", (0., 0., 0.), None, "fixture:root"),
            CanonicalDeformJointIRV2("T1", (0., 0., 1.), "BASE", "fixture:middle"),
            CanonicalDeformJointIRV2("T2", (0., 0., 2.), "T1", "fixture:tip"),
        ),
        local_frames=tuple(CanonicalJointFrameIRV2(jid, I, ()) for jid in ids),
        controls=(),
        chains=(CanonicalRigChainIRV2(
            "TRAIL", ids, "DECLARED_TRAILING_CHAIN", "fixture:chain",
        ),),
        constraints=(),
        attachment_anchors=(),
        deform_root_joint_ids=("BASE",),
        mesh_lineage_hash="fixture-mesh",
        feature_evidence_binding_hash="fixture-feature",
        proposal_evidence_hash="fixture-proposal",
        rig_program_qualification_hash="fixture-qualification",
        deform_structure_hash="fixture-deform",
        control_program_hash="fixture-control",
        rig_lineage_hash="fixture-rig",
    )


def _pose(x: float) -> MechanicsPoseIRV1:
    rows = tuple(SolvedJointTransformIRV1(
        jid, (0., 0., float(index)),
        (float(x), 0., float(index)), I,
    ) for index, jid in enumerate(("BASE", "T1", "T2")))
    return MechanicsPoseIRV1(rows, "fixture-rig", "fixture-control")


def _policy() -> SecondaryMotionPolicyV1:
    return SecondaryMotionPolicyV1(
        "fixture-rig", "fixture-control",
        (SecondaryChainPolicyV1("TRAIL", frequency_hz=2.,
                                max_deviation_rad=0.85),),
    )


def test_declared_secondary_follows_primary_without_moving_root():
    rig, policy = _rig(), _policy()
    initial = step_secondary_motion_v1(rig, _pose(0.), policy)
    assert initial.status == "PASS_SECONDARY_MOTION_BOUNDED"
    assert initial.state.step_count == 0
    first = step_secondary_motion_v1(
        rig, _pose(1.), policy, previous_state=initial.state,
        dt_seconds=1. / 60.,
    )
    positions = {j.canonical_joint_id: np.asarray(j.solved_P)
                 for j in first.pose.joints}
    assert positions["BASE"][0] == pytest.approx(1.)
    assert positions["T1"][0] < positions["BASE"][0] - .01
    assert np.linalg.norm(positions["T1"] - positions["BASE"]) == pytest.approx(1., abs=1e-9)
    assert np.linalg.norm(positions["T2"] - positions["T1"]) == pytest.approx(1., abs=1e-9)
    assert first.max_secondary_segment_error < 1e-8
    assert first.max_secondary_angular_deviation_rad <= .85 + 1e-8
    assert first.metadata["secondary_policy_is_not_source_truth"] is True


def test_secondary_replay_and_fixed_step_partition_are_deterministic():
    rig, policy = _rig(), _policy()
    initial = step_secondary_motion_v1(rig, _pose(0.), policy)
    args = dict(previous_state=initial.state, dt_seconds=1. / 60.)
    first = step_secondary_motion_v1(rig, _pose(1.), policy, **args)
    again = step_secondary_motion_v1(rig, _pose(1.), policy, **args)
    assert first == again
    single = step_secondary_motion_v1(
        rig, _pose(1.), policy, previous_state=initial.state,
        dt_seconds=1. / 30.,
    )
    partitioned = step_secondary_motion_v1(
        rig, _pose(1.), policy, previous_state=first.state,
        dt_seconds=1. / 60.,
    )
    for one, two in zip(single.pose.joints, partitioned.pose.joints):
        assert np.allclose(one.solved_P, two.solved_P, atol=1e-10, rtol=0.)
        assert np.allclose(one.rotation_basis, two.rotation_basis, atol=1e-10, rtol=0.)
    assert single.state.metadata["simulation_tick_count"] == 2
    assert partitioned.state.metadata["simulation_tick_count"] == 2


def test_secondary_never_overrides_declared_attachment_or_constraint():
    rig, policy = _rig(), _policy()
    attached = replace(rig, attachment_anchors=(
        CanonicalAttachmentAnchorIRV2(
            "WEAPON_SOCKET", parent_canonical_joint_id="T2",
            source_proposal_anchor_id="fixture:socket",
        ),
    ))
    with pytest.raises(QualificationError, match="SECONDARY_ATTACHMENT_OWNER_CONFLICT"):
        step_secondary_motion_v1(attached, _pose(0.), policy)
    with pytest.raises(QualificationError, match="SECONDARY_CHAIN_NOT_DECLARED"):
        step_secondary_motion_v1(
            rig, _pose(0.),
            replace(policy, chains=(SecondaryChainPolicyV1("UNKNOWN"),)),
        )


def test_policy_state_and_unobserved_physics_fail_closed():
    rig, policy = _rig(), _policy()
    initial = step_secondary_motion_v1(rig, _pose(0.), policy)
    with pytest.raises(QualificationError, match="SECONDARY_INIT_REQUIRES_NO_DT"):
        step_secondary_motion_v1(rig, _pose(0.), policy, dt_seconds=.02)
    with pytest.raises(QualificationError, match="SECONDARY_STATE_LINEAGE_OR_DOMAIN_DRIFT"):
        step_secondary_motion_v1(
            rig, _pose(1.), policy,
            previous_state=replace(initial.state, rig_lineage_hash="foreign"),
            dt_seconds=1. / 60.,
        )
    with pytest.raises(QualificationError, match="SECONDARY_STATE_POLICY_DRIFT_REQUIRES_RESET"):
        step_secondary_motion_v1(
            rig, _pose(1.),
            replace(policy, chains=(SecondaryChainPolicyV1("TRAIL", frequency_hz=3.),)),
            previous_state=initial.state, dt_seconds=1. / 60.,
        )
    with pytest.raises(QualificationError, match="SECONDARY_DYNAMICS_POLICY_INVALID"):
        step_secondary_motion_v1(
            rig, _pose(0.),
            replace(policy, chains=(SecondaryChainPolicyV1("TRAIL", frequency_hz=-1.),)),
        )
    assert initial.state.metadata["physical_truth_minted"] is False
    assert initial.state.metadata["training_authorized"] is False
