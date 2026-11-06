from __future__ import annotations

"""Mechanical secondary-chain behavior and live skin regressions.

No subject names, source physics, generated rig/skin or Graphics authority.
All tails/appendages below are explicitly declared synthetic joint chains.
"""

from dataclasses import replace
import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalAttachmentAnchorIRV2,
    CanonicalDeformJointIRV2, CanonicalJointFrameIRV2,
    CanonicalRigChainIRV2, CanonicalRigControlIRV2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinRowIRV2,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
)
from compiler.realsas_compiler_core.mechanics_secondary_motion_v1 import (
    SecondaryChainPolicyV1,
    SecondaryMotionPolicyV1,
    step_secondary_motion_v1,
)
from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import (
    MechanicsInterventionStepV1,
    evaluate_mechanics_intervention_frame_v1,
    evaluate_mechanics_intervention_sequence_v1,
)
from compiler.realsas_compiler_core.mechanics_prepared_skin_v1 import (
    prepare_mechanics_skin_deformer_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_live_multigoal_integration_v1 import _objects


I=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))


def _secondary_objects():
    mesh,rig,skin,mechanics=_objects()
    # Extend the declared rig with a completely independent trailing chain;
    # primary branch joint positions remain valid under ROOT translation.
    mesh=replace(
        mesh,
        vertices=tuple(replace(v,P=p) for v,p in zip(
            mesh.vertices,((0.,0.,0.),(0.,0.,1.),(0.,0.,2.)),
        )),
        mesh_lineage_hash="",
    )
    mesh=replace(mesh,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(mesh))
    rig=replace(
        rig,
        joints=rig.joints+(
            CanonicalDeformJointIRV2("T1",(0.,0.,1.),"BASE","src:t1"),
            CanonicalDeformJointIRV2("T2",(0.,0.,2.),"T1","src:t2"),
        ),
        local_frames=rig.local_frames+(
            CanonicalJointFrameIRV2("T1",I,()),
            CanonicalJointFrameIRV2("T2",I,()),
        ),
        controls=(
            CanonicalRigControlIRV2(
                "C_ROOT","ROOT",("BASE",),None,(0.,0.,0.),"src:root",
            ),
        ),
        chains=rig.chains+(
            CanonicalRigChainIRV2(
                "CHAIN_SECONDARY",("BASE","T1","T2"),
                "DECLARED_TRAILING_CHAIN","src:secondary",
            ),
        ),
        constraints=(),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
    )
    skin=replace(
        skin,
        rows=(
            CanonicalSkinRowIRV2("MV:0",(("BASE",1.),),"CV:0"),
            CanonicalSkinRowIRV2("MV:1",(("T1",1.),),"CV:1"),
            CanonicalSkinRowIRV2("MV:2",(("T2",1.),),"CV:2"),
        ),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
    )
    mechanics=replace(
        mechanics,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        constraint_schedule=(),
        executable_operator_ids=(),
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        deform_structure_hash=rig.deform_structure_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
    )
    policy=SecondaryMotionPolicyV1(
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        chains=(SecondaryChainPolicyV1(
            "CHAIN_SECONDARY",frequency_hz=2.0,max_deviation_rad=0.85,
        ),),
    )
    prepared=prepare_mechanics_skin_deformer_v1(mesh,rig,skin)
    return mesh,rig,skin,mechanics,policy,prepared


def _frame(objects, x, *,prev=None,dt=None, policy=None):
    mesh,rig,skin,mechanics,default_policy,prepared=objects
    return evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        overrides=(ControlTransformIRV1(
            "C_ROOT",translation=(float(x),0.,0.),
        ),),
        prepared_skin=prepared,
        secondary_motion_policy=default_policy if policy is None else policy,
        secondary_motion_previous_state=prev,
        secondary_motion_dt_seconds=dt,
    )


def test_secondary_spring_lags_primary_translation_and_preserves_every_segment():
    obj=_secondary_objects()
    first=_frame(obj,0.)
    assert first.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert first.secondary_motion_state is not None
    follow=_frame(obj,1.,prev=first.secondary_motion_state,dt=1./60.)
    assert follow.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    joint={j.canonical_joint_id:np.asarray(j.solved_P) for j in follow.pose.joints}
    assert joint["BASE"][0]==pytest.approx(1.)
    assert joint["T1"][0]<joint["BASE"][0]-0.01
    assert np.linalg.norm(joint["T1"]-joint["BASE"])==pytest.approx(1.,abs=1e-9)
    assert np.linalg.norm(joint["T2"]-joint["T1"])==pytest.approx(1.,abs=1e-9)
    assert follow.metadata["secondary_motion_max_segment_length_error"]<1e-9
    assert follow.metadata["secondary_motion_max_deviation_rad"]<=.85+1e-8
    assert follow.max_segment_length_relative_error<1e-9
    assert follow.metadata["secondary_motion_mode"]=="CRITICALLY_DAMPED_DECLARED_CHAIN_RESEARCH"
    assert follow.pose.metadata["primary_constraint_execution_unchanged"] is True
    # LBS reads the post-secondary joint positions: the movable tail skin
    # must not simply snap rigidly to a root-translation endpoint.
    assert follow.deformation.positions[2][0]<1.0


def test_secondary_motion_returns_deterministic_state_and_converges():
    obj=_secondary_objects()
    original=_frame(obj,0.)
    a=_frame(obj,1.,prev=original.secondary_motion_state,dt=1/60)
    b=_frame(obj,1.,prev=original.secondary_motion_state,dt=1/60)
    assert a.pose==b.pose
    assert a.secondary_motion_state==b.secondary_motion_state
    state=a.secondary_motion_state
    final=a
    for _ in range(180):
        final=_frame(obj,1.,prev=state,dt=1/60)
        state=final.secondary_motion_state
    by={j.canonical_joint_id:np.asarray(j.solved_P) for j in final.pose.joints}
    assert np.linalg.norm(by["T2"]-np.asarray((1.,0.,2.)))<0.005
    assert state.step_count==181
    assert all(j.canonical_joint_id in ("T1","T2") for j in state.joints)


def test_secondary_full_sequence_and_prepared_skin_match_base_primary_authority():
    obj=_secondary_objects()
    mesh,rig,skin,mech,policy,prepared=obj
    steps=tuple(MechanicsInterventionStepV1(
        time_seconds=i/30.,
        overrides=(ControlTransformIRV1("C_ROOT",
             translation=(float(i>0),0.,0.)),),
    ) for i in range(20))
    secondary=evaluate_mechanics_intervention_sequence_v1(
        mesh,rig,skin,mech,steps,prepared_skin=prepared,
        secondary_motion_policy=policy,
    )
    rigid=evaluate_mechanics_intervention_sequence_v1(
        mesh,rig,skin,mech,steps,prepared_skin=prepared,
    )
    assert secondary.status==rigid.status=="PASS_MECHANICS_SEQUENCE_DIAGNOSTIC"
    assert secondary.metadata["secondary_motion_mode"]=="CRITICALLY_DAMPED_DECLARED_CHAIN_RESEARCH"
    assert rigid.metadata["secondary_motion_mode"]=="DISABLED"
    assert secondary.frames[-1].secondary_motion_state.step_count==19
    assert secondary.frames[-1].deformation.positions[2]!=rigid.frames[-1].deformation.positions[2]
    assert secondary.frames[-1].pose.metadata["secondary_motion"]=="CRITICALLY_DAMPED_POSITION_LAG_RESEARCH_ONLY"
    for delayed,base in zip(secondary.frames,rigid.frames):
        assert delayed.deformation.positions[0]==base.deformation.positions[0]
        assert delayed.max_segment_length_relative_error<1e-8


def test_secondary_rejects_primary_ik_and_authoritative_attachment_overlap():
    mesh,rig,skin,mechanics,policy,prepared=_secondary_objects()
    pose=evaluate_mechanics_intervention_frame_v1(mesh,rig,skin,mechanics).pose
    incompatible=replace(policy,chains=(SecondaryChainPolicyV1("CHAIN_A"),))
    # CHAIN_A owns joints used by primary declared IK, cannot act as tail.
    overlapping=replace(rig,constraints=(
        # Real primary IK would own A1/A2 (here explicitly made to fail
        # before any secondary displacements can occur).
        replace(_objects()[1].constraints[0],canonical_chain_id="CHAIN_A"),
    ))
    with pytest.raises(QualificationError,match="PRIMARY_CONSTRAINT_OWNERSHIP_CONFLICT"):
        step_secondary_motion_v1(overlapping,pose,incompatible)
    attached=replace(rig,attachment_anchors=(
        CanonicalAttachmentAnchorIRV2(
            "ATTACH_T2",parent_canonical_joint_id="T2",
            local_P=(0.,0.,0.),source_proposal_anchor_id="src:att",
        ),
    ))
    with pytest.raises(QualificationError,match="ATTACHMENT_OWNER_CONFLICT"):
        step_secondary_motion_v1(attached,pose,policy)


def test_secondary_negative_provenance_timing_state_and_parameters():
    obj=_secondary_objects()
    _,rig,_,_,policy,_=obj
    first=_frame(obj,0.)
    state=first.secondary_motion_state
    for dt in (0.,-0.1,0.21,float("nan")):
        with pytest.raises(QualificationError,match="STEP_DT_INVALID"):
            _frame(obj,0.2,prev=state,dt=dt)
    with pytest.raises(QualificationError,match="INIT_REQUIRES_NO_DT"):
        _frame(obj,0.,dt=.02)
    with pytest.raises(QualificationError,match="STATE_LINEAGE_OR_DOMAIN_DRIFT"):
        _frame(obj,0.2,prev=replace(state,rig_lineage_hash="wrong"),dt=.05)
    with pytest.raises(QualificationError,match="DYNAMICS_POLICY_INVALID"):
        _frame(obj,0.,policy=replace(policy,chains=(
            SecondaryChainPolicyV1("CHAIN_SECONDARY",frequency_hz=-2.),
        )))
    with pytest.raises(QualificationError,match="CHAIN_NOT_DECLARED"):
        _frame(obj,0.,policy=replace(policy,chains=(
            SecondaryChainPolicyV1("NO_SUCH_CHAIN"),
        )))
    with pytest.raises(QualificationError,match="CHAIN_DUPLICATE"):
        _frame(obj,0.,policy=replace(policy,chains=policy.chains+policy.chains))
    with pytest.raises(QualificationError,match="POLICY_AUTHORITY_LINEAGE_DRIFT"):
        _frame(obj,0.,policy=replace(policy,control_program_hash="stale"))


def test_secondary_fixed_ticks_are_repeatable_across_frame_partition():
    obj=_secondary_objects()
    initialized=_frame(obj,0.)
    # The target primary pose is constant after one instantaneous root edit.
    # 1 x 1/30s must match 2 x 1/60s under fixed 60Hz secondary stepping.
    once=_frame(obj,1.,prev=initialized.secondary_motion_state,dt=1/30)
    half=_frame(obj,1.,prev=initialized.secondary_motion_state,dt=1/60)
    twice=_frame(obj,1.,prev=half.secondary_motion_state,dt=1/60)
    for p,q in zip(once.pose.joints,twice.pose.joints):
        assert np.allclose(p.solved_P,q.solved_P,atol=1e-11,rtol=0.)
        assert np.allclose(p.rotation_basis,q.rotation_basis,atol=1e-11,rtol=0.)
    assert once.secondary_motion_state.metadata["integration_substeps_this_frame"]==2
    assert twice.secondary_motion_state.metadata["simulation_tick_count"]==2
    assert once.secondary_motion_state.metadata["simulation_tick_count"]==2
    # Runtime frame count and internal physics tick count are distinct.
    assert once.secondary_motion_state.step_count==1
    assert twice.secondary_motion_state.step_count==2


def test_secondary_reset_explicitly_clears_lag_and_discards_prior_state():
    obj=_secondary_objects()
    base=_frame(obj,0.)
    moved=_frame(obj,1.,prev=base.secondary_motion_state,dt=1/60)
    reset=_frame(obj,1.)  # intentional seek/reset, no previous state
    assert reset.secondary_motion_state.step_count==0
    assert reset.secondary_motion_state.metadata["simulation_tick_count"]==0
    assert reset.pose != moved.pose
    assert np.allclose(reset.pose.joints[-1].solved_P,(1.,0.,2.),atol=1e-9)


def test_secondary_substep_budget_and_invalid_fixed_step_fail_closed():
    obj=_secondary_objects()
    base=_frame(obj,0.)
    mesh,rig,skin,mechanics,policy,prepared=obj
    limited=replace(policy,maximum_substeps=2)
    with pytest.raises(QualificationError,match="SUBSTEP_RESOURCE_BOUND"):
        _frame(obj,.1,prev=base.secondary_motion_state,dt=.1,policy=limited)
    bad=replace(policy,fixed_step_seconds=0.)
    with pytest.raises(QualificationError,match="SUBSTEP_POLICY_INVALID"):
        _frame(obj,0.,policy=bad)
    bad=replace(policy,maximum_substeps=0)
    with pytest.raises(QualificationError,match="SUBSTEP_POLICY_INVALID"):
        _frame(obj,0.,policy=bad)


def test_secondary_cannot_hide_failed_primary_grip_orientation():
    obj=_secondary_objects()
    mesh,rig,skin,mechanics,policy,prepared=obj
    anchor=CanonicalAttachmentAnchorIRV2(
        "GRIP_PROTECTED",
        parent_canonical_joint_id="BASE",
        local_P=(0.,0.,0.),
        source_proposal_anchor_id="fixture:grip",
        metadata={"local_basis":(
            (0.,-1.,0.),(1.,0.,0.),(0.,0.,1.)
        )},
    )
    rig=replace(rig,attachment_anchors=(anchor,))
    with pytest.raises(
        QualificationError,match="SECONDARY_REQUIRES_QUALIFIED_PRIMARY_MECHANICS"
    ):
        evaluate_mechanics_intervention_frame_v1(
            mesh,rig,skin,mechanics,prepared_skin=prepared,
            secondary_motion_policy=policy,
            required_attachment_world_orientations={
                "GRIP_PROTECTED":I,
            },
        )


def test_secondary_runtime_physics_policy_edit_requires_explicit_reset():
    obj=_secondary_objects()
    mesh,rig,skin,mechanics,policy,prepared=obj
    start=_frame(obj,0.)
    altered=replace(policy,chains=(replace(
        policy.chains[0],frequency_hz=3.4,
    ),))
    with pytest.raises(QualificationError,match="STATE_POLICY_DRIFT_REQUIRES_RESET"):
        _frame(obj,.5,prev=start.secondary_motion_state,dt=1/60,policy=altered)
    reset=_frame(obj,.5,policy=altered)
    assert reset.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert reset.secondary_motion_state.step_count==0
    assert (reset.secondary_motion_state.metadata["runtime_policy_sha256"]
            !=start.secondary_motion_state.metadata["runtime_policy_sha256"])
    next_frame=_frame(obj,.5,prev=reset.secondary_motion_state,dt=1/60,policy=altered)
    assert next_frame.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
