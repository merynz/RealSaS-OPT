from __future__ import annotations

"""Deterministic XPBD numerical court on declared canonical rig/skin fixtures."""

from dataclasses import replace
import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    solve_control_pose_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_domain_v1 import (
    DeclaredXPBDSecondaryRequestV1, prepare_declared_xpbd_secondary_domain_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_secondary_v1 import (
    XPBDSecondaryPolicyV1, XPBDSecondaryChainPolicyV1,
    XPBDPlaneColliderV1, XPBDSphereColliderV1,
    step_declared_xpbd_secondary_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_xpbd_domain_v1 import _fixture


def _setup(**policy_changes):
    mesh, rig, skin, mechanics = _fixture()
    domain = prepare_declared_xpbd_secondary_domain_v1(
        mesh, rig, skin, mechanics,
        DeclaredXPBDSecondaryRequestV1(("K0",)),
    )
    policy = XPBDSecondaryPolicyV1(
        domain_hash=domain.domain_hash,
        chains=(XPBDSecondaryChainPolicyV1("K0"),),
        **policy_changes,
    )
    return rig, domain, policy, solve_control_pose_v1(rig, ())


def _translated(pose, x):
    return replace(pose, joints=tuple(replace(
        j, solved_P=(j.solved_P[0]+x,j.solved_P[1],j.solved_P[2])
    ) for j in pose.joints))


def _step(rig, domain, policy, pose, **kwargs):
    return step_declared_xpbd_secondary_v1(
        rig, pose, domain, policy, **kwargs,
    )


def test_xpbd_initialization_has_no_simulated_physics_or_truth_claim():
    rig, domain, policy, pose = _setup()
    output = _step(rig, domain, policy, pose)
    assert output.status == "INITIALIZED_XPBD_UNSIMULATED"
    assert output.state.simulation_tick_count == 0
    assert output.state.frame_count == 0
    assert output.state.metadata["physical_truth_minted"] is False
    assert output.metadata["compliance_scaled_by_inverse_dt_squared"] is True
    assert output.metadata["cloth_surface_particles_simulated"] is False
    assert output.pose.joints == pose.joints
    assert output.state.domain_hash == domain.domain_hash
    assert output.state == _step(rig,domain,policy,pose).state


def test_xpbd_moving_root_is_pinned_and_secondary_trails_without_length_drift():
    rig, domain, policy, primary = _setup()
    initial = _step(rig, domain, policy, primary)
    driven = _translated(primary, .3)
    output = _step(rig,domain,policy,driven,
                   previous_state=initial.state,dt_seconds=1./60.)
    assert output.status == "PASS_XPBD_SECONDARY_RUNTIME"
    joints = {j.canonical_joint_id: np.asarray(j.solved_P)
              for j in output.pose.joints}
    assert joints["J0"][0] == pytest.approx(.3)
    assert joints["J1"][0] < .3-1e-3
    assert np.linalg.norm(joints["J1"]-joints["J0"]) == pytest.approx(
        1.,abs=.02
    )
    assert np.linalg.norm(joints["J2"]-joints["J1"]) == pytest.approx(
        1.,abs=.02
    )
    assert output.state.simulation_tick_count == 2
    assert output.state.frame_count == 1
    assert output.max_segment_relative_error < .02
    assert output.metadata["rig_skin_immutable"] is True


def test_xpbd_fixed_tick_partition_and_replay_determinism():
    rig,domain,policy,primary=_setup()
    start=_step(rig,domain,policy,primary)
    full=_step(rig,domain,policy,primary,previous_state=start.state,
               dt_seconds=1./60.)
    a=_step(rig,domain,policy,primary,previous_state=start.state,
            dt_seconds=1./120.)
    b=_step(rig,domain,policy,primary,previous_state=a.state,
            dt_seconds=1./120.)
    assert full.pose.joints == b.pose.joints
    assert full.state.joints == b.state.joints
    assert full.state.simulation_tick_count == b.state.simulation_tick_count == 2
    assert full == _step(rig,domain,policy,primary,
                         previous_state=start.state,dt_seconds=1./60.)


def test_xpbd_sphere_collision_projects_tip_and_preserves_parent():
    sphere=XPBDSphereColliderV1("BALL",(.1,2.,0.),.18)
    rig,domain,policy,pose=_setup(spheres=(sphere,),gravity_xyz=(0.,0.,0.))
    start=_step(rig,domain,policy,pose)
    output=_step(rig,domain,policy,pose,previous_state=start.state,
                 dt_seconds=1./60.)
    p={j.canonical_joint_id:np.asarray(j.solved_P)
       for j in output.pose.joints}
    assert np.linalg.norm(p["J2"]-np.asarray(sphere.center)) >= .18-1e-4
    assert p["J0"].tolist()==[0.,0.,0.]
    assert output.applied_contact_projections>0
    assert output.max_contact_penetration<1e-4
    assert output.status=="PASS_XPBD_SECONDARY_RUNTIME"


def test_xpbd_unsatisfiable_collision_is_not_reported_as_pass():
    plane=XPBDPlaneColliderV1("CAP",(0.,1.5,0.),(0.,-1.,0.))
    rig,domain,policy,pose=_setup(planes=(plane,),gravity_xyz=(0.,0.,0.))
    start=_step(rig,domain,policy,pose)
    output=_step(rig,domain,policy,pose,previous_state=start.state,
                 dt_seconds=1./60.)
    assert output.status=="FAIL_XPBD_RESIDUAL_OR_PENETRATION"
    assert output.max_segment_relative_error>.02


def test_xpbd_frame_snapshot_is_read_only_and_rejects_stale_primary():
    rig,domain,policy,pose=_setup(gravity_xyz=(0.,0.,0.))
    start=_step(rig,domain,policy,pose)
    snapshot=_step(rig,domain,policy,pose,previous_state=start.state)
    assert snapshot.status=="SNAPSHOT_XPBD_NO_STEP"
    assert snapshot.state==start.state
    assert snapshot.pose.joints==pose.joints
    with pytest.raises(QualificationError,match="SNAPSHOT_PRIMARY_CHANGED_REQUIRES_DT"):
        _step(rig,domain,policy,_translated(pose,.3),
              previous_state=start.state)


def test_xpbd_state_runtime_policy_and_time_fail_closed():
    rig,domain,policy,pose=_setup()
    start=_step(rig,domain,policy,pose)
    with pytest.raises(QualificationError,match="INITIALIZE_WITHOUT_DT"):
        _step(rig,domain,policy,pose,dt_seconds=1./60.)
    with pytest.raises(QualificationError,match="FIXED_TICK_PARTITION_INVALID"):
        _step(rig,domain,policy,pose,previous_state=start.state,
              dt_seconds=.01)
    changed=replace(policy,velocity_damping_per_second=7.)
    with pytest.raises(QualificationError,match="STATE_DOMAIN_OR_POLICY_DRIFT"):
        _step(rig,domain,changed,pose,previous_state=start.state,
              dt_seconds=1./60.)
    with pytest.raises(QualificationError,match="POLICY_DOMAIN_DRIFT"):
        _step(rig,domain,replace(policy,domain_hash="foreign"),pose)
    with pytest.raises(QualificationError,match="STATE_JOINT_COVERAGE_DRIFT"):
        _step(rig,domain,policy,pose,
              previous_state=replace(start.state,joints=()),dt_seconds=1./60.)


def test_xpbd_rejects_invalid_collision_and_unbounded_simulation():
    rig,domain,policy,pose=_setup()
    with pytest.raises(QualificationError,match="PLANE_NORMAL_NOT_UNIT"):
        _step(rig,domain,replace(policy,planes=(
            XPBDPlaneColliderV1("BAD",(0.,0.,0.),(0.,3.,0.)),
        )),pose)
    with pytest.raises(QualificationError,match="COLLIDER_ID_DUPLICATE"):
        _step(rig,domain,replace(policy,planes=(
            XPBDPlaneColliderV1("A",(0.,0.,0.),(0.,1.,0.)),
            XPBDPlaneColliderV1("A",(0.,0.,0.),(0.,1.,0.)),
        )),pose)
    with pytest.raises(QualificationError,match="COMPLIANCE_INVALID"):
        _step(rig,domain,replace(policy,chains=(
            replace(policy.chains[0],follow_compliance=-.1),
        )),pose)
    with pytest.raises(QualificationError,match="ITERATION_RESOURCE_BOUND"):
        _step(rig,domain,replace(policy,iterations=0),pose)


def test_xpbd_is_real_compliance_not_hidden_spring_integration():
    rig,domain,hard,pose=_setup(gravity_xyz=(0.,0.,0.))
    start=_step(rig,domain,hard,pose)
    target=_translated(pose,.2)
    hard_output=_step(rig,domain,hard,target,previous_state=start.state,
                      dt_seconds=1./60.)
    soft=replace(hard,chains=(replace(
        hard.chains[0],stretch_compliance=.03,
    ),),maximum_segment_relative_error=.5)
    soft_initial=_step(rig,domain,soft,pose)
    soft_output=_step(rig,domain,soft,target,
                      previous_state=soft_initial.state,dt_seconds=1./60.)
    assert soft_output.max_segment_relative_error>hard_output.max_segment_relative_error
    assert soft_output.state.runtime_policy_hash != hard_output.state.runtime_policy_hash
