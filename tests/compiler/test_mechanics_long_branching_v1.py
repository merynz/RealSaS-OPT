from __future__ import annotations

"""Non-humanoid long-tree Mechanics evaluation, no corpus/model input."""

from dataclasses import replace
import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanics_multigoal_solver_v1 import (
    solve_declared_multi_goal_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tools.mechanics.benchmark_long_branching_ik_v1 import (
    make_morphology_case_v1,solve_and_court_v1,_observability,
)


@pytest.mark.parametrize("dof_count,branches",[(24,2),(48,4),(72,6)])
def test_every_declared_branch_dof_reaches_a_tip_but_endpoint_observation_is_rank_deficient(
    dof_count,branches,
):
    case=make_morphology_case_v1(dof_count)
    obs=_observability(case)
    assert obs["declared_rotation_dof_count"]==dof_count
    assert obs["locally_nonzero_endpoint_columns"]==dof_count
    assert obs["endpoint_jacobian_rank"]<=3*branches
    assert obs["endpoint_jacobian_rank"]<dof_count
    assert obs["unidentifiable_joint_space_dimension"]>0
    assert obs["all_dofs_independently_identifiable"] is False
    assert len(case.rig.controls)==branches
    assert len(case.rig.joints)==1+branches*13


def test_branched_24_dof_reachable_world_contact_matches_independent_targets():
    case=make_morphology_case_v1(24)
    r=solve_and_court_v1(case,max_nfev=130,derivative_check=True)
    assert r["status"]=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert r["max_pin_residual_world"]<2e-3
    assert r["max_drag_residual_world"]<2e-3
    assert r["max_segment_length_absolute_error_world"]<1e-8
    assert r["differential_derivative_error"]<1e-6
    assert r["no_model_or_graphics_claim"] is True


def test_long_chain_wrong_contact_world_target_rejected_as_unsolved_not_rewritten():
    case=make_morphology_case_v1(24)
    anchors=dict(case.pin_world_targets)
    pinned=case.pins[0]
    anchors[pinned]=(100.,100.,100.)
    targets=dict(case.targets_world)
    targets[pinned]=anchors[pinned]
    # Keep independent expected goal aligned with the *intentionally altered*
    # declared world pin. Otherwise the reference court correctly reports
    # a forged target declaration, before testing unreachable mechanics.
    unsolvable=replace(case,pin_world_targets=anchors,targets_world=targets)
    result=solve_and_court_v1(unsolvable,max_nfev=30)
    assert result["status"]=="FAIL_MULTIGOAL_BEHAVIORAL_COURT"
    assert result["max_pin_residual_world"]>1.0


def test_morphology_policy_and_missing_contact_authority_fail_closed():
    with pytest.raises(ValueError,match="DOF_POLICY"):
        make_morphology_case_v1(23)
    case=make_morphology_case_v1(24)
    with pytest.raises(QualificationError,match="PIN_WORLD_TARGET_BINDING_DRIFT"):
        solve_declared_multi_goal_v1(
            case.rig,case.overrides,
            pinned_contact_control_ids=case.pins,
            pinned_world_targets={},
            jacobian_mode="ANALYTIC_RESEARCH_V1",
        )
    case72=make_morphology_case_v1(72)
    with pytest.raises(QualificationError,match="DOF_RESOURCE_BOUND"):
        solve_declared_multi_goal_v1(
            case72.rig,case72.overrides,
            pinned_contact_control_ids=case72.pins,
            pinned_world_targets=case72.pin_world_targets,
            maximum_dofs=71,
            jacobian_mode="ANALYTIC_RESEARCH_V1",
        )


def test_explicit_active_dof_policy_solves_reachable_drag_without_rewriting_rig():
    case=make_morphology_case_v1(24)
    from compiler.realsas_compiler_core.mechanics_multigoal_solver_v1 import (
        solve_declared_multi_goal_v1,
    )
    selected=(("J:1:0","RY"),)
    result=solve_declared_multi_goal_v1(
        case.rig,case.overrides,
        pinned_contact_control_ids=case.pins,
        pinned_world_targets=case.pin_world_targets,
        active_parameter_keys=selected,
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        regularization=1e-8,goal_tolerance=2e-3,max_nfev=80,
    )
    assert result.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert result.metadata["active_solver_parameter_count"]==1
    assert result.metadata["frozen_solver_parameter_count"]==23
    assert result.metadata["active_dof_selection_is_representation_policy_not_source_truth"] is True
    assert result.max_drag_residual<2e-3
    assert result.max_pin_residual<2e-3
    assert all(abs(angle)<1e-12 for jid,axis,angle in result.parameters_rad
               if (jid,axis) not in selected)
    assert len(case.rig.local_frames)==27


def test_explicit_active_dof_policy_does_not_fake_pass_when_target_needs_two_joints():
    case=make_morphology_case_v1(24)
    from tools.mechanics.benchmark_long_branching_ik_v1 import _rotated_about_axis
    by_id={j.canonical_joint_id:np.asarray(j.P) for j in case.rig.joints}
    p1=by_id["J:1:0"]
    p2=by_id["J:1:1"]
    tip=by_id["TIP:1"]
    axis=np.asarray((0.,1.,0.))
    # Independent two-joint forward: joint 1 rotates distal segment,
    # joint 0 rotates the entire descendant subtree.
    p_after_first=p1+_rotated_about_axis(p2-p1,axis,.12)
    p_target=p_after_first+_rotated_about_axis(
        _rotated_about_axis(tip-p2,axis,.25),axis,.12,
    )
    from compiler.realsas_compiler_core.mechanics_control_solver_v1 import ControlTransformIRV1
    new_edit=(
        ControlTransformIRV1("C:1",
            translation=tuple(map(float,p_target-tip))),
    )
    incomplete=solve_declared_multi_goal_v1(
        case.rig,new_edit,
        pinned_contact_control_ids=case.pins,
        pinned_world_targets=case.pin_world_targets,
        active_parameter_keys=(("J:1:0","RY"),),
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        regularization=1e-8,goal_tolerance=2e-3,max_nfev=90,
    )
    assert incomplete.status=="FAIL_MULTIGOAL_BEHAVIORAL_COURT"
    assert incomplete.max_drag_residual>2e-3
    full=solve_declared_multi_goal_v1(
        case.rig,new_edit,
        pinned_contact_control_ids=case.pins,
        pinned_world_targets=case.pin_world_targets,
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        regularization=1e-8,goal_tolerance=2e-3,max_nfev=90,
    )
    assert full.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert full.max_drag_residual<2e-3


def test_active_dof_policy_rejects_missing_duplicate_or_unknown_declared_axes():
    case=make_morphology_case_v1(24)
    for selected in (
        (), (("J:1:0","RX"),),
        (("J:1:0","RY"),("J:1:0","RY")),
    ):
        with pytest.raises(QualificationError,match="ACTIVE_DOF_POLICY_INVALID"):
            solve_declared_multi_goal_v1(
                case.rig,case.overrides,
                pinned_contact_control_ids=case.pins,
                pinned_world_targets=case.pin_world_targets,
                active_parameter_keys=selected,
                jacobian_mode="ANALYTIC_RESEARCH_V1",
            )


@pytest.mark.parametrize("dof_count,expected_rank",[(24,4),(48,8),(72,12)])
def test_generic_rank_policy_uses_only_declared_task_jacobian_and_restores_full_capability(
    dof_count,expected_rank,
):
    case=make_morphology_case_v1(dof_count)
    solved=solve_declared_multi_goal_v1(
        case.rig,case.overrides,
        pinned_contact_control_ids=case.pins,
        pinned_world_targets=case.pin_world_targets,
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        active_parameter_policy="RANK_REVEALING_RESEARCH_V1",
        goal_tolerance=2e-3,regularization=1e-8,max_nfev=90,
    )
    assert solved.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert solved.metadata["rank_revealing_candidate_task_rank"]==expected_rank
    assert len(solved.metadata["rank_revealing_candidate_parameter_keys"])==expected_rank
    assert solved.metadata["active_dof_selection_is_representation_policy_not_source_truth"] is True
    assert solved.max_pin_residual<2e-3
    assert solved.max_drag_residual<2e-3
    assert len(solved.parameters_rad)==dof_count


def test_rank_policy_mode_cannot_hallucinate_a_numeric_jacobian_or_combine_explicit_keys():
    case=make_morphology_case_v1(24)
    with pytest.raises(QualificationError,match="RANK_POLICY_REQUIRES_AUTO_ANALYTIC"):
        solve_declared_multi_goal_v1(
            case.rig,case.overrides,
            pinned_contact_control_ids=case.pins,
            pinned_world_targets=case.pin_world_targets,
            jacobian_mode="NUMERIC_V1",
            active_parameter_policy="RANK_REVEALING_RESEARCH_V1",
        )
    with pytest.raises(QualificationError,match="RANK_POLICY_REQUIRES_AUTO_ANALYTIC"):
        solve_declared_multi_goal_v1(
            case.rig,case.overrides,
            pinned_contact_control_ids=case.pins,
            pinned_world_targets=case.pin_world_targets,
            jacobian_mode="ANALYTIC_RESEARCH_V1",
            active_parameter_policy="RANK_REVEALING_RESEARCH_V1",
            active_parameter_keys=(("J:1:0","RY"),),
        )
    with pytest.raises(QualificationError,match="ACTIVE_DOF_POLICY_MODE_UNSUPPORTED"):
        solve_declared_multi_goal_v1(
            case.rig,active_parameter_policy="AUTOGENERATE_SOURCE_RIG",
        )


def test_rank_policy_falls_back_to_full_declared_dofs_but_does_not_hallucinate_nonlinear_success():
    from tests.compiler.test_mechanics_multigoal_solver_v1 import _branched_rig
    from compiler.realsas_compiler_core.mechanics_control_solver_v1 import ControlTransformIRV1
    rig=_branched_rig()
    # At a completely straight rest chain, the endpoint Jacobian has
    # rank 2 despite three declared DOFs. The inward/outward movement of
    # a bent two-link chain is a second-order effect at that singularity.
    edit=(ControlTransformIRV1(
        "A_HANDLE",translation=(-0.7,1.2,0.),
    ),)
    result=solve_declared_multi_goal_v1(
        rig,edit,pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        active_parameter_policy="RANK_REVEALING_RESEARCH_V1",
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        regularization=1e-9,goal_tolerance=2e-3,max_nfev=400,
    )
    assert result.metadata["rank_revealing_candidate_task_rank"]==2
    assert result.metadata["rank_revealing_fallback_to_all"] is True
    assert result.metadata["active_solver_parameter_count"]==3
    # Rank restoration gives every declared variable back to the optimizer;
    # it is NOT a guarantee that a local nonlinear solver can escape a
    # singular straight-chain starting pose. Fail closed, not fake PASS.
    assert result.status=="FAIL_MULTIGOAL_BEHAVIORAL_COURT"
    assert result.max_drag_residual>2e-3
    assert len(result.parameters_rad)==3


def test_rank_hysteresis_keeps_previous_independently_spanning_actuator_set():
    case=make_morphology_case_v1(24)
    first=solve_declared_multi_goal_v1(
        case.rig,case.overrides,
        pinned_contact_control_ids=case.pins,
        pinned_world_targets=case.pin_world_targets,
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        active_parameter_policy="RANK_REVEALING_RESEARCH_V1",
        regularization=1e-8,goal_tolerance=2e-3,
    )
    assert first.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    original=case.overrides[0]
    from compiler.realsas_compiler_core.mechanics_control_solver_v1 import ControlTransformIRV1
    second_edit=(ControlTransformIRV1(
        original.canonical_control_id,
        translation=(original.translation[0]+.0001,
                     original.translation[1],original.translation[2]),
    ),)
    again=solve_declared_multi_goal_v1(
        case.rig,second_edit,
        pinned_contact_control_ids=case.pins,
        pinned_world_targets=case.pin_world_targets,
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        active_parameter_policy="RANK_REVEALING_RESEARCH_V1",
        preferred_parameter_keys=first.metadata["rank_revealing_candidate_parameter_keys"],
        initial_parameters_rad=first.parameters_rad,
        regularization=1e-8,goal_tolerance=2e-3,
    )
    assert again.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert again.metadata["rank_revealing_prior_basis_reused"] is True
    assert (again.metadata["rank_revealing_candidate_parameter_keys"]
            ==first.metadata["rank_revealing_candidate_parameter_keys"])
    assert again.max_drag_residual<2e-3


def test_rank_hysteresis_rejects_unknown_and_non_rank_prior_basis():
    case=make_morphology_case_v1(24)
    args=dict(
        pinned_contact_control_ids=case.pins,
        pinned_world_targets=case.pin_world_targets,
        jacobian_mode="ANALYTIC_RESEARCH_V1",
    )
    with pytest.raises(QualificationError,match="HYSTERESIS_SCHEMA_INVALID"):
        solve_declared_multi_goal_v1(
            case.rig,case.overrides,
            active_parameter_policy="RANK_REVEALING_RESEARCH_V1",
            preferred_parameter_keys=(("NONEXISTENT","RY"),),**args,
        )
    with pytest.raises(QualificationError,match="HYSTERESIS_REQUIRES_RANK_POLICY"):
        solve_declared_multi_goal_v1(
            case.rig,case.overrides,
            preferred_parameter_keys=(("J:1:0","RY"),),**args,
        )
