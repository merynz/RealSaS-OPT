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
    unsolvable=replace(case,pin_world_targets=anchors)
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
