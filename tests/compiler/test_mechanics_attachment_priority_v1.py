from __future__ import annotations

"""Explicit Mechanics priority tests. Contact/IK is never soft."""

import math
import pytest

from compiler.realsas_compiler_core.mechanics_attachment_priority_v1 import (
    execute_explicit_attachment_priority_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_attachment_feasibility_v1 import (
    _two_grips,_world_targets,
)
from tests.compiler.test_mechanics_oriented_ik_v1 import _goals,_rz,I


def _kwargs(angle=.19):
    return dict(
        control_transforms=_goals(angle),
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        attachment_position_targets=_world_targets(angle),
        attachment_orientation_targets={
            "GRIP:A":_rz(angle),"GRIP:B":_rz(angle),
        },
        regularization=1e-9,
    )


def test_explicit_priority_keeps_all_targets_hard_by_default():
    rig=_two_grips()
    result=execute_explicit_attachment_priority_v1(rig,**_kwargs())
    assert result.status=="PASS_HARD_AND_ALL_REQUESTED_TARGETS"
    assert result.hard_contract_satisfied
    assert result.all_requested_targets_satisfied
    assert result.hard_solver.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert result.advisory_residuals==()
    assert result.metadata["control_or_pin_priorities"]=="ALWAYS_HARD_NO_DEMOTION"
    assert not result.full_requested_rigidity_court.witnesses


def test_explicit_advisory_position_is_reported_unmet_never_silently_qualified():
    rig=_two_grips()
    k=_kwargs()
    k["attachment_position_targets"]["GRIP:B"]=(20.,0.,0.)
    result=execute_explicit_attachment_priority_v1(
        rig,**k,
        position_priorities={"GRIP:A":"HARD","GRIP:B":"ADVISORY"},
    )
    assert result.status=="PASS_HARD_WITH_UNMET_ADVISORY_TARGETS"
    assert result.hard_contract_satisfied is True
    assert result.all_requested_targets_satisfied is False
    assert result.metadata["unmet_advisory_target_count"]==1
    assert result.metadata["full_requested_rigid_conflict_count"]>=1
    assert result.metadata["advisory_exemption_not_source_capability_proof"]
    assert result.hard_solver.max_pin_residual<1e-3
    advisory=result.advisory_residuals[0]
    assert advisory.canonical_anchor_id=="GRIP:B"
    assert advisory.track_kind=="POSITION"
    assert not advisory.met
    assert advisory.residual>10.


def test_explicit_advisory_orientation_is_independently_measured():
    rig=_two_grips()
    k=_kwargs()
    k["attachment_orientation_targets"]["GRIP:B"]=_rz(-1.)
    result=execute_explicit_attachment_priority_v1(
        rig,**k,
        orientation_priorities={"GRIP:A":"HARD","GRIP:B":"ADVISORY"},
    )
    assert result.status=="PASS_HARD_WITH_UNMET_ADVISORY_TARGETS"
    assert result.hard_contract_satisfied
    assert len(result.advisory_residuals)==1
    r=result.advisory_residuals[0]
    assert r.track_kind=="ORIENTATION"
    assert r.residual>1.
    assert not r.met


def test_implicit_all_hard_conflict_preflight_does_not_even_run_optimizer():
    rig=_two_grips()
    k=_kwargs()
    k["attachment_position_targets"]["GRIP:B"]=(20.,0.,0.)
    result=execute_explicit_attachment_priority_v1(rig,**k)
    assert result.status=="FAIL_HARD_RIGID_ATTACHMENT_CONFLICT"
    assert result.hard_solver is None
    assert result.metadata["optimizer_invoked"] is False
    assert result.metadata["hard_rigid_witness_count"]>=1
    assert not result.hard_contract_satisfied
    assert not result.all_requested_targets_satisfied


def test_no_rigid_pair_conflict_does_not_pretend_full_kinematic_feasibility():
    rig=_two_grips()
    k=_kwargs()
    # A single grip can be 100 units out of reach yet carry no rigid pair
    # inconsistency when no other attachment target is demanded.
    k["attachment_position_targets"]={"GRIP:A":(100.,0.,0.)}
    k["attachment_orientation_targets"]={}
    result=execute_explicit_attachment_priority_v1(rig,**k,max_nfev=40)
    assert not result.hard_rigidity_court.witnesses
    assert result.status=="FAIL_HARD_IK_OR_CONTACT_CONTRACT"
    assert result.hard_solver.status=="FAIL_MULTIGOAL_BEHAVIORAL_COURT"
    assert not result.hard_contract_satisfied


def test_priority_requires_exact_explicit_per_track_coverage_and_known_modes():
    rig=_two_grips()
    k=_kwargs()
    with pytest.raises(QualificationError,match="POSITION_COVERAGE_DRIFT"):
        execute_explicit_attachment_priority_v1(
            rig,**k,position_priorities={"GRIP:A":"HARD"},
        )
    with pytest.raises(QualificationError,match="POSITION_MODE_UNSUPPORTED"):
        execute_explicit_attachment_priority_v1(
            rig,**k,position_priorities={
                "GRIP:A":"HARD","GRIP:B":"AUTO_LOW_PRIORITY",
            },
        )
    with pytest.raises(QualificationError,match="ORIENTATION_COVERAGE_DRIFT"):
        execute_explicit_attachment_priority_v1(
            rig,**k,orientation_priorities={"GRIP:A":"HARD"},
        )
    with pytest.raises(QualificationError,match="ORIENTATION_MODE_UNSUPPORTED"):
        execute_explicit_attachment_priority_v1(
            rig,**k,orientation_priorities={
                "GRIP:A":"HARD","GRIP:B":"OPTIONAL_SKIP_REPORT",
            },
        )


def test_priority_does_not_make_missing_authority_or_forged_anchors_advisory():
    rig=_two_grips()
    with pytest.raises(QualificationError,match="UNKNOWN_ANCHOR"):
        execute_explicit_attachment_priority_v1(
            rig,_goals(.2),
            attachment_position_targets={"FAKE":(0.,0.,0.)},
            position_priorities={"FAKE":"ADVISORY"},
        )
