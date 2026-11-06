from __future__ import annotations

"""Explicit mandatory-vs-advisory attachment task policy, research-only.

All supplied attachment targets are HARD unless their own position/orientation
track is *explicitly* labeled ADVISORY. IK effectors and world contact pins are
ALWAYS HARD. Advisory targets are never silently dropped from reporting.

A PASS_HARD_* outcome is not full contract certification if an advisory
target remains unmet; this module neither reclassifies source truth nor
creates any CanonicalRig constraint, physical joint limit or graphics order.
"""

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .mechanics_control_solver_v1 import ControlTransformIRV1
from .mechanics_multigoal_solver_v1 import (
    MechanicsMultiGoalResultV1, solve_declared_multi_goal_v1,
)
from .mechanics_attachment_feasibility_v1 import (
    AttachmentRigidFeasibilityCourtV1,
    evaluate_declared_attachment_rigidity_v1,
)
from .mechanics_attachment_frames_v1 import evaluate_attachment_world_frames_v1
from .types import QualificationError


@dataclass(frozen=True)
class AdvisoryAttachmentResidualV1:
    canonical_anchor_id: str
    track_kind: str  # POSITION | ORIENTATION
    residual: float
    tolerance: float
    met: bool


@dataclass(frozen=True)
class MechanicsAttachmentPriorityResultV1:
    status: str
    hard_solver: MechanicsMultiGoalResultV1 | None
    hard_rigidity_court: AttachmentRigidFeasibilityCourtV1
    full_requested_rigidity_court: AttachmentRigidFeasibilityCourtV1
    advisory_residuals: tuple[AdvisoryAttachmentResidualV1, ...]
    hard_contract_satisfied: bool
    all_requested_targets_satisfied: bool
    metadata: dict = field(default_factory=dict)


def _priority(
    provided: Mapping[str, object],
    declared: Mapping[str, str] | None,
    kind: str,
) -> dict[str, str]:
    if declared is None:
        return {aid:"HARD" for aid in provided}
    if not isinstance(declared,Mapping) or set(declared)!=set(provided):
        raise QualificationError("ATTACHMENT_PRIORITY_" + kind + "_COVERAGE_DRIFT")
    values=dict(declared)
    if any(v not in {"HARD","ADVISORY"} for v in values.values()):
        raise QualificationError("ATTACHMENT_PRIORITY_" + kind + "_MODE_UNSUPPORTED")
    return values


def execute_explicit_attachment_priority_v1(
    rig: CanonicalRigIRV2,
    control_transforms: Sequence[ControlTransformIRV1] = (),
    *,
    pinned_contact_control_ids: Sequence[str] = (),
    pinned_world_targets: Mapping[str, Sequence[float]] | None = None,
    attachment_position_targets: Mapping[str, Sequence[float]] | None = None,
    attachment_orientation_targets: Mapping[str, Sequence[Sequence[float]]] | None = None,
    position_priorities: Mapping[str, str] | None = None,
    orientation_priorities: Mapping[str, str] | None = None,
    goal_tolerance: float = 1e-3,
    orientation_tolerance_rad: float = 1e-3,
    regularization: float = 1e-9,
    max_nfev: int = 200,
) -> MechanicsAttachmentPriorityResultV1:
    positions=dict(attachment_position_targets or {})
    orientations=dict(attachment_orientation_targets or {})
    p_priorities=_priority(positions,position_priorities,"POSITION")
    o_priorities=_priority(orientations,orientation_priorities,"ORIENTATION")

    hard_p={k:v for k,v in positions.items() if p_priorities[k]=="HARD"}
    hard_o={k:v for k,v in orientations.items() if o_priorities[k]=="HARD"}
    advisory_p={k:v for k,v in positions.items() if p_priorities[k]=="ADVISORY"}
    advisory_o={k:v for k,v in orientations.items() if o_priorities[k]=="ADVISORY"}

    full_court=evaluate_declared_attachment_rigidity_v1(
        rig,
        required_world_positions=positions,
        required_world_orientations=orientations,
        position_tolerance_world=goal_tolerance,
        orientation_tolerance_rad=orientation_tolerance_rad,
    )
    hard_court=evaluate_declared_attachment_rigidity_v1(
        rig,
        required_world_positions=hard_p,
        required_world_orientations=hard_o,
        position_tolerance_world=goal_tolerance,
        orientation_tolerance_rad=orientation_tolerance_rad,
    )
    metadata={
        "authority":"EXPLICIT_RUNTIME_PRIORITY_RESEARCH_ONLY",
        "source_truth_minted":False,
        "control_or_pin_priorities":"ALWAYS_HARD_NO_DEMOTION",
        "hard_position_target_count":len(hard_p),
        "hard_orientation_target_count":len(hard_o),
        "advisory_position_target_count":len(advisory_p),
        "advisory_orientation_target_count":len(advisory_o),
        "all_input_targets_still_diagnosed":True,
        "advisory_exemption_not_source_capability_proof":True,
        "rig_or_graphics_authority_changed":False,
    }
    if hard_court.witnesses:
        return MechanicsAttachmentPriorityResultV1(
            status="FAIL_HARD_RIGID_ATTACHMENT_CONFLICT",
            hard_solver=None,hard_rigidity_court=hard_court,
            full_requested_rigidity_court=full_court,
            advisory_residuals=(),hard_contract_satisfied=False,
            all_requested_targets_satisfied=False,
            metadata={**metadata,"optimizer_invoked":False,
                      "hard_rigid_witness_count":len(hard_court.witnesses)},
        )

    solved=solve_declared_multi_goal_v1(
        rig,control_transforms,
        pinned_contact_control_ids=pinned_contact_control_ids,
        pinned_world_targets=pinned_world_targets,
        attachment_position_targets=hard_p or None,
        attachment_orientation_targets=hard_o or None,
        goal_tolerance=goal_tolerance,
        orientation_tolerance_rad=orientation_tolerance_rad,
        regularization=regularization,
        max_nfev=max_nfev,
        jacobian_mode="ANALYTIC_RESEARCH_V1",
    )
    hard_ok=solved.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    if not hard_ok:
        return MechanicsAttachmentPriorityResultV1(
            status="FAIL_HARD_IK_OR_CONTACT_CONTRACT",
            hard_solver=solved,hard_rigidity_court=hard_court,
            full_requested_rigidity_court=full_court,
            advisory_residuals=(),hard_contract_satisfied=False,
            all_requested_targets_satisfied=False,
            metadata={**metadata,"optimizer_invoked":True,
                      "hard_solver_status":solved.status},
        )

    # Independently evaluate EVERY advisory target on the unmodified solved
    # pose. A secondary priority never turns an unmet target into success.
    court=evaluate_attachment_world_frames_v1(
        rig,solved.pose,
        required_world_positions=positions,
        required_world_orientations=orientations,
        position_tolerance_world=goal_tolerance,
        angular_tolerance_rad=orientation_tolerance_rad,
    )
    by_id={row.canonical_anchor_id:row for row in court.frames}
    advisories=[]
    for aid in sorted(advisory_p):
        r=by_id[aid].position_target_residual_world
        if r is None or not math.isfinite(r):
            raise QualificationError("ATTACHMENT_ADVISORY_POSITION_COURT_MISSING")
        advisories.append(AdvisoryAttachmentResidualV1(
            aid,"POSITION",r,goal_tolerance,r<=goal_tolerance,
        ))
    for aid in sorted(advisory_o):
        r=by_id[aid].angular_target_residual_rad
        if r is None or not math.isfinite(r):
            raise QualificationError("ATTACHMENT_ADVISORY_ORIENTATION_COURT_MISSING")
        advisories.append(AdvisoryAttachmentResidualV1(
            aid,"ORIENTATION",r,orientation_tolerance_rad,r<=orientation_tolerance_rad,
        ))
    all_met=all(item.met for item in advisories)
    return MechanicsAttachmentPriorityResultV1(
        status=("PASS_HARD_AND_ALL_REQUESTED_TARGETS" if all_met
                else "PASS_HARD_WITH_UNMET_ADVISORY_TARGETS"),
        hard_solver=solved,
        hard_rigidity_court=hard_court,
        full_requested_rigidity_court=full_court,
        advisory_residuals=tuple(advisories),
        hard_contract_satisfied=True,
        all_requested_targets_satisfied=all_met,
        metadata={**metadata,"optimizer_invoked":True,
                  "unmet_advisory_target_count":sum(not a.met for a in advisories),
                  "full_requested_rigid_conflict_count":len(full_court.witnesses)},
    )


__all__=[
    "AdvisoryAttachmentResidualV1",
    "MechanicsAttachmentPriorityResultV1",
    "execute_explicit_attachment_priority_v1",
]
