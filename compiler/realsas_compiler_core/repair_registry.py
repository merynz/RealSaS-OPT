from __future__ import annotations

"""Fail-closed authority registry for production repair operations.

A historical repair implementation or a populated RepairDirective is never enough
for execution.  Current main must contain a separately qualified operation record
whose owner, operation family, qualification identity and allowed semantic change
patterns match exactly.
"""

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any, Callable


class RepairOperationNotPromotedError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepairOperationAuthorityRecordV1:
    operation_id: str
    owner_id: str
    operation_family: str
    current_main_status: str
    qualification_hash: str
    allowed_change_patterns: tuple[str, ...]
    historical_reference: str
    promotion_rule: str
    module_name: str = ""
    callable_name: str = ""
    schema_version: str = "RealSaS.RepairOperationAuthorityRecord.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# These records are intentionally NON-executable. Source audit found valuable
# historical semantics but current V4 lacks the exact upstream representation
# needed to execute them without bypassing current qualification authority.
REPAIR_OPERATION_AUTHORITY: dict[str, RepairOperationAuthorityRecordV1] = {
    "RIG_PARENT_REQUALIFICATION_V1": RepairOperationAuthorityRecordV1(
        operation_id="RIG_PARENT_REQUALIFICATION_V1",
        owner_id="rig",
        operation_family="select_retained_rig_parent_candidate",
        current_main_status="BLOCKED_REQUIRES_CURRENT_PROPOSAL_REQUALIFICATION_SEAM",
        qualification_hash="",
        allowed_change_patterns=("mechanical.skeleton",),
        historical_reference="v0.5/realsas_synthesis.causal_repair.apply_repair_directive_to_rig",
        promotion_rule=(
            "do not edit QualifiedSkeletonIRV2 parents in place; re-express the bounded "
            "counterfactual through current proposal/optimizer qualification and prove lineage"
        ),
    ),
    "WEIGHT_RETAINED_CANDIDATE_V1": RepairOperationAuthorityRecordV1(
        operation_id="WEIGHT_RETAINED_CANDIDATE_V1",
        owner_id="weight",
        operation_family="select_retained_weight_candidate",
        current_main_status="BLOCKED_MISSING_CURRENT_RETAINED_CANDIDATE_PORTFOLIO",
        qualification_hash="",
        allowed_change_patterns=(
            "mechanical.skin",
            "directional_visual.direction.*.component.*.mesh_skin",
        ),
        historical_reference="v0.5/realsas_weight.causal_repair.apply_repair_directive_to_weight_plan",
        promotion_rule=(
            "current qualified skin does not preserve the historical retained alternative portfolio; "
            "promote only after a typed current candidate-retention/requalification seam exists"
        ),
    ),
}


def operation_authority_v1(operation_id: str) -> RepairOperationAuthorityRecordV1:
    try:
        return REPAIR_OPERATION_AUTHORITY[str(operation_id)]
    except KeyError as exc:
        raise KeyError(f"unknown repair operation_id:{operation_id}") from exc


def operation_authority_blockers_v1(operation) -> tuple[str, ...]:
    try:
        record = operation_authority_v1(operation.operation_id)
    except KeyError:
        return ("repair_operation_not_in_current_authority_registry",)
    blockers: list[str] = []
    if record.current_main_status != "CANONICAL_MAINLINE_EXECUTABLE":
        blockers.append(f"repair_operation_not_promoted:{record.current_main_status}")
    if str(operation.owner_id) != record.owner_id:
        blockers.append("repair_operation_owner_authority_mismatch")
    if str(operation.operation_family) != record.operation_family:
        blockers.append("repair_operation_family_authority_mismatch")
    if not record.qualification_hash or str(operation.qualification_hash) != record.qualification_hash:
        blockers.append("repair_operation_qualification_authority_mismatch")
    if tuple(operation.allowed_change_paths) != tuple(record.allowed_change_patterns):
        blockers.append("repair_operation_change_scope_authority_mismatch")
    return tuple(blockers)


def resolve_promoted_repair_operation_v1(operation_id: str) -> Callable[..., Any]:
    record = operation_authority_v1(operation_id)
    if record.current_main_status != "CANONICAL_MAINLINE_EXECUTABLE":
        raise RepairOperationNotPromotedError(
            f"{operation_id} is {record.current_main_status}; typed source-diff/qualification is required before execution"
        )
    if not record.qualification_hash or not record.module_name or not record.callable_name:
        raise RepairOperationNotPromotedError(f"{operation_id} lacks sealed executable authority")
    module = import_module(record.module_name)
    fn = getattr(module, record.callable_name)
    if not callable(fn):
        raise TypeError(f"repair operation target is not callable:{record.module_name}.{record.callable_name}")
    return fn
