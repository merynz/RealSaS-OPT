from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import sha256_json
from .persistence.models import (
    AttemptEventRow,
    FailureSignatureRow,
    OwnerAttributionRow,
    RepairDirectiveRow,
)
from .stage_graph import StageGraph


@dataclass(frozen=True)
class FailureRecord:
    failure_signature_id: UUID
    owner_attribution_id: UUID
    repair_directive_id: UUID
    owner_stage_id: str
    invalidated_stage_ids: tuple[str, ...]
    signature_sha256: str


def _owner_kind(failure_class: str) -> str:
    return {
        "INPUT": "INPUT",
        "INFRA": "INFRA",
        "POLICY": "STAGE",
        "CODE": "STAGE",
        "ARTIFACT": "STAGE",
    }.get(failure_class, "STAGE")


def _directive_type(failure_class: str) -> str:
    return {
        "INPUT": "CHANGE_INPUT",
        "INFRA": "RETRY_INFRA",
        "POLICY": "CHANGE_POLICY",
        "CODE": "CHANGE_IMPLEMENTATION",
        "ARTIFACT": "REBUILD_OWNER_STAGE",
    }.get(failure_class, "REEXECUTE_OWNER_STAGE")


def record_stage_failure(
    session: Session,
    *,
    graph: StageGraph,
    attempt_id: UUID,
    failing_stage_id: str,
    error_code: str,
    diagnostics: dict,
    failure_class: str = "STAGE",
    execution_id: UUID | None = None,
    reported_owner_stage_id: str | None = None,
) -> FailureRecord:
    """Persist compiler failure localization and a bounded repair directive.

    An adapter may attribute a failure to itself or one of its true ancestors.
    It may not blame an unrelated/downstream stage.
    """
    graph.get(failing_stage_id)
    owner = reported_owner_stage_id or failing_stage_id
    graph.get(owner)
    legal_owners = set(graph.ancestors_including([failing_stage_id]))
    if owner not in legal_owners:
        raise ValueError(f"ILLEGAL_FAILURE_OWNER:{failing_stage_id}:{owner}")

    invalidated = graph.descendants_including([owner])
    payload = {
        "schema": "RealSaS.FailureSignature.v1",
        "failing_stage_id": failing_stage_id,
        "error_code": str(error_code),
        "failure_class": str(failure_class),
        "diagnostics": diagnostics,
    }
    signature = sha256_json(payload)

    existing = session.execute(
        select(FailureSignatureRow).where(
            FailureSignatureRow.attempt_id == attempt_id,
            FailureSignatureRow.signature_sha256 == signature,
        )
    ).scalar_one_or_none()
    if existing is not None:
        attribution = session.execute(
            select(OwnerAttributionRow).where(
                OwnerAttributionRow.failure_signature_id == existing.id
            )
        ).scalar_one()
        directive = session.execute(
            select(RepairDirectiveRow).where(
                RepairDirectiveRow.failure_signature_id == existing.id,
                RepairDirectiveRow.status == "OPEN",
            )
        ).scalar_one()
        return FailureRecord(existing.id, attribution.id, directive.id, attribution.owner_stage_id, tuple(directive.invalidated_stage_ids["stage_ids"]), signature)

    failure = FailureSignatureRow(
        id=uuid4(),
        attempt_id=attempt_id,
        execution_id=execution_id,
        stage_id=failing_stage_id,
        code=str(error_code),
        severity="ERROR",
        signature_sha256=signature,
        payload=payload,
    )
    attribution = OwnerAttributionRow(
        id=uuid4(),
        failure_signature_id=failure.id,
        owner_stage_id=owner,
        owner_kind=_owner_kind(failure_class),
        reason=(
            f"reported owner {owner} is within dependency ancestry of failing stage {failing_stage_id}"
            if reported_owner_stage_id
            else f"default stage-local ownership: {failing_stage_id}"
        ),
    )
    directive = RepairDirectiveRow(
        id=uuid4(),
        failure_signature_id=failure.id,
        owner_stage_id=owner,
        directive_type=_directive_type(failure_class),
        invalidated_stage_ids={"stage_ids": list(invalidated)},
        payload={
            "schema": "RealSaS.RepairDirective.v1",
            "owner_stage_id": owner,
            "failing_stage_id": failing_stage_id,
            "error_code": str(error_code),
        },
        status="OPEN",
    )
    session.add_all([failure, attribution, directive])
    session.add(
        AttemptEventRow(
            attempt_id=attempt_id,
            event_type="FAILURE_LOCALIZED",
            payload={
                "failure_signature_id": str(failure.id),
                "owner_stage_id": owner,
                "repair_directive_id": str(directive.id),
                "invalidated_stage_ids": list(invalidated),
            },
        )
    )
    session.flush()
    return FailureRecord(failure.id, attribution.id, directive.id, owner, invalidated, signature)
