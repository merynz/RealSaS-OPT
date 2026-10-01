from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..persistence.models import (
    AttemptEventRow,
    AttemptRow,
    EngineReleaseRow,
    FailureSignatureRow,
    OwnerAttributionRow,
    RepairDirectiveRow,
)
from ..stage_graph import StageGraph


@dataclass(frozen=True)
class OpenFailureExplanation:
    failure_signature_id: UUID
    failing_stage_id: str
    error_code: str
    owner_stage_id: str
    directive_type: str
    invalidated_stage_ids: tuple[str, ...]
    diagnostics: dict


@dataclass(frozen=True)
class AttemptExplanation:
    attempt_id: UUID
    subject_id: UUID
    attempt_kind: str
    final_state: str
    engine_release_id: UUID | None
    engine_release_sha256: str | None
    direct_changed_stage_ids: tuple[str, ...]
    invalidated_by_change: tuple[str, ...]
    unchanged_stage_ids: tuple[str, ...]
    open_failures: tuple[OpenFailureExplanation, ...]
    recommended_resume_roots: tuple[str, ...]
    recommended_resume_scope: tuple[str, ...]


def _minimal_roots(graph: StageGraph, roots: set[str]) -> tuple[str, ...]:
    minimal=[]
    for candidate in sorted(roots, key=lambda x: graph.get(x).ordinal):
        if any(candidate in set(graph.descendants_including([prior])) for prior in minimal):
            continue
        minimal.append(candidate)
    return tuple(minimal)


def explain_attempt(
    session: Session,
    *,
    graph: StageGraph,
    attempt_id: UUID,
) -> AttemptExplanation:
    attempt=session.execute(
        select(AttemptRow).where(AttemptRow.id==attempt_id)
    ).scalar_one_or_none()
    if attempt is None:
        raise KeyError("ATTEMPT_NOT_FOUND")

    release_sha=None
    if attempt.engine_release_id is not None:
        release_sha=session.execute(
            select(EngineReleaseRow.release_sha256).where(
                EngineReleaseRow.id==attempt.engine_release_id
            )
        ).scalar_one_or_none()

    change_event=session.execute(
        select(AttemptEventRow)
        .where(
            AttemptEventRow.attempt_id==attempt_id,
            AttemptEventRow.event_type=="CODE_CHANGE_IMPACT_RESOLVED",
        )
        .order_by(AttemptEventRow.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    change_payload={} if change_event is None else dict(change_event.payload)
    direct=tuple(map(str,change_payload.get("direct_changed_stage_ids") or ()))
    invalidated=tuple(map(str,change_payload.get("invalidated_stage_ids") or ()))
    unchanged=tuple(map(str,change_payload.get("unchanged_stage_ids") or ()))

    rows=session.execute(
        select(FailureSignatureRow,OwnerAttributionRow,RepairDirectiveRow)
        .join(OwnerAttributionRow,OwnerAttributionRow.failure_signature_id==FailureSignatureRow.id)
        .join(RepairDirectiveRow,RepairDirectiveRow.failure_signature_id==FailureSignatureRow.id)
        .where(
            FailureSignatureRow.attempt_id==attempt_id,
            RepairDirectiveRow.status=="OPEN",
        )
        .order_by(FailureSignatureRow.created_at,FailureSignatureRow.id)
    ).all()
    failures=[]
    failure_roots=set()
    for failure,owner,directive in rows:
        stage_ids=tuple(map(str,(directive.invalidated_stage_ids or {}).get("stage_ids") or ()))
        failures.append(OpenFailureExplanation(
            failure_signature_id=failure.id,
            failing_stage_id=failure.stage_id,
            error_code=failure.code,
            owner_stage_id=owner.owner_stage_id,
            directive_type=directive.directive_type,
            invalidated_stage_ids=stage_ids,
            diagnostics=dict(failure.payload.get("diagnostics") or {}),
        ))
        failure_roots.add(owner.owner_stage_id)

    roots=_minimal_roots(graph,failure_roots or set(direct))
    scope=set()
    for root in roots:
        scope.update(graph.descendants_including([root]))
    ordered_scope=tuple(s.stage_id for s in graph.stages if s.stage_id in scope)

    return AttemptExplanation(
        attempt_id=attempt.id,
        subject_id=attempt.subject_id,
        attempt_kind=attempt.kind,
        final_state=attempt.final_state,
        engine_release_id=attempt.engine_release_id,
        engine_release_sha256=release_sha,
        direct_changed_stage_ids=direct,
        invalidated_by_change=invalidated,
        unchanged_stage_ids=unchanged,
        open_failures=tuple(failures),
        recommended_resume_roots=roots,
        recommended_resume_scope=ordered_scope,
    )
