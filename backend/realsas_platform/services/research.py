from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..developer import ChangeImpact, compare_stage_versions
from ..domain import sha256_json
from ..persistence.models import AttemptEventRow, AttemptRow, EngineReleaseRow, SubjectRow
from ..services.releases import load_stage_versions
from ..stage_graph import StageGraph


class ResearchAttemptRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class ResearchAttemptStart:
    attempt_id: UUID
    candidate_engine_release_id: UUID
    baseline_engine_release_id: UUID
    impact: ChangeImpact


def start_research_attempt(
    session: Session,
    *,
    graph: StageGraph,
    subject_id: UUID,
    baseline_engine_release_id: UUID,
    candidate_engine_release_id: UUID,
    created_by: str,
    parent_attempt_id: UUID | None = None,
) -> ResearchAttemptStart:
    if session.in_transaction():
        raise RuntimeError("RESEARCH_ATTEMPT_SERVICE_REQUIRES_CLEAN_SESSION")
    if not created_by:
        raise ValueError("created_by is required")

    with session.begin():
        if session.execute(select(SubjectRow.id).where(SubjectRow.id == subject_id)).scalar_one_or_none() is None:
            raise ResearchAttemptRejected("RESEARCH_SUBJECT_NOT_FOUND")

        baseline_release = session.execute(
            select(EngineReleaseRow).where(EngineReleaseRow.id == baseline_engine_release_id)
        ).scalar_one_or_none()
        candidate_release = session.execute(
            select(EngineReleaseRow).where(EngineReleaseRow.id == candidate_engine_release_id)
        ).scalar_one_or_none()
        if baseline_release is None or candidate_release is None:
            raise ResearchAttemptRejected("RESEARCH_ENGINE_RELEASE_NOT_FOUND")
        if candidate_release.purpose != "RESEARCH":
            raise ResearchAttemptRejected("RESEARCH_ATTEMPT_REQUIRES_RESEARCH_RELEASE")
        if parent_attempt_id is not None:
            parent = session.execute(
                select(AttemptRow).where(AttemptRow.id == parent_attempt_id)
            ).scalar_one_or_none()
            if parent is None or parent.subject_id != subject_id:
                raise ResearchAttemptRejected("RESEARCH_PARENT_ATTEMPT_SUBJECT_MISMATCH")

        baseline = load_stage_versions(
            session,
            release_id=baseline_engine_release_id,
            graph=graph,
            require_product=False,
        )
        candidate = load_stage_versions(
            session,
            release_id=candidate_engine_release_id,
            graph=graph,
            require_product=False,
        )
        impact = compare_stage_versions(graph, baseline, candidate)
        spec = {
            "schema": "RealSaS.ResearchAttemptSpec.v1",
            "subject_id": str(subject_id),
            "parent_attempt_id": None if parent_attempt_id is None else str(parent_attempt_id),
            "baseline_engine_release_id": str(baseline_engine_release_id),
            "baseline_release_sha256": baseline_release.release_sha256,
            "candidate_engine_release_id": str(candidate_engine_release_id),
            "candidate_release_sha256": candidate_release.release_sha256,
            "direct_changed_stage_ids": [x.stage_id for x in impact.direct_changes],
            "invalidated_stage_ids": list(impact.invalidated_stage_ids),
        }
        attempt_id = uuid4()
        session.add(
            AttemptRow(
                id=attempt_id,
                subject_id=subject_id,
                engine_release_id=candidate_engine_release_id,
                parent_attempt_id=parent_attempt_id,
                kind="research",
                spec_sha256=sha256_json(spec),
                created_by=created_by,
                final_state="OPEN",
            )
        )
        session.flush()
        session.add(
            AttemptEventRow(
                attempt_id=attempt_id,
                event_type="CODE_CHANGE_IMPACT_RESOLVED",
                payload={
                    **spec,
                    "direct_changes": [
                        {
                            "stage_id": x.stage_id,
                            "implementation_changed": x.implementation_changed,
                            "policy_changed": x.policy_changed,
                            "parameters_changed": x.parameters_changed,
                        }
                        for x in impact.direct_changes
                    ],
                    "unchanged_stage_ids": list(impact.unchanged_stage_ids),
                },
            )
        )
        return ResearchAttemptStart(
            attempt_id=attempt_id,
            candidate_engine_release_id=candidate_engine_release_id,
            baseline_engine_release_id=baseline_engine_release_id,
            impact=impact,
        )
