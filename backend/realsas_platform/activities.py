from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from temporalio import activity
from temporalio.exceptions import ApplicationError

from .compiler_bridge import CompilerBridgeRejected, CompilerStageOutcome, execute_bound_compiler_stage
from .diagnostics import record_stage_failure
from .persistence.models import (
    AttemptArtifactRow,
    AttemptEventRow,
    AttemptRow,
    ExecutionRow,
)
from .resolver import CompilePlanResolver
from .services.catalog import PostgresQualifiedArtifactCatalog
from .services.releases import load_stage_versions
from .stage_graph import StageGraph
from .workflows import CompileSubjectCommand


class CompilerExecutor(Protocol):
    def execute(
        self,
        session_factory: sessionmaker[Session],
        *,
        graph: StageGraph,
        attempt_id: UUID,
        stage_id: str,
        allowed_stage_ids: tuple[str, ...],
    ) -> CompilerStageOutcome: ...


class CanonicalCompilerExecutor:
    def execute(
        self,
        session_factory: sessionmaker[Session],
        *,
        graph: StageGraph,
        attempt_id: UUID,
        stage_id: str,
        allowed_stage_ids: tuple[str, ...],
    ) -> CompilerStageOutcome:
        return execute_bound_compiler_stage(
            session_factory,
            graph=graph,
            attempt_id=attempt_id,
            stage_id=stage_id,
            allowed_stage_ids=allowed_stage_ids,
        )


@dataclass(frozen=True)
class StageExecutionFailure:
    execution_id: UUID
    stage_id: str
    error_code: str
    failure_signature_id: UUID
    owner_stage_id: str
    repair_directive_id: UUID


def _next_retry_number(session: Session, *, workflow_id: str, stage_id: str) -> int:
    current = session.execute(
        select(func.max(ExecutionRow.retry_number)).where(
            ExecutionRow.workflow_id == workflow_id,
            ExecutionRow.stage_contract == stage_id,
        )
    ).scalar_one()
    return 0 if current is None else int(current) + 1


def resolve_compile_plan_once(
    session_factory: sessionmaker[Session],
    graph: StageGraph,
    command: CompileSubjectCommand,
) -> dict[str, Any]:
    with session_factory() as session:
        attempt = session.execute(
            select(AttemptRow).where(AttemptRow.id == UUID(command.attempt_id))
        ).scalar_one_or_none()
        if attempt is None or attempt.subject_id != UUID(command.subject_id):
            raise RuntimeError("COMPILE_ATTEMPT_SUBJECT_MISMATCH")
        if attempt.engine_release_id != UUID(command.engine_release_id):
            raise RuntimeError("COMPILE_ATTEMPT_ENGINE_RELEASE_DRIFT")
        versions = load_stage_versions(
            session,
            release_id=UUID(command.engine_release_id),
            graph=graph,
            require_product=True,
        )
        catalog = PostgresQualifiedArtifactCatalog(session)
        plan = CompilePlanResolver(graph, catalog).resolve(
            target_stage_id=command.target_stage_id,
            subject_semantic_sha256=command.subject_semantic_sha256,
            versions=versions,
        )
        return {
            "schema": "RealSaS.ResolvedCompilePlan.v1",
            "target_stage_id": plan.target_stage_id,
            "stages": [
                {
                    "stage_id": row.stage_id,
                    "action": row.action.value,
                    "expected_semantic_sha256": row.expected_semantic_sha256,
                    "reusable_artifact_id": None if row.reusable_artifact_id is None else str(row.reusable_artifact_id),
                    "reason": row.reason,
                }
                for row in plan.stages
            ],
        }


def bind_reused_stage_once(
    session_factory: sessionmaker[Session],
    *,
    attempt_id: UUID,
    stage_id: str,
    reusable_artifact_id: UUID,
) -> dict[str, Any]:
    role = f"stage:{stage_id}"
    with session_factory() as session:
        with session.begin():
            existing = session.execute(
                select(AttemptArtifactRow).where(
                    AttemptArtifactRow.attempt_id == attempt_id,
                    AttemptArtifactRow.role == role,
                )
            ).scalar_one_or_none()
            if existing is not None:
                if existing.artifact_id != reusable_artifact_id:
                    raise RuntimeError("ATTEMPT_REUSE_ARTIFACT_DRIFT")
                return {
                    "stage_id": stage_id,
                    "status": "REUSED",
                    "artifact_id": str(existing.artifact_id),
                }
            session.add(
                AttemptArtifactRow(
                    attempt_id=attempt_id,
                    role=role,
                    artifact_id=reusable_artifact_id,
                    origin="inherited",
                )
            )
            session.add(
                AttemptEventRow(
                    attempt_id=attempt_id,
                    event_type="STAGE_REUSED",
                    payload={
                        "stage_id": stage_id,
                        "artifact_id": str(reusable_artifact_id),
                    },
                )
            )
        return {
            "stage_id": stage_id,
            "status": "REUSED",
            "artifact_id": str(reusable_artifact_id),
        }


def execute_compile_stage_once(
    session_factory: sessionmaker[Session],
    *,
    graph: StageGraph,
    executor: CompilerExecutor,
    attempt_id: UUID,
    command_id: str,
    stage_id: str,
    allowed_stage_ids: tuple[str, ...],
) -> dict[str, Any]:
    workflow_id = f"realsas:compile:{command_id}"
    execution_id = uuid4()

    with session_factory() as session:
        with session.begin():
            attempt = session.execute(
                select(AttemptRow).where(AttemptRow.id == attempt_id)
            ).scalar_one_or_none()
            if attempt is None:
                raise RuntimeError("EXECUTION_ATTEMPT_NOT_FOUND")
            retry = _next_retry_number(
                session,
                workflow_id=workflow_id,
                stage_id=stage_id,
            )
            session.add(
                ExecutionRow(
                    id=execution_id,
                    attempt_id=attempt_id,
                    workflow_id=workflow_id,
                    stage_contract=stage_id,
                    status="RUNNING",
                    retry_number=retry,
                    started_at=datetime.now(timezone.utc),
                )
            )
            session.add(
                AttemptEventRow(
                    attempt_id=attempt_id,
                    event_type="STAGE_EXECUTION_STARTED",
                    payload={
                        "execution_id": str(execution_id),
                        "stage_id": stage_id,
                        "retry_number": retry,
                    },
                )
            )

    try:
        outcome = executor.execute(
            session_factory,
            graph=graph,
            attempt_id=attempt_id,
            stage_id=stage_id,
            allowed_stage_ids=allowed_stage_ids,
        )
    except CompilerBridgeRejected as exc:
        with session_factory() as session:
            with session.begin():
                row = session.execute(
                    select(ExecutionRow).where(ExecutionRow.id == execution_id)
                ).scalar_one()
                row.status = "FAIL"
                row.finished_at = datetime.now(timezone.utc)
                row.error_code = type(exc).__name__
                row.error_payload = {"message": str(exc)}
                failure = record_stage_failure(
                    session,
                    graph=graph,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    failing_stage_id=stage_id,
                    error_code=str(exc).split(":", 1)[0],
                    diagnostics={"message": str(exc), "class": "COMPILER_BRIDGE"},
                    failure_class="INFRA",
                )
        raise ApplicationError(
            str(exc),
            type="CompilerPlatformPlanFailure",
            non_retryable=True,
        ) from exc

    if outcome.status not in {"PASS", "PASS_DEMO_ONLY"}:
        error_code = outcome.blockers[0] if outcome.blockers else f"STAGE_{outcome.status}"
        with session_factory() as session:
            with session.begin():
                row = session.execute(
                    select(ExecutionRow).where(ExecutionRow.id == execution_id)
                ).scalar_one()
                row.status = "FAIL"
                row.finished_at = datetime.now(timezone.utc)
                row.error_code = error_code[:200]
                row.error_payload = {
                    "compiler_status": outcome.status,
                    "blockers": list(outcome.blockers),
                    "diagnostics_hash": outcome.diagnostics_hash,
                    "executed_stage_ids": list(outcome.executed_stage_ids),
                }
                failure = record_stage_failure(
                    session,
                    graph=graph,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    failing_stage_id=stage_id,
                    error_code=error_code,
                    diagnostics=dict(row.error_payload),
                    failure_class="STAGE",
                )
        raise ApplicationError(
            f"{stage_id}:{error_code}",
            type="CompilerStageFailure",
            non_retryable=True,
        )

    with session_factory() as session:
        with session.begin():
            row = session.execute(
                select(ExecutionRow).where(ExecutionRow.id == execution_id)
            ).scalar_one()
            row.status = "PASS"
            row.finished_at = datetime.now(timezone.utc)
            session.add(
                AttemptEventRow(
                    attempt_id=attempt_id,
                    event_type="STAGE_EXECUTION_PASSED",
                    payload={
                        "execution_id": str(execution_id),
                        "stage_id": stage_id,
                        "compiler_status": outcome.status,
                        "executed_stage_ids": list(outcome.executed_stage_ids),
                        "output_count": len(outcome.outputs),
                        "diagnostics_hash": outcome.diagnostics_hash,
                    },
                )
            )
    return {
        "execution_id": str(execution_id),
        "stage_id": stage_id,
        "status": "PASS",
        "compiler_status": outcome.status,
        "executed_stage_ids": list(outcome.executed_stage_ids),
        "outputs": list(outcome.outputs),
        "diagnostics_hash": outcome.diagnostics_hash,
    }


class CompilerActivities:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        graph: StageGraph,
        executor: CompilerExecutor | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.graph = graph
        self.executor = executor or CanonicalCompilerExecutor()

    @activity.defn(name="resolve_compile_plan")
    async def resolve_compile_plan(self, command: CompileSubjectCommand) -> dict[str, Any]:
        return resolve_compile_plan_once(self.session_factory, self.graph, command)

    @activity.defn(name="bind_reused_stage")
    async def bind_reused_stage(self, request: dict[str, Any]) -> dict[str, Any]:
        artifact_id = request["resolution"].get("reusable_artifact_id")
        if not artifact_id:
            raise ApplicationError(
                "REUSE_RESOLUTION_ARTIFACT_REQUIRED",
                type="ReuseResolutionFailure",
                non_retryable=True,
            )
        return bind_reused_stage_once(
            self.session_factory,
            attempt_id=UUID(str(request["attempt_id"])),
            stage_id=str(request["stage_id"]),
            reusable_artifact_id=UUID(str(artifact_id)),
        )

    @activity.defn(name="execute_compile_stage")
    async def execute_compile_stage(self, request: dict[str, Any]) -> dict[str, Any]:
        return execute_compile_stage_once(
            self.session_factory,
            graph=self.graph,
            executor=self.executor,
            attempt_id=UUID(str(request["attempt_id"])),
            command_id=str(request["command_id"]),
            stage_id=str(request["stage_id"]),
            allowed_stage_ids=tuple(map(str, request["allowed_execute_stage_ids"])),
        )
