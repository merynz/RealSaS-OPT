from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from temporalio.client import Client
from temporalio.exceptions import WorkflowAlreadyStartedError

from .persistence.models import CommandRow
from .services.outbox import ClaimedOutboxEvent, acknowledge_outbox_event, claim_outbox_batch, fail_outbox_event
from .workflows import CompileSubjectCommand, CompileSubjectWorkflow, RenderCommand, RenderWorkflow


@dataclass(frozen=True)
class PublishResult:
    workflow_id: str
    already_started: bool


class CommandPublisher(Protocol):
    async def publish(self, command_type: str, payload: dict[str, Any]) -> PublishResult: ...


class TemporalCommandPublisher:
    def __init__(
        self,
        client: Client,
        *,
        compile_task_queue: str = "realsas-compiler",
        render_task_queue: str = "realsas-runtime",
    ) -> None:
        self.client = client
        self.compile_task_queue = compile_task_queue
        self.render_task_queue = render_task_queue

    async def publish(self, command_type: str, payload: dict[str, Any]) -> PublishResult:
        command_id = str(payload["command_id"])
        if command_type == "COMPILE_SUBJECT":
            command = CompileSubjectCommand(
                command_id=command_id,
                subject_id=str(payload["subject_id"]),
                attempt_id=str(payload["attempt_id"]),
                engine_release_id=str(payload["engine_release_id"]),
                subject_semantic_sha256=str(payload["subject_semantic_sha256"]),
                target_stage_id=str(payload["target_stage_id"]),
            )
            workflow_id = f"realsas:compile:{command_id}"
            workflow = CompileSubjectWorkflow.run
            queue = self.compile_task_queue
        elif command_type == "RENDER_PRODUCT":
            command = RenderCommand(
                command_id=command_id,
                subject_id=str(payload["subject_id"]),
                product_revision_id=str(payload["product_revision_id"]),
                render_request_id=str(payload["render_request_id"]),
                render_request_semantic_sha256=str(payload["render_request_semantic_sha256"]),
            )
            workflow_id = f"realsas:render:{command_id}"
            workflow = RenderWorkflow.run
            queue = self.render_task_queue
        else:
            raise RuntimeError("UNSUPPORTED_COMMAND_TYPE:" + command_type)

        try:
            await self.client.start_workflow(
                workflow,
                command,
                id=workflow_id,
                task_queue=queue,
            )
        except WorkflowAlreadyStartedError:
            return PublishResult(workflow_id=workflow_id, already_started=True)
        return PublishResult(workflow_id=workflow_id, already_started=False)


def _load_command(session: Session, claim: ClaimedOutboxEvent) -> tuple[str, dict[str, Any]]:
    if claim.aggregate_type != "Command":
        raise RuntimeError("OUTBOX_AGGREGATE_NOT_COMMAND")
    row = session.execute(
        select(CommandRow).where(CommandRow.id == claim.aggregate_id)
    ).scalar_one_or_none()
    if row is None:
        raise RuntimeError("OUTBOX_COMMAND_NOT_FOUND")
    return row.command_type, dict(row.payload)


async def dispatch_outbox_once(
    session_factory: sessionmaker[Session],
    publisher: CommandPublisher,
    *,
    limit: int = 50,
) -> int:
    with session_factory() as session:
        claims = claim_outbox_batch(session, limit=limit)
    delivered = 0
    for claim in claims:
        try:
            with session_factory() as session:
                command_type, payload = _load_command(session, claim)
            await publisher.publish(command_type, payload)
            with session_factory() as session:
                acknowledge_outbox_event(session, claim)
            delivered += 1
        except Exception as exc:
            with session_factory() as session:
                fail_outbox_event(
                    session,
                    claim,
                    error=f"{type(exc).__name__}:{exc}",
                    retry_after=timedelta(seconds=min(300, 2 ** min(claim.delivery_attempt, 8))),
                )
    return delivered
