from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..domain import RenderRequestSpec, require_sha256, sha256_json
from ..persistence.models import (
    ArtifactRow,
    AttemptRow,
    AuditEventRow,
    CommandRow,
    EngineReleaseRow,
    OutboxEventRow,
    ProductRevisionRow,
    RenderRequestRow,
    SubjectRow,
)
from ..stage_graph import StageGraph


COMPILE_SUBJECT = "COMPILE_SUBJECT"
RENDER_PRODUCT = "RENDER_PRODUCT"


class CommandRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandReceipt:
    command_id: UUID
    command_type: str
    subject_id: UUID
    reused_idempotency_key: bool
    attempt_id: UUID | None = None
    render_request_id: UUID | None = None


def _require_clean_session(session: Session) -> None:
    if session.in_transaction():
        raise RuntimeError("COMMAND_SERVICE_REQUIRES_CLEAN_SESSION")


def _existing_by_idempotency(
    session: Session,
    *,
    idempotency_key: str,
    command_type: str,
    subject_id: UUID,
) -> CommandRow | None:
    row = session.execute(
        select(CommandRow).where(CommandRow.idempotency_key == idempotency_key)
    ).scalar_one_or_none()
    if row is None:
        return None
    if row.command_type != command_type or row.subject_id != subject_id:
        raise CommandRejected("IDEMPOTENCY_KEY_COMMAND_IDENTITY_CONFLICT")
    return row


def submit_compile_subject(
    session: Session,
    *,
    graph: StageGraph,
    subject_id: UUID,
    engine_release_id: UUID,
    subject_semantic_sha256: str,
    idempotency_key: str,
    requested_by: str,
    target_stage_id: str = "46_PRODUCT_CLOSURE_SEAL",
) -> CommandReceipt:
    _require_clean_session(session)
    require_sha256(subject_semantic_sha256, "subject_semantic_sha256")
    graph.get(target_stage_id)
    if not idempotency_key or not requested_by:
        raise ValueError("idempotency_key and requested_by are required")

    with session.begin():
        existing = _existing_by_idempotency(
            session,
            idempotency_key=idempotency_key,
            command_type=COMPILE_SUBJECT,
            subject_id=subject_id,
        )
        if existing is not None:
            return CommandReceipt(
                command_id=existing.id,
                command_type=existing.command_type,
                subject_id=subject_id,
                reused_idempotency_key=True,
                attempt_id=UUID(str(existing.payload["attempt_id"])),
            )

        if session.execute(select(SubjectRow.id).where(SubjectRow.id == subject_id)).scalar_one_or_none() is None:
            raise CommandRejected("COMPILE_SUBJECT_NOT_FOUND")
        release = session.execute(
            select(EngineReleaseRow).where(EngineReleaseRow.id == engine_release_id)
        ).scalar_one_or_none()
        if release is None:
            raise CommandRejected("COMPILE_ENGINE_RELEASE_NOT_FOUND")
        if release.purpose != "PRODUCT":
            raise CommandRejected("COMPILE_PRODUCT_RELEASE_REQUIRED")

        spec = {
            "schema": "RealSaS.CompileSubjectCommandSpec.v1",
            "subject_id": str(subject_id),
            "engine_release_id": str(engine_release_id),
            "engine_release_sha256": release.release_sha256,
            "subject_semantic_sha256": subject_semantic_sha256,
            "target_stage_id": target_stage_id,
        }
        attempt_id = uuid4()
        command_id = uuid4()
        session.add(
            AttemptRow(
                id=attempt_id,
                subject_id=subject_id,
                engine_release_id=engine_release_id,
                kind="compile_candidate",
                spec_sha256=sha256_json(spec),
                created_by=requested_by,
                final_state="OPEN",
            )
        )
        session.add(
            CommandRow(
                id=command_id,
                command_type=COMPILE_SUBJECT,
                subject_id=subject_id,
                idempotency_key=idempotency_key,
                payload={
                    **spec,
                    "attempt_id": str(attempt_id),
                    "command_id": str(command_id),
                },
            )
        )
        session.add(
            OutboxEventRow(
                aggregate_type="Command",
                aggregate_id=command_id,
                event_type="CompileSubjectRequested",
                payload={"command_id": str(command_id)},
            )
        )
        session.add(
            AuditEventRow(
                actor=requested_by,
                action="COMPILE_COMMAND_ACCEPTED",
                subject_id=subject_id,
                attempt_id=attempt_id,
                payload={
                    "command_id": str(command_id),
                    "engine_release_id": str(engine_release_id),
                    "target_stage_id": target_stage_id,
                },
            )
        )
        return CommandReceipt(command_id, COMPILE_SUBJECT, subject_id, False, attempt_id=attempt_id)


def submit_render_product(
    session: Session,
    *,
    subject_id: UUID,
    product_revision_id: UUID,
    motion_artifact_id: UUID,
    view_spec: dict,
    render_settings: dict,
    idempotency_key: str,
    requested_by: str,
) -> CommandReceipt:
    _require_clean_session(session)
    if not idempotency_key or not requested_by:
        raise ValueError("idempotency_key and requested_by are required")

    with session.begin():
        existing = _existing_by_idempotency(
            session,
            idempotency_key=idempotency_key,
            command_type=RENDER_PRODUCT,
            subject_id=subject_id,
        )
        if existing is not None:
            return CommandReceipt(
                command_id=existing.id,
                command_type=existing.command_type,
                subject_id=subject_id,
                reused_idempotency_key=True,
                render_request_id=UUID(str(existing.payload["render_request_id"])),
            )

        revision = session.execute(
            select(ProductRevisionRow).where(ProductRevisionRow.id == product_revision_id)
        ).scalar_one_or_none()
        if revision is None or revision.subject_id != subject_id:
            raise CommandRejected("RENDER_PRODUCT_REVISION_SUBJECT_MISMATCH")
        motion = session.execute(
            select(ArtifactRow).where(ArtifactRow.id == motion_artifact_id)
        ).scalar_one_or_none()
        if motion is None:
            raise CommandRejected("RENDER_MOTION_ARTIFACT_NOT_FOUND")

        spec = RenderRequestSpec(
            product_revision_manifest_sha256=revision.manifest_sha256,
            motion_artifact_semantic_sha256=motion.semantic_sha256,
            view_spec=view_spec,
            render_settings=render_settings,
        )
        render = session.execute(
            select(RenderRequestRow).where(RenderRequestRow.semantic_sha256 == spec.semantic_sha256)
        ).scalar_one_or_none()
        if render is None:
            render = RenderRequestRow(
                id=uuid4(),
                subject_id=subject_id,
                product_revision_id=product_revision_id,
                motion_artifact_id=motion_artifact_id,
                view_spec=dict(view_spec),
                render_settings=dict(render_settings),
                semantic_sha256=spec.semantic_sha256,
                idempotency_key="render:" + spec.semantic_sha256,
            )
            session.add(render)
            session.flush()

        command_id = uuid4()
        session.add(
            CommandRow(
                id=command_id,
                command_type=RENDER_PRODUCT,
                subject_id=subject_id,
                idempotency_key=idempotency_key,
                payload={
                    "schema": "RealSaS.RenderProductCommandSpec.v1",
                    "command_id": str(command_id),
                    "render_request_id": str(render.id),
                    "product_revision_id": str(product_revision_id),
                    "render_request_semantic_sha256": spec.semantic_sha256,
                },
            )
        )
        session.add(
            OutboxEventRow(
                aggregate_type="Command",
                aggregate_id=command_id,
                event_type="RenderProductRequested",
                payload={"command_id": str(command_id)},
            )
        )
        session.add(
            AuditEventRow(
                actor=requested_by,
                action="RENDER_COMMAND_ACCEPTED",
                subject_id=subject_id,
                product_revision_id=product_revision_id,
                payload={
                    "command_id": str(command_id),
                    "render_request_id": str(render.id),
                    "render_request_semantic_sha256": spec.semantic_sha256,
                },
            )
        )
        return CommandReceipt(command_id, RENDER_PRODUCT, subject_id, False, render_request_id=render.id)
