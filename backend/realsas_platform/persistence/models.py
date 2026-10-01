from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SubjectRow(Base):
    __tablename__ = "subjects"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(300), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ArtifactTypeRow(Base):
    __tablename__ = "artifact_types"
    __table_args__ = (UniqueConstraint("name", "schema_version", name="uq_artifact_type_schema"),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(100), nullable=False)
    domain: Mapped[str] = mapped_column(String(100), nullable=False)


class ArtifactRow(Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint("artifact_type_id", "semantic_sha256", name="uq_artifact_semantic_identity"),
        UniqueConstraint("content_sha256", "storage_key", name="uq_artifact_content_storage"),
        CheckConstraint("size_bytes >= 0", name="ck_artifact_size_nonnegative"),
    )
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    artifact_type_id: Mapped[UUID] = mapped_column(ForeignKey("artifact_types.id", ondelete="RESTRICT"), nullable=False)
    semantic_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    producer_contract: Mapped[str] = mapped_column(String(300), nullable=False)
    implementation_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    semantic_parameters: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ArtifactInputRow(Base):
    __tablename__ = "artifact_inputs"
    artifact_id: Mapped[UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), primary_key=True)
    input_role: Mapped[str] = mapped_column(String(200), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    input_artifact_id: Mapped[UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False)


class AttemptRow(Base):
    __tablename__ = "attempts"
    __table_args__ = (CheckConstraint("kind IN ('research','repair','compile_candidate')", name="ck_attempt_kind"),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    subject_id: Mapped[UUID] = mapped_column(ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False)
    parent_attempt_id: Mapped[UUID | None] = mapped_column(ForeignKey("attempts.id", ondelete="RESTRICT"))
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    spec_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_by: Mapped[str] = mapped_column(String(300), nullable=False)
    final_state: Mapped[str] = mapped_column(String(50), nullable=False, default="OPEN")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AttemptEventRow(Base):
    __tablename__ = "attempt_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    attempt_id: Mapped[UUID] = mapped_column(ForeignKey("attempts.id", ondelete="RESTRICT"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AttemptArtifactRow(Base):
    __tablename__ = "attempt_artifacts"
    __table_args__ = (CheckConstraint("origin IN ('inherited','produced')", name="ck_attempt_artifact_origin"),)
    attempt_id: Mapped[UUID] = mapped_column(ForeignKey("attempts.id", ondelete="RESTRICT"), primary_key=True)
    role: Mapped[str] = mapped_column(String(150), primary_key=True)
    artifact_id: Mapped[UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False)
    origin: Mapped[str] = mapped_column(String(20), nullable=False)


class ExecutionRow(Base):
    __tablename__ = "executions"
    __table_args__ = (UniqueConstraint("workflow_id", "stage_contract", "retry_number", name="uq_execution_retry"),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    attempt_id: Mapped[UUID | None] = mapped_column(ForeignKey("attempts.id", ondelete="RESTRICT"))
    workflow_id: Mapped[str] = mapped_column(String(300), nullable=False)
    workflow_run_id: Mapped[str | None] = mapped_column(String(300))
    stage_contract: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    worker_identity: Mapped[str | None] = mapped_column(String(300))
    retry_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(200))
    error_payload: Mapped[dict | None] = mapped_column(JSONB)


class ExecutionArtifactRow(Base):
    __tablename__ = "execution_artifacts"
    execution_id: Mapped[UUID] = mapped_column(ForeignKey("executions.id", ondelete="RESTRICT"), primary_key=True)
    relation: Mapped[str] = mapped_column(String(20), primary_key=True)
    role: Mapped[str] = mapped_column(String(150), primary_key=True)
    artifact_id: Mapped[UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), primary_key=True)
    __table_args__ = (CheckConstraint("relation IN ('input','output')", name="ck_execution_artifact_relation"),)


class ProofRow(Base):
    __tablename__ = "proofs"
    __table_args__ = (CheckConstraint("result IN ('PASS','FAIL','ABSTAIN')", name="ck_proof_result"),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    proof_type: Mapped[str] = mapped_column(String(200), nullable=False)
    subject_artifact_id: Mapped[UUID | None] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"))
    policy_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    implementation_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    result: Mapped[str] = mapped_column(String(20), nullable=False)
    report_artifact_id: Mapped[UUID | None] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class QualificationRow(Base):
    __tablename__ = "qualifications"
    __table_args__ = (
        UniqueConstraint("artifact_id", "qualification_type", "proof_id", name="uq_artifact_qualification_proof"),
        CheckConstraint("result IN ('PASS','FAIL','ABSTAIN')", name="ck_qualification_result"),
    )
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    artifact_id: Mapped[UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False)
    qualification_type: Mapped[str] = mapped_column(String(200), nullable=False)
    result: Mapped[str] = mapped_column(String(20), nullable=False)
    proof_id: Mapped[UUID | None] = mapped_column(ForeignKey("proofs.id", ondelete="RESTRICT"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ProductRevisionRow(Base):
    __tablename__ = "product_revisions"
    __table_args__ = (
        UniqueConstraint("subject_id", "revision_number", name="uq_product_revision_number"),
        UniqueConstraint("subject_id", "manifest_sha256", name="uq_product_revision_manifest"),
    )
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    subject_id: Mapped[UUID] = mapped_column(ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False)
    revision_number: Mapped[int] = mapped_column(BigInteger, nullable=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_from_attempt_id: Mapped[UUID] = mapped_column(ForeignKey("attempts.id", ondelete="RESTRICT"), nullable=False)
    sealed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProductRevisionArtifactRow(Base):
    __tablename__ = "product_revision_artifacts"
    product_revision_id: Mapped[UUID] = mapped_column(ForeignKey("product_revisions.id", ondelete="RESTRICT"), primary_key=True)
    role: Mapped[str] = mapped_column(String(150), primary_key=True)
    artifact_id: Mapped[UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False)


class PromotionRow(Base):
    __tablename__ = "promotions"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    subject_id: Mapped[UUID] = mapped_column(ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False)
    from_revision_id: Mapped[UUID | None] = mapped_column(ForeignKey("product_revisions.id", ondelete="RESTRICT"))
    to_revision_id: Mapped[UUID] = mapped_column(ForeignKey("product_revisions.id", ondelete="RESTRICT"), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    requested_by: Mapped[str] = mapped_column(String(300), nullable=False)
    qualification_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class SubjectCurrentRevisionRow(Base):
    __tablename__ = "subject_current_revision"
    subject_id: Mapped[UUID] = mapped_column(ForeignKey("subjects.id", ondelete="RESTRICT"), primary_key=True)
    product_revision_id: Mapped[UUID] = mapped_column(ForeignKey("product_revisions.id", ondelete="RESTRICT"), nullable=False)
    lock_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class RenderRequestRow(Base):
    __tablename__ = "render_requests"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    subject_id: Mapped[UUID] = mapped_column(ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False)
    product_revision_id: Mapped[UUID] = mapped_column(ForeignKey("product_revisions.id", ondelete="RESTRICT"), nullable=False)
    motion_artifact_id: Mapped[UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False)
    view_spec: Mapped[dict] = mapped_column(JSONB, nullable=False)
    render_settings: Mapped[dict] = mapped_column(JSONB, nullable=False)
    semantic_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class RenderOutputRow(Base):
    __tablename__ = "render_outputs"
    render_request_id: Mapped[UUID] = mapped_column(ForeignKey("render_requests.id", ondelete="RESTRICT"), primary_key=True)
    artifact_id: Mapped[UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class CommandRow(Base):
    __tablename__ = "commands"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    command_type: Mapped[str] = mapped_column(String(100), nullable=False)
    subject_id: Mapped[UUID | None] = mapped_column(ForeignKey("subjects.id", ondelete="RESTRICT"))
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class OutboxEventRow(Base):
    __tablename__ = "outbox_events"
    __table_args__ = (
        Index("ix_outbox_delivery_scan", "delivered_at", "available_at", "claimed_at", "id"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    aggregate_type: Mapped[str] = mapped_column(String(100), nullable=False)
    aggregate_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivery_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    claim_token: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)


class AuditEventRow(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(300), nullable=False)
    action: Mapped[str] = mapped_column(String(150), nullable=False)
    subject_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    attempt_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    product_revision_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    artifact_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
