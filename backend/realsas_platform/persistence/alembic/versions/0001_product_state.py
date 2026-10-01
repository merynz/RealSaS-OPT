"""Initial RealSaS professional product-state schema.

Revision ID: 0001_product_state
Revises:
Create Date: 2026-10-01
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_product_state"
down_revision = None
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB()


def upgrade() -> None:
    op.create_table(
        "subjects",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("slug", sa.String(200), nullable=False, unique=True),
        sa.Column("display_name", sa.String(300), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "artifact_types",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("name", sa.String(300), nullable=False),
        sa.Column("schema_version", sa.String(100), nullable=False),
        sa.Column("domain", sa.String(100), nullable=False),
        sa.UniqueConstraint("name", "schema_version", name="uq_artifact_type_schema"),
    )
    op.create_table(
        "artifacts",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("artifact_type_id", UUID, sa.ForeignKey("artifact_types.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("semantic_sha256", sa.String(64), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("producer_contract", sa.String(300), nullable=False),
        sa.Column("implementation_sha256", sa.String(64), nullable=False),
        sa.Column("policy_sha256", sa.String(64), nullable=False),
        sa.Column("semantic_parameters", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("artifact_type_id", "semantic_sha256", name="uq_artifact_semantic_identity"),
        sa.UniqueConstraint("content_sha256", "storage_key", name="uq_artifact_content_storage"),
        sa.CheckConstraint("size_bytes >= 0", name="ck_artifact_size_nonnegative"),
    )
    op.create_table(
        "artifact_inputs",
        sa.Column("artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("input_role", sa.String(200), primary_key=True),
        sa.Column("ordinal", sa.Integer(), primary_key=True),
        sa.Column("input_artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False),
    )
    op.create_table(
        "attempts",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("subject_id", UUID, sa.ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("parent_attempt_id", UUID, sa.ForeignKey("attempts.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("spec_sha256", sa.String(64), nullable=False),
        sa.Column("created_by", sa.String(300), nullable=False),
        sa.Column("final_state", sa.String(50), server_default=sa.text("'OPEN'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("kind IN ('research','repair','compile_candidate')", name="ck_attempt_kind"),
    )
    op.create_table(
        "attempt_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("attempt_id", UUID, sa.ForeignKey("attempts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "attempt_artifacts",
        sa.Column("attempt_id", UUID, sa.ForeignKey("attempts.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("role", sa.String(150), primary_key=True),
        sa.Column("artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("origin", sa.String(20), nullable=False),
        sa.CheckConstraint("origin IN ('inherited','produced')", name="ck_attempt_artifact_origin"),
    )
    op.create_table(
        "executions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("attempt_id", UUID, sa.ForeignKey("attempts.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("workflow_id", sa.String(300), nullable=False),
        sa.Column("workflow_run_id", sa.String(300), nullable=True),
        sa.Column("stage_contract", sa.String(200), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("worker_identity", sa.String(300), nullable=True),
        sa.Column("retry_number", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(200), nullable=True),
        sa.Column("error_payload", JSONB, nullable=True),
        sa.UniqueConstraint("workflow_id", "stage_contract", "retry_number", name="uq_execution_retry"),
    )
    op.create_table(
        "execution_artifacts",
        sa.Column("execution_id", UUID, sa.ForeignKey("executions.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("relation", sa.String(20), primary_key=True),
        sa.Column("role", sa.String(150), primary_key=True),
        sa.Column("artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), primary_key=True),
        sa.CheckConstraint("relation IN ('input','output')", name="ck_execution_artifact_relation"),
    )
    op.create_table(
        "proofs",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("proof_type", sa.String(200), nullable=False),
        sa.Column("subject_artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("policy_sha256", sa.String(64), nullable=False),
        sa.Column("implementation_sha256", sa.String(64), nullable=False),
        sa.Column("result", sa.String(20), nullable=False),
        sa.Column("report_artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("result IN ('PASS','FAIL','ABSTAIN')", name="ck_proof_result"),
    )
    op.create_table(
        "qualifications",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("qualification_type", sa.String(200), nullable=False),
        sa.Column("result", sa.String(20), nullable=False),
        sa.Column("proof_id", UUID, sa.ForeignKey("proofs.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("artifact_id", "qualification_type", "proof_id", name="uq_artifact_qualification_proof"),
        sa.CheckConstraint("result IN ('PASS','FAIL','ABSTAIN')", name="ck_qualification_result"),
    )
    op.create_table(
        "product_revisions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("subject_id", UUID, sa.ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("revision_number", sa.BigInteger(), nullable=False),
        sa.Column("manifest_sha256", sa.String(64), nullable=False),
        sa.Column("created_from_attempt_id", UUID, sa.ForeignKey("attempts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("sealed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("subject_id", "revision_number", name="uq_product_revision_number"),
        sa.UniqueConstraint("subject_id", "manifest_sha256", name="uq_product_revision_manifest"),
    )
    op.create_table(
        "product_revision_artifacts",
        sa.Column("product_revision_id", UUID, sa.ForeignKey("product_revisions.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("role", sa.String(150), primary_key=True),
        sa.Column("artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False),
    )
    op.create_table(
        "promotions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("subject_id", UUID, sa.ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("from_revision_id", UUID, sa.ForeignKey("product_revisions.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("to_revision_id", UUID, sa.ForeignKey("product_revisions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("requested_by", sa.String(300), nullable=False),
        sa.Column("qualification_snapshot", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "subject_current_revision",
        sa.Column("subject_id", UUID, sa.ForeignKey("subjects.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("product_revision_id", UUID, sa.ForeignKey("product_revisions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("lock_version", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "render_requests",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("subject_id", UUID, sa.ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("product_revision_id", UUID, sa.ForeignKey("product_revisions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("motion_artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("view_spec", JSONB, nullable=False),
        sa.Column("render_settings", JSONB, nullable=False),
        sa.Column("semantic_sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("idempotency_key", sa.String(200), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "render_outputs",
        sa.Column("render_request_id", UUID, sa.ForeignKey("render_requests.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "commands",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("command_type", sa.String(100), nullable=False),
        sa.Column("subject_id", UUID, sa.ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("idempotency_key", sa.String(200), nullable=False, unique=True),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "outbox_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("aggregate_type", sa.String(100), nullable=False),
        sa.Column("aggregate_id", UUID, nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )
    op.create_table(
        "audit_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("actor", sa.String(300), nullable=False),
        sa.Column("action", sa.String(150), nullable=False),
        sa.Column("subject_id", UUID, nullable=True),
        sa.Column("attempt_id", UUID, nullable=True),
        sa.Column("product_revision_id", UUID, nullable=True),
        sa.Column("artifact_id", UUID, nullable=True),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "audit_events", "outbox_events", "commands", "render_outputs", "render_requests",
        "subject_current_revision", "promotions", "product_revision_artifacts", "product_revisions",
        "qualifications", "proofs", "execution_artifacts", "executions", "attempt_artifacts",
        "attempt_events", "attempts", "artifact_inputs", "artifacts", "artifact_types", "subjects",
    ):
        op.drop_table(table)
