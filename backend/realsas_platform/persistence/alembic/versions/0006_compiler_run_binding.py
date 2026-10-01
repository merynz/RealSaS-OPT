"""Bind platform attempts to existing canonical compiler runs.

Revision ID: 0006_compiler_run_binding
Revises: 0005_attempt_engine_release
Create Date: 2026-10-01
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006_compiler_run_binding"
down_revision = "0005_attempt_engine_release"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "compiler_run_bindings",
        sa.Column("attempt_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("attempts.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("compiler_run_id", sa.String(300), nullable=False, unique=True),
        sa.Column("run_manifest_path", sa.Text(), nullable=False),
        sa.Column("run_ledger_path", sa.Text(), nullable=False),
        sa.Column("pipeline_plan_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("compiler_run_bindings")
