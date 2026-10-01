"""Add immutable engine release snapshots.

Revision ID: 0003_engine_releases
Revises: 0002_outbox_leasing
Create Date: 2026-10-01
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_engine_releases"
down_revision = "0002_outbox_leasing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "engine_releases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(300), nullable=False),
        sa.Column("release_sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("purpose", sa.String(20), nullable=False),
        sa.Column("created_by", sa.String(300), nullable=False),
        sa.Column("sealed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("purpose IN ('PRODUCT','RESEARCH')", name="ck_engine_release_purpose"),
    )
    op.create_table(
        "engine_release_stages",
        sa.Column("release_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("engine_releases.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("stage_id", sa.String(100), primary_key=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("implementation_sha256", sa.String(64), nullable=False),
        sa.Column("policy_sha256", sa.String(64), nullable=False),
        sa.Column("semantic_parameters", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint("release_id", "ordinal", name="uq_engine_release_stage_ordinal"),
    )


def downgrade() -> None:
    op.drop_table("engine_release_stages")
    op.drop_table("engine_releases")
