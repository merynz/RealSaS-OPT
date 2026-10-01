"""Bind attempts to immutable engine releases.

Revision ID: 0005_attempt_engine_release
Revises: 0004_failure_localization
Create Date: 2026-10-01
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_attempt_engine_release"
down_revision = "0004_failure_localization"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "attempts",
        sa.Column("engine_release_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_attempts_engine_release",
        "attempts",
        "engine_releases",
        ["engine_release_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_attempts_subject_release_created",
        "attempts",
        ["subject_id", "engine_release_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_attempts_subject_release_created", table_name="attempts")
    op.drop_constraint("fk_attempts_engine_release", "attempts", type_="foreignkey")
    op.drop_column("attempts", "engine_release_id")
