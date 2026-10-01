"""Add compiler failure localization and repair ownership.

Revision ID: 0004_failure_localization
Revises: 0003_engine_releases
Create Date: 2026-10-01
"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision="0004_failure_localization"
down_revision="0003_engine_releases"
branch_labels=None
depends_on=None


def upgrade() -> None:
    op.create_table(
        "failure_signatures",
        sa.Column("id",postgresql.UUID(as_uuid=True),primary_key=True),
        sa.Column("attempt_id",postgresql.UUID(as_uuid=True),sa.ForeignKey("attempts.id",ondelete="RESTRICT"),nullable=False),
        sa.Column("execution_id",postgresql.UUID(as_uuid=True),sa.ForeignKey("executions.id",ondelete="RESTRICT"),nullable=True),
        sa.Column("stage_id",sa.String(100),nullable=False),
        sa.Column("code",sa.String(200),nullable=False),
        sa.Column("severity",sa.String(20),nullable=False),
        sa.Column("signature_sha256",sa.String(64),nullable=False),
        sa.Column("payload",postgresql.JSONB(),nullable=False),
        sa.Column("created_at",sa.DateTime(timezone=True),server_default=sa.text("now()"),nullable=False),
        sa.UniqueConstraint("attempt_id","signature_sha256",name="uq_attempt_failure_signature"),
    )
    op.create_table(
        "owner_attributions",
        sa.Column("id",postgresql.UUID(as_uuid=True),primary_key=True),
        sa.Column("failure_signature_id",postgresql.UUID(as_uuid=True),sa.ForeignKey("failure_signatures.id",ondelete="RESTRICT"),nullable=False),
        sa.Column("owner_stage_id",sa.String(100),nullable=False),
        sa.Column("owner_kind",sa.String(30),nullable=False),
        sa.Column("reason",sa.Text(),nullable=False),
        sa.Column("created_at",sa.DateTime(timezone=True),server_default=sa.text("now()"),nullable=False),
    )
    op.create_table(
        "repair_directives",
        sa.Column("id",postgresql.UUID(as_uuid=True),primary_key=True),
        sa.Column("failure_signature_id",postgresql.UUID(as_uuid=True),sa.ForeignKey("failure_signatures.id",ondelete="RESTRICT"),nullable=False),
        sa.Column("owner_stage_id",sa.String(100),nullable=False),
        sa.Column("directive_type",sa.String(50),nullable=False),
        sa.Column("invalidated_stage_ids",postgresql.JSONB(),nullable=False),
        sa.Column("payload",postgresql.JSONB(),nullable=False),
        sa.Column("status",sa.String(20),nullable=False),
        sa.Column("created_at",sa.DateTime(timezone=True),server_default=sa.text("now()"),nullable=False),
        sa.CheckConstraint("status IN ('OPEN','APPLIED','SUPERSEDED')",name="ck_repair_directive_status"),
    )


def downgrade() -> None:
    op.drop_table("repair_directives")
    op.drop_table("owner_attributions")
    op.drop_table("failure_signatures")
