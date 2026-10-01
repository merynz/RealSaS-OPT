"""Add durable outbox leasing.

Revision ID: 0002_outbox_leasing
Revises: 0001_product_state
Create Date: 2026-10-01
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_outbox_leasing"
down_revision = "0001_product_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("outbox_events", sa.Column("claim_token", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("outbox_events", sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("outbox_events", sa.Column("available_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False))
    op.add_column("outbox_events", sa.Column("last_error", sa.Text(), nullable=True))
    op.create_index(
        "ix_outbox_delivery_scan",
        "outbox_events",
        ["delivered_at", "available_at", "claimed_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_outbox_delivery_scan", table_name="outbox_events")
    op.drop_column("outbox_events", "last_error")
    op.drop_column("outbox_events", "available_at")
    op.drop_column("outbox_events", "claimed_at")
    op.drop_column("outbox_events", "claim_token")
