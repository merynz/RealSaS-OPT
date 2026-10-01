from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..persistence.models import OutboxEventRow


@dataclass(frozen=True)
class ClaimedOutboxEvent:
    event_id: int
    claim_token: UUID
    aggregate_type: str
    aggregate_id: UUID
    event_type: str
    payload: dict
    delivery_attempt: int


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def claim_outbox_batch(
    session: Session,
    *,
    limit: int = 50,
    stale_after: timedelta = timedelta(minutes=5),
) -> tuple[ClaimedOutboxEvent, ...]:
    if limit < 1 or limit > 500:
        raise ValueError("outbox claim limit must be 1..500")
    now = _utcnow()
    stale_before = now - stale_after
    claimed: list[ClaimedOutboxEvent] = []
    with session.begin():
        rows = session.execute(
            select(OutboxEventRow)
            .where(
                OutboxEventRow.delivered_at.is_(None),
                OutboxEventRow.available_at <= now,
                or_(OutboxEventRow.claimed_at.is_(None), OutboxEventRow.claimed_at < stale_before),
            )
            .order_by(OutboxEventRow.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).scalars().all()
        for row in rows:
            token = uuid4()
            row.claim_token = token
            row.claimed_at = now
            row.delivery_attempts += 1
            claimed.append(
                ClaimedOutboxEvent(
                    event_id=row.id,
                    claim_token=token,
                    aggregate_type=row.aggregate_type,
                    aggregate_id=row.aggregate_id,
                    event_type=row.event_type,
                    payload=dict(row.payload),
                    delivery_attempt=row.delivery_attempts,
                )
            )
    return tuple(claimed)


def acknowledge_outbox_event(session: Session, claim: ClaimedOutboxEvent) -> None:
    with session.begin():
        row = session.execute(
            select(OutboxEventRow)
            .where(OutboxEventRow.id == claim.event_id)
            .with_for_update()
        ).scalar_one()
        if row.delivered_at is not None:
            return
        if row.claim_token != claim.claim_token:
            raise RuntimeError("OUTBOX_CLAIM_TOKEN_DRIFT")
        row.delivered_at = _utcnow()
        row.claim_token = None
        row.claimed_at = None
        row.last_error = None


def fail_outbox_event(
    session: Session,
    claim: ClaimedOutboxEvent,
    *,
    error: str,
    retry_after: timedelta,
) -> None:
    if retry_after.total_seconds() < 0:
        raise ValueError("retry_after must be non-negative")
    with session.begin():
        row = session.execute(
            select(OutboxEventRow)
            .where(OutboxEventRow.id == claim.event_id)
            .with_for_update()
        ).scalar_one()
        if row.delivered_at is not None:
            return
        if row.claim_token != claim.claim_token:
            raise RuntimeError("OUTBOX_CLAIM_TOKEN_DRIFT")
        row.last_error = error[:4000]
        row.available_at = _utcnow() + retry_after
        row.claim_token = None
        row.claimed_at = None
