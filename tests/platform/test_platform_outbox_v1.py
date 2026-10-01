from __future__ import annotations

from datetime import timedelta
import os
from uuid import uuid4

import pytest
from sqlalchemy import select

from backend.realsas_platform.persistence.models import OutboxEventRow
from backend.realsas_platform.persistence.session import create_platform_engine, create_session_factory
from backend.realsas_platform.services.outbox import acknowledge_outbox_event, claim_outbox_batch, fail_outbox_event

DB_URL=os.environ.get("REALSAS_PLATFORM_DATABASE_URL")
pytestmark=pytest.mark.skipif(not DB_URL, reason="PostgreSQL integration URL not configured")


def test_outbox_claim_ack_and_retry_are_durable():
    engine=create_platform_engine(DB_URL); Session=create_session_factory(engine)
    first_id=None; second_id=None
    with Session() as session:
        with session.begin():
            first=OutboxEventRow(aggregate_type="Command",aggregate_id=uuid4(),event_type="CompileRequested",payload={"command_id":"a"})
            second=OutboxEventRow(aggregate_type="Command",aggregate_id=uuid4(),event_type="RenderRequested",payload={"command_id":"b"})
            session.add_all([first,second]); session.flush(); first_id=first.id; second_id=second.id
        claims=claim_outbox_batch(session,limit=2)
        assert [c.event_id for c in claims]==[first_id,second_id]
        acknowledge_outbox_event(session,claims[0])
        fail_outbox_event(session,claims[1],error="temporary",retry_after=timedelta(seconds=0))
        retry=claim_outbox_batch(session,limit=10)
        assert len(retry)==1 and retry[0].event_id==second_id and retry[0].delivery_attempt==2
        acknowledge_outbox_event(session,retry[0])
        assert claim_outbox_batch(session,limit=10)==()
        delivered=session.execute(select(OutboxEventRow).where(OutboxEventRow.id.in_([first_id,second_id]))).scalars().all()
        assert all(x.delivered_at is not None for x in delivered)
    engine.dispose()
