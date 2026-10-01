from __future__ import annotations

import asyncio
from pathlib import Path
import os
from uuid import uuid4

import pytest
from sqlalchemy import select

from backend.realsas_platform.domain import EngineReleaseManifest, StageReleaseBinding
from backend.realsas_platform.persistence.models import (
    ArtifactRow, ArtifactTypeRow, CommandRow, OutboxEventRow, ProductRevisionRow,
    SubjectRow, AttemptRow,
)
from backend.realsas_platform.persistence.session import create_platform_engine, create_session_factory
from backend.realsas_platform.services.commands import submit_compile_subject, submit_render_product
from backend.realsas_platform.services.releases import seal_engine_release
from backend.realsas_platform.stage_graph import StageGraph
from backend.realsas_platform.temporal_dispatch import PublishResult, dispatch_outbox_once

DB_URL=os.environ.get("REALSAS_PLATFORM_DATABASE_URL")
pytestmark=pytest.mark.skipif(not DB_URL, reason="PostgreSQL integration URL not configured")


class FakePublisher:
    def __init__(self):
        self.calls=[]

    async def publish(self, command_type, payload):
        self.calls.append((command_type,dict(payload)))
        command_id=str(payload["command_id"])
        prefix="compile" if command_type=="COMPILE_SUBJECT" else "render"
        return PublishResult(f"realsas:{prefix}:{command_id}",False)


def _release(graph):
    return EngineReleaseManifest(
        name="command-ci-release",purpose="PRODUCT",
        stages=tuple(StageReleaseBinding(s.ordinal,s.stage_id,f"{s.ordinal:064x}",f"{s.ordinal+200:064x}",{}) for s in graph.stages),
    )


def test_compile_command_is_transactional_idempotent_and_dispatchable():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    engine=create_platform_engine(DB_URL); Session=create_session_factory(engine)
    subject_id=uuid4()
    with Session() as session:
        with session.begin():
            session.add(SubjectRow(id=subject_id,slug="command-compile",display_name="Command Compile"))
        release=seal_engine_release(session,manifest=_release(graph),graph=graph,created_by="ci")
        first=submit_compile_subject(session,graph=graph,subject_id=subject_id,engine_release_id=release.release_id,subject_semantic_sha256="a"*64,idempotency_key="compile-once",requested_by="ci")
        same=submit_compile_subject(session,graph=graph,subject_id=subject_id,engine_release_id=release.release_id,subject_semantic_sha256="a"*64,idempotency_key="compile-once",requested_by="ci")
        assert same.reused_idempotency_key is True
        assert same.command_id==first.command_id and same.attempt_id==first.attempt_id
    publisher=FakePublisher()
    assert asyncio.run(dispatch_outbox_once(Session,publisher))==1
    assert publisher.calls[0][0]=="COMPILE_SUBJECT"
    with Session() as session:
        event=session.execute(select(OutboxEventRow).where(OutboxEventRow.aggregate_id==first.command_id)).scalar_one()
        assert event.delivered_at is not None
    engine.dispose()


def test_render_command_binds_exact_revision_and_reuses_semantic_render_request():
    engine=create_platform_engine(DB_URL); Session=create_session_factory(engine)
    subject_id=uuid4(); attempt_id=uuid4(); revision_id=uuid4(); motion_id=uuid4()
    with Session() as session:
        with session.begin():
            session.add(SubjectRow(id=subject_id,slug="command-render",display_name="Command Render")); session.flush()
            session.add(AttemptRow(id=attempt_id,subject_id=subject_id,kind="compile_candidate",spec_sha256="1"*64,created_by="ci",final_state="QUALIFIED"))
            at=ArtifactTypeRow(id=uuid4(),name="RealSaS.Motion.v1",schema_version="v1",domain="motion"); session.add(at); session.flush()
            session.add(ArtifactRow(id=motion_id,artifact_type_id=at.id,semantic_sha256="2"*64,content_sha256="3"*64,storage_key="cas/motion",size_bytes=1,producer_contract="40_MOTION_COMPILE_RUN",implementation_sha256="4"*64,policy_sha256="5"*64,semantic_parameters={},verified_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc)))
            session.add(ProductRevisionRow(id=revision_id,subject_id=subject_id,revision_number=1,manifest_sha256="6"*64,created_from_attempt_id=attempt_id,sealed_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc)))
        first=submit_render_product(session,subject_id=subject_id,product_revision_id=revision_id,motion_artifact_id=motion_id,view_spec={"view":"front"},render_settings={"fps":24},idempotency_key="render-1",requested_by="ci")
        second=submit_render_product(session,subject_id=subject_id,product_revision_id=revision_id,motion_artifact_id=motion_id,view_spec={"view":"front"},render_settings={"fps":24},idempotency_key="render-2",requested_by="ci")
        assert first.render_request_id==second.render_request_id
    publisher=FakePublisher()
    assert asyncio.run(dispatch_outbox_once(Session,publisher))==2
    assert [x[0] for x in publisher.calls]==["RENDER_PRODUCT","RENDER_PRODUCT"]
    engine.dispose()
