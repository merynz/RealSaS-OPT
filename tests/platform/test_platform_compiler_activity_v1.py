from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from backend.realsas_platform.activities import CompilerExecutor, execute_compile_stage_once
from backend.realsas_platform.compiler_bridge import CompilerStageOutcome
from backend.realsas_platform.persistence.models import AttemptRow, ExecutionRow, FailureSignatureRow, SubjectRow
from backend.realsas_platform.persistence.session import create_platform_engine, create_session_factory
from backend.realsas_platform.stage_graph import StageGraph

DB_URL=os.environ.get("REALSAS_PLATFORM_DATABASE_URL")
pytestmark=pytest.mark.skipif(not DB_URL, reason="PostgreSQL integration URL not configured")


class FakeFailingExecutor:
    def execute(self, session_factory, *, graph, attempt_id, stage_id, allowed_stage_ids):
        return CompilerStageOutcome(
            stage_id=stage_id,
            status="FAIL",
            executed_stage_ids=(stage_id,),
            outputs=(),
            blockers=("MECHANICAL_ENVELOPE_VIOLATION",),
            diagnostics_hash="d"*64,
        )


class FakePassingExecutor:
    def execute(self, session_factory, *, graph, attempt_id, stage_id, allowed_stage_ids):
        return CompilerStageOutcome(
            stage_id=stage_id,
            status="PASS",
            executed_stage_ids=(stage_id,),
            outputs=({"path":"/tmp/example","sha256":"a"*64,"bytes":1,"schema":"RealSaS.Test.v1","authority_class":"TEST"},),
            blockers=(),
            diagnostics_hash="e"*64,
        )


def _attempt(Session):
    subject_id=uuid4(); attempt_id=uuid4()
    with Session() as session:
        with session.begin():
            session.add(SubjectRow(id=subject_id,slug="activity-"+attempt_id.hex,display_name="Activity")); session.flush()
            session.add(AttemptRow(id=attempt_id,subject_id=subject_id,kind="research",spec_sha256="a"*64,created_by="ci",final_state="OPEN"))
    return attempt_id


def test_stage_failure_persists_execution_and_failure_signature_before_raising():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    engine=create_platform_engine(DB_URL); Session=create_session_factory(engine)
    attempt_id=_attempt(Session)
    from temporalio.exceptions import ApplicationError
    with pytest.raises(ApplicationError,match="MECHANICAL_ENVELOPE_VIOLATION"):
        execute_compile_stage_once(
            Session,graph=graph,executor=FakeFailingExecutor(),attempt_id=attempt_id,
            command_id="cmd-fail",stage_id="35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
            allowed_stage_ids=graph.descendants_including(["35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"]),
        )
    with Session() as session:
        execution=session.execute(select(ExecutionRow).where(ExecutionRow.attempt_id==attempt_id)).scalar_one()
        failure=session.execute(select(FailureSignatureRow).where(FailureSignatureRow.attempt_id==attempt_id)).scalar_one()
        assert execution.status=="FAIL"
        assert failure.stage_id=="35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"
        assert failure.code=="MECHANICAL_ENVELOPE_VIOLATION"
    engine.dispose()


def test_stage_pass_persists_execution_pass():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    engine=create_platform_engine(DB_URL); Session=create_session_factory(engine)
    attempt_id=_attempt(Session)
    result=execute_compile_stage_once(
        Session,graph=graph,executor=FakePassingExecutor(),attempt_id=attempt_id,
        command_id="cmd-pass",stage_id="35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
        allowed_stage_ids=graph.descendants_including(["35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"]),
    )
    assert result["status"]=="PASS"
    with Session() as session:
        execution=session.execute(select(ExecutionRow).where(ExecutionRow.attempt_id==attempt_id)).scalar_one()
        assert execution.status=="PASS"
    engine.dispose()
