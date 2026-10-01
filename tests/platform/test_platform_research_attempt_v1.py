from __future__ import annotations

from pathlib import Path
import os
from uuid import uuid4

import pytest
from sqlalchemy import select

from backend.realsas_platform.domain import EngineReleaseManifest, StageReleaseBinding
from backend.realsas_platform.persistence.models import AttemptEventRow, AttemptRow, SubjectRow
from backend.realsas_platform.persistence.session import create_platform_engine, create_session_factory
from backend.realsas_platform.services.releases import seal_engine_release
from backend.realsas_platform.services.research import start_research_attempt
from backend.realsas_platform.stage_graph import StageGraph

DB_URL=os.environ.get("REALSAS_PLATFORM_DATABASE_URL")
pytestmark=pytest.mark.skipif(not DB_URL, reason="PostgreSQL integration URL not configured")


def _manifest(graph, name, purpose, stage35_impl):
    return EngineReleaseManifest(
        name=name,
        purpose=purpose,
        stages=tuple(
            StageReleaseBinding(
                ordinal=s.ordinal,
                stage_id=s.stage_id,
                implementation_sha256=stage35_impl if s.stage_id=="35_DYNAMIC_MECHANICAL_MESH_QUALIFIED" else "1"*64,
                policy_sha256="2"*64,
                semantic_parameters={},
            )
            for s in graph.stages
        ),
    )


def test_research_attempt_records_exact_change_impact_and_release():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    engine=create_platform_engine(DB_URL); Session=create_session_factory(engine)
    subject_id=uuid4()
    with Session() as session:
        with session.begin():
            session.add(SubjectRow(id=subject_id,slug="research-impact",display_name="Research Impact"))
        baseline=seal_engine_release(session,manifest=_manifest(graph,"baseline","PRODUCT","1"*64),graph=graph,created_by="ci")
        candidate=seal_engine_release(session,manifest=_manifest(graph,"candidate","RESEARCH","3"*64),graph=graph,created_by="ci")
        started=start_research_attempt(
            session,
            graph=graph,
            subject_id=subject_id,
            baseline_engine_release_id=baseline.release_id,
            candidate_engine_release_id=candidate.release_id,
            created_by="ci",
        )
        assert tuple(x.stage_id for x in started.impact.direct_changes)==("35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",)
        assert started.impact.invalidated_stage_ids==graph.descendants_including(["35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"])
        row=session.execute(select(AttemptRow).where(AttemptRow.id==started.attempt_id)).scalar_one()
        assert row.engine_release_id==candidate.release_id
        session.rollback()
        event=session.execute(select(AttemptEventRow).where(AttemptEventRow.attempt_id==started.attempt_id)).scalar_one()
        assert event.payload["direct_changed_stage_ids"]==["35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"]
        assert "10_IRIS_FIT" in event.payload["unchanged_stage_ids"]
    engine.dispose()
