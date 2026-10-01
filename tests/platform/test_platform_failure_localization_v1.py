from __future__ import annotations

from pathlib import Path
import os
from uuid import uuid4

import pytest

from backend.realsas_platform.diagnostics import record_stage_failure
from backend.realsas_platform.persistence.models import AttemptRow, SubjectRow
from backend.realsas_platform.persistence.session import create_platform_engine, create_session_factory
from backend.realsas_platform.stage_graph import StageGraph

DB_URL=os.environ.get("REALSAS_PLATFORM_DATABASE_URL")
pytestmark=pytest.mark.skipif(not DB_URL, reason="PostgreSQL integration URL not configured")


def test_failure_can_be_attributed_only_to_self_or_true_ancestor():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    engine=create_platform_engine(DB_URL); Session=create_session_factory(engine)
    subject_id=uuid4(); attempt_id=uuid4()
    with Session() as session:
        with session.begin():
            session.add(SubjectRow(id=subject_id,slug="failure-owner",display_name="Failure Owner")); session.flush()
            session.add(AttemptRow(id=attempt_id,subject_id=subject_id,kind="research",spec_sha256="a"*64,created_by="ci",final_state="OPEN"))
        with session.begin():
            record=record_stage_failure(
                session,
                graph=graph,
                attempt_id=attempt_id,
                failing_stage_id="37_QUALIFIED_PRESENTATION_STRUCTURE",
                reported_owner_stage_id="35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
                error_code="VISUAL_BINDING_MECHANICAL_INCOMPATIBILITY",
                diagnostics={"face":17},
                failure_class="CODE",
            )
        assert record.owner_stage_id=="35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"
        assert record.invalidated_stage_ids==graph.descendants_including(["35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"])
        with session.begin():
            with pytest.raises(ValueError,match="ILLEGAL_FAILURE_OWNER"):
                record_stage_failure(
                    session,
                    graph=graph,
                    attempt_id=attempt_id,
                    failing_stage_id="35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
                    reported_owner_stage_id="42_RUNTIME_PROJECTION_AND_CAA_BINDING",
                    error_code="BAD_BLAME",
                    diagnostics={},
                )
    engine.dispose()
