from pathlib import Path
import os

import pytest

from backend.realsas_platform.domain import EngineReleaseManifest, StageReleaseBinding
from backend.realsas_platform.persistence.session import create_platform_engine, create_session_factory
from backend.realsas_platform.services.releases import load_stage_versions, seal_engine_release
from backend.realsas_platform.stage_graph import StageGraph

DB_URL=os.environ.get("REALSAS_PLATFORM_DATABASE_URL")
pytestmark=pytest.mark.skipif(not DB_URL, reason="PostgreSQL integration URL not configured")


def _manifest(graph, purpose="PRODUCT"):
    return EngineReleaseManifest(
        name="ci-release",
        purpose=purpose,
        stages=tuple(
            StageReleaseBinding(
                ordinal=s.ordinal,
                stage_id=s.stage_id,
                implementation_sha256=f"{s.ordinal:064x}",
                policy_sha256=f"{s.ordinal+100:064x}",
                semantic_parameters={"contract_epoch":1},
            )
            for s in graph.stages
        ),
    )


def test_engine_release_is_exact_46_stage_immutable_snapshot():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    engine=create_platform_engine(DB_URL); Session=create_session_factory(engine)
    with Session() as session:
        sealed=seal_engine_release(session,manifest=_manifest(graph),graph=graph,created_by="ci")
        reused=seal_engine_release(session,manifest=_manifest(graph),graph=graph,created_by="ci")
        assert reused.reused is True and reused.release_id==sealed.release_id
        versions=load_stage_versions(session,release_id=sealed.release_id,graph=graph,require_product=True)
        assert tuple(versions)==tuple(s.stage_id for s in graph.stages)
        assert versions["42_RUNTIME_PROJECTION_AND_CAA_BINDING"].semantic_parameters["contract_epoch"]==1
    engine.dispose()


def test_product_compile_rejects_research_release():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    engine=create_platform_engine(DB_URL); Session=create_session_factory(engine)
    with Session() as session:
        sealed=seal_engine_release(session,manifest=_manifest(graph,purpose="RESEARCH"),graph=graph,created_by="ci")
        session.rollback()
        from backend.realsas_platform.services.releases import EngineReleaseRejected
        with pytest.raises(EngineReleaseRejected,match="PRODUCT_COMPILE_REQUIRES_PRODUCT_ENGINE_RELEASE"):
            load_stage_versions(session,release_id=sealed.release_id,graph=graph,require_product=True)
    engine.dispose()
