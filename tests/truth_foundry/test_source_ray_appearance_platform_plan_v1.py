from __future__ import annotations

from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.stage_inputs import released_execution_plan
from tools.platform_release_snapshot import snapshot
from tools.platform_source_ray_appearance_admission_v1 import CAPABILITY_ID, MANIFEST_KEY, STAGE_ID, admission_plan


def test_research_release_round_trips_shipped_appearance_admission_stage():
    plan = admission_plan()
    assert plan["stage_count"] == 1
    stage = plan["stages"][0]
    assert stage["id"] == STAGE_ID
    assert stage["manifest_keys"] == [MANIFEST_KEY]
    assert stage["policy"]["fail_closed"] is True
    assert stage["policy"]["cacheable"] is False
    assert stage["policy"]["product_pass_authority"] is False
    assert CAPABILITY_ID == "compiler.stage/" + STAGE_ID

    request = snapshot(
        plan,
        name="source-ray-appearance-platform-admission-source-test",
        purpose="RESEARCH",
        created_by="source-test",
    )
    reconstructed = released_execution_plan(
        mainline.load_json(mainline.PLAN_PATH),
        request["manifest"]["dag"],
        mode="RESEARCH",
    )
    assert mainline.content_sha256(reconstructed) == mainline.content_sha256(plan)
    assert request["manifest"]["pipeline_plan_sha256"] == mainline.content_sha256(plan)
