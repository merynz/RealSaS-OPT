from copy import deepcopy

import pytest

from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.stage_inputs import released_execution_plan
from tools.platform_release_snapshot import snapshot


def test_stage48_research_release_roundtrips_exact_plan_level_metadata():
    plan = mainline.load_json(mainline.ROOT / "canonical" / "TRUTH_CALIBRATION_EXECUTION_PLAN_V3.json")
    request = snapshot(
        plan,
        name="stage48-plan-metadata-regression",
        purpose="RESEARCH",
        created_by="ci",
    )
    dag = request["manifest"]["dag"]
    assert dag["plan_metadata"] == {
        key: value for key, value in plan.items() if key not in {"stage_count", "stages"}
    }
    restored = released_execution_plan(
        mainline.load_json(mainline.PLAN_PATH), dag, mode="RESEARCH"
    )
    assert restored == plan
    assert mainline.content_sha256(restored) == request["manifest"]["pipeline_plan_sha256"]


def test_release_plan_metadata_cannot_smuggle_stage_arrays():
    base = mainline.load_json(mainline.PLAN_PATH)
    plan = deepcopy(base)
    plan["stages"] = plan["stages"][:2]
    plan["stage_count"] = 2
    request = snapshot(plan, name="metadata-guard", purpose="RESEARCH", created_by="ci")
    dag = deepcopy(request["manifest"]["dag"])
    dag["plan_metadata"]["stages"] = []
    with pytest.raises(RuntimeError, match="PLAN_METADATA_STAGE_OVERRIDE_FORBIDDEN"):
        released_execution_plan(base, dag, mode="RESEARCH")
