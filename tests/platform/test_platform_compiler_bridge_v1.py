import pytest

from backend.realsas_platform.compiler_bridge import CompilerBridgeRejected, classify_execution_delta


def test_compiler_bridge_accepts_only_platform_planned_execution_delta():
    before={"34":1,"35":1,"36":0}
    after={"34":1,"35":2,"36":1}
    assert classify_execution_delta(
        before=before,
        after=after,
        allowed_stage_ids=("35","36"),
    )==("35","36")


def test_compiler_bridge_fails_closed_when_compiler_executes_unplanned_stage():
    with pytest.raises(CompilerBridgeRejected,match="COMPILER_PLATFORM_PLAN_DRIFT:34"):
        classify_execution_delta(
            before={"34":1,"35":1},
            after={"34":2,"35":2},
            allowed_stage_ids=("35",),
        )
