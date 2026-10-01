from pathlib import Path
from uuid import uuid4

import pytest

from backend.realsas_platform.developer import compare_stage_versions
from backend.realsas_platform.resolver import StageVersionIdentity
from backend.realsas_platform.stage_graph import StageGraph
from backend.realsas_platform.workflows import compile_activity_sequence


def _versions(graph, changed=None):
    changed=changed or {}
    return {
        s.stage_id: StageVersionIdentity(
            implementation_sha256=changed.get(s.stage_id,"1"*64),
            policy_sha256="2"*64,
            semantic_parameters={},
        )
        for s in graph.stages
    }


def test_developer_change_is_localized_to_direct_owner_and_true_descendants():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    impact=compare_stage_versions(graph,_versions(graph),_versions(graph,{"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED":"3"*64}))
    assert tuple(x.stage_id for x in impact.direct_changes)==("35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",)
    assert impact.invalidated_stage_ids==graph.descendants_including(["35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"])
    assert "10_IRIS_FIT" in impact.unchanged_stage_ids


def test_compile_workflow_has_one_durable_activity_per_resolved_stage():
    seq=compile_activity_sequence({"stages":[
        {"stage_id":"34_DEFORMATION_CAPABILITY_ENVELOPE","action":"REUSE"},
        {"stage_id":"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED","action":"EXECUTE"},
        {"stage_id":"36_QUALIFIED_MESH_SKIN_TRANSFER","action":"EXECUTE"},
    ]})
    assert seq==(
        ("34_DEFORMATION_CAPABILITY_ENVELOPE","bind_reused_stage"),
        ("35_DYNAMIC_MECHANICAL_MESH_QUALIFIED","execute_compile_stage"),
        ("36_QUALIFIED_MESH_SKIN_TRANSFER","execute_compile_stage"),
    )
    with pytest.raises(ValueError,match="UNKNOWN_RESOLUTION_ACTION"):
        compile_activity_sequence({"stages":[{"stage_id":"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED","action":"MAYBE"}]})
