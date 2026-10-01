from pathlib import Path
from backend.realsas_platform.developer import ChangeImpact, StageChange, explain_change_impact
from backend.realsas_platform.stage_graph import StageGraph


def test_change_impact_explains_direct_and_downstream_invalidations():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    root="35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"
    impact=ChangeImpact(
        direct_changes=(StageChange(root,True,False,False),),
        invalidated_stage_ids=graph.descendants_including([root]),
        unchanged_stage_ids=tuple(s.stage_id for s in graph.stages if s.stage_id not in set(graph.descendants_including([root]))),
    )
    reasons=explain_change_impact(graph,impact)
    assert reasons[0].stage_id==root and reasons[0].kind=="DIRECT_CHANGE"
    by_stage={r.stage_id:r for r in reasons}
    assert by_stage["42_RUNTIME_PROJECTION_AND_CAA_BINDING"].kind=="DEPENDENCY_DESCENDANT"
    assert by_stage["42_RUNTIME_PROJECTION_AND_CAA_BINDING"].roots==(root,)
