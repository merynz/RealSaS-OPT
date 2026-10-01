from pathlib import Path
from uuid import uuid4

from backend.realsas_platform.resolver import (
    CompilePlanResolver,
    ResolutionAction,
    StageVersionIdentity,
)
from backend.realsas_platform.stage_graph import StageGraph


class MemoryCatalog:
    def __init__(self):
        self.rows={}

    def find_qualified(self, *, artifact_type, schema_version, semantic_sha256):
        return self.rows.get((artifact_type,schema_version,semantic_sha256))

    def admit_plan(self, plan):
        for row in plan.stages:
            self.rows[("RealSaS.StageResultManifest","v1",row.expected_semantic_sha256)]=uuid4()


def versions(graph, *, stage42_impl="1"*64):
    rows={}
    for stage in graph.stages:
        rows[stage.stage_id]=StageVersionIdentity(
            implementation_sha256=stage42_impl if stage.stage_id=="42_RUNTIME_PROJECTION_AND_CAA_BINDING" else "1"*64,
            policy_sha256="2"*64,
            semantic_parameters={"contract_epoch":1},
        )
    return rows


def test_identical_compile_reuses_every_qualified_stage_result():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    catalog=MemoryCatalog(); resolver=CompilePlanResolver(graph,catalog)
    first=resolver.resolve(target_stage_id="46_PRODUCT_CLOSURE_SEAL",subject_semantic_sha256="a"*64,versions=versions(graph))
    assert all(x.action==ResolutionAction.EXECUTE for x in first.stages)
    catalog.admit_plan(first)
    second=resolver.resolve(target_stage_id="46_PRODUCT_CLOSURE_SEAL",subject_semantic_sha256="a"*64,versions=versions(graph))
    assert second.execute_stage_ids==()
    required=graph.ancestors_including(["46_PRODUCT_CLOSURE_SEAL"])
    assert second.reused_stage_ids==required


def test_stage42_implementation_change_invalidates_only_true_descendants():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    catalog=MemoryCatalog(); resolver=CompilePlanResolver(graph,catalog)
    baseline=resolver.resolve(target_stage_id="46_PRODUCT_CLOSURE_SEAL",subject_semantic_sha256="a"*64,versions=versions(graph))
    catalog.admit_plan(baseline)
    changed=resolver.resolve(target_stage_id="46_PRODUCT_CLOSURE_SEAL",subject_semantic_sha256="a"*64,versions=versions(graph,stage42_impl="3"*64))
    assert changed.execute_stage_ids==(
        "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
        "43_RSS_MATERIALIZE_COMPACT",
        "44_NATIVE_PACKAGE_OPEN_PLAYBACK",
        "45_DYNAMIC_VISUAL_INTEGRITY_PROOF",
        "46_PRODUCT_CLOSURE_SEAL",
    )
    assert "41_MOTION_DYNAMIC_PROOF" in changed.reused_stage_ids
    assert "10_IRIS_FIT" in changed.reused_stage_ids


def test_subject_identity_change_invalidates_entire_compile_graph():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    catalog=MemoryCatalog(); resolver=CompilePlanResolver(graph,catalog)
    baseline=resolver.resolve(target_stage_id="46_PRODUCT_CLOSURE_SEAL",subject_semantic_sha256="a"*64,versions=versions(graph))
    catalog.admit_plan(baseline)
    changed=resolver.resolve(target_stage_id="46_PRODUCT_CLOSURE_SEAL",subject_semantic_sha256="b"*64,versions=versions(graph))
    required=graph.ancestors_including(["46_PRODUCT_CLOSURE_SEAL"])
    assert changed.execute_stage_ids==required
