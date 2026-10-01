from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from compiler.realsas_compiler_services.orchestrator.mainline import (
    PLAN_PATH,
    content_sha256,
    implementation_closure_manifest,
    load_json,
    validate_plan,
)

from .domain import EngineReleaseManifest, StageReleaseBinding
from .resolver import StageVersionIdentity
from .stage_graph import StageGraph


@dataclass(frozen=True)
class StageChange:
    stage_id: str
    implementation_changed: bool
    policy_changed: bool
    parameters_changed: bool


@dataclass(frozen=True)
class ChangeImpact:
    direct_changes: tuple[StageChange, ...]
    invalidated_stage_ids: tuple[str, ...]
    unchanged_stage_ids: tuple[str, ...]


def build_checkout_engine_release(
    *,
    name: str,
    purpose: str = "RESEARCH",
) -> EngineReleaseManifest:
    """Snapshot the exact checkout into a semantic 46-stage EngineRelease.

    This converts source-code changes into stage-level implementation identities
    using the compiler's existing import-closure hash logic.
    """
    plan = load_json(PLAN_PATH)
    validate_plan(plan)
    closure = implementation_closure_manifest(plan)
    impl_by_stage = {
        str(row["stage_id"]): str(row["implementation_hash"])
        for row in closure["adapter_implementation_closures"]
    }
    rows = []
    for stage in sorted(plan["stages"], key=lambda x: int(x["ordinal"])):
        stage_id = str(stage["id"])
        rows.append(
            StageReleaseBinding(
                ordinal=int(stage["ordinal"]),
                stage_id=stage_id,
                implementation_sha256=impl_by_stage[stage_id],
                policy_sha256=content_sha256(stage["policy"]),
                semantic_parameters={
                    "adapter": str(stage["adapter"]),
                    "pipeline_plan_sha256": str(closure["pipeline_plan_sha256"]),
                },
            )
        )
    return EngineReleaseManifest(name=name, purpose=purpose, stages=tuple(rows))


def compare_stage_versions(
    graph: StageGraph,
    baseline: Mapping[str, StageVersionIdentity],
    candidate: Mapping[str, StageVersionIdentity],
) -> ChangeImpact:
    direct: list[StageChange] = []
    roots: list[str] = []
    for stage in graph.stages:
        a = baseline[stage.stage_id]
        b = candidate[stage.stage_id]
        row = StageChange(
            stage_id=stage.stage_id,
            implementation_changed=a.implementation_sha256 != b.implementation_sha256,
            policy_changed=a.policy_sha256 != b.policy_sha256,
            parameters_changed=dict(a.semantic_parameters) != dict(b.semantic_parameters),
        )
        if row.implementation_changed or row.policy_changed or row.parameters_changed:
            direct.append(row)
            roots.append(stage.stage_id)
    invalidated = graph.descendants_including(roots) if roots else ()
    invalid = set(invalidated)
    unchanged = tuple(s.stage_id for s in graph.stages if s.stage_id not in invalid)
    return ChangeImpact(tuple(direct), tuple(invalidated), unchanged)


@dataclass(frozen=True)
class ImpactReason:
    stage_id: str
    kind: str
    roots: tuple[str, ...]


def explain_change_impact(
    graph: StageGraph,
    impact: ChangeImpact,
) -> tuple[ImpactReason, ...]:
    direct = {x.stage_id for x in impact.direct_changes}
    reasons: list[ImpactReason] = []
    for stage_id in impact.invalidated_stage_ids:
        if stage_id in direct:
            reasons.append(ImpactReason(stage_id, "DIRECT_CHANGE", (stage_id,)))
            continue
        roots = tuple(
            root
            for root in sorted(direct)
            if stage_id in set(graph.descendants_including([root]))
        )
        reasons.append(ImpactReason(stage_id, "DEPENDENCY_DESCENDANT", roots))
    return tuple(reasons)
