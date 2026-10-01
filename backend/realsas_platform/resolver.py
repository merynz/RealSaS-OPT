from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Protocol
from uuid import UUID

from .domain import ArtifactInputIdentity, ArtifactSemanticDescriptor
from .stage_graph import StageGraph


STAGE_RESULT_ARTIFACT_TYPE = "RealSaS.StageResultManifest"
STAGE_RESULT_SCHEMA_VERSION = "v1"


class ResolutionAction(StrEnum):
    REUSE = "REUSE"
    EXECUTE = "EXECUTE"


@dataclass(frozen=True)
class StageVersionIdentity:
    implementation_sha256: str
    policy_sha256: str
    semantic_parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedStage:
    stage_id: str
    action: ResolutionAction
    expected_semantic_sha256: str
    reusable_artifact_id: UUID | None
    reason: str


@dataclass(frozen=True)
class ResolvedCompilePlan:
    target_stage_id: str
    stages: tuple[ResolvedStage, ...]

    @property
    def execute_stage_ids(self) -> tuple[str, ...]:
        return tuple(x.stage_id for x in self.stages if x.action == ResolutionAction.EXECUTE)

    @property
    def reused_stage_ids(self) -> tuple[str, ...]:
        return tuple(x.stage_id for x in self.stages if x.action == ResolutionAction.REUSE)


class QualifiedArtifactCatalog(Protocol):
    def find_qualified(
        self,
        *,
        artifact_type: str,
        schema_version: str,
        semantic_sha256: str,
    ) -> UUID | None: ...


class CompilePlanResolver:
    """Resolve minimal semantic execution from immutable StageResult manifests."""

    def __init__(self, graph: StageGraph, catalog: QualifiedArtifactCatalog) -> None:
        self.graph = graph
        self.catalog = catalog

    def resolve(
        self,
        *,
        target_stage_id: str,
        subject_semantic_sha256: str,
        versions: Mapping[str, StageVersionIdentity],
    ) -> ResolvedCompilePlan:
        required = self.graph.ancestors_including([target_stage_id])
        expected_by_stage: dict[str, str] = {}
        decisions: list[ResolvedStage] = []

        for stage_id in required:
            stage = next(s for s in self.graph.stages if s.stage_id == stage_id)
            version = versions.get(stage_id)
            if version is None:
                raise KeyError(f"missing stage version identity: {stage_id}")

            inputs = [
                ArtifactInputIdentity(
                    role="subject",
                    ordinal=0,
                    artifact_type="RealSaS.SubjectSemanticIdentity",
                    semantic_sha256=subject_semantic_sha256,
                )
            ]
            for index, parent in enumerate(stage.depends_on, start=1):
                if parent not in expected_by_stage:
                    raise RuntimeError(f"resolver dependency order drift: {stage_id} <- {parent}")
                inputs.append(
                    ArtifactInputIdentity(
                        role=f"stage:{parent}",
                        ordinal=index,
                        artifact_type=STAGE_RESULT_ARTIFACT_TYPE,
                        semantic_sha256=expected_by_stage[parent],
                    )
                )

            descriptor = ArtifactSemanticDescriptor(
                artifact_type=STAGE_RESULT_ARTIFACT_TYPE,
                schema_version=STAGE_RESULT_SCHEMA_VERSION,
                producer_contract=stage_id,
                implementation_sha256=version.implementation_sha256,
                policy_sha256=version.policy_sha256,
                inputs=tuple(inputs),
                semantic_parameters={
                    "stage_id": stage_id,
                    **dict(version.semantic_parameters),
                },
            )
            expected = descriptor.semantic_sha256
            expected_by_stage[stage_id] = expected
            reusable = self.catalog.find_qualified(
                artifact_type=STAGE_RESULT_ARTIFACT_TYPE,
                schema_version=STAGE_RESULT_SCHEMA_VERSION,
                semantic_sha256=expected,
            )
            decisions.append(
                ResolvedStage(
                    stage_id=stage_id,
                    action=ResolutionAction.REUSE if reusable is not None else ResolutionAction.EXECUTE,
                    expected_semantic_sha256=expected,
                    reusable_artifact_id=reusable,
                    reason="QUALIFIED_SEMANTIC_IDENTITY_HIT" if reusable is not None else "QUALIFIED_SEMANTIC_IDENTITY_MISS",
                )
            )

        return ResolvedCompilePlan(target_stage_id=target_stage_id, stages=tuple(decisions))
