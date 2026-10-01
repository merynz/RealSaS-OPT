from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..domain import EngineReleaseManifest
from ..persistence.models import AuditEventRow, EngineReleaseRow, EngineReleaseStageRow
from ..resolver import StageVersionIdentity
from ..stage_graph import StageGraph


class EngineReleaseRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class SealedEngineRelease:
    release_id: UUID
    release_sha256: str
    reused: bool


def _validate_manifest_against_graph(manifest: EngineReleaseManifest, graph: StageGraph) -> None:
    expected = [(s.ordinal, s.stage_id) for s in graph.stages]
    actual = [(s.ordinal, s.stage_id) for s in manifest.stages]
    if actual != expected:
        raise EngineReleaseRejected("ENGINE_RELEASE_CANONICAL_STAGE_GRAPH_DRIFT")


def seal_engine_release(
    session: Session,
    *,
    manifest: EngineReleaseManifest,
    graph: StageGraph,
    created_by: str,
) -> SealedEngineRelease:
    if not created_by:
        raise ValueError("created_by is required")
    _validate_manifest_against_graph(manifest, graph)
    with session.begin():
        existing = session.execute(
            select(EngineReleaseRow).where(
                EngineReleaseRow.release_sha256 == manifest.release_sha256
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.purpose != manifest.purpose or existing.name != manifest.name:
                raise EngineReleaseRejected("ENGINE_RELEASE_IDENTITY_METADATA_DRIFT")
            return SealedEngineRelease(existing.id, existing.release_sha256, True)

        row = EngineReleaseRow(
            id=uuid4(),
            name=manifest.name,
            release_sha256=manifest.release_sha256,
            purpose=manifest.purpose,
            created_by=created_by,
            sealed_at=datetime.now(timezone.utc),
        )
        session.add(row)
        session.flush()
        for stage in manifest.stages:
            session.add(
                EngineReleaseStageRow(
                    release_id=row.id,
                    stage_id=stage.stage_id,
                    ordinal=stage.ordinal,
                    implementation_sha256=stage.implementation_sha256,
                    policy_sha256=stage.policy_sha256,
                    semantic_parameters=dict(stage.semantic_parameters),
                )
            )
        session.add(
            AuditEventRow(
                actor=created_by,
                action="ENGINE_RELEASE_SEALED",
                payload={
                    "engine_release_id": str(row.id),
                    "release_sha256": manifest.release_sha256,
                    "purpose": manifest.purpose,
                    "stage_count": 46,
                },
            )
        )
        session.flush()
        return SealedEngineRelease(row.id, row.release_sha256, False)


def load_stage_versions(
    session: Session,
    *,
    release_id: UUID,
    graph: StageGraph,
    require_product: bool,
) -> dict[str, StageVersionIdentity]:
    release = session.execute(
        select(EngineReleaseRow).where(EngineReleaseRow.id == release_id)
    ).scalar_one_or_none()
    if release is None:
        raise EngineReleaseRejected("ENGINE_RELEASE_NOT_FOUND")
    if require_product and release.purpose != "PRODUCT":
        raise EngineReleaseRejected("PRODUCT_COMPILE_REQUIRES_PRODUCT_ENGINE_RELEASE")
    rows = session.execute(
        select(EngineReleaseStageRow)
        .where(EngineReleaseStageRow.release_id == release_id)
        .order_by(EngineReleaseStageRow.ordinal)
    ).scalars().all()
    if [(r.ordinal, r.stage_id) for r in rows] != [(s.ordinal, s.stage_id) for s in graph.stages]:
        raise EngineReleaseRejected("ENGINE_RELEASE_STAGE_SET_DRIFT")
    return {
        r.stage_id: StageVersionIdentity(
            implementation_sha256=r.implementation_sha256,
            policy_sha256=r.policy_sha256,
            semantic_parameters=dict(r.semantic_parameters),
        )
        for r in rows
    }
