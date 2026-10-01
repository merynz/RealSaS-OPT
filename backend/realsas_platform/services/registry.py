from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..artifacts import ArtifactStore, StoredObject
from ..domain import ArtifactSemanticDescriptor
from ..persistence.models import ArtifactInputRow, ArtifactRow, ArtifactTypeRow, AuditEventRow


class ArtifactRegistrationRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactInputBinding:
    role: str
    ordinal: int
    artifact_id: UUID


@dataclass(frozen=True)
class RegisteredArtifact:
    artifact_id: UUID
    semantic_sha256: str
    content_sha256: str
    storage_key: str
    reused: bool


def register_artifact(
    session: Session,
    store: ArtifactStore,
    *,
    descriptor: ArtifactSemanticDescriptor,
    schema_name: str,
    schema_version: str,
    domain: str,
    data: bytes,
    inputs: tuple[ArtifactInputBinding, ...],
    actor: str,
) -> RegisteredArtifact:
    """Store bytes first, verify them, then atomically expose registry metadata."""
    stored: StoredObject = store.put_bytes(data)
    if not store.verify(stored.content_sha256):
        raise ArtifactRegistrationRejected("ARTIFACT_STORE_VERIFICATION_FAILED")
    if [(x.role, x.ordinal) for x in inputs] != [(x.role, x.ordinal) for x in descriptor.inputs]:
        raise ArtifactRegistrationRejected("ARTIFACT_INPUT_DESCRIPTOR_DRIFT")

    with session.begin():
        artifact_type = session.execute(
            select(ArtifactTypeRow).where(
                ArtifactTypeRow.name == schema_name,
                ArtifactTypeRow.schema_version == schema_version,
            )
        ).scalar_one_or_none()
        if artifact_type is None:
            artifact_type = ArtifactTypeRow(id=uuid4(), name=schema_name, schema_version=schema_version, domain=domain)
            session.add(artifact_type)
            session.flush()

        existing = session.execute(
            select(ArtifactRow).where(
                ArtifactRow.artifact_type_id == artifact_type.id,
                ArtifactRow.semantic_sha256 == descriptor.semantic_sha256,
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.content_sha256 != stored.content_sha256:
                raise ArtifactRegistrationRejected("SEMANTIC_IDENTITY_CONTENT_DIVERGENCE")
            return RegisteredArtifact(existing.id, existing.semantic_sha256, existing.content_sha256, existing.storage_key, True)

        artifact = ArtifactRow(
            id=uuid4(),
            artifact_type_id=artifact_type.id,
            semantic_sha256=descriptor.semantic_sha256,
            content_sha256=stored.content_sha256,
            storage_key=stored.storage_key,
            size_bytes=stored.size_bytes,
            producer_contract=descriptor.producer_contract,
            implementation_sha256=descriptor.implementation_sha256,
            policy_sha256=descriptor.policy_sha256,
            semantic_parameters=dict(descriptor.semantic_parameters),
            verified_at=datetime.now(timezone.utc),
        )
        session.add(artifact)
        session.flush()
        for row in inputs:
            session.add(ArtifactInputRow(artifact_id=artifact.id, input_role=row.role, ordinal=row.ordinal, input_artifact_id=row.artifact_id))
        session.add(AuditEventRow(actor=actor, action="ARTIFACT_REGISTERED", artifact_id=artifact.id, payload={"semantic_sha256": artifact.semantic_sha256, "content_sha256": artifact.content_sha256, "producer_contract": descriptor.producer_contract}))
        session.flush()
        return RegisteredArtifact(artifact.id, artifact.semantic_sha256, artifact.content_sha256, artifact.storage_key, False)
