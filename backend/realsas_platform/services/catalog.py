from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..persistence.models import ArtifactRow, ArtifactTypeRow, QualificationRow


class PostgresQualifiedArtifactCatalog:
    def __init__(self, session: Session, *, qualification_type: str = "REUSE_ELIGIBLE") -> None:
        self.session = session
        self.qualification_type = qualification_type

    def find_qualified(
        self,
        *,
        artifact_type: str,
        schema_version: str,
        semantic_sha256: str,
    ) -> UUID | None:
        return self.session.execute(
            select(ArtifactRow.id)
            .join(ArtifactTypeRow, ArtifactTypeRow.id == ArtifactRow.artifact_type_id)
            .join(QualificationRow, QualificationRow.artifact_id == ArtifactRow.id)
            .where(
                ArtifactTypeRow.name == artifact_type,
                ArtifactTypeRow.schema_version == schema_version,
                ArtifactRow.semantic_sha256 == semantic_sha256,
                QualificationRow.qualification_type == self.qualification_type,
                QualificationRow.result == "PASS",
            )
            .limit(1)
        ).scalar_one_or_none()
