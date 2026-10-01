from __future__ import annotations

from datetime import datetime, timezone
import os
from uuid import uuid4

import pytest
from sqlalchemy import select

from backend.realsas_platform.persistence.models import (
    ArtifactRow, ArtifactTypeRow, AttemptRow, ProductRevisionArtifactRow,
    ProductRevisionRow, QualificationRow, SubjectCurrentRevisionRow, SubjectRow,
)
from backend.realsas_platform.persistence.session import create_platform_engine, create_session_factory
from backend.realsas_platform.services.promotion import REQUIRED_PRODUCT_ROLES, PromotionRejected, promote_product_revision


DB_URL=os.environ.get("REALSAS_PLATFORM_DATABASE_URL")
pytestmark=pytest.mark.skipif(not DB_URL, reason="PostgreSQL integration URL not configured")


def test_serializable_product_promotion_is_atomic_and_audited():
    engine=create_platform_engine(DB_URL)
    Session=create_session_factory(engine)
    subject_id=uuid4(); attempt_id=uuid4(); revision_id=uuid4()
    with Session() as session:
        with session.begin():
            session.add(SubjectRow(id=subject_id,slug="promotion-smoke",display_name="Promotion Smoke"))
            session.flush()
            session.add(AttemptRow(id=attempt_id,subject_id=subject_id,kind="compile_candidate",spec_sha256="a"*64,created_by="ci",final_state="QUALIFIED"))
            artifact_type=ArtifactTypeRow(id=uuid4(),name="RealSaS.PlatformPromotionSmoke.v1",schema_version="v1",domain="test")
            session.add(artifact_type); session.flush()
            artifact_ids={}
            for index,role in enumerate(sorted(REQUIRED_PRODUCT_ROLES)):
                artifact=ArtifactRow(
                    id=uuid4(),artifact_type_id=artifact_type.id,semantic_sha256=f"{index+1:064x}",
                    content_sha256=f"{index+101:064x}",storage_key=f"cas/test/{role}",size_bytes=1,
                    producer_contract=f"TEST_{role}",implementation_sha256="b"*64,policy_sha256="c"*64,
                    semantic_parameters={},verified_at=datetime.now(timezone.utc),
                )
                session.add(artifact); session.flush()
                session.add(QualificationRow(id=uuid4(),artifact_id=artifact.id,qualification_type="PRODUCT_PROMOTION",result="PASS"))
                artifact_ids[role]=artifact.id
            session.add(ProductRevisionRow(id=revision_id,subject_id=subject_id,revision_number=1,manifest_sha256="d"*64,created_from_attempt_id=attempt_id,sealed_at=datetime.now(timezone.utc)))
            session.flush()
            for role,artifact_id in artifact_ids.items():
                session.add(ProductRevisionArtifactRow(product_revision_id=revision_id,role=role,artifact_id=artifact_id))

        result=promote_product_revision(session,subject_id=subject_id,target_revision_id=revision_id,requested_by="ci",reason="integration smoke")
        assert result.to_revision_id==revision_id
        current=session.execute(select(SubjectCurrentRevisionRow).where(SubjectCurrentRevisionRow.subject_id==subject_id)).scalar_one()
        assert current.product_revision_id==revision_id
        assert current.lock_version==1
        session.rollback()
        with pytest.raises(PromotionRejected,match="PROMOTION_TARGET_ALREADY_CURRENT"):
            promote_product_revision(session,subject_id=subject_id,target_revision_id=revision_id,requested_by="ci",reason="duplicate")
    engine.dispose()
