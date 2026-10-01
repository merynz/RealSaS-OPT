from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..persistence.models import AuditEventRow, OutboxEventRow, ProductRevisionArtifactRow, ProductRevisionRow, PromotionRow, QualificationRow, SubjectCurrentRevisionRow

REQUIRED_PRODUCT_ROLES=frozenset({"observation","geometry","mechanical_mesh","skeleton","skin","appearance","visual_presentation","motion_library","runtime_compatibility"})

@dataclass(frozen=True)
class PromotionResult:
    promotion_id: UUID
    subject_id: UUID
    from_revision_id: UUID | None
    to_revision_id: UUID
    new_lock_version: int

class PromotionRejected(RuntimeError):
    pass

def promote_product_revision(session: Session, *, subject_id: UUID, target_revision_id: UUID, requested_by: str, reason: str) -> PromotionResult:
    """Atomic product-current switch. Session engine must be SERIALIZABLE."""
    if not requested_by or not reason: raise ValueError("requested_by and reason are required")
    with session.begin():
        target=session.execute(select(ProductRevisionRow).where(ProductRevisionRow.id==target_revision_id)).scalar_one_or_none()
        if target is None or target.subject_id!=subject_id: raise PromotionRejected("PROMOTION_TARGET_SUBJECT_MISMATCH")
        bindings=session.execute(select(ProductRevisionArtifactRow).where(ProductRevisionArtifactRow.product_revision_id==target_revision_id)).scalars().all()
        roles={row.role for row in bindings}; missing=sorted(REQUIRED_PRODUCT_ROLES-roles)
        if missing: raise PromotionRejected("PROMOTION_REQUIRED_ROLES_MISSING:"+",".join(missing))
        for binding in bindings:
            passed=session.execute(select(QualificationRow.id).where(QualificationRow.artifact_id==binding.artifact_id,QualificationRow.result=="PASS").limit(1)).scalar_one_or_none()
            if passed is None: raise PromotionRejected("PROMOTION_ARTIFACT_NOT_QUALIFIED:"+binding.role)
        current=session.execute(select(SubjectCurrentRevisionRow).where(SubjectCurrentRevisionRow.subject_id==subject_id).with_for_update()).scalar_one_or_none()
        from_id=None if current is None else current.product_revision_id
        if from_id==target_revision_id: raise PromotionRejected("PROMOTION_TARGET_ALREADY_CURRENT")
        promotion=PromotionRow(id=uuid4(),subject_id=subject_id,from_revision_id=from_id,to_revision_id=target_revision_id,reason=reason,requested_by=requested_by,qualification_snapshot={"required_roles":sorted(REQUIRED_PRODUCT_ROLES),"bound_roles":sorted(roles),"all_bound_artifacts_have_pass_qualification":True})
        session.add(promotion)
        if current is None:
            current=SubjectCurrentRevisionRow(subject_id=subject_id,product_revision_id=target_revision_id,lock_version=1); session.add(current)
        else:
            current.product_revision_id=target_revision_id; current.lock_version+=1
        session.add(AuditEventRow(actor=requested_by,action="PRODUCT_REVISION_PROMOTED",subject_id=subject_id,product_revision_id=target_revision_id,payload={"promotion_id":str(promotion.id),"from_revision_id":None if from_id is None else str(from_id),"to_revision_id":str(target_revision_id),"reason":reason}))
        session.add(OutboxEventRow(aggregate_type="Subject",aggregate_id=subject_id,event_type="ProductRevisionPromoted",payload={"promotion_id":str(promotion.id),"product_revision_id":str(target_revision_id)}))
        session.flush()
        return PromotionResult(promotion.id,subject_id,from_id,target_revision_id,current.lock_version)
