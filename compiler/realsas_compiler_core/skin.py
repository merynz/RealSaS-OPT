from __future__ import annotations
from collections import defaultdict
from math import isfinite
from .types import SkinProposalIR, QualifiedSkinIR, QualifiedSkinRow, QualifiedSkeletonIR, RiggingSurfaceIR, QualificationError
from .hashing import content_sha256


def qualify_skin(
    surface:RiggingSurfaceIR,
    skeleton:QualifiedSkeletonIR,
    proposal:SkinProposalIR,
    *,
    max_simplex_repair_l1:float=0.02,
    max_total_correction_l1:float=0.02,
    negative_tolerance:float=1e-8,
    max_influences:int|None=None,
    authority_bindings:dict[str,str]|None=None,
)->QualifiedSkinIR:
    if max_simplex_repair_l1 < 0.0 or max_total_correction_l1 < 0.0 or negative_tolerance < 0.0:
        raise ValueError("skin repair budgets/tolerance must be nonnegative")
    if proposal.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("stale/mismatched skin proposal: surface lineage mismatch")
    if proposal.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("stale/mismatched skin proposal: skeleton lineage mismatch")
    surface_ids={n.surface_id for n in surface.surface_nodes}; joint_ids={j.canonical_joint_id for j in skeleton.joints}
    rows=defaultdict(dict)
    tiny_negative_mass=0.0
    tiny_negative_count=0
    for inf in proposal.influences:
        if inf.surface_id not in surface_ids: raise QualificationError(f"unknown surface influence:{inf.surface_id}")
        if inf.canonical_joint_id not in joint_ids: raise QualificationError(f"unknown canonical joint influence:{inf.canonical_joint_id}")
        key=(inf.surface_id,inf.canonical_joint_id)
        if inf.canonical_joint_id in rows[inf.surface_id]:
            raise QualificationError(f"duplicate skin influence pair:{inf.surface_id}:{inf.canonical_joint_id}")
        w=float(inf.weight)
        if not isfinite(w): raise QualificationError("non-finite skin weight")
        if w < -negative_tolerance: raise QualificationError("material negative skin weight")
        if w < 0.0:
            tiny_negative_mass += -w
            tiny_negative_count += 1
        rows[inf.surface_id][inf.canonical_joint_id]=w
    qualified=[]; corrected=0; max_res=0.0; total_corr=0.0; total_discarded=0.0
    for sid in sorted(surface_ids):
        raw=dict(rows.get(sid,{}))
        if not raw: raise QualificationError(f"surface has no skin influences:{sid}")
        raw_sum=sum(raw.values()); res=abs(raw_sum-1.0); max_res=max(max_res,res)
        clipped={jid:max(0.0,w) for jid,w in raw.items()}
        items=sorted(clipped.items())
        discarded=0.0
        if max_influences is not None and len(items)>max_influences:
            ranked=sorted(items,key=lambda x:(-x[1],x[0])); keep=ranked[:max_influences]; discarded=sum(w for _,w in ranked[max_influences:])
            items=sorted(keep); total_discarded+=discarded
        s=sum(w for _,w in items)
        if s<=1e-12: raise QualificationError(f"zero skin row:{sid}")
        norm=tuple((jid,w/s) for jid,w in items)
        final=dict(norm)
        # Exact end-to-end row correction from the submitted proposal row to the
        # admitted row. Intermediate clipping/sparsification/renormalization is not
        # double-counted.
        corr=sum(abs(final.get(jid,0.0)-raw.get(jid,0.0)) for jid in set(raw)|set(final))
        if corr>max_simplex_repair_l1+1e-15:
            raise QualificationError(f"skin row correction exceeds bounded repair budget:{sid}:{corr}")
        total_corr+=corr
        if total_corr>max_total_correction_l1+1e-15:
            raise QualificationError(f"skin aggregate correction exceeds bounded repair budget:{total_corr}")
        if corr>1e-12: corrected+=1
        qualified.append(QualifiedSkinRow(sid,norm,res,corr))
    proposal_meta=dict(proposal.metadata or {})
    declared_supervision_coverage=proposal_meta.get("supervision_coverage")
    if declared_supervision_coverage is None:
        declared_supervision_coverage=proposal_meta.get("teacher_coverage")
    if declared_supervision_coverage is not None:
        declared_supervision_coverage=float(declared_supervision_coverage)
        if not isfinite(declared_supervision_coverage) or not (0.0 <= declared_supervision_coverage <= 1.0):
            raise QualificationError("skin proposal supervision coverage invalid")
    confidence_summary=proposal_meta.get("row_confidence_summary")
    confidence_semantics=proposal_meta.get("row_confidence_semantics")
    row_confidence_available=confidence_summary is not None or confidence_semantics is not None
    report={
        "row_count":len(qualified),
        "prediction_surface_coverage":len(qualified)/max(len(surface_ids),1),
        "supervision_coverage":declared_supervision_coverage,
        "supervision_coverage_semantics":(
            str(proposal_meta.get("supervision_coverage_semantics"))
            if proposal_meta.get("supervision_coverage_semantics") is not None
            else ("DECLARED_BY_PROPOSAL" if declared_supervision_coverage is not None else "NOT_DECLARED_BY_PROPOSAL")
        ),
        "row_confidence_available":bool(row_confidence_available),
        "row_confidence_semantics":(
            str(confidence_semantics) if confidence_semantics is not None else "NOT_DECLARED_BY_PROPOSAL"
        ),
        "uncovered_row_semantics":str(
            proposal_meta.get("uncovered_row_semantics")
            or "UNDECLARED_SUPERVISION_ROWS_REQUIRE_DOWNSTREAM_MECHANICAL_COMPATIBILITY_PROOF"
        ),
        "product_skin_evidence_complete":bool(
            declared_supervision_coverage is not None and row_confidence_available
        ),
        "corrected_row_count":corrected,
        "max_simplex_residual_before":max_res,
        "total_correction_l1":total_corr,
        "bounded_repair_limit_l1_per_row":max_simplex_repair_l1,
        "bounded_repair_limit_l1_total":max_total_correction_l1,
        "total_sparsification_discarded_mass":total_discarded,
        "tiny_negative_clipped_count":tiny_negative_count,
        "tiny_negative_clipped_mass":tiny_negative_mass,
        "duplicate_pairs_rejected":True,
        "silent_clipping_forbidden":True,
    }
    authority_bindings={
        str(k):str(v)
        for k,v in sorted(dict(authority_bindings or {}).items())
    }
    if any(not k or not v for k,v in authority_bindings.items()):
        raise QualificationError("SKIN_AUTHORITY_BINDING_INVALID")
    if authority_bindings:
        report["authority_bindings"]=authority_bindings
    if corrected>len(qualified):
        raise QualificationError("skin repair accounting invariant violated")
    lineage=content_sha256({"surface":surface.geometry_lineage_hash,"skeleton":skeleton.skeleton_lineage_hash,"proposal":proposal.to_dict(),"report":report,"rows":[r.to_dict() for r in qualified]})
    return QualifiedSkinIR(tuple(qualified),surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,report,lineage)
