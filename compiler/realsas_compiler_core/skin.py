from __future__ import annotations
from collections import defaultdict
from math import isfinite
from .types import SkinProposalIR, QualifiedSkinIR, QualifiedSkinRow, QualifiedSkeletonIR, RiggingSurfaceIR, QualificationError
from .hashing import content_sha256

def qualify_skin(surface:RiggingSurfaceIR, skeleton:QualifiedSkeletonIR, proposal:SkinProposalIR, *, max_simplex_repair_l1:float=0.02, negative_tolerance:float=1e-8, max_influences:int|None=None)->QualifiedSkinIR:
    if proposal.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("stale/mismatched skin proposal: surface lineage mismatch")
    if proposal.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("stale/mismatched skin proposal: skeleton lineage mismatch")
    surface_ids={n.surface_id for n in surface.surface_nodes}; joint_ids={j.canonical_joint_id for j in skeleton.joints}
    rows=defaultdict(lambda:defaultdict(float))
    for inf in proposal.influences:
        if inf.surface_id not in surface_ids: raise QualificationError(f"unknown surface influence:{inf.surface_id}")
        if inf.canonical_joint_id not in joint_ids: raise QualificationError(f"unknown canonical joint influence:{inf.canonical_joint_id}")
        w=float(inf.weight)
        if not isfinite(w): raise QualificationError("non-finite skin weight")
        if w < -negative_tolerance: raise QualificationError("material negative skin weight")
        rows[inf.surface_id][inf.canonical_joint_id]+=max(0.0,w)
    qualified=[]; corrected=0; max_res=0.0; total_corr=0.0
    for sid in sorted(surface_ids):
        row=rows.get(sid,{})
        if not row: raise QualificationError(f"surface has no skin influences:{sid}")
        items=sorted(row.items())
        if max_influences is not None and len(items)>max_influences:
            # Top-k selection is deterministic but only admitted if discarded mass is within the same bounded correction budget.
            ranked=sorted(items,key=lambda x:(-x[1],x[0])); keep=ranked[:max_influences]; discarded=sum(w for _,w in ranked[max_influences:])
            if discarded>max_simplex_repair_l1: raise QualificationError(f"influence sparsification would exceed repair budget:{sid}:{discarded}")
            items=sorted(keep); total_corr+=discarded; corrected+=1
        s=sum(w for _,w in items); res=abs(s-1.0); max_res=max(max_res,res)
        if s<=1e-12: raise QualificationError(f"zero skin row:{sid}")
        if res>max_simplex_repair_l1: raise QualificationError(f"simplex residual exceeds bounded repair budget:{sid}:{res}")
        norm=tuple((jid,w/s) for jid,w in items)
        corr=sum(abs(w2-w) for (jid,w),(jid2,w2) in zip(items,norm))
        total_corr+=corr
        if corr>1e-12: corrected+=1
        qualified.append(QualifiedSkinRow(sid,norm,res,corr))
    report={"row_count":len(qualified),"corrected_row_count":corrected,"max_simplex_residual_before":max_res,"total_correction_l1":total_corr,"bounded_repair_limit_l1":max_simplex_repair_l1,"silent_clipping_forbidden":True}
    lineage=content_sha256({"surface":surface.geometry_lineage_hash,"skeleton":skeleton.skeleton_lineage_hash,"proposal":proposal.to_dict(),"report":report,"rows":[r.to_dict() for r in qualified]})
    return QualifiedSkinIR(tuple(qualified),surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,report,lineage)
