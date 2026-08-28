from __future__ import annotations
from .types import SkeletonProposalIR, QualifiedSkeletonIR, QualifiedJoint, RiggingSurfaceIR, QualificationError
from .hashing import content_sha256
from realsas_contracts.technical_part_graph import CanonicalGraphNodeCandidate, CanonicalGraphEdgeCandidate, CanonicalGraphOptimizationRequest
from realsas_synthesis.canonical_graph_optimizer import optimize_canonical_graph_v18_98

def qualify_skeleton(surface:RiggingSurfaceIR, proposal:SkeletonProposalIR, *, run_ilp_shadow:bool=False)->QualifiedSkeletonIR:
    if proposal.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("stale/mismatched skeleton proposal: surface lineage mismatch")
    if not proposal.joints: raise QualificationError("empty skeleton proposal")
    joint_by={j.proposal_id:j for j in proposal.joints}
    if len(joint_by)!=len(proposal.joints): raise QualificationError("duplicate proposal joint id")
    surface_ids={n.surface_id for n in surface.surface_nodes}
    internal={pid:"CGC:"+content_sha256({"proposal_id":pid,"position":joint_by[pid].position,"surface":surface.geometry_lineage_hash})[:20] for pid in sorted(joint_by)}
    reverse={v:k for k,v in internal.items()}
    nodes=[]
    for pid in sorted(joint_by):
        j=joint_by[pid]
        missing=set(j.support_surface_ids)-surface_ids
        if missing: raise QualificationError(f"joint {pid} references unknown surface ids:{sorted(missing)}")
        nodes.append(CanonicalGraphNodeCandidate(
            canonical_control_id=internal[pid],root_score01=float(j.root_score),confidence01=float(j.confidence),
            uncertainty01=max(0.0,1.0-float(j.confidence)),required=True,
            provenance={"source_proposal_id":pid,"support_surface_ids":tuple(j.support_surface_ids)},
        ))
    edges=[]
    for e in sorted(proposal.edges,key=lambda x:x.edge_id):
        if e.parent_proposal_id not in internal or e.child_proposal_id not in internal:
            raise QualificationError(f"edge endpoint missing:{e.edge_id}")
        edges.append(CanonicalGraphEdgeCandidate(
            edge_key="CGE:"+content_sha256(e.to_dict())[:20],
            parent_control_id=internal[e.parent_proposal_id],child_control_id=internal[e.child_proposal_id],
            evidence_score01=float(e.score),authority_tier="MODEL_PROPOSAL_EVIDENCE",objective_score=float(e.score),
            uncertainty01=max(0.0,1.0-float(e.confidence)),reason=e.reason,hard_required=e.hard_required,hard_forbidden=e.hard_forbidden,
            provenance={"source_edge_id":e.edge_id},
        ))
    req=CanonicalGraphOptimizationRequest(
        request_id="RIGQ:"+content_sha256({"surface":surface.geometry_lineage_hash,"proposal":proposal.to_dict()})[:24],
        attempt_id="CURRENT",nodes=tuple(nodes),edges=tuple(edges),root_candidate_ids=tuple(sorted(internal.values())),
        run_ilp_shadow=run_ilp_shadow,ilp_authority_enabled=False,calibration_locked=False,
        metadata={"proposal_ids_are_not_canonical":True,"surface_hash":surface.geometry_lineage_hash},
    )
    res=optimize_canonical_graph_v18_98(req)
    if not res.passed:
        raise QualificationError("skeleton qualification failed:"+";".join(res.blockers or (res.status,)))
    # Canonical product IDs are newly minted by compiler; neither model proposal ids nor optimizer candidate ids survive as product IDs.
    canonical={cid:"J:"+content_sha256({"qualified_graph":res.content_sha256,"candidate":cid})[:20] for cid in res.selected_node_ids}
    q=[]
    for cid in sorted(res.selected_node_ids):
        pid=reverse[cid]; j=joint_by[pid]; pcid=res.parent_by_child.get(cid)
        q.append(QualifiedJoint(canonical[cid],tuple(map(float,j.position)),None if pcid is None else canonical[pcid],tuple(j.support_surface_ids),pid))
    root=canonical[res.selected_root_control_id]
    report={"solver":res.solver,"status":res.status,"objective":res.objective_value,"optimality_proven":res.optimality_proven,"blockers":res.blockers,"warnings":res.warnings,"optimizer_result_sha256":res.content_sha256,"proposal_to_candidate":internal,"candidate_to_canonical":canonical}
    lineage=content_sha256({"surface":surface.geometry_lineage_hash,"proposal":proposal.to_dict(),"report":report,"joints":[j.to_dict() for j in q]})
    return QualifiedSkeletonIR(tuple(q),root,report,lineage)
