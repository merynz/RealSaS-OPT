from __future__ import annotations
from dataclasses import replace
from .types import SkeletonProposalIR, QualifiedSkeletonIR, QualifiedJoint, RiggingSurfaceIR, QualificationError
from .v4_types import QualifiedSkeletonIRV2
from .hashing import content_sha256
from .canonical_graph_optimizer_authority import optimize_canonical_graph_v18_98
from realsas_contracts.technical_part_graph import CanonicalGraphNodeCandidate, CanonicalGraphEdgeCandidate, CanonicalGraphOptimizationRequest

def qualified_skeleton_semantic_identity(value) -> dict:
    """Canonical skeleton semantics independent of compiler-owned joint IDs.

    Proposal identity, parent relation, position and support ownership define
    the semantic tree. Compiler-minted canonical IDs, telemetry and diagnostic
    ordering are intentionally excluded.
    """
    source_by_canonical = {}
    joints_by_source = {}
    for joint in value.joints:
        source = str(joint.source_proposal_id)
        canonical = str(joint.canonical_joint_id)
        if not source:
            raise QualificationError("SKELETON_SEMANTIC_SOURCE_ID_MISSING")
        if source in joints_by_source:
            raise QualificationError(
                f"SKELETON_SEMANTIC_SOURCE_ID_DUPLICATE:{source}"
            )
        if canonical in source_by_canonical:
            raise QualificationError(
                f"SKELETON_SEMANTIC_CANONICAL_ID_DUPLICATE:{canonical}"
            )
        joints_by_source[source] = joint
        source_by_canonical[canonical] = source

    rows = []
    for source in sorted(joints_by_source):
        joint = joints_by_source[source]
        parent_source = None
        if joint.parent_canonical_id is not None:
            parent_canonical = str(joint.parent_canonical_id)
            if parent_canonical not in source_by_canonical:
                raise QualificationError(
                    f"SKELETON_SEMANTIC_PARENT_UNKNOWN:{parent_canonical}"
                )
            parent_source = source_by_canonical[parent_canonical]
        rows.append(
            {
                "source_proposal_id": source,
                "parent_source_proposal_id": parent_source,
                "position": [float(value) for value in joint.position],
                "support_surface_ids": sorted(
                    map(str, joint.support_surface_ids)
                ),
            }
        )
    return {
        "schema": "RealSaS.SemanticSkeletonTreeByProposalIdentity.v1",
        "joints": rows,
    }


def qualified_skeleton_semantic_sha256(value) -> str:
    return content_sha256(qualified_skeleton_semantic_identity(value))


def _optimizer_semantic_identity(res) -> dict:
    """Stable authority identity for one qualified graph selection.

    Runtime/solver telemetry such as elapsed_seconds and diagnostic shadow
    payloads must never mint canonical joint IDs or skeleton lineage.
    """
    return {
        "request_id": str(res.request_id),
        "selected_root_control_id": str(res.selected_root_control_id),
        "selected_node_ids": tuple(sorted(map(str, res.selected_node_ids))),
        "selected_edge_keys": tuple(sorted(map(str, res.selected_edge_keys))),
        "parent_by_child": {
            str(child): str(parent)
            for child, parent in sorted(dict(res.parent_by_child).items())
        },
        "solver": str(res.solver),
        "status": str(res.status),
        "objective_value": float(res.objective_value),
        "optimality_proven": bool(res.optimality_proven),
        "feasible": bool(res.feasible),
        "blockers": tuple(sorted(map(str, res.blockers or ()))),
    }


def _optimizer_semantic_sha256(res) -> str:
    return content_sha256(_optimizer_semantic_identity(res))


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
    if res.optimality_proven is not True:
        raise QualificationError("skeleton qualification failed:CANONICAL_GRAPH_OPTIMALITY_NOT_PROVEN")
    optimizer_semantic_sha256=_optimizer_semantic_sha256(res)
    canonical={
        cid:"J:"+content_sha256(
            {
                "qualified_graph_semantic":optimizer_semantic_sha256,
                "candidate":cid,
            }
        )[:20]
        for cid in res.selected_node_ids
    }
    q=[]
    for cid in sorted(res.selected_node_ids):
        pid=reverse[cid]; j=joint_by[pid]; pcid=res.parent_by_child.get(cid)
        q.append(QualifiedJoint(canonical[cid],tuple(map(float,j.position)),None if pcid is None else canonical[pcid],tuple(j.support_surface_ids),pid))
    root=canonical[res.selected_root_control_id]
    diagnostic_warning_prefixes=(
        "ilp_shadow_",
        "arborescence_ilp_shadow_",
    )
    authority_warnings=tuple(
        warning
        for warning in (res.warnings or ())
        if not str(warning).startswith(diagnostic_warning_prefixes)
    )
    report={
        "solver":res.solver,
        "status":res.status,
        "objective":res.objective_value,
        "optimality_proven":res.optimality_proven,
        "blockers":res.blockers,
        "warnings":authority_warnings,
        # Backward field name retained, but its value is now the stable
        # authority-semantic hash. The optimizer object's raw content hash
        # includes non-authoritative telemetry and is forbidden from identity.
        "optimizer_result_sha256":optimizer_semantic_sha256,
        "optimizer_identity_contract":"GRAPH_SELECTION_SEMANTICS_V1",
        "optimizer_runtime_telemetry_in_identity":False,
        "ilp_shadow_diagnostics_in_identity":False,
        "proposal_to_candidate":internal,
        "candidate_to_canonical":canonical,
    }
    lineage=content_sha256({
        "surface":surface.geometry_lineage_hash,
        "proposal":proposal.to_dict(),
        "report":report,
        "joints":[j.to_dict() for j in q],
    })
    return QualifiedSkeletonIR(tuple(q),root,report,lineage)

def qualify_skeleton_v2(surface:RiggingSurfaceIR, proposal:SkeletonProposalIR, *, run_ilp_shadow:bool=False)->QualifiedSkeletonIRV2:
    """V4 product-facing skeleton qualifier.

    Current restored optimizer still emits one arborescence. The product type no
    longer conflates that implementation fact with the semantic contract: genuine
    deform roots are represented as a set, while any future technical assembly root
    is a separate explicitly non-deforming binding.
    """
    legacy=qualify_skeleton(surface,proposal,run_ilp_shadow=run_ilp_shadow)
    roots=tuple(sorted(j.canonical_joint_id for j in legacy.joints if j.parent_canonical_id is None))
    if not roots:
        raise QualificationError("qualified skeleton has no deform root")
    report=dict(legacy.qualification_report)
    report.update({
        "schema_upgrade":"RealSaS.QualifiedSkeletonIR.v2",
        "deform_root_semantics":"EXPLICIT_SET",
        "assembly_root_semantics":"SEPARATE_NON_DEFORMING_BINDING",
        "current_optimizer_shape":"SINGLE_ARBORESCENCE_COMPATIBILITY",
    })
    value=QualifiedSkeletonIRV2(tuple(legacy.joints),roots,{},report,"")
    payload=value.to_dict(); payload.pop("skeleton_lineage_hash",None)
    return replace(value,skeleton_lineage_hash=content_sha256(payload))
