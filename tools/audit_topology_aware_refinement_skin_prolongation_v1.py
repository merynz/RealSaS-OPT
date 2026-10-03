"""Audit topology-aware skin prolongation from a coarse surface to its exact refinement.

No nearest-neighbour search over unrelated geometry is used. Every target node has
an exact dense-source parent from base_inverse/final_inverse. Interpolation stencils
are restricted to that parent's Stage15 local-relation graph neighbours.
"""
from __future__ import annotations
import argparse, json, math, shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, qualified_skeleton_from_dict,
    qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.product_mesh_skin_v1 import _skin_support_coefficients
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import (
    QualifiedJoint, QualifiedSkeletonIR, SkinInfluenceProposal, SkinProposalIR,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from tools.demo.render_knight_motion_preview_v1 import _ctx

METHODS=("PARENT_CONSTANT","GRAPH_EDGE_LINEAR","GRAPH_IDW_P1","GRAPH_IDW_P2","GRAPH_IDW_P4")

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,sort_keys=True,indent=2)+"\n")
def q(a,p): return float(np.quantile(np.asarray(a,dtype=np.float64),p))

def target_parent_map(source_surface,target_surface,inverse_npz):
    with np.load(inverse_npz,allow_pickle=False) as z:
        src=np.asarray(z["base_inverse"])
        dst=np.asarray(z["final_inverse"])
    if src.shape!=dst.shape or src.ndim!=1 or src.dtype.kind not in "iu" or dst.dtype.kind not in "iu":
        raise RuntimeError("PROLONGATION_INVERSE_CONTRACT_INVALID")
    if np.any(src<0) or np.any(src>=len(source_surface.surface_nodes)) or np.any(dst<0) or np.any(dst>=len(target_surface.surface_nodes)):
        raise RuntimeError("PROLONGATION_INVERSE_RANGE_INVALID")
    pairs=np.unique(np.column_stack((dst.astype(np.int64),src.astype(np.int64))),axis=0)
    if len(pairs)!=len(target_surface.surface_nodes):
        raise RuntimeError("PROLONGATION_TARGET_CROSSES_COARSE_PARENT")
    parent=np.full(len(target_surface.surface_nodes),-1,np.int64)
    parent[pairs[:,0]]=pairs[:,1]
    if np.any(parent<0): raise RuntimeError("PROLONGATION_PARENT_ACCOUNTING_INVALID")
    return parent

def source_graph(surface):
    sid_to_i={str(n.surface_id):i for i,n in enumerate(surface.surface_nodes)}
    if len(sid_to_i)!=len(surface.surface_nodes): raise RuntimeError("PROLONGATION_SOURCE_ID_DUPLICATE")
    nbr=[set() for _ in surface.surface_nodes]
    kinds=defaultdict(int)
    for rel in surface.local_relations:
        a=sid_to_i.get(str(rel.a_surface_id)); b=sid_to_i.get(str(rel.b_surface_id))
        if a is None or b is None or a==b: raise RuntimeError("PROLONGATION_LOCAL_RELATION_INVALID")
        nbr[a].add(b); nbr[b].add(a); kinds[str(rel.relation_kind)]+=1
    return tuple(tuple(sorted(x)) for x in nbr),dict(sorted(kinds.items()))

def skin_matrix(surface,skeleton,skin):
    sids=tuple(str(n.surface_id) for n in surface.surface_nodes)
    jids=tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    ji={x:i for i,x in enumerate(jids)}
    rows={str(r.surface_id):r for r in skin.rows}
    if set(rows)!=set(sids): raise RuntimeError("PROLONGATION_SOURCE_SKIN_ACCOUNTING")
    W=np.zeros((len(sids),len(jids)),np.float64)
    for r,sid in enumerate(sids):
        for jid,w in rows[sid].influences:
            if jid not in ji: raise RuntimeError("PROLONGATION_SOURCE_SKIN_JOINT")
            W[r,ji[jid]]=float(w)
    if np.any(W<0) or not np.allclose(W.sum(1),1.0,atol=1e-8,rtol=0):
        raise RuntimeError("PROLONGATION_SOURCE_SKIN_SIMPLEX")
    return W,jids

def rebound_skeleton(source_skeleton,source_surface,target_surface,parent):
    source_sid_to_i={str(n.surface_id):i for i,n in enumerate(source_surface.surface_nodes)}
    children=[[] for _ in source_surface.surface_nodes]
    for ti,pi in enumerate(parent): children[int(pi)].append(str(target_surface.surface_nodes[ti].surface_id))
    joints=[]
    for j in source_skeleton.joints:
        supports=[]
        for sid in j.support_surface_ids:
            idx=source_sid_to_i.get(str(sid))
            if idx is None: raise RuntimeError("PROLONGATION_SKELETON_SOURCE_SUPPORT_UNKNOWN")
            supports.extend(children[idx])
        joints.append(QualifiedJoint(
            canonical_joint_id=str(j.canonical_joint_id), position=tuple(map(float,j.position)),
            parent_canonical_id=None if j.parent_canonical_id is None else str(j.parent_canonical_id),
            support_surface_ids=tuple(sorted(set(supports))),
            source_proposal_id=str(j.source_proposal_id),
        ))
    report={
        "status":"DIAGNOSTIC_EXACT_REFINEMENT_SUPPORT_REBOUND",
        "product_authority_minted":False,
        "kinematics_source":"ARCHIVED_STAGE28",
        "support_rebound":"EXACT_BASE_INVERSE_TO_FINAL_INVERSE_CHILDREN",
        "source_skeleton_lineage_hash":source_skeleton.skeleton_lineage_hash,
    }
    provisional=QualifiedSkeletonIR(tuple(joints),str(source_skeleton.root_id),report,"")
    payload=provisional.to_dict();payload.pop("skeleton_lineage_hash",None)
    return replace(provisional,skeleton_lineage_hash=content_sha256(payload))

def child_weight(method,x,p,nbr,P,W,eps):
    if method=="PARENT_CONSTANT" or not nbr[p]:
        return W[p].copy(),1.0
    if method=="GRAPH_EDGE_LINEAR":
        best=None
        for n in nbr[p]:
            v=P[n]-P[p]; vv=float(np.dot(v,v))
            if vv<=eps*eps: continue
            raw=float(np.dot(x-P[p],v)/vv)
            t=min(1.0,max(0.0,raw))
            proj=P[p]+t*v
            residual=float(np.linalg.norm(x-proj)/(math.sqrt(vv)+eps))
            key=(residual,-t,int(n))
            if best is None or key<best[0]: best=(key,t,n)
        if best is None or best[1]<=0.0: return W[p].copy(),1.0
        t,n=best[1],best[2]
        return (1.0-t)*W[p]+t*W[n],1.0-t
    power=float(method.rsplit("P",1)[1])
    ids=np.asarray((p,)+nbr[p],dtype=np.int64)
    d=np.linalg.norm(P[ids]-x[None,:],axis=1)
    exact=np.flatnonzero(d<=eps)
    if len(exact):
        coeff=np.zeros(len(ids),np.float64); coeff[int(exact[0])]=1.0
    else:
        raw=1.0/np.power(d,power); coeff=raw/raw.sum()
    return coeff@W[ids],float(coeff[0])

def unique_candidate_edges(candidate):
    vi={v.candidate_vertex_id:i for i,v in enumerate(candidate.vertices)}
    edges=set()
    for a,b,c in candidate.faces:
        ia,ib,ic=vi[a],vi[b],vi[c]
        edges.add(tuple(sorted((ia,ib))));edges.add(tuple(sorted((ib,ic))));edges.add(tuple(sorted((ic,ia))))
    return np.asarray(sorted(edges),np.int64),vi

def candidate_gradient(candidate,target_surface,skin,skeleton):
    srows={r.surface_id:r for r in skin.rows};jids=tuple(j.canonical_joint_id for j in skeleton.joints);ji={x:i for i,x in enumerate(jids)}
    V=np.zeros((len(candidate.vertices),len(jids)),np.float64)
    pos=np.asarray([v.P for v in candidate.vertices],np.float64)
    for r,v in enumerate(candidate.vertices):
        for sid,c in _skin_support_coefficients(v):
            for jid,w in srows[sid].influences: V[r,ji[jid]]+=float(c)*float(w)
    edges,_=unique_candidate_edges(candidate)
    span=float(np.max(pos.max(0)-pos.min(0)));length=np.linalg.norm(pos[edges[:,0]]-pos[edges[:,1]],axis=1)/span
    delta=np.abs(V[edges[:,0]]-V[edges[:,1]]).sum(1);grad=delta/length
    return {"edge_count":len(edges),"weight_l1_gt_1":int(np.count_nonzero(delta>1.0)),
        "tiny_lt_1e3_and_l1_gt_1":int(np.count_nonzero((length<1e-3)&(delta>1.0))),
        "grad_p95":q(grad,.95),"grad_p99":q(grad,.99),"grad_p999":q(grad,.999),"grad_max":float(np.max(grad))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True);ap.add_argument("--run-id",required=True)
    ap.add_argument("--target-surface-json",type=Path,required=True);ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True);ap.add_argument("--out-root",type=Path,required=True)
    a=ap.parse_args();ctx=_ctx(a.authority_root,a.run_id)
    source_surface=rigging_surface_from_dict(stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    source_skeleton=qualified_skeleton_from_dict(stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    source_skin=qualified_skin_from_dict(stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))
    target_surface=rigging_surface_from_dict(read(a.target_surface_json));candidate=canonical_mesh_candidate_from_dict(read(a.candidate_json))
    parent=target_parent_map(source_surface,target_surface,a.inverse_npz)
    nbr,kinds=source_graph(source_surface); W0,jids=skin_matrix(source_surface,source_skeleton,source_skin)
    target_skeleton=rebound_skeleton(source_skeleton,source_surface,target_surface,parent)
    P=np.asarray([n.P for n in source_surface.surface_nodes],np.float64)
    X=np.asarray([n.P for n in target_surface.surface_nodes],np.float64)
    span=float(np.max(P.max(0)-P.min(0)));eps=max(1e-14,span*1e-12)
    degree=np.asarray([len(x) for x in nbr],np.int64)

    manifest={"schema":"RealSaS.TopologyAwareRefinementSkinProlongationAudit.v1","status":"DIAGNOSTIC_ONLY",
        "product_authority_minted":False,"training_used":False,"global_knn_used":False,
        "exact_dense_parent_mapping_used":True,"source_node_count":len(P),"target_node_count":len(X),
        "source_graph_relation_kind_counts":kinds,"source_degree":{"min":int(degree.min()),"p50":q(degree,.5),"p95":q(degree,.95),"max":int(degree.max())},
        "source_skin_lineage_hash":source_skin.skin_lineage_hash,"source_skeleton_lineage_hash":source_skeleton.skeleton_lineage_hash,
        "target_skeleton_lineage_hash":target_skeleton.skeleton_lineage_hash,"variants":[]}
    a.out_root.mkdir(parents=True,exist_ok=True)
    for method in METHODS:
        W=np.empty((len(X),len(jids)),np.float64);parent_coeff=np.empty(len(X),np.float64)
        for t,x in enumerate(X):
            W[t],parent_coeff[t]=child_weight(method,x,int(parent[t]),nbr,P,W0,eps)
        if np.any(W< -1e-12) or not np.allclose(W.sum(1),1.0,atol=1e-8,rtol=0):
            raise RuntimeError("PROLONGATION_OUTPUT_SIMPLEX_INVALID:"+method)
        W=np.maximum(W,0.0);W/=W.sum(1,keepdims=True)
        influences=tuple(SkinInfluenceProposal(
            str(target_surface.surface_nodes[t].surface_id),jid,float(W[t,j])
        ) for t in range(len(X)) for j,jid in enumerate(jids))
        proposal=SkinProposalIR(influences,target_surface.geometry_lineage_hash,target_skeleton.skeleton_lineage_hash,
            model_provenance="COMPILER_EXACT_REFINEMENT_TOPOLOGY_AWARE_PROLONGATION_AUDIT_V1",
            metadata={"method":method,"global_knn_used":False,"source_graph_only":True,"product_authority_minted":False})
        skin=qualify_skin(target_surface,target_skeleton,proposal,max_simplex_repair_l1=1e-8,max_total_correction_l1=1e-4)
        parent_delta=np.abs(W-W0[parent]).sum(1)
        grad=candidate_gradient(candidate,target_surface,skin,target_skeleton)
        row={"method":method,"parent_coeff_mean":float(np.mean(parent_coeff)),"parent_coeff_p05":q(parent_coeff,.05),
            "parent_coeff_p50":q(parent_coeff,.5),"parent_coeff_min":float(np.min(parent_coeff)),
            "child_vs_parent_l1_mean":float(np.mean(parent_delta)),"child_vs_parent_l1_p95":q(parent_delta,.95),
            "child_vs_parent_l1_max":float(np.max(parent_delta)),**grad,"skin_lineage_hash":skin.skin_lineage_hash}
        manifest["variants"].append(row)
        out=a.out_root/method.lower();out.mkdir(parents=True,exist_ok=True)
        write(out/"fresh_qualified_skeleton.json",target_skeleton.to_dict());write(out/"fresh_qualified_skin.json",skin.to_dict())
        write(out/"rig_REPORT.json",{"status":"DIAGNOSTIC_ARCHIVED_KINEMATICS_EXACT_SUPPORT_REBOUND","product_authority_minted":False})
        write(out/"skin_REPORT.json",{"status":"DIAGNOSTIC_TOPOLOGY_AWARE_PROLONGATION","method":method,"product_authority_minted":False,**row})
        write(out/"PROLONGATION_STATS.json",row)
        print("PROLONGATION_PREP="+json.dumps(row,sort_keys=True),flush=True)
    write(a.out_root/"PROLONGATION_MANIFEST.json",manifest)

if __name__=="__main__":main()
