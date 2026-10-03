"""Attribute remaining V9 dynamic stretch to source-topology adjacency classes."""
from __future__ import annotations
import argparse,json
from collections import deque
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,qualified_camera_set_from_dict,
    qualified_skin_from_dict,qualified_skeleton_from_dict,rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.product_mesh_skin_v1 import _skin_support_coefficients
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import SkinInfluenceProposal,SkinProposalIR
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from tools.audit_topology_aware_refinement_skin_prolongation_v1 import (
    child_weight,rebound_skeleton,skin_matrix,source_graph,target_parent_map,
)
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import CLIPS,FULL_MOTION_SAMPLES
from tools.demo.render_knight_motion_preview_v1 import _ctx,_skin,_tracks_for_clip

METHODS=("PARENT_CONSTANT","GRAPH_IDW_P4")

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,sort_keys=True,indent=2)+"\n")
def q(a,p): return float(np.quantile(np.asarray(a,dtype=np.float64),p))

def build_skin(method,target_surface,target_skeleton,parent,nbr,P,W0,jids):
    X=np.asarray([n.P for n in target_surface.surface_nodes],np.float64)
    span=float(np.max(P.max(0)-P.min(0)));eps=max(1e-14,span*1e-12)
    W=np.empty((len(X),len(jids)),np.float64)
    for t,x in enumerate(X):
        W[t],_=child_weight(method,x,int(parent[t]),nbr,P,W0,eps)
    W=np.maximum(W,0.0);W/=W.sum(1,keepdims=True)
    inf=[]
    for t,node in enumerate(target_surface.surface_nodes):
        for j,jid in enumerate(jids):
            w=float(W[t,j])
            if w>1e-15: inf.append(SkinInfluenceProposal(str(node.surface_id),jid,w))
    prop=SkinProposalIR(tuple(inf),target_surface.geometry_lineage_hash,target_skeleton.skeleton_lineage_hash,
        model_provenance="V9_SOURCE_TOPOLOGY_ATTRIBUTION_"+method,
        metadata={"diagnostic_only":True,"product_authority_minted":False})
    return qualify_skin(target_surface,target_skeleton,prop,max_simplex_repair_l1=1e-8,max_total_correction_l1=1e-4)

def unique_edges(candidate):
    vi={v.candidate_vertex_id:i for i,v in enumerate(candidate.vertices)}
    edges=set()
    for a,b,c in candidate.faces:
        ia,ib,ic=vi[a],vi[b],vi[c]
        edges.add(tuple(sorted((ia,ib))));edges.add(tuple(sorted((ib,ic))));edges.add(tuple(sorted((ic,ia))))
    return np.asarray(sorted(edges),np.int64)

def components(nbr):
    comp=np.full(len(nbr),-1,np.int64);cid=0
    for s in range(len(nbr)):
        if comp[s]>=0: continue
        dq=[s];comp[s]=cid
        while dq:
            u=dq.pop()
            for v in nbr[u]:
                if comp[v]<0: comp[v]=cid;dq.append(v)
        cid+=1
    return comp,cid

def vertex_parent_sets(candidate,target_surface,parent):
    tsi={str(n.surface_id):i for i,n in enumerate(target_surface.surface_nodes)}
    out=[]
    for v in candidate.vertices:
        ps=set()
        for sid,c in _skin_support_coefficients(v):
            if float(c)>0:
                ti=tsi.get(str(sid))
                if ti is None: raise RuntimeError("ATTRIBUTION_TARGET_SUPPORT_UNKNOWN")
                ps.add(int(parent[ti]))
        if not ps: raise RuntimeError("ATTRIBUTION_VERTEX_PARENT_SET_EMPTY")
        out.append(tuple(sorted(ps)))
    return tuple(out)

def edge_class(A,B,nbr,two_ring,comp):
    sa=set(A);sb=set(B)
    if sa&sb:return "HOP0_SHARED_PARENT"
    for a in sa:
        if any(b in nbr[a] for b in sb): return "HOP1_LOCAL_RELATION"
    for a in sa:
        if any(b in two_ring[a] for b in sb): return "HOP2"
    if any(comp[a]==comp[b] for a in sa for b in sb): return "SAME_COMPONENT_GT2"
    return "DISCONNECTED_COMPONENTS"

def motion_edge_max(candidate,surface,skeleton,skin,ctx,cameras):
    rest,W,_faces=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin)
    edges=unique_edges(candidate)
    rlen=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
    if np.any(rlen<=1e-12): raise RuntimeError("ATTRIBUTION_DEGENERATE_REST_EDGE")
    max_ratio=np.ones(len(edges),np.float64);owner=[None]*len(edges)
    source_report=read("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json")
    joint_ids=tuple(j.canonical_joint_id for j in skeleton.joints)
    for clip in CLIPS:
        path=ctx["run_root"]/"inputs/motion/quaternius_knight_v1"/(clip+".motion.json")
        payload=read(path);tracks,_mapping=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(0.,float(payload["duration_seconds"]),FULL_MOTION_SAMPLES,endpoint=not bool(payload.get("loop")))
        for fi,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,W,joint_ids,mats)
            plen=np.linalg.norm(posed[edges[:,0]]-posed[edges[:,1]],axis=1)
            ratio=plen/rlen
            upd=ratio>max_ratio
            for idx in np.flatnonzero(upd): owner[int(idx)]=(clip,int(fi),float(t))
            max_ratio=np.maximum(max_ratio,ratio)
    return rest,W,edges,rlen,max_ratio,owner

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True);ap.add_argument("--run-id",required=True)
    ap.add_argument("--target-surface-json",type=Path,required=True);ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();ctx=_ctx(a.authority_root,a.run_id)
    source_surface=rigging_surface_from_dict(stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    source_skeleton=qualified_skeleton_from_dict(stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    source_skin=qualified_skin_from_dict(stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))
    cameraset=qualified_camera_set_from_dict(stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    cameras=tuple(sorted(cameraset.cameras,key=lambda c:c.view_index))
    target_surface=rigging_surface_from_dict(read(a.target_surface_json));candidate=canonical_mesh_candidate_from_dict(read(a.candidate_json))
    parent=target_parent_map(source_surface,target_surface,a.inverse_npz);nbr,_=source_graph(source_surface)
    W0,jids=skin_matrix(source_surface,source_skeleton,source_skin);P=np.asarray([n.P for n in source_surface.surface_nodes],np.float64)
    target_skeleton=rebound_skeleton(source_skeleton,source_surface,target_surface,parent)
    vparents=vertex_parent_sets(candidate,target_surface,parent);comp,ncomp=components(nbr)
    two=[]
    for i in range(len(nbr)):
        s=set(nbr[i])
        for n in nbr[i]: s.update(nbr[n])
        s.discard(i);two.append(frozenset(s))
    two=tuple(two)
    source_span=float(np.max(P.max(0)-P.min(0)))
    result={"schema":"RealSaS.V9DynamicStretchSourceTopologyAttribution.v1","status":"MEASURED_DIAGNOSTIC_ONLY",
        "product_authority_minted":False,"source_component_count":int(ncomp),"methods":[]}
    for method in METHODS:
        skin=build_skin(method,target_surface,target_skeleton,parent,nbr,P,W0,jids)
        rest,W,edges,rlen,ratio,owner=motion_edge_max(candidate,target_surface,target_skeleton,skin,ctx,cameras)
        span=float(np.max(rest.max(0)-rest.min(0)));classes=[]
        rows=[]
        for ei,(u,v) in enumerate(edges):
            cls=edge_class(vparents[int(u)],vparents[int(v)],nbr,two,comp);classes.append(cls)
            pa=vparents[int(u)];pb=vparents[int(v)]
            mind=min(float(np.linalg.norm(P[x]-P[y])) for x in pa for y in pb)/source_span
            w_l1=float(np.abs(W[int(u)]-W[int(v)]).sum())
            rows.append((float(ratio[ei]),ei,cls,mind,w_l1))
        classes=np.asarray(classes,dtype=object)
        by={}
        for cls in sorted(set(classes.tolist())):
            mask=classes==cls;r=ratio[mask]
            by[cls]={"edge_count":int(mask.sum()),"ratio_p95":q(r,.95),"ratio_p99":q(r,.99),"ratio_max":float(r.max()),
                "gt4":int(np.count_nonzero(r>4.0)),"gt10":int(np.count_nonzero(r>10.0))}
        top=[]
        for rr,ei,cls,mind,w_l1 in sorted(rows,reverse=True)[:50]:
            u,v=map(int,edges[ei])
            top.append({"edge_index":int(ei),"vertex_u":u,"vertex_v":v,"source_topology_class":cls,
                "rest_length_over_span":float(rlen[ei]/span),"max_motion_edge_ratio":rr,
                "owner":owner[ei],"endpoint_u_source_parents":list(vparents[u]),"endpoint_v_source_parents":list(vparents[v]),
                "min_source_parent_distance_over_span":mind,"rest_weight_l1":w_l1})
        m={"method":method,"max_motion_edge_ratio":float(ratio.max()),"gt4":int(np.count_nonzero(ratio>4.0)),
            "gt10":int(np.count_nonzero(ratio>10.0)),"by_source_topology_class":by,"top_edges":top}
        result["methods"].append(m)
        print("SOURCE_TOPOLOGY_ATTRIBUTION="+json.dumps({"method":method,"max":m["max_motion_edge_ratio"],
            "gt4":m["gt4"],"gt10":m["gt10"],"by_class":by,
            "top10":[{"ratio":x["max_motion_edge_ratio"],"class":x["source_topology_class"],
                      "rest_len":x["rest_length_over_span"],"w_l1":x["rest_weight_l1"]} for x in top[:10]]},sort_keys=True),flush=True)
    write(a.out,result)

if __name__=="__main__":main()
