"""Audit finite-element skin prolongation from the archived coarse carrier to V9.

The archived Stage18 candidate defines a piecewise-linear skin field at its mesh
vertices. Each refined V9 surface node has an exact coarse parent from the shared
dense-source inverse maps. We evaluate only coarse faces incident to that parent
(or its one-ring as a fail-closed fallback), project the refined point to the
nearest such triangle, and barycentrically interpolate the archived skin field.

This is topology-local remesh transfer, not learned inference and not product
authority.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,qualified_skeleton_from_dict,
    qualified_skin_from_dict,rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.product_mesh_skin_v1 import _skin_support_coefficients
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import SkinInfluenceProposal,SkinProposalIR
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from tools.audit_topology_aware_refinement_skin_prolongation_v1 import (
    candidate_gradient,rebound_skeleton,source_graph,target_parent_map,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,sort_keys=True,indent=2)+"\n")
def q(a,p): return float(np.quantile(np.asarray(a,dtype=np.float64),p))

def closest_triangle_barycentric(p,a,b,c):
    ab=b-a; ac=c-a; ap=p-a
    d1=float(np.dot(ab,ap)); d2=float(np.dot(ac,ap))
    if d1<=0.0 and d2<=0.0: return np.array([1.,0.,0.]),a
    bp=p-b; d3=float(np.dot(ab,bp)); d4=float(np.dot(ac,bp))
    if d3>=0.0 and d4<=d3: return np.array([0.,1.,0.]),b
    vc=d1*d4-d3*d2
    if vc<=0.0 and d1>=0.0 and d3<=0.0:
        v=d1/(d1-d3); bary=np.array([1.-v,v,0.]); return bary,a+v*ab
    cp=p-c; d5=float(np.dot(ab,cp)); d6=float(np.dot(ac,cp))
    if d6>=0.0 and d5<=d6: return np.array([0.,0.,1.]),c
    vb=d5*d2-d1*d6
    if vb<=0.0 and d2>=0.0 and d6<=0.0:
        w=d2/(d2-d6); bary=np.array([1.-w,0.,w]); return bary,a+w*ac
    va=d3*d6-d5*d4
    if va<=0.0 and (d4-d3)>=0.0 and (d5-d6)>=0.0:
        w=(d4-d3)/((d4-d3)+(d5-d6)); bary=np.array([0.,1.-w,w]); return bary,b+w*(c-b)
    denom=va+vb+vc
    if abs(denom)<=1e-30: raise RuntimeError("FEM_COARSE_TRIANGLE_DEGENERATE")
    inv=1.0/denom; v=vb*inv; w=vc*inv
    bary=np.array([1.-v-w,v,w],dtype=np.float64)
    return bary,bary[0]*a+bary[1]*b+bary[2]*c

def coarse_face_candidates(coarse,source_surface,nbr):
    sid_to_i={str(n.surface_id):i for i,n in enumerate(source_surface.surface_nodes)}
    faces_by=[set() for _ in source_surface.surface_nodes]
    vertex_support=[]
    for v in coarse.vertices:
        ids=set()
        for sid,c in _skin_support_coefficients(v):
            if float(c)<=0: continue
            idx=sid_to_i.get(str(sid))
            if idx is None: raise RuntimeError("FEM_COARSE_VERTEX_SUPPORT_UNKNOWN")
            ids.add(idx)
        if not ids: raise RuntimeError("FEM_COARSE_VERTEX_SUPPORT_EMPTY")
        vertex_support.append(ids)
    vi={v.candidate_vertex_id:i for i,v in enumerate(coarse.vertices)}
    faces=[]
    for fi,f in enumerate(coarse.faces):
        idx=tuple(vi[x] for x in f); faces.append(idx)
        support=set()
        for v in idx: support.update(vertex_support[v])
        for s in support: faces_by[s].add(fi)
    return tuple(faces),tuple(tuple(sorted(x)) for x in faces_by)

def nearest_topology_faces(start,faces_by,nbr,cache):
    if start in cache: return cache[start]
    if faces_by[start]:
        value=(faces_by[start],0)
        cache[start]=value
        return value
    seen={int(start)}
    frontier={int(start)}
    hop=0
    while frontier:
        hop+=1
        nxt=set()
        for u in frontier:
            for v in nbr[u]:
                if v not in seen:
                    seen.add(v);nxt.add(v)
        if not nxt: break
        found=set()
        for u in nxt: found.update(faces_by[u])
        if found:
            value=(tuple(sorted(found)),hop)
            cache[start]=value
            return value
        frontier=nxt
    raise RuntimeError("FEM_SOURCE_COMPONENT_HAS_NO_COARSE_FACE")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True);ap.add_argument("--run-id",required=True)
    ap.add_argument("--target-surface-json",type=Path,required=True);ap.add_argument("--v9-candidate-json",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();ctx=_ctx(a.authority_root,a.run_id)
    source_surface=rigging_surface_from_dict(stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    source_skeleton=qualified_skeleton_from_dict(stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    source_skin=qualified_skin_from_dict(stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))
    coarse=canonical_mesh_candidate_from_dict(stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    target_surface=rigging_surface_from_dict(read(a.target_surface_json))
    v9=canonical_mesh_candidate_from_dict(read(a.v9_candidate_json))
    parent=target_parent_map(source_surface,target_surface,a.inverse_npz);nbr,kinds=source_graph(source_surface)
    target_skeleton=rebound_skeleton(source_skeleton,source_surface,target_surface,parent)

    coarse_pos,coarse_W,coarse_faces_matrix=_candidate_skin_matrix(
        coarse,surface=source_surface,skeleton=source_skeleton,skin=source_skin
    )
    coarse_pos=np.asarray(coarse_pos,np.float64);coarse_W=np.asarray(coarse_W,np.float64)
    coarse_faces,faces_by=coarse_face_candidates(coarse,source_surface,nbr)
    if not np.array_equal(np.asarray(coarse_faces,np.int64),np.asarray(coarse_faces_matrix,np.int64)):
        raise RuntimeError("FEM_COARSE_FACE_ORDER_DRIFT")
    jids=tuple(j.canonical_joint_id for j in source_skeleton.joints)
    X=np.asarray([n.P for n in target_surface.surface_nodes],np.float64)
    span=float(np.max(coarse_pos.max(0)-coarse_pos.min(0)))
    W=np.zeros((len(X),len(jids)),np.float64)
    residual=np.empty(len(X),np.float64);fallback_hops=np.empty(len(X),np.int64);bary_min=np.empty(len(X),np.float64)
    face_choice=np.empty(len(X),np.int64)
    topology_face_cache={}

    for t,x in enumerate(X):
        p=int(parent[t]); candidates,hop=nearest_topology_faces(p,faces_by,nbr,topology_face_cache)
        fallback_hops[t]=int(hop)
        best=None
        for fi in candidates:
            ia,ib,ic=coarse_faces[fi]
            bary,proj=closest_triangle_barycentric(x,coarse_pos[ia],coarse_pos[ib],coarse_pos[ic])
            d2=float(np.dot(x-proj,x-proj))
            key=(d2,int(fi))
            if best is None or key<best[0]: best=(key,bary,(ia,ib,ic),proj)
        _,bary,idx,proj=best
        W[t]=bary[0]*coarse_W[idx[0]]+bary[1]*coarse_W[idx[1]]+bary[2]*coarse_W[idx[2]]
        residual[t]=float(np.linalg.norm(x-proj)/span);bary_min[t]=float(np.min(bary));face_choice[t]=int(best[0][1])
    if np.any(W<-1e-12) or not np.allclose(W.sum(1),1.0,atol=1e-8,rtol=0):
        raise RuntimeError("FEM_PROLONGATED_SKIN_SIMPLEX_INVALID")
    W=np.maximum(W,0);W/=W.sum(1,keepdims=True)

    influences=[]
    for t,node in enumerate(target_surface.surface_nodes):
        for j,jid in enumerate(jids):
            w=float(W[t,j])
            if w>1e-15: influences.append(SkinInfluenceProposal(str(node.surface_id),jid,w))
    proposal=SkinProposalIR(tuple(influences),target_surface.geometry_lineage_hash,target_skeleton.skeleton_lineage_hash,
        model_provenance="COMPILER_TOPOLOGY_LOCAL_COARSE_MESH_FEM_PROLONGATION_AUDIT_V1",
        metadata={"diagnostic_only":True,"product_authority_minted":False,"global_knn_used":False,
                  "coarse_field_source":"ARCHIVED_STAGE18_PLUS_STAGE32"})
    skin=qualify_skin(target_surface,target_skeleton,proposal,max_simplex_repair_l1=1e-8,max_total_correction_l1=1e-4)
    grad=candidate_gradient(v9,target_surface,skin,target_skeleton)
    stats={"schema":"RealSaS.CoarseMeshFEMSkinProlongationAudit.v1","status":"DIAGNOSTIC_ONLY",
        "product_authority_minted":False,"training_used":False,"global_knn_used":False,
        "method":"TOPOLOGY_LOCAL_CLOSEST_COARSE_TRIANGLE_BARYCENTRIC",
        "source_relation_kind_counts":kinds,"source_node_count":len(source_surface.surface_nodes),
        "target_node_count":len(target_surface.surface_nodes),"coarse_vertex_count":len(coarse.vertices),
        "coarse_face_count":len(coarse.faces),
        "local_face_fallback_count":int(np.count_nonzero(fallback_hops)),
        "topology_face_fallback_hops":{"p50":q(fallback_hops,.5),"p95":q(fallback_hops,.95),
            "p99":q(fallback_hops,.99),"max":int(np.max(fallback_hops))},
        "projection_residual_over_span":{"p50":q(residual,.5),"p95":q(residual,.95),"p99":q(residual,.99),"max":float(residual.max())},
        "minimum_barycentric":{"min":float(bary_min.min()),"p01":q(bary_min,.01),"p50":q(bary_min,.5)},
        **grad,"skin_lineage_hash":skin.skin_lineage_hash,"target_skeleton_lineage_hash":target_skeleton.skeleton_lineage_hash}
    a.out_dir.mkdir(parents=True,exist_ok=True)
    write(a.out_dir/"fresh_qualified_skeleton.json",target_skeleton.to_dict())
    write(a.out_dir/"fresh_qualified_skin.json",skin.to_dict())
    write(a.out_dir/"rig_REPORT.json",{"status":"DIAGNOSTIC_ARCHIVED_KINEMATICS_EXACT_SUPPORT_REBOUND","product_authority_minted":False})
    write(a.out_dir/"skin_REPORT.json",stats);write(a.out_dir/"FEM_STATS.json",stats)
    print("FEM_PROLONGATION_PREP="+json.dumps(stats,sort_keys=True),flush=True)

if __name__=="__main__":main()
