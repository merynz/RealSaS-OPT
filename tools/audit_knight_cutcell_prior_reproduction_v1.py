from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask,
    exact,
    face_indices,
    l1_summary,
    load,
    motion_metrics,
    stress_arbitrary_weights,
    teacher_matrix,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
)

ALPHAS=(0.0,0.25,0.5,0.75,1.0)
GAMMAS=(0.25,0.5,0.75)


def _quant_key(p: np.ndarray, tol: float):
    return tuple(np.rint(np.asarray(p,dtype=np.float64)/tol).astype(np.int64).tolist())


def _point_segment_distance(points,p0,p1):
    x=np.asarray(points,dtype=np.float64)
    a=np.asarray(p0,dtype=np.float64); b=np.asarray(p1,dtype=np.float64)
    ab=b-a; d=float(np.dot(ab,ab))
    if d<=1e-20:
        return np.linalg.norm(x-a[None,:],axis=1)
    t=np.clip(((x-a[None,:])@ab)/d,0.0,1.0)
    q=a[None,:]+t[:,None]*ab[None,:]
    return np.linalg.norm(x-q,axis=1)


def _bone_segments(skeleton):
    joints=tuple(skeleton.joints)
    ids=tuple(str(j.canonical_joint_id) for j in joints)
    pos={str(j.canonical_joint_id):np.asarray(j.position,dtype=np.float64) for j in joints}
    parent={str(j.canonical_joint_id):(None if j.parent_canonical_id is None else str(j.parent_canonical_id)) for j in joints}
    out=[]
    for jid in ids:
        par=parent[jid]
        if par is None:
            p0=pos[jid]; p1=pos[jid]
        else:
            p0=pos[par]; p1=pos[jid]
        out.append((jid,p0,p1))
    return ids,out


def _build_cutcell_graph(vertices,faces,resolution):
    import trimesh
    V=np.asarray(vertices,dtype=np.float64)
    F=np.asarray(faces,dtype=np.int64)
    mesh=trimesh.Trimesh(vertices=V,faces=F,process=False,validate=False)

    lo=V.min(axis=0); hi=V.max(axis=0)
    span=hi-lo; extent=float(np.max(span))
    if extent<=1e-12: raise RuntimeError("CUT_CELL_ZERO_EXTENT")
    pad=max(extent*1e-4,1e-7)
    grid_axes=[np.linspace(lo[d],hi[d],int(resolution),dtype=np.float64) for d in range(3)]
    tol=max(extent*1e-8,1e-10)

    coords=[p.copy() for p in V]
    key_to_node={_quant_key(p,tol):i for i,p in enumerate(coords)}
    mesh_count=len(coords)
    edge_w={}
    surface_links=[]

    def node(p):
        k=_quant_key(p,tol)
        i=key_to_node.get(k)
        if i is None:
            i=len(coords); key_to_node[k]=i; coords.append(np.asarray(p,dtype=np.float64).copy())
        return i

    def add_edge(a,b):
        if a==b:return
        x,y=(a,b) if a<b else (b,a)
        w=float(np.linalg.norm(np.asarray(coords[x])-np.asarray(coords[y])))
        if not np.isfinite(w) or w<=1e-15:return
        old=edge_w.get((x,y))
        if old is None or w<old: edge_w[(x,y)]=w

    # Paper Algorithm 1 includes original mesh edges.
    for tri in F.tolist():
        a,b,c=map(int,tri)
        add_edge(a,b);add_edge(b,c);add_edge(c,a)

    intersector=trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    candidate_segments=[]
    classifier_meta={"gwn_available":False,"classifier":"RAY_PARITY_FALLBACK"}

    for axis in range(3):
        orth=[d for d in range(3) if d!=axis]
        origins=[]; dirs=[]
        ray_ij=[]
        for i,u in enumerate(grid_axes[orth[0]]):
            for j,v in enumerate(grid_axes[orth[1]]):
                o=np.zeros(3,dtype=np.float64)
                o[axis]=lo[axis]-pad
                o[orth[0]]=u; o[orth[1]]=v
                d=np.zeros(3,dtype=np.float64); d[axis]=1.0
                origins.append(o);dirs.append(d);ray_ij.append((i,j))
        origins=np.asarray(origins);dirs=np.asarray(dirs)
        loc,iray,itri=intersector.intersects_location(origins,dirs,multiple_hits=True)
        grouped=[[] for _ in range(len(origins))]
        for p,r,t in zip(loc,iray,itri):
            grouped[int(r)].append((float(p[axis]),np.asarray(p,dtype=np.float64),int(t)))

        for rid,hits in enumerate(grouped):
            hits.sort(key=lambda x:x[0])
            # Deduplicate coincident triangle hits on shared edges.
            uniq=[]
            for h in hits:
                if uniq and abs(h[0]-uniq[-1][0])<=tol:
                    uniq[-1][2].add(h[2])
                else:
                    uniq.append([h[0],h[1],{h[2]}])
            if not uniq:
                continue
            i,j=ray_ij[rid]
            fixed=np.zeros(3,dtype=np.float64)
            fixed[orth[0]]=grid_axes[orth[0]][i]
            fixed[orth[1]]=grid_axes[orth[1]][j]

            breaks=[]
            for k,x in enumerate(grid_axes[axis]):
                p=fixed.copy();p[axis]=x
                breaks.append((float(x),p,False,set()))
            for x,p,ts in uniq:
                if lo[axis]-tol<=x<=hi[axis]+tol:
                    breaks.append((x,p,True,set(ts)))
            breaks.sort(key=lambda x:x[0])
            ded=[]
            for b in breaks:
                if ded and abs(b[0]-ded[-1][0])<=tol:
                    ded[-1][2]=ded[-1][2] or b[2]
                    ded[-1][3].update(b[3])
                else:
                    ded.append([b[0],b[1].copy(),bool(b[2]),set(b[3])])

            hit_positions=np.asarray([u[0] for u in uniq],dtype=np.float64)
            for b0,b1 in zip(ded[:-1],ded[1:]):
                if b1[0]-b0[0]<=tol:continue
                mid=0.5*(b0[1]+b1[1])
                # A ray begins outside the bbox: odd number of crossings before midpoint => inside.
                parity_inside=bool(np.count_nonzero(hit_positions < mid[axis]-tol)%2==1)
                candidate_segments.append((b0,b1,mid,parity_inside))

            for b in ded:
                if b[2] and b[3]:
                    si=node(b[1])
                    for ti in b[3]:
                        tri=F[int(ti)]
                        for mv in tri.tolist():
                            surface_links.append((si,int(mv)))

    # Prefer the paper's generalized winding-number classification when libigl is available.
    mids=np.asarray([x[2] for x in candidate_segments],dtype=np.float64)
    inside=None
    if len(mids):
        try:
            import igl
            fn=getattr(igl,"fast_winding_number_for_meshes",None)
            if fn is not None:
                vals=[]
                chunk=8192
                for s in range(0,len(mids),chunk):
                    vals.append(np.asarray(fn(V,F,mids[s:s+chunk]),dtype=np.float64).reshape(-1))
                wn=np.concatenate(vals)
                inside=wn>=0.5
                classifier_meta={"gwn_available":True,"classifier":"LIBIGL_FAST_WINDING_NUMBER_GE_0_5","winding_min":float(wn.min()),"winding_max":float(wn.max())}
        except Exception as e:
            classifier_meta={"gwn_available":False,"classifier":"RAY_PARITY_FALLBACK","gwn_error":repr(e)}
    if inside is None:
        inside=np.asarray([bool(x[3]) for x in candidate_segments],dtype=bool)

    kept_segments=0
    for keep,(b0,b1,mid,parity) in zip(inside,candidate_segments):
        if not bool(keep):continue
        a=node(b0[1]);b=node(b1[1]);add_edge(a,b);kept_segments+=1
    for a,b in surface_links:add_edge(a,b)

    C=np.asarray(coords,dtype=np.float64)
    nbr=[[] for _ in range(len(C))]
    for (a,b),w in edge_w.items():
        nbr[a].append((b,w));nbr[b].append((a,w))

    return C,nbr,{
        "mesh_vertex_count":int(mesh_count),
        "graph_vertex_count":int(len(C)),
        "graph_edge_count":int(len(edge_w)),
        "candidate_segment_count":int(len(candidate_segments)),
        "interior_segment_count":int(kept_segments),
        "surface_link_count":int(len(surface_links)),
        "resolution":int(resolution),
        **classifier_meta,
        "mesh_is_watertight_trimesh":bool(mesh.is_watertight),
    }


def _multisource_dijkstra(nbr,sources,initial):
    n=len(nbr)
    dist=np.full(n,np.inf,dtype=np.float64)
    heap=[]
    for s,d in zip(sources,initial):
        s=int(s);d=float(d)
        if d<dist[s]:
            dist[s]=d;heapq.heappush(heap,(d,s))
    while heap:
        d,u=heapq.heappop(heap)
        if d!=dist[u]:continue
        for v,w in nbr[u]:
            nd=d+w
            if nd<dist[v]:
                dist[v]=nd;heapq.heappush(heap,(nd,v))
    return dist


def _cutcell_distances(graph_pos,nbr,mesh_vertex_count,bones,samples=5):
    tree=cKDTree(graph_pos)
    D=np.full((mesh_vertex_count,len(bones)),np.inf,dtype=np.float64)
    for bi,(jid,p0,p1) in enumerate(bones):
        q=np.linspace(np.asarray(p0),np.asarray(p1),int(samples),dtype=np.float64)
        _,src=tree.query(q,k=1)
        src=np.unique(np.asarray(src,dtype=np.int64))
        delta=_point_segment_distance(graph_pos[src],p0,p1)
        D[:,bi]=_multisource_dijkstra(nbr,src,delta)[:mesh_vertex_count]

    # Algorithm 2 fallback: only vertices unreachable from every bone use top-3 Euclidean bones.
    unreachable=~np.isfinite(D).any(axis=1)
    if np.any(unreachable):
        P=graph_pos[:mesh_vertex_count]
        eu=np.stack([_point_segment_distance(P,p0,p1) for _,p0,p1 in bones],axis=1)
        for vi in np.where(unreachable)[0]:
            order=np.argsort(eu[vi])[:min(3,len(bones))]
            D[vi,order]=eu[vi,order]
    return D,{"unreachable_all_bones_before_fallback":int(np.count_nonzero(unreachable))}


def _prior_from_dist(D,extent,alpha):
    x=D/max(float(extent),1e-12)
    q=(1.0-float(alpha))*x+float(alpha)*x*x
    raw=np.zeros_like(q)
    finite=np.isfinite(q)
    raw[finite]=1.0/np.maximum(q[finite]*q[finite],1e-16)
    s=raw.sum(axis=1,keepdims=True)
    bad=(s[:,0]<=1e-20)
    if np.any(bad):raise RuntimeError(f"CUT_CELL_ZERO_WEIGHT_ROWS:{int(np.count_nonzero(bad))}")
    return raw/s


def _simplex(W):
    X=np.maximum(np.asarray(W,dtype=np.float64),0.0)
    s=X.sum(axis=1,keepdims=True)
    if np.any(s<=1e-15):raise RuntimeError("FUSION_SIMPLEX_COLLAPSE")
    return X/s


def _motion_score(m):
    return (int(m["max_edge_gt_10"]),int(m["max_edge_gt_4"]),float(m["max_edge_p99"]),float(m["worst_edge_max"]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--resolution",type=int,default=32)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    skin=exact(rr,"skin",qualified_skin_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)

    P=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    F=face_indices(cand)
    dense=dense_supported_face_mask(rr,cand,surface)
    Fgraph=F[dense]
    jids,Wpred=_candidate_skin_weights(cand,skin,sk)
    sjids,bones=_bone_segments(sk)
    if tuple(jids)!=tuple(sjids):
        raise RuntimeError("CUT_CELL_JOINT_ORDER_DRIFT")
    extent=float(np.max(P.max(axis=0)-P.min(axis=0)))

    G,nbr,gmeta=_build_cutcell_graph(P,Fgraph,a.resolution)
    D,dmeta=_cutcell_distances(G,nbr,len(P),bones,samples=5)

    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    baseline_motion=motion_metrics(P,Wpred,F,jids,sk,cams,rr,source_report)
    baseline_stress=stress_arbitrary_weights(P,Wpred,F,jids,sk,cams,env,policy)
    unsafe_v=np.zeros(len(P),dtype=bool)
    unsafe_idx=np.asarray(baseline_stress["unsafe_face_indices"],dtype=np.int64)
    if len(unsafe_idx):
        unsafe_v[np.unique(F[unsafe_idx].reshape(-1))]=True

    alpha_rows=[]
    priors={}
    for alpha in ALPHAS:
        W=_prior_from_dist(D,extent,alpha)
        priors[alpha]=W
        mot=motion_metrics(P,W,F,jids,sk,cams,rr,source_report)
        alpha_rows.append({"alpha":alpha,"motion":{k:v for k,v in mot.items() if k!="frames"},"score":list(_motion_score(mot))})
    # Teacher-free selection: actual-motion mechanical score only.
    best_row=min(alpha_rows,key=lambda r:tuple(r["score"]))
    best_alpha=float(best_row["alpha"]);Wcut=priors[best_alpha]

    variants={"ARACHNE":Wpred,"CUT_CELL":Wcut}
    for gamma in GAMMAS:
        variants[f"BLEND_{gamma:.2f}"]=_simplex((1.0-gamma)*Wpred+gamma*Wcut)
    Wreplace=Wpred.copy();Wreplace[unsafe_v]=Wcut[unsafe_v];variants["UNSAFE_VERTEX_REPLACE"]=Wreplace

    variant_rows={}
    for name,W in variants.items():
        mot=motion_metrics(P,W,F,jids,sk,cams,rr,source_report)
        variant_rows[name]={"motion":{k:v for k,v in mot.items() if k!="frames"},"score":list(_motion_score(mot))}
    best_variant=min((k for k in variants if k!="ARACHNE"),key=lambda k:tuple(variant_rows[k]["score"]))

    # Full synthetic stress only on baseline, pure prior, and best teacher-free fusion.
    for name in ("ARACHNE","CUT_CELL",best_variant):
        if "stress" not in variant_rows[name]:
            st=stress_arbitrary_weights(P,variants[name],F,jids,sk,cams,env,policy)
            variant_rows[name]["stress"]={k:v for k,v in st.items() if k!="unsafe_face_indices"}

    # Teacher is evaluation only.
    Wteach,valid=teacher_matrix(a.teacher_bank,sk,cand,jids)
    for name,W in variants.items():
        variant_rows[name]["teacher_eval_only"]={
            "valid":l1_summary(W,Wteach,valid),
            "invalid":l1_summary(W,Wteach,~valid),
        }

    weights_out=a.out.with_suffix(".npz")
    np.savez_compressed(
        weights_out,
        joint_ids=np.asarray(jids),
        arachne=Wpred.astype(np.float32),
        cutcell=Wcut.astype(np.float32),
        selected=variants[best_variant].astype(np.float32),
        selected_variant=np.asarray(best_variant),
        selected_alpha=np.asarray(best_alpha,dtype=np.float64),
        unsafe_vertex_mask=unsafe_v.astype(np.uint8),
    )

    report={
        "schema":"RealSaS.KnightCutCellPriorReproduction.v1",
        "status":"PAPER_REPRO_DIAGNOSTIC__NO_PRODUCT_MUTATION",
        "paper":{
            "title":"A Geodesic Cut-Cell Prior for Neural Skinning",
            "arxiv":"2608.11272",
            "algorithm_1_adaptation":"CUT_CELL_GRAPH_MESH_PLUS_GRID_PLUS_SURFACE_INTERSECTIONS",
            "algorithm_2":"N5_BONE_SAMPLES_MULTI_SOURCE_DIJKSTRA_TOP3_EUCLIDEAN_ALL_UNREACHABLE_FALLBACK",
            "eq1":"w=1/(((1-alpha)*(d/D)+alpha*(d/D)^2)^2), then partition-of-unity",
            "deviation":"Graph interior uses libigl fast winding number when available; deterministic ray-parity fallback otherwise. Input is current dense-face-supported canonical mesh, not teacher topology.",
        },
        "teacher_used_by_solver":False,
        "graph":{**gmeta,**dmeta,"dense_supported_input_face_count":int(len(Fgraph)),"candidate_face_count":int(len(F))},
        "alpha_sweep":alpha_rows,
        "selected_alpha_by_teacher_free_actual_motion":best_alpha,
        "unsafe_vertex_count_from_baseline_stage35":int(np.count_nonzero(unsafe_v)),
        "variants":variant_rows,
        "selected_variant_by_teacher_free_actual_motion":best_variant,
        "weights_npz_path":str(weights_out),
        "finding":{
            "cutcell_improves_actual_motion_over_arachne":bool(tuple(variant_rows["CUT_CELL"]["score"])<tuple(variant_rows["ARACHNE"]["score"])),
            "some_cutcell_fusion_improves_actual_motion_over_arachne":bool(tuple(variant_rows[best_variant]["score"])<tuple(variant_rows["ARACHNE"]["score"])),
            "best_variant":best_variant,
        },
        "claim_boundary":"This reproduces the paper's geometric prior and tests simple inference-time fusions. The paper's strongest neural results retrain models with the prior; a negative fusion result does not falsify the paper. Teacher weights are evaluation-only.",
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_CUTCELL_PRIOR_REPRO_PASS",json.dumps({
        "graph":report["graph"],
        "best_alpha":best_alpha,
        "best_variant":best_variant,
        "baseline":variant_rows["ARACHNE"],
        "cutcell":variant_rows["CUT_CELL"],
        "best":variant_rows[best_variant],
        "finding":report["finding"],
    },sort_keys=True))

if __name__=="__main__":
    main()
