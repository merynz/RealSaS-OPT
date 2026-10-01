from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import _pose_skin_matrices
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask,
    exact,
    face_indices,
    load,
    motion_metrics,
    stress_arbitrary_weights,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import (
    teacher_weights,
)
from tools.audit_knight_topology_region_pair_decomposition_court_v1 import (
    source_components,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


EPSILON=0.05
PROBE_DEGREES=30.0
MAX_MEANSHIFT_SEEDS=4096
PCA_COMPONENTS=32


def _triangle_orientation(P,F):
    tri=P[F]
    e1=tri[:,1]-tri[:,0]
    e2=tri[:,2]-tri[:,0]
    n=np.cross(e1,e2)
    nn=np.linalg.norm(n,axis=1)
    valid=nn>1e-12
    n=n/np.maximum(nn[:,None],1e-12)
    O=np.stack((e1,e2,n),axis=2)
    return O,valid


def _polar_rotations(Fm):
    U,s,Vt=np.linalg.svd(Fm,full_matrices=False)
    R=np.einsum("nij,njk->nik",U,Vt)
    det=np.linalg.det(R)
    bad=det<0.0
    if np.any(bad):
        U2=U.copy()
        U2[bad,:,2]*=-1.0
        R[bad]=np.einsum("nij,njk->nik",U2[bad],Vt[bad])
    return R


def _rotation_signatures(P,F,W,jids,skeleton,cameras,probe_degrees):
    O0,valid=_triangle_orientation(P,F)
    invO=np.zeros_like(O0)
    invO[valid]=np.linalg.inv(O0[valid])
    frames=derive_joint_frames_from_skeleton(skeleton,cameras=cameras)
    hom=np.concatenate((P,np.ones((len(P),1),dtype=np.float64)),axis=1)
    probes=[]
    for jid in sorted(jids):
        for axis in range(3):
            for sign in (-1.0,1.0):
                probes.append((jid,axis,sign*float(probe_degrees)))
    S=len(probes)
    Z=np.zeros((len(F),9*S),dtype=np.float32)
    for pi,(jid,axis,degrees) in enumerate(probes):
        sm=_pose_skin_matrices(
            skeleton,frames,joint_id=jid,local_axis_index=axis,degrees=degrees
        )
        mats=np.stack([sm[x] for x in jids],axis=0)
        per=np.stack([(hom@mats[k].T)[:,:3] for k in range(len(jids))],axis=1)
        posed=np.sum(per*W[:,:,None],axis=1)
        Ot,_=_triangle_orientation(posed,F)
        grad=np.einsum("nij,njk->nik",Ot,invO)
        R=_polar_rotations(grad)
        Z[:,9*pi:9*(pi+1)]=R.reshape(len(F),9).astype(np.float32)
    # James/Twigg physical tolerance: normalize RMS matrix-entry distance so epsilon
    # remains independent of 9S dimensionality.
    Z/=math.sqrt(9.0*S)
    return Z,valid,probes


def _deterministic_seed_indices(points,max_seeds):
    n=len(points)
    if n<=max_seeds:return np.arange(n,dtype=np.int64)
    # Deterministic evenly-spaced sample; no outcome-dependent selection.
    return np.linspace(0,n-1,int(max_seeds),dtype=np.int64)


def _cluster_signatures(Z,face_mask,epsilon):
    idx=np.where(np.asarray(face_mask,dtype=bool))[0]
    if len(idx)<2:raise RuntimeError("JAMES_TWIGG_TOO_FEW_CORE_FACES")
    X=Z[idx].astype(np.float64)
    nc=min(PCA_COMPONENTS,X.shape[0]-1,X.shape[1])
    pca=PCA(n_components=nc,svd_solver="randomized",random_state=0)
    Xp=pca.fit_transform(X)
    seeds=_deterministic_seed_indices(Xp,MAX_MEANSHIFT_SEEDS)
    ms=MeanShift(
        bandwidth=float(epsilon),
        bin_seeding=False,
        cluster_all=True,
        max_iter=200,
        n_jobs=-1,
    )
    ms.fit(Xp[seeds])
    centers=np.asarray(ms.cluster_centers_,dtype=np.float64)
    if not len(centers):raise RuntimeError("JAMES_TWIGG_NO_MODES")
    tree=cKDTree(centers)
    dist,label=tree.query(Xp,k=1)
    face_label=np.full(len(Z),-1,dtype=np.int64)
    face_label[idx]=np.asarray(label,dtype=np.int64)
    # Paper's core criterion is h/4 after mean shift. We cannot cheaply mean-shift all
    # 25k points; preserve a strict proxy: original projected signature within h/4
    # of the discovered mode.
    core=np.zeros(len(Z),dtype=bool)
    core[idx]=np.asarray(dist)<=float(epsilon)/4.0
    return face_label,core,{
        "cluster_count":int(len(centers)),
        "fit_face_count":int(len(idx)),
        "seed_face_count":int(len(seeds)),
        "pca_components":int(nc),
        "pca_explained_variance_ratio_sum":float(np.sum(pca.explained_variance_ratio_)),
        "core_face_count":int(np.count_nonzero(core)),
        "core_fraction_of_fit":float(np.count_nonzero(core)/max(1,len(idx))),
        "mode_distance_p50":float(np.quantile(dist,.5)),
        "mode_distance_p95":float(np.quantile(dist,.95)),
        "mode_distance_max":float(np.max(dist)),
    }


def _face_areas(P,F):
    t=P[F]
    return 0.5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)


def _vertex_labels_from_faces(P,F,face_labels,core,dense_supported,safe_faces):
    n=len(P)
    area=_face_areas(P,F)
    votes=[defaultdict(float) for _ in range(n)]
    usable=core & dense_supported & safe_faces & (face_labels>=0)
    for fi in np.where(usable)[0]:
        lab=int(face_labels[fi]);w=float(area[fi])
        for v in F[fi]:
            votes[int(v)][lab]+=w
    label=np.full(n,-1,dtype=np.int64)
    confidence=np.zeros(n,dtype=np.float64)
    for v,d in enumerate(votes):
        if not d:continue
        rows=sorted(d.items(),key=lambda kv:(-kv[1],kv[0]))
        total=sum(x[1] for x in rows)
        label[v]=int(rows[0][0])
        confidence[v]=float(rows[0][1]/max(total,1e-15))

    # Propagate only through dense-supported, mechanically safe face edges.
    edge_w={}
    for fi in np.where(dense_supported & safe_faces)[0]:
        a,b,c=map(int,F[fi])
        for x,y in ((a,b),(b,c),(c,a)):
            x,y=(x,y) if x<y else (y,x)
            w=float(np.linalg.norm(P[x]-P[y]))
            if w<=1e-15:continue
            edge_w[(x,y)]=min(edge_w.get((x,y),np.inf),w)
    nbr=[[] for _ in range(n)]
    for (a,b),w in edge_w.items():
        nbr[a].append((b,w));nbr[b].append((a,w))
    seeds=np.where(label>=0)[0]
    if not len(seeds):raise RuntimeError("JAMES_TWIGG_NO_VERTEX_SEEDS")
    # Exact multi-source shortest-path propagation on the mechanically-safe graph.
    # This avoids an O(#seeds * #vertices) distance matrix.
    import heapq
    best=np.full(n,np.inf,dtype=np.float64)
    propagated=np.full(n,-1,dtype=np.int64)
    heap=[]
    for s in seeds.tolist():
        best[int(s)]=0.0;propagated[int(s)]=int(label[int(s)])
        heapq.heappush(heap,(0.0,int(propagated[int(s)]),int(s)))
    while heap:
        d,lab,u=heapq.heappop(heap)
        if d!=best[u] or lab!=propagated[u]:continue
        for v,w in nbr[u]:
            nd=d+w
            if nd<best[v]-1e-15 or (abs(nd-best[v])<=1e-15 and (propagated[v]<0 or lab<propagated[v])):
                best[v]=nd;propagated[v]=lab
                heapq.heappush(heap,(nd,lab,v))
    unresolved=~np.isfinite(best)
    missing=(label<0)&(~unresolved)
    label[missing]=propagated[missing]
    confidence[missing]=0.5
    return label,confidence,{
        "initial_labeled_vertex_count":int(np.count_nonzero(seeds)),
        "propagated_vertex_count":int(np.count_nonzero(missing)),
        "unresolved_vertex_count":int(np.count_nonzero(unresolved)),
        "safe_graph_edge_count":int(len(edge_w)),
    }


def _split_holeless(P0,W0,F,label):
    P=[x.copy() for x in P0]
    W=[x.copy() for x in W0]
    region=[int(x) for x in label]
    seam_cache={}
    centroid_cache={}

    def area(face):
        a,b,c=np.asarray(P,dtype=np.float64)[np.asarray(face,dtype=np.int64)]
        return float(np.linalg.norm(np.cross(b-a,c-a)))

    def seam(a,b,r):
        key=(min(a,b),max(a,b),int(r))
        if key in seam_cache:return seam_cache[key]
        candidates=[v for v in (a,b) if int(label[v])==int(r)]
        if not candidates:return None
        pos=0.5*(P0[a]+P0[b])
        ww=np.mean(W0[np.asarray(candidates,dtype=np.int64)],axis=0)
        ww=np.maximum(ww,0.0);ww/=max(float(ww.sum()),1e-15)
        i=len(P);P.append(pos);W.append(ww);region.append(int(r));seam_cache[key]=i
        return i

    def centroid(fi,r,face):
        key=(int(fi),int(r))
        if key in centroid_cache:return centroid_cache[key]
        candidates=[v for v in face if int(label[v])==int(r)]
        if not candidates:return None
        pos=P0[np.asarray(face,dtype=np.int64)].mean(axis=0)
        ww=np.mean(W0[np.asarray(candidates,dtype=np.int64)],axis=0)
        ww=np.maximum(ww,0.0);ww/=max(float(ww.sum()),1e-15)
        i=len(P);P.append(pos);W.append(ww);region.append(int(r));centroid_cache[key]=i
        return i

    out=[];mixed=0;unsplit_unresolved=0;area_err=[]
    for fi,row in enumerate(F.tolist()):
        face=list(map(int,row));labs=[int(label[v]) for v in face]
        if any(x<0 for x in labs):
            out.append(tuple(face));unsplit_unresolved+=1;continue
        unique=sorted(set(labs))
        before=area(face);made=[]
        if len(unique)==1:
            made=[tuple(face)]
        elif len(unique)==2:
            mixed+=1
            for r in unique:
                own=[i for i in range(3) if labs[i]==r];other=[i for i in range(3) if labs[i]!=r]
                if len(own)==1:
                    i=own[0];j,k=other;vi=face[i]
                    a=seam(vi,face[j],r);b=seam(vi,face[k],r)
                    if a is not None and b is not None:made.append((vi,a,b))
                else:
                    i,j=own;k=other[0];vi,vj,vk=face[i],face[j],face[k]
                    a=seam(vi,vk,r);b=seam(vj,vk,r)
                    if a is not None and b is not None:made.extend(((vi,vj,b),(vi,b,a)))
        elif len(unique)==3:
            mixed+=1
            for i in range(3):
                r=labs[i];vi=face[i];vj=face[(i+1)%3];vk=face[(i-1)%3]
                a=seam(vi,vj,r);b=seam(vk,vi,r);cen=centroid(fi,r,face)
                if a is not None and b is not None and cen is not None:
                    made.extend(((vi,a,cen),(vi,cen,b)))
        if not made:
            out.append(tuple(face));unsplit_unresolved+=1
        else:
            after=sum(area(f) for f in made)
            area_err.append(abs(after-before)/max(before,1e-15))
            out.extend(made)
    P=np.asarray(P,dtype=np.float64);W=np.asarray(W,dtype=np.float64);F2=np.asarray(out,dtype=np.int64)
    W=np.maximum(W,0.0);W/=np.maximum(W.sum(axis=1,keepdims=True),1e-15)
    R=np.asarray(region,dtype=np.int64)
    pure=(R[F2[:,0]]==R[F2[:,1]])&(R[F2[:,1]]==R[F2[:,2]])
    return P,W,F2,{
        "vertex_count_before":int(len(P0)),"vertex_count_after":int(len(P)),
        "face_count_before":int(len(F)),"face_count_after":int(len(F2)),
        "mixed_face_count":int(mixed),"face_deletion_count":0,
        "seam_vertex_copy_count":int(len(seam_cache)),"centroid_copy_count":int(len(centroid_cache)),
        "unsplit_unresolved_face_count":int(unsplit_unresolved),
        "region_pure_output_fraction":float(np.mean(pure)),
        "rest_area_error_max":float(max(area_err,default=0.0)),
        "rest_area_error_p99":float(np.quantile(area_err,.99)) if area_err else 0.0,
    }


def _teacher_region_eval(bank,source,skeleton,candidate,inferred,F,unsafe):
    Wt,jids,tri,sf=teacher_weights(bank,source,skeleton,candidate)
    src_comp=source_components(int(sf.max())+1,sf)
    tri_comp=np.asarray([src_comp[int(row[0])] for row in sf],dtype=np.int64)
    truth=np.asarray([tri_comp[int(t)] for t in tri],dtype=np.int64)
    mask=inferred>=0
    ari=adjusted_rand_score(truth[mask],inferred[mask]) if np.count_nonzero(mask)>1 else None
    nmi=normalized_mutual_info_score(truth[mask],inferred[mask]) if np.count_nonzero(mask)>1 else None
    truth_mixed=np.asarray([len(set(truth[f].tolist()))>1 for f in F],dtype=bool)
    pred_mixed=np.asarray([len(set(inferred[f].tolist()))>1 if np.all(inferred[f]>=0) else False for f in F],dtype=bool)
    tp=int(np.count_nonzero(truth_mixed&pred_mixed));fp=int(np.count_nonzero((~truth_mixed)&pred_mixed));fn=int(np.count_nonzero(truth_mixed&(~pred_mixed)))
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    unsafe_truth=unsafe&truth_mixed
    unsafe_recall=float(np.count_nonzero(unsafe_truth&pred_mixed)/max(1,np.count_nonzero(unsafe_truth)))
    return {
        "vertex_ari":None if ari is None else float(ari),
        "vertex_nmi":None if nmi is None else float(nmi),
        "truth_mixed_face_count":int(np.count_nonzero(truth_mixed)),
        "predicted_mixed_face_count":int(np.count_nonzero(pred_mixed)),
        "mixed_face_precision":float(precision),
        "mixed_face_recall":float(recall),
        "unsafe_truth_cross_region_face_count":int(np.count_nonzero(unsafe_truth)),
        "unsafe_truth_cross_region_recall":unsafe_recall,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--weights-npz",type=Path,required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--teacher-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)

    P=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    F=face_indices(cand)
    with np.load(a.weights_npz,allow_pickle=False) as z:
        jids=tuple(map(str,z["joint_ids"].tolist()))
        W=np.asarray(z["selected"],dtype=np.float64)
        selected_variant=str(np.asarray(z["selected_variant"]).item())
        selected_alpha=float(np.asarray(z["selected_alpha"]).item())
    if W.shape!=(len(P),len(jids)):raise RuntimeError("JAMES_TWIGG_WEIGHT_SHAPE_DRIFT")

    surface=load(
        rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",
        rigging_surface_from_dict,
    )
    dense=dense_supported_face_mask(rr,cand,surface)
    base_stress=stress_arbitrary_weights(P,W,F,jids,sk,cams,env,policy)
    unsafe=np.zeros(len(F),dtype=bool)
    unsafe[np.asarray(base_stress["unsafe_face_indices"],dtype=np.int64)]=True
    safe=(~unsafe)&dense

    Z,valid,probes=_rotation_signatures(P,F,W,jids,sk,cams,PROBE_DEGREES)
    fit=safe&valid
    face_label,core,cluster_meta=_cluster_signatures(Z,fit,EPSILON)
    vlabel,vconf,prop_meta=_vertex_labels_from_faces(P,F,face_label,core,dense,safe)

    teacher_eval=_teacher_region_eval(a.teacher_bank,a.teacher_source,sk,cand,vlabel,F,unsafe)

    P2,W2,F2,split_meta=_split_holeless(P,W,F,vlabel)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    before_motion=motion_metrics(P,W,F,jids,sk,cams,rr,source_report)
    after_motion=motion_metrics(P2,W2,F2,jids,sk,cams,rr,source_report)
    after_stress=stress_arbitrary_weights(P2,W2,F2,jids,sk,cams,env,policy)

    report={
        "schema":"RealSaS.KnightJamesTwiggRegionInferenceCourt.v1",
        "status":"PAPER_INSPIRED_TEACHER_FREE_REGION_INFERENCE__NO_PRODUCT_MUTATION",
        "paper":{
            "title":"Skinning Mesh Animations",
            "authors":"Doug L. James; Christopher D. Twigg",
            "year":2005,
            "rotation_feature":"polar(F_t)=R_t W_t; concatenate row-major vec(R_t) over probes",
            "epsilon":EPSILON,
            "adaptation":"Stage35-style controlled joint probes replace an observed mesh animation sequence. PCA and deterministic seed subsampling accelerate mean-shift while preserving the paper's rotation-sequence feature and epsilon-scale intent.",
        },
        "teacher_used_by_region_inference":False,
        "cutcell_input":{"selected_variant":selected_variant,"selected_alpha":selected_alpha},
        "probe":{"degrees":PROBE_DEGREES,"probe_count":len(probes)},
        "clustering":cluster_meta,
        "vertex_label_propagation":prop_meta,
        "teacher_evaluation_only":teacher_eval,
        "holeless_split":split_meta,
        "mechanics":{
            "before_stress":{k:v for k,v in base_stress.items() if k!="unsafe_face_indices"},
            "after_stress":{k:v for k,v in after_stress.items() if k!="unsafe_face_indices"},
            "before_motion":{k:v for k,v in before_motion.items() if k!="frames"},
            "after_motion":{k:v for k,v in after_motion.items() if k!="frames"},
        },
        "finding":{
            "region_inference_labels_all_vertices":bool(prop_meta["unresolved_vertex_count"]==0),
            "teacher_free_holeless_split_reduces_gt10":bool(after_motion["max_edge_gt_10"]<before_motion["max_edge_gt_10"]),
            "teacher_free_holeless_split_reduces_gt4":bool(after_motion["max_edge_gt_4"]<before_motion["max_edge_gt_4"]),
            "teacher_free_holeless_split_reduces_synthetic_unsafe":bool(after_stress["unsafe_face_count"]<base_stress["unsafe_face_count"]),
        },
        "claim_boundary":"Teacher source topology is evaluation-only. This experiment tests whether James/Twigg rotation-sequence clustering can replace the teacher component labels used by the successful holeless split oracle. PCA/seed subsampling are acceleration adaptations, so a negative result would not falsify the original 2005 method.",
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_JAMES_TWIGG_REGION_INFERENCE_PASS",json.dumps({
        "clustering":cluster_meta,
        "propagation":prop_meta,
        "teacher_eval":teacher_eval,
        "split":split_meta,
        "before_motion":report["mechanics"]["before_motion"],
        "after_motion":report["mechanics"]["after_motion"],
        "before_stress":report["mechanics"]["before_stress"],
        "after_stress":report["mechanics"]["after_stress"],
        "finding":report["finding"],
    },sort_keys=True))

if __name__=="__main__":
    main()
