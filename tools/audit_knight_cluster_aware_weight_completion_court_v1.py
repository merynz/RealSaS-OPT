from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

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
    build_safe_graph,
    dense_supported_face_mask,
    exact,
    face_indices,
    harmonic_complete,
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


def point_segment_distance(P,A,B):
    AB=B-A
    den=np.sum(AB*AB,axis=1)
    AP=P[:,None,:]-A[None,:,:]
    t=np.sum(AP*AB[None,:,:],axis=2)/np.maximum(den[None,:],1e-15)
    t=np.clip(t,0.0,1.0)
    Q=A[None,:,:]+t[:,:,None]*AB[None,:,:]
    return np.linalg.norm(P[:,None,:]-Q,axis=2)


def neighbors_from_edges(n,edges):
    nbr=[set() for _ in range(n)]
    for a,b in edges:
        nbr[int(a)].add(int(b));nbr[int(b)].add(int(a))
    return [tuple(sorted(x)) for x in nbr]


def suspect_components(
    W,
    pred_rank,
    unsafe_vertex,
    graph_edges,
    *,
    rank_min,
    confidence_min,
):
    dom=np.argmax(W,axis=1)
    conf=np.max(W,axis=1)
    eligible=(pred_rank>int(rank_min))&(conf>=float(confidence_min))
    nbr=neighbors_from_edges(len(W),graph_edges.keys())
    seen=np.zeros(len(W),dtype=bool)
    selected=np.zeros(len(W),dtype=bool)
    components=[]
    for s in np.where(eligible)[0]:
        if seen[s]:continue
        target=int(dom[s])
        q=[int(s)];seen[s]=True;comp=[]
        while q:
            u=q.pop();comp.append(u)
            for v in nbr[u]:
                if seen[v] or not eligible[v] or int(dom[v])!=target:
                    continue
                seen[v]=True;q.append(v)
        touches=bool(np.any(unsafe_vertex[np.asarray(comp,dtype=np.int64)]))
        if touches:
            selected[np.asarray(comp,dtype=np.int64)]=True
        components.append({
            "joint_index":target,
            "size":len(comp),
            "touches_stage35_unsafe":touches,
            "selected":touches,
        })
    components.sort(key=lambda x:x["size"],reverse=True)
    return selected,components


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
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

    faces=face_indices(cand)
    rest=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    jids,W0=_candidate_skin_weights(cand,skin,sk)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    full=run_skin_topology_compatibility_v1(
        cand,surface=surface,skeleton=sk,skin=skin,envelope=env,cameras=cams,policy=policy,
        risk_l1_min=0.0,stress_all_faces=True,
    )
    unsafe_face=np.zeros(len(faces),dtype=bool)
    unsafe_face[np.asarray(full["unsafe_face_indices"],dtype=np.int64)]=True
    unsafe_vertex=np.zeros(len(W0),dtype=bool)
    unsafe_vertex[np.unique(faces[unsafe_face].reshape(-1))]=True

    dense_supported=dense_supported_face_mask(rr,cand,surface)
    # Cluster detection uses actual dense-face-supported graph. Unlike Harmonic V1,
    # mechanically unsafe faces are NOT removed from graph connectivity, because doing
    # so stranded the very wrong clusters that need boundary anchors.
    graph_edges=build_safe_graph(rest,faces,dense_supported)

    joints={str(j.canonical_joint_id):j for j in sk.joints}
    pos={jid:np.asarray(j.position,dtype=np.float64) for jid,j in joints.items()}
    A=[];B=[]
    for jid in jids:
        j=joints[jid]
        if j.parent_canonical_id is None:
            A.append(pos[jid]);B.append(pos[jid])
        else:
            A.append(pos[str(j.parent_canonical_id)]);B.append(pos[jid])
    D=point_segment_distance(rest,np.asarray(A),np.asarray(B))
    order=np.argsort(D,axis=1)
    rank=np.empty_like(order)
    rows=np.arange(len(rest))[:,None]
    rank[rows,order]=np.arange(len(jids))[None,:]
    pred=np.argmax(W0,axis=1)
    pred_rank=rank[np.arange(len(rest)),pred]+1

    baseline_stress=stress_arbitrary_weights(rest,W0,faces,jids,sk,cams,env,policy)
    baseline_motion=motion_metrics(rest,W0,faces,jids,sk,cams,rr,source_report)

    cases=[]
    matrices={}
    for rmin in (4,8,12):
        for conf in (0.90,0.99):
            unknown,components=suspect_components(
                W0,pred_rank,unsafe_vertex,graph_edges,
                rank_min=rmin,confidence_min=conf,
            )
            W,completion=harmonic_complete(W0,unknown,graph_edges)
            st=stress_arbitrary_weights(rest,W,faces,jids,sk,cams,env,policy)
            mot=motion_metrics(rest,W,faces,jids,sk,cams,rr,source_report)
            key=f"r{rmin}_c{conf:.2f}"
            meta={
                "case_id":key,
                "rank_min":rmin,
                "confidence_min":conf,
                "suspect_vertex_count":int(unknown.sum()),
                "selected_component_count":sum(bool(x["selected"]) for x in components),
                "largest_selected_component_sizes":[x["size"] for x in components if x["selected"]][:32],
                "completion":completion,
            }
            score=(int(st["unsafe_face_count"]),float(st["max_edge_p95"]),float(st["max_edge_max"]),int(unknown.sum()))
            cases.append({
                "meta":meta,
                "synthetic_stress":st,
                "actual_motion":{k:v for k,v in mot.items() if k!="frames"},
                "selection_score":list(score),
            })
            matrices[key]=W

    chosen=min(cases,key=lambda x:tuple(x["selection_score"]))
    key=chosen["meta"]["case_id"];Wbest=matrices[key]

    Wteach,valid=teacher_matrix(a.teacher_bank,sk,cand,jids)
    teacher_eval={
        "baseline_valid":l1_summary(W0,Wteach,valid),
        "chosen_valid":l1_summary(Wbest,Wteach,valid),
        "baseline_invalid":l1_summary(W0,Wteach,~valid),
        "chosen_invalid":l1_summary(Wbest,Wteach,~valid),
    }

    report={
        "schema":"RealSaS.KnightClusterAwareWeightCompletionCourt.v1",
        "status":"DIAGNOSTIC__NO_PRODUCT_AUTHORITY_MUTATION",
        "teacher_used_for_detection":False,
        "teacher_used_for_selection":False,
        "operator":{
            "cluster_graph":"DENSE_FACE_SUPPORTED_CURRENT_CANONICAL_GRAPH",
            "cluster_key":"SAME_PREDICTED_DOMINANT_JOINT",
            "cluster_filter":"NONLOCAL_DOMINANT_JOINT + HIGH_PREDICTED_CONFIDENCE",
            "promotion_filter":"CONNECTED_CLUSTER_TOUCHES_STAGE35_ALL_FACE_UNSAFE_VERTEX",
            "completion":"DIRICHLET_GRAPH_HARMONIC_WITH_ENTIRE_SUSPECT_CLUSTER_REMOVED_FROM_ANCHORS",
        },
        "baseline":{
            "stage35_unsafe_face_count":int(full["unsafe_face_count"]),
            "synthetic_stress":baseline_stress,
            "actual_motion":{k:v for k,v in baseline_motion.items() if k!="frames"},
        },
        "cases":cases,
        "chosen_case_id":key,
        "teacher_evaluation_only":teacher_eval,
        "finding":{
            "chosen_reduces_synthetic_unsafe":bool(chosen["synthetic_stress"]["unsafe_face_count"]<baseline_stress["unsafe_face_count"]),
            "chosen_reduces_actual_gt10":bool(chosen["actual_motion"]["max_edge_gt_10"]<baseline_motion["max_edge_gt_10"]),
            "chosen_reduces_actual_gt4":bool(chosen["actual_motion"]["max_edge_gt_4"]<baseline_motion["max_edge_gt_4"]),
            "chosen_improves_teacher_invalid_mean":bool(teacher_eval["chosen_invalid"]["mean"]<teacher_eval["baseline_invalid"]["mean"]),
        },
        "claim_boundary":"This tests whether Harmonic V1 failed because coherent wrong clusters were left as anchors. Teacher is post-hoc evaluation only."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_CLUSTER_AWARE_WEIGHT_COMPLETION_COURT_PASS",json.dumps({
        "baseline":report["baseline"],"chosen_case_id":key,"chosen":chosen,
        "teacher_eval":teacher_eval,"finding":report["finding"]
    },sort_keys=True))

if __name__=="__main__":main()
