from __future__ import annotations

import argparse, json
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


def point_segment_distance(P,A,B):
    AB=B-A
    den=np.sum(AB*AB,axis=1)
    AP=P[:,None,:]-A[None,:,:]
    t=np.sum(AP*AB[None,:,:],axis=2)/np.maximum(den[None,:],1e-15)
    t=np.clip(t,0.0,1.0)
    Q=A[None,:,:]+t[:,:,None]*AB[None,:,:]
    return np.linalg.norm(P[:,None,:]-Q,axis=2)


def neighbors_from_faces(faces,n):
    nbr=[set() for _ in range(n)]
    for a,b,c in faces.tolist():
        nbr[a].update((b,c));nbr[b].update((a,c));nbr[c].update((a,b))
    return [tuple(sorted(x)) for x in nbr]


def candidate_weights(
    W0,
    *,
    topk,
    order,
    pred_rank,
    unsafe_vertex,
    neighbors,
    distances,
):
    W=np.asarray(W0,dtype=np.float64).copy()
    changed=[]
    fallback_current=0
    fallback_neighbors=0
    fallback_distance=0
    for i in np.where(unsafe_vertex & (pred_rank>int(topk)))[0]:
        allowed=np.asarray(order[i,:int(topk)],dtype=np.int64)
        proposal=np.zeros(W.shape[1],dtype=np.float64)

        # First preserve any product prediction mass that already lies in the
        # mechanically local carrier set.
        mass=float(W0[i,allowed].sum())
        if mass>1e-8:
            proposal[allowed]=W0[i,allowed]
            fallback_current+=1
        else:
            # Then ask immediate surface neighbors, but only for weights that also
            # satisfy this vertex's own bone-locality constraint.
            rows=neighbors[int(i)]
            if rows:
                avg=W0[np.asarray(rows,dtype=np.int64)].mean(axis=0)
                m=float(avg[allowed].sum())
                if m>1e-8:
                    proposal[allowed]=avg[allowed]
                    fallback_neighbors+=1
            if proposal.sum()<=1e-12:
                # Last bounded fallback: inverse-distance prior over the local carrier
                # set. This never invents a remote joint.
                d=np.asarray(distances[i,allowed],dtype=np.float64)
                scale=max(float(np.median(d)),1e-6)
                score=np.exp(-d/scale)
                if not np.isfinite(score).all() or float(score.sum())<=1e-12:
                    proposal[int(allowed[0])]=1.0
                else:
                    proposal[allowed]=score
                fallback_distance+=1

        proposal=np.maximum(proposal,0.0)
        s=float(proposal.sum())
        if s<=1e-12:
            raise RuntimeError("BONE_LOCALITY_PROPOSAL_EMPTY")
        proposal/=s
        if float(np.sum(np.abs(proposal-W0[i])))>1e-10:
            W[i]=proposal
            changed.append(int(i))

    return W,{
        "top_k":int(topk),
        "changed_vertex_count":len(changed),
        "fallback_current_mass":fallback_current,
        "fallback_neighbor_mass":fallback_neighbors,
        "fallback_distance_prior":fallback_distance,
        "changed_vertices":changed[:512],
    }


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
    joint_ids,W0=_candidate_skin_weights(cand,skin,sk)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    full=run_skin_topology_compatibility_v1(
        cand,surface=surface,skeleton=sk,skin=skin,envelope=env,cameras=cams,policy=policy,
        risk_l1_min=0.0,stress_all_faces=True,
    )
    unsafe_face=np.zeros(len(faces),dtype=bool)
    unsafe_face[np.asarray(full["unsafe_face_indices"],dtype=np.int64)]=True
    unsafe_vertex=np.zeros(len(rest),dtype=bool)
    unsafe_vertex[np.unique(faces[unsafe_face].reshape(-1))]=True

    joints={str(j.canonical_joint_id):j for j in sk.joints}
    pos={jid:np.asarray(j.position,dtype=np.float64) for jid,j in joints.items()}
    A=[];B=[]
    for jid in joint_ids:
        j=joints[jid]
        if j.parent_canonical_id is None:
            A.append(pos[jid]);B.append(pos[jid])
        else:
            A.append(pos[str(j.parent_canonical_id)]);B.append(pos[jid])
    D=point_segment_distance(rest,np.asarray(A),np.asarray(B))
    order=np.argsort(D,axis=1)
    rank=np.empty_like(order)
    rows=np.arange(len(rest))[:,None]
    rank[rows,order]=np.arange(len(joint_ids))[None,:]
    pred_dom=np.argmax(W0,axis=1)
    pred_rank=rank[np.arange(len(rest)),pred_dom]+1
    nbr=neighbors_from_faces(faces,len(rest))

    baseline_stress=stress_arbitrary_weights(rest,W0,faces,joint_ids,sk,cams,env,policy)
    baseline_motion=motion_metrics(rest,W0,faces,joint_ids,sk,cams,rr,source_report)

    cases=[]
    matrices={}
    for k in (2,4,6,8,12):
        W,meta=candidate_weights(
            W0,topk=k,order=order,pred_rank=pred_rank,unsafe_vertex=unsafe_vertex,
            neighbors=nbr,distances=D,
        )
        st=stress_arbitrary_weights(rest,W,faces,joint_ids,sk,cams,env,policy)
        mot=motion_metrics(rest,W,faces,joint_ids,sk,cams,rr,source_report)
        score=(
            int(st["unsafe_face_count"]),
            float(st["max_edge_p95"]),
            float(st["max_edge_max"]),
            int(meta["changed_vertex_count"]),
        )
        cases.append({"meta":meta,"synthetic_stress":st,"actual_motion":{k0:v for k0,v in mot.items() if k0!="frames"},"selection_score":list(score)})
        matrices[int(k)]=W

    chosen=min(cases,key=lambda x:tuple(x["selection_score"]))
    chosen_k=int(chosen["meta"]["top_k"])
    Wbest=matrices[chosen_k]

    Wteach,teacher_valid=teacher_matrix(a.teacher_bank,sk,cand,joint_ids)
    teacher_eval={
        "baseline_valid":l1_summary(W0,Wteach,teacher_valid),
        "chosen_valid":l1_summary(Wbest,Wteach,teacher_valid),
        "baseline_invalid":l1_summary(W0,Wteach,~teacher_valid),
        "chosen_invalid":l1_summary(Wbest,Wteach,~teacher_valid),
    }

    report={
        "schema":"RealSaS.KnightBoneLocalityConstrainedWeightCourt.v1",
        "status":"DIAGNOSTIC__NO_PRODUCT_AUTHORITY_MUTATION",
        "selection_rule":"LEXICOGRAPHIC_SYNTHETIC_ONLY: unsafe_count, edge_p95, edge_max, changed_count",
        "teacher_used_for_selection":False,
        "baseline":{
            "initial_stage35_unsafe":int(full["unsafe_face_count"]),
            "unsafe_vertex_count":int(unsafe_vertex.sum()),
            "synthetic_stress":baseline_stress,
            "actual_motion":{k:v for k,v in baseline_motion.items() if k!="frames"},
        },
        "cases":cases,
        "chosen_top_k":chosen_k,
        "teacher_evaluation_only":teacher_eval,
        "finding":{
            "chosen_reduces_synthetic_unsafe":bool(chosen["synthetic_stress"]["unsafe_face_count"]<baseline_stress["unsafe_face_count"]),
            "chosen_reduces_actual_gt10":bool(chosen["actual_motion"]["max_edge_gt_10"]<baseline_motion["max_edge_gt_10"]),
            "chosen_reduces_actual_gt4":bool(chosen["actual_motion"]["max_edge_gt_4"]<baseline_motion["max_edge_gt_4"]),
            "chosen_improves_teacher_invalid_mean":bool(teacher_eval["chosen_invalid"]["mean"]<teacher_eval["baseline_invalid"]["mean"]),
        },
        "claim_boundary":"No teacher data enters detection, correction, candidate selection, or stress scoring. Teacher is post-hoc evaluation only."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_BONE_LOCALITY_CONSTRAINED_WEIGHT_COURT_PASS",json.dumps({
        "baseline":report["baseline"],"chosen_top_k":chosen_k,"chosen":chosen,
        "teacher_eval":teacher_eval,"finding":report["finding"]
    },sort_keys=True))

if __name__=="__main__":main()
