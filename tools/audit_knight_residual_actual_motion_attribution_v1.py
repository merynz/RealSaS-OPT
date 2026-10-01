from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from tools.audit_knight_preloaded_mutex_causal_court_v1 import _solve_preloaded
from tools.audit_knight_mechanical_mutex_evidence_policy_court_v1 import _edge_jsd
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import (
    _edge_stretch,
    _edge_table,
    _truth_region,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    CLIPS,
    _joint_pose_v2,
    _skin,
    _tracks_for_clip,
    dense_supported_face_mask,
    exact,
    face_indices,
    load,
    stress_arbitrary_weights,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.demo.render_knight_motion_preview_v1 import _ctx

PREREG=Path("canonical/KNIGHT_RESIDUAL_ACTUAL_MOTION_ATTRIBUTION_PREREG_V1_20260929.json")


def _actual_face_max(rest, weights, faces, joint_ids, skeleton, cameras, rr, source_report):
    max_edge=np.ones(len(faces),dtype=np.float64)
    worst_clip=np.full(len(faces),"REST",dtype=object)
    worst_frame=np.full(len(faces),-1,dtype=np.int64)
    for clip in CLIPS:
        payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text())
        tracks,_=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        for frame,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,weights,joint_ids,mats)
            r=rest[faces];p=posed[faces]
            rl=np.stack([
                np.linalg.norm(r[:,1]-r[:,0],axis=1),
                np.linalg.norm(r[:,2]-r[:,1],axis=1),
                np.linalg.norm(r[:,0]-r[:,2],axis=1),
            ],axis=1)
            pl=np.stack([
                np.linalg.norm(p[:,1]-p[:,0],axis=1),
                np.linalg.norm(p[:,2]-p[:,1],axis=1),
                np.linalg.norm(p[:,0]-p[:,2],axis=1),
            ],axis=1)
            e=(pl/np.maximum(rl,1e-15)).max(axis=1)
            hit=e>max_edge
            max_edge[hit]=e[hit]
            worst_clip[hit]=clip
            worst_frame[hit]=int(frame)
    return max_edge,worst_clip,worst_frame


def _bucket(mask, actual, unsafe):
    idx=np.where(mask)[0]
    out={"face_count":int(len(idx))}
    for th in (4.0,10.0):
        cat=idx[actual[idx]>th]
        unsafe_cat=cat[unsafe[cat]]
        safe_cat=cat[~unsafe[cat]]
        out[f"actual_gt_{int(th)}"]={
            "count":int(len(cat)),
            "stage35_unsafe_count":int(len(unsafe_cat)),
            "stage35_safe_count":int(len(safe_cat)),
            "stage35_safe_fraction":float(len(safe_cat)/max(1,len(cat))),
        }
    if len(idx):
        out["actual_edge_p50"]=float(np.quantile(actual[idx],.5))
        out["actual_edge_p95"]=float(np.quantile(actual[idx],.95))
        out["actual_edge_max"]=float(np.max(actual[idx]))
    else:
        out["actual_edge_p50"]=out["actual_edge_p95"]=out["actual_edge_max"]=None
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--weights-npz",type=Path,required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--teacher-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    prereg=json.loads(PREREG.read_text())
    if prereg.get("status")!="FROZEN_BEFORE_RESIDUAL_ATTRIBUTION_RESULT":
        raise RuntimeError("RESIDUAL_ATTRIBUTION_PREREG_DRIFT")

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)
    surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)

    P=np.asarray([v.P for v in cand.vertices],np.float64)
    F=face_indices(cand)
    dense=dense_supported_face_mask(rr,cand,surface)

    with np.load(a.weights_npz,allow_pickle=False) as z:
        jids=tuple(map(str,z["joint_ids"].tolist()))
        Wa=np.asarray(z["arachne"],np.float64)
    Wa=np.maximum(Wa,0.0);Wa/=np.maximum(Wa.sum(axis=1,keepdims=True),1e-15)

    # Freeze teacher-free Stage35 scope and partition first.
    st=stress_arbitrary_weights(P,Wa,F,jids,sk,cams,env,policy)
    unsafe=np.zeros(len(F),bool)
    unsafe[np.asarray(st["unsafe_face_indices"],np.int64)]=True

    edges,edge_faces,fei=_edge_table(P,F)
    stretch=_edge_stretch(P,Wa,edges,jids,sk,cams,env)
    l1=np.sum(np.abs(Wa[edges[:,0]]-Wa[edges[:,1]]),axis=1)
    jsd=_edge_jsd(Wa,edges)
    dense_edge=np.zeros(len(edges),bool);scope=np.zeros(len(edges),bool)
    for ei,e in enumerate(map(tuple,edges.tolist())):
        fs=edge_faces[e]
        dense_edge[ei]=any(bool(dense[fi]) for fi in fs)
        scope[ei]=any(bool(dense[fi] and unsafe[fi]) for fi in fs)
    repulsive=scope&(stretch>4.0)
    pred,solver=_solve_preloaded(len(P),edges,dense_edge,repulsive,jsd)
    pred_mixed=np.asarray([len(set(int(pred[v]) for v in row))>1 for row in F],dtype=bool)

    # Teacher evaluation starts here.
    truth=_truth_region(a.teacher_bank,a.teacher_source,sk,cand)
    truth_mixed=np.asarray([len(set(int(truth[v]) for v in row))>1 for row in F],dtype=bool)

    Wt,tjids,_,_=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    tix={str(j):i for i,j in enumerate(tjids)}
    missing=[j for j in jids if str(j) not in tix]
    if missing:
        raise RuntimeError("RESIDUAL_ATTRIBUTION_TEACHER_JOINT_DRIFT:"+json.dumps(missing))
    Wt=np.stack([Wt[:,tix[str(j)]] for j in jids],axis=1)
    Wt=np.maximum(Wt,0.0);Wt/=np.maximum(Wt.sum(axis=1,keepdims=True),1e-15)

    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    actual,wclip,wframe=_actual_face_max(P,Wt,F,jids,sk,cams,rr,source_report)

    masks={
      "TRUTH_MIXED_PRED_MIXED":truth_mixed&pred_mixed,
      "TRUTH_MIXED_PRED_SAME":truth_mixed&(~pred_mixed),
      "TRUTH_SAME_PRED_MIXED":(~truth_mixed)&pred_mixed,
      "TRUTH_SAME_PRED_SAME":(~truth_mixed)&(~pred_mixed),
    }
    buckets={name:_bucket(mask,actual,unsafe) for name,mask in masks.items()}

    missed=masks["TRUTH_MIXED_PRED_SAME"]
    missed_gt10=np.where(missed&(actual>10.0))[0]
    missed_gt4=np.where(missed&(actual>4.0))[0]
    safe_fraction_10=float(np.count_nonzero(~unsafe[missed_gt10])/max(1,len(missed_gt10)))
    safe_fraction_4=float(np.count_nonzero(~unsafe[missed_gt4])/max(1,len(missed_gt4)))
    if safe_fraction_10>=.70 and safe_fraction_4>=.70:
        diagnosis="STAGE35_SCOPE_OWNER"
    elif safe_fraction_10<=.30 and safe_fraction_4<=.30:
        diagnosis="PARTITION_OWNER_INSIDE_EXISTING_SCOPE"
    else:
        diagnosis="MIXED_SCOPE_AND_PARTITION"

    top=np.argsort(actual)[::-1][:100]
    top_rows=[{
      "face_index":int(fi),
      "actual_max_edge":float(actual[fi]),
      "worst_clip":str(wclip[fi]),
      "worst_frame":int(wframe[fi]),
      "stage35_unsafe":bool(unsafe[fi]),
      "truth_mixed":bool(truth_mixed[fi]),
      "pred_mixed":bool(pred_mixed[fi]),
      "truth_regions":sorted(set(int(truth[v]) for v in F[fi])),
      "pred_regions":sorted(set(int(pred[v]) for v in F[fi])),
    } for fi in top]

    report={
      "schema":"RealSaS.KnightResidualActualMotionAttributionCourt.v1",
      "status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
      "preregistration":str(PREREG),
      "teacher_used_by_partition":False,
      "partition_solver":solver,
      "stage35_unsafe_face_count":int(np.count_nonzero(unsafe)),
      "truth_mixed_face_count":int(np.count_nonzero(truth_mixed)),
      "pred_mixed_face_count":int(np.count_nonzero(pred_mixed)),
      "buckets":buckets,
      "missed_truth_boundary_scope": {
        "actual_gt10_stage35_safe_fraction":safe_fraction_10,
        "actual_gt4_stage35_safe_fraction":safe_fraction_4,
      },
      "diagnosis":diagnosis,
      "top_actual_motion_faces":top_rows,
      "claim_boundary":"Stage35 unsafe scope and preloaded-mutex partition are frozen from Arachne evidence before teacher topology and teacher weights are loaded."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_RESIDUAL_ACTUAL_MOTION_ATTRIBUTION_PASS",json.dumps({
      "diagnosis":diagnosis,
      "stage35_unsafe_face_count":report["stage35_unsafe_face_count"],
      "truth_mixed_face_count":report["truth_mixed_face_count"],
      "pred_mixed_face_count":report["pred_mixed_face_count"],
      "buckets":buckets,
      "missed_truth_boundary_scope":report["missed_truth_boundary_scope"],
      "top10":top_rows[:10],
    },sort_keys=True))


if __name__=="__main__":
    main()
