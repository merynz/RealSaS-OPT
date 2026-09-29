from __future__ import annotations

import argparse
import json
from collections import Counter
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
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import _pose_skin_matrices
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import _stress_angle
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.audit_knight_mechanical_mutex_evidence_policy_court_v1 import _edge_jsd, _solve
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import _edge_stretch, _edge_table
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    CLIPS,
    dense_supported_face_mask,
    exact,
    face_indices,
    load,
    stress_arbitrary_weights,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.audit_knight_topology_region_pair_decomposition_court_v1 import source_components
from tools.demo.render_knight_motion_preview_v1 import _ctx, _skin, _tracks_for_clip

PREREG=Path("canonical/KNIGHT_RESIDUAL_TOPOLOGY_SEVERITY_PREREG_V1_20260929.json")


def _teacher_truth(bank,source,sk,cand):
    _,_,tri,sf=teacher_weights(bank,source,sk,cand)
    src_comp=source_components(int(sf.max())+1,sf)
    tri_comp=np.asarray([src_comp[int(row[0])] for row in sf],dtype=np.int64)
    return np.asarray([tri_comp[int(t)] for t in tri],dtype=np.int64)


def _actual_face_edge_max(rest,weights,faces,jids,sk,cams,rr,source_report):
    maxe=np.ones(len(faces),dtype=np.float64)
    worst_clip=np.full(len(faces),"REST",dtype=object)
    worst_frame=np.full(len(faces),-1,dtype=np.int64)
    r=rest[faces]
    rl=np.stack([
        np.linalg.norm(r[:,1]-r[:,0],axis=1),
        np.linalg.norm(r[:,2]-r[:,1],axis=1),
        np.linalg.norm(r[:,0]-r[:,2],axis=1),
    ],axis=1)
    for clip in CLIPS:
        payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text())
        tracks,_=_tracks_for_clip(payload,sk,cams,source_report)
        times=np.linspace(0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        for frame,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams)
            posed=_skin(rest,weights,jids,mats)
            p=posed[faces]
            pl=np.stack([
                np.linalg.norm(p[:,1]-p[:,0],axis=1),
                np.linalg.norm(p[:,2]-p[:,1],axis=1),
                np.linalg.norm(p[:,0]-p[:,2],axis=1),
            ],axis=1)
            e=(pl/np.maximum(rl,1e-15)).max(axis=1)
            upd=e>maxe
            maxe[upd]=e[upd]
            worst_clip[upd]=clip
            worst_frame[upd]=int(frame)
    return maxe,worst_clip,worst_frame


def _synthetic_face_edge_max(rest,weights,faces,jids,sk,cams,env):
    stress_angle=_stress_angle(env)
    frames=derive_joint_frames_from_skeleton(sk,cameras=cams)
    hom=np.concatenate([rest,np.ones((len(rest),1),dtype=np.float64)],axis=1)
    r=rest[faces]
    rl=np.stack([
        np.linalg.norm(r[:,1]-r[:,0],axis=1),
        np.linalg.norm(r[:,2]-r[:,1],axis=1),
        np.linalg.norm(r[:,0]-r[:,2],axis=1),
    ],axis=1)
    maxe=np.ones(len(faces),dtype=np.float64)
    for jid in sorted(jids):
        for axis in range(3):
            for sign in (-1.0,1.0):
                sm=_pose_skin_matrices(sk,frames,joint_id=jid,local_axis_index=axis,degrees=sign*float(stress_angle))
                mats=np.stack([sm[x] for x in jids],axis=0)
                per=np.stack([(hom@mats[k].T)[:,:3] for k in range(len(jids))],axis=1)
                posed=np.sum(per*weights[:,:,None],axis=1)
                p=posed[faces]
                pl=np.stack([
                    np.linalg.norm(p[:,1]-p[:,0],axis=1),
                    np.linalg.norm(p[:,2]-p[:,1],axis=1),
                    np.linalg.norm(p[:,0]-p[:,2],axis=1),
                ],axis=1)
                maxe=np.maximum(maxe,(pl/np.maximum(rl,1e-15)).max(axis=1))
    return maxe


def _group(mask,actual_e,synthetic_e,raw_rep_count,max_stretch,max_l1,max_jsd):
    idx=np.where(mask)[0]
    if not len(idx):
        return {"face_count":0}
    def q(a,p): return float(np.quantile(a[idx],p))
    return {
      "face_count":int(len(idx)),
      "actual_gt4":int(np.count_nonzero(actual_e[idx]>4.0)),
      "actual_gt10":int(np.count_nonzero(actual_e[idx]>10.0)),
      "actual_edge_p50":q(actual_e,.5),
      "actual_edge_p95":q(actual_e,.95),
      "actual_edge_max":float(np.max(actual_e[idx])),
      "synthetic_gt4":int(np.count_nonzero(synthetic_e[idx]>4.0)),
      "synthetic_edge_p50":q(synthetic_e,.5),
      "synthetic_edge_p95":q(synthetic_e,.95),
      "synthetic_edge_max":float(np.max(synthetic_e[idx])),
      "raw_repulsive_edge_multiplicity":dict(sorted(Counter(map(int,raw_rep_count[idx].tolist())).items())),
      "max_stage35_stretch_p50":q(max_stretch,.5),
      "max_stage35_stretch_p95":q(max_stretch,.95),
      "max_arachne_l1_p50":q(max_l1,.5),
      "max_arachne_l1_p95":q(max_l1,.95),
      "max_arachne_jsd_p50":q(max_jsd,.5),
      "max_arachne_jsd_p95":q(max_jsd,.95),
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
    if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_RESIDUAL_TOPOLOGY_SEVERITY_RESULT":
        raise RuntimeError("RESIDUAL_TOPOLOGY_SEVERITY_PREREG_DRIFT")

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

    base=stress_arbitrary_weights(P,Wa,F,jids,sk,cams,env,policy)
    unsafe=np.zeros(len(F),bool)
    unsafe[np.asarray(base["unsafe_face_indices"],np.int64)]=True
    edges,edge_faces,fei=_edge_table(P,F)
    stretch=_edge_stretch(P,Wa,edges,jids,sk,cams,env)
    l1=np.sum(np.abs(Wa[edges[:,0]]-Wa[edges[:,1]]),axis=1)
    jsd=_edge_jsd(Wa,edges)
    dense_edge=np.zeros(len(edges),bool);scope=np.zeros(len(edges),bool)
    for ei,e in enumerate(map(tuple,edges.tolist())):
        fs=edge_faces[e]
        dense_edge[ei]=any(bool(dense[fi]) for fi in fs)
        scope[ei]=any(bool(dense[fi] and unsafe[fi]) for fi in fs)
    rep=scope&(stretch>4.0)&(l1>1.0)

    inferred,solver=_solve(len(P),edges,dense_edge,rep,jsd)
    pred_mixed=np.asarray([len(set(int(inferred[v]) for v in row))>1 for row in F],bool)

    # Freeze teacher-free face evidence before teacher data enters.
    raw_rep_count=np.sum(rep[fei],axis=1)
    max_stretch=np.max(stretch[fei],axis=1)
    max_l1=np.max(l1[fei],axis=1)
    max_jsd=np.max(jsd[fei],axis=1)

    # Teacher begins here.
    truth=_teacher_truth(a.teacher_bank,a.teacher_source,sk,cand)
    truth_mixed=np.asarray([len(set(int(truth[v]) for v in row))>1 for row in F],bool)

    Wt,tjids,_,_=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    tix={str(j):i for i,j in enumerate(tjids)}
    Wt=np.stack([Wt[:,tix[str(j)]] for j in jids],axis=1)
    Wt=np.maximum(Wt,0.0);Wt/=np.maximum(Wt.sum(axis=1,keepdims=True),1e-15)

    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    actual_e,worst_clip,worst_frame=_actual_face_edge_max(P,Wt,F,jids,sk,cams,rr,source_report)
    synthetic_e=_synthetic_face_edge_max(P,Wt,F,jids,sk,cams,env)

    groups={
      "TRUE_POSITIVE_CROSS_REGION":truth_mixed&pred_mixed,
      "FALSE_NEGATIVE_CROSS_REGION":truth_mixed&(~pred_mixed),
      "FALSE_POSITIVE_SPLIT":(~truth_mixed)&pred_mixed,
      "TRUE_SAME_REGION":(~truth_mixed)&(~pred_mixed),
      "UNSAFE_FALSE_NEGATIVE_CROSS_REGION":unsafe&truth_mixed&(~pred_mixed),
    }
    summaries={k:_group(v,actual_e,synthetic_e,raw_rep_count,max_stretch,max_l1,max_jsd) for k,v in groups.items()}

    catastrophic10=actual_e>10.0
    catastrophic4=actual_e>4.0
    attr={
      "actual_gt10_total":int(np.count_nonzero(catastrophic10)),
      "actual_gt10_false_negative_cross_region":int(np.count_nonzero(catastrophic10&truth_mixed&(~pred_mixed))),
      "actual_gt10_true_positive_cross_region":int(np.count_nonzero(catastrophic10&truth_mixed&pred_mixed)),
      "actual_gt10_truth_same_region":int(np.count_nonzero(catastrophic10&(~truth_mixed))),
      "actual_gt4_total":int(np.count_nonzero(catastrophic4)),
      "actual_gt4_false_negative_cross_region":int(np.count_nonzero(catastrophic4&truth_mixed&(~pred_mixed))),
      "actual_gt4_true_positive_cross_region":int(np.count_nonzero(catastrophic4&truth_mixed&pred_mixed)),
      "actual_gt4_truth_same_region":int(np.count_nonzero(catastrophic4&(~truth_mixed))),
    }

    worst_idx=np.argsort(-actual_e)[:50]
    worst=[{
      "face_index":int(fi),
      "actual_edge_max":float(actual_e[fi]),
      "synthetic_edge_max":float(synthetic_e[fi]),
      "truth_mixed":bool(truth_mixed[fi]),
      "pred_mixed":bool(pred_mixed[fi]),
      "raw_repulsive_edge_count":int(raw_rep_count[fi]),
      "max_stage35_stretch":float(max_stretch[fi]),
      "max_arachne_l1":float(max_l1[fi]),
      "max_arachne_jsd":float(max_jsd[fi]),
      "worst_clip":str(worst_clip[fi]),
      "worst_frame":int(worst_frame[fi]),
    } for fi in worst_idx]

    report={
      "schema":"RealSaS.KnightResidualTopologySeverityCourt.v1",
      "status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
      "preregistration":str(PREREG),
      "teacher_used_by_partition_solver":False,
      "solver":solver,
      "groups":summaries,
      "catastrophic_attribution":attr,
      "worst_faces":worst,
      "finding":{
        "all_actual_gt10_are_false_negative_cross_region":bool(attr["actual_gt10_total"]>0 and attr["actual_gt10_total"]==attr["actual_gt10_false_negative_cross_region"]),
        "all_actual_gt4_are_false_negative_cross_region":bool(attr["actual_gt4_total"]>0 and attr["actual_gt4_total"]==attr["actual_gt4_false_negative_cross_region"]),
      },
      "claim_boundary":"Partition, repulsive mask and all Arachne-derived edge scores are frozen before teacher topology/weights enter. Teacher data only attributes residual failure severity."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_RESIDUAL_TOPOLOGY_SEVERITY_PASS",json.dumps({
      "solver":solver,
      "groups":summaries,
      "catastrophic_attribution":attr,
      "finding":report["finding"],
      "worst_faces":worst[:10],
    },sort_keys=True))


if __name__=="__main__":
    main()
