from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.audit_knight_teacher_free_weight_completion_court_v1 import exact,face_indices
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.demo.render_knight_motion_preview_v1 import _ctx,_skin,_tracks_for_clip

CLIPS=("demo_idle_v1","demo_run_v1","demo_slash_v1")

def entropy(W):
    X=np.maximum(np.asarray(W,float),1e-15)
    return -np.sum(X*np.log(X),axis=1)

def summarize_displacement(rest,posed):
    d=np.linalg.norm(posed-rest,axis=1)
    return {
      "mean":float(np.mean(d)),"p50":float(np.quantile(d,.5)),
      "p95":float(np.quantile(d,.95)),"max":float(np.max(d))
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
    rest=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    with np.load(a.weights_npz,allow_pickle=False) as z:
        jids=tuple(map(str,z["joint_ids"].tolist()))
        Wa=np.asarray(z["arachne"],dtype=np.float64)
        Wc=np.asarray(z["cutcell"],dtype=np.float64)
        Ws=np.asarray(z["selected"],dtype=np.float64)
        selected=str(np.asarray(z["selected_variant"]).item())
    Wt,tjids,_,_=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    if tuple(jids)!=tuple(tjids):
        raise RuntimeError("TEACHER_JOINT_ORDER_DRIFT")

    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    variants={"ARACHNE":Wa,"CUT_CELL":Wc,"SELECTED":Ws}
    frames=[]
    sqerr={k:0.0 for k in variants};count=0
    teacher_motion_energy=0.0
    variant_motion_energy={k:0.0 for k in variants}
    cos_num={k:0.0 for k in variants};cos_da={k:0.0 for k in variants};cos_dt=0.0

    for clip in CLIPS:
        payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text())
        tracks,_=_tracks_for_clip(payload,sk,cams,source_report)
        times=np.linspace(0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        for fi,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams)
            pt=_skin(rest,Wt,jids,mats)
            dt=pt-rest
            teacher_motion_energy+=float(np.sum(dt*dt))
            cos_dt+=float(np.sum(dt*dt))
            row={"clip_id":clip,"frame":int(fi),"time_seconds":float(t),"teacher_displacement":summarize_displacement(rest,pt),"variants":{}}
            for name,W in variants.items():
                pp=_skin(rest,W,jids,mats)
                diff=pp-pt
                sq=float(np.sum(diff*diff));sqerr[name]+=sq
                dp=pp-rest
                variant_motion_energy[name]+=float(np.sum(dp*dp))
                cos_num[name]+=float(np.sum(dp*dt));cos_da[name]+=float(np.sum(dp*dp))
                per=np.linalg.norm(diff,axis=1)
                row["variants"][name]={
                    "rmse_xyz":float(np.sqrt(np.mean(diff*diff))),
                    "vertex_error_mean":float(np.mean(per)),
                    "vertex_error_p95":float(np.quantile(per,.95)),
                    "vertex_error_max":float(np.max(per)),
                    "displacement":summarize_displacement(rest,pp),
                }
            frames.append(row);count+=len(rest)*3

    char_extent=float(np.max(rest.max(axis=0)-rest.min(axis=0)))
    global_metrics={}
    for name,W in variants.items():
        rmse=float(np.sqrt(sqerr[name]/max(count,1)))
        motion_ratio=float(variant_motion_energy[name]/max(teacher_motion_energy,1e-15))
        cosine=float(cos_num[name]/max(np.sqrt(cos_da[name]*cos_dt),1e-15))
        dom=np.argmax(W,axis=1)
        binc=np.bincount(dom,minlength=len(jids))
        global_metrics[name]={
            "trajectory_rmse_xyz":rmse,
            "trajectory_rmse_over_character_extent":rmse/max(char_extent,1e-15),
            "motion_energy_ratio_vs_teacher":motion_ratio,
            "motion_vector_cosine_vs_teacher":cosine,
            "weight_entropy_mean":float(np.mean(entropy(W))),
            "dominant_joint_count_used":int(np.count_nonzero(binc)),
            "largest_dominant_joint_fraction":float(np.max(binc)/len(W)),
            "joint_total_mass_top10":[
                {"joint_id":jids[int(i)],"mass":float(W[:,i].sum())}
                for i in np.argsort(W.sum(axis=0))[::-1][:10]
            ],
        }

    report={
      "schema":"RealSaS.KnightCutCellMotionFidelityCourt.v1",
      "status":"TEACHER_EVALUATION_ONLY__NO_PRODUCT_MUTATION",
      "selected_variant":selected,
      "teacher_used_by_cutcell_solver":False,
      "metrics":global_metrics,
      "frames":frames,
      "finding":{
        "cutcell_is_not_rigid_collapse":bool(global_metrics["CUT_CELL"]["dominant_joint_count_used"]>=4 and global_metrics["CUT_CELL"]["largest_dominant_joint_fraction"]<0.9 and global_metrics["CUT_CELL"]["motion_energy_ratio_vs_teacher"]>0.1),
        "cutcell_motion_direction_correlates_with_teacher":bool(global_metrics["CUT_CELL"]["motion_vector_cosine_vs_teacher"]>0.7),
        "cutcell_motion_energy_within_2x_teacher":bool(0.5<=global_metrics["CUT_CELL"]["motion_energy_ratio_vs_teacher"]<=2.0),
        "cutcell_trajectory_rmse_better_than_arachne":bool(global_metrics["CUT_CELL"]["trajectory_rmse_xyz"]<global_metrics["ARACHNE"]["trajectory_rmse_xyz"]),
      },
      "claim_boundary":"Teacher weights are used only to evaluate deformation trajectories under identical skeleton and motion. This court detects the trivial failure mode where a low-stretch geometric prior merely rigidifies or otherwise semantically destroys the motion."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_CUTCELL_MOTION_FIDELITY_COURT_PASS",json.dumps({"metrics":global_metrics,"finding":report["finding"]},sort_keys=True))

if __name__=="__main__":main()
