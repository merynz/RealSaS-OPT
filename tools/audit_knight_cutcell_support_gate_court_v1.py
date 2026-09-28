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
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    exact,
    face_indices,
    load,
    motion_metrics,
    stress_arbitrary_weights,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.demo.render_knight_motion_preview_v1 import _ctx, _skin, _tracks_for_clip

CLIPS=("demo_idle_v1","demo_run_v1","demo_slash_v1")
BETAS=(0.05,0.10,0.20,0.35,0.50,0.75,1.0)
TOPKS=(2,3,4,6,8)
RELS=(0.01,0.025,0.05,0.10,0.20)


def simplex(W):
    X=np.maximum(np.asarray(W,dtype=np.float64),0.0)
    s=X.sum(axis=1,keepdims=True)
    if np.any(s<=1e-15):
        raise RuntimeError("SUPPORT_GATE_SIMPLEX_COLLAPSE")
    return X/s


def product_gate(Wa,Wc,beta):
    # CutCell acts only as a geometric likelihood; Arachne remains semantic authority.
    eps=1e-8
    return simplex(np.maximum(Wa,eps)*np.power(np.maximum(Wc,eps),float(beta)))


def topk_gate(Wa,Wc,k):
    k=min(int(k),Wc.shape[1])
    idx=np.argpartition(Wc,-k,axis=1)[:,-k:]
    mask=np.zeros_like(Wa,dtype=bool)
    rows=np.arange(len(Wa))[:,None]
    mask[rows,idx]=True
    # Fail-safe: never delete Arachne's own dominant semantic bone.
    mask[np.arange(len(Wa)),np.argmax(Wa,axis=1)]=True
    return simplex(np.where(mask,Wa,0.0))


def relative_gate(Wa,Wc,tau):
    mx=np.max(Wc,axis=1,keepdims=True)
    mask=Wc>=float(tau)*np.maximum(mx,1e-15)
    mask[np.arange(len(Wa)),np.argmax(Wa,axis=1)]=True
    return simplex(np.where(mask,Wa,0.0))


def entropy(W):
    X=np.maximum(np.asarray(W,dtype=np.float64),1e-15)
    return -np.sum(X*np.log(X),axis=1)


def change_metrics(W,Wa):
    d=np.sum(np.abs(W-Wa),axis=1)
    return {
      "row_l1_change_mean":float(np.mean(d)),
      "row_l1_change_p95":float(np.quantile(d,.95)),
      "row_l1_change_p99":float(np.quantile(d,.99)),
      "row_l1_change_max":float(np.max(d)),
      "rows_changed_gt_0_1":int(np.count_nonzero(d>.1)),
      "rows_changed_gt_0_5":int(np.count_nonzero(d>.5)),
      "dominant_joint_change_fraction":float(np.mean(np.argmax(W,axis=1)!=np.argmax(Wa,axis=1))),
      "entropy_mean":float(np.mean(entropy(W))),
    }


def teacher_fidelity(P,variants,jids,sk,cams,rr,source_report,Wt):
    sq={k:0.0 for k in variants}; count=0
    te=0.0; ve={k:0.0 for k in variants}
    num={k:0.0 for k in variants}; da={k:0.0 for k in variants}; dt2=0.0
    for clip in CLIPS:
        payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text())
        tracks,_=_tracks_for_clip(payload,sk,cams,source_report)
        times=np.linspace(0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        for t in times:
            mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams)
            pt=_skin(P,Wt,jids,mats); dtrue=pt-P
            te+=float(np.sum(dtrue*dtrue)); dt2+=float(np.sum(dtrue*dtrue))
            for name,W in variants.items():
                pp=_skin(P,W,jids,mats); diff=pp-pt; dp=pp-P
                sq[name]+=float(np.sum(diff*diff)); ve[name]+=float(np.sum(dp*dp))
                num[name]+=float(np.sum(dp*dtrue)); da[name]+=float(np.sum(dp*dp))
            count+=len(P)*3
    ext=float(np.max(P.max(axis=0)-P.min(axis=0)))
    out={}
    for name in variants:
        rmse=float(np.sqrt(sq[name]/max(count,1)))
        out[name]={
          "trajectory_rmse_xyz":rmse,
          "trajectory_rmse_over_character_extent":rmse/max(ext,1e-15),
          "motion_energy_ratio_vs_teacher":float(ve[name]/max(te,1e-15)),
          "motion_vector_cosine_vs_teacher":float(num[name]/max(np.sqrt(da[name]*dt2),1e-15)),
        }
    return out


def score(m,change):
    # Teacher-free mechanics first, then minimal semantic intervention.
    return (
      int(m["max_edge_gt_10"]),
      int(m["max_edge_gt_4"]),
      float(m["max_edge_p99"]),
      float(change["row_l1_change_mean"]),
      float(m["worst_edge_max"]),
    )


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
        Wa=np.asarray(z["arachne"],dtype=np.float64)
        Wc=np.asarray(z["cutcell"],dtype=np.float64)
    if Wa.shape!=Wc.shape or Wa.shape[0]!=len(P):
        raise RuntimeError("CUTCELL_GATE_WEIGHT_SHAPE_DRIFT")

    variants={"ARACHNE":Wa,"CUT_CELL":Wc}
    for b in BETAS:
        variants[f"PRODUCT_BETA_{b:.2f}"]=product_gate(Wa,Wc,b)
    for k in TOPKS:
        variants[f"TOPK_{k}"]=topk_gate(Wa,Wc,k)
    for t in RELS:
        variants[f"REL_{t:.3f}"]=relative_gate(Wa,Wc,t)

    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    rows={}
    for name,W in variants.items():
        mot=motion_metrics(P,W,F,jids,sk,cams,rr,source_report)
        ch=change_metrics(W,Wa)
        rows[name]={
          "actual_motion":{k:v for k,v in mot.items() if k!="frames"},
          "change_from_arachne":ch,
          "teacher_free_score":list(score(mot,ch)),
        }

    # Stress only on nontrivial Pareto candidates plus baselines.
    candidates=sorted(
        (n for n in variants if n not in ("ARACHNE","CUT_CELL")),
        key=lambda n:tuple(rows[n]["teacher_free_score"])
    )
    shortlist=["ARACHNE","CUT_CELL"]+candidates[:6]
    for name in shortlist:
        st=stress_arbitrary_weights(P,variants[name],F,jids,sk,cams,env,policy)
        rows[name]["synthetic_stress"]={k:v for k,v in st.items() if k!="unsafe_face_indices"}

    # Teacher is evaluation-only and never selects or constructs a gate.
    Wt0,tjids,_,_=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    tix={str(j):i for i,j in enumerate(tjids)}
    missing=[j for j in jids if str(j) not in tix]
    if missing: raise RuntimeError("TEACHER_JOINT_ID_MISSING:"+json.dumps(missing))
    Wt=np.stack([Wt0[:,tix[str(j)]] for j in jids],axis=1)
    eval_names=["ARACHNE","CUT_CELL"]+candidates[:10]
    fidelity=teacher_fidelity(P,{n:variants[n] for n in eval_names},jids,sk,cams,rr,source_report,Wt)
    for name,m in fidelity.items():
        rows[name]["teacher_eval_only_motion_fidelity"]=m

    best=candidates[0]
    report={
      "schema":"RealSaS.KnightCutCellSupportGateCourt.v1",
      "status":"PAPER_INSPIRED_PRIOR_AUGMENTATION__NO_PRODUCT_MUTATION",
      "paper_basis":{
        "title":"A Geodesic Cut-Cell Prior for Neural Skinning",
        "principle":"CUT_CELL_IS_GEOMETRIC_PRIOR_TO_AUGMENT_DATA_DRIVEN_SKINNING_NOT_REPLACE_SEMANTICS",
      },
      "teacher_used_by_gate":False,
      "operator_families":{
        "PRODUCT":"normalize(Arachne * CutCell^beta)",
        "TOPK":"retain Arachne only on CutCell top-k support, union Arachne dominant joint",
        "REL":"retain Arachne where CutCell >= tau*row_max, union Arachne dominant joint",
      },
      "variants":rows,
      "selected_by_teacher_free_mechanical_score":best,
      "shortlist_stress_evaluated":shortlist,
      "finding":{
        "some_support_gate_eliminates_gt10":bool(any(rows[n]["actual_motion"]["max_edge_gt_10"]==0 for n in candidates)),
        "some_support_gate_reduces_gt4_vs_arachne":bool(any(rows[n]["actual_motion"]["max_edge_gt_4"]<rows["ARACHNE"]["actual_motion"]["max_edge_gt_4"] for n in candidates)),
        "some_support_gate_preserves_dominant_joint_under_5pct_change":bool(any(rows[n]["change_from_arachne"]["dominant_joint_change_fraction"]<.05 for n in candidates)),
        "best_teacher_free_variant":best,
      },
      "claim_boundary":"This is an inference-time paper-inspired augmentation test, not the paper's full neural integration/retraining protocol. Teacher weights are evaluation-only."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_CUTCELL_SUPPORT_GATE_COURT_PASS",json.dumps({
      "best":best,
      "best_row":rows[best],
      "arachne":rows["ARACHNE"],
      "cutcell":rows["CUT_CELL"],
      "finding":report["finding"],
    },sort_keys=True))


if __name__=="__main__":
    main()
