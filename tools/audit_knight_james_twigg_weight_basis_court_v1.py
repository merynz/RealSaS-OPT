from __future__ import annotations

import argparse,json
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
from tools.audit_knight_james_twigg_region_inference_v1 import (
    EPSILON,
    PROBE_DEGREES,
    _cluster_signatures,
    _rotation_signatures,
    _split_holeless,
    _teacher_region_eval,
    _vertex_labels_from_faces,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask,
    exact,
    face_indices,
    load,
    motion_metrics,
    stress_arbitrary_weights,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


def run_basis(
    name,Z,valid,P,F,Wsplit,jids,sk,cams,env,policy,dense,
    stress_for_safe,bank,source,cand,rr,source_report
):
    unsafe=np.zeros(len(F),dtype=bool)
    unsafe[np.asarray(stress_for_safe["unsafe_face_indices"],dtype=np.int64)]=True
    safe=(~unsafe)&dense&valid
    face_label,core,cluster_meta=_cluster_signatures(Z,safe,EPSILON)
    vlabel,vconf,prop_meta=_vertex_labels_from_faces(P,F,face_label,core,dense,~unsafe)
    teacher_eval=_teacher_region_eval(bank,source,sk,cand,vlabel,F,unsafe)
    P2,W2,F2,split_meta=_split_holeless(P,Wsplit,F,vlabel)
    before_motion=motion_metrics(P,Wsplit,F,jids,sk,cams,rr,source_report)
    after_motion=motion_metrics(P2,W2,F2,jids,sk,cams,rr,source_report)
    before_stress=stress_arbitrary_weights(P,Wsplit,F,jids,sk,cams,env,policy)
    after_stress=stress_arbitrary_weights(P2,W2,F2,jids,sk,cams,env,policy)
    return {
      "basis":name,
      "clustering":cluster_meta,
      "propagation":prop_meta,
      "teacher_eval_only":teacher_eval,
      "holeless_split":split_meta,
      "mechanics":{
        "before_motion":{k:v for k,v in before_motion.items() if k!="frames"},
        "after_motion":{k:v for k,v in after_motion.items() if k!="frames"},
        "before_stress":{k:v for k,v in before_stress.items() if k!="unsafe_face_indices"},
        "after_stress":{k:v for k,v in after_stress.items() if k!="unsafe_face_indices"},
      },
      "score":{
        "unsafe_cross_region_recall":float(teacher_eval["unsafe_truth_cross_region_recall"]),
        "mixed_recall":float(teacher_eval["mixed_face_recall"]),
        "mixed_precision":float(teacher_eval["mixed_face_precision"]),
        "after_edge_gt10":int(after_motion["max_edge_gt_10"]),
        "after_edge_gt4":int(after_motion["max_edge_gt_4"]),
        "after_worst_edge":float(after_motion["worst_edge_max"]),
        "after_synthetic_unsafe":int(after_stress["unsafe_face_count"]),
      }
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
    surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)

    P=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    F=face_indices(cand)
    dense=dense_supported_face_mask(rr,cand,surface)
    with np.load(a.weights_npz,allow_pickle=False) as z:
        jids=tuple(map(str,z["joint_ids"].tolist()))
        Wa=np.asarray(z["arachne"],dtype=np.float64)
        Wc=np.asarray(z["cutcell"],dtype=np.float64)
    if Wa.shape!=Wc.shape or Wa.shape[0]!=len(P):
        raise RuntimeError("WEIGHT_BASIS_SHAPE_DRIFT")

    stress_a=stress_arbitrary_weights(P,Wa,F,jids,sk,cams,env,policy)
    stress_c=stress_arbitrary_weights(P,Wc,F,jids,sk,cams,env,policy)
    Za,va,probes=_rotation_signatures(P,F,Wa,jids,sk,cams,PROBE_DEGREES)
    Zc,vc,_=_rotation_signatures(P,F,Wc,jids,sk,cams,PROBE_DEGREES)
    # Each normalized signature has RMS-style scaling already. Concatenating and
    # dividing by sqrt(2) keeps epsilon on the same physical scale.
    Zd=np.concatenate((Za,Zc),axis=1)/np.sqrt(2.0)
    vd=va&vc

    src_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    results=[]
    # Always split the ORIGINAL Arachne field so region inference is the only
    # intervention; CutCell is merely an alternate teacher-free motion lens.
    results.append(run_basis("ARACHNE_SIGNATURE",Za,va,P,F,Wa,jids,sk,cams,env,policy,dense,stress_a,a.teacher_bank,a.teacher_source,cand,rr,src_report))
    results.append(run_basis("CUTCELL_SIGNATURE",Zc,vc,P,F,Wa,jids,sk,cams,env,policy,dense,stress_c,a.teacher_bank,a.teacher_source,cand,rr,src_report))
    # Dual safe set: a face must be mechanically safe under both lenses.
    combined={
      "unsafe_face_indices":sorted(set(stress_a["unsafe_face_indices"])|set(stress_c["unsafe_face_indices"]))
    }
    results.append(run_basis("DUAL_SIGNATURE",Zd,vd,P,F,Wa,jids,sk,cams,env,policy,dense,combined,a.teacher_bank,a.teacher_source,cand,rr,src_report))

    best=max(results,key=lambda r:(
        r["score"]["unsafe_cross_region_recall"],
        r["score"]["mixed_recall"],
        r["score"]["mixed_precision"],
        -r["score"]["after_edge_gt10"],
        -r["score"]["after_edge_gt4"],
    ))

    report={
      "schema":"RealSaS.KnightJamesTwiggWeightBasisCourt.v1",
      "status":"PAPER_INSPIRED_TEACHER_FREE_REGION_INFERENCE__NO_PRODUCT_MUTATION",
      "teacher_used_by_region_inference":False,
      "split_weight_authority":"ORIGINAL_ARACHNE_ONLY",
      "probe_degrees":PROBE_DEGREES,
      "epsilon":EPSILON,
      "probe_count":len(probes),
      "bases":results,
      "selected_basis_by_teacher_evaluation_for_RESEARCH_ONLY":best["basis"],
      "finding":{
        "any_basis_recovers_half_unsafe_cross_region_boundaries":bool(max(x["score"]["unsafe_cross_region_recall"] for x in results)>=0.5),
        "any_basis_reduces_arachne_edge_gt10_after_holeless_split":bool(any(x["score"]["after_edge_gt10"]<630 for x in results)),
        "best_basis":best["basis"],
      },
      "claim_boundary":"Teacher topology selects the best basis only for research interpretation after all teacher-free labels are produced. This selection is not a product inference rule. Every holeless split is evaluated on the original Arachne weight field so topology-label causality is isolated."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_JAMES_TWIGG_WEIGHT_BASIS_COURT_PASS",json.dumps({
      "scores":{x["basis"]:x["score"] for x in results},
      "finding":report["finding"]
    },sort_keys=True))

if __name__=="__main__":main()
