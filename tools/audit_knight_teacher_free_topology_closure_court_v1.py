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
    rigging_surface_from_dict,
)
from tools.audit_knight_james_twigg_region_inference_v1 import (
    EPSILON, PROBE_DEGREES,
    _cluster_signatures, _rotation_signatures, _split_holeless,
    _teacher_region_eval, _vertex_labels_from_faces,
)
from tools.audit_knight_james_twigg_unsafe_patch_completion_court_v1 import (
    complete_unsafe_patches,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask, exact, face_indices, load,
    motion_metrics, stress_arbitrary_weights,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import (
    teacher_weights,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx

PREREG=Path("canonical/KNIGHT_TEACHER_FREE_TOPOLOGY_CLOSURE_PREREG_V1_20260929.json")

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
    if prereg.get("status")!="FROZEN_BEFORE_TEACHER_FREE_TOPOLOGY_ISOLATION_RESULT":
        raise RuntimeError("TOPOLOGY_CLOSURE_PREREG_DRIFT")
    th=prereg["closure_thresholds"]

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

    # IMPORTANT: Arachne is used only to infer the topology labels.
    with np.load(a.weights_npz,allow_pickle=False) as z:
        jids=tuple(map(str,z["joint_ids"].tolist()))
        Wa=np.asarray(z["arachne"],dtype=np.float64)
    if Wa.shape!=(len(P),len(jids)):
        raise RuntimeError("TOPOLOGY_ISOLATION_ARACHNE_SHAPE_DRIFT")

    # Teacher-free inference path.
    infer_stress=stress_arbitrary_weights(P,Wa,F,jids,sk,cams,env,policy)
    unsafe=np.zeros(len(F),dtype=bool)
    unsafe[np.asarray(infer_stress["unsafe_face_indices"],dtype=np.int64)]=True
    safe=(~unsafe)&dense
    Z,valid,probes=_rotation_signatures(P,F,Wa,jids,sk,cams,PROBE_DEGREES)
    face_label,core,cluster_meta=_cluster_signatures(Z,safe&valid,EPSILON)
    initial_label,conf,prop_meta=_vertex_labels_from_faces(P,F,face_label,core,dense,safe)
    inferred_label,patch_reports,patch_meta=complete_unsafe_patches(
        P,F,dense,unsafe,initial_label
    )

    # Teacher topology appears only here, after labels are frozen.
    teacher_eval=_teacher_region_eval(
        a.teacher_bank,a.teacher_source,sk,cand,inferred_label,F,unsafe
    )

    # Teacher weights appear only here, after topology inference is frozen.
    Wt,tjids,_,_=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    tix={str(j):i for i,j in enumerate(tjids)}
    missing=[j for j in jids if str(j) not in tix]
    if missing:
        raise RuntimeError("TOPOLOGY_ISOLATION_TEACHER_JOINT_DRIFT:"+json.dumps(missing))
    Wt=np.stack([Wt[:,tix[str(j)]] for j in jids],axis=1)
    Wt=np.maximum(Wt,0.0)
    Wt/=np.maximum(Wt.sum(axis=1,keepdims=True),1e-15)

    P2,W2,F2,split_meta=_split_holeless(P,Wt,F,inferred_label)

    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    before_motion=motion_metrics(P,Wt,F,jids,sk,cams,rr,source_report)
    after_motion=motion_metrics(P2,W2,F2,jids,sk,cams,rr,source_report)
    before_stress=stress_arbitrary_weights(P,Wt,F,jids,sk,cams,env,policy)
    after_stress=stress_arbitrary_weights(P2,W2,F2,jids,sk,cams,env,policy)

    checks={
      "no_face_deletion":int(split_meta["face_deletion_count"])<=int(th["face_deletion_count_max"]),
      "rest_area_preserved":float(split_meta["rest_area_error_max"])<=float(th["rest_area_error_max"]),
      "actual_gt10":int(after_motion["max_edge_gt_10"])<=int(th["actual_motion_max_edge_gt_10_max"]),
      "actual_gt4":int(after_motion["max_edge_gt_4"])<=int(th["actual_motion_max_edge_gt_4_max"]),
      "actual_worst_edge":float(after_motion["worst_edge_max"])<=float(th["actual_motion_worst_edge_max"]),
      "synthetic_unsafe":int(after_stress["unsafe_face_count"])<=int(th["synthetic_unsafe_face_count_max"]),
      "mixed_precision":float(teacher_eval["mixed_face_precision"])>=float(th["mixed_face_precision_min"]),
      "unsafe_cross_region_recall":float(teacher_eval["unsafe_truth_cross_region_recall"])>=float(th["unsafe_cross_region_recall_min"]),
    }
    closure=all(checks.values())

    report={
      "schema":"RealSaS.KnightTeacherFreeTopologyClosureCourt.v1",
      "status":"PASS_TOPOLOGY_CLOSED_FOR_KNIGHT" if closure else "FAIL_TOPOLOGY_REMAINS_OPEN",
      "preregistration":str(PREREG),
      "teacher_used_by_region_inference":False,
      "teacher_topology_used_for_evaluation_only":True,
      "teacher_weights_used_for_post_inference_mechanical_isolation_only":True,
      "inference":{
        "probe_degrees":PROBE_DEGREES,
        "probe_count":len(probes),
        "clustering":cluster_meta,
        "initial_propagation":prop_meta,
        "patch_completion":{**patch_meta,
          "completed_patch_count":sum(r["status"]=="COMPLETED" for r in patch_reports),
          "partial_patch_count":sum(r["status"]=="PARTIAL" for r in patch_reports),
          "abstained_patch_count":sum(r["status"].startswith("ABSTAIN") for r in patch_reports),
        },
      },
      "teacher_evaluation_only":teacher_eval,
      "holeless_split":split_meta,
      "topology_isolated_mechanics":{
        "before_motion":{k:v for k,v in before_motion.items() if k!="frames"},
        "after_motion":{k:v for k,v in after_motion.items() if k!="frames"},
        "before_stress":{k:v for k,v in before_stress.items() if k!="unsafe_face_indices"},
        "after_stress":{k:v for k,v in after_stress.items() if k!="unsafe_face_indices"},
      },
      "frozen_thresholds":th,
      "checks":checks,
      "closure_pass":closure,
      "claim_boundary":"A PASS closes the Knight topology owner under teacher-free inference with teacher weights used only after inference to isolate topology mechanics. It does not close Arachne weight generation or unseen generalization."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TEACHER_FREE_TOPOLOGY_CLOSURE_"+("PASS" if closure else "FAIL"),json.dumps({
      "teacher_eval":teacher_eval,
      "split":split_meta,
      "before_motion":report["topology_isolated_mechanics"]["before_motion"],
      "after_motion":report["topology_isolated_mechanics"]["after_motion"],
      "before_stress":report["topology_isolated_mechanics"]["before_stress"],
      "after_stress":report["topology_isolated_mechanics"]["after_stress"],
      "checks":checks,
      "closure_pass":closure,
    },sort_keys=True))

if __name__=="__main__":main()
