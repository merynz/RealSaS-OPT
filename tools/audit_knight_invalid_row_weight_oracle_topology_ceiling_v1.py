from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.artifact_codec_v2 import (
 canonical_mesh_candidate_from_dict,deformation_envelope_from_dict,mesh_policy_from_dict,
 qualified_camera_set_from_dict,qualified_skeleton_from_dict,rigging_surface_from_dict)
from tools.audit_knight_james_twigg_region_inference_v1 import _split_holeless,_teacher_region_eval
from tools.audit_knight_mechanical_mutex_evidence_policy_court_v1 import _edge_jsd,_solve
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import _edge_stretch,_edge_table
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
 dense_supported_face_mask,exact,face_indices,load,motion_metrics,stress_arbitrary_weights)
from tools.audit_knight_weight_outlier_signal_court_v1 import teacher_matrix
from tools.demo.render_knight_motion_preview_v1 import _ctx

PREREG=Path("canonical/KNIGHT_INVALID_ROW_WEIGHT_ORACLE_TOPOLOGY_CEILING_PREREG_V1_20260929.json")

def checks(th,ev,split,am,ast):
 return {
  "no_face_deletion":int(split["face_deletion_count"])<=int(th["face_deletion_count_max"]),
  "rest_area_preserved":float(split["rest_area_error_max"])<=float(th["rest_area_error_max"]),
  "actual_gt10":int(am["max_edge_gt_10"])<=int(th["actual_motion_max_edge_gt_10_max"]),
  "actual_gt4":int(am["max_edge_gt_4"])<=int(th["actual_motion_max_edge_gt_4_max"]),
  "actual_worst_edge":float(am["worst_edge_max"])<=float(th["actual_motion_worst_edge_max"]),
  "synthetic_unsafe":int(ast["unsafe_face_count"])<=int(th["synthetic_unsafe_face_count_max"]),
  "mixed_precision":float(ev["mixed_face_precision"])>=float(th["mixed_face_precision_min"]),
  "unsafe_cross_region_recall":float(ev["unsafe_truth_cross_region_recall"])>=float(th["unsafe_cross_region_recall_min"]),
 }

def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True)
 p.add_argument("--weights-npz",type=Path,required=True);p.add_argument("--teacher-bank",type=Path,required=True);p.add_argument("--teacher-source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 prereg=json.loads(PREREG.read_text())
 if prereg.get("status")!="FROZEN_BEFORE_INVALID_ROW_WEIGHT_ORACLE_RESULT":raise RuntimeError("INVALID_ROW_ORACLE_PREREG_DRIFT")
 th=prereg["closure_thresholds"]
 rr=_ctx(a.authority_root,a.run_id)["run_root"];cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict);sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
 cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
 env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
 policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)
 surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
 P=np.asarray([v.P for v in cand.vertices],np.float64);F=face_indices(cand);dense=dense_supported_face_mask(rr,cand,surface)
 with np.load(a.weights_npz,allow_pickle=False) as z:
  jids=tuple(map(str,z["joint_ids"].tolist()));Wa=np.asarray(z["arachne"],np.float64)
 Wa=np.maximum(Wa,0.0);Wa/=np.maximum(Wa.sum(axis=1,keepdims=True),1e-15)
 Wt,valid=teacher_matrix(a.teacher_bank,sk,cand,jids)
 Wt=np.maximum(Wt,0.0);Wt/=np.maximum(Wt.sum(axis=1,keepdims=True),1e-15)
 Wi=Wa.copy();Wi[~valid]=Wt[~valid]

 # Inference stress/masks are recomputed after the oracle intervention.
 base=stress_arbitrary_weights(P,Wi,F,jids,sk,cams,env,policy);unsafe=np.zeros(len(F),bool);unsafe[np.asarray(base["unsafe_face_indices"],np.int64)]=True
 edges,edge_faces,fei=_edge_table(P,F);stretch=_edge_stretch(P,Wi,edges,jids,sk,cams,env);l1=np.sum(np.abs(Wi[edges[:,0]]-Wi[edges[:,1]]),axis=1);jsd=_edge_jsd(Wi,edges)
 dense_edge=np.zeros(len(edges),bool);scope=np.zeros(len(edges),bool)
 for ei,e in enumerate(map(tuple,edges.tolist())):
  fs=edge_faces[e];dense_edge[ei]=any(bool(dense[fi]) for fi in fs);scope[ei]=any(bool(dense[fi] and unsafe[fi]) for fi in fs)
 masks={
  "STRETCH_ONLY_JSD":scope&(stretch>4.0),
  "STRETCH_OR_L1_JSD":scope&((stretch>4.0)|(l1>1.0)),
  "STRETCH_AND_L1_JSD":scope&(stretch>4.0)&(l1>1.0),
 }
 labels={};solvers={}
 for name,mask in masks.items():labels[name],solvers[name]=_solve(len(P),edges,dense_edge,mask,jsd)

 # Teacher topology is evaluation only, after labels are frozen.
 ev={name:_teacher_region_eval(a.teacher_bank,a.teacher_source,sk,cand,lab,F,unsafe) for name,lab in labels.items()}
 source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
 rows={}
 for name,lab in labels.items():
  # Full teacher weights only measure isolated topology mechanics after inference.
  P2,W2,F2,split=_split_holeless(P,Wt,F,lab)
  am=motion_metrics(P2,W2,F2,jids,sk,cams,rr,source_report);ast=stress_arbitrary_weights(P2,W2,F2,jids,sk,cams,env,policy);ck=checks(th,ev[name],split,am,ast)
  rows[name]={"solver":solvers[name],"teacher_evaluation_only":ev[name],"holeless_split":split,
   "after_motion":{k:v for k,v in am.items() if k!="frames"},"after_stress":{k:v for k,v in ast.items() if k!="unsafe_face_indices"},
   "checks":ck,"closure_pass":all(ck.values())}
 report={"schema":"RealSaS.KnightInvalidRowWeightOracleTopologyCeiling.v1","status":"ORACLE_CEILING__NO_PRODUCT_MUTATION",
  "preregistration":str(PREREG),"ORACLE_NOT_PRODUCT":True,
  "intervention":{"teacher_invalid_vertex_count":int(np.count_nonzero(~valid)),"teacher_valid_vertex_count":int(np.count_nonzero(valid)),"intervened_fraction":float(np.mean(~valid))},
  "inference_uses_teacher_region_labels":False,"variants":rows,
  "finding":{"any_variant_closes_topology":bool(any(r["closure_pass"] for r in rows.values()))},
  "claim_boundary":"Teacher validity/weights are used only as an explicit causal oracle intervention on rows known invalid. No teacher topology label participates in inference. This cannot be shipped."}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_INVALID_ROW_WEIGHT_ORACLE_TOPOLOGY_CEILING_PASS",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
