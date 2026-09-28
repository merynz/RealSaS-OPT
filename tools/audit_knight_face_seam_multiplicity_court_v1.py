from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
 canonical_mesh_candidate_from_dict,deformation_envelope_from_dict,mesh_policy_from_dict,
 qualified_camera_set_from_dict,qualified_skeleton_from_dict,rigging_surface_from_dict)
from tools.audit_knight_edge_seam_two_core_court_v1 import _two_core
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import _edge_stretch,_edge_table,_truth_region
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
 dense_supported_face_mask,exact,face_indices,load,stress_arbitrary_weights)
from tools.demo.render_knight_motion_preview_v1 import _ctx

PREREG=Path("canonical/KNIGHT_FACE_SEAM_MULTIPLICITY_PREREG_V1_20260929.json")

def metrics(name,pred_edge,fei,truth,unsafe,F):
 count=np.sum(pred_edge[fei],axis=1)
 pred=count>=2
 truth_mixed=np.asarray([len(set(int(truth[v]) for v in row))>1 for row in F],bool)
 tp=int(np.count_nonzero(pred&truth_mixed));fp=int(np.count_nonzero(pred&(~truth_mixed)))
 ut=unsafe&truth_mixed;hit=int(np.count_nonzero(pred&ut))
 table={}
 for c in range(4):
  table[str(c)]={
   "face_count":int(np.count_nonzero(count==c)),
   "truth_mixed":int(np.count_nonzero((count==c)&truth_mixed)),
   "truth_same":int(np.count_nonzero((count==c)&(~truth_mixed))),
   "unsafe_truth_mixed":int(np.count_nonzero((count==c)&ut)),
  }
 return {
  "name":name,
  "predicted_splittable_face_count":int(np.count_nonzero(pred)),
  "true_positive":tp,"false_positive":fp,
  "mixed_face_precision":float(tp/max(1,tp+fp)),
  "unsafe_truth_cross_region_face_count":int(np.count_nonzero(ut)),
  "unsafe_truth_cross_region_recovered":hit,
  "unsafe_truth_cross_region_recall":float(hit/max(1,np.count_nonzero(ut))),
  "seam_edge_multiplicity":table,
 }

def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True)
 p.add_argument("--weights-npz",type=Path,required=True);p.add_argument("--teacher-bank",type=Path,required=True)
 p.add_argument("--teacher-source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_FACE_SEAM_MULTIPLICITY_RESULT":raise RuntimeError("FACE_SEAM_MULTIPLICITY_PREREG_DRIFT")
 rr=_ctx(a.authority_root,a.run_id)["run_root"];cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
 sk=exact(rr,"skeleton",qualified_skeleton_from_dict);cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
 env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
 policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)
 surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
 P=np.asarray([v.P for v in cand.vertices],np.float64);F=face_indices(cand);dense=dense_supported_face_mask(rr,cand,surface)
 with np.load(a.weights_npz,allow_pickle=False) as z:
  jids=tuple(map(str,z["joint_ids"].tolist()));W=np.asarray(z["arachne"],np.float64)
 st=stress_arbitrary_weights(P,W,F,jids,sk,cams,env,policy);unsafe=np.zeros(len(F),bool);unsafe[np.asarray(st["unsafe_face_indices"],np.int64)]=True
 edges,efs,fei=_edge_table(P,F);stretch=_edge_stretch(P,W,edges,jids,sk,cams,env);l1=np.sum(np.abs(W[edges[:,0]]-W[edges[:,1]]),axis=1)
 ins=np.zeros(len(edges),bool);bv=np.zeros(len(P),bool)
 for ei,e in enumerate(map(tuple,edges.tolist())):
  fs=efs[e];ins[ei]=any(bool(unsafe[fi] and dense[fi]) for fi in fs)
  if len(fs)==1:bv[int(e[0])]=True;bv[int(e[1])]=True
 raw_and=ins&(stretch>4.0)&(l1>1.0)
 rules={
  "RAW_STRETCH_GT4_AND_WEIGHT_L1_GT1":raw_and,
  "TWO_CORE_STRETCH_GT4_AND_WEIGHT_L1_GT1":_two_core(edges,raw_and,bv),
  "RAW_WEIGHT_L1_GT1":ins&(l1>1.0),
  "RAW_STRETCH_GT4":ins&(stretch>4.0),
 }
 # Teacher begins here.
 truth=_truth_region(a.teacher_bank,a.teacher_source,sk,cand)
 rows={name:metrics(name,pred,fei,truth,unsafe,F) for name,pred in rules.items()}
 passes={name:bool(r["mixed_face_precision"]>=.90 and r["unsafe_truth_cross_region_recall"]>=.70) for name,r in rows.items()}
 report={"schema":"RealSaS.KnightFaceSeamMultiplicityCourt.v1","status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
  "preregistration":str(PREREG),"teacher_used_by_operator":False,"results_teacher_eval_only":rows,"passes_label_gates":passes,
  "claim_boundary":"Every edge score, seam rule, 2-core mask and per-face seam multiplicity is frozen before teacher topology is loaded."}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_FACE_SEAM_MULTIPLICITY_PASS",json.dumps({"passes":passes,"results":rows},sort_keys=True))
if __name__=="__main__":main()
