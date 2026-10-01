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
PREREG=Path("canonical/KNIGHT_EDGE_SEAM_CUTCELL_CORROBORATION_PREREG_V1_20260929.json")

def metrics(mask,truth_edge,fei,truth,unsafe,F):
 tp=int(np.count_nonzero(mask&truth_edge));fp=int(np.count_nonzero(mask&(~truth_edge)))
 pf=np.any(mask[fei],axis=1)
 tm=np.asarray([len(set(int(truth[v]) for v in row))>1 for row in F],bool)
 ftp=int(np.count_nonzero(pf&tm));ffp=int(np.count_nonzero(pf&(~tm)));ut=unsafe&tm;hit=int(np.count_nonzero(pf&ut))
 return {"edge_count":int(mask.sum()),"edge_precision":float(tp/max(1,tp+fp)),
  "predicted_face_count":int(pf.sum()),"face_precision":float(ftp/max(1,ftp+ffp)),
  "unsafe_recall":float(hit/max(1,ut.sum())),"unsafe_recovered":hit}

def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True)
 p.add_argument("--weights-npz",type=Path,required=True);p.add_argument("--teacher-bank",type=Path,required=True)
 p.add_argument("--teacher-source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_CUTCELL_CORROBORATION_RESULT":raise RuntimeError("CUTCELL_CORR_PREREG_DRIFT")
 rr=_ctx(a.authority_root,a.run_id)["run_root"];cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
 sk=exact(rr,"skeleton",qualified_skeleton_from_dict);cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
 env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
 policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)
 surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
 P=np.asarray([v.P for v in cand.vertices],np.float64);F=face_indices(cand);dense=dense_supported_face_mask(rr,cand,surface)
 with np.load(a.weights_npz,allow_pickle=False) as z:
  jids=tuple(map(str,z["joint_ids"].tolist()));Wa=np.asarray(z["arachne"],np.float64);Wc=np.asarray(z["cutcell"],np.float64)
 if Wa.shape!=Wc.shape:raise RuntimeError("CUTCELL_CORR_SHAPE_DRIFT")
 st=stress_arbitrary_weights(P,Wa,F,jids,sk,cams,env,policy);unsafe=np.zeros(len(F),bool);unsafe[np.asarray(st["unsafe_face_indices"],np.int64)]=True
 edges,efs,fei=_edge_table(P,F);stretch=_edge_stretch(P,Wa,edges,jids,sk,cams,env);l1=np.sum(np.abs(Wa[edges[:,0]]-Wa[edges[:,1]]),axis=1)
 ins=np.zeros(len(edges),bool);bv=np.zeros(len(P),bool)
 for ei,e in enumerate(map(tuple,edges.tolist())):
  fs=efs[e];ins[ei]=any(bool(unsafe[fi] and dense[fi]) for fi in fs)
  if len(fs)==1:bv[int(e[0])]=True;bv[int(e[1])]=True
 seam=_two_core(edges,ins&(stretch>4)&(l1>1),bv)
 dc=np.argmax(Wc,axis=1);u=edges[:,0];v=edges[:,1]
 strata={"CUTCELL_DOM_DIFFERENT":seam&(dc[u]!=dc[v]),"CUTCELL_DOM_SAME":seam&(dc[u]==dc[v])}
 truth=_truth_region(a.teacher_bank,a.teacher_source,sk,cand);te=truth[u]!=truth[v]
 out={n:metrics(m,te,fei,truth,unsafe,F) for n,m in strata.items()}
 report={"schema":"RealSaS.KnightEdgeSeamCutCellCorroborationCourt.v1","status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
  "preregistration":str(PREREG),"teacher_used_by_stratification":False,"base_edge_count":int(seam.sum()),
  "strata_teacher_eval_only":out,"claim_boundary":"CutCell is used only as an independent teacher-free locality prior; teacher topology is loaded after strata freeze."}
 if sum(int(m.sum()) for m in strata.values())!=int(seam.sum()):raise RuntimeError("CUTCELL_CORR_PARTITION_FAIL")
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_EDGE_SEAM_CUTCELL_CORROBORATION_PASS",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
