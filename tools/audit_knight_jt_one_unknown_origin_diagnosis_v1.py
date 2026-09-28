from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.artifact_codec_v2 import (
 canonical_mesh_candidate_from_dict,deformation_envelope_from_dict,mesh_policy_from_dict,
 qualified_camera_set_from_dict,qualified_skeleton_from_dict,rigging_surface_from_dict)
from tools.audit_knight_edge_seam_two_core_court_v1 import _two_core
from tools.audit_knight_james_twigg_region_inference_v1 import (
 EPSILON,PROBE_DEGREES,_cluster_signatures,_rotation_signatures,_vertex_labels_from_faces)
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import _edge_stretch,_edge_table,_truth_region
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
 dense_supported_face_mask,exact,face_indices,load,stress_arbitrary_weights)
from tools.demo.render_knight_motion_preview_v1 import _ctx

PREREG=Path("canonical/KNIGHT_JT_ONE_UNKNOWN_ORIGIN_DIAG_PREREG_V1_20260929.json")

def eval_edge(mask,truth):
 tp=int(np.count_nonzero(mask&truth));fp=int(np.count_nonzero(mask&(~truth)))
 return {"edge_count":int(np.count_nonzero(mask)),"tp":tp,"fp":fp,"precision":float(tp/max(1,tp+fp))}

def eval_face(mask,fei,truth_v,unsafe,F):
 pred=np.any(mask[fei],axis=1)
 tm=np.asarray([len(set(int(truth_v[v]) for v in row))>1 for row in F],dtype=bool)
 tp=int(np.count_nonzero(pred&tm));fp=int(np.count_nonzero(pred&(~tm)))
 ut=unsafe&tm;hit=int(np.count_nonzero(pred&ut))
 return {"predicted_face_count":int(np.count_nonzero(pred)),"tp":tp,"fp":fp,
  "precision":float(tp/max(1,tp+fp)),"unsafe_recall":float(hit/max(1,np.count_nonzero(ut))),
  "unsafe_recovered":hit}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True)
 p.add_argument("--weights-npz",type=Path,required=True);p.add_argument("--teacher-bank",type=Path,required=True)
 p.add_argument("--teacher-source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
 a=p.parse_args()
 if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_ONE_UNKNOWN_ORIGIN_RESULT":
  raise RuntimeError("ONE_UNKNOWN_ORIGIN_PREREG_DRIFT")
 rr=_ctx(a.authority_root,a.run_id)["run_root"]
 cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict);sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
 cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
 env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
 policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)
 surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
 P=np.asarray([v.P for v in cand.vertices],np.float64);F=face_indices(cand);dense=dense_supported_face_mask(rr,cand,surface)
 with np.load(a.weights_npz,allow_pickle=False) as z:
  jids=tuple(map(str,z["joint_ids"].tolist()));W=np.asarray(z["arachne"],np.float64)
 stress=stress_arbitrary_weights(P,W,F,jids,sk,cams,env,policy)
 unsafe=np.zeros(len(F),bool);unsafe[np.asarray(stress["unsafe_face_indices"],np.int64)]=True;safe=(~unsafe)&dense
 Z,valid,_=_rotation_signatures(P,F,W,jids,sk,cams,PROBE_DEGREES)
 fl,core,_=_cluster_signatures(Z,safe&valid,EPSILON)
 vlabel,_,_= _vertex_labels_from_faces(P,F,fl,core,dense,safe)
 # Direct core-seed vertices, before graph propagation.
 direct=np.zeros(len(P),bool)
 usable=core&dense&safe&(fl>=0)
 for fi in np.where(usable)[0]:
  direct[F[int(fi)]]=True
 edges,efs,fei=_edge_table(P,F);stretch=_edge_stretch(P,W,edges,jids,sk,cams,env)
 l1=np.sum(np.abs(W[edges[:,0]]-W[edges[:,1]]),axis=1)
 ins=np.zeros(len(edges),bool);bv=np.zeros(len(P),bool)
 for ei,e in enumerate(map(tuple,edges.tolist())):
  fs=efs[e];ins[ei]=any(bool(unsafe[fi] and dense[fi]) for fi in fs)
  if len(fs)==1:bv[int(e[0])]=True;bv[int(e[1])]=True
 seam=_two_core(edges,ins&(stretch>4)&(l1>1),bv)
 u=edges[:,0];v=edges[:,1];ku=vlabel[u]>=0;kv=vlabel[v]>=0
 one=seam&(ku^kv)
 known_vertex=np.where(ku,u,v)
 known_direct=direct[known_vertex]
 strata={"KNOWN_CORE_SEED":one&known_direct,"KNOWN_PROPAGATED_ONLY":one&(~known_direct)}
 truth=_truth_region(a.teacher_bank,a.teacher_source,sk,cand);te=truth[u]!=truth[v]
 out={}
 for n,m in strata.items():
  out[n]={"edge":eval_edge(m,te),"face":eval_face(m,fei,truth,unsafe,F),
   "stretch_p50":float(np.quantile(stretch[m],.5)) if np.any(m) else None,
   "l1_p50":float(np.quantile(l1[m],.5)) if np.any(m) else None}
 report={"schema":"RealSaS.KnightJTOneUnknownOriginDiagnosis.v1","status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
  "preregistration":str(PREREG),"teacher_used_by_stratification":False,
  "one_unknown_edge_count":int(np.count_nonzero(one)),"strata_teacher_eval_only":out,
  "claim_boundary":"Direct-core versus propagated-only origin is frozen before teacher topology is loaded."}
 if sum(np.count_nonzero(m) for m in strata.values())!=np.count_nonzero(one):raise RuntimeError("ONE_UNKNOWN_PARTITION_FAIL")
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_JT_ONE_UNKNOWN_ORIGIN_DIAG_PASS",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
