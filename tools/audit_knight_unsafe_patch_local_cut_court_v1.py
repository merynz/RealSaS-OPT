from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.artifact_codec_v2 import (
 canonical_mesh_candidate_from_dict,deformation_envelope_from_dict,mesh_policy_from_dict,
 qualified_camera_set_from_dict,qualified_skeleton_from_dict,rigging_surface_from_dict)
from tools.audit_knight_edge_seam_two_core_court_v1 import _two_core
from tools.audit_knight_james_twigg_unsafe_patch_completion_court_v1 import unsafe_components
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import _edge_stretch,_edge_table,_truth_region
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
 dense_supported_face_mask,exact,face_indices,load,stress_arbitrary_weights)
from tools.demo.render_knight_motion_preview_v1 import _ctx
PREREG=Path("canonical/KNIGHT_UNSAFE_PATCH_LOCAL_CUT_PREREG_V1_20260929.json")

def local_components(F,comp,seam_keys):
 verts=sorted(set(int(v) for fi in comp for v in F[int(fi)]));vset=set(verts)
 nbr={v:set() for v in verts}
 for fi in comp:
  a,b,c=map(int,F[int(fi)])
  for x,y in ((a,b),(b,c),(c,a)):
   key=(x,y) if x<y else (y,x)
   if key in seam_keys:continue
   nbr[x].add(y);nbr[y].add(x)
 lab={};sizes=[];cid=0
 for s in verts:
  if s in lab:continue
  stack=[s];lab[s]=cid;n=0
  while stack:
   u=stack.pop();n+=1
   for v in nbr[u]:
    if v not in lab:lab[v]=cid;stack.append(v)
  sizes.append(n);cid+=1
 return lab,sizes

def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True)
 p.add_argument("--weights-npz",type=Path,required=True);p.add_argument("--teacher-bank",type=Path,required=True)
 p.add_argument("--teacher-source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_LOCAL_PATCH_CUT_RESULT":raise RuntimeError("LOCAL_PATCH_CUT_PREREG_DRIFT")
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
 seam=_two_core(edges,ins&(stretch>4)&(l1>1),bv)
 seam_keys={tuple(map(int,edges[i])) for i in np.where(seam)[0]}
 comps=unsafe_components(F,unsafe,dense)
 pred=np.zeros(len(F),bool);patches=[]
 for ci,comp in enumerate(comps):
  lab,sizes=local_components(F,comp,seam_keys)
  mixed=0
  for fi in comp:
   row=F[int(fi)]
   if len(set(lab[int(v)] for v in row))>1:pred[int(fi)]=True;mixed+=1
  patches.append({"component_index":ci,"unsafe_face_count":len(comp),"local_region_count":len(sizes),
   "largest_local_region_sizes":sorted(sizes,reverse=True)[:10],"predicted_mixed_face_count":mixed})
 # Teacher starts here.
 truth=_truth_region(a.teacher_bank,a.teacher_source,sk,cand)
 tm=np.asarray([len(set(int(truth[v]) for v in row))>1 for row in F],bool)
 tp=int(np.count_nonzero(pred&tm));fp=int(np.count_nonzero(pred&(~tm)));ut=unsafe&tm;hit=int(np.count_nonzero(pred&ut))
 m={"predicted_mixed_face_count":int(pred.sum()),"true_positive":tp,"false_positive":fp,
  "mixed_face_precision":float(tp/max(1,tp+fp)),"unsafe_truth_cross_region_face_count":int(ut.sum()),
  "unsafe_truth_cross_region_recovered":hit,"unsafe_truth_cross_region_recall":float(hit/max(1,ut.sum()))}
 report={"schema":"RealSaS.KnightUnsafePatchLocalCutCourt.v1","status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
  "preregistration":str(PREREG),"teacher_used_by_operator":False,"seam_edge_count":int(seam.sum()),
  "unsafe_patch_count":len(comps),"metrics_teacher_eval_only":m,"patches":patches,
  "passes_frozen_label_gates":bool(m["mixed_face_precision"]>=.90 and m["unsafe_truth_cross_region_recall"]>=.70),
  "claim_boundary":"Local patch cut components and predicted mixed faces are frozen before teacher topology is loaded."}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_UNSAFE_PATCH_LOCAL_CUT_PASS",json.dumps({"metrics":m,"passes":report["passes_frozen_label_gates"],"top_patches":patches[:10]},sort_keys=True))
if __name__=="__main__":main()
