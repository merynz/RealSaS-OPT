from __future__ import annotations
import argparse,heapq,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.artifact_codec_v2 import (
 canonical_mesh_candidate_from_dict,deformation_envelope_from_dict,mesh_policy_from_dict,
 qualified_camera_set_from_dict,qualified_skeleton_from_dict,rigging_surface_from_dict)
from tools.audit_knight_james_twigg_region_inference_v1 import _split_holeless,_teacher_region_eval
from tools.audit_knight_mechanical_mutex_evidence_policy_court_v1 import _edge_jsd,_solve
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import _edge_stretch,_edge_table
from tools.audit_knight_teacher_free_weight_completion_court_v1 import dense_supported_face_mask,exact,face_indices,load,motion_metrics,stress_arbitrary_weights
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.audit_knight_weight_outlier_signal_court_v1 import teacher_matrix
from tools.demo.render_knight_motion_preview_v1 import _ctx
PREREG=Path("canonical/KNIGHT_RELIABILITY_MASK_ONLY_TOPOLOGY_CEILING_PREREG_V1_20260929.json")

def complete_invalid(P,edges,dense_edge,valid,labels):
 n=len(P);nbr=[[] for _ in range(n)]
 for ei in np.where(dense_edge)[0]:
  a,b=map(int,edges[int(ei)]);w=float(np.linalg.norm(P[a]-P[b]))
  nbr[a].append((b,w));nbr[b].append((a,w))
 invalid=~np.asarray(valid,bool);seen=np.zeros(n,bool);out=np.asarray(labels,int).copy()
 reports=[]
 for s in np.where(invalid)[0]:
  if seen[s]:continue
  stack=[int(s)];seen[s]=True;comp=[]
  while stack:
   u=stack.pop();comp.append(u)
   for v,_ in nbr[u]:
    if invalid[v] and not seen[v]:seen[v]=True;stack.append(v)
  cset=set(comp);seeds={}
  for u in comp:
   for v,_ in nbr[u]:
    if valid[v] and out[v]>=0:seeds[int(v)]=int(out[v])
  labs=sorted(set(seeds.values()))
  if not labs:
   reports.append({"size":len(comp),"boundary_seed_count":0,"boundary_label_count":0,"status":"ABSTAIN_ZERO_BOUNDARY"});continue
  if len(labs)==1:
   for u in comp:out[u]=labs[0]
   reports.append({"size":len(comp),"boundary_seed_count":len(seeds),"boundary_label_count":1,"status":"FILLED_SINGLE_BOUNDARY"});continue
  allowed=cset|set(seeds);best={v:float("inf") for v in allowed};bl={v:-1 for v in allowed};heap=[]
  for v,lab in sorted(seeds.items()):best[v]=0.;bl[v]=lab;heapq.heappush(heap,(0.,lab,v))
  while heap:
   d,lab,u=heapq.heappop(heap)
   if d!=best[u] or lab!=bl[u]:continue
   for v,w in nbr[u]:
    if v not in allowed:continue
    nd=d+w
    if nd<best[v]-1e-15 or (abs(nd-best[v])<=1e-15 and (bl[v]<0 or lab<bl[v])):
     best[v]=nd;bl[v]=lab;heapq.heappush(heap,(nd,lab,v))
  miss=0
  for u in comp:
   if bl[u]>=0:out[u]=bl[u]
   else:miss+=1
  reports.append({"size":len(comp),"boundary_seed_count":len(seeds),"boundary_label_count":len(labs),"status":"FILLED_MULTI_BOUNDARY" if miss==0 else "PARTIAL","unresolved":miss})
 return out,reports

def ck(th,ev,split,am,ast):
 return {"no_face_deletion":split["face_deletion_count"]<=th["face_deletion_count_max"],"rest_area_preserved":split["rest_area_error_max"]<=th["rest_area_error_max"],
 "actual_gt10":am["max_edge_gt_10"]<=th["actual_motion_max_edge_gt_10_max"],"actual_gt4":am["max_edge_gt_4"]<=th["actual_motion_max_edge_gt_4_max"],"actual_worst_edge":am["worst_edge_max"]<=th["actual_motion_worst_edge_max"],
 "synthetic_unsafe":ast["unsafe_face_count"]<=th["synthetic_unsafe_face_count_max"],"mixed_precision":ev["mixed_face_precision"]>=th["mixed_face_precision_min"],"unsafe_cross_region_recall":ev["unsafe_truth_cross_region_recall"]>=th["unsafe_cross_region_recall_min"]}

def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--weights-npz",type=Path,required=True);p.add_argument("--teacher-bank",type=Path,required=True);p.add_argument("--teacher-source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 pr=json.loads(PREREG.read_text())
 if pr.get("status")!="FROZEN_BEFORE_RELIABILITY_MASK_ONLY_RESULT":raise RuntimeError("RELIABILITY_MASK_ORACLE_PREREG_DRIFT")
 th=pr["closure_thresholds"];rr=_ctx(a.authority_root,a.run_id)["run_root"];cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict);sk=exact(rr,"skeleton",qualified_skeleton_from_dict);cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
 env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict);policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict);surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
 P=np.asarray([v.P for v in cand.vertices],float);F=face_indices(cand);dense=dense_supported_face_mask(rr,cand,surface)
 with np.load(a.weights_npz,allow_pickle=False) as z:jids=tuple(map(str,z["joint_ids"].tolist()));W=np.asarray(z["arachne"],float)
 W=np.maximum(W,0);W/=np.maximum(W.sum(1,keepdims=True),1e-15)
 # Oracle reliability mask only; teacher weights are deliberately not read here.
 _,valid=teacher_matrix(a.teacher_bank,sk,cand,jids)
 base=stress_arbitrary_weights(P,W,F,jids,sk,cams,env,policy);unsafe=np.zeros(len(F),bool);unsafe[np.asarray(base["unsafe_face_indices"],int)]=True
 edges,efs,fei=_edge_table(P,F);stretch=_edge_stretch(P,W,edges,jids,sk,cams,env);l1=np.sum(np.abs(W[edges[:,0]]-W[edges[:,1]]),1);jsd=_edge_jsd(W,edges)
 dense_edge=np.zeros(len(edges),bool);scope=np.zeros(len(edges),bool);trusted_edge=np.zeros(len(edges),bool)
 for ei,e in enumerate(map(tuple,edges.tolist())):
  fs=efs[e];dense_edge[ei]=any(bool(dense[fi]) for fi in fs);scope[ei]=any(bool(dense[fi] and unsafe[fi]) for fi in fs);trusted_edge[ei]=dense_edge[ei] and bool(valid[e[0]] and valid[e[1]])
 variants={"VALID_STRETCH_ONLY_JSD":scope&trusted_edge&(stretch>4),"VALID_STRETCH_AND_L1_JSD":scope&trusted_edge&(stretch>4)&(l1>1)}
 frozen={};solvers={};completion={}
 for name,rep in variants.items():
  lab,sol=_solve(len(P),edges,trusted_edge,rep,jsd)
  # Invalid singleton labels are not authority; erase then complete from trusted labels.
  lab=np.asarray(lab,int);lab[~valid]=-1
  lab2,reports=complete_invalid(P,edges,dense_edge,valid,lab)
  frozen[name]=lab2;solvers[name]=sol;completion[name]=reports
 # Teacher topology begins after final labels are frozen.
 ev={n:_teacher_region_eval(a.teacher_bank,a.teacher_source,sk,cand,l,F,unsafe) for n,l in frozen.items()}
 # Teacher weights only for mechanical isolation.
 Wt,_,_,_=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand); # align via teacher_matrix for exact jids
 Wt,_=teacher_matrix(a.teacher_bank,sk,cand,jids);Wt=np.maximum(Wt,0);Wt/=np.maximum(Wt.sum(1,keepdims=True),1e-15)
 src=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text());rows={}
 for n,l in frozen.items():
  P2,W2,F2,split=_split_holeless(P,Wt,F,l);am=motion_metrics(P2,W2,F2,jids,sk,cams,rr,src);ast=stress_arbitrary_weights(P2,W2,F2,jids,sk,cams,env,policy);checks=ck(th,ev[n],split,am,ast)
  rows[n]={"solver":solvers[n],"completion":{"component_count":len(completion[n]),"zero_boundary":sum(r["status"]=="ABSTAIN_ZERO_BOUNDARY" for r in completion[n]),"multi_boundary":sum(r["status"]=="FILLED_MULTI_BOUNDARY" for r in completion[n]),"reports":completion[n][:64]},"teacher_evaluation_only":ev[n],"holeless_split":split,"after_motion":{k:v for k,v in am.items() if k!="frames"},"after_stress":{k:v for k,v in ast.items() if k!="unsafe_face_indices"},"checks":checks,"closure_pass":all(checks.values())}
 report={"schema":"RealSaS.KnightReliabilityMaskOnlyTopologyCeiling.v1","status":"ORACLE_CEILING__NO_PRODUCT_MUTATION","preregistration":str(PREREG),"ORACLE_NOT_PRODUCT":True,"teacher_weights_used_by_inference":False,"teacher_region_labels_used_by_inference":False,"reliability_mask":{"valid":int(valid.sum()),"invalid":int((~valid).sum())},"variants":rows,"finding":{"any_variant_closes_topology":any(r["closure_pass"] for r in rows.values())},"claim_boundary":"Only teacher_valid_mask is used as an oracle reliability mask during inference. Teacher weights and topology are not used until after labels freeze."}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print("KNIGHT_RELIABILITY_MASK_ONLY_TOPOLOGY_CEILING_PASS",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
