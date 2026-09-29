from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from compiler.realsas_compiler_core.artifact_codec_v2 import (
 canonical_mesh_candidate_from_dict,deformation_envelope_from_dict,mesh_policy_from_dict,
 qualified_camera_set_from_dict,qualified_skeleton_from_dict,rigging_surface_from_dict)
from tools.audit_knight_bone_locality_signal_court_v1 import point_segment_distance
from tools.audit_knight_mechanical_mutex_evidence_policy_court_v1 import _edge_jsd,_solve
from tools.audit_knight_residual_topology_severity_court_v1 import _actual_face_edge_max
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import _edge_stretch,_edge_table
from tools.audit_knight_teacher_free_weight_completion_court_v1 import dense_supported_face_mask,exact,face_indices,load,stress_arbitrary_weights
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.audit_knight_topology_region_pair_decomposition_court_v1 import source_components
from tools.audit_knight_weight_outlier_signal_court_v1 import teacher_matrix,build_neighbors
from tools.demo.render_knight_motion_preview_v1 import _ctx
PREREG=Path("canonical/KNIGHT_DETERMINISTIC_WEIGHT_RELIABILITY_FEATURE_PREREG_V1_20260929.json")

def truth_region(bank,source,sk,cand):
 _,_,tri,sf=teacher_weights(bank,source,sk,cand);sc=source_components(int(sf.max())+1,sf);tc=np.asarray([sc[int(r[0])] for r in sf],int);return np.asarray([tc[int(t)] for t in tri],int)

def qs(x,mask):
 a=np.asarray(x,float)[np.asarray(mask,bool)]
 if not len(a):return {"count":0}
 return {"count":int(len(a)),"p05":float(np.quantile(a,.05)),"p50":float(np.quantile(a,.5)),"p95":float(np.quantile(a,.95)),"max":float(np.max(a))}

def auc(y,x):
 y=np.asarray(y,bool);x=np.asarray(x,float);m=np.isfinite(x)
 if len(np.unique(y[m]))<2:return None
 return float(roc_auc_score(y[m].astype(int),x[m]))

def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--weights-npz",type=Path,required=True);p.add_argument("--teacher-bank",type=Path,required=True);p.add_argument("--teacher-source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_RELIABILITY_FEATURE_RESULT":raise RuntimeError("RELIABILITY_FEATURE_PREREG_DRIFT")
 rr=_ctx(a.authority_root,a.run_id)["run_root"];cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict);sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
 cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
 env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict);policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict);surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
 P=np.asarray([v.P for v in cand.vertices],float);F=face_indices(cand);dense=dense_supported_face_mask(rr,cand,surface)
 with np.load(a.weights_npz,allow_pickle=False) as z:jids=tuple(map(str,z["joint_ids"].tolist()));W=np.asarray(z["arachne"],float)
 W=np.maximum(W,0);W/=np.maximum(W.sum(1,keepdims=True),1e-15)

 # Product-only reliability features.
 joints=tuple(sk.joints);obj={str(j.canonical_joint_id):j for j in joints};pos={str(j.canonical_joint_id):np.asarray(j.position,float) for j in joints}
 A=[];B=[]
 for jid in jids:
  j=obj[jid];A.append(pos[jid] if j.parent_canonical_id is None else pos[str(j.parent_canonical_id)]);B.append(pos[jid])
 D=point_segment_distance(P,np.asarray(A),np.asarray(B));order=np.argsort(D,axis=1);rank=np.empty_like(order);rows=np.arange(len(P))[:,None];rank[rows,order]=np.arange(len(jids))[None,:]
 dom=np.argmax(W,axis=1);nearest=order[:,0];dom_rank=(rank[np.arange(len(P)),dom]+1).astype(float);dist_ratio=D[np.arange(len(P)),dom]/np.maximum(D[np.arange(len(P)),nearest],1e-8)

 nbr=build_neighbors(F,len(P));deviation=np.full(len(P),np.nan);dispersion=np.full(len(P),np.nan)
 for i,ns in enumerate(nbr):
  if not ns:continue
  X=W[np.asarray(ns,int)];m=np.median(X,axis=0);m=np.maximum(m,0);s=float(m.sum())
  if s<=1e-12:continue
  m/=s;deviation[i]=float(np.sum(np.abs(W[i]-m)));dispersion[i]=float(np.median(np.sum(np.abs(X-m[None]),axis=1)))

 face_feat={
  "max_dom_bone_rank":np.max(dom_rank[F],axis=1),
  "mean_dom_bone_rank":np.mean(dom_rank[F],axis=1),
  "max_dom_distance_ratio":np.max(dist_ratio[F],axis=1),
  "max_neighbor_median_deviation":np.nanmax(deviation[F],axis=1),
  "min_neighbor_consensus_dispersion":np.nanmin(dispersion[F],axis=1),
  "max_neighbor_consensus_dispersion":np.nanmax(dispersion[F],axis=1),
 }

 # Freeze current partition before teacher.
 base=stress_arbitrary_weights(P,W,F,jids,sk,cams,env,policy);unsafe=np.zeros(len(F),bool);unsafe[np.asarray(base["unsafe_face_indices"],int)]=True
 edges,efs,fei=_edge_table(P,F);stretch=_edge_stretch(P,W,edges,jids,sk,cams,env);l1=np.sum(np.abs(W[edges[:,0]]-W[edges[:,1]]),1);jsd=_edge_jsd(W,edges)
 de=np.zeros(len(edges),bool);scope=np.zeros(len(edges),bool)
 for ei,e in enumerate(map(tuple,edges.tolist())):
  fs=efs[e];de[ei]=any(bool(dense[fi]) for fi in fs);scope[ei]=any(bool(dense[fi] and unsafe[fi]) for fi in fs)
 rep=scope&(stretch>4)&(l1>1);lab,solver=_solve(len(P),edges,de,rep,jsd);pred=np.asarray([len(set(int(lab[v]) for v in r))>1 for r in F],bool)

 # Teacher begins here.
 truth=truth_region(a.teacher_bank,a.teacher_source,sk,cand);tm=np.asarray([len(set(int(truth[v]) for v in r))>1 for r in F],bool)
 Wt,valid=teacher_matrix(a.teacher_bank,sk,cand,jids);face_invalid=np.any((~valid)[F],axis=1)
 Wfull,_,_,_=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand); # align via teacher_matrix is already exact joint order
 Wfull=Wt
 source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text());actual,_,_=_actual_face_edge_max(P,Wfull,F,jids,sk,cams,rr,source_report)
 groups={
  "FALSE_POSITIVE_SPLIT":(~tm)&pred,
  "TRUE_POSITIVE_CROSS_REGION":tm&pred,
  "UNSAFE_CATASTROPHIC_FALSE_NEGATIVE":unsafe&tm&(~pred)&(actual>10),
  "TRUE_SAME_REGION":(~tm)&(~pred),
 }
 auc_invalid={k:auc(face_invalid,v) for k,v in face_feat.items()}
 fp_vs_tp={}
 sel=groups["FALSE_POSITIVE_SPLIT"]|groups["TRUE_POSITIVE_CROSS_REGION"];y=groups["FALSE_POSITIVE_SPLIT"][sel]
 for k,v in face_feat.items():fp_vs_tp[k]=auc(y,v[sel])
 report={"schema":"RealSaS.KnightDeterministicWeightReliabilityFeatureCourt.v1","status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION","preregistration":str(PREREG),"teacher_used_by_features_or_partition":False,"solver":solver,
  "teacher_eval_only":{"face_invalid_fraction":float(np.mean(face_invalid)),"feature_auc_for_teacher_invalid_face":auc_invalid,"feature_auc_false_positive_vs_true_positive_split":fp_vs_tp,
   "groups":{g:{k:qs(v,m) for k,v in face_feat.items()}|{"face_count":int(np.count_nonzero(m)),"teacher_invalid_fraction":float(np.mean(face_invalid[m])) if np.count_nonzero(m) else None} for g,m in groups.items()}},
  "claim_boundary":"All reliability features and the topology partition are computed before teacher validity/topology/weights are loaded. No threshold is selected."}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print("KNIGHT_DETERMINISTIC_WEIGHT_RELIABILITY_FEATURE_PASS",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
