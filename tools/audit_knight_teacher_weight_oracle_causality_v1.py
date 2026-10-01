from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, qualified_camera_set_from_dict,
    qualified_skeleton_from_dict, qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights,_ctx,_skin,_tracks_for_clip

EXACT={
"candidate":("artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json","0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3"),
"skeleton":("artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json","e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987"),
"skin":("artifacts/32_SKIN_QUALIFIED/qualified_skin.json","f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09"),
"cameras":("artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json","312a9b1afe4ea1fcdc481232ac951fb6a6815fee532fbb93646e97e609b5b80a"),
}
CASES=(("demo_run_v1",0.4166666666666667),("demo_slash_v1",0.2777777777777778))

def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(rr,k,codec):
 rel,h=EXACT[k];p=rr/rel
 if sh(p)!=h:raise RuntimeError("SHA_DRIFT:"+k)
 return codec(json.loads(p.read_text()))
def faces_idx(c):
 d={str(v.candidate_vertex_id):i for i,v in enumerate(c.vertices)}
 return np.asarray([[d[str(x)] for x in f] for f in c.faces],dtype=np.int64)
def src_cat(a,b,sets):
 if a==b:return "SAME_FACE"
 n=len(sets[a]&sets[b])
 return "SHARE_EDGE" if n==2 else "SHARE_VERTEX_ONLY" if n==1 else "NONINCIDENT"
def metrics(rest,posed,faces):
 r=rest[faces];p=posed[faces]
 re=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],axis=1)
 pe=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
 edge=(pe/np.maximum(re,1e-15)).max(axis=1)
 r1=r[:,1]-r[:,0];r2=r[:,2]-r[:,0];p1=p[:,1]-p[:,0];p2=p[:,2]-p[:,0]
 l=np.linalg.norm(r1,axis=1);u=r1/np.maximum(l[:,None],1e-15);x2=np.sum(r2*u,axis=1);perp=r2-x2[:,None]*u;y2=np.linalg.norm(perp,axis=1)
 inv=np.zeros((len(faces),2,2));inv[:,0,0]=1/np.maximum(l,1e-15);inv[:,0,1]=-x2/np.maximum(l*y2,1e-15);inv[:,1,1]=1/np.maximum(y2,1e-15)
 s=np.linalg.svd(np.einsum("nij,njk->nik",np.stack((p1,p2),axis=2),inv),compute_uv=False)
 return edge,s[:,0]/np.maximum(s[:,1],1e-15),s[:,0]*s[:,1]
def summ(e,c,a,mask):
 e=e[mask];c=c[mask];a=a[mask]
 return {"count":int(len(e)),"edge_p95":float(np.quantile(e,.95)),"edge_p99":float(np.quantile(e,.99)),"edge_max":float(np.max(e)),"edge_gt_4":int(np.count_nonzero(e>4)),"edge_gt_10":int(np.count_nonzero(e>10)),"condition_p95":float(np.quantile(c,.95)),"condition_gt_16":int(np.count_nonzero(c>16)),"area_p95":float(np.quantile(a,.95)),"area_gt_20":int(np.count_nonzero(a>20))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--bank",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 rr=_ctx(a.authority_root,a.run_id)["run_root"]
 cand=load(rr,"candidate",canonical_mesh_candidate_from_dict);sk=load(rr,"skeleton",qualified_skeleton_from_dict);skin=load(rr,"skin",qualified_skin_from_dict);cams=tuple(sorted(load(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
 source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
 rest=np.asarray([v.P for v in cand.vertices],dtype=np.float64);faces=faces_idx(cand);jids,Wpred=_candidate_skin_weights(cand,skin,sk);ji={j:i for i,j in enumerate(jids)}
 with np.load(a.bank,allow_pickle=False) as z:
  sids=tuple(map(str,z["surface_ids"].tolist()));BW=np.asarray(z["weights"],dtype=np.float64);tri=np.asarray(z["source_triangle_index"],dtype=np.int64);bpos=np.asarray(z["target_positions_world"],dtype=np.float64);bpar=np.asarray(z["target_parent_indices"],dtype=np.int64)
 with np.load(a.source,allow_pickle=False) as z:sf=np.asarray(z["faces"],dtype=np.int64)
 # Align bank target columns to Stage28 and require exact parent graph preservation.
 joints=tuple(sk.joints);ids=[str(j.canonical_joint_id) for j in joints];ididx={x:i for i,x in enumerate(ids)};spos=np.asarray([j.position for j in joints],dtype=np.float64);spar=np.asarray([-1 if j.parent_canonical_id is None else ididx[str(j.parent_canonical_id)] for j in joints])
 C=np.linalg.norm(bpos[:,None]-spos[None],axis=2);ri,ci=linear_sum_assignment(C);m=np.empty(len(bpos),dtype=np.int64);m[ri]=ci
 graph_ok=0
 for b in range(len(bpar)):
  expected=-1 if bpar[b]<0 else int(m[int(bpar[b])]);actual=int(spar[int(m[b])]);graph_ok+=int(expected==actual)
 if graph_ok!=len(bpar):raise RuntimeError(f"BANK_STAGE28_GRAPH_ALIGNMENT_NOT_EXACT:{graph_ok}/{len(bpar)}")
 # bank col -> sorted joint_ids matrix used by skinning
 Wteacher_all=np.zeros((len(sids),len(jids)),dtype=np.float64)
 for bc in range(len(m)):
  stage_jid=ids[int(m[bc])];Wteacher_all[:,ji[stage_jid]]=BW[:,bc]
 bankrow={sid:i for i,sid in enumerate(sids)}
 crows=[]
 for v in cand.vertices:
  coeff=tuple(v.support_binding.coefficients)
  if len(coeff)!=1 or abs(float(coeff[0][1])-1)>1e-12:raise RuntimeError("NONIDENTITY_SUPPORT")
  crows.append(bankrow[str(coeff[0][0])])
 crows=np.asarray(crows,dtype=np.int64);Wteach=Wteacher_all[crows]
 # topology masks
 sets=[set(map(int,x)) for x in sf.tolist()];strict=[];noninc=[]
 for f in faces:
  ts=[int(tri[crows[int(v)]]) for v in f];cats=(src_cat(ts[0],ts[1],sets),src_cat(ts[1],ts[2],sets),src_cat(ts[2],ts[0],sets))
  strict.append(all(x in ("SAME_FACE","SHARE_EDGE") for x in cats));noninc.append("NONINCIDENT" in cats)
 strict=np.asarray(strict,dtype=bool);noninc=np.asarray(noninc,dtype=bool)
 # predicted-vs-teacher row L1 in exact current joint basis
 rowerr=np.sum(np.abs(Wpred-Wteach),axis=1)
 wf=Wpred[faces];pred_face_l1=np.maximum.reduce([np.sum(np.abs(wf[:,0]-wf[:,1]),axis=1),np.sum(np.abs(wf[:,1]-wf[:,2]),axis=1),np.sum(np.abs(wf[:,2]-wf[:,0]),axis=1)])
 tw=Wteach[faces];teach_face_l1=np.maximum.reduce([np.sum(np.abs(tw[:,0]-tw[:,1]),axis=1),np.sum(np.abs(tw[:,1]-tw[:,2]),axis=1),np.sum(np.abs(tw[:,2]-tw[:,0]),axis=1)])
 out={"schema":"RealSaS.KnightTeacherWeightOracleCausality.v1","status":"DIAGNOSTIC_ORACLE_ONLY__NO_REPAIR","alignment":{"graph_edge_match_count":graph_ok,"graph_edge_total":len(bpar),"max_position_error":float(np.max(C[ri,ci]))},"weight_error":{"row_l1_p95":float(np.quantile(rowerr,.95)),"row_l1_p99":float(np.quantile(rowerr,.99)),"row_l1_max":float(np.max(rowerr)),"rows_gt_0_1":int(np.count_nonzero(rowerr>0.1)),"strict_faces_pred_l1_gt_1":int(np.count_nonzero(strict&(pred_face_l1>1))),"strict_faces_teacher_l1_gt_1":int(np.count_nonzero(strict&(teach_face_l1>1)))},"cases":[]}
 for clip,t in CASES:
  payload=json.loads((rr/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip}.motion.json").read_text());tracks,_=_tracks_for_clip(payload,sk,cams,source_report);mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams)
  pp=_skin(rest,Wpred,jids,mats);pt=_skin(rest,Wteach,jids,mats);ep,cp,ap=metrics(rest,pp,faces);et,ct,at=metrics(rest,pt,faces)
  groups={"STRICT_SOURCE_LOCAL":strict,"NONINCIDENT":noninc,"STRICT_PRED_WEIGHT_L1_GT_1":strict&(pred_face_l1>1),"STRICT_TEACHER_WEIGHT_L1_LE_0_1":strict&(teach_face_l1<=.1)}
  out["cases"].append({"clip_id":clip,"time_seconds":t,"groups":{g:{"predicted":summ(ep,cp,ap,mask),"teacher_oracle":summ(et,ct,at,mask)} for g,mask in groups.items()}})
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_TEACHER_WEIGHT_ORACLE_CAUSALITY_PASS",json.dumps({"weight_error":out["weight_error"],"cases":out["cases"]},sort_keys=True))
if __name__=="__main__":main()
