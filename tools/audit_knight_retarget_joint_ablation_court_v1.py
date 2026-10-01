from __future__ import annotations

import argparse, hashlib, json
from dataclasses import replace
from pathlib import Path
from collections import Counter

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, qualified_camera_set_from_dict,
    qualified_skeleton_from_dict, qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights,_ctx,_mapping,_skin,_tracks_for_clip

EXACT={
"candidate":("artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json","0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3"),
"skeleton":("artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json","e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987"),
"skin":("artifacts/32_SKIN_QUALIFIED/qualified_skin.json","f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09"),
"cameras":("artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json","312a9b1afe4ea1fcdc481232ac951fb6a6815fee532fbb93646e97e609b5b80a"),
}

def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(rr,k,codec):
 rel,h=EXACT[k];p=rr/rel
 if sh(p)!=h:raise RuntimeError("SHA_DRIFT:"+k)
 return codec(json.loads(p.read_text()))

def faces_idx(c):
 d={str(v.candidate_vertex_id):i for i,v in enumerate(c.vertices)}
 return np.asarray([[d[str(x)] for x in f] for f in c.faces],dtype=np.int64)

def metrics(rest,posed,faces):
 r=rest[faces];p=posed[faces]
 re=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],axis=1)
 pe=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
 edge=(pe/np.maximum(re,1e-15)).max(axis=1)
 r1=r[:,1]-r[:,0];r2=r[:,2]-r[:,0];p1=p[:,1]-p[:,0];p2=p[:,2]-p[:,0]
 l1=np.linalg.norm(r1,axis=1);u=r1/np.maximum(l1[:,None],1e-15);x2=np.sum(r2*u,axis=1);perp=r2-x2[:,None]*u;y2=np.linalg.norm(perp,axis=1)
 inv=np.zeros((len(faces),2,2));inv[:,0,0]=1/np.maximum(l1,1e-15);inv[:,0,1]=-x2/np.maximum(l1*y2,1e-15);inv[:,1,1]=1/np.maximum(y2,1e-15)
 F=np.einsum("nij,njk->nik",np.stack((p1,p2),axis=2),inv);s=np.linalg.svd(F,compute_uv=False);cond=s[:,0]/np.maximum(s[:,1],1e-15);area=s[:,0]*s[:,1]
 return edge,cond,area

def summary(edge,cond,area,mask=None):
 if mask is None:mask=np.ones(len(edge),dtype=bool)
 e=edge[mask];c=cond[mask];a=area[mask]
 return {
  "count":int(len(e)),"edge_p95":float(np.quantile(e,.95)),"edge_p99":float(np.quantile(e,.99)),"edge_max":float(np.max(e)),
  "edge_gt_4":int(np.count_nonzero(e>4)),"edge_gt_10":int(np.count_nonzero(e>10)),
  "condition_p95":float(np.quantile(c,.95)),"condition_gt_16":int(np.count_nonzero(c>16)),
  "area_p95":float(np.quantile(a,.95)),"area_gt_20":int(np.count_nonzero(a>20)),
 }

def identity_track(track):
 keys=[]
 for k in track.keyframes:
  keys.append(replace(k,local_rotation_quat_xyzw=(0.,0.,0.,1.),local_translation_xyz=(0.,0.,0.),local_scale_xyz=(1.,1.,1.)))
 return replace(track,keyframes=tuple(keys))

def ancestors(parent,node):
 out=set();cur=parent.get(node)
 while cur is not None and cur not in out:out.add(cur);cur=parent.get(cur)
 return out

def bad_children(payload,skeleton,source_report):
 mapping,details,srows,trows,tpar=_mapping(payload,skeleton,source_report)
 spar={str(r["source_joint_id"]):(None if r.get("parent_source_joint_id") is None else str(r.get("parent_source_joint_id"))) for r in srows}
 bad=[]
 for ch,pa in tpar.items():
  if pa is None:continue
  sp=mapping[str(pa)];sc=mapping[str(ch)]
  if sp==sc or sp in ancestors(spar,sc):continue
  bad.append({"target_child":str(ch),"target_parent":str(pa),"source_child":sc,"source_parent":sp,"relation":"REVERSED" if sc in ancestors(spar,sp) else "UNRELATED"})
 return mapping,bad

def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out",type=Path,required=True)
 a=p.parse_args();ctx=_ctx(a.authority_root,a.run_id);rr=ctx["run_root"]
 cand=load(rr,"candidate",canonical_mesh_candidate_from_dict);skel=load(rr,"skeleton",qualified_skeleton_from_dict);skin=load(rr,"skin",qualified_skin_from_dict)
 cams=tuple(sorted(load(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda c:int(c.view_index)))
 source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
 rest=np.asarray([v.P for v in cand.vertices],dtype=np.float64);faces=faces_idx(cand);jids,W=_candidate_skin_weights(cand,skin,skel)
 # Pure measured weight-discontinuity mask, independent of topology labels.
 wf=W[faces]
 l1=np.maximum.reduce([np.sum(np.abs(wf[:,0]-wf[:,1]),axis=1),np.sum(np.abs(wf[:,1]-wf[:,2]),axis=1),np.sum(np.abs(wf[:,2]-wf[:,0]),axis=1)])
 high=l1>1.0
 cases=(("demo_run_v1",0.4166666666666667),("demo_slash_v1",0.2777777777777778))
 out={"schema":"RealSaS.KnightRetargetJointAblationCourt.v1","status":"MEASURED__NO_REPAIR","cases":[]}
 for clip,t in cases:
  payload=json.loads((rr/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip}.motion.json").read_text())
  tracks,details=_tracks_for_clip(payload,skel,cams,source_report);mapping,bad=bad_children(payload,skel,source_report)
  def eval_tracks(T):
   mats,_,_= _joint_pose_v2(skeleton=skel,tracks=T,time_seconds=float(t),cameras=cams)
   posed=_skin(rest,W,jids,mats);e,c,a0=metrics(rest,posed,faces)
   return {"all":summary(e,c,a0),"high_weight_l1_gt_1":summary(e,c,a0,high)}
  base=eval_tracks(tracks)
  rows=[]
  for jid in sorted(tracks):
   T=dict(tracks);T[jid]=identity_track(tracks[jid]);m=eval_tracks(T)
   rows.append({
    "target_joint_id":jid,"source_joint_id":mapping.get(jid),"is_bad_hierarchy_child":any(x["target_child"]==jid for x in bad),
    "metrics":m,
    "edge_gt_10_reduction_all":base["all"]["edge_gt_10"]-m["all"]["edge_gt_10"],
    "edge_gt_10_reduction_high":base["high_weight_l1_gt_1"]["edge_gt_10"]-m["high_weight_l1_gt_1"]["edge_gt_10"],
    "edge_p99_reduction_high":base["high_weight_l1_gt_1"]["edge_p99"]-m["high_weight_l1_gt_1"]["edge_p99"],
   })
  rows.sort(key=lambda x:(x["edge_gt_10_reduction_high"],x["edge_p99_reduction_high"]),reverse=True)
  T=dict(tracks)
  for b in bad:T[b["target_child"]]=identity_track(tracks[b["target_child"]])
  bad_all=eval_tracks(T)
  out["cases"].append({"clip_id":clip,"time_seconds":t,"baseline":base,"bad_hierarchy_edges":bad,"all_bad_children_identity_ablation":bad_all,"joint_leave_one_out_ranked":rows})
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_RETARGET_JOINT_ABLATION_COURT_PASS",json.dumps({c["clip_id"]:{"baseline_high_edge_gt10":c["baseline"]["high_weight_l1_gt_1"]["edge_gt_10"],"bad_children_ablation_high_edge_gt10":c["all_bad_children_identity_ablation"]["high_weight_l1_gt_1"]["edge_gt_10"],"top":c["joint_leave_one_out_ranked"][:5]} for c in out["cases"]},sort_keys=True))
if __name__=="__main__":main()
