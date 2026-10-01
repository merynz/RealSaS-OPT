from __future__ import annotations

import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, qualified_camera_set_from_dict,
    qualified_skeleton_from_dict, qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.preproduct_authority_v1 import signed_zero_surface_from_dict, normalization_domain_from_dict
from compiler.realsas_compiler_core.substrate.scene_first_signed import mesh_connected_component_labels_v1
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights,_ctx,_skin,_tracks_for_clip

EXACT={
"candidate":("artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json","0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3"),
"skeleton":("artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json","e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987"),
"skin":("artifacts/32_SKIN_QUALIFIED/qualified_skin.json","f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09"),
"cameras":("artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json","312a9b1afe4ea1fcdc481232ac951fb6a6815fee532fbb93646e97e609b5b80a"),
}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def exact(rr,k,codec):
 rel,h=EXACT[k];p=rr/rel
 if sha(p)!=h:raise RuntimeError("SHA_DRIFT:"+k)
 return codec(json.loads(p.read_text()))
def compact_inv(p,f,div,comp):
 lo=p.min(0);span=np.maximum(p.max(0)-lo,1e-12);keys=np.floor((p-lo)/span*div).astype(int);keys=np.clip(keys,0,div-1)
 if comp:keys=np.column_stack((mesh_connected_component_labels_v1(len(p),f),keys))
 _,inv=np.unique(keys,axis=0,return_inverse=True);return inv
def mapped_set(f,inv):
 m=inv[f];k=(m[:,0]!=m[:,1])&(m[:,1]!=m[:,2])&(m[:,2]!=m[:,0]);return {tuple(map(int,x)) for x in np.unique(np.sort(m[k],1),axis=0).tolist()}
def tri_components(nv,faces):
 labels=mesh_connected_component_labels_v1(nv,faces)
 return np.asarray([labels[int(f[0])] for f in faces],int)
def srccat(a,b,sets):
 if a==b:return "SAME_FACE"
 n=len(sets[a]&sets[b])
 return "SHARE_EDGE" if n==2 else "SHARE_VERTEX_ONLY" if n==1 else "NONINCIDENT"
def edge_ratio(rest,posed,faces):
 r=rest[faces];p=posed[faces]
 rl=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],1)
 pl=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],1)
 return (pl/np.maximum(rl,1e-15)).max(1)
def counts(mask,edge):
 m=np.asarray(mask,bool)
 return {"faces":int(m.sum()),"edge_gt_4":int(np.count_nonzero(m&(edge>4))),"edge_gt_10":int(np.count_nonzero(m&(edge>10))),"edge_gt_20":int(np.count_nonzero(m&(edge>20))),"max":float(edge[m].max()) if m.any() else None}
def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--bank",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 rr=_ctx(a.authority_root,a.run_id)["run_root"]
 cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict);sk=exact(rr,"skeleton",qualified_skeleton_from_dict);skin=exact(rr,"skin",qualified_skin_from_dict);cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
 surf=rigging_surface_from_dict(json.loads((rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json").read_text()))
 zero=signed_zero_surface_from_dict(json.loads((rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json").read_text()));norm=normalization_domain_from_dict(json.loads((rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json").read_text()))
 with np.load(Path(zero.npz_path),allow_pickle=False) as z:vn=np.asarray(z["vertices_normalized"],float);df=np.asarray(z["faces"],int)
 world=np.asarray(norm.center_xyz,float)[None]+vn*float(norm.half_extent);md=dict(surf.metadata or {});inv=compact_inv(world,df,int(md["compact_voxel_divisions"]),bool(md["component_aware_compaction"]));mset=mapped_set(df,inv)
 with np.load(a.bank,allow_pickle=False) as z:
  sids=tuple(map(str,z["surface_ids"].tolist()));tri=np.asarray(z["source_triangle_index"],int);valid=np.asarray(z["teacher_valid_mask"],np.uint8).astype(bool);dist=np.asarray(z["source_surface_distance"],float)
 with np.load(a.source,allow_pickle=False) as z:sf=np.asarray(z["faces"],int);sv=np.asarray(z["vertices_source"],float)
 srcsets=[set(map(int,x)) for x in sf.tolist()];tcomp=tri_components(len(sv),sf)
 br={sid:i for i,sid in enumerate(sids)};surfidx={str(n.surface_id):i for i,n in enumerate(surf.surface_nodes)};cvi={str(v.candidate_vertex_id):i for i,v in enumerate(cand.vertices)}
 faces=np.asarray([[cvi[str(x)] for x in f] for f in cand.faces],int)
 crow=[];csurf=[]
 for v in cand.vertices:
  co=tuple(v.support_binding.coefficients)
  if len(co)!=1 or abs(float(co[0][1])-1)>1e-12:raise RuntimeError("NONIDENTITY")
  sid=str(co[0][0]);crow.append(br[sid]);csurf.append(surfidx[sid])
 crow=np.asarray(crow,int);csurf=np.asarray(csurf,int)
 dense=[];topo=[];crosscomp=[];anyinvalid=[];maxdist=[]
 for f in faces:
  compact=tuple(sorted(map(int,csurf[f])));dense.append(compact in mset)
  ts=[int(tri[crow[int(v)]]) for v in f];pc=(srccat(ts[0],ts[1],srcsets),srccat(ts[1],ts[2],srcsets),srccat(ts[2],ts[0],srcsets))
  topo.append("NONINCIDENT" if "NONINCIDENT" in pc else ("SHARE_VERTEX_ONLY" if "SHARE_VERTEX_ONLY" in pc else "STRICT_LOCAL"))
  crosscomp.append(len({int(tcomp[t]) for t in ts})>1);anyinvalid.append(not bool(np.all(valid[crow[f]])));maxdist.append(float(np.max(dist[crow[f]])))
 dense=np.asarray(dense,bool);topo=np.asarray(topo,object);crosscomp=np.asarray(crosscomp,bool);anyinvalid=np.asarray(anyinvalid,bool);maxdist=np.asarray(maxdist,float)
 rest=np.asarray([v.P for v in cand.vertices],float);jids,W=_candidate_skin_weights(cand,skin,sk);wf=W[faces];pl1=np.maximum.reduce([np.abs(wf[:,0]-wf[:,1]).sum(1),np.abs(wf[:,1]-wf[:,2]).sum(1),np.abs(wf[:,2]-wf[:,0]).sum(1)])
 src=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
 report={"schema":"RealSaS.KnightCatastrophicFaceContingencyCourt.v1","status":"MEASURED__NO_REPAIR","static":{"face_count":len(faces),"dense":int(dense.sum()),"invented":int((~dense).sum()),"source_topology_counts":dict(Counter(topo.tolist())),"cross_source_component":int(crosscomp.sum()),"any_teacher_invalid":int(anyinvalid.sum())},"frames":[]}
 masks={
  "ALL":np.ones(len(faces),bool),
  "DENSE":dense,"INVENTED":~dense,
  "STRICT_LOCAL":topo=="STRICT_LOCAL","SHARE_VERTEX_ONLY":topo=="SHARE_VERTEX_ONLY","NONINCIDENT":topo=="NONINCIDENT",
  "NONINCIDENT_DENSE":(topo=="NONINCIDENT")&dense,
  "NONINCIDENT_DENSE_CROSS_SOURCE_COMPONENT":(topo=="NONINCIDENT")&dense&crosscomp,
  "NONINCIDENT_DENSE_CROSS_SOURCE_COMPONENT_ANY_INVALID":(topo=="NONINCIDENT")&dense&crosscomp&anyinvalid,
  "NONINCIDENT_DENSE_CROSS_SOURCE_COMPONENT_ALL_VALID":(topo=="NONINCIDENT")&dense&crosscomp&~anyinvalid,
  "NONINCIDENT_DENSE_SAME_SOURCE_COMPONENT":(topo=="NONINCIDENT")&dense&~crosscomp,
  "NONINCIDENT_DENSE_SAME_SOURCE_COMPONENT_ANY_INVALID":(topo=="NONINCIDENT")&dense&~crosscomp&anyinvalid,
  "NONINCIDENT_DENSE_SAME_SOURCE_COMPONENT_ALL_VALID":(topo=="NONINCIDENT")&dense&~crosscomp&~anyinvalid,
  "INVENTED_ANY_TEACHER_INVALID":(~dense)&anyinvalid,
  "INVENTED_ALL_TEACHER_VALID":(~dense)&~anyinvalid,
  "STRICT_LOCAL_ANY_TEACHER_INVALID":(topo=="STRICT_LOCAL")&anyinvalid,
  "STRICT_LOCAL_ALL_TEACHER_VALID":(topo=="STRICT_LOCAL")&~anyinvalid,
  "SHARE_VERTEX_ANY_TEACHER_INVALID":(topo=="SHARE_VERTEX_ONLY")&anyinvalid,
  "NONINCIDENT_ANY_TEACHER_INVALID":(topo=="NONINCIDENT")&anyinvalid,
  "ANY_TEACHER_INVALID":anyinvalid,"ALL_TEACHER_VALID":~anyinvalid,
  "PRED_WEIGHT_L1_GT_1":pl1>1,
  "DIST_LE_0_01":maxdist<=.01,"DIST_0_01_TO_0_05":(maxdist>.01)&(maxdist<=.05),"DIST_GT_0_05":maxdist>.05,
  "NONINCIDENT_DIST_LE_0_01":(topo=="NONINCIDENT")&(maxdist<=.01),
  "NONINCIDENT_DIST_GT_0_05":(topo=="NONINCIDENT")&(maxdist>.05),
 }
 for clip in ("demo_run_v1","demo_slash_v1"):
  payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text());tracks,_=_tracks_for_clip(payload,sk,cams,src);times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
  for fi,t in enumerate(times):
   mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams);posed=_skin(rest,W,jids,mats);e=edge_ratio(rest,posed,faces)
   report["frames"].append({"clip_id":clip,"frame":fi,"time":float(t),"groups":{k:counts(v,e) for k,v in masks.items()}})
 # Pick worst frame by total >10.
 worst=max(report["frames"],key=lambda x:x["groups"]["ALL"]["edge_gt_10"])
 report["worst_frame_summary"]=worst
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_CATASTROPHIC_FACE_CONTINGENCY_COURT_PASS",json.dumps({"static":report["static"],"worst":worst},sort_keys=True))
if __name__=="__main__":main()
