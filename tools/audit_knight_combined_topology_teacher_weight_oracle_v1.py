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
CLIPS=(("demo_idle_v1","idle"),("demo_run_v1","run"),("demo_slash_v1","slash"))

def sh(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def load(rr,k,codec):
    rel,h=EXACT[k];p=rr/rel
    if sh(p)!=h:raise RuntimeError("SHA_DRIFT:"+k)
    return codec(json.loads(p.read_text()))
def fi(c):
    d={str(v.candidate_vertex_id):i for i,v in enumerate(c.vertices)}
    return np.asarray([[d[str(x)] for x in f] for f in c.faces],dtype=np.int64)
def cat(a,b,sets):
    if a==b:return "SAME_FACE"
    n=len(sets[a]&sets[b])
    return "SHARE_EDGE" if n==2 else "SHARE_VERTEX_ONLY" if n==1 else "NONINCIDENT"
def met(rest,posed,faces):
    r=rest[faces];p=posed[faces]
    re=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],axis=1)
    pe=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
    edge=(pe/np.maximum(re,1e-15)).max(axis=1)
    r1=r[:,1]-r[:,0];r2=r[:,2]-r[:,0];p1=p[:,1]-p[:,0];p2=p[:,2]-p[:,0]
    l=np.linalg.norm(r1,axis=1);u=r1/np.maximum(l[:,None],1e-15);x2=np.sum(r2*u,axis=1);perp=r2-x2[:,None]*u;y2=np.linalg.norm(perp,axis=1)
    inv=np.zeros((len(faces),2,2));inv[:,0,0]=1/np.maximum(l,1e-15);inv[:,0,1]=-x2/np.maximum(l*y2,1e-15);inv[:,1,1]=1/np.maximum(y2,1e-15)
    s=np.linalg.svd(np.einsum("nij,njk->nik",np.stack((p1,p2),axis=2),inv),compute_uv=False)
    return edge,s[:,0]/np.maximum(s[:,1],1e-15),s[:,0]*s[:,1]
def sm(e,c,a):
    return {"count":int(len(e)),"edge_p50":float(np.quantile(e,.5)),"edge_p95":float(np.quantile(e,.95)),"edge_p99":float(np.quantile(e,.99)),"edge_max":float(np.max(e)),
    "edge_gt_2":int(np.count_nonzero(e>2)),"edge_gt_4":int(np.count_nonzero(e>4)),"edge_gt_10":int(np.count_nonzero(e>10)),
    "condition_p95":float(np.quantile(c,.95)),"condition_p99":float(np.quantile(c,.99)),"condition_max":float(np.max(c)),"condition_gt_16":int(np.count_nonzero(c>16)),
    "area_p95":float(np.quantile(a,.95)),"area_p99":float(np.quantile(a,.99)),"area_max":float(np.max(a)),"area_gt_20":int(np.count_nonzero(a>20))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--bank",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=load(rr,"candidate",canonical_mesh_candidate_from_dict);sk=load(rr,"skeleton",qualified_skeleton_from_dict);skin=load(rr,"skin",qualified_skin_from_dict);cams=tuple(sorted(load(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    rest=np.asarray([v.P for v in cand.vertices],float);faces=fi(cand);jids,Wpred=_candidate_skin_weights(cand,skin,sk);ji={j:i for i,j in enumerate(jids)}
    with np.load(a.bank,allow_pickle=False) as z:
        sids=tuple(map(str,z["surface_ids"].tolist()));BW=np.asarray(z["weights"],float);tri=np.asarray(z["source_triangle_index"],int);bpos=np.asarray(z["target_positions_world"],float);bpar=np.asarray(z["target_parent_indices"],int)
    with np.load(a.source,allow_pickle=False) as z:sf=np.asarray(z["faces"],int)
    # independently align target columns and require graph exactness
    joints=tuple(sk.joints);ids=[str(x.canonical_joint_id) for x in joints];idx={x:i for i,x in enumerate(ids)};spos=np.asarray([x.position for x in joints],float);spar=np.asarray([-1 if x.parent_canonical_id is None else idx[str(x.parent_canonical_id)] for x in joints],int)
    C=np.linalg.norm(bpos[:,None]-spos[None],axis=2);ri,ci=linear_sum_assignment(C);m=np.empty(len(bpos),int);m[ri]=ci
    matches=sum(int((-1 if bpar[b]<0 else m[bpar[b]])==spar[m[b]]) for b in range(len(bpar)))
    if matches!=len(bpar):raise RuntimeError("TEACHER_TARGET_GRAPH_NOT_EXACT")
    WT=np.zeros((len(sids),len(jids)),float)
    for bc in range(len(m)):WT[:,ji[ids[m[bc]]]]=BW[:,bc]
    br={sid:i for i,sid in enumerate(sids)};vid={str(v.candidate_vertex_id):i for i,v in enumerate(cand.vertices)};crow=[]
    for v in cand.vertices:
        co=tuple(v.support_binding.coefficients)
        if len(co)!=1 or abs(float(co[0][1])-1)>1e-12:raise RuntimeError("NONIDENTITY_SUPPORT")
        crow.append(br[str(co[0][0])])
    crow=np.asarray(crow,int);Wteach=WT[crow]
    sets=[set(map(int,x)) for x in sf.tolist()]
    keep=[];strict=[];sharev=[]
    for f in faces:
        ts=[int(tri[crow[int(v)]]) for v in f];cs=[cat(ts[0],ts[1],sets),cat(ts[1],ts[2],sets),cat(ts[2],ts[0],sets)]
        keep.append("NONINCIDENT" not in cs);strict.append(all(x in ("SAME_FACE","SHARE_EDGE") for x in cs));sharev.append(("NONINCIDENT" not in cs) and ("SHARE_VERTEX_ONLY" in cs))
    keep=np.asarray(keep,bool);strict=np.asarray(strict,bool);sharev=np.asarray(sharev,bool);rf=faces[keep]
    report={"schema":"RealSaS.KnightCombinedTopologyTeacherWeightOracle.v1","status":"DIAGNOSTIC_ORACLE_ONLY__NO_REPAIR","intervention":{"removed_nonincident_faces":int(np.count_nonzero(~keep)),"retained_faces":int(np.count_nonzero(keep)),"strict_retained":int(np.count_nonzero(strict)),"share_vertex_only_retained":int(np.count_nonzero(sharev)),"skeleton_mutated":False,"motion_mutated":False,"retarget_mutated":False,"rest_positions_mutated":False},"clips":[]}
    for clip,short in CLIPS:
        payload=json.loads((rr/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip}.motion.json").read_text());tracks,_=_tracks_for_clip(payload,sk,cams,source_report);times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")));frames=[]
        for n,t in enumerate(times):
            mats,_,_= _joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams)
            pp=_skin(rest,Wpred,jids,mats);pt=_skin(rest,Wteach,jids,mats)
            ep,cp,ap=met(rest,pp,faces);et,ct,at=met(rest,pt,faces)
            frames.append({"frame":n,"time":float(t),"baseline_all":sm(ep,cp,ap),"combined_oracle_retained":sm(et[keep],ct[keep],at[keep]),"teacher_weight_strict":sm(et[strict],ct[strict],at[strict]),"teacher_weight_share_vertex_only":sm(et[sharev],ct[sharev],at[sharev]) if np.any(sharev) else None})
        report["clips"].append({"clip_id":clip,"frames":frames})
    worst=max(f["combined_oracle_retained"]["edge_max"] for c in report["clips"] for f in c["frames"])
    worstp99=max(f["combined_oracle_retained"]["edge_p99"] for c in report["clips"] for f in c["frames"])
    gt10=max(f["combined_oracle_retained"]["edge_gt_10"] for c in report["clips"] for f in c["frames"])
    cond16=max(f["combined_oracle_retained"]["condition_gt_16"] for c in report["clips"] for f in c["frames"])
    report["aggregate_verdict_inputs"]={"worst_retained_edge_max":worst,"worst_retained_edge_p99":worstp99,"max_retained_edge_gt_10":gt10,"max_retained_condition_gt_16":cond16}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print("KNIGHT_COMBINED_ORACLE_PASS",json.dumps(report["aggregate_verdict_inputs"],sort_keys=True))
if __name__=="__main__":main()
