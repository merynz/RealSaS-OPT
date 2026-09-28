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
from compiler.realsas_compiler_core.mesh.conditioning_v1 import triangle_rest_metric
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights,_ctx,_skin,_tracks_for_clip

EXACT={
"candidate":("artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json","0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3"),
"skeleton":("artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json","e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987"),
"skin":("artifacts/32_SKIN_QUALIFIED/qualified_skin.json","f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09"),
"cameras":("artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json","312a9b1afe4ea1fcdc481232ac951fb6a6815fee532fbb93646e97e609b5b80a"),
}

def sh(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def load(rr,k,codec):
    rel,h=EXACT[k];p=rr/rel
    if sh(p)!=h:raise RuntimeError("SHA_DRIFT:"+k)
    return codec(json.loads(p.read_text()))
def cat(a,b,sets):
    if a==b:return "SAME_FACE"
    n=len(sets[a]&sets[b])
    return "SHARE_EDGE" if n==2 else "SHARE_VERTEX_ONLY" if n==1 else "NONINCIDENT"
def deform_metrics(rest,posed,faces):
    r=rest[faces];p=posed[faces]
    re=np.stack([np.linalg.norm(r[:,1]-r[:,0],1),np.linalg.norm(r[:,2]-r[:,1],1),np.linalg.norm(r[:,0]-r[:,2],1)])
    # vectorized edge ratios
    rl=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],axis=1)
    pl=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
    edge=pl/np.maximum(rl,1e-15)
    r1=r[:,1]-r[:,0];r2=r[:,2]-r[:,0];p1=p[:,1]-p[:,0];p2=p[:,2]-p[:,0]
    l=np.linalg.norm(r1,axis=1);u=r1/np.maximum(l[:,None],1e-15);x2=np.sum(r2*u,axis=1);perp=r2-x2[:,None]*u;y2=np.linalg.norm(perp,axis=1)
    inv=np.zeros((len(faces),2,2));inv[:,0,0]=1/np.maximum(l,1e-15);inv[:,0,1]=-x2/np.maximum(l*y2,1e-15);inv[:,1,1]=1/np.maximum(y2,1e-15)
    F=np.einsum("nij,njk->nik",np.stack((p1,p2),axis=2),inv)
    s=np.linalg.svd(F,compute_uv=False)
    return edge,s
def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--bank",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=load(rr,"candidate",canonical_mesh_candidate_from_dict);sk=load(rr,"skeleton",qualified_skeleton_from_dict);skin=load(rr,"skin",qualified_skin_from_dict);cams=tuple(sorted(load(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    rest=np.asarray([v.P for v in cand.vertices],float);vid={str(v.candidate_vertex_id):i for i,v in enumerate(cand.vertices)}
    faces=np.asarray([[vid[str(x)] for x in f] for f in cand.faces],int);jids,Wpred=_candidate_skin_weights(cand,skin,sk);ji={j:i for i,j in enumerate(jids)}
    with np.load(a.bank,allow_pickle=False) as z:
        sids=tuple(map(str,z["surface_ids"].tolist()));BW=np.asarray(z["weights"],float);tri=np.asarray(z["source_triangle_index"],int);bary=np.asarray(z["source_triangle_barycentric"],float);dist=np.asarray(z["source_surface_distance"],float);bpos=np.asarray(z["target_positions_world"],float);bpar=np.asarray(z["target_parent_indices"],int)
    with np.load(a.source,allow_pickle=False) as z:sf=np.asarray(z["faces"],int)
    joints=tuple(sk.joints);ids=[str(x.canonical_joint_id) for x in joints];idx={x:i for i,x in enumerate(ids)};spos=np.asarray([x.position for x in joints],float);spar=np.asarray([-1 if x.parent_canonical_id is None else idx[str(x.parent_canonical_id)] for x in joints],int)
    C=np.linalg.norm(bpos[:,None]-spos[None],axis=2);ri,ci=linear_sum_assignment(C);m=np.empty(len(bpos),int);m[ri]=ci
    if sum(int((-1 if bpar[b]<0 else m[bpar[b]])==spar[m[b]]) for b in range(len(bpar)))!=len(bpar):raise RuntimeError("GRAPH_ALIGN_FAIL")
    WT=np.zeros((len(sids),len(jids)),float)
    for bc in range(len(m)):WT[:,ji[ids[m[bc]]]]=BW[:,bc]
    br={sid:i for i,sid in enumerate(sids)};crow=[]
    for v in cand.vertices:
        co=tuple(v.support_binding.coefficients)
        if len(co)!=1 or abs(float(co[0][1])-1)>1e-12:raise RuntimeError("NONIDENTITY")
        crow.append(br[str(co[0][0])])
    crow=np.asarray(crow,int);Wteach=WT[crow]
    sets=[set(map(int,x)) for x in sf.tolist()]
    cats=[];keep=[]
    for f in faces:
        ts=[int(tri[crow[int(v)]]) for v in f];cs=(cat(ts[0],ts[1],sets),cat(ts[1],ts[2],sets),cat(ts[2],ts[0],sets));cats.append(cs);keep.append("NONINCIDENT" not in cs)
    keep=np.asarray(keep,bool)
    tw=Wteach[faces]
    teach_l1=np.maximum.reduce([np.sum(np.abs(tw[:,0]-tw[:,1]),axis=1),np.sum(np.abs(tw[:,1]-tw[:,2]),axis=1),np.sum(np.abs(tw[:,2]-tw[:,0]),axis=1)])
    restq=[triangle_rest_metric(tuple(tuple(map(float,rest[i])) for i in f)) for f in faces]

    report={"schema":"RealSaS.KnightCombinedOracleResidualCourt.v1","status":"MEASURED__NO_REPAIR","clips":[]}
    seen={}
    for clip in ("demo_run_v1","demo_slash_v1"):
        payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text());tracks,_=_tracks_for_clip(payload,sk,cams,source_report);times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")));frames=[]
        for frame,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams);posed=_skin(rest,Wteach,jids,mats);edge,s=deform_metrics(rest,posed,faces);cond=s[:,0]/np.maximum(s[:,1],1e-15);maxedge=edge.max(axis=1)
            mask=keep&((cond>16)|(maxedge>2))
            rows=[]
            for fi in np.where(mask)[0]:
                fi=int(fi);f=faces[fi];q=restq[fi];key=str(fi)
                row={
                    "face_index":fi,"frame":int(frame),"time":float(t),
                    "pair_categories":list(cats[fi]),"share_vertex_only":bool("SHARE_VERTEX_ONLY" in cats[fi]),
                    "edge_ratios":[float(x) for x in edge[fi]],"edge_max":float(maxedge[fi]),
                    "singular_values":[float(x) for x in s[fi]],"condition":float(cond[fi]),"area_scale":float(s[fi,0]*s[fi,1]),
                    "rest_min_angle_deg":float(q["min_angle_deg"]),"rest_aspect":float(q["aspect_longest_over_min_altitude"]),"rest_degenerate":bool(q["degenerate"]),
                    "teacher_face_weight_l1":float(teach_l1[fi]),
                    "source_surface_distance_max":float(np.max(dist[crow[f]])),
                    "source_triangle_indices":[int(tri[crow[int(v)]]) for v in f],
                    "vertex_ids":[str(cand.vertices[int(v)].candidate_vertex_id) for v in f],
                    "dominant_joints":[jids[int(np.argmax(Wteach[int(v)]))] for v in f],
                    "dominant_weights":[float(np.max(Wteach[int(v)])) for v in f],
                }
                rows.append(row)
                seen.setdefault(fi,{"rest_min_angle_deg":row["rest_min_angle_deg"],"rest_aspect":row["rest_aspect"],"share_vertex_only":row["share_vertex_only"],"max_condition":0.0,"max_edge":0.0,"occurrences":0})
                seen[fi]["max_condition"]=max(seen[fi]["max_condition"],row["condition"]);seen[fi]["max_edge"]=max(seen[fi]["max_edge"],row["edge_max"]);seen[fi]["occurrences"]+=1
            frames.append({"frame":int(frame),"time":float(t),"residual_count":len(rows),"rows":rows})
        report["clips"].append({"clip_id":clip,"frames":frames})
    uniq=[{"face_index":fi,**v} for fi,v in sorted(seen.items(),key=lambda kv:(kv[1]["max_condition"],kv[1]["max_edge"]),reverse=True)]
    report["aggregate"]={
        "unique_residual_face_count":len(uniq),
        "share_vertex_only_residual_face_count":sum(bool(x["share_vertex_only"]) for x in uniq),
        "rest_min_angle_below_7_5_count":sum(float(x["rest_min_angle_deg"])<7.5 for x in uniq),
        "rest_aspect_above_16_count":sum(float(x["rest_aspect"])>16 for x in uniq),
        "worst_condition":max((float(x["max_condition"]) for x in uniq),default=0.0),
        "worst_edge":max((float(x["max_edge"]) for x in uniq),default=0.0),
    }
    report["unique_residual_faces"]=uniq
    report["finding"]={
        "all_residuals_explained_by_frozen_triangle_quality_violation":bool(uniq and all(float(x["rest_min_angle_deg"])<7.5 or float(x["rest_aspect"])>16 for x in uniq)),
        "all_residuals_are_share_vertex_only":bool(uniq and all(bool(x["share_vertex_only"]) for x in uniq)),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_COMBINED_ORACLE_RESIDUAL_COURT_PASS",json.dumps({**report["aggregate"],**report["finding"]},sort_keys=True))
if __name__=="__main__":main()
