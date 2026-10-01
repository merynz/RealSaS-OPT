from __future__ import annotations

import argparse, hashlib, json
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
def compact_inverse(points,faces,divisions,component_aware):
    p=np.asarray(points,float);f=np.asarray(faces,int)
    lo=p.min(0);span=np.maximum(p.max(0)-lo,1e-12)
    keys=np.floor((p-lo)/span*int(divisions)).astype(np.int64);keys=np.clip(keys,0,int(divisions)-1)
    if component_aware:
        keys=np.column_stack((mesh_connected_component_labels_v1(len(p),f),keys))
    _,inv=np.unique(keys,axis=0,return_inverse=True)
    return np.asarray(inv,int)
def mapped_face_set(faces,inv):
    mf=inv[np.asarray(faces,int)]
    keep=(mf[:,0]!=mf[:,1])&(mf[:,1]!=mf[:,2])&(mf[:,2]!=mf[:,0])
    return {tuple(map(int,x)) for x in np.unique(np.sort(mf[keep],axis=1),axis=0).tolist()}
def deform(rest,posed,faces):
    r=rest[faces];p=posed[faces]
    rl=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],1)
    pl=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],1)
    edge=(pl/np.maximum(rl,1e-15)).max(1)
    r1=r[:,1]-r[:,0];r2=r[:,2]-r[:,0];p1=p[:,1]-p[:,0];p2=p[:,2]-p[:,0]
    l=np.linalg.norm(r1,axis=1);u=r1/np.maximum(l[:,None],1e-15);x2=(r2*u).sum(1);perp=r2-x2[:,None]*u;y2=np.linalg.norm(perp,axis=1)
    inv=np.zeros((len(faces),2,2));inv[:,0,0]=1/np.maximum(l,1e-15);inv[:,0,1]=-x2/np.maximum(l*y2,1e-15);inv[:,1,1]=1/np.maximum(y2,1e-15)
    s=np.linalg.svd(np.einsum("nij,njk->nik",np.stack((p1,p2),2),inv),compute_uv=False)
    return edge,s[:,0]/np.maximum(s[:,1],1e-15),s[:,0]*s[:,1]
def group(edge,cond,area,mask):
    x=edge[mask];c=cond[mask];a=area[mask]
    return {"count":int(len(x)),"edge_p95":float(np.quantile(x,.95)) if len(x) else None,"edge_p99":float(np.quantile(x,.99)) if len(x) else None,"edge_max":float(x.max()) if len(x) else None,"edge_gt_4":int(np.count_nonzero(x>4)),"edge_gt_10":int(np.count_nonzero(x>10)),"condition_gt_16":int(np.count_nonzero(c>16)),"area_gt_20":int(np.count_nonzero(a>20))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict);sk=exact(rr,"skeleton",qualified_skeleton_from_dict);skin=exact(rr,"skin",qualified_skin_from_dict);cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    surf=rigging_surface_from_dict(json.loads((rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json").read_text()))
    zero=signed_zero_surface_from_dict(json.loads((rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json").read_text()))
    norm=normalization_domain_from_dict(json.loads((rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json").read_text()))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],float);df=np.asarray(z["faces"],int)
    world=np.asarray(norm.center_xyz,float)[None]+vn*float(norm.half_extent)
    md=dict(surf.metadata or {});inv=compact_inverse(world,df,int(md["compact_voxel_divisions"]),bool(md["component_aware_compaction"]));mset=mapped_face_set(df,inv)
    sid={str(n.surface_id):i for i,n in enumerate(surf.surface_nodes)}
    cvi={str(v.candidate_vertex_id):i for i,v in enumerate(cand.vertices)}
    faces=np.asarray([[cvi[str(x)] for x in f] for f in cand.faces],int)
    compact_face=[]
    for f in cand.faces:
        row=[]
        for x in f:
            v=cand.vertices[cvi[str(x)]];co=tuple(v.support_binding.coefficients)
            if len(co)!=1 or abs(float(co[0][1])-1)>1e-12:raise RuntimeError("NONIDENTITY")
            row.append(sid[str(co[0][0])])
        compact_face.append(tuple(sorted(row)))
    dense=np.asarray([x in mset for x in compact_face],bool);invented=~dense
    if int(invented.sum())!=515:raise RuntimeError(f"INVENTED_COUNT_DRIFT:{invented.sum()}")
    rest=np.asarray([v.P for v in cand.vertices],float);jids,W=_candidate_skin_weights(cand,skin,sk)
    wf=W[faces];l1=np.maximum.reduce([np.abs(wf[:,0]-wf[:,1]).sum(1),np.abs(wf[:,1]-wf[:,2]).sum(1),np.abs(wf[:,2]-wf[:,0]).sum(1)])
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    report={"schema":"RealSaS.KnightStage18InventedFaceMotionContribution.v1","status":"MEASURED__NO_REPAIR","static":{"face_count":len(faces),"dense_mapped":int(dense.sum()),"clique_invented":int(invented.sum()),"invented_high_weight_l1_gt_1":int(np.count_nonzero(invented&(l1>1))),"dense_high_weight_l1_gt_1":int(np.count_nonzero(dense&(l1>1)))},"clips":[]}
    for clip in ("demo_idle_v1","demo_run_v1","demo_slash_v1"):
        payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text());tracks,_=_tracks_for_clip(payload,sk,cams,source_report);times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")));frames=[]
        for fi,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams);posed=_skin(rest,W,jids,mats);e,c0,a0=deform(rest,posed,faces)
            frames.append({"frame":fi,"time":float(t),"ALL":group(e,c0,a0,np.ones(len(faces),bool)),"DENSE_MAPPED":group(e,c0,a0,dense),"CLIQUE_INVENTED":group(e,c0,a0,invented)})
        report["clips"].append({"clip_id":clip,"frames":frames})
    # Worst actual-motion counts, and fraction of catastrophic faces attributable to invented set.
    b=[]
    for c in report["clips"]:
        if c["clip_id"]=="demo_idle_v1":continue
        for f in c["frames"]:
            b.append({
              "all_gt10":f["ALL"]["edge_gt_10"],"inv_gt10":f["CLIQUE_INVENTED"]["edge_gt_10"],
              "all_gt4":f["ALL"]["edge_gt_4"],"inv_gt4":f["CLIQUE_INVENTED"]["edge_gt_4"],
            })
    report["aggregate"]={
      "max_all_edge_gt_10":max(x["all_gt10"] for x in b),"max_invented_edge_gt_10":max(x["inv_gt10"] for x in b),
      "max_all_edge_gt_4":max(x["all_gt4"] for x in b),"max_invented_edge_gt_4":max(x["inv_gt4"] for x in b),
      "invented_fraction_of_gt10_at_each_frame":[float(x["inv_gt10"]/x["all_gt10"]) if x["all_gt10"] else 0.0 for x in b],
      "invented_fraction_of_gt4_at_each_frame":[float(x["inv_gt4"]/x["all_gt4"]) if x["all_gt4"] else 0.0 for x in b],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STAGE18_INVENTED_FACE_MOTION_CONTRIBUTION_PASS",json.dumps({**report["static"],**report["aggregate"]},sort_keys=True))
if __name__=="__main__":main()
