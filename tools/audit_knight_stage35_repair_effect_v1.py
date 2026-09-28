from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np

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

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def load(rr,k,codec):
    rel,h=EXACT[k];p=rr/rel
    if sha(p)!=h:raise RuntimeError("SHA_DRIFT:"+k)
    return codec(json.loads(p.read_text()))
def face_indices(candidate,base_vertex_ids):
    idx={str(v):i for i,v in enumerate(base_vertex_ids)}
    return np.asarray([[idx[str(x)] for x in f] for f in candidate.faces],dtype=np.int64)
def metrics(rest,posed,faces):
    r=rest[faces];p=posed[faces]
    rl=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],axis=1)
    pl=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
    edge=(pl/np.maximum(rl,1e-15)).max(axis=1)
    r1=r[:,1]-r[:,0];r2=r[:,2]-r[:,0];p1=p[:,1]-p[:,0];p2=p[:,2]-p[:,0]
    l=np.linalg.norm(r1,axis=1);u=r1/np.maximum(l[:,None],1e-15);x2=np.sum(r2*u,axis=1);perp=r2-x2[:,None]*u;y2=np.linalg.norm(perp,axis=1)
    inv=np.zeros((len(faces),2,2));inv[:,0,0]=1/np.maximum(l,1e-15);inv[:,0,1]=-x2/np.maximum(l*y2,1e-15);inv[:,1,1]=1/np.maximum(y2,1e-15)
    s=np.linalg.svd(np.einsum("nij,njk->nik",np.stack((p1,p2),axis=2),inv),compute_uv=False)
    cond=s[:,0]/np.maximum(s[:,1],1e-15);area=s[:,0]*s[:,1]
    return {"edge_p95":float(np.quantile(edge,.95)),"edge_p99":float(np.quantile(edge,.99)),"edge_max":float(edge.max()),"edge_gt_2":int(np.count_nonzero(edge>2)),"edge_gt_4":int(np.count_nonzero(edge>4)),"edge_gt_10":int(np.count_nonzero(edge>10)),"condition_gt_16":int(np.count_nonzero(cond>16)),"area_gt_20":int(np.count_nonzero(area>20))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    base=load(rr,"candidate",canonical_mesh_candidate_from_dict);sk=load(rr,"skeleton",qualified_skeleton_from_dict);skin=load(rr,"skin",qualified_skin_from_dict);cams=tuple(sorted(load(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    compat_path=rr/"artifacts/35_DYNAMIC_MECHANICAL_MESH_QUALIFIED/skin_topology_compatibility.json"
    repaired_path=rr/"artifacts/35_DYNAMIC_MECHANICAL_MESH_QUALIFIED/repaired_stage18_candidate.json"
    directive_path=rr/"artifacts/35_DYNAMIC_MECHANICAL_MESH_QUALIFIED/skin_topology_repair_directive.json"
    existence={"compatibility":compat_path.is_file(),"repaired":repaired_path.is_file(),"directive":directive_path.is_file()}
    if not all(existence.values()):raise RuntimeError("STAGE35_REPAIR_ARTIFACTS_MISSING:"+json.dumps(existence,sort_keys=True))
    compat=json.loads(compat_path.read_text());directive=json.loads(directive_path.read_text());repaired=canonical_mesh_candidate_from_dict(json.loads(repaired_path.read_text()))
    if str(directive.get("source_candidate_lineage_hash"))!=base.candidate_lineage_hash:raise RuntimeError("REPAIR_SOURCE_LINEAGE_DRIFT")
    if str(directive.get("repaired_candidate_lineage_hash"))!=repaired.candidate_lineage_hash:raise RuntimeError("REPAIR_TARGET_LINEAGE_DRIFT")
    base_ids=[str(v.candidate_vertex_id) for v in base.vertices]
    rep_ids=[str(v.candidate_vertex_id) for v in repaired.vertices]
    if rep_ids!=base_ids:raise RuntimeError("SEAM_CUT_VERTEX_SET_MUTATED")
    rest=np.asarray([v.P for v in base.vertices],float);jids,W=_candidate_skin_weights(base,skin,sk)
    bf=face_indices(base,base_ids);rf=face_indices(repaired,base_ids)
    # exact removed face identity check
    rep_set={tuple(map(str,f)) for f in repaired.faces};removed=[i for i,f in enumerate(base.faces) if tuple(map(str,f)) not in rep_set]
    declared=sorted(map(int,compat.get("unsafe_face_indices") or ()))
    if sorted(removed)!=declared:raise RuntimeError("REPAIR_REMOVAL_SET_DRIFT")
    out={"schema":"RealSaS.KnightStage35RepairEffectAudit.v1","status":"MEASURED__NO_NEW_REPAIR","compatibility":{"passed":bool(compat.get("passed")),"unsafe_face_count":int(compat.get("unsafe_face_count",-1)),"risky_face_count":int(compat.get("risky_face_count",-1)),"risk_l1_min":compat.get("risk_l1_min"),"max_edge_ratio_limit":compat.get("max_edge_ratio_limit"),"report_hash":compat.get("report_hash")},"directive":directive,"face_counts":{"before":len(base.faces),"removed":len(removed),"after":len(repaired.faces)},"clips":[]}
    for clip in ("demo_run_v1","demo_slash_v1"):
        payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text());tracks,_=_tracks_for_clip(payload,sk,cams,source_report);times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")));frames=[]
        for frame,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams);posed=_skin(rest,W,jids,mats)
            frames.append({"frame":frame,"time":float(t),"before":metrics(rest,posed,bf),"after_stage35_seam_cut":metrics(rest,posed,rf)})
        out["clips"].append({"clip_id":clip,"frames":frames})
    out["aggregate"]={
        "before_worst_edge_max":max(f["before"]["edge_max"] for c in out["clips"] for f in c["frames"]),
        "after_worst_edge_max":max(f["after_stage35_seam_cut"]["edge_max"] for c in out["clips"] for f in c["frames"]),
        "before_max_edge_gt_10":max(f["before"]["edge_gt_10"] for c in out["clips"] for f in c["frames"]),
        "after_max_edge_gt_10":max(f["after_stage35_seam_cut"]["edge_gt_10"] for c in out["clips"] for f in c["frames"]),
        "before_max_edge_gt_4":max(f["before"]["edge_gt_4"] for c in out["clips"] for f in c["frames"]),
        "after_max_edge_gt_4":max(f["after_stage35_seam_cut"]["edge_gt_4"] for c in out["clips"] for f in c["frames"]),
    }
    out["finding"]={"existing_stage35_repair_eliminates_catastrophic_gt10_on_actual_motion":bool(out["aggregate"]["after_max_edge_gt_10"]==0),"existing_stage35_repair_eliminates_gt4_on_actual_motion":bool(out["aggregate"]["after_max_edge_gt_4"]==0)}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STAGE35_REPAIR_EFFECT_AUDIT_PASS",json.dumps({**out["face_counts"],**out["aggregate"],**out["finding"]},sort_keys=True))
if __name__=="__main__":main()
