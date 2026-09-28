from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, deformation_envelope_from_dict,
    mechanical_partition_from_dict, mesh_policy_from_dict,
    qualified_camera_set_from_dict, qualified_skeleton_from_dict,
    qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1, seam_cut_candidate_v1,
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
def exact(rr,k,codec):
    rel,h=EXACT[k];p=rr/rel
    if sha(p)!=h:raise RuntimeError("SHA_DRIFT:"+k)
    return codec(json.loads(p.read_text()))
def load_schema(rr,rel,codec):
    p=rr/rel
    if not p.is_file():raise RuntimeError("MISSING:"+rel)
    return codec(json.loads(p.read_text()))
def face_indices(candidate, ids):
    ix={str(x):i for i,x in enumerate(ids)}
    return np.asarray([[ix[str(v)] for v in f] for f in candidate.faces],dtype=np.int64)
def metrics(rest,posed,faces):
    r=rest[faces];p=posed[faces]
    rl=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],axis=1)
    pl=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
    er=pl/np.maximum(rl,1e-15); edge=er.max(axis=1)
    r1=r[:,1]-r[:,0];r2=r[:,2]-r[:,0];p1=p[:,1]-p[:,0];p2=p[:,2]-p[:,0]
    l=np.linalg.norm(r1,axis=1);u=r1/np.maximum(l[:,None],1e-15);x2=np.sum(r2*u,axis=1);perp=r2-x2[:,None]*u;y2=np.linalg.norm(perp,axis=1)
    inv=np.zeros((len(faces),2,2));inv[:,0,0]=1/np.maximum(l,1e-15);inv[:,0,1]=-x2/np.maximum(l*y2,1e-15);inv[:,1,1]=1/np.maximum(y2,1e-15)
    s=np.linalg.svd(np.einsum("nij,njk->nik",np.stack((p1,p2),axis=2),inv),compute_uv=False)
    cond=s[:,0]/np.maximum(s[:,1],1e-15); area=s[:,0]*s[:,1]
    return {
      "count":int(len(faces)),
      "edge_p95":float(np.quantile(edge,.95)),"edge_p99":float(np.quantile(edge,.99)),"edge_max":float(edge.max()),
      "edge_gt_2":int(np.count_nonzero(edge>2)),"edge_gt_4":int(np.count_nonzero(edge>4)),"edge_gt_10":int(np.count_nonzero(edge>10)),
      "condition_p95":float(np.quantile(cond,.95)),"condition_gt_16":int(np.count_nonzero(cond>16)),
      "area_gt_20":int(np.count_nonzero(area>20)),
    }
def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    skin=exact(rr,"skin",qualified_skin_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    surface=load_schema(rr,"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
    envelope=load_schema(rr,"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load_schema(rr,"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)

    comp=run_skin_topology_compatibility_v1(
      cand,surface=surface,skeleton=sk,skin=skin,envelope=envelope,cameras=cams,policy=policy
    )
    if comp["passed"]:
        repaired=cand
        directive={"status":"NO_REPAIR_REQUIRED","removed_face_count":0}
    else:
        repaired,directive=seam_cut_candidate_v1(cand,comp["unsafe_face_indices"],report_hash=comp["report_hash"])

    ids=[str(v.candidate_vertex_id) for v in cand.vertices]
    rids=[str(v.candidate_vertex_id) for v in repaired.vertices]
    if rids!=ids:raise RuntimeError("SEAM_CUT_VERTEX_ID_ORDER_MUTATED")
    bf=face_indices(cand,ids); rf=face_indices(repaired,ids)
    rest=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    jids,W=_candidate_skin_weights(cand,skin,sk)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    report={
      "schema":"RealSaS.KnightStage35InMemoryRepairEffect.v1","status":"MEASURED__NO_AUTHORITY_MUTATION",
      "compatibility":comp,"directive":directive,
      "face_counts":{"before":len(cand.faces),"after":len(repaired.faces),"removed":len(cand.faces)-len(repaired.faces)},
      "repaired_candidate_lineage_hash":repaired.candidate_lineage_hash,
      "clips":[],
    }
    for clip in ("demo_idle_v1","demo_run_v1","demo_slash_v1"):
        payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text())
        tracks,_=_tracks_for_clip(payload,sk,cams,source_report)
        times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        frames=[]
        for fi,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams)
            posed=_skin(rest,W,jids,mats)
            frames.append({"frame":fi,"time":float(t),"before":metrics(rest,posed,bf),"after":metrics(rest,posed,rf)})
        report["clips"].append({"clip_id":clip,"frames":frames})
    report["aggregate"]={
      "before_worst_edge_max":max(fr["before"]["edge_max"] for c in report["clips"] for fr in c["frames"]),
      "after_worst_edge_max":max(fr["after"]["edge_max"] for c in report["clips"] for fr in c["frames"]),
      "before_max_edge_gt_10":max(fr["before"]["edge_gt_10"] for c in report["clips"] for fr in c["frames"]),
      "after_max_edge_gt_10":max(fr["after"]["edge_gt_10"] for c in report["clips"] for fr in c["frames"]),
      "before_max_edge_gt_4":max(fr["before"]["edge_gt_4"] for c in report["clips"] for fr in c["frames"]),
      "after_max_edge_gt_4":max(fr["after"]["edge_gt_4"] for c in report["clips"] for fr in c["frames"]),
      "after_max_condition_gt_16":max(fr["after"]["condition_gt_16"] for c in report["clips"] for fr in c["frames"]),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STAGE35_IN_MEMORY_REPAIR_EFFECT_PASS",json.dumps({"compat_passed":comp["passed"],"unsafe":comp["unsafe_face_count"],**report["face_counts"],**report["aggregate"]},sort_keys=True))
if __name__=="__main__":main()
