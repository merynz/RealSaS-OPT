from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, deformation_envelope_from_dict,
    mesh_policy_from_dict, qualified_camera_set_from_dict,
    qualified_skeleton_from_dict, qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1, seam_cut_candidate_v1, _face_indices, _skin_l1_per_face,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights,_ctx,_skin,_tracks_for_clip

def load(p,codec): return codec(json.loads(p.read_text()))
def face_indices(candidate,ids):
    ix={str(x):i for i,x in enumerate(ids)}
    return np.asarray([[ix[str(v)] for v in f] for f in candidate.faces],dtype=np.int64)
def actual(rest,posed,faces):
    r=rest[faces];p=posed[faces]
    rl=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],axis=1)
    pl=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
    e=(pl/np.maximum(rl,1e-15)).max(axis=1)
    return {"edge_max":float(e.max()),"edge_gt_4":int(np.count_nonzero(e>4)),"edge_gt_10":int(np.count_nonzero(e>10))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json",canonical_mesh_candidate_from_dict)
    surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
    sk=load(rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json",qualified_skeleton_from_dict)
    skin=load(rr/"artifacts/32_SKIN_QUALIFIED/qualified_skin.json",qualified_skin_from_dict)
    cams=tuple(sorted(load(rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)

    rest0,w0,faces0=_candidate_skin_matrix(cand,surface=surface,skeleton=sk,skin=skin)
    l1=_skin_l1_per_face(w0,faces0)
    exact_zero=np.isclose(l1,0.0,atol=1e-12)
    risk0=run_skin_topology_compatibility_v1(cand,surface=surface,skeleton=sk,skin=skin,envelope=env,cameras=cams,policy=policy,risk_l1_min=0.0,stress_all_faces=False)
    full=run_skin_topology_compatibility_v1(cand,surface=surface,skeleton=sk,skin=skin,envelope=env,cameras=cams,policy=policy,risk_l1_min=0.0,stress_all_faces=True)
    r0=set(map(int,risk0["unsafe_face_indices"])); fa=set(map(int,full["unsafe_face_indices"]))
    extra=sorted(fa-r0)
    missing=sorted(r0-fa)
    repaired,directive=seam_cut_candidate_v1(cand,full["unsafe_face_indices"],report_hash=full["report_hash"])
    after=run_skin_topology_compatibility_v1(repaired,surface=surface,skeleton=sk,skin=skin,envelope=env,cameras=cams,policy=policy,risk_l1_min=0.0,stress_all_faces=True)

    ids=[str(v.candidate_vertex_id) for v in cand.vertices]; rest=np.asarray([v.P for v in cand.vertices],float);jids,W=_candidate_skin_weights(cand,skin,sk)
    src=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    rf=face_indices(repaired,ids);frames=[]
    for clip in ("demo_run_v1","demo_slash_v1"):
      payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text());tracks,_=_tracks_for_clip(payload,sk,cams,src)
      times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
      for fi,t in enumerate(times):
        mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams);posed=_skin(rest,W,jids,mats)
        frames.append({"clip":clip,"frame":fi,"time":float(t),**actual(rest,posed,rf)})
    report={
      "schema":"RealSaS.KnightAllFaceSkinTopologyStressCourt.v1","status":"MEASURED__NO_AUTHORITY_MUTATION",
      "static":{"face_count":len(cand.faces),"exact_zero_l1_face_count":int(exact_zero.sum()),"risk0_unsafe":len(r0),"all_face_unsafe":len(fa),"all_face_extra_unsafe":len(extra),"risk0_missing_from_all_face":len(missing)},
      "risk0":{"passed":risk0["passed"],"risky":risk0["risky_face_count"],"unsafe":risk0["unsafe_face_count"],"hash":risk0["report_hash"]},
      "all_face":{"passed":full["passed"],"risky":full["risky_face_count"],"unsafe":full["unsafe_face_count"],"hash":full["report_hash"]},
      "extra_unsafe":[{"face_index":i,"skin_l1":float(l1[i]),"exact_zero_l1":bool(exact_zero[i]),"vertex_ids":list(map(str,cand.faces[i]))} for i in extra],
      "repair":{"removed":len(cand.faces)-len(repaired.faces),"face_count":len(repaired.faces),"lineage":repaired.candidate_lineage_hash,"post_repair_compatibility_passed":bool(after["passed"]),"post_repair_unsafe":int(after["unsafe_face_count"]),"directive":directive},
      "actual_motion":{"frames":frames,"worst_edge_max":max(x["edge_max"] for x in frames),"max_edge_gt_4":max(x["edge_gt_4"] for x in frames),"max_edge_gt_10":max(x["edge_gt_10"] for x in frames)},
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_ALL_FACE_SKIN_TOPOLOGY_STRESS_COURT_PASS",json.dumps({**report["static"],**report["repair"],**{k:v for k,v in report["actual_motion"].items() if k!="frames"}},sort_keys=True))
if __name__=="__main__":main()
