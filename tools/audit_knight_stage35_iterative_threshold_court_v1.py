from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, deformation_envelope_from_dict,
    mesh_policy_from_dict, qualified_camera_set_from_dict,
    qualified_skeleton_from_dict, qualified_skin_from_dict, rigging_surface_from_dict,
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
def load(rr,rel,codec):return codec(json.loads((rr/rel).read_text()))
def idx_faces(candidate,ids):
    ix={str(x):i for i,x in enumerate(ids)}
    return np.asarray([[ix[str(v)] for v in f] for f in candidate.faces],dtype=np.int64)
def actual_metrics(rest,posed,faces):
    r=rest[faces];p=posed[faces]
    rl=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],1)
    pl=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],1)
    edge=(pl/np.maximum(rl,1e-15)).max(1)
    r1=r[:,1]-r[:,0];r2=r[:,2]-r[:,0];p1=p[:,1]-p[:,0];p2=p[:,2]-p[:,0]
    l=np.linalg.norm(r1,axis=1);u=r1/np.maximum(l[:,None],1e-15);x2=(r2*u).sum(1);perp=r2-x2[:,None]*u;y2=np.linalg.norm(perp,axis=1)
    inv=np.zeros((len(faces),2,2));inv[:,0,0]=1/np.maximum(l,1e-15);inv[:,0,1]=-x2/np.maximum(l*y2,1e-15);inv[:,1,1]=1/np.maximum(y2,1e-15)
    s=np.linalg.svd(np.einsum("nij,njk->nik",np.stack((p1,p2),2),inv),compute_uv=False)
    cond=s[:,0]/np.maximum(s[:,1],1e-15);area=s[:,0]*s[:,1]
    return {"face_count":int(len(faces)),"edge_gt_4":int(np.count_nonzero(edge>4)),"edge_gt_10":int(np.count_nonzero(edge>10)),"edge_max":float(edge.max()),"edge_p99":float(np.quantile(edge,.99)),"condition_gt_16":int(np.count_nonzero(cond>16)),"area_gt_20":int(np.count_nonzero(area>20))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    original=exact(rr,"candidate",canonical_mesh_candidate_from_dict);sk=exact(rr,"skeleton",qualified_skeleton_from_dict);skin=exact(rr,"skin",qualified_skin_from_dict);cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    surface=load(rr,"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
    envelope=load(rr,"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr,"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)
    ids=[str(v.candidate_vertex_id) for v in original.vertices];rest=np.asarray([v.P for v in original.vertices],float);jids,W=_candidate_skin_weights(original,skin,sk)
    src=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    motion_frames=[]
    for clip in ("demo_run_v1","demo_slash_v1"):
        payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text());tracks,_=_tracks_for_clip(payload,sk,cams,src);times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        for fi,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=sk,tracks=tracks,time_seconds=float(t),cameras=cams);motion_frames.append((clip,fi,float(t),_skin(rest,W,jids,mats)))
    def evaluate_actual(candidate):
        faces=idx_faces(candidate,ids); rows=[]
        for clip,fi,t,posed in motion_frames:rows.append({"clip_id":clip,"frame":fi,"time":t,**actual_metrics(rest,posed,faces)})
        return {"worst_edge_max":max(r["edge_max"] for r in rows),"max_edge_gt_10":max(r["edge_gt_10"] for r in rows),"max_edge_gt_4":max(r["edge_gt_4"] for r in rows),"max_condition_gt_16":max(r["condition_gt_16"] for r in rows),"max_area_gt_20":max(r["area_gt_20"] for r in rows),"rows":rows}
    report={"schema":"RealSaS.KnightStage35IterativeThresholdCourt.v1","status":"MEASURED__NO_AUTHORITY_MUTATION","thresholds":[]}
    for risk in (0.5,0.1,0.0):
        candidate=original; iterations=[]; stopped="MAX_ITERATIONS"
        for it in range(1,5):
            comp=run_skin_topology_compatibility_v1(candidate,surface=surface,skeleton=sk,skin=skin,envelope=envelope,cameras=cams,policy=policy,risk_l1_min=float(risk))
            actual=evaluate_actual(candidate)
            row={"iteration_input":it-1,"candidate_lineage_hash":candidate.candidate_lineage_hash,"face_count":len(candidate.faces),"compatibility_passed":bool(comp["passed"]),"risky_face_count":int(comp["risky_face_count"]),"unsafe_face_count":int(comp["unsafe_face_count"]),"compat_report_hash":comp["report_hash"],"actual_motion":actual}
            iterations.append(row)
            if comp["passed"]:
                stopped="COMPATIBILITY_PASS"
                break
            candidate,directive=seam_cut_candidate_v1(candidate,comp["unsafe_face_indices"],report_hash=comp["report_hash"],max_iterations=4)
            row["repair_directive"]=directive
        final_actual=evaluate_actual(candidate)
        report["thresholds"].append({"risk_l1_min":risk,"stop_reason":stopped,"iterations":iterations,"final_candidate_lineage_hash":candidate.candidate_lineage_hash,"final_face_count":len(candidate.faces),"total_removed":len(original.faces)-len(candidate.faces),"final_actual_motion":final_actual})
    report["finding"]={}
    for row in report["thresholds"]:
        key=str(row["risk_l1_min"]).replace(".","_")
        last=row["iterations"][-1]
        report["finding"][f"risk_{key}_gate_passes_while_actual_gt4_remains"]=bool(last["compatibility_passed"] and last["actual_motion"]["max_edge_gt_4"]>0)
        report["finding"][f"risk_{key}_gate_passes_while_actual_gt10_remains"]=bool(last["compatibility_passed"] and last["actual_motion"]["max_edge_gt_10"]>0)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STAGE35_ITERATIVE_THRESHOLD_COURT_PASS",json.dumps({str(x["risk_l1_min"]):{"stop":x["stop_reason"],"removed":x["total_removed"],"faces":x["final_face_count"],"actual":{k:v for k,v in x["final_actual_motion"].items() if k!="rows"}} for x in report["thresholds"]},sort_keys=True))
if __name__=="__main__":main()
