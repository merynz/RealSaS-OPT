from __future__ import annotations

import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import _pose_skin_matrices
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    _stress_angle,
    _triangle_metrics_batch,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.demo.render_knight_motion_preview_v1 import _ctx, _skin, _tracks_for_clip

EXACT = {
    "candidate": (
        "artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json",
        "0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3",
    ),
    "skeleton": (
        "artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json",
        "e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987",
    ),
    "cameras": (
        "artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json",
        "312a9b1afe4ea1fcdc481232ac951fb6a6815fee532fbb93646e97e609b5b80a",
    ),
}
CLIPS=("demo_idle_v1","demo_run_v1","demo_slash_v1")


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def exact(rr,key,codec):
    rel,h=EXACT[key]; p=rr/rel
    if sha(p)!=h: raise RuntimeError("SHA_DRIFT:"+key)
    return codec(json.loads(p.read_text()))


def load(p,codec): return codec(json.loads(p.read_text()))


def faces_index(candidate):
    vi={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    return np.asarray([[vi[str(x)] for x in f] for f in candidate.faces],dtype=np.int64)


def surface_ids(candidate):
    out=[]
    for v in candidate.vertices:
        c=tuple(v.support_binding.coefficients)
        if len(c)!=1 or abs(float(c[0][1])-1.0)>1e-12: raise RuntimeError("NONIDENTITY_SUPPORT")
        out.append(str(c[0][0]))
    return tuple(out)


def teacher_weights(bank_path,source_path,skeleton,candidate):
    with np.load(bank_path,allow_pickle=False) as z:
        bank_ids=tuple(map(str,z["surface_ids"].tolist()))
        BW=np.asarray(z["weights"],dtype=np.float64)
        tri=np.asarray(z["source_triangle_index"],dtype=np.int64)
        bpos=np.asarray(z["target_positions_world"],dtype=np.float64)
        bpar=np.asarray(z["target_parent_indices"],dtype=np.int64)
    with np.load(source_path,allow_pickle=False) as z:
        sf=np.asarray(z["faces"],dtype=np.int64)

    joints=tuple(skeleton.joints)
    jids=tuple(str(j.canonical_joint_id) for j in joints)
    idx={x:i for i,x in enumerate(jids)}
    spos=np.asarray([j.position for j in joints],dtype=np.float64)
    spar=np.asarray([-1 if j.parent_canonical_id is None else idx[str(j.parent_canonical_id)] for j in joints],dtype=np.int64)
    C=np.linalg.norm(bpos[:,None,:]-spos[None,:,:],axis=2)
    ri,ci=linear_sum_assignment(C)
    assign=np.empty(len(bpos),dtype=np.int64); assign[ri]=ci
    matches=sum(int((-1 if bpar[i]<0 else assign[int(bpar[i])])==spar[int(assign[i])]) for i in range(len(bpar)))
    if matches!=len(bpar): raise RuntimeError("TARGET_GRAPH_ALIGNMENT_NOT_EXACT")

    Wb=np.zeros((len(bank_ids),len(jids)),dtype=np.float64)
    for bc in range(len(assign)):
        Wb[:,int(assign[bc])]=BW[:,bc]
    row={sid:i for i,sid in enumerate(bank_ids)}
    crows=np.asarray([row[sid] for sid in surface_ids(candidate)],dtype=np.int64)
    return Wb[crows],jids,tri[crows],sf


def source_cat(a,b,sets):
    if a==b:return "SAME_FACE"
    n=len(sets[a]&sets[b])
    return "SHARE_EDGE" if n==2 else "SHARE_VERTEX_ONLY" if n==1 else "NONINCIDENT"


def stress(rest,W,faces,skeleton,cameras,envelope,policy):
    angle=_stress_angle(envelope)
    frames=derive_joint_frames_from_skeleton(skeleton,cameras=cameras)
    jids=tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    hom=np.concatenate([rest,np.ones((len(rest),1))],axis=1)
    max_area=np.ones(len(faces));min_area=np.ones(len(faces));max_cond=np.ones(len(faces));max_edge=np.ones(len(faces))
    for jid in sorted(jids):
        for axis in range(3):
            for sign in (-1.0,1.0):
                sm=_pose_skin_matrices(skeleton,frames,joint_id=jid,local_axis_index=axis,degrees=sign*angle)
                mats=np.stack([sm[x] for x in jids],axis=0)
                per=np.stack([(hom@mats[k].T)[:,:3] for k in range(len(jids))],axis=1)
                posed=np.sum(per*W[:,:,None],axis=1)
                area,cond,_,emax=_triangle_metrics_batch(rest,posed,faces)
                max_area=np.maximum(max_area,area);min_area=np.minimum(min_area,area);max_cond=np.maximum(max_cond,cond);max_edge=np.maximum(max_edge,emax)
    unsafe=(max_area>float(policy.g3_max_dynamic_area_ratio))|(min_area<float(policy.g3_min_dynamic_area_ratio))|(max_cond>float(policy.g3_max_dynamic_condition_number))|(max_edge>4.0)
    return unsafe,{"unsafe_face_count":int(unsafe.sum()),"edge_max":float(max_edge.max()),"edge_p95":float(np.quantile(max_edge,.95)),"condition_max":float(max_cond.max()),"area_max":float(max_area.max()),"area_min":float(min_area.min())}


def edge_incidence(faces,mask=None):
    if mask is None: rows=faces
    else: rows=faces[np.asarray(mask,dtype=bool)]
    cnt=Counter()
    for a,b,c in rows.tolist():
        for x,y in ((a,b),(b,c),(c,a)):
            cnt[tuple(sorted((int(x),int(y))))]+=1
    return cnt


def motion(rest,W,faces,jids,skeleton,cameras,rr,source_report):
    rows=[]
    for clip in CLIPS:
        payload=json.loads((rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text())
        tracks,_=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        for fi,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,W,jids,mats)
            r=rest[faces];p=posed[faces]
            rl=np.stack([np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)],axis=1)
            pl=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
            e=(pl/np.maximum(rl,1e-15)).max(axis=1)
            rows.append({"clip":clip,"frame":fi,"time":float(t),"edge_p99":float(np.quantile(e,.99)),"edge_max":float(e.max()),"edge_gt4":int((e>4).sum()),"edge_gt10":int((e>10).sum())})
    return {"frames":rows,"worst_edge_max":max(x["edge_max"] for x in rows),"max_edge_gt4":max(x["edge_gt4"] for x in rows),"max_edge_gt10":max(x["edge_gt10"] for x in rows)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--teacher-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)

    faces=faces_index(cand)
    rest=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    W,jids,tri,sf=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    unsafe,stress_summary=stress(rest,W,faces,sk,cams,env,policy)

    sets=[set(map(int,x)) for x in sf.tolist()]
    classes=[]
    for face in faces:
        ts=[int(tri[int(v)]) for v in face]
        cats=(source_cat(ts[0],ts[1],sets),source_cat(ts[1],ts[2],sets),source_cat(ts[2],ts[0],sets))
        classes.append("NONINCIDENT" if "NONINCIDENT" in cats else "STRICT_OR_SHARE_VERTEX")
    classes=np.asarray(classes,dtype=object)

    replay=json.loads(Path("canonical/KNIGHT_STAGE18_DENSE_FACE_REPLAY_AUDIT_V1_20260928.json").read_text())
    invented_idx=set()
    # Reconstruct exact dense-supported mask using report's listed invented compact index triples is not candidate-index keyed.
    # Use the already sealed count only for total; classify unsafe invention by exact source report below when available.
    invented_total=int(replay["face_provenance"]["current_face_invented_by_edge_clique"])

    base_inc=edge_incidence(faces)
    keep=~unsafe
    cut_inc=edge_incidence(faces,keep)
    base_boundary={e for e,n in base_inc.items() if n==1}
    cut_boundary={e for e,n in cut_inc.items() if n==1}
    new_boundary=cut_boundary-base_boundary

    # Connected components of retained vertex graph.
    nbr=defaultdict(set)
    for e,n in cut_inc.items():
        if n>0:
            a0,b0=e;nbr[a0].add(b0);nbr[b0].add(a0)
    used=set(int(x) for x in faces[keep].reshape(-1))
    seen=set(); comps=[]
    for s in sorted(used):
        if s in seen:continue
        stack=[s];seen.add(s);c=0
        while stack:
            u=stack.pop();c+=1
            for v in nbr[u]:
                if v not in seen:seen.add(v);stack.append(v)
        comps.append(c)
    comps.sort(reverse=True)

    src_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    mot=motion(rest,W,faces,jids,sk,cams,rr,src_report)

    report={
        "schema":"RealSaS.KnightTeacherWeightTopologyOnlyCourt.v1",
        "status":"ORACLE_DIAGNOSTIC_ONLY__NO_REPAIR",
        "teacher_role":"WEIGHT_ORACLE_ONLY",
        "stress":stress_summary,
        "actual_motion":mot,
        "unsafe_classification":{
            "unsafe_total":int(unsafe.sum()),
            "unsafe_source_nonincident":int(np.count_nonzero(unsafe&(classes=="NONINCIDENT"))),
            "unsafe_source_local_or_share_vertex":int(np.count_nonzero(unsafe&(classes!="NONINCIDENT"))),
            "candidate_total_source_nonincident":int(np.count_nonzero(classes=="NONINCIDENT")),
            "sealed_stage18_clique_invented_total":invented_total,
        },
        "seam_cut_topology_consequence":{
            "face_count_before":int(len(faces)),
            "face_count_after_delete":int(keep.sum()),
            "deleted_face_count":int(unsafe.sum()),
            "boundary_edges_before":int(len(base_boundary)),
            "boundary_edges_after":int(len(cut_boundary)),
            "new_boundary_edges_created":int(len(new_boundary)),
            "retained_vertex_component_count":int(len(comps)),
            "retained_vertex_component_sizes":comps[:64],
        },
        "finding":{
            "correct_weights_still_leave_topology_unsafe_faces":bool(unsafe.sum()>0),
            "unsafe_is_dominated_by_source_nonincident":bool(np.count_nonzero(unsafe&(classes=="NONINCIDENT"))>0.8*max(1,int(unsafe.sum()))),
            "face_deletion_creates_new_open_boundaries":bool(len(new_boundary)>0),
        }
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TEACHER_WEIGHT_TOPOLOGY_ONLY_COURT_PASS",json.dumps({
        "stress":stress_summary,"motion":{k:v for k,v in mot.items() if k!="frames"},
        "unsafe":report["unsafe_classification"],"cut":report["seam_cut_topology_consequence"],"finding":report["finding"]
    },sort_keys=True))

if __name__=="__main__":main()
