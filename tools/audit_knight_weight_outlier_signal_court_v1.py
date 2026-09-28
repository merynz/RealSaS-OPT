from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,
)
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights, _ctx

EXACT = {
    "candidate": (
        "artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json",
        "0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3",
    ),
    "skeleton": (
        "artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json",
        "e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987",
    ),
    "skin": (
        "artifacts/32_SKIN_QUALIFIED/qualified_skin.json",
        "f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09",
    ),
    "cameras": (
        "artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json",
        "312a9b1afe4ea1fcdc481232ac951fb6a6815fee532fbb93646e97e609b5b80a",
    ),
}


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def exact(rr,key,codec):
    rel,expected=EXACT[key]; p=rr/rel
    actual=sha(p)
    if actual!=expected: raise RuntimeError(f"SHA_DRIFT:{key}:{actual}:{expected}")
    return codec(json.loads(p.read_text()))


def load(path,codec): return codec(json.loads(path.read_text()))


def face_indices(candidate):
    vi={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    return np.asarray([[vi[str(x)] for x in f] for f in candidate.faces],dtype=np.int64)


def surface_ids(candidate):
    out=[]
    for v in candidate.vertices:
        c=tuple(v.support_binding.coefficients)
        if len(c)!=1 or abs(float(c[0][1])-1.0)>1e-12:
            raise RuntimeError("NON_IDENTITY_BINDING")
        out.append(str(c[0][0]))
    return tuple(out)


def teacher_matrix(bank_path,skeleton,candidate,joint_ids):
    with np.load(bank_path,allow_pickle=False) as z:
        sids=tuple(map(str,z["surface_ids"].tolist()))
        BW=np.asarray(z["weights"],dtype=np.float64)
        valid=np.asarray(z["teacher_valid_mask"],dtype=np.uint8).astype(bool)
        bpos=np.asarray(z["target_positions_world"],dtype=np.float64)
        bpar=np.asarray(z["target_parent_indices"],dtype=np.int64)
    joints=tuple(skeleton.joints)
    ids=[str(j.canonical_joint_id) for j in joints]
    idx={x:i for i,x in enumerate(ids)}
    spos=np.asarray([j.position for j in joints],dtype=np.float64)
    spar=np.asarray([-1 if j.parent_canonical_id is None else idx[str(j.parent_canonical_id)] for j in joints],dtype=np.int64)
    C=np.linalg.norm(bpos[:,None,:]-spos[None,:,:],axis=2)
    ri,ci=linear_sum_assignment(C)
    assign=np.empty(len(bpos),dtype=np.int64); assign[ri]=ci
    matches=sum(int((-1 if bpar[i]<0 else assign[int(bpar[i])])==spar[int(assign[i])]) for i in range(len(bpar)))
    if matches!=len(bpar): raise RuntimeError("TEACHER_TARGET_GRAPH_ALIGNMENT_NOT_EXACT")
    jix={j:i for i,j in enumerate(joint_ids)}
    WT=np.zeros((len(sids),len(joint_ids)),dtype=np.float64)
    for bc in range(len(assign)):
        WT[:,jix[ids[int(assign[bc])]]]=BW[:,bc]
    row={sid:i for i,sid in enumerate(sids)}
    ci=np.asarray([row[sid] for sid in surface_ids(candidate)],dtype=np.int64)
    return WT[ci],valid[ci]


def build_neighbors(faces,n):
    nbr=[set() for _ in range(n)]
    for a,b,c in faces.tolist():
        nbr[a].update((b,c)); nbr[b].update((a,c)); nbr[c].update((a,b))
    return [tuple(sorted(x)) for x in nbr]


def connected_components(mask,neighbors):
    mask=np.asarray(mask,dtype=bool)
    seen=np.zeros(len(mask),dtype=bool)
    comps=[]
    for s in np.where(mask)[0]:
        if seen[s]: continue
        q=[int(s)];seen[s]=True;comp=[]
        while q:
            u=q.pop();comp.append(u)
            for v in neighbors[u]:
                if mask[v] and not seen[v]:
                    seen[v]=True;q.append(int(v))
        comps.append(comp)
    return sorted(comps,key=len,reverse=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    skin=exact(rr,"skin",qualified_skin_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)

    faces=face_indices(cand)
    joint_ids,W=_candidate_skin_weights(cand,skin,sk)
    Wt,valid=teacher_matrix(a.teacher_bank,sk,cand,joint_ids)
    terr=np.sum(np.abs(W-Wt),axis=1)

    full=run_skin_topology_compatibility_v1(
        cand,surface=surface,skeleton=sk,skin=skin,envelope=env,cameras=cams,policy=policy,
        risk_l1_min=0.0,stress_all_faces=True,
    )
    unsafe=np.zeros(len(faces),dtype=bool)
    unsafe[np.asarray(full["unsafe_face_indices"],dtype=np.int64)]=True
    unsafe_inc=np.zeros(len(W),dtype=np.int64)
    np.add.at(unsafe_inc,faces[unsafe].reshape(-1),1)

    nbr=build_neighbors(faces,len(W))
    med=np.zeros_like(W)
    dispersion=np.full(len(W),np.inf,dtype=np.float64)
    deviation=np.full(len(W),np.inf,dtype=np.float64)
    degree=np.asarray([len(x) for x in nbr],dtype=np.int64)
    for i,rows in enumerate(nbr):
        if not rows: continue
        X=W[np.asarray(rows,dtype=np.int64)]
        m=np.median(X,axis=0)
        m=np.maximum(m,0.0)
        s=float(m.sum())
        if s<=1e-12: continue
        m/=s
        med[i]=m
        deviation[i]=float(np.sum(np.abs(W[i]-m)))
        dispersion[i]=float(np.median(np.sum(np.abs(X-m[None,:]),axis=1)))

    bad1=terr>1.0
    bad01=terr>0.1
    thresholds=[]
    for dev in (0.25,0.5,0.75,1.0):
        for disp in (0.10,0.25,0.50):
            sel=(unsafe_inc>0)&(degree>=3)&(deviation>dev)&(dispersion<disp)
            thresholds.append({
                "deviation_min":dev,"dispersion_max":disp,
                "selected":int(sel.sum()),
                "teacher_invalid_selected":int(np.count_nonzero(sel&~valid)),
                "teacher_valid_selected":int(np.count_nonzero(sel&valid)),
                "bad_l1_gt1_selected":int(np.count_nonzero(sel&bad1)),
                "bad_l1_gt1_total":int(bad1.sum()),
                "bad_l1_gt1_recall":float(np.count_nonzero(sel&bad1)/max(1,bad1.sum())),
                "selected_precision_for_l1_gt1":float(np.count_nonzero(sel&bad1)/max(1,sel.sum())),
                "bad_l1_gt0_1_recall":float(np.count_nonzero(sel&bad01)/max(1,bad01.sum())),
            })

    comps=connected_components(bad1,nbr)
    top=[]
    for i in np.argsort(terr)[::-1][:256]:
        top.append({
            "vertex_index":int(i),
            "surface_id":surface_ids(cand)[int(i)],
            "teacher_valid":bool(valid[i]),
            "teacher_l1_error":float(terr[i]),
            "unsafe_incident_face_count":int(unsafe_inc[i]),
            "degree":int(degree[i]),
            "neighbor_median_deviation_l1":float(deviation[i]),
            "neighbor_consensus_dispersion_l1":float(dispersion[i]),
            "pred_dominant_joint":joint_ids[int(np.argmax(W[i]))],
            "teacher_dominant_joint":joint_ids[int(np.argmax(Wt[i]))],
            "pred_dominant_weight":float(np.max(W[i])),
            "teacher_dominant_weight":float(np.max(Wt[i])),
        })

    report={
        "schema":"RealSaS.KnightWeightOutlierSignalCourt.v1",
        "status":"DIAGNOSTIC__NO_REPAIR",
        "static":{
            "vertex_count":len(W),
            "unsafe_face_count":int(unsafe.sum()),
            "vertices_incident_to_unsafe":int(np.count_nonzero(unsafe_inc>0)),
            "teacher_l1_gt1_count":int(bad1.sum()),
            "teacher_l1_gt0_1_count":int(bad01.sum()),
            "teacher_l1_gt1_invalid_count":int(np.count_nonzero(bad1&~valid)),
            "teacher_l1_gt1_valid_count":int(np.count_nonzero(bad1&valid)),
            "bad_gt1_component_count":len(comps),
            "bad_gt1_component_sizes":[len(x) for x in comps[:64]],
        },
        "signal_distribution":{
            "bad_gt1_deviation_p50":float(np.quantile(deviation[bad1],.5)),
            "bad_gt1_deviation_p95":float(np.quantile(deviation[bad1],.95)),
            "good_le0_1_deviation_p50":float(np.quantile(deviation[~bad01],.5)),
            "good_le0_1_deviation_p95":float(np.quantile(deviation[~bad01],.95)),
            "bad_gt1_dispersion_p50":float(np.quantile(dispersion[bad1],.5)),
            "bad_gt1_unsafe_incidence_nonzero":int(np.count_nonzero(bad1&(unsafe_inc>0))),
        },
        "threshold_court":thresholds,
        "top_error_rows":top,
        "claim_boundary":"Teacher is evaluation-only. Product-available signals are Stage35 unsafe incidence and predicted-weight neighborhood statistics.",
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_WEIGHT_OUTLIER_SIGNAL_COURT_PASS",json.dumps({
        "static":report["static"],"signal_distribution":report["signal_distribution"],
        "best_by_recall":sorted(thresholds,key=lambda x:(x["bad_l1_gt1_recall"],x["selected_precision_for_l1_gt1"]),reverse=True)[:5]
    },sort_keys=True))

if __name__=="__main__":main()
