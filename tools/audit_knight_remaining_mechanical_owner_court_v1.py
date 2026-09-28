from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
    _mapping,
    _skin,
    _tracks_for_clip,
)

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
CLIPS = (
    ("demo_idle_v1", "idle"),
    ("demo_run_v1", "run"),
    ("demo_slash_v1", "slash"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_exact(run_root: Path, key: str) -> dict:
    rel, expected = EXACT[key]
    p = run_root / rel
    if not p.is_file():
        raise RuntimeError(f"MISSING:{rel}")
    actual = sha256(p)
    if actual != expected:
        raise RuntimeError(f"SHA_DRIFT:{rel}:{actual}:{expected}")
    return json.loads(p.read_text())


def faces_index(candidate) -> np.ndarray:
    vi = {str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)}
    return np.asarray([[vi[str(x)] for x in face] for face in candidate.faces], dtype=np.int64)


def identity_surface_ids(candidate):
    out=[]
    for v in candidate.vertices:
        coeffs=tuple(v.support_binding.coefficients)
        if len(coeffs)!=1 or abs(float(coeffs[0][1])-1.0)>1e-12:
            raise RuntimeError("NON_IDENTITY_SUPPORT_IN_EXACT_KNIGHT")
        out.append(str(coeffs[0][0]))
    return tuple(out)


def source_category(ta:int,tb:int,source_sets)->str:
    if ta==tb: return "SAME_FACE"
    common=len(source_sets[ta] & source_sets[tb])
    if common==2: return "SHARE_EDGE"
    if common==1: return "SHARE_VERTEX_ONLY"
    return "NONINCIDENT"


def teacher_face_classes(candidate, bank_path:Path, source_path:Path):
    with np.load(bank_path,allow_pickle=False) as z:
        bank_ids=tuple(map(str,z["surface_ids"].tolist()))
        tri_idx=np.asarray(z["source_triangle_index"],dtype=np.int64)
    with np.load(source_path,allow_pickle=False) as z:
        source_faces=np.asarray(z["faces"],dtype=np.int64)
    row={sid:i for i,sid in enumerate(bank_ids)}
    source_sets=[set(map(int,x)) for x in source_faces.tolist()]
    tri_by_vertex=[]
    for sid in identity_surface_ids(candidate):
        if sid not in row: raise RuntimeError(f"SURFACE_NOT_IN_BANK:{sid}")
        tri_by_vertex.append(int(tri_idx[row[sid]]))
    vi={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    classes=[]
    pair_cats=[]
    for face in candidate.faces:
        ids=[vi[str(x)] for x in face]
        tris=[tri_by_vertex[i] for i in ids]
        cats=(
            source_category(tris[0],tris[1],source_sets),
            source_category(tris[1],tris[2],source_sets),
            source_category(tris[2],tris[0],source_sets),
        )
        pair_cats.append(cats)
        classes.append("NONINCIDENT" if "NONINCIDENT" in cats else "TEACHER_LOCAL")
    return np.asarray(classes,dtype=object), pair_cats


def intrinsic_metrics(rest:np.ndarray, posed:np.ndarray, faces:np.ndarray):
    r=rest[faces]; p=posed[faces]
    re=np.stack((
        np.linalg.norm(r[:,1]-r[:,0],axis=1),
        np.linalg.norm(r[:,2]-r[:,1],axis=1),
        np.linalg.norm(r[:,0]-r[:,2],axis=1)),axis=1)
    pe=np.stack((
        np.linalg.norm(p[:,1]-p[:,0],axis=1),
        np.linalg.norm(p[:,2]-p[:,1],axis=1),
        np.linalg.norm(p[:,0]-p[:,2],axis=1)),axis=1)
    edge=pe/np.maximum(re,1e-15)
    edge_max=edge.max(axis=1)
    edge_min=edge.min(axis=1)

    r1=r[:,1]-r[:,0]; r2=r[:,2]-r[:,0]
    p1=p[:,1]-p[:,0]; p2=p[:,2]-p[:,0]
    l1=np.linalg.norm(r1,axis=1)
    u=r1/np.maximum(l1[:,None],1e-15)
    x2=np.sum(r2*u,axis=1)
    perp=r2-x2[:,None]*u
    y2=np.linalg.norm(perp,axis=1)
    valid=(l1>1e-12)&(y2>1e-12)
    inv=np.zeros((len(faces),2,2),dtype=np.float64)
    inv[:,0,0]=1.0/np.maximum(l1,1e-15)
    inv[:,0,1]=-x2/np.maximum(l1*y2,1e-15)
    inv[:,1,1]=1.0/np.maximum(y2,1e-15)
    pedges=np.stack((p1,p2),axis=2)
    F=np.einsum("nij,njk->nik",pedges,inv)
    s=np.linalg.svd(F,compute_uv=False)
    smax=s[:,0]; smin=s[:,1]
    area=smax*smin
    condition=smax/np.maximum(smin,1e-15)
    area[~valid]=np.inf; condition[~valid]=np.inf
    return {"edge_max":edge_max,"edge_min":edge_min,"area":area,"condition":condition,"valid":valid}


def quant(x,q):
    x=np.asarray(x,dtype=np.float64)
    x=x[np.isfinite(x)]
    return float(np.quantile(x,q)) if len(x) else None


def summarize(mask, m):
    mask=np.asarray(mask,dtype=bool)
    n=int(np.count_nonzero(mask))
    if not n: return {"face_count":0}
    edge=m["edge_max"][mask]; area=m["area"][mask]; cond=m["condition"][mask]
    return {
        "face_count":n,
        "edge_max_p50":quant(edge,.50),
        "edge_max_p95":quant(edge,.95),
        "edge_max_p99":quant(edge,.99),
        "edge_max_max":float(np.max(edge)),
        "area_p95":quant(area,.95),
        "area_p99":quant(area,.99),
        "area_max":float(np.max(area)),
        "condition_p95":quant(cond,.95),
        "condition_p99":quant(cond,.99),
        "condition_max":float(np.max(cond)),
        "edge_gt_2":int(np.count_nonzero(edge>2.0)),
        "edge_gt_4":int(np.count_nonzero(edge>4.0)),
        "edge_gt_10":int(np.count_nonzero(edge>10.0)),
        "condition_gt_4":int(np.count_nonzero(cond>4.0)),
        "condition_gt_16":int(np.count_nonzero(cond>16.0)),
    }


def weight_features(W:np.ndarray,faces:np.ndarray):
    wf=W[faces]
    l01=np.sum(np.abs(wf[:,0]-wf[:,1]),axis=1)
    l12=np.sum(np.abs(wf[:,1]-wf[:,2]),axis=1)
    l20=np.sum(np.abs(wf[:,2]-wf[:,0]),axis=1)
    max_l1=np.maximum(np.maximum(l01,l12),l20)
    dom=np.argmax(W,axis=1)
    dom_w=np.max(W,axis=1)
    fd=dom[faces]
    same_dom=(fd[:,0]==fd[:,1])&(fd[:,1]==fd[:,2])
    min_dom=np.min(dom_w[faces],axis=1)
    exact_onehot_vertex=np.isclose(dom_w,1.0,atol=1e-12)&(np.sum(W>1e-12,axis=1)==1)
    eoh=exact_onehot_vertex[faces]
    exact_same_onehot=same_dom & np.all(eoh,axis=1)
    # Exact identical full weight vectors across the three vertices.
    same_vec=(np.max(np.abs(wf[:,0]-wf[:,1]),axis=1)<1e-12)&(np.max(np.abs(wf[:,1]-wf[:,2]),axis=1)<1e-12)
    return {
        "max_l1":max_l1,
        "same_dominant_joint":same_dom,
        "min_dominant_weight":min_dom,
        "exact_same_onehot_joint":exact_same_onehot,
        "exact_same_weight_vector":same_vec,
        "dominant_joint_index":dom,
        "dominant_weight":dom_w,
    }


def ancestors(parent:dict[str,str|None], node:str):
    out=set(); cur=parent.get(node)
    while cur is not None and cur not in out:
        out.add(cur); cur=parent.get(cur)
    return out


def mapping_hierarchy(payload,skeleton,source_report):
    mapping,details,srows,trows,tpar=_mapping(payload,skeleton,source_report)
    spar={str(r["source_joint_id"]):(None if r.get("parent_source_joint_id") is None else str(r.get("parent_source_joint_id"))) for r in srows}
    rows=[]
    counts=Counter()
    for child,parent in sorted(tpar.items()):
        if parent is None: continue
        sp=mapping[str(parent)]; sc=mapping[str(child)]
        if sp==sc:
            rel="SAME_SOURCE"
        elif sp in ancestors(spar,sc):
            rel="SOURCE_DESCENDANT"
        elif sc in ancestors(spar,sp):
            rel="SOURCE_REVERSED_ANCESTRY"
        else:
            rel="SOURCE_UNRELATED_BRANCH"
        counts[rel]+=1
        rows.append({"target_parent":str(parent),"target_child":str(child),"source_parent_map":sp,"source_child_map":sc,"relation":rel})
    duplicates=defaultdict(list)
    for tid,sid in mapping.items(): duplicates[sid].append(tid)
    dup={sid:sorted(v) for sid,v in duplicates.items() if len(v)>1}
    return mapping,details,rows,dict(counts),dup


def matrix_rigidity(mats:dict):
    rows=[]; worst_orth=0.0; worst_det=0.0
    for jid,M in mats.items():
        A=np.asarray(M,dtype=np.float64)[:3,:3]
        orth=float(np.linalg.norm(A.T@A-np.eye(3),ord="fro"))
        det=float(np.linalg.det(A))
        det_err=abs(det-1.0)
        worst_orth=max(worst_orth,orth); worst_det=max(worst_det,det_err)
        rows.append((orth,det_err,str(jid),det))
    rows.sort(reverse=True)
    return {
        "max_rotation_orthogonality_fro_error":worst_orth,
        "max_abs_det_minus_one":worst_det,
        "worst": [
            {"joint_id":jid,"orthogonality_fro_error":o,"det":d,"abs_det_minus_one":de}
            for o,de,jid,d in rows[:12]
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--teacher-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    ctx=_ctx(a.authority_root,a.run_id)
    rr=ctx["run_root"]
    candidate=canonical_mesh_candidate_from_dict(read_exact(rr,"candidate"))
    skeleton=qualified_skeleton_from_dict(read_exact(rr,"skeleton"))
    skin=qualified_skin_from_dict(read_exact(rr,"skin"))
    cameras=tuple(sorted(qualified_camera_set_from_dict(read_exact(rr,"cameras")).cameras,key=lambda x:int(x.view_index)))
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    rest=np.asarray([v.P for v in candidate.vertices],dtype=np.float64)
    faces=faces_index(candidate)
    classes,pair_cats=teacher_face_classes(candidate,a.teacher_bank,a.teacher_source)
    local=classes=="TEACHER_LOCAL"; nonincident=~local
    joint_ids,W=_candidate_skin_weights(candidate,skin,skeleton)
    wf=weight_features(W,faces)

    component=np.asarray([str(candidate.vertices[i].component_id) for i in faces[:,0]],dtype=object)
    # Faces are required by candidate validation to be within one component; independently verify.
    comp_cross=0
    for fi,row in enumerate(faces):
        cs={str(candidate.vertices[int(i)].component_id) for i in row}
        if len(cs)!=1: comp_cross+=1

    report={
        "schema":"RealSaS.KnightRemainingMechanicalOwnerCourt.v1",
        "status":"MEASURED__NO_REPAIR",
        "exact_authority_sha256":{k:v[1] for k,v in EXACT.items()},
        "static":{
            "vertex_count":int(len(rest)),
            "face_count":int(len(faces)),
            "teacher_local_face_count":int(np.count_nonzero(local)),
            "teacher_nonincident_face_count":int(np.count_nonzero(nonincident)),
            "cross_component_face_count_measured":int(comp_cross),
            "component_face_counts":dict(Counter(component.tolist())),
            "face_weight_l1":{
                "p50":quant(wf["max_l1"],.5),"p95":quant(wf["max_l1"],.95),
                "p99":quant(wf["max_l1"],.99),"max":float(np.max(wf["max_l1"])),
                "gt_0_5":int(np.count_nonzero(wf["max_l1"]>0.5)),
                "gt_1_0":int(np.count_nonzero(wf["max_l1"]>1.0)),
                "gt_1_9":int(np.count_nonzero(wf["max_l1"]>1.9)),
            },
            "exact_same_onehot_joint_face_count":int(np.count_nonzero(wf["exact_same_onehot_joint"])),
            "exact_same_weight_vector_face_count":int(np.count_nonzero(wf["exact_same_weight_vector"])),
        },
        "clips":[],
    }

    for clip_id,short in CLIPS:
        path=rr/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json"
        payload=json.loads(path.read_text())
        mapping,details,hrows,hcounts,duplicates=mapping_hierarchy(payload,skeleton,source_report)
        tracks,_=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        frames=[]
        for frame_index,t in enumerate(times):
            mats,_,frame_hash=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,W,joint_ids,mats)
            met=intrinsic_metrics(rest,posed,faces)
            groups={
                "ALL":np.ones(len(faces),dtype=bool),
                "TEACHER_LOCAL":local,
                "TEACHER_NONINCIDENT":nonincident,
                "LOW_WEIGHT_DISCONTINUITY_L1_LE_0_1":wf["max_l1"]<=0.1,
                "HIGH_WEIGHT_DISCONTINUITY_L1_GT_1_0":wf["max_l1"]>1.0,
                "EXACT_SAME_ONEHOT_JOINT":wf["exact_same_onehot_joint"],
                "EXACT_SAME_WEIGHT_VECTOR":wf["exact_same_weight_vector"],
                "TEACHER_LOCAL_LOW_WEIGHT_L1_LE_0_1":local&(wf["max_l1"]<=0.1),
                "TEACHER_LOCAL_HIGH_WEIGHT_L1_GT_1_0":local&(wf["max_l1"]>1.0),
            }
            summaries={name:summarize(mask,met) for name,mask in groups.items()}

            # Top teacher-local offenders: if these are catastrophic, topology cannot be sole owner.
            score=np.maximum.reduce([
                met["edge_max"],
                np.sqrt(np.maximum(met["area"],1e-15)),
                np.sqrt(np.maximum(met["condition"],1e-15)),
            ])
            idx=np.nonzero(local)[0]
            idx=idx[np.argsort(score[idx])[::-1]]
            offenders=[]
            for fi in idx[:40]:
                face=faces[int(fi)]
                offenders.append({
                    "face_index":int(fi),
                    "component_id":str(component[int(fi)]),
                    "edge_max":float(met["edge_max"][fi]),
                    "area":float(met["area"][fi]),
                    "condition":float(met["condition"][fi]),
                    "max_weight_l1":float(wf["max_l1"][fi]),
                    "same_dominant_joint":bool(wf["same_dominant_joint"][fi]),
                    "min_dominant_weight":float(wf["min_dominant_weight"][fi]),
                    "exact_same_onehot_joint":bool(wf["exact_same_onehot_joint"][fi]),
                    "pair_categories":list(pair_cats[int(fi)]),
                    "vertex_ids":[str(candidate.vertices[int(i)].candidate_vertex_id) for i in face],
                    "dominant_joints":[joint_ids[int(wf["dominant_joint_index"][int(i)])] for i in face],
                })

            frames.append({
                "frame_index":int(frame_index),"time_seconds":float(t),"motion_frame_hash":frame_hash,
                "skin_matrix_rigidity":matrix_rigidity(mats),
                "groups":summaries,
                "top_teacher_local_offenders":offenders,
            })
        report["clips"].append({
            "clip_id":clip_id,
            "motion_sha256":sha256(path),
            "mapping":{
                "target_joint_count":len(mapping),
                "hierarchy_relation_counts":hcounts,
                "hierarchy_rows":hrows,
                "duplicate_source_mappings":duplicates,
                "details":details,
            },
            "frames":frames,
        })

    # Court findings are factual predicates, not a final owner verdict.
    run_clip=next(x for x in report["clips"] if x["clip_id"]=="demo_run_v1")
    local_edge_p95=max(float(f["groups"]["TEACHER_LOCAL"]["edge_max_p95"] or 0.0) for f in run_clip["frames"])
    local_edge_max=max(float(f["groups"]["TEACHER_LOCAL"]["edge_max_max"] or 0.0) for f in run_clip["frames"])
    bad_hierarchy=max(
        int(x["mapping"]["hierarchy_relation_counts"].get("SOURCE_REVERSED_ANCESTRY",0))
        + int(x["mapping"]["hierarchy_relation_counts"].get("SOURCE_UNRELATED_BRANCH",0))
        for x in report["clips"]
    )
    rigid_matrix_error=max(
        float(f["skin_matrix_rigidity"]["max_rotation_orthogonality_fro_error"])
        for x in report["clips"] for f in x["frames"]
    )
    report["measured_findings"]={
        "teacher_local_faces_still_have_large_intrinsic_stretch":bool(local_edge_p95>2.0 or local_edge_max>4.0),
        "run_teacher_local_edge_p95_max":local_edge_p95,
        "run_teacher_local_edge_max_max":local_edge_max,
        "retarget_mapping_has_hierarchy_reversal_or_cross_branch":bool(bad_hierarchy>0),
        "max_bad_target_edges":bad_hierarchy,
        "joint_skin_matrices_are_numerically_rigid":bool(rigid_matrix_error<1e-8),
        "max_skin_matrix_orthogonality_error":rigid_matrix_error,
        "claim_boundary":(
            "These measurements can exclude sole-owner hypotheses and identify concrete structural violations. "
            "They do not assign final causal percentages until ablations hold all other authorities fixed."
        ),
    }

    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_REMAINING_MECHANICAL_OWNER_COURT_PASS",json.dumps(report["measured_findings"],sort_keys=True))

if __name__=="__main__":
    main()
