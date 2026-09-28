from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx

EXACT={
 "candidate":("artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json","0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3"),
 "skeleton":("artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json","e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987"),
 "skin":("artifacts/32_SKIN_QUALIFIED/qualified_skin.json","f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09"),
}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(rr,key,codec):
    rel,h=EXACT[key]; p=rr/rel
    if sha(p)!=h: raise RuntimeError("SHA_DRIFT:"+key)
    return codec(json.loads(p.read_text()))

def q(x,p):
    x=np.asarray(x,dtype=np.float64)
    return float(np.quantile(x,p)) if x.size else None

def pairmax(W, faces):
    x=W[faces]
    return np.maximum.reduce([
        np.sum(np.abs(x[:,0]-x[:,1]),axis=1),
        np.sum(np.abs(x[:,1]-x[:,2]),axis=1),
        np.sum(np.abs(x[:,2]-x[:,0]),axis=1),
    ])

def src_cat(a,b,sets):
    if a==b:return "SAME_FACE"
    c=len(sets[a]&sets[b])
    if c==2:return "SHARE_EDGE"
    if c==1:return "SHARE_VERTEX_ONLY"
    return "NONINCIDENT"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=load(rr,"candidate",canonical_mesh_candidate_from_dict)
    skel=load(rr,"skeleton",qualified_skeleton_from_dict)
    pred=load(rr,"skin",qualified_skin_from_dict)

    with np.load(a.bank,allow_pickle=False) as z:
        sids=tuple(map(str,z["surface_ids"].tolist()))
        bankW=np.asarray(z["weights"],dtype=np.float64)
        tri=np.asarray(z["source_triangle_index"],dtype=np.int64)
        bary=np.asarray(z["source_triangle_barycentric"],dtype=np.float64)
        dist=np.asarray(z["source_surface_distance"],dtype=np.float64)
        tpos=np.asarray(z["target_positions_world"],dtype=np.float64)
        tpar=np.asarray(z["target_parent_indices"],dtype=np.int64)
        tsource=np.asarray(z["target_source_indices_provenance_only"],dtype=np.int64)
        valid=np.asarray(z["teacher_valid_mask"],dtype=np.uint8).astype(bool)
    with np.load(a.source,allow_pickle=False) as z:
        sf=np.asarray(z["faces"],dtype=np.int64)
        sskin=np.asarray(z["skin"],dtype=np.float64)
        sv=np.asarray(z["vertices_source"],dtype=np.float64)

    # Independently align bank target columns to Stage28 joints by world positions.
    joints=tuple(skel.joints)
    spos=np.asarray([j.position for j in joints],dtype=np.float64)
    C=np.linalg.norm(tpos[:,None,:]-spos[None,:,:],axis=2)
    ri,ci=linear_sum_assignment(C)
    assign=np.empty(len(tpos),dtype=np.int64); assign[ri]=ci
    errors=C[ri,ci]
    if len(set(assign.tolist()))!=len(assign): raise RuntimeError("TARGET_ASSIGN_NOT_BIJECTIVE")
    joint_ids=tuple(str(j.canonical_joint_id) for j in joints)

    # Predicted Stage32 matrix in bank column order after measured positional alignment.
    pred_by={str(r.surface_id):dict(r.influences) for r in pred.rows}
    PW=np.zeros_like(bankW)
    for bi,sid in enumerate(sids):
        row=pred_by.get(sid)
        if row is None: raise RuntimeError("PRED_ROW_MISSING:"+sid)
        for bank_col,target_joint_index in enumerate(assign):
            PW[bi,bank_col]=float(row.get(joint_ids[int(target_joint_index)],0.0))
    pred_err=np.sum(np.abs(PW-bankW),axis=1)

    # Artist-source skin interpolated directly at the bank's own triangle+bary provenance.
    tri_vertices=sf[tri]
    source_projected_W=np.einsum("ni,nij->nj",bary,sskin[tri_vertices])

    # Independent geometric reconstruction error of provenance.
    srcP=np.einsum("ni,nij->nj",bary,sv[tri_vertices])

    bank_index={sid:i for i,sid in enumerate(sids)}
    vids={str(v.candidate_vertex_id):i for i,v in enumerate(cand.vertices)}
    csids=[]
    cbank=[]
    positions=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    for v in cand.vertices:
        coeff=tuple(v.support_binding.coefficients)
        if len(coeff)!=1 or abs(float(coeff[0][1])-1)>1e-12: raise RuntimeError("NONIDENTITY_SUPPORT")
        sid=str(coeff[0][0]); csids.append(sid); cbank.append(bank_index[sid])
    cbank=np.asarray(cbank,dtype=np.int64)
    faces=np.asarray([[vids[str(x)] for x in face] for face in cand.faces],dtype=np.int64)

    bank_face_l1=pairmax(bankW[cbank],faces)
    source_face_l1=pairmax(source_projected_W[cbank],faces)
    pred_face_l1=pairmax(PW[cbank],faces)

    source_sets=[set(map(int,x)) for x in sf.tolist()]
    cats=[]; strict=[]
    for face in faces:
        tis=[int(tri[cbank[int(v)]]) for v in face]
        pc=(src_cat(tis[0],tis[1],source_sets),src_cat(tis[1],tis[2],source_sets),src_cat(tis[2],tis[0],source_sets))
        cats.append(pc)
        strict.append(all(x in ("SAME_FACE","SHARE_EDGE") for x in pc))
    strict=np.asarray(strict,dtype=bool)
    noninc=np.asarray(["NONINCIDENT" in x for x in cats],dtype=bool)
    share_vertex_only=(~noninc)&(~strict)

    face_dist=np.max(dist[cbank[faces]],axis=1)
    # Compare bank's recorded distance to direct candidate-to-provenance reconstructed point distance.
    direct_dist=np.linalg.norm(positions-srcP[cbank],axis=1)
    dist_err=np.abs(direct_dist-dist[cbank])

    def group(mask):
        mask=np.asarray(mask,dtype=bool)
        return {
          "count":int(np.count_nonzero(mask)),
          "bank_target_weight_l1_p95":q(bank_face_l1[mask],.95),
          "bank_target_weight_l1_max":float(np.max(bank_face_l1[mask])) if np.any(mask) else None,
          "source_artist_41d_weight_l1_p95":q(source_face_l1[mask],.95),
          "source_artist_41d_weight_l1_max":float(np.max(source_face_l1[mask])) if np.any(mask) else None,
          "predicted_stage32_weight_l1_p95":q(pred_face_l1[mask],.95),
          "source_distance_p95":q(face_dist[mask],.95),
          "source_distance_max":float(np.max(face_dist[mask])) if np.any(mask) else None,
          "bank_l1_gt_1":int(np.count_nonzero(mask&(bank_face_l1>1.0))),
          "source_l1_gt_1":int(np.count_nonzero(mask&(source_face_l1>1.0))),
        }

    high=bank_face_l1>1.0
    rows=[]
    idx=np.argsort(bank_face_l1)[::-1]
    for fi in idx[:200]:
        f=faces[int(fi)]
        rows.append({
          "face_index":int(fi),
          "pair_categories":list(cats[int(fi)]),
          "strict_source_local":bool(strict[fi]),
          "bank_target_weight_l1":float(bank_face_l1[fi]),
          "source_artist_41d_weight_l1":float(source_face_l1[fi]),
          "stage32_pred_weight_l1":float(pred_face_l1[fi]),
          "max_source_surface_distance":float(face_dist[fi]),
          "vertex_surface_ids":[csids[int(v)] for v in f],
          "source_triangle_indices":[int(tri[cbank[int(v)]]) for v in f],
        })

    report={
      "schema":"RealSaS.KnightTeacherProjectionIntegrityAudit.v1",
      "status":"MEASURED__NO_REPAIR",
      "target_column_alignment":{
        "method":"Hungarian minimum world-position distance; measured, not assumed",
        "max_position_error":float(np.max(errors)),
        "p95_position_error":q(errors,.95),
        "mapping":[{"bank_col":int(i),"stage28_joint_id":joint_ids[int(assign[i])],"position_error":float(C[i,assign[i]]),"source_index_provenance_only":int(tsource[i]),"parent_bank_col":int(tpar[i])} for i in range(len(assign))]
      },
      "stage32_vs_teacher_bank":{
        "surface_row_count":len(sids),
        "l1_mean":float(np.mean(pred_err)),
        "l1_p95":q(pred_err,.95),
        "l1_p99":q(pred_err,.99),
        "l1_max":float(np.max(pred_err)),
        "rows_gt_0_01":int(np.count_nonzero(pred_err>0.01)),
        "rows_gt_0_1":int(np.count_nonzero(pred_err>0.1)),
      },
      "provenance_geometry_check":{
        "candidate_to_reconstructed_source_distance_p50":q(direct_dist,.5),
        "p95":q(direct_dist,.95),"p99":q(direct_dist,.99),"max":float(np.max(direct_dist)),
        "recorded_vs_direct_abs_error_max":float(np.max(dist_err)),
        "recorded_vs_direct_abs_error_p95":q(dist_err,.95),
        "teacher_valid_count":int(np.count_nonzero(valid)),
      },
      "face_groups":{
        "ALL":group(np.ones(len(faces),dtype=bool)),
        "STRICT_SOURCE_LOCAL_SAME_OR_SHARE_EDGE":group(strict),
        "SHARE_VERTEX_ONLY_BUT_NOT_NONINCIDENT":group(share_vertex_only),
        "NONINCIDENT":group(noninc),
        "BANK_TARGET_L1_GT_1":group(high),
        "BANK_TARGET_L1_GT_1_AND_STRICT_LOCAL":group(high&strict),
        "BANK_TARGET_L1_GT_1_AND_DISTANCE_GT_0_05":group(high&(face_dist>0.05)),
        "BANK_TARGET_L1_GT_1_AND_DISTANCE_LE_0_01":group(high&(face_dist<=0.01)),
      },
      "correlations":{
        "face_bank_l1_vs_source_distance":float(np.corrcoef(bank_face_l1,face_dist)[0,1]),
        "face_bank_l1_vs_source_artist_l1":float(np.corrcoef(bank_face_l1,source_face_l1)[0,1]),
      },
      "top_bank_discontinuity_faces":rows,
      "measured_findings":{
        "stage32_reproduces_bank_pointwise":bool(q(pred_err,.99)<0.05),
        "bank_has_large_face_discontinuities":bool(np.count_nonzero(bank_face_l1>1.0)>0),
        "artist_source_projection_has_large_discontinuities_same_locations":bool(np.count_nonzero(source_face_l1>1.0)>0),
        "strict_local_bank_discontinuity_count":int(np.count_nonzero(strict&(bank_face_l1>1.0))),
        "strict_local_artist_source_discontinuity_count":int(np.count_nonzero(strict&(source_face_l1>1.0))),
      }
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TEACHER_PROJECTION_INTEGRITY_AUDIT_PASS",json.dumps(report["measured_findings"],sort_keys=True))
if __name__=="__main__":main()
