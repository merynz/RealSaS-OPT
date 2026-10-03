"""Audit deformation-sensitive quadratic skin/carrier compatibility repair.

Minimizes minimum correction to the model field while regularizing carrier-edge
weight differences in a subject-free joint-motion metric derived only from the
generic G3 +/-10 degree local-frame micro-stress probes.

No teacher data, animation clips, fitting, or product authority are used.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    G3_LOCAL_MICRO_STRESS_ANGLE_DEG,
    _pose_skin_matrices,
)
from compiler.realsas_compiler_core.product_mesh_skin_v1 import _skin_support_coefficients
from compiler.realsas_compiler_core.types import QualifiedSkinIR, QualifiedSkinRow
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from tools.demo.render_knight_motion_preview_v1 import _ctx

LAMBDAS=(0.1,1.0,10.0,100.0)
LENGTH_POWERS=(1.0,2.0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")


def q(a,p):
    return float(np.quantile(np.asarray(a,dtype=np.float64),p))


def unique_edges(candidate,vertex_index):
    edges=set()
    for face in candidate.faces:
        a,b,c=(vertex_index[x] for x in face)
        edges.add(tuple(sorted((a,b))))
        edges.add(tuple(sorted((b,c))))
        edges.add(tuple(sorted((c,a))))
    return np.asarray(sorted(edges),dtype=np.int64)


def skin_matrix(surface,skeleton,skin):
    sids=tuple(str(n.surface_id) for n in surface.surface_nodes)
    jids=tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    rows={str(r.surface_id):r for r in skin.rows}
    if set(rows)!=set(sids):
        raise RuntimeError("DEFORM_REPAIR_SKIN_SURFACE_ACCOUNTING")
    ji={jid:i for i,jid in enumerate(jids)}
    W=np.zeros((len(sids),len(jids)),dtype=np.float64)
    for r,sid in enumerate(sids):
        for jid,w in rows[sid].influences:
            if jid not in ji:
                raise RuntimeError("DEFORM_REPAIR_SKIN_UNKNOWN_JOINT")
            W[r,ji[jid]]=float(w)
    if np.any(W<0.0) or not np.allclose(W.sum(1),1.0,atol=1e-8,rtol=0.0):
        raise RuntimeError("DEFORM_REPAIR_SOURCE_SKIN_NOT_SIMPLEX")
    return sids,jids,W


def carrier_projection(candidate,sids):
    si={sid:i for i,sid in enumerate(sids)}
    vertices=tuple(candidate.vertices)
    vi={v.candidate_vertex_id:i for i,v in enumerate(vertices)}
    prow=[];pcol=[];pdata=[]
    for r,v in enumerate(vertices):
        total=0.0
        for sid,c in _skin_support_coefficients(v):
            if sid not in si:
                raise RuntimeError("DEFORM_REPAIR_VERTEX_SUPPORT_UNKNOWN")
            c=float(c)
            if c<0.0 or not math.isfinite(c):
                raise RuntimeError("DEFORM_REPAIR_VERTEX_SUPPORT_INVALID")
            total+=c;prow.append(r);pcol.append(si[sid]);pdata.append(c)
        if abs(total-1.0)>1e-9:
            raise RuntimeError("DEFORM_REPAIR_VERTEX_SUPPORT_NOT_SIMPLEX")
    P=sparse.csr_matrix((pdata,(prow,pcol)),shape=(len(vertices),len(sids)),dtype=np.float64)
    edges=unique_edges(candidate,vi)
    rest=np.asarray([v.P for v in vertices],dtype=np.float64)
    span=float(np.max(rest.max(0)-rest.min(0)))
    lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)/span
    if span<=1e-12 or np.any(lengths<=1e-12):
        raise RuntimeError("DEFORM_REPAIR_CARRIER_GEOMETRY_INVALID")
    er=np.repeat(np.arange(len(edges)),2)
    ec=edges.reshape(-1)
    ed=np.tile(np.asarray([1.0,-1.0]),len(edges))
    C=sparse.csr_matrix((ed,(er,ec)),shape=(len(edges),len(vertices)),dtype=np.float64)
    return P,edges,rest,span,lengths,(C@P).tocsr()


def joint_motion_metric(skeleton,cameras,body_span):
    frames=derive_joint_frames_from_skeleton(skeleton,cameras=cameras)
    jids=tuple(j.canonical_joint_id for j in skeleton.joints)
    blocks=[]
    probe_count=0
    for active in sorted(jids):
        for axis in range(3):
            for sign in (-1.0,1.0):
                mats=_pose_skin_matrices(
                    skeleton,frames,joint_id=active,local_axis_index=axis,
                    degrees=sign*G3_LOCAL_MICRO_STRESS_ANGLE_DEG,
                )
                feat=[]
                for jid in jids:
                    M=np.asarray(mats[jid],dtype=np.float64)
                    row=np.concatenate(((M[:3,:3]-np.eye(3)).reshape(-1),M[:3,3]/body_span))
                    feat.append(row)
                block=np.asarray(feat,dtype=np.float64)
                # Common-mode rigid motion does not create a simplex weight-gradient failure.
                block-=block.mean(axis=0,keepdims=True)
                blocks.append(block)
                probe_count+=1
    F=np.concatenate(blocks,axis=1)/math.sqrt(max(probe_count,1))
    H=F@F.T
    J=len(jids)
    center=np.eye(J)-np.ones((J,J),dtype=np.float64)/float(J)
    H=center@H@center
    H=(H+H.T)*0.5
    evals=np.linalg.eigvalsh(H)
    if evals.min() < -1e-10:
        raise RuntimeError("DEFORM_REPAIR_JOINT_METRIC_NOT_PSD")
    diag=np.diag(H)
    dist=diag[:,None]+diag[None,:]-2.0*H
    positive=dist[np.triu_indices(J,1)]
    positive=positive[positive>1e-14]
    scale=float(np.quantile(positive,0.95)) if len(positive) else 1.0
    if not math.isfinite(scale) or scale<=1e-14:
        raise RuntimeError("DEFORM_REPAIR_JOINT_METRIC_DEGENERATE")
    H/=scale
    H=(H+H.T)*0.5
    evals,U=np.linalg.eigh(H)
    evals=np.maximum(evals,0.0)
    return H,evals,U,{
        "probe_count":probe_count,
        "g3_angle_deg":G3_LOCAL_MICRO_STRESS_ANGLE_DEG,
        "pairwise_distance_p95_normalizer":scale,
        "eigenvalue_min":float(evals.min()),
        "eigenvalue_p50":q(evals,.5),
        "eigenvalue_max":float(evals.max()),
        "effective_rank_gt_1e8":int(np.count_nonzero(evals>1e-8)),
    }


def solve_mode_system(K,b,scale):
    if scale<=1e-14:
        return b.copy(),0
    A=sparse.eye(K.shape[0],format="csr",dtype=np.float64)+float(scale)*K
    d=np.asarray(A.diagonal(),dtype=np.float64)
    if np.any(d<=0.0) or not np.isfinite(d).all():
        raise RuntimeError("DEFORM_REPAIR_LINEAR_SYSTEM_DIAGONAL_INVALID")
    M=LinearOperator(A.shape,matvec=lambda x:x/d,dtype=np.float64)
    x,info=cg(A,b,rtol=1e-9,atol=1e-11,maxiter=1500,M=M)
    if info!=0 or not np.isfinite(x).all():
        raise RuntimeError(f"DEFORM_REPAIR_CG_FAIL:{info}")
    residual=float(np.linalg.norm(A@x-b)/max(np.linalg.norm(b),1e-12))
    if residual>1e-7:
        raise RuntimeError(f"DEFORM_REPAIR_CG_RESIDUAL:{residual}")
    return x,residual


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--surface-json",type=Path,required=True)
    ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--fresh-inference-dir",type=Path,required=True)
    ap.add_argument("--out-root",type=Path,required=True)
    a=ap.parse_args()

    ctx=_ctx(a.authority_root,a.run_id)
    cameraset=qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    cameras=tuple(sorted(cameraset.cameras,key=lambda c:int(c.view_index)))
    surface=rigging_surface_from_dict(read(a.surface_json))
    candidate=canonical_mesh_candidate_from_dict(read(a.candidate_json))
    skeleton=qualified_skeleton_from_dict(read(a.fresh_inference_dir/"fresh_qualified_skeleton.json"))
    skin=qualified_skin_from_dict(read(a.fresh_inference_dir/"fresh_qualified_skin.json"))

    sids,jids,W0=skin_matrix(surface,skeleton,skin)
    P,edges,rest,span,lengths,Q=carrier_projection(candidate,sids)
    H,heig,U,hreport=joint_motion_metric(skeleton,cameras,span)
    base_vertex=np.asarray(P@W0,dtype=np.float64)
    base_diff=base_vertex[edges[:,0]]-base_vertex[edges[:,1]]
    base_l1=np.abs(base_diff).sum(1)
    base_motion_energy=np.einsum("ei,ij,ej->e",base_diff,H,base_diff,optimize=True)

    manifest={
        "schema":"RealSaS.DeformationSensitiveSkinProjectionAudit.v1",
        "status":"DIAGNOSTIC_ONLY",
        "product_authority_minted":False,
        "training_used":False,
        "teacher_data_used":False,
        "motion_clip_data_used":False,
        "joint_motion_metric":"GENERIC_G3_LOCAL_FRAME_MICRO_STRESS_TRANSFORM_SIGNATURE",
        "objective":"L2_TO_MODEL_PLUS_CARRIER_EDGE_DIRICHLET_IN_G3_JOINT_MOTION_METRIC",
        "surface_node_count":len(sids),
        "carrier_vertex_count":len(candidate.vertices),
        "carrier_edge_count":len(edges),
        "joint_count":len(jids),
        "body_span":span,
        "source_skin_lineage_hash":skin.skin_lineage_hash,
        "joint_metric":hreport,
        "baseline":{
            "carrier_weight_gradient_p99":q(base_l1/lengths,.99),
            "carrier_weight_gradient_max":float(np.max(base_l1/lengths)),
            "motion_metric_edge_energy_p99":q(base_motion_energy,.99),
            "motion_metric_edge_energy_max":float(base_motion_energy.max()),
        },
        "variants":[],
    }
    a.out_root.mkdir(parents=True,exist_ok=True)

    B=W0@U
    for power in LENGTH_POWERS:
        median=float(np.median(lengths))
        edge_weight=np.power(median/lengths,float(power))
        D=sparse.diags(edge_weight,format="csr")
        K=(Q.T@D@Q).tocsr()
        kd=np.asarray(K.diagonal(),dtype=np.float64)
        positive=kd[kd>1e-14]
        knorm=float(np.median(positive)) if len(positive) else 1.0
        if not math.isfinite(knorm) or knorm<=1e-14:
            raise RuntimeError("DEFORM_REPAIR_K_NORMALIZER_INVALID")
        K=K/knorm

        for lam in LAMBDAS:
            Z=np.empty_like(B)
            max_residual=0.0
            for k,h in enumerate(heig):
                Z[:,k],residual=solve_mode_system(K,B[:,k],float(lam)*float(h))
                max_residual=max(max_residual,float(residual))
            W=Z@U.T
            if not np.isfinite(W).all():
                raise RuntimeError("DEFORM_REPAIR_OUTPUT_NONFINITE")
            negative_mass=float(-np.minimum(W,0.0).sum())
            W=np.maximum(W,0.0)
            mass=W.sum(1,keepdims=True)
            if np.any(mass<=1e-12):
                raise RuntimeError("DEFORM_REPAIR_ZERO_ROW")
            W/=mass
            correction=np.abs(W-W0).sum(1)
            vertex=np.asarray(P@W,dtype=np.float64)
            diff=vertex[edges[:,0]]-vertex[edges[:,1]]
            l1=np.abs(diff).sum(1)
            motion_energy=np.einsum("ei,ij,ej->e",diff,H,diff,optimize=True)

            tag=f"p{int(power)}_lambda_{str(lam).replace('.','p')}"
            out=a.out_root/tag
            out.mkdir(parents=True,exist_ok=True)
            qrows=tuple(QualifiedSkinRow(
                surface_id=sid,
                influences=tuple((jid,float(W[r,c])) for c,jid in enumerate(jids)),
                simplex_residual_before=0.0,
                correction_l1=float(correction[r]),
            ) for r,sid in enumerate(sids))
            report={
                "status":"DIAGNOSTIC_DEFORMATION_SENSITIVE_COMPATIBILITY_PROJECTION",
                "product_authority_minted":False,
                "training_used":False,
                "teacher_data_used":False,
                "motion_clip_data_used":False,
                "source_skin_lineage_hash":skin.skin_lineage_hash,
                "lambda":float(lam),
                "length_power":float(power),
                "K_diagonal_median_normalizer":knorm,
                "negative_mass_clipped":negative_mass,
                "cg_max_relative_residual":max_residual,
                "mean_row_correction_l1":float(np.mean(correction)),
                "p95_row_correction_l1":q(correction,.95),
                "max_row_correction_l1":float(np.max(correction)),
                "carrier_weight_gradient_p95":q(l1/lengths,.95),
                "carrier_weight_gradient_p99":q(l1/lengths,.99),
                "carrier_weight_gradient_max":float(np.max(l1/lengths)),
                "motion_metric_edge_energy_p95":q(motion_energy,.95),
                "motion_metric_edge_energy_p99":q(motion_energy,.99),
                "motion_metric_edge_energy_max":float(np.max(motion_energy)),
                "joint_metric":hreport,
            }
            provisional=QualifiedSkinIR(
                qrows,surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,report,"")
            payload=provisional.to_dict();payload.pop("skin_lineage_hash",None)
            qskin=QualifiedSkinIR(
                qrows,surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,
                report,content_sha256(payload))
            write(out/"fresh_qualified_skin.json",qskin.to_dict())
            shutil.copy2(a.fresh_inference_dir/"fresh_qualified_skeleton.json",out/"fresh_qualified_skeleton.json")
            shutil.copy2(a.fresh_inference_dir/"rig_REPORT.json",out/"rig_REPORT.json")
            write(out/"skin_REPORT.json",report)
            write(out/"PROJECTION_STATS.json",{**report,"skin_lineage_hash":qskin.skin_lineage_hash})
            manifest["variants"].append({"tag":tag,**report,"skin_lineage_hash":qskin.skin_lineage_hash})
            print("DEFORM_SENSITIVE_PREP="+json.dumps({
                "tag":tag,"p95_corr":report["p95_row_correction_l1"],
                "grad_p99":report["carrier_weight_gradient_p99"],
                "motion_energy_p99":report["motion_metric_edge_energy_p99"],
                "negative_mass":negative_mass,
            },sort_keys=True),flush=True)
    write(a.out_root/"PROJECTION_MANIFEST.json",manifest)


if __name__=="__main__":
    main()
