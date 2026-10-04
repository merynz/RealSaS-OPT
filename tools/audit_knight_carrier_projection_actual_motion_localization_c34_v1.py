from __future__ import annotations

"""C3.4 exact-motion owner localization on the current carrier.

Replays the exact C3.3 architecture arm:
- frozen MIRA semantic field in GSA domain,
- deterministic carrier-bound mechanical support projection,
- exact Stage19 carrier,
- A2-verified Stage28 rig,
- 17 samples for each idle/run/slash clip (51 frames).

Unlike C3.3, this court retains per-face dynamic metrics and classifies every
actual-motion offender by carrier support domain. It does not mutate rig, skin,
carrier, motion or proof policy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    _triangle_metrics_batch_exact,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    static_mesh_qualification_from_dict,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)
from models.mira.carrier_query_v1 import build_mira_mechanical_carrier_query_v1
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import (
    CLIPS,
    FULL_MOTION_SAMPLES,
)
from tools.audit_knight_carrier_first_c3_v1 import (
    _decode_legacy,
    _run_mira_backbone,
)
from tools.audit_knight_mira_support_mixture_c32_v1 import (
    _legacy_transfer_ceiling,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _ctx,
    _skin,
    _tracks_for_clip,
)
from tools.inference.refined_surface_rig_skin_v1 import write


CLASS_NAMES=(
    "ALL_IDENTITY",
    "ONE_NON_IDENTITY",
    "TWO_PLUS_NON_IDENTITY",
)


def _faces(candidate, ordered_vertex_ids):
    row={str(v):i for i,v in enumerate(ordered_vertex_ids)}
    out=[]
    for face in candidate.faces:
        ids=tuple(map(str,face))
        if len(ids)!=3 or len(set(ids))!=3 or any(x not in row for x in ids):
            raise RuntimeError("C34_FACE_AXIS_DRIFT")
        out.append(tuple(row[x] for x in ids))
    return np.asarray(out,np.int64)


def _vertex_support_class(candidate, ordered_vertex_ids):
    by={str(v.candidate_vertex_id):v for v in candidate.vertices}
    non=np.zeros(len(ordered_vertex_ids),bool)
    modes=[]
    support_count=np.zeros(len(ordered_vertex_ids),np.int64)
    for i,vid in enumerate(ordered_vertex_ids):
        v=by[str(vid)]
        coeffs=tuple(v.support_binding.coefficients)
        identity=(
            str(v.support_binding.mode)=="IDENTITY_SURFACE_NODE"
            and len(coeffs)==1
            and abs(float(coeffs[0][1])-1.0)<=1e-12
        )
        non[i]=not identity
        modes.append(str(v.support_binding.mode))
        support_count[i]=len(coeffs)
    return non,tuple(modes),support_count


def _face_classes(faces,non_identity):
    count=np.sum(non_identity[faces],axis=1)
    labels=np.empty(len(faces),dtype=object)
    labels[count==0]="ALL_IDENTITY"
    labels[count==1]="ONE_NON_IDENTITY"
    labels[count>=2]="TWO_PLUS_NON_IDENTITY"
    return labels,count


def _weight_l1(weights,faces):
    w=np.asarray(weights,np.float64)[faces]
    return np.maximum.reduce((
        np.abs(w[:,0]-w[:,1]).sum(axis=1),
        np.abs(w[:,1]-w[:,2]).sum(axis=1),
        np.abs(w[:,2]-w[:,0]).sum(axis=1),
    ))


def _rest_quality(rest,faces):
    tri=np.asarray(rest,np.float64)[faces]
    e=np.stack((
        np.linalg.norm(tri[:,1]-tri[:,0],axis=1),
        np.linalg.norm(tri[:,2]-tri[:,1],axis=1),
        np.linalg.norm(tri[:,0]-tri[:,2],axis=1),
    ),axis=1)
    longest=np.max(e,axis=1)
    shortest=np.min(e,axis=1)
    cross=np.linalg.norm(
        np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    min_alt=cross/np.maximum(longest,1e-15)
    aspect=longest/np.maximum(min_alt,1e-15)
    s=np.sort(e,axis=1)
    a,b,c=s[:,0],s[:,1],s[:,2]
    cosang=np.clip(
        (b*b+c*c-a*a)/np.maximum(2.0*b*c,1e-30),-1.0,1.0)
    min_angle=np.degrees(np.arccos(cosang))
    return {
        "shortest_edge":shortest,
        "longest_edge":longest,
        "aspect":aspect,
        "min_angle_deg":min_angle,
    }


def _class_summary(
    *,
    labels,
    union_bad,
    union_edge4,
    union_edge10,
    worst_condition,
    worst_edge,
    min_area,
    max_area,
    weight_l1,
    rest_quality,
):
    out={}
    for name in CLASS_NAMES:
        mask=labels==name
        bad=mask & union_bad
        def q(x,p):
            arr=np.asarray(x,np.float64)
            return float(np.quantile(arr,p)) if len(arr) else None
        out[name]={
            "face_count":int(np.count_nonzero(mask)),
            "actual_bad_face_count":int(np.count_nonzero(bad)),
            "actual_bad_rate":float(np.count_nonzero(bad)/max(np.count_nonzero(mask),1)),
            "edge_gt4_union_count":int(np.count_nonzero(mask & union_edge4)),
            "edge_gt10_union_count":int(np.count_nonzero(mask & union_edge10)),
            "worst_condition_max":(
                float(np.max(worst_condition[mask])) if np.any(mask) else None),
            "worst_edge_max":(
                float(np.max(worst_edge[mask])) if np.any(mask) else None),
            "minimum_area_ratio_min":(
                float(np.min(min_area[mask])) if np.any(mask) else None),
            "maximum_area_ratio_max":(
                float(np.max(max_area[mask])) if np.any(mask) else None),
            "weight_pair_l1_p95":q(weight_l1[mask],0.95),
            "bad_weight_pair_l1_mean":(
                float(np.mean(weight_l1[bad])) if np.any(bad) else None),
            "bad_weight_pair_l1_p95":q(weight_l1[bad],0.95),
            "rest_aspect_p95":q(rest_quality["aspect"][mask],0.95),
            "bad_rest_aspect_p95":q(rest_quality["aspect"][bad],0.95),
            "rest_min_angle_p05":q(rest_quality["min_angle_deg"][mask],0.05),
            "bad_rest_min_angle_p05":q(rest_quality["min_angle_deg"][bad],0.05),
        }
    return out


def main(args):
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("C34_CUDA_NOT_AVAILABLE")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

    ctx=_ctx(args.authority_root,args.run_id)
    rr=ctx["run_root"]
    surface=rigging_surface_from_dict(stage_output_payload(
        ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    skeleton=qualified_skeleton_from_dict(stage_output_payload(
        ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    addressing=surface_addressing_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.SurfaceAddressingIR.v1"))
    static=static_mesh_qualification_from_dict(stage_output_payload(
        ctx,"19_STATIC_CANONICAL_MESH_QUALIFIED","RealSaS.StaticCanonicalMeshQualificationIR.v1"))
    carrier=build_mechanical_carrier_evidence_v1(
        candidate,static_qualification=static,surface_addressing=addressing)
    policy=mesh_policy_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1"))
    camera_set=qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    cameras=tuple(sorted(camera_set.cameras,key=lambda x:int(x.view_index)))
    surface_tensor=tensorize_rigging_surface_v1(surface)

    conditioning,ci,memory,tokens,decoder,mira_result,telemetry=_run_mira_backbone(
        surface=surface,skeleton=skeleton,arm_dir=args.arm_dir,
        device=args.device,query_chunk=args.backbone_query_chunk)
    legacy=_decode_legacy(
        conditioning=conditioning,ci=ci,memory=memory,tokens=tokens,
        decoder=decoder,device=args.device,chunk=args.readout_chunk)
    query=build_mira_mechanical_carrier_query_v1(
        candidate=candidate,carrier_evidence=carrier,
        surface_tensor=surface_tensor,conditioning=conditioning)
    weights=_legacy_transfer_ceiling(query=query,legacy_weights=legacy)

    rest=np.asarray(carrier.positions,np.float64)
    faces=_faces(candidate,query.carrier_vertex_ids)
    non_identity,modes,support_count=_vertex_support_class(
        candidate,query.carrier_vertex_ids)
    labels,non_identity_per_face=_face_classes(faces,non_identity)
    weight_l1=_weight_l1(weights,faces)
    rest_quality=_rest_quality(rest,faces)

    F=len(faces)
    union_bad=np.zeros(F,bool)
    union_edge4=np.zeros(F,bool)
    union_edge10=np.zeros(F,bool)
    worst_condition=np.ones(F,np.float64)
    worst_edge=np.ones(F,np.float64)
    min_area=np.ones(F,np.float64)
    max_area=np.ones(F,np.float64)
    worst_clip=np.full(F,"REST",dtype=object)
    worst_frame=np.full(F,-1,np.int64)
    worst_time=np.zeros(F,np.float64)
    bad_frame_hits=np.zeros(F,np.int64)

    source_report=json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    frame_rows=[]
    mapping_rows={}
    for clip in CLIPS:
        payload=json.loads(
            (rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json").read_text())
        tracks,mapping=_tracks_for_clip(payload,skeleton,cameras,source_report)
        mapping_rows[clip]=mapping
        times=np.linspace(
            0.0,float(payload["duration_seconds"]),FULL_MOTION_SAMPLES,
            endpoint=not bool(payload.get("loop")))
        for frame_index,t in enumerate(times):
            mats,_,_=_joint_pose_v2(
                skeleton=skeleton,tracks=tracks,
                time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,weights,query.joint_ids,mats)
            area,condition,edge_min,edge_max,smin=_triangle_metrics_batch_exact(
                rest,posed,faces)
            finite=(
                np.isfinite(area)&np.isfinite(condition)
                &np.isfinite(edge_min)&np.isfinite(edge_max)&np.isfinite(smin))
            bad=(
                (~finite)
                |(area<float(policy.g3_min_dynamic_area_ratio))
                |(area>float(policy.g3_max_dynamic_area_ratio))
                |(condition>float(policy.g3_max_dynamic_condition_number))
            )
            edge4=edge_max>4.0
            edge10=edge_max>10.0
            union_bad|=bad
            union_edge4|=edge4
            union_edge10|=edge10
            bad_frame_hits+=bad.astype(np.int64)

            improve=condition>worst_condition
            worst_condition=np.maximum(worst_condition,condition)
            worst_edge=np.maximum(worst_edge,edge_max)
            min_area=np.minimum(min_area,area)
            max_area=np.maximum(max_area,area)
            for fi in np.flatnonzero(improve):
                worst_clip[fi]=clip
                worst_frame[fi]=int(frame_index)
                worst_time[fi]=float(t)

            by_class={}
            for name in CLASS_NAMES:
                mask=labels==name
                b=bad&mask
                by_class[name]={
                    "face_count":int(np.count_nonzero(mask)),
                    "bad_face_count":int(np.count_nonzero(b)),
                    "edge_gt4_count":int(np.count_nonzero(edge4&mask)),
                    "edge_gt10_count":int(np.count_nonzero(edge10&mask)),
                    "maximum_condition":(
                        float(np.max(condition[mask])) if np.any(mask) else None),
                    "maximum_edge_ratio":(
                        float(np.max(edge_max[mask])) if np.any(mask) else None),
                    "minimum_area_ratio":(
                        float(np.min(area[mask])) if np.any(mask) else None),
                    "maximum_area_ratio":(
                        float(np.max(area[mask])) if np.any(mask) else None),
                }
            frame_rows.append({
                "clip_id":clip,
                "frame_index":int(frame_index),
                "time_seconds":float(t),
                "bad_face_count":int(np.count_nonzero(bad)),
                "edge_gt4_count":int(np.count_nonzero(edge4)),
                "edge_gt10_count":int(np.count_nonzero(edge10)),
                "maximum_condition":float(np.max(condition)),
                "maximum_edge_ratio":float(np.max(edge_max)),
                "minimum_area_ratio":float(np.min(area)),
                "maximum_area_ratio":float(np.max(area)),
                "by_support_class":by_class,
            })

    class_summary=_class_summary(
        labels=labels,union_bad=union_bad,
        union_edge4=union_edge4,union_edge10=union_edge10,
        worst_condition=worst_condition,worst_edge=worst_edge,
        min_area=min_area,max_area=max_area,
        weight_l1=weight_l1,rest_quality=rest_quality)

    bad_ids=np.flatnonzero(union_bad)
    top=sorted(
        map(int,bad_ids),
        key=lambda i:(-float(worst_condition[i]),-float(worst_edge[i]),i),
    )[:200]
    top_rows=[{
        "face_index":int(fi),
        "support_class":str(labels[fi]),
        "non_identity_vertex_count":int(non_identity_per_face[fi]),
        "vertex_ids":[str(query.carrier_vertex_ids[int(v)]) for v in faces[fi]],
        "vertex_support_modes":[modes[int(v)] for v in faces[fi]],
        "vertex_support_cardinality":[int(support_count[int(v)]) for v in faces[fi]],
        "weight_pair_l1":float(weight_l1[fi]),
        "bad_frame_hits":int(bad_frame_hits[fi]),
        "worst_condition":float(worst_condition[fi]),
        "worst_edge_ratio":float(worst_edge[fi]),
        "minimum_area_ratio":float(min_area[fi]),
        "maximum_area_ratio":float(max_area[fi]),
        "rest_aspect":float(rest_quality["aspect"][fi]),
        "rest_min_angle_deg":float(rest_quality["min_angle_deg"][fi]),
        "worst_condition_clip":str(worst_clip[fi]),
        "worst_condition_frame":int(worst_frame[fi]),
        "worst_condition_time_seconds":float(worst_time[fi]),
    } for fi in top]

    identity_bad=int(np.count_nonzero(union_bad&(labels=="ALL_IDENTITY")))
    total_bad=int(np.count_nonzero(union_bad))
    nonidentity_bad=total_bad-identity_bad
    if total_bad==0:
        diagnosis="ACTUAL_MOTION_CLOSED"
    elif identity_bad==0:
        diagnosis="COMPILED_NON_IDENTITY_CARRIER_DOMAIN_OWNER"
    elif identity_bad/total_bad>=0.70:
        diagnosis="IDENTITY_DOMAIN_FAILURE__RIG_MOTION_OR_BASE_SKIN_OWNER"
    else:
        diagnosis="MIXED_IDENTITY_AND_COMPILED_DOMAIN_FAILURE"

    report={
        "schema":"RealSaS.KnightCarrierProjectionActualMotionLocalization.v1",
        "status":"COMPLETE__ATTRIBUTION_ONLY__NO_PRODUCT_MUTATION",
        "run_id":args.run_id,
        "carrier_evidence_hash":carrier.carrier_evidence_hash,
        "carrier_topology_hash":carrier.topology_hash,
        "mira_checkpoint_sha256":mira_result["model_sha256"],
        "query_hash":query.query_hash,
        "frame_count":len(frame_rows),
        "clip_count":len(CLIPS),
        "full_motion_samples_per_clip":FULL_MOTION_SAMPLES,
        "actual_motion":{
            "failed_frame_count":int(sum(row["bad_face_count"]>0 for row in frame_rows)),
            "unique_bad_face_count":total_bad,
            "unique_identity_bad_face_count":identity_bad,
            "unique_nonidentity_touching_bad_face_count":nonidentity_bad,
            "identity_fraction_of_unique_bad":float(identity_bad/max(total_bad,1)),
            "unique_edge_gt4_face_count":int(np.count_nonzero(union_edge4)),
            "unique_edge_gt10_face_count":int(np.count_nonzero(union_edge10)),
            "maximum_condition":float(np.max(worst_condition)),
            "maximum_edge_ratio":float(np.max(worst_edge)),
            "minimum_area_ratio":float(np.min(min_area)),
            "maximum_area_ratio":float(np.max(max_area)),
        },
        "by_support_class":class_summary,
        "frame_rows":frame_rows,
        "retarget_mapping":mapping_rows,
        "top_bad_faces":top_rows,
        "diagnosis":diagnosis,
        "training_used":False,
        "teacher_weights_used":False,
        "product_authority_minted":False,
        "claim_boundary":[
            "This is the exact C3.3 projected frozen MIRA field on the exact current carrier.",
            "All 51 idle/run/slash samples are replayed with the current demo retarget mapping.",
            "Face classification uses carrier support identity only and does not use teacher topology.",
            "The court attributes actual-motion failures; it does not repair or promote any owner.",
        ],
    }
    write(args.out_dir/"REPORT.json",report)
    print("MIRA_C34_RESULT="+json.dumps({
        "status":report["status"],
        "diagnosis":diagnosis,
        "actual_motion":report["actual_motion"],
        "by_support_class":class_summary,
        "top_bad_faces":top_rows[:12],
    },sort_keys=True),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--arm-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--backbone-query-chunk",type=int,default=256)
    ap.add_argument("--readout-chunk",type=int,default=128)
    args=ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True,exist_ok=True)
        write(args.out_dir/"ERROR.json",{
            "type":type(exc).__name__,"message":str(exc)
        })
        raise
