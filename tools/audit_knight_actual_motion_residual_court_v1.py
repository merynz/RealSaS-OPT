from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_camera_set_from_dict, qualified_skeleton_from_dict,
    qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_holeless_partitioned_dense_candidate
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    DEFAULT_MAX_EDGE_RATIO, _skin_l1_per_face,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR, build_component_carrier_policy,
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.product_mesh_skin_v1 import _skin_support_coefficients
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import (
    load_json, replay_compacted_dense_face_provenance,
)
from tools.audit_knight_global_skin_region_sweep_v1 import skin_matrix
from tools.audit_knight_probe_conditioned_region_court_v1 import (
    probe_edge_risk, build_probe_partition,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx, _skin, _tracks_for_clip

CLIPS=("demo_idle_v1","demo_run_v1","demo_slash_v1")


def quantiles(x):
    a=np.asarray(x,dtype=np.float64)
    if not len(a): return {"count":0}
    return {k:float(np.quantile(a,q)) for k,q in (
        ("min",0.0),("p50",.5),("p90",.9),("p95",.95),("p99",.99),("max",1.0)
    )} | {"count":int(len(a))}


def face_geometry(rest,faces):
    r=rest[faces]
    e01=np.linalg.norm(r[:,1]-r[:,0],axis=1)
    e12=np.linalg.norm(r[:,2]-r[:,1],axis=1)
    e20=np.linalg.norm(r[:,0]-r[:,2],axis=1)
    edges=np.stack([e01,e12,e20],axis=1)
    mn=edges.min(axis=1); mx=edges.max(axis=1)
    cross=np.linalg.norm(np.cross(r[:,1]-r[:,0],r[:,2]-r[:,0]),axis=1)
    min_alt=cross/np.maximum(mx,1e-15)
    aspect=mx/np.maximum(min_alt,1e-15)
    # smallest angle is opposite shortest edge.
    s=np.sort(edges,axis=1)
    a,b,c=s[:,0],s[:,1],s[:,2]
    cosang=np.clip((b*b+c*c-a*a)/np.maximum(2*b*c,1e-30),-1.0,1.0)
    min_angle=np.degrees(np.arccos(cosang))
    return mn,mx,aspect,min_angle


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--skin-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    surface=rigging_surface_from_dict(load_json(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    skeleton=qualified_skeleton_from_dict(load_json(rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"))
    cameras=tuple(sorted(qualified_camera_set_from_dict(
        load_json(rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json")
    ).cameras,key=lambda x:int(x.view_index)))
    skin=qualified_skin_from_dict(load_json(a.skin_json))
    explicit,prov=replay_compacted_dense_face_provenance(rr,surface)
    sids,jids,W=skin_matrix(surface,skeleton,skin)

    # Reproduce the sealed 73-region OWNER_COPY candidate exactly at the operator level.
    # probe_edge_risk needs the deformation envelope; load through the same codec lazily.
    from compiler.realsas_compiler_core.artifact_codec_v2 import deformation_envelope_from_dict
    envelope=deformation_envelope_from_dict(load_json(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json"))
    risk=probe_edge_risk(surface,skeleton,cameras,envelope,sids,W)
    part,unsafe_edges,crossing_edges,closure=build_probe_partition(
        surface,risk,max_edge_ratio=DEFAULT_MAX_EDGE_RATIO)
    carrier=build_component_carrier_policy(
        partition=part,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id,"MESH",("ACTUAL_MOTION_RESIDUAL_COURT",),
            metadata={"automatic":False,"audit_only":True}) for c in part.components),
        metadata={"audit_only":True})
    candidate=build_holeless_partitioned_dense_candidate(
        surface,part,carrier,
        producer_policy_hash=content_sha256({
            "audit":"ACTUAL_MOTION_RESIDUAL_COURT",
            "mechanical_rule":"OWNER_COPY",
            "partition_components":len(part.components),
        }),
        explicit_face_provenance=explicit)
    validate_canonical_mesh_candidate(candidate,surface=surface,partition=part,carrier_policy=carrier)

    rest,weights,faces=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin)
    rest=np.asarray(rest,dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64)
    faces=np.asarray(faces,dtype=np.int64)
    skin_l1=_skin_l1_per_face(weights,faces)
    rest_min_edge,rest_max_edge,rest_aspect,rest_min_angle=face_geometry(rest,faces)

    by_vid={str(v.candidate_vertex_id):v for v in candidate.vertices}
    sid_to_component={}
    for c in part.components:
        for sid in c.surface_ids:
            sid_to_component[str(sid)]=str(c.component_id)

    support_class=[]
    face_components=[]
    face_support_sids=[]
    for face in candidate.faces:
        vs=[by_vid[str(x)] for x in face]
        modes=[str(v.support_binding.mode) for v in vs]
        if any(m=="SEAM_GEOMETRY_INTERPOLATION" for m in modes):
            cls="HAS_GENERATED_SEAM"
        elif all(m=="IDENTITY_SURFACE_NODE" for m in modes):
            cls="IDENTITY_ONLY"
        else:
            cls="OTHER_SUPPORT_MODE"
        support_class.append(cls)
        sset=set(); cset=set()
        for v in vs:
            for sid,coeff in _skin_support_coefficients(v):
                if float(coeff)>1e-12:
                    sid=str(sid); sset.add(sid)
                    cset.add(sid_to_component[sid])
        face_support_sids.append(tuple(sorted(sset)))
        face_components.append(tuple(sorted(cset)))

    source_report=load_json(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))
    max_ratio=np.ones(len(faces),dtype=np.float64)
    witness=[None]*len(faces)
    frame_rows=[]
    for clip in CLIPS:
        payload=load_json(rr/"inputs/motion/quaternius_knight_v1"/f"{clip}.motion.json")
        tracks,_=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(0.0,float(payload["duration_seconds"]),4,
                          endpoint=not bool(payload.get("loop")))
        for frame,t in enumerate(times):
            mats,_,_=_joint_pose_v2(
                skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,weights,jids,mats)
            r=rest[faces]; p=posed[faces]
            rl=np.stack([
                np.linalg.norm(r[:,1]-r[:,0],axis=1),
                np.linalg.norm(r[:,2]-r[:,1],axis=1),
                np.linalg.norm(r[:,0]-r[:,2],axis=1)],axis=1)
            pl=np.stack([
                np.linalg.norm(p[:,1]-p[:,0],axis=1),
                np.linalg.norm(p[:,2]-p[:,1],axis=1),
                np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
            e=np.max(pl/np.maximum(rl,1e-15),axis=1)
            improved=e>max_ratio
            for fi in np.nonzero(improved)[0].tolist():
                witness[fi]={
                    "clip_id":clip,"frame":int(frame),"time_seconds":float(t),
                    "edge_ratio":float(e[fi]),
                }
            max_ratio=np.maximum(max_ratio,e)
            frame_rows.append({
                "clip_id":clip,"frame":int(frame),"time_seconds":float(t),
                "edge_gt2":int(np.count_nonzero(e>2.0)),
                "edge_gt3":int(np.count_nonzero(e>3.0)),
                "edge_gt4":int(np.count_nonzero(e>4.0)),
                "edge_gt6":int(np.count_nonzero(e>6.0)),
                "edge_gt10":int(np.count_nonzero(e>10.0)),
                "edge_max":float(np.max(e)),
                "edge_p99":float(np.quantile(e,.99)),
            })

    offender=np.nonzero(max_ratio>4.0)[0]
    top=[]
    for fi in sorted(offender.tolist(),key=lambda i:(-max_ratio[i],i)):
        top.append({
            "face_index":int(fi),
            "max_actual_edge_ratio":float(max_ratio[fi]),
            "witness":witness[fi],
            "support_class":support_class[fi],
            "mechanical_component_ids":list(face_components[fi]),
            "mechanical_component_count":len(face_components[fi]),
            "support_surface_ids":list(face_support_sids[fi]),
            "skin_l1":float(skin_l1[fi]),
            "rest_min_edge":float(rest_min_edge[fi]),
            "rest_max_edge":float(rest_max_edge[fi]),
            "rest_aspect":float(rest_aspect[fi]),
            "rest_min_angle_deg":float(rest_min_angle[fi]),
        })

    report={
        "schema":"RealSaS.KnightActualMotionResidualCourt.v1",
        "status":"COMPLETE__AUDIT_ONLY__NO_PARENT_MUTATION",
        "candidate":{
            "component_count":len(part.components),
            "vertex_count":len(candidate.vertices),
            "face_count":len(candidate.faces),
            "unsafe_source_edge_seed_count":len(unsafe_edges),
            "cut_closed_crossing_edge_count":len(crossing_edges),
            "closure":closure,
        },
        "face_provenance_replay":prov,
        "actual_motion":{
            "sampled_frame_count":len(frame_rows),
            "max_edge_ratio":float(np.max(max_ratio)),
            "p99_face_worst_ratio":float(np.quantile(max_ratio,.99)),
            "face_gt2":int(np.count_nonzero(max_ratio>2.0)),
            "face_gt3":int(np.count_nonzero(max_ratio>3.0)),
            "face_gt4":int(np.count_nonzero(max_ratio>4.0)),
            "face_gt6":int(np.count_nonzero(max_ratio>6.0)),
            "face_gt10":int(np.count_nonzero(max_ratio>10.0)),
            "frames":frame_rows,
        },
        "gt4_summary":{
            "count":len(offender),
            "seam_count":sum(support_class[i]=="HAS_GENERATED_SEAM" for i in offender),
            "identity_only_count":sum(support_class[i]=="IDENTITY_ONLY" for i in offender),
            "other_support_count":sum(support_class[i]=="OTHER_SUPPORT_MODE" for i in offender),
            "multi_component_face_count":sum(len(face_components[i])>1 for i in offender),
            "skin_l1_gt_0p5_count":int(np.count_nonzero(skin_l1[offender]>.5)),
            "rest_aspect_gt_policy_count":int(np.count_nonzero(rest_aspect[offender]>16.0)),
            "rest_min_angle_lt_7p5_count":int(np.count_nonzero(rest_min_angle[offender]<7.5)),
            "max_ratio":quantiles(max_ratio[offender]),
            "skin_l1":quantiles(skin_l1[offender]),
            "rest_min_edge":quantiles(rest_min_edge[offender]),
            "rest_aspect":quantiles(rest_aspect[offender]),
            "rest_min_angle_deg":quantiles(rest_min_angle[offender]),
        },
        "offenders_gt4":top,
        "claim_boundary":[
            "Only exact Knight idle/run/slash clip samples used by the existing motion court are measured.",
            "No 120-degree arbitrary stress is rerun.",
            "QualifiedSkinIR, partition and candidate are not mutated.",
            "This court diagnoses the remaining actual-motion tail only."
        ]
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("ACTUAL_MOTION_RESIDUAL="+json.dumps({
        "components":len(part.components),
        "faces":len(candidate.faces),
        "gt4":report["actual_motion"]["face_gt4"],
        "gt6":report["actual_motion"]["face_gt6"],
        "gt10":report["actual_motion"]["face_gt10"],
        "worst":report["actual_motion"]["max_edge_ratio"],
        "summary":report["gt4_summary"],
        "top":top[:8],
    },sort_keys=True))

if __name__=="__main__": main()
