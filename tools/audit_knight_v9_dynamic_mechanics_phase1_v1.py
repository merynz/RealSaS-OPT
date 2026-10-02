from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np

from tools.audit_knight_repaired_quality_collapse_v1 import loadj,build_repaired_surface,report_quality
from tools.audit_knight_v9_source_fidelity_v1 import sha
from compiler.realsas_compiler_core.preproduct_authority_v1 import signed_zero_surface_from_dict,normalization_domain_from_dict
from compiler.realsas_compiler_core.refined_surface_skin_proposal_v1 import propose_refined_surface_skin_v1
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import (
    CLIPS,FULL_MOTION_SAMPLES,VISIBLE_FLIP_SAMPLE_INDICES,DEMO_VIEWS,
    _faces,_triangle_geometry,_projected_flip_metrics,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    component_carrier_policy_from_dict,
    deformation_envelope_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.dynamic_geometry_integrity_v2 import unexpected_intersection_pairs
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import run_g3_local_frame_micro_stress_v2
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,build_component_carrier_policy,
    validate_canonical_mesh_candidate,
    validate_component_carrier_policy,
    validate_mechanical_partition,
)
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from tools.demo.render_knight_motion_preview_v1 import _ctx,_skin,_tracks_for_clip

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--partition-json",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--compaction-seal",type=Path,required=True)
    ap.add_argument("--static-report",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    ctx=_ctx(a.authority_root,a.run_id)
    rr=ctx["run_root"]
    candidate=canonical_mesh_candidate_from_dict(loadj(a.candidate_json))
    source_surface=rigging_surface_from_dict(stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    partition=mechanical_partition_from_dict(loadj(a.partition_json))
    seal=loadj(a.compaction_seal)
    zero=signed_zero_surface_from_dict(stage_output_payload(ctx,"12_ZERO_SURFACE_DECODED","RealSaS.SignedZeroSurfaceSealIR.v1"))
    if sha(a.inverse_npz)!=seal["inverse_npz_sha256"] or sha(Path(zero.npz_path))!=zero.npz_sha256 or zero.npz_sha256!=seal["source_zero_surface_sha256"]:
        raise RuntimeError("V9_DYNAMIC_REFINEMENT_SOURCE_BYTES_DRIFT")
    surface,_explicit=build_repaired_surface(rr,a.inverse_npz)
    carrier=build_component_carrier_policy(
        partition=partition,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id,"MESH",("REPAIRED_STATIC_QUALITY_ALTERNATING_V3",),
            metadata={"automatic":True,"semantic_recognition_used":False}
        ) for c in partition.components),
        metadata={"default_carrier":"MESH","automatic":True},
    )
    skeleton=qualified_skeleton_from_dict(stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    source_skin=qualified_skin_from_dict(stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))
    camera_set=qualified_camera_set_from_dict(stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    cameras=tuple(sorted(camera_set.cameras,key=lambda x:int(x.view_index)))
    envelope=deformation_envelope_from_dict(stage_output_payload(ctx,"34_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1"))
    policy=mesh_policy_from_dict(stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1"))

    validate_mechanical_partition(partition,surface)
    validate_component_carrier_policy(carrier,partition)
    validate_canonical_mesh_candidate(candidate,surface=surface,partition=partition,carrier_policy=carrier)
    static_report=loadj(a.static_report)
    quality=report_quality(candidate,policy)
    if (static_report["final_candidate_lineage_hash"]!=candidate.candidate_lineage_hash
            or quality["policy_violating_face_count"]!=0 or not _manifold_report(candidate.faces)["passed"]):
        raise RuntimeError("V9_DYNAMIC_STATIC_COURT_NOT_PASS")
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        world=np.asarray(norm.center_xyz)[None,:]+np.asarray(z["vertices_normalized"],np.float64)*float(norm.half_extent)
    with np.load(a.inverse_npz,allow_pickle=False) as z:
        proposal,transfer_receipt=propose_refined_surface_skin_v1(
            source_surface=source_surface,target_surface=surface,skeleton=skeleton,source_skin=source_skin,
            dense_positions=world,source_inverse=z["base_inverse"],target_inverse=z["final_inverse"],
            dense_source_sha256=zero.npz_sha256,
        )
    # Numeric simplex requalification only; actual mechanics remain unproven.
    skin=qualify_skin(surface,skeleton,proposal,max_simplex_repair_l1=1e-8,max_total_correction_l1=1e-4)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for name,value in (("refined_surface.json",surface.to_dict()),("transferred_skin.json",skin.to_dict()),
                       ("transfer_receipt.json",transfer_receipt)):
        (a.out.parent/name).write_text(json.dumps(value,sort_keys=True)+"\n")
    print("V9_DYNAMIC_INPUT_BINDINGS="+json.dumps(transfer_receipt,sort_keys=True),flush=True)

    faces=_faces(candidate)
    rest,W,faces_support=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin)
    rest=np.asarray(rest,dtype=np.float64); W=np.asarray(W,dtype=np.float64)
    faces_support=np.asarray(faces_support,dtype=np.int64)
    if not np.array_equal(faces,faces_support):
        raise RuntimeError("V9_DYNAMIC_SUPPORT_FACE_ORDER_DRIFT")
    simplex_residual=float(np.max(np.abs(W.sum(axis=1)-1.0)))
    negative=int(np.count_nonzero(W < -1e-10))
    nonfinite=int(np.count_nonzero(~np.isfinite(W)))

    t0=time.perf_counter()
    g3=run_g3_local_frame_micro_stress_v2(
        candidate,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy
    )
    g3_seconds=time.perf_counter()-t0
    (a.out.parent/"g3_report.json").write_text(json.dumps(g3.to_dict(),sort_keys=True)+"\n")
    print("V9_DYNAMIC_G3="+json.dumps({"passed":g3.passed,"failures":g3.failure_invariants,
          "seconds":g3_seconds,"condition":g3.maximum_condition_number}),flush=True)

    source_report=loadj(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))
    joint_ids=tuple(j.canonical_joint_id for j in skeleton.joints)
    clip_rows=[]
    global_worst_key=(-1,-1,-1.0)
    global_worst_pose=None
    global_worst_locator=None
    max_edge_gt4=max_edge_gt10=max_posed_degenerate=0
    worst_edge=0.0
    max_visible_projected_flips=0
    max_all_projected_flips=0

    for clip_id in CLIPS:
        payload=loadj(rr/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json")
        tracks,mapping=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(
            0.0,float(payload["duration_seconds"]),FULL_MOTION_SAMPLES,
            endpoint=not bool(payload.get("loop")),
        )
        frames=[]
        clip_worst=(-1,-1,-1.0)
        clip_worst_index=0
        for frame_index,time_seconds in enumerate(times):
            mats,_joint_pos,frame_hash=_joint_pose_v2(
                skeleton=skeleton,tracks=tracks,time_seconds=float(time_seconds),cameras=cameras
            )
            posed=np.asarray(_skin(rest,W,joint_ids,mats),dtype=np.float64)
            mech=_triangle_geometry(rest,posed,faces)
            proj=_projected_flip_metrics(
                candidate=candidate,faces=faces,rest=rest,posed=posed,cameras=cameras,
                do_visible_census=frame_index in VISIBLE_FLIP_SAMPLE_INDICES,
            )
            key=(int(mech["edge_gt_10"]),int(mech["edge_gt_4"]),float(mech["edge_max"]))
            if key>clip_worst:
                clip_worst=key;clip_worst_index=frame_index
            if key>global_worst_key:
                global_worst_key=key
                global_worst_pose=posed.copy()
                global_worst_locator={
                    "clip_id":clip_id,"frame_index":int(frame_index),
                    "time_seconds":float(time_seconds),
                }
            max_edge_gt4=max(max_edge_gt4,int(mech["edge_gt_4"]))
            max_edge_gt10=max(max_edge_gt10,int(mech["edge_gt_10"]))
            worst_edge=max(worst_edge,float(mech["edge_max"]))
            max_posed_degenerate=max(max_posed_degenerate,int(mech["posed_degenerate_face_count"]))
            max_visible_projected_flips=max(max_visible_projected_flips,int(proj["max_visible_projected_flip_count"]))
            max_all_projected_flips=max(max_all_projected_flips,int(proj["max_all_measurable_projected_flip_count"]))
            frames.append({
              "frame_index":int(frame_index),"time_seconds":float(time_seconds),
              "motion_frame_hash":str(frame_hash),"mechanical":mech,
              "projected_orientation_diagnostic":proj,
            })
        clip_rows.append({
          "clip_id":clip_id,"mapping":mapping,"sample_count":len(times),
          "worst_frame_index":clip_worst_index,"worst_key":clip_worst,"frames":frames,
        })
        (a.out.parent/"motion_checkpoint.json").write_text(json.dumps(clip_rows,sort_keys=True)+"\n")
        print("V9_DYNAMIC_CLIP="+json.dumps({"clip":clip_id,"worst":clip_worst,
              "max_posed_degenerate":max_posed_degenerate}),flush=True)

    if global_worst_pose is None:
        raise RuntimeError("V9_DYNAMIC_GLOBAL_WORST_MISSING")

    # Full all-face intersection census only at rest and global worst actual-motion frame.
    inter_t0=time.perf_counter()
    rest_pairs=set(unexpected_intersection_pairs(vertices=rest,faces=faces))
    print("V9_DYNAMIC_REST_INTERSECTIONS="+str(len(rest_pairs)),flush=True)
    worst_pairs=set(unexpected_intersection_pairs(vertices=global_worst_pose,faces=faces))
    full_new=sorted(worst_pairs-rest_pairs)
    full_intersection_seconds=time.perf_counter()-inter_t0

    # Render-visible subset census in demo views, matching historical diagnostic.
    visible_union=set()
    for camera in cameras:
        if int(camera.view_index) not in DEMO_VIEWS:
            continue
        vis=rasterize_visible_owner(candidate,camera,positions=global_worst_pose,coverage_scale=1)
        owner=np.asarray(vis.owner_face_index,dtype=np.int64)
        visible_union.update(int(x) for x in np.unique(owner[owner>=0]).tolist())
    visible_indices=np.asarray(sorted(visible_union),dtype=np.int64)
    # The exact all-face census already includes these pairs. Preserve global IDs.
    vis_rest={p for p in rest_pairs if p[0] in visible_union and p[1] in visible_union}
    vis_worst={p for p in worst_pairs if p[0] in visible_union and p[1] in visible_union}
    visible_new=sorted(vis_worst-vis_rest)

    mechanical_pass=bool(
        simplex_residual<=1e-8 and negative==0 and nonfinite==0
        and bool(g3.passed)
        and max_edge_gt4==0 and max_edge_gt10==0
        and max_posed_degenerate==0
        and len(full_new)==0
    )
    report={
      "schema":"RealSaS.KnightV9DynamicMechanicsPhase1.v1",
      "status":"PASS_PHASE1_MECHANICS" if mechanical_pass else "FAIL_PHASE1_MECHANICS",
      "candidate_lineage_hash":candidate.candidate_lineage_hash,
      "surface_lineage_hash":surface.geometry_lineage_hash,
      "skin_lineage_hash":skin.skin_lineage_hash,
      "skin_transfer":{
        "method":"EXACT_DENSE_CLUSTER_PARENT_CONSTANT_THEN_CANDIDATE_CONVEX_TRANSFER",
        "receipt":transfer_receipt,
        "candidate_weight_simplex_residual_max":simplex_residual,
        "negative_weight_count":negative,
        "nonfinite_weight_count":nonfinite,
        "model_refit_requirement":"UNDETERMINED_BY_THIS_COURT",
      },
      "g3":{
        "passed":bool(g3.passed),"failure_invariants":list(g3.failure_invariants),
        "probe_count":int(g3.probe_count),"face_count":int(g3.face_count),
        "minimum_area_ratio":float(g3.minimum_area_ratio),
        "maximum_area_ratio":float(g3.maximum_area_ratio),
        "maximum_condition_number":float(g3.maximum_condition_number),
        "minimum_edge_ratio":float(g3.minimum_edge_ratio),
        "maximum_edge_ratio":float(g3.maximum_edge_ratio),
        "seconds":g3_seconds,
      },
      "actual_motion":{
        "clips":clip_rows,
        "aggregate":{
          "max_edge_gt_4":int(max_edge_gt4),
          "max_edge_gt_10":int(max_edge_gt10),
          "worst_edge_max":float(worst_edge),
          "max_posed_degenerate_face_count":int(max_posed_degenerate),
          "global_worst_frame":global_worst_locator,
          "projected_orientation_diagnostic":{
            "max_visible_projected_flip_count_demo_views":int(max_visible_projected_flips),
            "max_all_measurable_projected_flip_count":int(max_all_projected_flips),
            "mechanical_fail_criterion":False,
          },
        },
      },
      "self_intersection_worst_frame":{
        "full_rest_pair_count":len(rest_pairs),
        "full_worst_pair_count":len(worst_pairs),
        "full_new_pair_count":len(full_new),
        "full_new_pairs_sample":[[int(x),int(y)] for x,y in full_new[:64]],
        "full_census_seconds":full_intersection_seconds,
        "visible_face_count":len(visible_indices),
        "visible_rest_pair_count":len(vis_rest),
        "visible_worst_pair_count":len(vis_worst),
        "visible_new_pair_count":len(visible_new),
        "visible_new_pairs_sample":[[int(x),int(y)] for x,y in visible_new[:64]],
      },
      "verdict":{
        "mechanical_pass":mechanical_pass,
        "product_authority_minted":False,
        "scope":"EXACT_PARENT_CONSTANT_SKIN_HYPOTHESIS__SAMPLED_MOTION_ONLY",
        "source_fidelity_sealed":False,
        "projected_flips_reclassified_as_presentation_diagnostic":True,
        "next_if_pass":"EXHAUSTIVE_ALL_51_SAMPLED_FRAMES_SELF_INTERSECTION_COURT",
        "next_if_fail":"LOCALIZE_EXACT_DYNAMIC_FAILURE_OWNER",
      },
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_V9_DYNAMIC_MECHANICS_PHASE1="+json.dumps({
      "status":report["status"],"g3":g3.passed,
      "g3_max_condition":g3.maximum_condition_number,
      "motion_gt4":max_edge_gt4,"motion_gt10":max_edge_gt10,
      "motion_worst_edge":worst_edge,"posed_degenerate":max_posed_degenerate,
      "full_rest_intersections":len(rest_pairs),
      "full_worst_intersections":len(worst_pairs),
      "full_new_intersections":len(full_new),
      "visible_new_intersections":len(visible_new),
      "visible_projected_flips_diagnostic":max_visible_projected_flips,
      "mechanical_pass":mechanical_pass,
    },sort_keys=True),flush=True)
    if not mechanical_pass:
        raise SystemExit(2)

if __name__=="__main__":
    main()
