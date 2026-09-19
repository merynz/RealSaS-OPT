from __future__ import annotations

"""Stage-35 canonical dynamic motion proof adapter."""

from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import build_qualified_dynamic_motion_v2
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    canonical_puppet_state_from_dict,
    motion_compile_constraint_set_v2_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_mesh_from_dict,
    qualified_mesh_skin_from_dict,
    qualified_motion_v2_from_dict,
    qualified_presentation_graph_from_dict,
    qualified_skeleton_from_dict,
    read_json,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CarrierCoverageThresholdIR,
    MeshQualificationPolicyIR,
)
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _load_file_ref,
    _sha256,
    _stage_output_payload,
    _write_ir,
)


def _mesh_policy(ctx)->MeshQualificationPolicyIR:
    payload=_stage_output_payload(
        ctx,"26_MESH_CANDIDATE_BUILD","RealSaS.MeshQualificationPolicyIR.v1"
    )
    rows=tuple(
        CarrierCoverageThresholdIR(
            str(row["carrier_class"]),float(row["min_recall"]),float(row["min_precision"]),
            float(row["max_largest_coherent_hole_fraction"]),float(row["max_interior_uncovered_fraction"])
        )
        for row in payload.get("coverage_thresholds") or ()
    )
    return MeshQualificationPolicyIR(
        float(payload["g1_max_normal_refinement_ratio"]),
        float(payload["g1_max_tangential_to_normal_ratio"]),
        float(payload["g3_min_angle_deg"]),
        float(payload["g3_max_aspect_longest_over_min_altitude"]),
        rows,
        str(payload["qualification_policy_lineage_hash"]),
        float(payload.get("g3_min_dynamic_area_ratio",0.05)),
        float(payload.get("g3_max_dynamic_area_ratio",20.0)),
        float(payload.get("g3_max_dynamic_condition_number",16.0)),
        schema_version=str(payload.get("schema_version") or "RealSaS.MeshQualificationPolicyIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _source_foreground_masks(ctx,observation):
    cfg=dict(ctx["run_manifest"].get("observation") or {})
    rows=tuple(cfg.get("source_foreground_masks") or ())
    if len(rows)!=8 or {int(r["view_index"]) for r in rows}!=set(range(8)):
        raise ValueError("MOTION_DYNAMIC_FOREGROUND_MATRIX_INCOMPLETE")
    authority={int(v.view_index):v for v in observation.views}
    out={}
    for row in rows:
        vi=int(row["view_index"])
        path=_load_file_ref(dict(row.get("mask") or {}),json_required=False)
        if _sha256(path)!=authority[vi].foreground_mask_sha256:
            raise ValueError("MOTION_DYNAMIC_FOREGROUND_AUTHORITY_DRIFT")
        raw=path.read_bytes()
        expected=int(authority[vi].width)*int(authority[vi].height)
        if len(raw)!=expected or any(x not in (0,1) for x in raw):
            raise ValueError("MOTION_DYNAMIC_FOREGROUND_MASK_INVALID")
        out[vi]=raw
    return out


def prove_dynamic_motion_stage(ctx:dict)->dict:
    skeleton=qualified_skeleton_from_dict(
        _stage_output_payload(ctx,"18_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1")
    )
    mesh=qualified_mesh_from_dict(
        _stage_output_payload(ctx,"27_QUALIFIED_MESH_GATE","RealSaS.QualifiedMeshIR.v1")
    )
    mesh_skin=qualified_mesh_skin_from_dict(
        _stage_output_payload(ctx,"28_QUALIFIED_MESH_SKIN_TRANSFER","RealSaS.QualifiedMeshSkinIR.v1")
    )
    product_state=canonical_puppet_state_from_dict(
        _stage_output_payload(ctx,"29_CANONICAL_PUPPET_STATE_SEALED","RealSaS.CanonicalPuppetStateIR.v1")
    )
    presentation=qualified_presentation_graph_from_dict(
        _stage_output_payload(ctx,"30_QUALIFIED_PRESENTATION_GRAPH","RealSaS.QualifiedPresentationGraphIR.v1")
    )
    constraints=motion_compile_constraint_set_v2_from_dict(
        _stage_output_payload(ctx,"34_MOTION_COMPILE_RUN","RealSaS.MotionCompileConstraintSetIR.v2")
    )
    motion=qualified_motion_v2_from_dict(
        _stage_output_payload(ctx,"34_MOTION_COMPILE_RUN","RealSaS.QualifiedMotionIR.v2")
    )
    cameras=qualified_camera_set_from_dict(
        _stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    observation=qualified_observation_set_from_dict(
        _stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
    )
    proof=build_qualified_dynamic_motion_v2(
        motion=motion,constraints=constraints,product_state=product_state,skeleton=skeleton,
        mesh=mesh,mesh_skin=mesh_skin,presentation=presentation,mesh_policy=_mesh_policy(ctx),
        cameras=cameras.cameras,source_foreground_masks=_source_foreground_masks(ctx,observation),
    )
    root=ctx["run_root"]/"artifacts"/"35_MOTION_DYNAMIC_PROOF"
    return {
        "status":"PASS",
        "outputs":[_write_ir(root/"qualified_dynamic_motion.json",proof,authority_class="QUALIFIED_DYNAMIC_MOTION")],
        "diagnostics":{
            "dynamic_motion_hash":proof.dynamic_motion_hash,
            "clip_count":len(proof.clips),
            "professional_motion_clip_count":proof.qualification_report["professional_motion_clip_count"],
            "dynamic_proof_passed":True,
            "canonical_3d_single_mesh_truth":True,
            "full_3d_local_quaternion_motion":True,
            "truly_unseen_dynamic_exposure_passed":proof.qualification_report["truly_unseen_dynamic_exposure_passed"],
            "truly_unseen_source_face_count":proof.qualification_report["truly_unseen_source_face_count"],
        },
    }
