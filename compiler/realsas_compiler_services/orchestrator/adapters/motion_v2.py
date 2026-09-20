from __future__ import annotations

"""V2 Stage39-41 motion authorization, full-3D compile and dynamic proof."""

from dataclasses import replace
import json

from compiler.realsas_compiler_core.motion_compile_v2 import (
    build_qualified_motion_v2,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import (
    build_qualified_dynamic_motion_v2,
)
from compiler.realsas_compiler_core.motion_source_v1 import (
    QualifiedMotionSourceSealIR,
    build_motion_source_asset,
    build_motion_source_set,
    motion_source_seal_hash,
)
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    canonical_puppet_state_from_dict,
    motion_compile_constraint_set_v2_from_dict,
    motion_source_set_from_dict,
    qualified_camera_set_from_dict,
    qualified_mesh_from_dict,
    qualified_mesh_skin_from_dict,
    qualified_motion_source_seal_from_dict,
    qualified_motion_v2_from_dict,
    qualified_observation_set_from_dict,
    qualified_presentation_graph_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CarrierCoverageThresholdIR,
    MeshQualificationPolicyIR,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v1 import (
    qualified_dynamic_motion_hash,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    load_file_ref,
    sha256_file,
    stage_output_payload,
    write_ir,
)
from compiler.realsas_compiler_core.types import QualificationError


def _motion_assets(ctx: dict):
    cfg = dict(ctx["run_manifest"].get("motion") or {})
    rows = tuple(cfg.get("sources") or ())
    if not rows:
        raise QualificationError("MOTION_V2_SOURCE_SET_EMPTY")
    assets = []
    payloads = {}
    for raw in rows:
        row = dict(raw)
        if str(row.get("source_kind") or "") != "EXTERNAL_ARTIST_CLIP_V1":
            raise QualificationError(
                "MOTION_V2_CURRENT_PRODUCT_REQUIRES_EXTERNAL_ARTIST_CLIP_V2"
            )
        path = load_file_ref(dict(row.get("file") or {}), json_required=False)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("schema") or payload.get("schema_version") or "") != "RealSaS.MotionSourceClip.v2":
            raise QualificationError("MOTION_V2_SOURCE_SCHEMA_REQUIRED")
        if str(payload.get("source_space") or "") != "SOURCE_RIG_TRACKS_V2":
            raise QualificationError("MOTION_V2_SOURCE_RIG_TRACKS_V2_REQUIRED")
        clip_id = str(payload.get("clip_id") or row.get("clip_id") or "")
        if not clip_id or clip_id in payloads:
            raise QualificationError("MOTION_V2_CLIP_ID_INVALID")
        asset = build_motion_source_asset(
            clip_id=clip_id,
            clip_kind=str(payload.get("clip_kind") or row.get("clip_kind") or ""),
            source_kind="EXTERNAL_ARTIST_CLIP_V1",
            source_space="SOURCE_RIG_TRACKS_V2",
            duration_seconds=float(payload.get("duration_seconds", 0.0)),
            loop=bool(payload.get("loop", False)),
            channel_contract=tuple(map(str, payload.get("channel_contract") or ())),
            source_payload=payload,
            source_ref=f"file:{path.name}:{sha256_file(path)}",
            metadata={
                "external_file_sha256": sha256_file(path),
                "professional_source": True,
                "full_3d_motion_required": True,
                "v2_authorization": True,
            },
        )
        assets.append(asset)
        payloads[clip_id] = payload
    return tuple(assets), payloads


def seal_motion_source_stage(ctx: dict) -> dict:
    mechanical = canonical_puppet_state_from_dict(
        stage_output_payload(
            ctx,
            "38_CANONICAL_PUPPET_SEALED",
            "RealSaS.CanonicalPuppetStateIR.v1",
        )
    )
    rest_proof = stage_output_payload(
        ctx,
        "25_CAA_REFERENCE_REST_RENDER_PROOF",
        "RealSaS.CAARestRenderProofIR.v2",
    )
    if str(rest_proof.get("qualification_report", {}).get("status")) != "PASS_CAA_REFERENCE_REST":
        raise QualificationError("MOTION_V2_CAA_REST_PRECONDITION_NOT_PASS")

    assets, _payloads = _motion_assets(ctx)
    source_set = build_motion_source_set(
        assets,
        metadata={
            "v2_motion_authorization": True,
            "caa_rest_precondition_hash": str(rest_proof["proof_hash"]),
        },
    )
    seal = QualifiedMotionSourceSealIR(
        source_set_binding_hash=source_set.source_set_hash,
        product_state_binding_hash=mechanical.product_state_hash,
        rest_preservation_binding_hash=str(rest_proof["proof_hash"]),
        source_assets=source_set.assets,
        qualification_report={
            "status": "PASS_SOURCE_IDENTITY_AND_PRECONDITION_ONLY",
            "source_asset_count": len(source_set.assets),
            "retargeting_performed": False,
            "motion_compilation_performed": False,
            "motion_quality_claimed": False,
            "stage40_compile_required": True,
            "stage41_dynamic_proof_required": True,
            "rest_precondition_kind": "CAA_REFERENCE_REST_PROOF_V2",
        },
        motion_source_seal_hash="",
        metadata={
            "legacy_rest_preservation_ir_used": False,
            "caa_rest_proof_binding_hash": str(rest_proof["proof_hash"]),
            "professional_motion_source_required": True,
        },
    )
    seal = replace(seal, motion_source_seal_hash=motion_source_seal_hash(seal))
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "motion_source_set.json",
                source_set,
                authority_class="MOTION_SOURCE_IDENTITY",
            ),
            write_ir(
                root / "qualified_motion_source_seal.json",
                seal,
                authority_class="QUALIFIED_MOTION_SOURCE_SEAL_V2",
            ),
        ],
        "diagnostics": {
            "source_set_hash": source_set.source_set_hash,
            "motion_source_seal_hash": seal.motion_source_seal_hash,
            "clip_ids": [asset.clip_id for asset in source_set.assets],
            "caa_rest_proof_binding_hash": str(rest_proof["proof_hash"]),
        },
    }


def compile_motion_stage(ctx: dict) -> dict:
    cfg = dict(ctx["run_manifest"].get("motion") or {})
    unknown = set(cfg) - {"sources", "compiler", "dynamic"}
    if unknown:
        return {
            "status": "BLOCKED",
            "blockers": ["MOTION_V2_CONFIG_UNSUPPORTED"],
            "diagnostics": {"unsupported_keys": sorted(unknown)},
        }

    mechanical = canonical_puppet_state_from_dict(
        stage_output_payload(
            ctx,
            "38_CANONICAL_PUPPET_SEALED",
            "RealSaS.CanonicalPuppetStateIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
        )
    )
    envelope_payload = stage_output_payload(
        ctx,
        "34_DEFORMATION_CAPABILITY_ENVELOPE",
        "RealSaS.DeformationCapabilityEnvelopeIR.v1",
    )
    from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
        deformation_envelope_from_dict,
    )
    envelope = deformation_envelope_from_dict(envelope_payload)
    presentation = qualified_presentation_graph_from_dict(
        stage_output_payload(
            ctx,
            "38_CANONICAL_PUPPET_SEALED",
            "RealSaS.QualifiedPresentationGraphIR.v1",
        )
    )
    source_set = motion_source_set_from_dict(
        stage_output_payload(
            ctx,
            "39_MOTION_SOURCE_OR_PRESET_SEAL",
            "RealSaS.MotionSourceSetIR.v1",
        )
    )
    source_seal = qualified_motion_source_seal_from_dict(
        stage_output_payload(
            ctx,
            "39_MOTION_SOURCE_OR_PRESET_SEAL",
            "RealSaS.QualifiedMotionSourceSealIR.v1",
        )
    )
    _assets, payloads = _motion_assets(ctx)
    constraints, motion = build_qualified_motion_v2(
        source_set=source_set,
        source_seal=source_seal,
        source_payloads=payloads,
        product_state=mechanical,
        skeleton=skeleton,
        envelope=envelope,
        presentation=presentation,
        compiler_config=dict(cfg.get("compiler") or {}),
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "motion_compile_constraints.json",
                constraints,
                authority_class="MOTION_COMPILE_CONSTRAINTS_V2",
            ),
            write_ir(
                root / "qualified_motion.json",
                motion,
                authority_class="QUALIFIED_FULL_3D_MOTION_V2",
            ),
        ],
        "diagnostics": {
            "constraint_set_hash": constraints.constraint_set_hash,
            "motion_lineage_hash": motion.motion_lineage_hash,
            "clip_count": len(motion.clips),
            "full_3d_local_quaternion_motion": True,
            "motion_quality_claimed": False,
        },
    }


def _mesh_policy(ctx: dict) -> MeshQualificationPolicyIR:
    payload = stage_output_payload(
        ctx,
        "18_CANONICAL_MESH_ADDRESSING_BUILD",
        "RealSaS.MeshQualificationPolicyIR.v1",
    )
    rows = tuple(
        CarrierCoverageThresholdIR(
            str(row["carrier_class"]),
            float(row["min_recall"]),
            float(row["min_precision"]),
            float(row["max_largest_coherent_hole_fraction"]),
            float(row["max_interior_uncovered_fraction"]),
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
        float(payload.get("g3_min_dynamic_area_ratio", 0.05)),
        float(payload.get("g3_max_dynamic_area_ratio", 20.0)),
        float(payload.get("g3_max_dynamic_condition_number", 16.0)),
        schema_version=str(
            payload.get("schema_version") or "RealSaS.MeshQualificationPolicyIR.v1"
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def _source_foreground_masks(ctx: dict, observation):
    cfg = dict(ctx["run_manifest"].get("observation") or {})
    rows = tuple(cfg.get("source_foreground_masks") or ())
    if len(rows) != 8 or {int(row["view_index"]) for row in rows} != set(range(8)):
        raise QualificationError("MOTION_V2_FOREGROUND_MATRIX_INCOMPLETE")
    authority = {int(view.view_index): view for view in observation.views}
    output = {}
    for row in rows:
        view = int(row["view_index"])
        path = load_file_ref(dict(row.get("mask") or {}), json_required=False)
        if sha256_file(path) != authority[view].foreground_mask_sha256:
            raise QualificationError("MOTION_V2_FOREGROUND_AUTHORITY_DRIFT")
        raw = path.read_bytes()
        if len(raw) != int(authority[view].width) * int(authority[view].height):
            raise QualificationError("MOTION_V2_FOREGROUND_SIZE_INVALID")
        output[view] = raw
    return output


def prove_dynamic_motion_stage(ctx: dict) -> dict:
    mechanical = canonical_puppet_state_from_dict(
        stage_output_payload(
            ctx,
            "38_CANONICAL_PUPPET_SEALED",
            "RealSaS.CanonicalPuppetStateIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
        )
    )
    mesh = qualified_mesh_from_dict(
        stage_output_payload(
            ctx,
            "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
            "RealSaS.QualifiedMeshIR.v1",
        )
    )
    mesh_skin = qualified_mesh_skin_from_dict(
        stage_output_payload(
            ctx,
            "36_QUALIFIED_MESH_SKIN_TRANSFER",
            "RealSaS.QualifiedMeshSkinIR.v1",
        )
    )
    presentation = qualified_presentation_graph_from_dict(
        stage_output_payload(
            ctx,
            "38_CANONICAL_PUPPET_SEALED",
            "RealSaS.QualifiedPresentationGraphIR.v1",
        )
    )
    constraints = motion_compile_constraint_set_v2_from_dict(
        stage_output_payload(
            ctx,
            "40_MOTION_COMPILE_RUN",
            "RealSaS.MotionCompileConstraintSetIR.v2",
        )
    )
    motion = qualified_motion_v2_from_dict(
        stage_output_payload(
            ctx,
            "40_MOTION_COMPILE_RUN",
            "RealSaS.QualifiedMotionIR.v2",
        )
    )
    cameras = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    proof = build_qualified_dynamic_motion_v2(
        motion=motion,
        constraints=constraints,
        product_state=mechanical,
        skeleton=skeleton,
        mesh=mesh,
        mesh_skin=mesh_skin,
        presentation=presentation,
        mesh_policy=_mesh_policy(ctx),
        cameras=cameras.cameras,
        source_foreground_masks=_source_foreground_masks(ctx, observation),
    )
    if proof.dynamic_motion_hash != qualified_dynamic_motion_hash(proof):
        raise QualificationError("MOTION_V2_DYNAMIC_HASH_DRIFT")
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "qualified_dynamic_motion.json",
                proof,
                authority_class="QUALIFIED_DYNAMIC_MOTION_V2",
            )
        ],
        "diagnostics": {
            "dynamic_motion_hash": proof.dynamic_motion_hash,
            "dynamic_proof_passed": True,
            "full_3d_local_quaternion_motion": True,
            "rest_unseen_exposure_gate_removed": True,
            "rest_unseen_exposed_fraction": proof.qualification_report.get(
                "rest_unseen_dynamic_exposed_fraction", 0.0
            ),
            "compiled_unobserved_appearance_budget_gate_stage": 45,
        },
    }
