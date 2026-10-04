from __future__ import annotations

"""V2 Stage39-41 motion authorization, full-3D compile and dynamic proof."""

from dataclasses import replace
import json

from compiler.realsas_compiler_core.motion_compile_v2 import (
    build_qualified_motion_v2,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import (
    DYNAMIC_EVALUATOR_SEMANTIC_VERSION_V2,
    DynamicMotionProofFailure,
    build_qualified_dynamic_motion_v2,
)
from compiler.realsas_compiler_core.motion_source_v1 import (
    QualifiedMotionSourceSealIR,
    build_motion_source_asset,
    build_motion_source_set,
    motion_source_seal_hash,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
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
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import (
    qualified_dynamic_motion_v2_hash,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    load_file_ref,
    sha256_file,
    stage_output_payload,
    write_ir,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.proof.failure_signatures import (
    derive_failure_signatures,
    no_owner_attribution,
)
from compiler.realsas_compiler_services.proof.stage41_failure_context_v1 import (
    build_stage41_failure_attribution_context_v1,
)


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


def _demo_failed_rest_precondition_admissible(
    ctx: dict,
    rest_proof: dict,
) -> bool:
    if str(ctx["ledger"].get("execution_class") or "") != "DEMO_WITNESS":
        return False
    demo = dict(ctx["run_manifest"].get("demo_execution") or {})
    if (
        demo.get("stage13_scientific_pass") is not False
        or demo.get("product_authority_claimed") is not False
    ):
        return False
    stage25 = next(
        (
            row
            for row in ctx["ledger"].get("stages") or ()
            if str(row.get("id") or "")
            == "25_CAA_REFERENCE_REST_RENDER_PROOF"
        ),
        None,
    )
    if (
        stage25 is None
        or str(stage25.get("status") or "") != "PASS_DEMO_ONLY"
    ):
        return False
    report = dict(rest_proof.get("qualification_report") or {})
    metadata = dict(rest_proof.get("metadata") or {})
    return (
        report.get("status") == "FAIL_CAA_REFERENCE_REST"
        and report.get("every_direction_passed") is False
        and report.get("demo_only_measured_failure_admission") is True
        and report.get("strict_rest_proof_passed") is False
        and report.get("product_pass") is False
        and report.get("product_authority_claimed") is False
        and metadata.get("demo_only_measurement") is True
        and metadata.get("strict_failure_preserved") is True
        and metadata.get("product_authority_claimed") is False
    )


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
    strict_rest_pass = (
        str(
            rest_proof.get("qualification_report", {}).get("status")
        )
        == "PASS_CAA_REFERENCE_REST"
    )
    demo_rest_measurement = (
        not strict_rest_pass
        and _demo_failed_rest_precondition_admissible(
            ctx, rest_proof
        )
    )
    if not strict_rest_pass and not demo_rest_measurement:
        raise QualificationError(
            "MOTION_V2_CAA_REST_PRECONDITION_NOT_PASS"
        )

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
            "rest_precondition_kind": (
                "CAA_REFERENCE_REST_PROOF_V2"
                if strict_rest_pass
                else "DEMO_ONLY_MEASURED_CAA_REFERENCE_REST_FAILURE_V2"
            ),
            "strict_rest_precondition_passed": bool(
                strict_rest_pass
            ),
            "demo_only_measured_rest_failure": bool(
                demo_rest_measurement
            ),
            "product_authority_claimed": False,
        },
        motion_source_seal_hash="",
        metadata={
            "legacy_rest_preservation_ir_used": False,
            "caa_rest_proof_binding_hash": str(rest_proof["proof_hash"]),
            "professional_motion_source_required": True,
            "demo_only_rest_measurement": bool(
                demo_rest_measurement
            ),
            "product_authority_claimed": False,
        },
    )
    seal = replace(seal, motion_source_seal_hash=motion_source_seal_hash(seal))
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": (
            "PASS_DEMO_ONLY"
            if demo_rest_measurement
            else "PASS"
        ),
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
            "strict_rest_precondition_passed": bool(
                strict_rest_pass
            ),
            "demo_only_measured_rest_failure": bool(
                demo_rest_measurement
            ),
            "product_authority_claimed": False,
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
    from compiler.realsas_compiler_core.artifact_codec_v2 import (
        deformation_envelope_from_dict,
    )
    envelope = deformation_envelope_from_dict(envelope_payload)
    presentation = qualified_presentation_graph_from_dict(
        stage_output_payload(
            ctx,
            "38_CANONICAL_PUPPET_SEALED",
            "RealSaS.QualifiedPresentationGraphIR.v2",
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
            "RealSaS.QualifiedPresentationGraphIR.v2",
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
    mesh_policy = _mesh_policy(ctx)
    try:
        proof = build_qualified_dynamic_motion_v2(
            motion=motion,
            constraints=constraints,
            product_state=mechanical,
            skeleton=skeleton,
            mesh=mesh,
            mesh_skin=mesh_skin,
            presentation=presentation,
            mesh_policy=mesh_policy,
            cameras=cameras.cameras,
            source_foreground_masks=_source_foreground_masks(ctx, observation),
        )
    except DynamicMotionProofFailure as exc:
        measurements=dict(exc.measurements)
        measurements.setdefault(
            "frozen_policy_thresholds",
            {
                "g3_min_dynamic_area_ratio":float(mesh_policy.g3_min_dynamic_area_ratio),
                "g3_max_dynamic_area_ratio":float(mesh_policy.g3_max_dynamic_area_ratio),
                "g3_max_dynamic_condition_number":float(
                    mesh_policy.g3_max_dynamic_condition_number
                ),
            },
        )
        signatures=derive_failure_signatures(
            "MOTION",
            measurements,
            status="FAIL",
        )
        bindings={
            "mechanical_state_binding_hash":mechanical.product_state_hash,
            "skeleton_binding_hash":skeleton.skeleton_lineage_hash,
            "mesh_binding_hash":mesh.mesh_lineage_hash,
            "mesh_skin_binding_hash":mesh_skin.mesh_skin_lineage_hash,
            "qualified_motion_binding_hash":motion.motion_lineage_hash,
            "constraint_set_binding_hash":constraints.constraint_set_hash,
            "presentation_binding_hash":presentation.presentation_lineage_hash,
        }
        attribution_context=build_stage41_failure_attribution_context_v1(
            measurements=measurements,
            bindings=bindings,
            mesh_policy_hash=mesh_policy.qualification_policy_lineage_hash,
            camera_binding_hashes=cameras.camera_binding_hashes,
            observation_set_hash=observation.observation_set_hash,
            evaluator_semantic_version=DYNAMIC_EVALUATOR_SEMANTIC_VERSION_V2,
        )
        return {
            "status":"FAIL",
            "blockers":[exc.failure_code],
            "diagnostics":{
                "failure_code":exc.failure_code,
                "proof_domain":"MOTION",
                "stage41_exact_motion":True,
                "measurements":measurements,
                "failure_signatures":list(signatures),
                "owner_attribution":list(no_owner_attribution()),
                "causal_owner_attribution":"NOT_PERFORMED",
                "owner_attribution_requires_controlled_counterfactual":True,
                "repair_authorized":False,
                "same_probe_reproof_required":True,
                "bindings":bindings,
                "owner_attribution_context":attribution_context,
            },
        }
    if proof.dynamic_motion_hash != qualified_dynamic_motion_v2_hash(proof):
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
