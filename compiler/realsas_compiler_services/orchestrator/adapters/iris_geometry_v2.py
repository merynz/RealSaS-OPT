from __future__ import annotations

"""RealSaS V2 geometry lane: IRIS -> dense surface -> geometry substrate -> GSA."""

from dataclasses import asdict, replace
from hashlib import sha256
import math

import numpy as np

from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)

from compiler.realsas_compiler_core.geometry_substrate_v2 import (
    GeometrySubstrateViewIR,
    build_geometry_substrate_qualification,
    geometry_substrate_from_dict,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics,
    rasterize_triangles_half_integer_top_left,
    source_connected_component_recall_metrics,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import project_points_xyz_v3
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    ModelCheckpointSealIR,
    RiggingSurfaceQualificationIR,
    SignedZeroSurfaceSealIR,
    model_checkpoint_seal_from_dict,
    model_checkpoint_seal_hash,
    model_fit_execution_from_dict,
    model_fit_preregistration_from_dict,
    normalization_domain_from_dict,
    rigging_surface_qualification_hash,
    signed_zero_surface_from_dict,
    signed_zero_surface_hash,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.silhouette_metrics_v2 import (
    silhouette_distance_metrics,
)
from compiler.realsas_compiler_core.substrate.adequacy_v1 import (
    select_adequate_rigging_surface_v1,
    substrate_adequacy_report_hash_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    load_file_ref,
    resolved_path,
    sha256_file,
    stage_output_payload,
    write_ir,
    write_json,
)
from compiler.realsas_compiler_services.orchestrator.adapters.model_execution_common_v2 import (
    build_fit_execution,
    build_fit_preregistration,
    seal_model_checkpoint,
)


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return content_sha256(
        {
            "dtype": str(value.dtype),
            "shape": tuple(map(int, value.shape)),
            "bytes_sha256": sha256(value.tobytes(order="C")).hexdigest(),
        }
    )


def _load_zero_arrays(seal: SignedZeroSurfaceSealIR):
    path = resolved_path(seal.npz_path)
    if not path.is_file() or sha256_file(path) != seal.npz_sha256:
        raise QualificationError("ZERO_SURFACE_NPZ_BYTES_DRIFT")
    with np.load(path, allow_pickle=False) as data:
        required = {"vertices_normalized", "faces", "implicit_normals"}
        if not required.issubset(set(data.files)):
            raise QualificationError("ZERO_SURFACE_NPZ_ARRAYS_MISSING")
        vertices = np.asarray(data["vertices_normalized"]).copy()
        faces = np.asarray(data["faces"]).copy()
        normals = np.asarray(data["implicit_normals"]).copy()
    if (
        _array_hash(vertices) != seal.vertices_sha256
        or _array_hash(faces) != seal.faces_sha256
        or _array_hash(normals) != seal.implicit_normals_sha256
    ):
        raise QualificationError("ZERO_SURFACE_ARRAY_HASH_DRIFT")
    return vertices, faces, normals


_IRIS_DENSE_COVERAGE_POLICY_SCHEMA = "RealSaS.IRISDenseSourceCoveragePolicy.v2"
_IRIS_DENSE_COVERAGE_POLICY_AUTHORITY = (
    "IRIS_V2_TRAINING_OBJECTIVE_POLICY__NO_KNIGHT_OR_MAGE_RESULT"
)


def _iris_dense_coverage_policy(ctx: dict):
    cfg = dict(ctx["run_manifest"].get("iris_fit") or {})
    ref = dict(cfg.get("dense_source_coverage_policy") or {})
    payload = load_file_ref(ref, expected_schema=_IRIS_DENSE_COVERAGE_POLICY_SCHEMA)
    path = resolved_path(str(ref.get("path") or ""))
    digest = sha256_file(path)
    if str(payload.get("status") or "") != "FROZEN_PRE_KNIGHT_SUBJECT_FREE":
        raise QualificationError("IRIS_DENSE_COVERAGE_POLICY_NOT_FROZEN")
    if str(payload.get("authority") or "") != _IRIS_DENSE_COVERAGE_POLICY_AUTHORITY:
        raise QualificationError("IRIS_DENSE_COVERAGE_POLICY_AUTHORITY_DRIFT")
    if bool(payload.get("mask_is_forward_input", True)):
        raise QualificationError("IRIS_DENSE_COVERAGE_MASK_FORWARD_INPUT_FORBIDDEN")
    if int(payload.get("target_resolution", 0)) != 1024:
        raise QualificationError("IRIS_DENSE_COVERAGE_RESOLUTION_DRIFT")
    if str(payload.get("production_q_domain_authority") or "") != "RGB_CAMERA_FULL_FRAME_LATTICE_V1":
        raise QualificationError("IRIS_DENSE_COVERAGE_Q_DOMAIN_AUTHORITY_DRIFT")
    schedule = dict(payload.get("view_schedule") or {})
    if (
        str(schedule.get("mode") or "") != "ROUND_ROBIN_V0_TO_V7"
        or not bool(schedule.get("all_views_required", False))
        or int(schedule.get("maximum_total_step_count_imbalance", -1)) != 1
    ):
        raise QualificationError("IRIS_DENSE_COVERAGE_VIEW_SCHEDULE_DRIFT")
    return ref, digest, payload


def _validate_dense_coverage_prereg(prereg, *, policy_sha256: str, policy: dict) -> None:
    qualification = dict(prereg.qualification_policy or {})
    required = {
        "dense_source_coverage_required": True,
        "dense_source_coverage_policy_sha256": str(policy_sha256),
        "dense_source_coverage_target_resolution": int(policy["target_resolution"]),
        "dense_source_coverage_view_schedule": str(
            policy["view_schedule"]["mode"]
        ),
        "dense_source_coverage_top_level_loss_weight": float(
            policy["loss"]["top_level_loss_weight"]
        ),
    }
    for key, expected in required.items():
        actual = qualification.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-12:
                raise QualificationError(
                    f"IRIS_DENSE_COVERAGE_PREREG_POLICY_DRIFT:{key}"
                )
        elif actual != expected:
            raise QualificationError(
                f"IRIS_DENSE_COVERAGE_PREREG_POLICY_DRIFT:{key}"
            )


def _validate_dense_coverage_execution_metadata(
    metadata: dict,
    *,
    policy_sha256: str,
    policy: dict,
) -> dict:
    dense = dict(metadata.get("dense_source_coverage") or {})
    if str(dense.get("policy_sha256") or "") != str(policy_sha256):
        raise QualificationError("IRIS_DENSE_COVERAGE_EXECUTION_POLICY_DRIFT")
    if bool(dense.get("source_foreground_masks_forward_input", True)):
        raise QualificationError("IRIS_DENSE_COVERAGE_EXECUTION_MASK_LEAK")
    if int(dense.get("target_resolution", 0)) != int(policy["target_resolution"]):
        raise QualificationError("IRIS_DENSE_COVERAGE_EXECUTION_RESOLUTION_DRIFT")
    if str(dense.get("q_domain_authority") or "") != str(
        policy["production_q_domain_authority"]
    ):
        raise QualificationError("IRIS_DENSE_COVERAGE_EXECUTION_Q_DOMAIN_DRIFT")
    if str(dense.get("view_schedule") or "") != str(
        policy["view_schedule"]["mode"]
    ):
        raise QualificationError("IRIS_DENSE_COVERAGE_EXECUTION_VIEW_SCHEDULE_DRIFT")
    counts = tuple(int(value) for value in dense.get("anchor_view_step_counts") or ())
    if len(counts) != 8 or min(counts, default=0) <= 0:
        raise QualificationError("IRIS_DENSE_COVERAGE_EXECUTION_VIEW_COVERAGE_INCOMPLETE")
    if max(counts) - min(counts) > int(
        policy["view_schedule"]["maximum_total_step_count_imbalance"]
    ):
        raise QualificationError("IRIS_DENSE_COVERAGE_EXECUTION_VIEW_IMBALANCE")
    terms = set(map(str, dense.get("reported_loss_terms") or ()))
    if "dense_source_coverage" not in terms:
        raise QualificationError("IRIS_DENSE_COVERAGE_EXECUTION_LOSS_TERM_MISSING")
    return {
        "anchor_view_step_counts": counts,
        "total_dense_supervised_steps": int(sum(counts)),
        "maximum_step_imbalance": int(max(counts) - min(counts)),
    }



_DEMO_IRIS_PREREG_SCHEMA = "RealSaS.DemoHistoricalIrisPreregEvidence.v1"
_DEMO_IRIS_EXEC_SCHEMA = "RealSaS.DemoHistoricalIrisExecutionEvidence.v1"


def _demo_frozen_iris_import_cfg(ctx: dict) -> dict | None:
    if str(ctx["ledger"].get("execution_class") or "") != "DEMO_WITNESS":
        return None
    cfg = dict(ctx["run_manifest"].get("iris_fit") or {})
    demo = dict(cfg.get("demo_frozen_import") or {})
    if not demo:
        return None
    if demo.get("enabled") is not True:
        raise QualificationError("DEMO_IRIS_IMPORT_NOT_EXPLICITLY_ENABLED")
    if demo.get("product_authority_claimed") is not False:
        raise QualificationError("DEMO_IRIS_IMPORT_PRODUCT_AUTHORITY_FORBIDDEN")
    return demo


def _demo_frozen_zero_import_cfg(ctx: dict) -> dict | None:
    if str(ctx["ledger"].get("execution_class") or "") != "DEMO_WITNESS":
        return None
    cfg = dict(ctx["run_manifest"].get("geometry_decode") or {})
    demo = dict(cfg.get("demo_frozen_import") or {})
    if not demo:
        return None
    if demo.get("enabled") is not True:
        raise QualificationError("DEMO_ZERO_IMPORT_NOT_EXPLICITLY_ENABLED")
    if demo.get("product_authority_claimed") is not False:
        raise QualificationError("DEMO_ZERO_IMPORT_PRODUCT_AUTHORITY_FORBIDDEN")
    return demo


def _demo_ref_payload(ref: dict, *, expected_schema: str | None = None):
    payload = load_file_ref(
        dict(ref),
        expected_schema=expected_schema,
    )
    path = resolved_path(str(ref.get("path") or ""))
    return path, sha256_file(path), payload


def _demo_evidence_hash(payload: dict, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return content_sha256(body)


def preregister_iris_fit_stage(ctx: dict) -> dict:
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    normalization = normalization_domain_from_dict(
        stage_output_payload(
            ctx,
            "08_NORMALIZATION_DOMAIN_QUALIFIED",
            "RealSaS.NormalizationDomainIR.v1",
        )
    )

    demo = _demo_frozen_iris_import_cfg(ctx)
    if demo is not None:
        prereg_ref = dict(demo.get("research_preregistration") or {})
        closure_ref = dict(demo.get("research_closure") or {})
        prereg_path, prereg_sha, research = _demo_ref_payload(
            prereg_ref,
            expected_schema="RealSaS.IRIS.V5TP64LongHorizonLowLRPreregistration.v1",
        )
        closure_path, closure_sha, closure = _demo_ref_payload(
            closure_ref,
            expected_schema="RealSaS.IRIS.V5TP64LongHorizonLowLRClosure.v1",
        )
        frozen = dict(research.get("frozen_inputs") or {})
        if str(frozen.get("observation_set_binding_hash") or "") != observation.observation_set_hash:
            raise QualificationError("DEMO_IRIS_IMPORT_OBSERVATION_BINDING_DRIFT")
        if str(frozen.get("normalization_binding_hash") or "") != normalization.normalization_hash:
            raise QualificationError("DEMO_IRIS_IMPORT_NORMALIZATION_BINDING_DRIFT")
        architecture = str((research.get("architecture") or {}).get("id") or "")
        if not architecture:
            raise QualificationError("DEMO_IRIS_IMPORT_ARCHITECTURE_MISSING")
        arm = dict((closure.get("arms") or {}).get("C") or {})
        selected = str(arm.get("selected_checkpoint_sha256") or "")
        expected_checkpoint = str((demo.get("checkpoint") or {}).get("sha256") or "")
        if selected != expected_checkpoint or len(selected) != 64:
            raise QualificationError("DEMO_IRIS_IMPORT_SELECTED_CHECKPOINT_DRIFT")

        payload = {
            "schema": _DEMO_IRIS_PREREG_SCHEMA,
            "lane": "IRIS",
            "status": "PASS_DEMO_ONLY",
            "research_preregistration_path": str(prereg_path),
            "research_preregistration_sha256": prereg_sha,
            "research_closure_path": str(closure_path),
            "research_closure_sha256": closure_sha,
            "architecture_id": architecture,
            "historical_run_id": str(closure.get("run_id") or ""),
            "selected_arm": "C_DIRECT_FSTAR_PLUS_SOURCE_SILHOUETTE",
            "selected_checkpoint_sha256": selected,
            "observation_set_binding_hash": observation.observation_set_hash,
            "normalization_binding_hash": normalization.normalization_hash,
            "stage08_evidence_sha256": str(frozen.get("stage08_evidence_sha256") or ""),
            "preexecution_research_preregistration_verified": True,
            "current_product_fit_preregistration_claimed": False,
            "product_authority_claimed": False,
            "evidence_hash": "",
        }
        payload["evidence_hash"] = _demo_evidence_hash(payload, "evidence_hash")
        root = ctx["run_root"] / "artifacts" / "09_IRIS_FIT_PREREGISTERED"
        return {
            "status": "PASS",
            "outputs": [
                write_json(
                    root / "demo_historical_iris_prereg_evidence.json",
                    payload,
                    authority_class="DEMO_HISTORICAL_IRIS_PREREG_EVIDENCE",
                    schema=_DEMO_IRIS_PREREG_SCHEMA,
                )
            ],
            "diagnostics": {
                "demo_historical_import": True,
                "evidence_hash": payload["evidence_hash"],
                "selected_checkpoint_sha256": selected,
                "product_authority_claimed": False,
            },
        }

    _dense_ref, dense_policy_sha, dense_policy = _iris_dense_coverage_policy(ctx)
    prereg, output = build_fit_preregistration(
        ctx,
        lane="IRIS",
        section_key="iris_fit",
        upstream_bindings={
            "observation_set": observation.observation_set_hash,
            "normalization": normalization.normalization_hash,
        },
        stage_id="09_IRIS_FIT_PREREGISTERED",
    )
    _validate_dense_coverage_prereg(
        prereg,
        policy_sha256=dense_policy_sha,
        policy=dense_policy,
    )
    return {
        "status": "PASS",
        "outputs": [output],
        "diagnostics": {
            "preregistration_hash": prereg.preregistration_hash,
            "architecture_id": prereg.architecture_id,
            "executor_kind": prereg.executor_kind,
            "geometry_role": "SIGNED_SURFACE_EVIDENCE_NOT_APPEARANCE_AUTHORITY",
            "dense_source_coverage_policy_sha256": dense_policy_sha,
            "dense_source_coverage_required": True,
        },
    }


def execute_iris_fit_stage(ctx: dict) -> dict:
    demo = _demo_frozen_iris_import_cfg(ctx)
    if demo is not None:
        prereg = stage_output_payload(
            ctx,
            "09_IRIS_FIT_PREREGISTERED",
            _DEMO_IRIS_PREREG_SCHEMA,
        )
        if prereg.get("evidence_hash") != _demo_evidence_hash(prereg, "evidence_hash"):
            raise QualificationError("DEMO_IRIS_PREREG_EVIDENCE_HASH_DRIFT")

        checkpoint_ref = dict(demo.get("checkpoint") or {})
        result_ref = dict(demo.get("result") or {})
        selected_ref = dict(demo.get("selected_checkpoint_report") or {})
        checkpoint_path = load_file_ref(checkpoint_ref, json_required=False)
        result_path, result_sha, result = _demo_ref_payload(
            result_ref,
            expected_schema="RealSaS.IRIS.V5TP64LongHorizonLowLR.v1",
        )
        selected_path, selected_sha, selected = _demo_ref_payload(selected_ref)
        checkpoint_sha = sha256_file(checkpoint_path)
        if checkpoint_sha != str(prereg.get("selected_checkpoint_sha256") or ""):
            raise QualificationError("DEMO_IRIS_EXECUTION_CHECKPOINT_DRIFT")
        if str(selected.get("selected_checkpoint_sha256") or "") != checkpoint_sha:
            raise QualificationError("DEMO_IRIS_SELECTED_REPORT_CHECKPOINT_DRIFT")
        if int(selected.get("selected_absolute_effective_step", -1)) != 19200:
            raise QualificationError("DEMO_IRIS_SELECTED_REPORT_STEP_DRIFT")
        if str(result.get("run_id") or "") != str(prereg.get("historical_run_id") or ""):
            raise QualificationError("DEMO_IRIS_RESULT_RUN_ID_DRIFT")
        if str((result.get("foundation") or {}).get("architecture_id") or "") != str(
            prereg.get("architecture_id") or ""
        ):
            raise QualificationError("DEMO_IRIS_RESULT_ARCHITECTURE_DRIFT")
        arm = dict((result.get("arms") or {}).get("C_DIRECT_FSTAR_PLUS_SOURCE_SILHOUETTE") or {})
        if str(arm.get("selected_checkpoint_sha256") or "") != checkpoint_sha:
            raise QualificationError("DEMO_IRIS_RESULT_SELECTED_CHECKPOINT_DRIFT")

        payload = {
            "schema": _DEMO_IRIS_EXEC_SCHEMA,
            "lane": "IRIS",
            "status": "PASS_DEMO_ONLY",
            "prereg_evidence_hash": str(prereg["evidence_hash"]),
            "architecture_id": str(prereg["architecture_id"]),
            "historical_run_id": str(prereg["historical_run_id"]),
            "selected_arm": str(prereg["selected_arm"]),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "result_path": str(result_path),
            "result_sha256": result_sha,
            "selected_checkpoint_report_path": str(selected_path),
            "selected_checkpoint_report_sha256": selected_sha,
            "selected_absolute_effective_step": int(selected["selected_absolute_effective_step"]),
            "optimizer_state_saved": bool(selected.get("optimizer_state_saved", False)),
            "scheduler_state_saved": bool(selected.get("scheduler_state_saved", False)),
            "teacher_training_supervision_used": True,
            "teacher_inference_inputs_used": False,
            "current_adapter_fit_executed": False,
            "historical_external_gpu_fit_evidence_admitted": True,
            "product_authority_claimed": False,
            "evidence_hash": "",
        }
        if not payload["optimizer_state_saved"] or not payload["scheduler_state_saved"]:
            raise QualificationError("DEMO_IRIS_SELECTED_STATE_NOT_FULLY_SEALED")
        payload["evidence_hash"] = _demo_evidence_hash(payload, "evidence_hash")
        root = ctx["run_root"] / "artifacts" / "10_IRIS_FIT"
        return {
            "status": "PASS",
            "outputs": [
                write_json(
                    root / "demo_historical_iris_execution_evidence.json",
                    payload,
                    authority_class="DEMO_HISTORICAL_IRIS_EXECUTION_EVIDENCE",
                    schema=_DEMO_IRIS_EXEC_SCHEMA,
                )
            ],
            "diagnostics": {
                "demo_historical_import": True,
                "checkpoint_sha256": checkpoint_sha,
                "result_sha256": result_sha,
                "evidence_hash": payload["evidence_hash"],
                "product_authority_claimed": False,
            },
        }

    _dense_ref, dense_policy_sha, dense_policy = _iris_dense_coverage_policy(ctx)
    prereg = model_fit_preregistration_from_dict(
        stage_output_payload(
            ctx,
            "09_IRIS_FIT_PREREGISTERED",
            "RealSaS.ModelFitPreregistrationIR.v1",
        )
    )
    _validate_dense_coverage_prereg(
        prereg,
        policy_sha256=dense_policy_sha,
        policy=dense_policy,
    )
    execution, output = build_fit_execution(
        ctx,
        lane="IRIS",
        section_key="iris_fit",
        prereg=prereg,
        stage_id="10_IRIS_FIT",
        proposal_required=False,
    )
    dense_execution = _validate_dense_coverage_execution_metadata(
        dict(execution.metadata.get("receipt_metadata") or {}),
        policy_sha256=dense_policy_sha,
        policy=dense_policy,
    )
    return {
        "status": "PASS",
        "outputs": [output],
        "diagnostics": {
            "execution_hash": execution.execution_hash,
            "checkpoint_sha256": execution.checkpoint_sha256,
            "result_sha256": execution.result_sha256,
            "product_authority_minted": False,
            "dense_source_coverage_policy_sha256": dense_policy_sha,
            "dense_source_coverage_execution": dense_execution,
        },
    }


def seal_iris_checkpoint_stage(ctx: dict) -> dict:
    demo = _demo_frozen_iris_import_cfg(ctx)
    if demo is not None:
        execution = stage_output_payload(
            ctx,
            "10_IRIS_FIT",
            _DEMO_IRIS_EXEC_SCHEMA,
        )
        if execution.get("evidence_hash") != _demo_evidence_hash(execution, "evidence_hash"):
            raise QualificationError("DEMO_IRIS_EXECUTION_EVIDENCE_HASH_DRIFT")
        checkpoint_path = resolved_path(str(execution.get("checkpoint_path") or ""))
        result_path = resolved_path(str(execution.get("result_path") or ""))
        if (
            not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != str(execution.get("checkpoint_sha256") or "")
        ):
            raise QualificationError("DEMO_IRIS_CHECKPOINT_BYTES_DRIFT")
        if (
            not result_path.is_file()
            or sha256_file(result_path) != str(execution.get("result_sha256") or "")
        ):
            raise QualificationError("DEMO_IRIS_RESULT_BYTES_DRIFT")
        value = ModelCheckpointSealIR(
            "IRIS",
            str(execution["evidence_hash"]),
            str(checkpoint_path),
            str(execution["checkpoint_sha256"]),
            str(execution["result_sha256"]),
            None,
            "",
            metadata={
                "architecture_id": str(execution["architecture_id"]),
                "historical_demo_import": True,
                "current_adapter_fit_executed": False,
                "product_authority_claimed": False,
                "historical_external_gpu_fit_evidence_hash": str(execution["evidence_hash"]),
            },
        )
        value = replace(
            value,
            checkpoint_seal_hash=model_checkpoint_seal_hash(value),
        )
        root = ctx["run_root"] / "artifacts" / "11_IRIS_CHECKPOINT_SEALED"
        output = write_ir(
            root / "model_checkpoint_seal.json",
            value,
            authority_class="DEMO_IRIS_CHECKPOINT_SEAL",
        )
        return {
            "status": "PASS",
            "outputs": [output],
            "diagnostics": {
                "checkpoint_seal_hash": value.checkpoint_seal_hash,
                "checkpoint_sha256": value.checkpoint_sha256,
                "historical_demo_import": True,
                "product_authority_claimed": False,
            },
        }

    execution = model_fit_execution_from_dict(
        stage_output_payload(
            ctx,
            "10_IRIS_FIT",
            "RealSaS.ModelFitExecutionIR.v1",
        )
    )
    value, output = seal_model_checkpoint(
        ctx,
        execution=execution,
        qualified_output_binding_hash=None,
        stage_id="11_IRIS_CHECKPOINT_SEALED",
    )
    return {
        "status": "PASS",
        "outputs": [output],
        "diagnostics": {
            "checkpoint_seal_hash": value.checkpoint_seal_hash,
            "checkpoint_sha256": value.checkpoint_sha256,
        },
    }


def decode_zero_surface_stage(ctx: dict) -> dict:
    checkpoint = model_checkpoint_seal_from_dict(
        stage_output_payload(
            ctx,
            "11_IRIS_CHECKPOINT_SEALED",
            "RealSaS.ModelCheckpointSealIR.v1",
        )
    )
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    normalization = normalization_domain_from_dict(
        stage_output_payload(
            ctx,
            "08_NORMALIZATION_DOMAIN_QUALIFIED",
            "RealSaS.NormalizationDomainIR.v1",
        )
    )

    demo = _demo_frozen_zero_import_cfg(ctx)
    if demo is not None:
        npz_ref = dict(demo.get("zero_surface_npz") or {})
        decoder_ref = dict(demo.get("decoder_result") or {})
        npz_path = load_file_ref(npz_ref, json_required=False)
        decoder_path, decoder_sha, decoder = _demo_ref_payload(decoder_ref)
        npz_sha = sha256_file(npz_path)
        if str(decoder.get("arm") or "") != "C_DIRECT_FSTAR_PLUS_SOURCE_SILHOUETTE":
            raise QualificationError("DEMO_ZERO_SURFACE_ARM_DRIFT")
        if str(decoder.get("decoder_id") or "") != (
            "RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT."
            "ExactShellContainment.v5_1"
        ):
            raise QualificationError("DEMO_ZERO_SURFACE_DECODER_DRIFT")
        containment = dict(decoder.get("containment") or {})
        if containment.get("passed") is not True:
            raise QualificationError("DEMO_ZERO_SURFACE_CONTAINMENT_NOT_PASS")
        mesh = dict(decoder.get("mesh") or {})
        if str(mesh.get("npz_sha256") or "") != npz_sha:
            raise QualificationError("DEMO_ZERO_SURFACE_NPZ_BINDING_DRIFT")
        diagnostics = dict(mesh.get("diagnostics") or {})
        normal_diag = dict(diagnostics.get("implicit_normal_diagnostics") or {})
        if bool(normal_diag.get("teacher_truth_used", True)):
            raise QualificationError("DEMO_ZERO_SURFACE_TEACHER_TRUTH_USED")

        with np.load(npz_path, allow_pickle=False) as data:
            required = {"vertices_normalized", "faces", "implicit_normals"}
            if not required.issubset(set(data.files)):
                raise QualificationError("ZERO_SURFACE_NPZ_ARRAYS_MISSING")
            vertices = np.asarray(data["vertices_normalized"]).copy()
            faces = np.asarray(data["faces"]).copy()
            normals = np.asarray(data["implicit_normals"]).copy()
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) < 4
            or not np.isfinite(vertices).all()
            or faces.ndim != 2
            or faces.shape[1] != 3
            or len(faces) == 0
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0)
            or np.any(faces >= len(vertices))
            or normals.shape != vertices.shape
            or not np.isfinite(normals).all()
            or np.any(np.linalg.norm(normals, axis=1) <= 1e-12)
        ):
            raise QualificationError("DEMO_ZERO_SURFACE_ARRAYS_INVALID")
        vertices_sha = _array_hash(vertices)
        faces_sha = _array_hash(faces)
        normals_sha = _array_hash(normals)
        expected_arrays = {
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "vertices_sha256": vertices_sha,
            "faces_sha256": faces_sha,
            "implicit_normals_sha256": normals_sha,
        }
        for key, actual in expected_arrays.items():
            expected = mesh.get(key)
            if isinstance(actual, int):
                if int(expected or -1) != actual:
                    raise QualificationError(f"DEMO_ZERO_SURFACE_MESH_DRIFT:{key}")
            elif str(expected or "") != actual:
                raise QualificationError(f"DEMO_ZERO_SURFACE_MESH_DRIFT:{key}")

        value = SignedZeroSurfaceSealIR(
            checkpoint.checkpoint_seal_hash,
            observation.observation_set_hash,
            normalization.normalization_hash,
            str(npz_path),
            npz_sha,
            str(decoder_path),
            decoder_sha,
            int(len(vertices)),
            int(len(faces)),
            vertices_sha,
            faces_sha,
            normals_sha,
            "",
            metadata={
                "decoder_id": str(decoder["decoder_id"]),
                "teacher_truth_used": False,
                "historical_demo_import": True,
                "decoder_evidence_schema": "V5_LONG_HORIZON_DECODER_RESULT",
                "exact_containment_passed": True,
                "normalized_coordinate_domain": "[-1,1]^3",
                "v2_role": "DENSE_GEOMETRY_EVIDENCE",
                "product_authority_claimed": False,
            },
        )
        value = replace(value, zero_surface_hash=signed_zero_surface_hash(value))
        root = ctx["run_root"] / "artifacts" / "12_ZERO_SURFACE_DECODED"
        return {
            "status": "PASS",
            "outputs": [
                write_ir(
                    root / "signed_zero_surface_seal.json",
                    value,
                    authority_class="DEMO_IRIS_SIGNED_ZERO_SURFACE_SEAL",
                )
            ],
            "diagnostics": {
                "zero_surface_hash": value.zero_surface_hash,
                "vertex_count": value.vertex_count,
                "face_count": value.face_count,
                "historical_demo_import": True,
                "product_authority_claimed": False,
            },
        }

    cfg = dict(ctx["run_manifest"].get("geometry_decode") or {})
    npz_ref = dict(cfg.get("zero_surface_npz") or {})
    metadata_ref = dict(cfg.get("zero_surface_metadata") or {})
    npz_path = load_file_ref(npz_ref, json_required=False)
    metadata = load_file_ref(
        metadata_ref,
        expected_schema="RealSaS.SignedZeroSurfaceArtifact.v1",
    )
    metadata_path = resolved_path(str(metadata_ref.get("path") or ""))
    npz_sha = sha256_file(npz_path)
    metadata_sha = sha256_file(metadata_path)

    expected = {
        "checkpoint_seal_binding_hash": checkpoint.checkpoint_seal_hash,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "observation_set_binding_hash": observation.observation_set_hash,
        "normalization_binding_hash": normalization.normalization_hash,
        "npz_sha256": npz_sha,
    }
    for key, wanted in expected.items():
        if str(metadata.get(key) or "") != wanted:
            raise QualificationError(f"ZERO_SURFACE_METADATA_BINDING_DRIFT:{key}")
    if bool(metadata.get("teacher_truth_used", False)):
        raise QualificationError("ZERO_SURFACE_TEACHER_TRUTH_FORBIDDEN")

    with np.load(npz_path, allow_pickle=False) as data:
        required = {"vertices_normalized", "faces", "implicit_normals"}
        if not required.issubset(set(data.files)):
            raise QualificationError("ZERO_SURFACE_NPZ_ARRAYS_MISSING")
        vertices = np.asarray(data["vertices_normalized"]).copy()
        faces = np.asarray(data["faces"]).copy()
        normals = np.asarray(data["implicit_normals"]).copy()

    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 4
        or not np.isfinite(vertices).all()
    ):
        raise QualificationError("ZERO_SURFACE_VERTICES_INVALID")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) == 0
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        raise QualificationError("ZERO_SURFACE_FACES_INVALID")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise QualificationError("ZERO_SURFACE_FACE_INDEX_INVALID")
    if (
        normals.shape != vertices.shape
        or not np.isfinite(normals).all()
        or np.any(np.linalg.norm(normals, axis=1) <= 1e-12)
    ):
        raise QualificationError("ZERO_SURFACE_NORMALS_INVALID")

    value = SignedZeroSurfaceSealIR(
        checkpoint.checkpoint_seal_hash,
        observation.observation_set_hash,
        normalization.normalization_hash,
        str(npz_path),
        npz_sha,
        str(metadata_path),
        metadata_sha,
        int(len(vertices)),
        int(len(faces)),
        _array_hash(vertices),
        _array_hash(faces),
        _array_hash(normals),
        "",
        metadata={
            "decoder_id": str(
                metadata.get("decoder_id")
                or "IRIS_SIGNED_ZERO_SURFACE_EXTERNAL_DECODE_V1"
            ),
            "teacher_truth_used": False,
            "normalized_coordinate_domain": "[-1,1]^3",
            "v2_role": "DENSE_GEOMETRY_EVIDENCE",
        },
    )
    value = replace(value, zero_surface_hash=signed_zero_surface_hash(value))
    root = ctx["run_root"] / "artifacts" / "12_ZERO_SURFACE_DECODED"
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "signed_zero_surface_seal.json",
                value,
                authority_class="IRIS_SIGNED_ZERO_SURFACE_SEAL",
            )
        ],
        "diagnostics": {
            "zero_surface_hash": value.zero_surface_hash,
            "vertex_count": value.vertex_count,
            "face_count": value.face_count,
        },
    }


def _source_foreground(ctx: dict, observation):
    cfg = dict(ctx["run_manifest"].get("observation") or {})
    rows = tuple(cfg.get("source_foreground_masks") or ())
    if len(rows) != 8 or {int(row["view_index"]) for row in rows} != set(range(8)):
        raise QualificationError("GEOMETRY_SUBSTRATE_FOREGROUND_MATRIX_INCOMPLETE")
    authority = {int(view.view_index): view for view in observation.views}
    output = {}
    for row in rows:
        view_index = int(row["view_index"])
        ref = dict(row.get("mask") or {})
        path = load_file_ref(ref, json_required=False)
        if sha256_file(path) != authority[view_index].foreground_mask_sha256:
            raise QualificationError("GEOMETRY_SUBSTRATE_FOREGROUND_AUTHORITY_DRIFT")
        raw = path.read_bytes()
        expected_size = (
            int(authority[view_index].width) * int(authority[view_index].height)
        )
        if len(raw) != expected_size or any(value not in (0, 1) for value in raw):
            raise QualificationError("GEOMETRY_SUBSTRATE_FOREGROUND_MASK_INVALID")
        output[view_index] = raw
    return output


def _demo_frozen_stage13_import_cfg(ctx: dict) -> dict | None:
    if str(ctx["ledger"].get("execution_class") or "") != "DEMO_WITNESS":
        return None
    cfg = dict(ctx["run_manifest"].get("geometry_gate") or {})
    demo = dict(cfg.get("demo_frozen_stage13_import") or {})
    if not demo:
        return None
    if demo.get("enabled") is not True:
        raise QualificationError("DEMO_STAGE13_IMPORT_NOT_EXPLICITLY_ENABLED")
    if demo.get("product_authority_claimed") is not False:
        raise QualificationError("DEMO_STAGE13_IMPORT_PRODUCT_AUTHORITY_FORBIDDEN")
    return demo


def _geometry_policy(cfg: dict) -> dict | None:
    keys = (
        "min_recall",
        "min_precision",
        "max_largest_coherent_hole_fraction",
        "max_interior_uncovered_fraction",
        "min_component_recall",
        "component_min_foreground_fraction",
        "max_silhouette_edge_p95_px",
    )
    if any(key not in cfg for key in keys):
        return None
    policy = {key: float(cfg[key]) for key in keys}
    if not (
        0 <= policy["min_recall"] <= 1
        and 0 <= policy["min_precision"] <= 1
        and 0 <= policy["max_largest_coherent_hole_fraction"] <= 1
        and 0 <= policy["max_interior_uncovered_fraction"] <= 1
        and 0 <= policy["min_component_recall"] <= 1
        and 0 <= policy["component_min_foreground_fraction"] <= 1
        and math.isfinite(policy["max_silhouette_edge_p95_px"])
        and policy["max_silhouette_edge_p95_px"] >= 0
    ):
        raise QualificationError("GEOMETRY_SUBSTRATE_THRESHOLD_RANGE_INVALID")
    return policy


def qualify_geometry_substrate_stage(ctx: dict) -> dict:
    zero = signed_zero_surface_from_dict(
        stage_output_payload(
            ctx,
            "12_ZERO_SURFACE_DECODED",
            "RealSaS.SignedZeroSurfaceSealIR.v1",
        )
    )
    normalization = normalization_domain_from_dict(
        stage_output_payload(
            ctx,
            "08_NORMALIZATION_DOMAIN_QUALIFIED",
            "RealSaS.NormalizationDomainIR.v1",
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

    cfg = dict(ctx["run_manifest"].get("geometry_gate") or {})
    policy = _geometry_policy(cfg)
    if policy is None:
        return {
            "status": "BLOCKED",
            "blockers": ["GEOMETRY_SUBSTRATE_EXPLICIT_THRESHOLDS_REQUIRED"],
            "diagnostics": {},
        }

    demo_import = _demo_frozen_stage13_import_cfg(ctx)
    if demo_import is not None:
        evidence_ref = dict(demo_import.get("evidence") or {})
        evidence_path, evidence_sha, evidence = _demo_ref_payload(
            evidence_ref,
            expected_schema="RealSaS.IRIS.V5TP64LongHorizon.Stage13Result.v1",
        )
        expected_npz = str(demo_import.get("zero_surface_npz_sha256") or "")
        if expected_npz != zero.npz_sha256:
            raise QualificationError("DEMO_STAGE13_ZERO_SURFACE_BINDING_DRIFT")
        if str(evidence.get("arm") or "") != "C_DIRECT_FSTAR_PLUS_SOURCE_SILHOUETTE":
            raise QualificationError("DEMO_STAGE13_ARM_DRIFT")
        if evidence.get("stage13_v2_pass") is not False:
            raise QualificationError("DEMO_STAGE13_SCIENTIFIC_STATE_DRIFT")
        evidence_policy = {
            str(key): float(value)
            for key, value in dict(evidence.get("policy") or {}).items()
        }
        if content_sha256(evidence_policy) != content_sha256(policy):
            raise QualificationError("DEMO_STAGE13_POLICY_DRIFT")

        source = _source_foreground(ctx, observation)
        evidence_rows = tuple(evidence.get("per_view") or ())
        if (
            len(evidence_rows) != 8
            or {int(row.get("view_index", -1)) for row in evidence_rows} != set(range(8))
        ):
            raise QualificationError("DEMO_STAGE13_VIEW_SET_INVALID")
        if str(evidence.get("policy_profile") or "") != "P999":
            raise QualificationError("DEMO_STAGE13_POLICY_PROFILE_DRIFT")
        summary = dict(evidence.get("summary") or {})
        extrema = {
            "min_precision": min(float(row["precision"]) for row in evidence_rows),
            "min_recall": min(float(row["recall"]) for row in evidence_rows),
            "min_component_recall": min(
                float(row["minimum_eligible_component_recall"])
                for row in evidence_rows
            ),
            "max_interior_uncovered_fraction": max(
                float(row["interior_uncovered_fraction"])
                for row in evidence_rows
            ),
            "max_largest_coherent_hole_fraction": max(
                float(row["largest_coherent_hole_fraction"])
                for row in evidence_rows
            ),
            "max_silhouette_edge_p95_px": max(
                float(row["silhouette_edge_p95_px"])
                for row in evidence_rows
            ),
            "max_silhouette_edge_max_px": max(
                float(row["silhouette_edge_max_px"])
                for row in evidence_rows
            ),
        }
        for key, actual in extrema.items():
            if key not in summary or abs(float(summary[key]) - actual) > 1e-15:
                raise QualificationError(
                    f"DEMO_STAGE13_EVIDENCE_SUMMARY_DRIFT:{key}"
                )

        observations = {int(view.view_index): view for view in observation.views}
        rows = []
        for raw in sorted(evidence_rows, key=lambda value: int(value["view_index"])):
            view_index = int(raw["view_index"])
            authority = observations[view_index]
            source_count = int(sum(source[view_index]))
            recall = float(raw["recall"])
            precision = float(raw["precision"])
            if source_count <= 0 or recall <= 0 or precision <= 0:
                raise QualificationError("DEMO_STAGE13_PIXEL_ACCOUNTING_INVALID")
            true_positive = int(round(recall * source_count))
            predicted_count = int(round(true_positive / precision))
            if (
                abs(true_positive / source_count - recall) > 1e-15
                or abs(true_positive / predicted_count - precision) > 1e-15
            ):
                raise QualificationError("DEMO_STAGE13_INTEGER_METRIC_RECONSTRUCTION_DRIFT")

            rows.append(
                GeometrySubstrateViewIR(
                    view_index=view_index,
                    silhouette_recall=recall,
                    silhouette_precision=precision,
                    largest_coherent_hole_fraction=float(
                        raw["largest_coherent_hole_fraction"]
                    ),
                    interior_uncovered_fraction=float(
                        raw["interior_uncovered_fraction"]
                    ),
                    source_foreground_pixel_count=source_count,
                    predicted_foreground_pixel_count=predicted_count,
                    component_recall=float(
                        raw["minimum_eligible_component_recall"]
                    ),
                    silhouette_edge_p95_px=float(raw["silhouette_edge_p95_px"]),
                    passed=bool(raw["passed"]),
                    metadata={
                        "source_observation_hash": authority.source_observation_hash,
                        "silhouette_edge_mean_px": float(
                            raw["silhouette_edge_mean_px"]
                        ),
                        "silhouette_edge_max_px": float(
                            raw["silhouette_edge_max_px"]
                        ),
                        "historical_external_measurement": True,
                        "evidence_sha256": evidence_sha,
                    },
                )
            )

        value = build_geometry_substrate_qualification(
            zero_surface_binding_hash=zero.zero_surface_hash,
            observation_set_binding_hash=observation.observation_set_hash,
            camera_set_binding_hash=cameras.camera_set_hash,
            normalization_binding_hash=normalization.normalization_hash,
            policy=policy,
            views=tuple(rows),
            metadata={
                "metric_contract": "GEOMETRY_SUBSTRATE_V2",
                "historical_external_stage13_measurement_import": True,
                "historical_evidence_path": str(evidence_path),
                "historical_evidence_sha256": evidence_sha,
                "appearance_proxy_forbidden": True,
                "final_visual_fidelity_claimed": False,
                "product_authority_claimed": False,
            },
        )
        if value.qualification_report["every_view_passed"]:
            raise QualificationError("DEMO_STAGE13_IMPORT_UNEXPECTED_SCIENTIFIC_PASS")
        root = ctx["run_root"] / "artifacts" / "13_GEOMETRY_SUBSTRATE_QUALIFIED"
        return {
            "status": "PASS_DEMO_ONLY",
            "outputs": [
                write_ir(
                    root / "geometry_substrate_qualification.json",
                    value,
                    authority_class="DEMO_ONLY_GEOMETRY_SUBSTRATE",
                )
            ],
            "diagnostics": {
                "scientific_pass": False,
                "demo_admitted": True,
                "historical_external_measurement_import": True,
                "evidence_sha256": evidence_sha,
                "per_view": [row.to_dict() for row in rows],
                "substrate_hash": value.substrate_hash,
                "product_authority_claimed": False,
            },
        }

    vertices, faces, _normals = _load_zero_arrays(zero)
    world = (
        np.asarray(normalization.center_xyz, dtype=np.float64)[None, :]
        + np.asarray(vertices, dtype=np.float64) * float(normalization.half_extent)
    )
    source = _source_foreground(ctx, observation)
    observations = {int(view.view_index): view for view in observation.views}
    rows = []
    for camera in sorted(cameras.cameras, key=lambda value: value.view_index):
        view_index = int(camera.view_index)
        authority = observations[view_index]
        projected = project_points_xyz_v3(world, camera)
        if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0.0):
            raise QualificationError(
                "GEOMETRY_SUBSTRATE_OUTSIDE_CAMERA_FORWARD_DOMAIN"
            )
        triangles = []
        for face in np.asarray(faces, dtype=np.int64):
            a, b, c = (projected[int(index)] for index in face)
            triangles.append(
                (
                    (float(a[0]), float(a[1])),
                    (float(b[0]), float(b[1])),
                    (float(c[0]), float(c[1])),
                )
            )
        predicted = rasterize_triangles_half_integer_top_left(
            triangles,
            width=int(authority.width),
            height=int(authority.height),
        )
        metrics = coverage_metrics(
            source[view_index],
            predicted,
            width=int(authority.width),
            height=int(authority.height),
        )
        source_mask = np.frombuffer(
            source[view_index], dtype=np.uint8
        ).reshape(int(authority.height), int(authority.width)).astype(bool)
        predicted_mask = np.frombuffer(
            predicted, dtype=np.uint8
        ).reshape(int(authority.height), int(authority.width)).astype(bool)
        (
            silhouette_edge_mean_px,
            silhouette_edge_p95_px,
            silhouette_edge_max_px,
        ) = silhouette_distance_metrics(source_mask, predicted_mask)
        component_metrics = source_connected_component_recall_metrics(
            source[view_index],
            predicted,
            width=int(authority.width),
            height=int(authority.height),
            minimum_foreground_fraction=policy[
                "component_min_foreground_fraction"
            ],
        )
        if (
            metrics["foreground_pixel_count"] > 0
            and component_metrics["eligible_component_count"] <= 0
        ):
            raise QualificationError(
                "GEOMETRY_SUBSTRATE_COMPONENT_POLICY_SELECTS_NO_FOREGROUND"
            )

        component_recall = float(
            component_metrics["minimum_eligible_component_recall"]
        )
        passed = (
            metrics["recall"] >= policy["min_recall"]
            and metrics["precision"] >= policy["min_precision"]
            and metrics["largest_coherent_hole_fraction"]
            <= policy["max_largest_coherent_hole_fraction"]
            and metrics["interior_uncovered_fraction"]
            <= policy["max_interior_uncovered_fraction"]
            and component_recall >= policy["min_component_recall"]
            and silhouette_edge_p95_px
            <= policy["max_silhouette_edge_p95_px"]
        )
        rows.append(
            GeometrySubstrateViewIR(
                view_index=view_index,
                silhouette_recall=float(metrics["recall"]),
                silhouette_precision=float(metrics["precision"]),
                largest_coherent_hole_fraction=float(
                    metrics["largest_coherent_hole_fraction"]
                ),
                interior_uncovered_fraction=float(
                    metrics["interior_uncovered_fraction"]
                ),
                source_foreground_pixel_count=int(
                    metrics["foreground_pixel_count"]
                ),
                predicted_foreground_pixel_count=int(
                    metrics["predicted_pixel_count"]
                ),
                component_recall=component_recall,
                silhouette_edge_p95_px=float(silhouette_edge_p95_px),
                passed=bool(passed),
                metadata={
                    "source_observation_hash": authority.source_observation_hash,
                    "eligible_component_count": int(
                        component_metrics["eligible_component_count"]
                    ),
                    "silhouette_edge_mean_px": float(
                        silhouette_edge_mean_px
                    ),
                    "silhouette_edge_max_px": float(
                        silhouette_edge_max_px
                    ),
                },
            )
        )

    value = build_geometry_substrate_qualification(
        zero_surface_binding_hash=zero.zero_surface_hash,
        observation_set_binding_hash=observation.observation_set_hash,
        camera_set_binding_hash=cameras.camera_set_hash,
        normalization_binding_hash=normalization.normalization_hash,
        policy=policy,
        views=tuple(rows),
        metadata={
            "raster_fill": "HALF_INTEGER_TOP_LEFT",
            "metric_contract": "GEOMETRY_SUBSTRATE_V2",
            "silhouette_metric": "SYMMETRIC_NEAREST_BOUNDARY_DISTANCE_PIXELS",
            "appearance_proxy_forbidden": True,
            "final_visual_fidelity_claimed": False,
        },
    )
    if not value.qualification_report["every_view_passed"]:
        demo = dict(ctx["run_manifest"].get("demo_execution") or {})
        if (
            str(ctx["ledger"].get("execution_class") or "") == "DEMO_WITNESS"
            and demo.get("allow_stage13_scientific_fail_for_demo") is True
            and demo.get("product_authority_claimed") is False
            and demo.get("stage13_scientific_pass") is False
        ):
            root = ctx["run_root"] / "artifacts" / "13_GEOMETRY_SUBSTRATE_QUALIFIED"
            return {
                "status": "PASS_DEMO_ONLY",
                "outputs": [
                    write_ir(
                        root / "geometry_substrate_qualification.json",
                        value,
                        authority_class="DEMO_ONLY_GEOMETRY_SUBSTRATE",
                    )
                ],
                "diagnostics": {
                    "scientific_pass": False,
                    "demo_admitted": True,
                    "per_view": [row.to_dict() for row in rows],
                    "substrate_hash": value.substrate_hash,
                    "product_authority_claimed": False,
                },
            }
        return {
            "status": "FAIL",
            "blockers": ["GEOMETRY_SUBSTRATE_QUALIFICATION_FAILED"],
            "diagnostics": {
                "per_view": [row.to_dict() for row in rows],
                "substrate_hash": value.substrate_hash,
            },
        }

    root = ctx["run_root"] / "artifacts" / "13_GEOMETRY_SUBSTRATE_QUALIFIED"
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "geometry_substrate_qualification.json",
                value,
                authority_class="QUALIFIED_GEOMETRY_SUBSTRATE",
            )
        ],
        "diagnostics": {
            "substrate_hash": value.substrate_hash,
            "all_views_passed": True,
            "appearance_authority_used": False,
        },
    }


def _validate_gsa_policy_document(cfg: dict, policy_document: dict):
    if (
        str(policy_document.get("schema") or "")
        != "RealSaS.Stage14SubstrateAdequacyPolicy.v2"
    ):
        raise QualificationError("GSA_POLICY_DOCUMENT_SCHEMA_INVALID")
    if not str(policy_document.get("status") or "").startswith("FROZEN_"):
        raise QualificationError("GSA_POLICY_DOCUMENT_NOT_FROZEN")
    document_gsa = dict(policy_document.get("gsa") or {})
    document_adequacy = dict(document_gsa.get("adequacy_policy") or {})

    normal_k = int(cfg["normal_k"])
    tolerance = float(cfg["visibility_depth_tolerance_norm"])
    if normal_k < 3 or not math.isfinite(tolerance) or tolerance <= 0:
        raise QualificationError("GSA_POLICY_INVALID")
    if normal_k != int(document_gsa.get("normal_k", -1)):
        raise QualificationError("GSA_POLICY_DOCUMENT_DRIFT:normal_k")
    if (
        abs(
            tolerance
            - float(document_gsa.get("visibility_depth_tolerance_norm", -1.0))
        )
        > 1e-15
    ):
        raise QualificationError(
            "GSA_POLICY_DOCUMENT_DRIFT:visibility_depth_tolerance_norm"
        )
    if content_sha256(dict(cfg.get("adequacy_policy") or {})) != content_sha256(
        document_adequacy
    ):
        raise QualificationError("GSA_POLICY_DOCUMENT_DRIFT:adequacy_policy")
    return normal_k, tolerance, document_adequacy


def build_gsa_stage(ctx: dict) -> dict:
    zero = signed_zero_surface_from_dict(
        stage_output_payload(
            ctx,
            "12_ZERO_SURFACE_DECODED",
            "RealSaS.SignedZeroSurfaceSealIR.v1",
        )
    )
    geometry = geometry_substrate_from_dict(
        stage_output_payload(
            ctx,
            "13_GEOMETRY_SUBSTRATE_QUALIFIED",
            "RealSaS.GeometrySubstrateQualificationIR.v2",
        )
    )
    normalization = normalization_domain_from_dict(
        stage_output_payload(
            ctx,
            "08_NORMALIZATION_DOMAIN_QUALIFIED",
            "RealSaS.NormalizationDomainIR.v1",
        )
    )
    cameras = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    checkpoint = model_checkpoint_seal_from_dict(
        stage_output_payload(
            ctx,
            "11_IRIS_CHECKPOINT_SEALED",
            "RealSaS.ModelCheckpointSealIR.v1",
        )
    )
    vertices, faces, normals = _load_zero_arrays(zero)
    cfg = dict(ctx["run_manifest"].get("gsa") or {})
    required = (
        "policy_document",
        "normal_k",
        "visibility_depth_tolerance_norm",
        "adequacy_policy",
    )
    if any(key not in cfg for key in required):
        return {
            "status": "BLOCKED",
            "blockers": ["GSA_EXPLICIT_POLICY_REQUIRED"],
            "diagnostics": {"required": list(required)},
        }

    policy_ref = dict(cfg.get("policy_document") or {})
    policy_document = load_file_ref(
        policy_ref,
        expected_schema="RealSaS.Stage14SubstrateAdequacyPolicy.v2",
    )
    normal_k, tolerance, document_adequacy = _validate_gsa_policy_document(
        cfg, policy_document
    )
    camera_dicts = tuple(
        asdict(camera)
        for camera in sorted(cameras.cameras, key=lambda value: value.view_index)
    )
    surface, adequacy = select_adequate_rigging_surface_v1(
        vertices,
        faces,
        normals,
        camera_dicts,
        normalization_center=normalization.center_xyz,
        normalization_half_extent=normalization.half_extent,
        authority_label="IRIS_SCENE_FIRST_SIGNED_ZERO_SURFACE_V2",
        source_run_id=str(ctx["ledger"].get("run_id") or ""),
        source_checkpoint_sha256=checkpoint.checkpoint_sha256,
        source_zero_surface_sha256=zero.npz_sha256,
        normal_k=normal_k,
        visibility_depth_tolerance_norm=tolerance,
        adequacy_policy=document_adequacy,
        metadata={
            "geometry_substrate_hash": geometry.substrate_hash,
            "observation_set_hash": zero.observation_set_binding_hash,
            "stage14_policy_document_path": str(policy_ref.get("path") or ""),
            "stage14_policy_document_sha256": str(
                policy_ref.get("sha256") or ""
            ),
            "appearance_authority_used": False,
        },
    )
    adequacy["policy_document_path"] = str(policy_ref.get("path") or "")
    adequacy["policy_document_sha256"] = str(policy_ref.get("sha256") or "")
    adequacy["geometry_substrate_hash"] = geometry.substrate_hash
    adequacy["adequacy_report_hash"] = ""
    adequacy["adequacy_report_hash"] = substrate_adequacy_report_hash_v1(
        adequacy
    )
    if surface is None:
        return {
            "status": "FAIL",
            "blockers": ["SUBSTRATE_ADEQUACY_NO_PASSING_CANDIDATE"],
            "diagnostics": adequacy,
        }

    root = ctx["run_root"] / "artifacts" / "14_GSA_BUILD"
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "rigging_surface_candidate.json",
                surface,
                authority_class="GSA_RIGGING_SURFACE_CANDIDATE",
            ),
            write_json(
                root / "substrate_adequacy_report.json",
                adequacy,
                authority_class="SUBSTRATE_ADEQUACY_REPORT",
                schema="RealSaS.SubstrateAdequacyReport.v1",
            ),
        ],
        "diagnostics": {
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "node_count": len(surface.surface_nodes),
            "relation_count": len(surface.local_relations),
            "adequacy_report_hash": adequacy["adequacy_report_hash"],
            "selected_target_node_cap": adequacy["selected_target_node_cap"],
        },
    }


def qualify_rigging_surface_stage(ctx: dict) -> dict:
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx,
            "14_GSA_BUILD",
            "RealSaS.RiggingSurfaceIR.v1",
        )
    )
    adequacy = stage_output_payload(
        ctx,
        "14_GSA_BUILD",
        "RealSaS.SubstrateAdequacyReport.v1",
    )
    if (
        adequacy.get("adequacy_report_hash")
        != substrate_adequacy_report_hash_v1(adequacy)
    ):
        raise QualificationError("RIGGING_SURFACE_ADEQUACY_HASH_DRIFT")
    if (
        str(adequacy.get("status")) != "PASS"
        or str(adequacy.get("selected_surface_lineage_hash"))
        != surface.geometry_lineage_hash
    ):
        raise QualificationError("RIGGING_SURFACE_ADEQUACY_BINDING_DRIFT")
    if int(adequacy.get("selected_actual_node_count", -1)) != len(
        surface.surface_nodes
    ):
        raise QualificationError("RIGGING_SURFACE_ADEQUACY_NODE_COUNT_DRIFT")

    tensor = tensorize_rigging_surface_v1(surface, require_scene_first=True)
    if tensor.source_surface_hash != surface.geometry_lineage_hash:
        raise QualificationError("RIGGING_SURFACE_TENSORIZATION_BINDING_DRIFT")
    observed = int(np.count_nonzero(tensor.observed))
    completed = int(np.count_nonzero(tensor.completed))
    if observed <= 0 or observed + completed != tensor.node_count:
        raise QualificationError(
            "RIGGING_SURFACE_OBSERVED_COMPLETED_ACCOUNTING_INVALID"
        )

    value = RiggingSurfaceQualificationIR(
        surface.geometry_lineage_hash,
        tensor.tensorization_hash,
        tensor.certificate_hash,
        tensor.node_count,
        tensor.edge_count,
        observed,
        completed,
        {
            "status": "PASS",
            "scene_first_signed_geometry": True,
            "teacher_truth_contamination_rejected": True,
            "lossless_fieldwise_tensorization_passed": True,
            "local_relation_graph_present": tensor.edge_count > 0,
            "substrate_adequacy_passed": True,
            "substrate_adequacy_report_hash": adequacy["adequacy_report_hash"],
            "appearance_authority_used": False,
        },
        "",
        metadata={
            "tensorization_schema": tensor.schema_version,
            "substrate_adequacy_report_hash": adequacy["adequacy_report_hash"],
            "geometry_substrate_hash": adequacy["geometry_substrate_hash"],
            "v2_geometry_lane": True,
        },
    )
    value = replace(
        value,
        qualification_hash=rigging_surface_qualification_hash(value),
    )
    root = ctx["run_root"] / "artifacts" / "15_RIGGING_SURFACE_QUALIFIED"
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "qualified_rigging_surface.json",
                surface,
                authority_class="QUALIFIED_RIGGING_SURFACE",
            ),
            write_ir(
                root / "rigging_surface_qualification.json",
                value,
                authority_class="RIGGING_SURFACE_QUALIFICATION",
            ),
        ],
        "diagnostics": {
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "qualification_hash": value.qualification_hash,
            "tensorization_hash": value.tensorization_hash,
            "node_count": value.node_count,
            "edge_count": value.edge_count,
        },
    }
