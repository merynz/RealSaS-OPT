from __future__ import annotations

"""CAS-native executor for the stable product.mesh role.

The executor consumes only typed capability artifacts. Shape supplies normalized
learned topology evidence; Compiler-owned normalization/surface/partition courts
bind that proposal into world space, repair it under the frozen release policy,
replay static source-fidelity qualification, then mint one immutable
CanonicalMeshIR. No legacy run path, moving alias, or second TESSA inference is
permitted here.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


MESH_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductMeshCompile.v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MESH_POLICY_PATH = _REPO_ROOT / "canonical" / "QUALIFIED_MESH_PRODUCT_POLICY_V2_20260919.json"


def _one(
    request: dict[str, Any], artifact_type: str, schema_version: str
) -> dict[str, Any]:
    matches = [
        dict(ref)
        for ref in request.get("input_artifacts") or ()
        if str(ref.get("artifact_type") or "") == artifact_type
        and str(ref.get("schema_version") or "") == schema_version
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"MESH_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(matches)}"
        )
    return matches[0]


def _json_payload(read_object, ref: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(read_object(ref).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "MESH_TYPED_JSON_DECODE_FAILED:" + str(ref.get("artifact_type") or "")
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("MESH_TYPED_JSON_OBJECT_REQUIRED")
    return value


def _json_bytes(value: Any) -> bytes:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _seal(
    value: Any,
    *,
    role: str,
    artifact_type: str,
    schema_version: str,
    authority_class: str,
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    sealed = put_bytes(_json_bytes(value))
    return {
        "role": role,
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "storage_key": sealed["storage_key"],
        "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"],
        "authority_class": authority_class,
    }


def execute_mesh_capability_v1(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != MESH_EXECUTOR_CONTRACT_V1:
        raise RuntimeError("MESH_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("MESH_CAPABILITY_KIND_DRIFT")
    if request.get("model_releases"):
        raise RuntimeError("MESH_LEARNED_MODEL_RELEASE_FORBIDDEN")

    # Imports stay local so generic worker discovery does not pull model/science
    # dependencies unless product.mesh actually executes.
    from compiler.realsas_compiler_core.artifact_codec_v2 import (
        component_carrier_policy_from_dict,
        mechanical_partition_from_dict,
        qualified_camera_set_from_dict,
        qualified_observation_set_from_dict,
        rigging_surface_from_dict,
    )
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import mint_canonical_mesh_v1
    from compiler.realsas_compiler_core.geometry_substrate_v2 import geometry_substrate_from_dict
    from compiler.realsas_compiler_core.mesh.product_policy_v1 import load_mesh_policy_document
    from compiler.realsas_compiler_core.preproduct_authority_v1 import normalization_domain_from_dict
    from compiler.realsas_compiler_core.shape_mesh_bridge_v1 import bind_shape_proposal_to_tessa_v1
    from compiler.realsas_compiler_core.static_mesh_qualification_v1 import relation_parent_quality_report_v1
    from compiler.realsas_compiler_core.static_mesh_qualification_v2 import qualify_static_mesh_v2
    from compiler.realsas_compiler_core.surface_addressing_v1 import build_surface_addressing
    from compiler.realsas_compiler_core.tessa_stage18_v1 import build_tessa_static_candidate_from_proposal_v1
    from compiler.realsas_compiler_services.platform_worker.shape_capability_v2 import decode_shape_proposal_v2
    from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
        decode_source_observation_arrays_v1,
        derive_foreground_masks_v1,
    )

    shape_ref = _one(request, "RealSaS.ShapeProposal", "v2")
    source_ref = _one(request, "RealSaS.SourceObservation", "v1")
    camera_ref = _one(request, "RealSaS.QualifiedCameraSetIR", "v1")
    observation_ref = _one(request, "RealSaS.QualifiedObservationSetIR", "v1")
    normalization_ref = _one(request, "RealSaS.NormalizationDomainIR", "v1")
    geometry_ref = _one(request, "RealSaS.GeometrySubstrateQualificationIR", "v2")
    surface_ref = _one(request, "RealSaS.RiggingSurfaceIR", "v1")
    partition_ref = _one(request, "RealSaS.MechanicalPartitionIR", "v1")
    carrier_policy_ref = _one(request, "RealSaS.ComponentCarrierPolicyIR", "v1")

    shape = decode_shape_proposal_v2(read_object(shape_ref))
    source = decode_source_observation_arrays_v1(read_object(source_ref))
    cameras = qualified_camera_set_from_dict(_json_payload(read_object, camera_ref))
    observation = qualified_observation_set_from_dict(
        _json_payload(read_object, observation_ref)
    )
    normalization = normalization_domain_from_dict(
        _json_payload(read_object, normalization_ref)
    )
    geometry = geometry_substrate_from_dict(_json_payload(read_object, geometry_ref))
    surface = rigging_surface_from_dict(_json_payload(read_object, surface_ref))
    partition = mechanical_partition_from_dict(_json_payload(read_object, partition_ref))
    carrier_policy = component_carrier_policy_from_dict(
        _json_payload(read_object, carrier_policy_ref)
    )

    if normalization.camera_set_binding_hash != cameras.camera_set_hash:
        raise RuntimeError("MESH_NORMALIZATION_CAMERA_BINDING_DRIFT")
    if normalization.observation_set_binding_hash != observation.observation_set_hash:
        raise RuntimeError("MESH_NORMALIZATION_OBSERVATION_BINDING_DRIFT")
    if geometry.camera_set_binding_hash != cameras.camera_set_hash:
        raise RuntimeError("MESH_GEOMETRY_CAMERA_BINDING_DRIFT")
    if geometry.observation_set_binding_hash != observation.observation_set_hash:
        raise RuntimeError("MESH_GEOMETRY_OBSERVATION_BINDING_DRIFT")

    expected_mask_sha = {
        int(view.view_index): str(view.foreground_mask_sha256)
        for view in observation.views
    }
    foreground = derive_foreground_masks_v1(
        source, expected_sha256_by_view=expected_mask_sha
    )

    expected_policy_sha = str(metadata.get("mesh_policy_document_sha256") or "")
    if len(expected_policy_sha) != 64 or not _MESH_POLICY_PATH.is_file():
        raise RuntimeError("MESH_RELEASE_POLICY_BINDING_REQUIRED")
    actual_policy_sha = hashlib.sha256(_MESH_POLICY_PATH.read_bytes()).hexdigest()
    if actual_policy_sha != expected_policy_sha:
        raise RuntimeError("MESH_RELEASE_POLICY_SOURCE_DRIFT")
    mesh_policy = load_mesh_policy_document(_MESH_POLICY_PATH)

    proposal, frame_binding = bind_shape_proposal_to_tessa_v1(
        shape=shape,
        shape_proposal_content_sha256=str(shape_ref.get("content_sha256") or ""),
        normalization=normalization,
        surface=surface,
    )
    support_config = dict(metadata.get("support_binding") or {})
    repair_config = dict(metadata.get("static_repair") or {})
    candidate_bundle = build_tessa_static_candidate_from_proposal_v1(
        proposal=proposal,
        proposal_evidence_sha256=str(shape_ref.get("content_sha256") or ""),
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        mesh_policy=mesh_policy,
        support_config=support_config,
        static_repair_config=repair_config,
        relation_quality_report=relation_parent_quality_report_v1,
    )
    candidate = candidate_bundle.repaired_candidate
    addressing = build_surface_addressing(candidate)
    static_result = qualify_static_mesh_v2(
        candidate=candidate,
        surface_addressing=addressing,
        geometry=geometry,
        mesh_policy=mesh_policy,
        partition=partition,
        cameras=cameras,
        observation=observation,
        source_foreground=foreground,
    )
    if static_result.status != "PASS":
        return {
            "capability_id": str(request.get("capability_id") or ""),
            "status": "FAIL",
            "outputs": [],
            "diagnostics": dict(static_result.diagnostics),
            "failure": {
                "code": str(
                    static_result.blockers[0]
                    if static_result.blockers
                    else "MESH_STATIC_QUALIFICATION_FAILED"
                ),
                "class": "COMPILER_MESH",
                "reported_owner_module_id": str(request.get("owner_module_id") or ""),
                "diagnostics": dict(static_result.diagnostics),
            },
        }
    if static_result.qualification is None or static_result.carrier_evidence is None:
        raise RuntimeError("MESH_STATIC_PASS_WITHOUT_AUTHORITY")

    proposal_refs = (
        str(shape_ref.get("content_sha256") or ""),
        frame_binding.binding_hash,
        candidate_bundle.material_support_field.field_lineage_hash,
        candidate_bundle.bridge_evidence.evidence_hash,
        candidate_bundle.repair_evidence.evidence_hash,
        candidate_bundle.reference_binding.binding_hash,
    )
    mesh = mint_canonical_mesh_v1(
        candidate,
        static_qualification=static_result.qualification,
        carrier_evidence=static_result.carrier_evidence,
        proposal_evidence_refs=proposal_refs,
    )

    outputs = [
        _seal(
            mesh,
            role="canonical_mesh",
            artifact_type="RealSaS.CanonicalMeshIR",
            schema_version="v1",
            authority_class="COMPILER_CANONICAL_MESH",
            put_bytes=put_bytes,
        ),
        _seal(
            static_result.qualification,
            role="static_mesh_qualification",
            artifact_type="RealSaS.StaticMeshQualificationIR",
            schema_version="v2",
            authority_class="STATIC_CANONICAL_MESH_QUALIFICATION",
            put_bytes=put_bytes,
        ),
        _seal(
            static_result.carrier_evidence,
            role="mechanical_carrier_evidence",
            artifact_type="RealSaS.MechanicalCarrierEvidenceIR",
            schema_version="v1",
            authority_class="STATIC_QUALIFIED_MECHANICAL_CARRIER_EVIDENCE",
            put_bytes=put_bytes,
        ),
        _seal(
            addressing,
            role="surface_addressing",
            artifact_type="RealSaS.SurfaceAddressingIR",
            schema_version="v1",
            authority_class="CANONICAL_SURFACE_ADDRESSING",
            put_bytes=put_bytes,
        ),
        _seal(
            frame_binding,
            role="shape_mesh_frame_binding",
            artifact_type="RealSaS.ShapeMeshFrameBinding",
            schema_version="v1",
            authority_class="COMPILER_SHAPE_FRAME_BINDING",
            put_bytes=put_bytes,
        ),
        _seal(
            candidate_bundle.material_support_field,
            role="proposal_material_support",
            artifact_type="RealSaS.TESSAMaterialSupportFieldIR",
            schema_version="v1",
            authority_class="PROPOSAL_MATERIAL_SUPPORT_EVIDENCE",
            put_bytes=put_bytes,
        ),
        _seal(
            candidate_bundle.bridge_evidence,
            role="proposal_candidate_bridge",
            artifact_type="RealSaS.TESSACandidateBridgeEvidenceIR",
            schema_version="v1",
            authority_class="PROPOSAL_CANDIDATE_BRIDGE_EVIDENCE",
            put_bytes=put_bytes,
        ),
        _seal(
            candidate_bundle.repair_evidence,
            role="proposal_static_repair",
            artifact_type="RealSaS.TESSAStaticRepairEvidenceIR",
            schema_version="v1",
            authority_class="COMPILER_STATIC_REPAIR_EVIDENCE",
            put_bytes=put_bytes,
        ),
        _seal(
            candidate_bundle.reference_binding,
            role="proposal_reference_binding",
            artifact_type="RealSaS.TESSAReferenceBindingIR",
            schema_version="v1",
            authority_class="LEARNED_PROPOSAL_REFERENCE_BINDING",
            put_bytes=put_bytes,
        ),
    ]
    diagnostics = {
        "schema": "RealSaS.ProductMeshCompileDiagnostics.v1",
        "mesh_lineage_hash": mesh.mesh_lineage_hash,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "static_qualification_hash": static_result.qualification.qualification_hash,
        "carrier_evidence_hash": static_result.carrier_evidence.carrier_evidence_hash,
        "frame_binding_hash": frame_binding.binding_hash,
        "shape_proposal_content_sha256": str(shape_ref.get("content_sha256") or ""),
        "mesh_policy_document_sha256": actual_policy_sha,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "second_tessa_inference_performed": False,
        "presentation_direction_dependency": False,
        "graphics_authority_claimed": False,
        "rig_authority_claimed": False,
        "skin_authority_claimed": False,
        "product_revision_pass_claimed": False,
        **dict(static_result.diagnostics),
    }
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": diagnostics,
        "failure": None,
    }


__all__ = ["execute_mesh_capability_v1"]
