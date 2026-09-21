from __future__ import annotations

"""V2 Stage37-38 presentation and complete puppet sealing."""

from PIL import Image
import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    complete_appearance_asset_from_dict,
    complete_appearance_qualification_from_dict,
)
from compiler.realsas_compiler_core.canonical_puppet_state_v1 import (
    build_canonical_puppet_state,
)
from compiler.realsas_compiler_core.output_presentation_v1 import (
    output_direction_set_from_dict,
)
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_uv,
    load_provenance_atlas,
)
from compiler.realsas_compiler_core.presentation_partition_v2 import (
    build_presentation_partition_evidence,
    presentation_partition_evidence_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    component_carrier_policy_from_dict,
    deformation_envelope_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_mesh_from_dict,
    qualified_mesh_skin_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.product_state_v2 import (
    build_caa_bound_presentation_graph,
    build_complete_puppet_state_v2,
    build_presentation_structure_v2,
    presentation_structure_v2_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    load_file_ref,
    resolved_path,
    sha256_file,
    stage_output_payload,
    write_ir,
)
from compiler.realsas_compiler_core.types import QualificationError


def _require_caa_final_mesh_candidate_binding(mesh, asset) -> str:
    source_candidate_hash = str(
        mesh.metadata.get("source_candidate_lineage_hash") or ""
    )
    if not source_candidate_hash:
        raise QualificationError(
            "PRESENTATION_V2_MESH_SOURCE_CANDIDATE_BINDING_MISSING"
        )
    if str(asset.candidate_mesh_binding_hash) != source_candidate_hash:
        raise QualificationError(
            "PRESENTATION_V2_CAA_MESH_CANDIDATE_BINDING_DRIFT"
        )
    return source_candidate_hash


def qualify_presentation_structure_stage(ctx: dict) -> dict:
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
    partition = mechanical_partition_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.MechanicalPartitionIR.v1",
        )
    )
    carrier = component_carrier_policy_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.ComponentCarrierPolicyIR.v1",
        )
    )
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    appearance = complete_appearance_qualification_from_dict(
        stage_output_payload(
            ctx,
            "24_COMPLETE_APPEARANCE_QUALIFIED",
            "RealSaS.CompleteAppearanceQualificationIR.v2",
        )
    )
    if appearance.asset_binding_hash != asset.asset_hash:
        raise QualificationError("PRESENTATION_V2_APPEARANCE_BINDING_DRIFT")
    _require_caa_final_mesh_candidate_binding(mesh, asset)
    if str(appearance.qualification_report.get("status") or "") != "PASS_COMPLETE_APPEARANCE":
        raise QualificationError("PRESENTATION_V2_APPEARANCE_NOT_QUALIFIED")

    cfg = dict(ctx["run_manifest"].get("presentation") or {})
    unknown = set(cfg) - {
        "mode",
        "policy_document",
    }
    if unknown:
        return {
            "status": "BLOCKED",
            "blockers": ["PRESENTATION_V2_CONFIG_UNSUPPORTED"],
            "diagnostics": {"unsupported_keys": sorted(unknown)},
        }
    if str(cfg.get("mode") or "AUTO_ROLE_FREE_V2") != "AUTO_ROLE_FREE_V2":
        return {
            "status": "BLOCKED",
            "blockers": ["PRESENTATION_V2_MODE_UNSUPPORTED"],
            "diagnostics": {"mode": cfg.get("mode")},
        }

    policy_ref = dict(cfg.get("policy_document") or {})
    if not policy_ref:
        return {
            "status": "BLOCKED",
            "blockers": ["PRESENTATION_V2_FROZEN_POLICY_DOCUMENT_REQUIRED"],
            "diagnostics": {},
        }
    policy = load_file_ref(
        policy_ref,
        expected_schema="RealSaS.PresentationPartitionPolicy.v1",
    )
    if not str(policy.get("status") or "").startswith("FROZEN_SUBJECT_FREE"):
        raise QualificationError("PRESENTATION_V2_POLICY_NOT_SUBJECT_FREE_FROZEN")
    forbidden = set(map(str, policy.get("forbidden_inputs") or ()))
    if not {"KNIGHT_RESULT", "SUBJECT_ID", "CATEGORY_LABEL"}.issubset(forbidden):
        raise QualificationError("PRESENTATION_V2_POLICY_FORBIDDEN_INPUTS_INCOMPLETE")
    mechanical_policy = dict(policy.get("mechanical_binding_policy") or {})
    required_mechanical = {
        "min_rigid_owner_weight",
        "max_rigid_other_mass",
        "rigidity_noop_required",
        "rigidity_noop_relative_edge_tolerance",
        "rigidity_noop_probe_rotation_degrees",
        "legacy_weight_thresholds_final_authority",
    }
    if not required_mechanical.issubset(mechanical_policy):
        raise QualificationError("PRESENTATION_V2_MECHANICAL_POLICY_INCOMPLETE")
    if bool(mechanical_policy["rigidity_noop_required"]) is not True:
        raise QualificationError("PRESENTATION_V2_RIGIDITY_NOOP_REQUIRED")
    if bool(mechanical_policy["legacy_weight_thresholds_final_authority"]):
        raise QualificationError(
            "PRESENTATION_V2_LEGACY_WEIGHT_THRESHOLD_AUTHORITY_FORBIDDEN"
        )

    uv_path = resolved_path(asset.uv_npz_path)
    provenance_path = resolved_path(asset.provenance_npz_path)
    if not uv_path.is_file() or sha256_file(uv_path) != asset.uv_npz_sha256:
        raise QualificationError("PRESENTATION_V2_CAA_UV_BYTES_DRIFT")
    if (
        not provenance_path.is_file()
        or sha256_file(provenance_path) != asset.provenance_npz_sha256
    ):
        raise QualificationError("PRESENTATION_V2_CAA_PROVENANCE_BYTES_DRIFT")
    face_uv = load_face_uv(asset)
    provenance = load_provenance_atlas(asset)
    textures = {}
    for row in asset.textures:
        path = resolved_path(row.transport_png_path)
        if not path.is_file() or sha256_file(path) != row.transport_png_sha256:
            raise QualificationError("PRESENTATION_V2_CAA_TEXTURE_BYTES_DRIFT")
        textures[int(row.direction_index)] = np.asarray(
            Image.open(path).convert("RGBA"),
            dtype=np.uint8,
        )

    evidence = build_presentation_partition_evidence(
        mesh=mesh,
        appearance_asset_hash=asset.asset_hash,
        appearance_qualification_hash=appearance.qualification_hash,
        face_uv=face_uv,
        textures_by_direction=textures,
        provenance_by_direction=provenance,
        policy=policy,
    )
    structure = build_presentation_structure_v2(
        skeleton=skeleton,
        mesh=mesh,
        mesh_skin=mesh_skin,
        partition=partition,
        carrier_policy=carrier,
        min_rigid_owner_weight=float(mechanical_policy["min_rigid_owner_weight"]),
        max_rigid_other_mass=float(mechanical_policy["max_rigid_other_mass"]),
        rigidity_noop_relative_edge_tolerance=float(
            mechanical_policy["rigidity_noop_relative_edge_tolerance"]
        ),
        rigidity_noop_probe_rotation_degrees=float(
            mechanical_policy["rigidity_noop_probe_rotation_degrees"]
        ),
        presentation_cut_face_pairs=evidence.cut_face_pairs,
        presentation_partition_evidence_hash=evidence.evidence_hash,
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "presentation_partition_evidence_v2.json",
                evidence,
                authority_class="QUALIFIED_PRESENTATION_PARTITION_EVIDENCE_V2",
            ),
            write_ir(
                root / "qualified_presentation_structure_v2.json",
                structure,
                authority_class="QUALIFIED_PRESENTATION_STRUCTURE_V2",
            ),
        ],
        "diagnostics": {
            "structure_hash": structure.structure_hash,
            "slot_count": len(structure.slots),
            "attachment_count": len(structure.attachments),
            "presentation_partition_evidence_hash": evidence.evidence_hash,
            "evaluated_shared_edge_count": evidence.evaluated_shared_edge_count,
            "source_supported_edge_count": evidence.source_supported_edge_count,
            "appearance_boundary_cut_count": len(evidence.cut_face_pairs),
            "categorical_recognition_used": False,
            "conceptual_object_identity_claimed": False,
            "appearance_authority_minted": False,
        },
    }


def seal_complete_puppet_stage(ctx: dict) -> dict:
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx,
            "15_RIGGING_SURFACE_QUALIFIED",
            "RealSaS.RiggingSurfaceIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx, "32_SKIN_QUALIFIED", "RealSaS.QualifiedSkinIR.v1"
        )
    )
    partition = mechanical_partition_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.MechanicalPartitionIR.v1",
        )
    )
    carrier = component_carrier_policy_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.ComponentCarrierPolicyIR.v1",
        )
    )
    envelope = deformation_envelope_from_dict(
        stage_output_payload(
            ctx,
            "34_DEFORMATION_CAPABILITY_ENVELOPE",
            "RealSaS.DeformationCapabilityEnvelopeIR.v1",
        )
    )
    policy = mesh_policy_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.MeshQualificationPolicyIR.v1",
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
    structure = presentation_structure_v2_from_dict(
        stage_output_payload(
            ctx,
            "37_QUALIFIED_PRESENTATION_STRUCTURE",
            "RealSaS.QualifiedPresentationStructureIR.v2",
        )
    )
    partition_evidence = presentation_partition_evidence_from_dict(
        stage_output_payload(
            ctx,
            "37_QUALIFIED_PRESENTATION_STRUCTURE",
            "RealSaS.PresentationPartitionEvidenceIR.v2",
        )
    )
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    appearance = complete_appearance_qualification_from_dict(
        stage_output_payload(
            ctx,
            "24_COMPLETE_APPEARANCE_QUALIFIED",
            "RealSaS.CompleteAppearanceQualificationIR.v2",
        )
    )
    directions = output_direction_set_from_dict(
        stage_output_payload(
            ctx,
            "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
            "RealSaS.OutputPresentationDirectionSetIR.v1",
        )
    )
    if appearance.asset_binding_hash != asset.asset_hash:
        raise ValueError("COMPLETE_PUPPET_CAA_QUALIFICATION_ASSET_DRIFT")
    _require_caa_final_mesh_candidate_binding(mesh, asset)
    if partition_evidence.mesh_binding_hash != mesh.mesh_lineage_hash:
        raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_MESH_DRIFT")
    if partition_evidence.appearance_asset_binding_hash != asset.asset_hash:
        raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_ASSET_DRIFT")
    if (
        partition_evidence.appearance_qualification_binding_hash
        != appearance.qualification_hash
    ):
        raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_QUALIFICATION_DRIFT")
    if (
        str(structure.metadata.get("presentation_partition_evidence_hash") or "")
        != partition_evidence.evidence_hash
    ):
        raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_EVIDENCE_DRIFT")
    if (
        str(appearance.qualification_report.get("status") or "")
        != "PASS_COMPLETE_APPEARANCE"
    ):
        raise ValueError("COMPLETE_PUPPET_CAA_QUALIFICATION_NOT_PASS")
    if (
        float(appearance.total_defined_fraction) != 1.0
        or not bool(appearance.qualification_report.get("totality_passed", False))
    ):
        raise ValueError("COMPLETE_PUPPET_CAA_TOTALITY_NOT_PROVEN")

    mechanical = build_canonical_puppet_state(
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        carrier_policy=carrier,
        envelope=envelope,
        policy=policy,
        mesh=mesh,
        mesh_skin=mesh_skin,
        metadata={
            "v2_role": "MECHANICAL_SUBSTATE_OF_COMPLETE_PUPPET",
            "appearance_bound_externally": True,
        },
    )
    graph = build_caa_bound_presentation_graph(
        structure=structure,
        product_state=mechanical,
        skeleton=skeleton,
        mesh=mesh,
        partition=partition,
        carrier_policy=carrier,
        directions=directions,
        appearance_asset_hash=asset.asset_hash,
        appearance_qualification_hash=appearance.qualification_hash,
    )
    complete = build_complete_puppet_state_v2(
        mechanical_state=mechanical,
        structure=structure,
        presentation_graph=graph,
        skeleton=skeleton,
        mesh=mesh,
        mesh_skin=mesh_skin,
        appearance_asset_hash=asset.asset_hash,
        appearance_qualification_hash=appearance.qualification_hash,
        output_direction_set_hash=directions.direction_set_hash,
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "canonical_mechanical_puppet_state.json",
                mechanical,
                authority_class="CANONICAL_MECHANICAL_PUPPET_STATE",
            ),
            write_ir(
                root / "qualified_presentation_graph.json",
                graph,
                authority_class="CAA_BOUND_QUALIFIED_PRESENTATION_GRAPH",
            ),
            write_ir(
                root / "complete_puppet_state_v2.json",
                complete,
                authority_class="COMPLETE_PUPPET_STATE_V2",
            ),
        ],
        "diagnostics": {
            "mechanical_state_hash": mechanical.product_state_hash,
            "presentation_lineage_hash": graph.presentation_lineage_hash,
            "complete_puppet_hash": complete.complete_puppet_hash,
            "complete_appearance_qualification_hash": appearance.qualification_hash,
            "geometry_mechanics_appearance_coequal": True,
        },
    }
