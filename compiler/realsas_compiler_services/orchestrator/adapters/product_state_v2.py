from __future__ import annotations

"""V2 Stage37-38 presentation and complete puppet sealing."""

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
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
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
    stage_output_payload,
    write_ir,
)


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
    cfg = dict(ctx["run_manifest"].get("presentation") or {})
    unknown = set(cfg) - {
        "mode",
        "min_rigid_owner_weight",
        "max_rigid_other_mass",
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

    structure = build_presentation_structure_v2(
        skeleton=skeleton,
        mesh=mesh,
        mesh_skin=mesh_skin,
        partition=partition,
        carrier_policy=carrier,
        min_rigid_owner_weight=float(cfg.get("min_rigid_owner_weight", 0.999)),
        max_rigid_other_mass=float(cfg.get("max_rigid_other_mass", 0.001)),
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "qualified_presentation_structure_v2.json",
                structure,
                authority_class="QUALIFIED_PRESENTATION_STRUCTURE_V2",
            )
        ],
        "diagnostics": {
            "structure_hash": structure.structure_hash,
            "slot_count": len(structure.slots),
            "attachment_count": len(structure.attachments),
            "categorical_recognition_used": False,
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
