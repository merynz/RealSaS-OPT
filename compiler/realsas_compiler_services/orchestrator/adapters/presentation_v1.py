from __future__ import annotations

"""Mainline stage-30 qualified presentation adapter."""

from compiler.realsas_compiler_core.presentation_graph_v1 import build_qualified_presentation_bundle
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    canonical_puppet_state_from_dict,
    component_carrier_policy_from_dict,
    deformation_envelope_from_dict,
    mechanical_partition_from_dict,
    qualified_mesh_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _stage_output_payload,
    _write_ir,
)


def qualify_presentation_graph_stage(ctx:dict)->dict:
    config=dict(ctx["run_manifest"].get("presentation") or {})
    unknown=set(config)-{"mode"}
    if unknown:
        return {
            "status":"BLOCKED",
            "blockers":["PRESENTATION_CONFIG_UNSUPPORTED"],
            "diagnostics":{"unsupported_keys":sorted(unknown)},
        }
    mode=str(config.get("mode") or "AUTO_ROLE_FREE_V1")
    if mode!="AUTO_ROLE_FREE_V1":
        return {
            "status":"BLOCKED",
            "blockers":["PRESENTATION_MODE_UNSUPPORTED"],
            "diagnostics":{"mode":mode},
        }

    observation_set=qualified_observation_set_from_dict(
        _stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
    )
    surface=rigging_surface_from_dict(
        _stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1")
    )
    skeleton=qualified_skeleton_from_dict(
        _stage_output_payload(ctx,"18_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1")
    )
    skin=qualified_skin_from_dict(
        _stage_output_payload(ctx,"22_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1")
    )
    partition=mechanical_partition_from_dict(
        _stage_output_payload(ctx,"24_MECHANICAL_PARTITION_QUALIFIED","RealSaS.MechanicalPartitionIR.v1")
    )
    carrier_policy=component_carrier_policy_from_dict(
        _stage_output_payload(ctx,"24_MECHANICAL_PARTITION_QUALIFIED","RealSaS.ComponentCarrierPolicyIR.v1")
    )
    envelope=deformation_envelope_from_dict(
        _stage_output_payload(ctx,"25_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1")
    )
    mesh=qualified_mesh_from_dict(
        _stage_output_payload(ctx,"27_QUALIFIED_MESH_GATE","RealSaS.QualifiedMeshIR.v1")
    )
    product_state=canonical_puppet_state_from_dict(
        _stage_output_payload(ctx,"29_CANONICAL_PUPPET_STATE_SEALED","RealSaS.CanonicalPuppetStateIR.v1")
    )

    structure,appearance,composition,graph=build_qualified_presentation_bundle(
        surface=surface,skeleton=skeleton,skin=skin,mesh=mesh,partition=partition,
        carrier_policy=carrier_policy,envelope=envelope,product_state=product_state,
        observation_set=observation_set,
    )
    root=ctx["run_root"]/"artifacts"/"30_QUALIFIED_PRESENTATION_GRAPH"
    outputs=[
        _write_ir(root/"qualified_presentation_structure.json",structure,authority_class="QUALIFIED_PRESENTATION_STRUCTURE"),
        _write_ir(root/"qualified_appearance_set.json",appearance,authority_class="QUALIFIED_SOURCE_APPEARANCE"),
        _write_ir(root/"qualified_composition_set.json",composition,authority_class="QUALIFIED_PRESENTATION_COMPOSITION"),
        _write_ir(root/"qualified_presentation_graph.json",graph,authority_class="QUALIFIED_PRESENTATION_GRAPH"),
    ]
    return {
        "status":"PASS",
        "outputs":outputs,
        "diagnostics":{
            "presentation_lineage_hash":graph.presentation_lineage_hash,
            "structure_lineage_hash":structure.structure_lineage_hash,
            "appearance_set_hash":appearance.appearance_set_hash,
            "composition_set_hash":composition.composition_set_hash,
            "slot_count":len(structure.slots),
            "attachment_count":len(structure.attachments),
            "view_count":len(graph.view_overlays),
            "mode":mode,
        },
    }
