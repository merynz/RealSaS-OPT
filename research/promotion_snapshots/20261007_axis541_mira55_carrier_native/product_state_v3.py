from __future__ import annotations

"""Carrier-native Stage38 mechanical/complete puppet seal.

Stage37 remains presentation-only and consumes QualifiedMeshSkinIR. Stage38 uses
QualifiedCarrierSkinIR as the semantic W_M authority and the Stage36 artifact only
as an identity re-key onto the exact same M.
"""

from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict, complete_appearance_qualification_from_dict
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    component_carrier_policy_from_dict,deformation_envelope_from_dict,mechanical_partition_from_dict,mesh_policy_from_dict,
    qualified_mesh_from_dict,qualified_mesh_skin_from_dict,qualified_skeleton_from_dict,rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_puppet_state_v2 import build_canonical_puppet_state_carrier_v2
from compiler.realsas_compiler_core.carrier_skin_v1 import qualified_carrier_skin_from_dict
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import mechanical_carrier_evidence_from_dict
from compiler.realsas_compiler_core.output_presentation_v1 import output_direction_set_from_dict
from compiler.realsas_compiler_core.presentation_partition_v2 import presentation_partition_evidence_from_dict
from compiler.realsas_compiler_core.product_state_v2 import build_caa_bound_presentation_graph,build_complete_puppet_state_v2,presentation_structure_v2_from_dict
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import visual_mesh_set_from_dict
from compiler.realsas_compiler_core.visual_presentation_v1 import qualified_visual_presentation_set_from_dict
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload,write_ir
from compiler.realsas_compiler_services.orchestrator.adapters.product_state_legacy_v2 import _require_caa_final_mesh_candidate_binding


def seal_complete_puppet_stage(ctx:dict)->dict:
    surface=rigging_surface_from_dict(stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    skeleton=qualified_skeleton_from_dict(stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    carrier_skin=qualified_carrier_skin_from_dict(stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedCarrierSkinIR.v1"))
    carrier_evidence=mechanical_carrier_evidence_from_dict(stage_output_payload(ctx,"19_STATIC_CANONICAL_MESH_QUALIFIED","RealSaS.MechanicalCarrierEvidenceIR.v1"))
    partition=mechanical_partition_from_dict(stage_output_payload(ctx,"17_MECHANICAL_PARTITION_QUALIFIED","RealSaS.MechanicalPartitionIR.v1"))
    carrier_policy=component_carrier_policy_from_dict(stage_output_payload(ctx,"17_MECHANICAL_PARTITION_QUALIFIED","RealSaS.ComponentCarrierPolicyIR.v1"))
    envelope=deformation_envelope_from_dict(stage_output_payload(ctx,"34_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1"))
    policy=mesh_policy_from_dict(stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1"))
    mesh=qualified_mesh_from_dict(stage_output_payload(ctx,"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED","RealSaS.QualifiedMeshIR.v1"))
    mesh_skin=qualified_mesh_skin_from_dict(stage_output_payload(ctx,"36_QUALIFIED_MESH_SKIN_TRANSFER","RealSaS.QualifiedMeshSkinIR.v1"))
    structure=presentation_structure_v2_from_dict(stage_output_payload(ctx,"37_QUALIFIED_PRESENTATION_STRUCTURE","RealSaS.QualifiedPresentationStructureIR.v2"))
    partition_evidence=presentation_partition_evidence_from_dict(stage_output_payload(ctx,"37_QUALIFIED_PRESENTATION_STRUCTURE","RealSaS.PresentationPartitionEvidenceIR.v2"))
    asset=complete_appearance_asset_from_dict(stage_output_payload(ctx,"23_COMPLETE_APPEARANCE_ASSET_BAKED","RealSaS.CompleteAppearanceAssetIR.v2"))
    appearance=complete_appearance_qualification_from_dict(stage_output_payload(ctx,"24_COMPLETE_APPEARANCE_QUALIFIED","RealSaS.CompleteAppearanceQualificationIR.v2"))
    directions=output_direction_set_from_dict(stage_output_payload(ctx,"16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED","RealSaS.OutputPresentationDirectionSetIR.v1"))

    if appearance.asset_binding_hash!=asset.asset_hash: raise ValueError("COMPLETE_PUPPET_CAA_QUALIFICATION_ASSET_DRIFT")
    _require_caa_final_mesh_candidate_binding(mesh,asset)
    if partition_evidence.mesh_binding_hash!=mesh.mesh_lineage_hash: raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_MESH_DRIFT")
    if partition_evidence.appearance_asset_binding_hash!=asset.asset_hash: raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_ASSET_DRIFT")
    if partition_evidence.appearance_qualification_binding_hash!=appearance.qualification_hash: raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_QUALIFICATION_DRIFT")
    if str(structure.metadata.get("presentation_partition_evidence_hash") or "")!=partition_evidence.evidence_hash: raise ValueError("COMPLETE_PUPPET_PRESENTATION_PARTITION_EVIDENCE_DRIFT")
    if str(appearance.qualification_report.get("status") or "")!="PASS_COMPLETE_APPEARANCE": raise ValueError("COMPLETE_PUPPET_CAA_QUALIFICATION_NOT_PASS")
    if float(appearance.total_defined_fraction)!=1.0 or not bool(appearance.qualification_report.get("totality_passed",False)): raise ValueError("COMPLETE_PUPPET_CAA_TOTALITY_NOT_PROVEN")

    source_owned_visual_mode=bool(dict(asset.metadata or {}).get("source_owned_visual_mesh_mode"))
    visual_mesh_set_hash=""; source_visual_mesh_set_hash=""
    if source_owned_visual_mode:
        source_visual_set=visual_mesh_set_from_dict(stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.VisualMeshSetIR.v1"))
        qualified_visual=qualified_visual_presentation_set_from_dict(stage_output_payload(ctx,"37_QUALIFIED_PRESENTATION_STRUCTURE","RealSaS.QualifiedVisualPresentationSetIR.v1"))
        source_visual_mesh_set_hash=str(source_visual_set.set_hash); visual_mesh_set_hash=str(qualified_visual.set_hash)
        if qualified_visual.source_visual_mesh_set_binding_hash!=source_visual_mesh_set_hash: raise ValueError("COMPLETE_PUPPET_SOURCE_VISUAL_SUBSTRATE_BINDING_DRIFT")
        if qualified_visual.mechanical_mesh_binding_hash!=mesh.mesh_lineage_hash: raise ValueError("COMPLETE_PUPPET_QUALIFIED_VISUAL_MESH_BINDING_DRIFT")
        if qualified_visual.appearance_asset_binding_hash!=asset.asset_hash or qualified_visual.appearance_qualification_binding_hash!=appearance.qualification_hash: raise ValueError("COMPLETE_PUPPET_QUALIFIED_VISUAL_APPEARANCE_DRIFT")
        if str(dict(asset.metadata or {}).get("visual_mesh_set_binding_hash") or "")!=source_visual_mesh_set_hash: raise ValueError("COMPLETE_PUPPET_SOURCE_VISUAL_ASSET_BINDING_DRIFT")
        for label,value in (("STRUCTURE",dict(structure.metadata or {}).get("qualified_visual_presentation_set_binding_hash")),("PARTITION_EVIDENCE",dict(partition_evidence.metadata or {}).get("qualified_visual_presentation_set_binding_hash"))):
            if str(value or "")!=visual_mesh_set_hash: raise ValueError("COMPLETE_PUPPET_QUALIFIED_VISUAL_BINDING_DRIFT:"+label)
        for label,value in (("STRUCTURE",dict(structure.metadata or {}).get("source_visual_mesh_set_binding_hash")),("PARTITION_EVIDENCE",dict(partition_evidence.metadata or {}).get("source_visual_mesh_set_binding_hash"))):
            if str(value or "")!=source_visual_mesh_set_hash: raise ValueError("COMPLETE_PUPPET_SOURCE_VISUAL_BINDING_DRIFT:"+label)
        if dict(asset.metadata or {}).get("mechanical_mesh_render_authority") is not False or dict(structure.metadata or {}).get("mechanical_mesh_render_authority") is not False:
            raise ValueError("COMPLETE_PUPPET_MECHANICAL_RENDER_AUTHORITY_DRIFT")

    mechanical=build_canonical_puppet_state_carrier_v2(
        surface=surface,skeleton=skeleton,carrier_skin=carrier_skin,carrier=carrier_evidence,
        partition=partition,carrier_policy=carrier_policy,envelope=envelope,policy=policy,mesh=mesh,mesh_skin=mesh_skin,
        metadata={"v2_role":"MECHANICAL_SUBSTATE_OF_COMPLETE_PUPPET","appearance_bound_externally":True},
    )
    graph=build_caa_bound_presentation_graph(structure=structure,product_state=mechanical,skeleton=skeleton,mesh=mesh,partition=partition,carrier_policy=carrier_policy,directions=directions,appearance_asset_hash=asset.asset_hash,appearance_qualification_hash=appearance.qualification_hash,visual_mesh_set_hash=visual_mesh_set_hash)
    complete=build_complete_puppet_state_v2(mechanical_state=mechanical,structure=structure,presentation_graph=graph,skeleton=skeleton,mesh=mesh,mesh_skin=mesh_skin,appearance_asset_hash=asset.asset_hash,appearance_qualification_hash=appearance.qualification_hash,output_direction_set_hash=directions.direction_set_hash,visual_mesh_set_hash=visual_mesh_set_hash)
    root=ctx["run_root"]/"artifacts"/ctx["stage"]["id"]
    return {"status":"PASS","outputs":[
        write_ir(root/"canonical_mechanical_puppet_state.json",mechanical,authority_class="CANONICAL_MECHANICAL_PUPPET_STATE"),
        write_ir(root/"qualified_presentation_graph.json",graph,authority_class="CAA_BOUND_QUALIFIED_PRESENTATION_GRAPH"),
        write_ir(root/"complete_puppet_state_v2.json",complete,authority_class="COMPLETE_PUPPET_STATE_V2"),
    ],"diagnostics":{
        "mechanical_state_hash":mechanical.product_state_hash,"presentation_lineage_hash":graph.presentation_lineage_hash,"complete_puppet_hash":complete.complete_puppet_hash,
        "complete_appearance_qualification_hash":appearance.qualification_hash,"geometry_mechanics_appearance_coequal":True,
        "source_owned_visual_mesh_mode":source_owned_visual_mode,"visual_mesh_set_binding_hash":visual_mesh_set_hash,"source_visual_mesh_set_binding_hash":source_visual_mesh_set_hash,
        "mechanical_mesh_render_authority":False if source_owned_visual_mode else True,"dynamic_visual_composition_authority_claimed":False,
        "skin_authority":"CARRIER_NATIVE_W_M","semantic_skin_transfer_performed":False,
    }}

__all__=["seal_complete_puppet_stage"]
