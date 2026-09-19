from __future__ import annotations

"""Stage-40 exact product closure adapter."""

from compiler.realsas_compiler_core.product_closure_v1 import build_product_closure_seal
from compiler.realsas_compiler_core.authoring_bundle_v1 import materialize_editable_puppet_bundle_v1
from compiler.realsas_compiler_core.motion_dynamic_proof_v1 import qualified_dynamic_motion_from_dict
from compiler.realsas_compiler_core.runtime_projection_v1 import runtime_v4_projection_from_dict
from compiler.realsas_compiler_core.runtime_package_v1 import runtime_v4_package_seal_from_dict
from compiler.realsas_compiler_core.runtime_native_proof_v1 import native_playback_from_dict,visual_motion_from_dict
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    canonical_puppet_state_from_dict,qualified_rest_source_preservation_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _stage_output_payload,_write_ir,
)


def seal_product_closure_stage(ctx:dict)->dict:
    state=canonical_puppet_state_from_dict(
        _stage_output_payload(ctx,"29_CANONICAL_PUPPET_STATE_SEALED","RealSaS.CanonicalPuppetStateIR.v1")
    )
    rest=qualified_rest_source_preservation_from_dict(
        _stage_output_payload(ctx,"32_REST_SOURCE_PRESERVATION_GATE","RealSaS.QualifiedRestSourcePreservationIR.v1")
    )
    dynamic=qualified_dynamic_motion_from_dict(
        _stage_output_payload(ctx,"35_MOTION_DYNAMIC_PROOF","RealSaS.QualifiedDynamicMotionIR.v1")
    )
    projection=runtime_v4_projection_from_dict(
        _stage_output_payload(ctx,"36_RUNTIME_V4_PROJECT","RealSaS.RuntimeV4ProjectionIR.v1")
    )
    package=runtime_v4_package_seal_from_dict(
        _stage_output_payload(ctx,"37_RSS_MATERIALIZE_COMPACT","RealSaS.RuntimeV4PackageSealIR.v1")
    )
    native=native_playback_from_dict(
        _stage_output_payload(ctx,"38_NATIVE_PACKAGE_OPEN_PLAYBACK","RealSaS.QualifiedNativePlaybackIR.v1")
    )
    visual=visual_motion_from_dict(
        _stage_output_payload(ctx,"39_VISUAL_MOTION_RENDER_BAKE","RealSaS.QualifiedVisualMotionEvidenceIR.v1")
    )
    skeleton_payload=_stage_output_payload(ctx,"18_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1")
    mesh_payload=_stage_output_payload(ctx,"27_QUALIFIED_MESH_GATE","RealSaS.QualifiedMeshIR.v1")
    mesh_skin_payload=_stage_output_payload(ctx,"28_QUALIFIED_MESH_SKIN_TRANSFER","RealSaS.QualifiedMeshSkinIR.v1")
    presentation_payload=_stage_output_payload(ctx,"30_QUALIFIED_PRESENTATION_GRAPH","RealSaS.QualifiedPresentationGraphIR.v1")
    appearance_payload=_stage_output_payload(ctx,"30_QUALIFIED_PRESENTATION_GRAPH","RealSaS.QualifiedAppearanceSetIR.v1")
    motion_payload=_stage_output_payload(ctx,"34_MOTION_COMPILE_RUN","RealSaS.QualifiedMotionIR.v1")
    state_payload=_stage_output_payload(ctx,"29_CANONICAL_PUPPET_STATE_SEALED","RealSaS.CanonicalPuppetStateIR.v1")
    root=ctx["run_root"]/"artifacts"/"40_PRODUCT_CLOSURE_SEAL"
    authoring=materialize_editable_puppet_bundle_v1(
        out_path=root/"editable_puppet.rsedit",
        state_payload=state_payload,skeleton_payload=skeleton_payload,
        mesh_payload=mesh_payload,mesh_skin_payload=mesh_skin_payload,
        presentation_payload=presentation_payload,appearance_payload=appearance_payload,
        motion_payload=motion_payload,projection=projection,
    )
    seal=build_product_closure_seal(
        product_state_hash=state.product_state_hash,rest=rest,dynamic=dynamic,projection=projection,
        package=package,native=native,visual=visual,authoring=authoring,
    )
    return {
        "status":"PASS",
        "outputs":[
            _write_ir(root/"editable_puppet_bundle_seal.json",authoring,authority_class="EDITABLE_PUPPET_AUTHORING_SEAL"),
            {"path":authoring.archive_path,"sha256":authoring.archive_sha256,"authority_class":"EDITABLE_PUPPET_AUTHORING_ARCHIVE","schema":"application/x-realsas-editable"},
            _write_ir(root/"product_closure_seal.json",seal,authority_class="PRODUCT_PASS_CLOSURE_SEAL"),
        ],
        "diagnostics":{
            "product_closure_hash":seal.product_closure_hash,
            "product_pass":True,
            "professional_motion_clip_ids":list(seal.professional_motion_clip_ids),
            "founder_visual_pass_claimed":False,
            "editable_authoring_bundle_hash":authoring.authoring_bundle_hash,
            "editable_export_passed":True,
        },
    }
