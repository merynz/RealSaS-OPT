from __future__ import annotations

"""Stage-40 exact product closure adapter."""

from compiler.realsas_compiler_core.product_closure_v1 import build_product_closure_seal
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
    seal=build_product_closure_seal(
        product_state_hash=state.product_state_hash,rest=rest,dynamic=dynamic,projection=projection,
        package=package,native=native,visual=visual,
    )
    root=ctx["run_root"]/"artifacts"/"40_PRODUCT_CLOSURE_SEAL"
    return {
        "status":"PASS",
        "outputs":[_write_ir(root/"product_closure_seal.json",seal,authority_class="PRODUCT_PASS_CLOSURE_SEAL")],
        "diagnostics":{
            "product_closure_hash":seal.product_closure_hash,
            "product_pass":True,
            "professional_motion_clip_ids":list(seal.professional_motion_clip_ids),
            "founder_visual_pass_claimed":False,
        },
    }
