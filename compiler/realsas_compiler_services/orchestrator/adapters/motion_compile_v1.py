from __future__ import annotations

"""Stage-34 canonical motion compiler adapter.

Compilation is mechanics qualification only. Stage 35 owns dynamic/render/contact/
phase evidence and any motion-quality claim.
"""

import json

from compiler.realsas_compiler_core.motion_compile_v2 import build_qualified_motion_v2
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    canonical_puppet_state_from_dict,
    deformation_envelope_from_dict,
    motion_source_set_from_dict,
    qualified_motion_source_seal_from_dict,
    qualified_presentation_graph_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _load_file_ref,
    _stage_output_payload,
    _write_ir,
)


def _source_payloads(ctx:dict,source_set)->dict:
    cfg=dict(ctx["run_manifest"].get("motion") or {})
    rows=tuple(cfg.get("sources") or ())
    by_id={}
    for raw in rows:
        row=dict(raw)
        kind=str(row.get("source_kind") or "")
        if kind=="INLINE_PRESET_SPEC_V1":
            clip_id=str(row.get("clip_id") or "")
            payload=dict(row.get("spec") or {})
        elif kind=="EXTERNAL_ARTIST_CLIP_V1":
            path=_load_file_ref(dict(row.get("file") or {}),json_required=False)
            payload=json.loads(path.read_text(encoding="utf-8"))
            if str(payload.get("schema") or payload.get("schema_version") or "")!="RealSaS.MotionSourceClip.v2":
                raise QualificationError("MOTION_STAGE34_PROFESSIONAL_CLIP_REQUIRES_V2")
            clip_id=str(payload.get("clip_id") or row.get("clip_id") or "")
        else:
            raise QualificationError("MOTION_STAGE34_SOURCE_KIND_UNSUPPORTED")
        if not clip_id or clip_id in by_id:
            raise QualificationError("MOTION_STAGE34_MANIFEST_SOURCE_ID_INVALID")
        by_id[clip_id]=(row,payload)

    payloads={}
    for asset in source_set.assets:
        pair=by_id.get(asset.clip_id)
        if pair is None:
            raise QualificationError("MOTION_STAGE34_MANIFEST_SOURCE_MISSING")
        row,payload=pair
        if str(row.get("source_kind") or "")!=asset.source_kind:
            raise QualificationError("MOTION_STAGE34_MANIFEST_SOURCE_KIND_DRIFT")
        payloads[asset.clip_id]=payload
    if set(by_id)!={a.clip_id for a in source_set.assets}:
        raise QualificationError("MOTION_STAGE34_MANIFEST_SOURCE_ACCOUNTING_DRIFT")
    return payloads


def compile_motion_stage(ctx:dict)->dict:
    motion_cfg=dict(ctx["run_manifest"].get("motion") or {})
    unknown=set(motion_cfg)-{"sources","compiler"}
    if unknown:
        return {"status":"BLOCKED","blockers":["MOTION_STAGE34_CONFIG_UNSUPPORTED"],"diagnostics":{"unsupported_keys":sorted(unknown)}}

    skeleton=qualified_skeleton_from_dict(
        _stage_output_payload(ctx,"18_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1")
    )
    envelope=deformation_envelope_from_dict(
        _stage_output_payload(ctx,"25_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1")
    )
    product_state=canonical_puppet_state_from_dict(
        _stage_output_payload(ctx,"29_CANONICAL_PUPPET_STATE_SEALED","RealSaS.CanonicalPuppetStateIR.v1")
    )
    presentation=qualified_presentation_graph_from_dict(
        _stage_output_payload(ctx,"30_QUALIFIED_PRESENTATION_GRAPH","RealSaS.QualifiedPresentationGraphIR.v1")
    )
    source_set=motion_source_set_from_dict(
        _stage_output_payload(ctx,"33_MOTION_SOURCE_OR_PRESET_SEAL","RealSaS.MotionSourceSetIR.v1")
    )
    source_seal=qualified_motion_source_seal_from_dict(
        _stage_output_payload(ctx,"33_MOTION_SOURCE_OR_PRESET_SEAL","RealSaS.QualifiedMotionSourceSealIR.v1")
    )
    constraints,motion=build_qualified_motion_v2(
        source_set=source_set,
        source_seal=source_seal,
        source_payloads=_source_payloads(ctx,source_set),
        product_state=product_state,
        skeleton=skeleton,
        envelope=envelope,
        presentation=presentation,
        compiler_config=dict(motion_cfg.get("compiler") or {}),
    )
    root=ctx["run_root"]/"artifacts"/"34_MOTION_COMPILE_RUN"
    outputs=[
        _write_ir(root/"motion_compile_constraints.json",constraints,authority_class="MOTION_COMPILE_CONSTRAINTS"),
        _write_ir(root/"qualified_motion.json",motion,authority_class="QUALIFIED_MOTION"),
    ]
    return {
        "status":"PASS",
        "outputs":outputs,
        "diagnostics":{
            "constraint_set_hash":constraints.constraint_set_hash,
            "motion_lineage_hash":motion.motion_lineage_hash,
            "clip_count":len(motion.clips),
            "artist_source_clip_count":motion.qualification_report["artist_source_clip_count"],
            "mechanical_probe_clip_count":motion.qualification_report["mechanical_probe_clip_count"],
            "dynamic_proof_passed":False,
            "motion_quality_claimed":False,
            "stage35_dynamic_proof_required":True,
            "full_3d_local_quaternion_motion":True,
        },
    }
