from __future__ import annotations

"""Stage-33 motion source/preset identity and authorization seal."""

import json

from compiler.realsas_compiler_core.motion_source_v1 import (
    build_motion_source_asset,
    build_motion_source_set,
    build_qualified_motion_source_seal,
)
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    canonical_puppet_state_from_dict,
    qualified_rest_source_preservation_from_dict,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _load_file_ref,
    _sha256,
    _stage_output_payload,
    _write_ir,
)


def _asset_from_manifest_row(row:dict):
    kind=str(row.get("source_kind") or "")
    if kind=="INLINE_PRESET_SPEC_V1":
        spec=dict(row.get("spec") or {})
        return build_motion_source_asset(
            clip_id=str(row.get("clip_id") or ""),
            clip_kind=str(row.get("clip_kind") or ""),
            source_kind=kind,
            source_space="PROCEDURAL_PARAMETER_SPACE_V1",
            duration_seconds=float(row.get("duration_seconds",0.0)),
            loop=bool(row.get("loop",False)),
            channel_contract=tuple(map(str,row.get("channel_contract") or ())),
            source_payload=spec,
            source_ref=f"manifest:inline:{row.get('clip_id') or ''}",
            metadata={
                "preset_spec":spec,
                "external_file_used":False,
                "stage33_only_source_identity":True,
            },
        )
    if kind=="EXTERNAL_ARTIST_CLIP_V1":
        ref=dict(row.get("file") or {})
        path=_load_file_ref(ref,expected_schema="RealSaS.MotionSourceClip.v1")
        payload=json.loads(path.read_text(encoding="utf-8"))
        return build_motion_source_asset(
            clip_id=str(payload.get("clip_id") or row.get("clip_id") or ""),
            clip_kind=str(payload.get("clip_kind") or row.get("clip_kind") or ""),
            source_kind=kind,
            source_space=str(payload.get("source_space") or ""),
            duration_seconds=float(payload.get("duration_seconds",0.0)),
            loop=bool(payload.get("loop",False)),
            channel_contract=tuple(map(str,payload.get("channel_contract") or ())),
            source_payload=payload,
            source_ref=f"file:{path.name}:{_sha256(path)}",
            metadata={
                "external_file_sha256":_sha256(path),
                "external_file_schema":"RealSaS.MotionSourceClip.v1",
                "stage33_only_source_identity":True,
            },
        )
    raise QualificationError("MOTION_STAGE33_SOURCE_KIND_UNSUPPORTED")


def seal_motion_source_or_preset_stage(ctx:dict)->dict:
    cfg=dict(ctx["run_manifest"].get("motion") or {})
    rows=tuple(cfg.get("sources") or ())
    if not rows:
        return {"status":"BLOCKED","blockers":["MOTION_SOURCE_SET_EMPTY"],"diagnostics":{}}
    product_state=canonical_puppet_state_from_dict(
        _stage_output_payload(ctx,"29_CANONICAL_PUPPET_STATE_SEALED","RealSaS.CanonicalPuppetStateIR.v1")
    )
    rest=qualified_rest_source_preservation_from_dict(
        _stage_output_payload(ctx,"32_REST_SOURCE_PRESERVATION_GATE","RealSaS.QualifiedRestSourcePreservationIR.v1")
    )
    assets=tuple(_asset_from_manifest_row(dict(row)) for row in rows)
    source_set=build_motion_source_set(
        assets,
        metadata={
            "manifest_source_count":len(rows),
            "stage34_retarget_or_compile_required":True,
        },
    )
    seal=build_qualified_motion_source_seal(
        source_set=source_set,product_state=product_state,rest_preservation=rest
    )
    root=ctx["run_root"]/"artifacts"/"33_MOTION_SOURCE_OR_PRESET_SEAL"
    outputs=[
        _write_ir(root/"motion_source_set.json",source_set,authority_class="MOTION_SOURCE_IDENTITY"),
        _write_ir(root/"qualified_motion_source_seal.json",seal,authority_class="QUALIFIED_MOTION_SOURCE_SEAL"),
    ]
    return {
        "status":"PASS",
        "outputs":outputs,
        "diagnostics":{
            "source_set_hash":source_set.source_set_hash,
            "motion_source_seal_hash":seal.motion_source_seal_hash,
            "source_asset_count":len(source_set.assets),
            "clip_ids":[row.clip_id for row in source_set.assets],
            "retargeting_performed":False,
            "motion_compilation_performed":False,
            "motion_quality_claimed":False,
        },
    }
