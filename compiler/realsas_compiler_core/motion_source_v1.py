from __future__ import annotations

"""Typed motion-source identity and run-local authorization seal.

Stage 33 does not retarget, compile, evaluate or prove motion. It preserves reusable
source identity separately from the exact product/rest-preservation authorization that
permits Stage 34 to attempt compilation.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Iterable, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json=dict[str,Any]

SOURCE_KINDS={"INLINE_PRESET_SPEC_V1","EXTERNAL_ARTIST_CLIP_V1"}
SOURCE_SPACES={
    "PROCEDURAL_PARAMETER_SPACE_V1",
    "CANONICAL_JOINT_TRACKS_V1",
    "SOURCE_RIG_TRACKS_V1",
    "SOURCE_RIG_TRACKS_V2",
}
CHANNELS={
    "ROTATION_DEG","TRANSLATION_XY","SCALE_XY","DEPTH_OFFSET",
    "ORDER","VISIBILITY",
    "LOCAL_ROTATION_QUAT_XYZW","LOCAL_TRANSLATION_XYZ","LOCAL_SCALE_XYZ",
}


@dataclass(frozen=True)
class MotionSourceAssetIR:
    clip_id:str
    clip_kind:str
    source_kind:str
    source_space:str
    duration_seconds:float
    loop:bool
    channel_contract:tuple[str,...]
    source_payload_hash:str
    source_ref:str
    source_asset_hash:str
    schema_version:str="RealSaS.MotionSourceAssetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class MotionSourceSetIR:
    assets:tuple[MotionSourceAssetIR,...]
    source_set_hash:str
    schema_version:str="RealSaS.MotionSourceSetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedMotionSourceSealIR:
    source_set_binding_hash:str
    product_state_binding_hash:str
    rest_preservation_binding_hash:str
    source_assets:tuple[MotionSourceAssetIR,...]
    qualification_report:Json
    motion_source_seal_hash:str
    schema_version:str="RealSaS.QualifiedMotionSourceSealIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def motion_source_asset_hash(value:MotionSourceAssetIR)->str:
    payload=value.to_dict(); payload.pop("source_asset_hash",None)
    return content_sha256(payload)


def motion_source_set_hash(value:MotionSourceSetIR)->str:
    payload=value.to_dict(); payload.pop("source_set_hash",None)
    return content_sha256(payload)


def motion_source_seal_hash(value:QualifiedMotionSourceSealIR)->str:
    payload=value.to_dict(); payload.pop("motion_source_seal_hash",None)
    return content_sha256(payload)


def _validate_asset(value:MotionSourceAssetIR)->None:
    if not value.clip_id or not value.clip_kind or not value.source_ref:
        raise QualificationError("MOTION_SOURCE_ASSET_IDENTITY_MISSING")
    if value.source_kind not in SOURCE_KINDS:
        raise QualificationError("MOTION_SOURCE_KIND_UNSUPPORTED")
    if value.source_space not in SOURCE_SPACES:
        raise QualificationError("MOTION_SOURCE_SPACE_UNSUPPORTED")
    if not math.isfinite(float(value.duration_seconds)) or float(value.duration_seconds)<=0.0:
        raise QualificationError("MOTION_SOURCE_DURATION_INVALID")
    channels=tuple(map(str,value.channel_contract))
    if not channels or len(channels)!=len(set(channels)) or any(ch not in CHANNELS for ch in channels):
        raise QualificationError("MOTION_SOURCE_CHANNEL_CONTRACT_INVALID")
    if len(value.source_payload_hash)!=64:
        raise QualificationError("MOTION_SOURCE_PAYLOAD_HASH_INVALID")
    if value.source_kind=="INLINE_PRESET_SPEC_V1" and value.source_space!="PROCEDURAL_PARAMETER_SPACE_V1":
        raise QualificationError("MOTION_PRESET_SOURCE_SPACE_INVALID")
    if value.source_kind=="EXTERNAL_ARTIST_CLIP_V1" and value.source_space=="PROCEDURAL_PARAMETER_SPACE_V1":
        raise QualificationError("MOTION_ARTIST_SOURCE_SPACE_INVALID")
    if value.source_asset_hash!=motion_source_asset_hash(value):
        raise QualificationError("MOTION_SOURCE_ASSET_HASH_MISMATCH")


def build_motion_source_asset(
    *,
    clip_id:str,
    clip_kind:str,
    source_kind:str,
    source_space:str,
    duration_seconds:float,
    loop:bool,
    channel_contract:Iterable[str],
    source_payload:Mapping[str,Any],
    source_ref:str,
    metadata:Json|None=None,
)->MotionSourceAssetIR:
    value=MotionSourceAssetIR(
        clip_id=str(clip_id),
        clip_kind=str(clip_kind),
        source_kind=str(source_kind),
        source_space=str(source_space),
        duration_seconds=float(duration_seconds),
        loop=bool(loop),
        channel_contract=tuple(map(str,channel_contract)),
        source_payload_hash=content_sha256(dict(source_payload)),
        source_ref=str(source_ref),
        source_asset_hash="",
        metadata=dict(metadata or {}),
    )
    value=replace(value,source_asset_hash=motion_source_asset_hash(value))
    _validate_asset(value)
    return value


def validate_motion_source_set(value:MotionSourceSetIR)->None:
    if not value.assets:
        raise QualificationError("MOTION_SOURCE_SET_EMPTY")
    clip_ids=set()
    asset_hashes=set()
    for asset in value.assets:
        _validate_asset(asset)
        if asset.clip_id in clip_ids:
            raise QualificationError("MOTION_SOURCE_CLIP_ID_DUPLICATE")
        if asset.source_asset_hash in asset_hashes:
            raise QualificationError("MOTION_SOURCE_ASSET_DUPLICATE")
        clip_ids.add(asset.clip_id); asset_hashes.add(asset.source_asset_hash)
    if value.source_set_hash!=motion_source_set_hash(value):
        raise QualificationError("MOTION_SOURCE_SET_HASH_MISMATCH")


def build_motion_source_set(assets:Iterable[MotionSourceAssetIR],*,metadata:Json|None=None)->MotionSourceSetIR:
    rows=tuple(sorted(tuple(assets),key=lambda x:x.clip_id))
    value=MotionSourceSetIR(
        assets=rows,
        source_set_hash="",
        metadata={
            "retargeting_performed":False,
            "motion_compilation_performed":False,
            "motion_quality_claimed":False,
            **dict(metadata or {}),
        },
    )
    value=replace(value,source_set_hash=motion_source_set_hash(value))
    validate_motion_source_set(value)
    return value


def validate_qualified_motion_source_seal(
    value:QualifiedMotionSourceSealIR,
    *,
    source_set:MotionSourceSetIR,
    product_state,
    rest_preservation,
)->None:
    validate_motion_source_set(source_set)
    if value.source_set_binding_hash!=source_set.source_set_hash:
        raise QualificationError("MOTION_SOURCE_SEAL_SOURCE_SET_DRIFT")
    if value.product_state_binding_hash!=product_state.product_state_hash:
        raise QualificationError("MOTION_SOURCE_SEAL_PRODUCT_STATE_DRIFT")
    if value.rest_preservation_binding_hash!=rest_preservation.preservation_lineage_hash:
        raise QualificationError("MOTION_SOURCE_SEAL_REST_PRESERVATION_DRIFT")
    if tuple(value.source_assets)!=tuple(source_set.assets):
        raise QualificationError("MOTION_SOURCE_SEAL_ASSET_DRIFT")
    if not bool(rest_preservation.qualification_report.get("motion_authorization_precondition_satisfied",False)):
        raise QualificationError("MOTION_SOURCE_SEAL_REST_PRESERVATION_NOT_AUTHORIZED")
    if any(str(row.status)!="PASS" for row in rest_preservation.view_decisions):
        raise QualificationError("MOTION_SOURCE_SEAL_REST_VIEW_NOT_PASS")
    report=dict(value.qualification_report or {})
    if report.get("status")!="PASS_SOURCE_IDENTITY_AND_PRECONDITION_ONLY":
        raise QualificationError("MOTION_SOURCE_SEAL_STATUS_INVALID")
    if bool(report.get("retargeting_performed",True)):
        raise QualificationError("MOTION_SOURCE_SEAL_RETARGETING_FORBIDDEN")
    if bool(report.get("motion_compilation_performed",True)):
        raise QualificationError("MOTION_SOURCE_SEAL_COMPILATION_FORBIDDEN")
    if bool(report.get("motion_quality_claimed",True)):
        raise QualificationError("MOTION_SOURCE_SEAL_QUALITY_CLAIM_FORBIDDEN")
    if value.motion_source_seal_hash!=motion_source_seal_hash(value):
        raise QualificationError("MOTION_SOURCE_SEAL_HASH_MISMATCH")


def build_qualified_motion_source_seal(
    *,
    source_set:MotionSourceSetIR,
    product_state,
    rest_preservation,
)->QualifiedMotionSourceSealIR:
    validate_motion_source_set(source_set)
    if not bool(rest_preservation.qualification_report.get("motion_authorization_precondition_satisfied",False)):
        raise QualificationError("MOTION_SOURCE_SEAL_REST_PRESERVATION_NOT_AUTHORIZED")
    if any(str(row.status)!="PASS" for row in rest_preservation.view_decisions):
        raise QualificationError("MOTION_SOURCE_SEAL_REST_VIEW_NOT_PASS")
    value=QualifiedMotionSourceSealIR(
        source_set_binding_hash=source_set.source_set_hash,
        product_state_binding_hash=product_state.product_state_hash,
        rest_preservation_binding_hash=rest_preservation.preservation_lineage_hash,
        source_assets=source_set.assets,
        qualification_report={
            "status":"PASS_SOURCE_IDENTITY_AND_PRECONDITION_ONLY",
            "source_asset_count":len(source_set.assets),
            "retargeting_performed":False,
            "motion_compilation_performed":False,
            "motion_quality_claimed":False,
            "stage34_compile_required":True,
            "stage35_dynamic_proof_required":True,
        },
        motion_source_seal_hash="",
        metadata={
            "reusable_source_identity_separate_from_run_authorization":True,
            "rotation_only_preset_is_not_professional_motion_evidence":True,
        },
    )
    value=replace(value,motion_source_seal_hash=motion_source_seal_hash(value))
    validate_qualified_motion_source_seal(
        value,source_set=source_set,product_state=product_state,
        rest_preservation=rest_preservation,
    )
    return value
