from __future__ import annotations

"""Stage-34 canonical motion compilation and mechanics qualification.

This module intentionally stops before dynamic/product-quality proof. It compiles
sealed motion sources into canonical puppet-local joint tracks, applies the exact
Stage-25 deformation envelope as the only joint-limit authority, preserves exact
product/presentation identity, and carries contact declarations forward for Stage 35.

Historical ARAP/XPBD/contact implementations are not invoked here. A contact
constraint being represented is not a claim that contact quality/satisfaction passed.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from .hashing import content_sha256
from .motion_source_v1 import (
    MotionSourceSetIR,
    QualifiedMotionSourceSealIR,
    motion_source_seal_hash,
    validate_motion_source_set,
)
from .product_authority_v1 import (
    DeformationCapabilityEnvelopeIR,
    QualifiedPresentationGraphIR,
    deformation_envelope_lineage_hash,
    qualified_presentation_lineage_hash,
)
from .canonical_puppet_state_v1 import CanonicalPuppetStateIR, canonical_puppet_state_hash
from .types import QualifiedSkeletonIR, QualificationError

Json=dict[str,Any]
Point2=tuple[float,float]

COMPILE_CHANNELS={"ROTATION_DEG","TRANSLATION_XY","SCALE_XY"}
ROOT_TRAJECTORY_MODES={"IN_PLACE","SOURCE_EXPLICIT"}
CONTACT_MODES={"PLANT_2D"}
MOTION_CLASSIFICATIONS={"MECHANICAL_PROBE_ONLY","ARTIST_SOURCE"}


@dataclass(frozen=True)
class MotionKeyframeIR:
    time_seconds:float
    rotation_deg:float=0.0
    translation_xy:Point2=(0.0,0.0)
    scale_xy:Point2=(1.0,1.0)
    metadata:Json=field(default_factory=dict)
    schema_version:str="RealSaS.MotionKeyframeIR.v1"
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class CanonicalJointTrackIR:
    canonical_joint_id:str
    source_joint_id:str
    channel_contract:tuple[str,...]
    keyframes:tuple[MotionKeyframeIR,...]
    metadata:Json=field(default_factory=dict)
    schema_version:str="RealSaS.CanonicalJointTrackIR.v1"
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class MotionContactConstraintIR:
    contact_id:str
    clip_id:str
    canonical_joint_id:str
    start_time_seconds:float
    end_time_seconds:float
    mode:str
    metadata:Json=field(default_factory=dict)
    schema_version:str="RealSaS.MotionContactConstraintIR.v1"
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class MotionCompileConstraintSetIR:
    product_state_binding_hash:str
    skeleton_binding_hash:str
    envelope_binding_hash:str
    presentation_binding_hash:str
    root_trajectory_modes:tuple[tuple[str,str],...]
    retarget_maps:tuple[tuple[str,tuple[tuple[str,str],...]],...]
    contact_constraints:tuple[MotionContactConstraintIR,...]
    dynamic_attachment_policy:str
    constraint_set_hash:str
    schema_version:str="RealSaS.MotionCompileConstraintSetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class CompiledMotionClipIR:
    clip_id:str
    clip_kind:str
    duration_seconds:float
    loop:bool
    source_asset_hash:str
    source_space:str
    classification:str
    root_trajectory_mode:str
    tracks:tuple[CanonicalJointTrackIR,...]
    clip_lineage_hash:str
    schema_version:str="RealSaS.CompiledMotionClipIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedMotionIR:
    motion_source_seal_binding_hash:str
    source_set_binding_hash:str
    product_state_binding_hash:str
    skeleton_binding_hash:str
    envelope_binding_hash:str
    presentation_binding_hash:str
    constraint_set_binding_hash:str
    clips:tuple[CompiledMotionClipIR,...]
    qualification_report:Json
    motion_lineage_hash:str
    schema_version:str="RealSaS.QualifiedMotionIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def _hash_without(value,field_name:str)->str:
    payload=value.to_dict(); payload.pop(field_name,None)
    return content_sha256(payload)


def motion_constraint_set_hash(value:MotionCompileConstraintSetIR)->str:
    return _hash_without(value,"constraint_set_hash")


def compiled_motion_clip_hash(value:CompiledMotionClipIR)->str:
    return _hash_without(value,"clip_lineage_hash")


def qualified_motion_hash(value:QualifiedMotionIR)->str:
    return _hash_without(value,"motion_lineage_hash")


def _finite(value:float)->bool:
    return math.isfinite(float(value))


def _point2(raw,*,code:str)->Point2:
    if not isinstance(raw,(list,tuple)) or len(raw)!=2:
        raise QualificationError(code)
    out=(float(raw[0]),float(raw[1]))
    if not all(_finite(x) for x in out):
        raise QualificationError(code)
    return out


def _joint_ranges(envelope:DeformationCapabilityEnvelopeIR):
    return {row.canonical_joint_id:row for row in envelope.joint_ranges}


def _check_product_bindings(*,source_set,source_seal,product_state,skeleton,envelope,presentation)->None:
    validate_motion_source_set(source_set)
    if source_seal.motion_source_seal_hash!=motion_source_seal_hash(source_seal):
        raise QualificationError("MOTION_COMPILE_SOURCE_SEAL_HASH_MISMATCH")
    if product_state.product_state_hash!=canonical_puppet_state_hash(product_state):
        raise QualificationError("MOTION_COMPILE_PRODUCT_STATE_HASH_MISMATCH")
    if envelope.envelope_lineage_hash!=deformation_envelope_lineage_hash(envelope):
        raise QualificationError("MOTION_COMPILE_ENVELOPE_HASH_MISMATCH")
    if presentation.presentation_lineage_hash!=qualified_presentation_lineage_hash(presentation):
        raise QualificationError("MOTION_COMPILE_PRESENTATION_HASH_MISMATCH")
    if source_seal.source_set_binding_hash!=source_set.source_set_hash:
        raise QualificationError("MOTION_COMPILE_SOURCE_SET_SEAL_DRIFT")
    if tuple(source_seal.source_assets)!=tuple(source_set.assets):
        raise QualificationError("MOTION_COMPILE_SOURCE_ASSET_SEAL_DRIFT")
    if source_seal.product_state_binding_hash!=product_state.product_state_hash:
        raise QualificationError("MOTION_COMPILE_SOURCE_SEAL_PRODUCT_DRIFT")
    if product_state.skeleton_lineage_hash!=skeleton.skeleton_lineage_hash:
        raise QualificationError("MOTION_COMPILE_SKELETON_PRODUCT_DRIFT")
    if product_state.deformation_envelope_lineage_hash!=envelope.envelope_lineage_hash:
        raise QualificationError("MOTION_COMPILE_ENVELOPE_PRODUCT_DRIFT")
    if envelope.skeleton_lineage_hash!=skeleton.skeleton_lineage_hash:
        raise QualificationError("MOTION_COMPILE_ENVELOPE_SKELETON_DRIFT")
    if presentation.product_state_binding_hash!=product_state.product_state_hash:
        raise QualificationError("MOTION_COMPILE_PRESENTATION_PRODUCT_DRIFT")
    if presentation.skeleton_binding_hash!=skeleton.skeleton_lineage_hash:
        raise QualificationError("MOTION_COMPILE_PRESENTATION_SKELETON_DRIFT")
    if source_seal.qualification_report.get("status")!="PASS_SOURCE_IDENTITY_AND_PRECONDITION_ONLY":
        raise QualificationError("MOTION_COMPILE_SOURCE_SEAL_NOT_AUTHORIZED")
    if not bool(source_seal.qualification_report.get("stage34_compile_required",False)):
        raise QualificationError("MOTION_COMPILE_STAGE34_NOT_AUTHORIZED")


def _validate_keyframe(
    key:MotionKeyframeIR,
    *,
    channels:tuple[str,...],
    joint_id:str,
    root_id:str,
    root_mode:str,
    envelope,
    duration_seconds:float,
)->None:
    if not _finite(key.time_seconds) or key.time_seconds<0.0 or key.time_seconds>duration_seconds+1e-9:
        raise QualificationError("MOTION_COMPILE_KEYFRAME_TIME_INVALID")
    row=envelope[joint_id]
    if "ROTATION_DEG" in channels:
        if key.rotation_deg<row.min_rotation_deg-1e-9 or key.rotation_deg>row.max_rotation_deg+1e-9:
            raise QualificationError("MOTION_COMPILE_ROTATION_OUTSIDE_ENVELOPE")
    elif abs(key.rotation_deg)>1e-12:
        raise QualificationError("MOTION_COMPILE_UNDECLARED_ROTATION")
    if "TRANSLATION_XY" in channels:
        radius=math.hypot(float(key.translation_xy[0]),float(key.translation_xy[1]))
        if radius>float(row.translation_radius)+1e-9:
            raise QualificationError("MOTION_COMPILE_TRANSLATION_OUTSIDE_ENVELOPE")
        if joint_id==root_id and root_mode=="IN_PLACE" and radius>1e-12:
            raise QualificationError("MOTION_COMPILE_IN_PLACE_ROOT_TRANSLATION")
    elif any(abs(float(x))>1e-12 for x in key.translation_xy):
        raise QualificationError("MOTION_COMPILE_UNDECLARED_TRANSLATION")
    if "SCALE_XY" in channels:
        if any(float(x)<row.min_scale-1e-9 or float(x)>row.max_scale+1e-9 for x in key.scale_xy):
            raise QualificationError("MOTION_COMPILE_SCALE_OUTSIDE_ENVELOPE")
    elif any(abs(float(x)-1.0)>1e-12 for x in key.scale_xy):
        raise QualificationError("MOTION_COMPILE_UNDECLARED_SCALE")


def _keyframe_from_payload(raw:Mapping[str,Any],channels:tuple[str,...])->MotionKeyframeIR:
    allowed={"time_seconds","rotation_deg","translation_xy","scale_xy","metadata"}
    unknown=set(raw)-allowed
    if unknown:
        raise QualificationError("MOTION_COMPILE_KEYFRAME_FIELD_UNSUPPORTED")
    missing=[]
    if "ROTATION_DEG" in channels and "rotation_deg" not in raw: missing.append("rotation_deg")
    if "TRANSLATION_XY" in channels and "translation_xy" not in raw: missing.append("translation_xy")
    if "SCALE_XY" in channels and "scale_xy" not in raw: missing.append("scale_xy")
    if missing:
        raise QualificationError("MOTION_COMPILE_DECLARED_CHANNEL_VALUE_MISSING")
    rotation=float(raw.get("rotation_deg",0.0))
    translation=_point2(raw.get("translation_xy",(0.0,0.0)),code="MOTION_COMPILE_TRANSLATION_INVALID")
    scale=_point2(raw.get("scale_xy",(1.0,1.0)),code="MOTION_COMPILE_SCALE_INVALID")
    if not _finite(rotation):
        raise QualificationError("MOTION_COMPILE_ROTATION_INVALID")
    return MotionKeyframeIR(
        time_seconds=float(raw["time_seconds"]),
        rotation_deg=rotation,
        translation_xy=translation,
        scale_xy=scale,
        metadata=dict(raw.get("metadata") or {}),
    )


def _tree_features(rows, *, id_key, parent_key, pos_key):
    ids=[str(r[id_key]) for r in rows]
    if not ids or len(ids)!=len(set(ids)):
        raise QualificationError("MOTION_RETARGET_SOURCE_SKELETON_ID_INVALID")
    by={str(r[id_key]):r for r in rows}
    roots=[jid for jid in ids if rget(by[jid],parent_key) in (None,"")]
    if len(roots)!=1:
        raise QualificationError("MOTION_RETARGET_SOURCE_SKELETON_ROOT_INVALID")
    root=roots[0]
    children={jid:[] for jid in ids}
    parent={}
    for jid in ids:
        p=rget(by[jid],parent_key)
        if p in (None,""):
            parent[jid]=None
        else:
            p=str(p)
            if p not in by or p==jid:
                raise QualificationError("MOTION_RETARGET_SOURCE_SKELETON_PARENT_INVALID")
            parent[jid]=p; children[p].append(jid)
    depth={}
    stack=[(root,0)]
    while stack:
        jid,d=stack.pop()
        if jid in depth:
            raise QualificationError("MOTION_RETARGET_SOURCE_SKELETON_CYCLE")
        depth[jid]=d
        for c in children[jid]:
            stack.append((c,d+1))
    if len(depth)!=len(ids):
        raise QualificationError("MOTION_RETARGET_SOURCE_SKELETON_DISCONNECTED")
    pos={}
    for jid in ids:
        p=np.asarray(by[jid][pos_key],dtype=np.float64)
        if p.shape!=(3,) or not np.isfinite(p).all():
            raise QualificationError("MOTION_RETARGET_SOURCE_SKELETON_POSITION_INVALID")
        pos[jid]=p
    root_pos=pos[root]
    scale=max((float(np.linalg.norm(p-root_pos)) for p in pos.values()),default=0.0)
    if scale<=1e-9:
        scale=1.0
    subtree={}
    def size(jid):
        if jid in subtree: return subtree[jid]
        subtree[jid]=1+sum(size(c) for c in children[jid])
        return subtree[jid]
    size(root)
    feat={}
    for jid in ids:
        feat[jid]=np.asarray([
            *(pos[jid]-root_pos)/scale,
            float(depth[jid])/max(max(depth.values()),1),
            float(len(children[jid]))/max(max((len(x) for x in children.values()),default=1),1),
            float(subtree[jid])/float(len(ids)),
            1.0 if jid==root else 0.0,
        ],dtype=np.float64)
    return ids,parent,children,feat,root

def rget(row,key):
    return row.get(key)

def _automatic_retarget_map(payload:Mapping[str,Any], skeleton:QualifiedSkeletonIR)->dict[str,str]:
    source_rows=tuple(payload.get("source_skeleton") or ())
    if not source_rows:
        raise QualificationError("MOTION_COMPILE_SOURCE_RIG_RETARGET_MISSING")
    src_ids,src_parent,src_children,src_feat,src_root=_tree_features(
        source_rows,id_key="source_joint_id",parent_key="parent_source_joint_id",pos_key="rest_position"
    )
    target_rows=tuple({
        "source_joint_id":j.canonical_joint_id,
        "parent_source_joint_id":j.parent_canonical_id,
        "rest_position":j.position,
    } for j in skeleton.joints)
    tgt_ids,tgt_parent,tgt_children,tgt_feat,tgt_root=_tree_features(
        target_rows,id_key="source_joint_id",parent_key="parent_source_joint_id",pos_key="rest_position"
    )
    if len(src_ids)!=len(tgt_ids):
        raise QualificationError("MOTION_COMPILE_AUTO_RETARGET_TOPOLOGY_CARDINALITY_MISMATCH")
    cost=np.zeros((len(src_ids),len(tgt_ids)),dtype=np.float64)
    for i,sid in enumerate(src_ids):
        for j,tid in enumerate(tgt_ids):
            sf,tf=src_feat[sid],tgt_feat[tid]
            c=float(np.linalg.norm(sf[:3]-tf[:3])) + 0.75*abs(sf[3]-tf[3]) + 0.5*abs(sf[4]-tf[4]) + 0.5*abs(sf[5]-tf[5])
            if (sid==src_root)!=(tid==tgt_root):
                c+=1000.0
            if len(src_children[sid])!=len(tgt_children[tid]):
                c+=100.0*abs(len(src_children[sid])-len(tgt_children[tid]))
            cost[i,j]=c
    ri,ci=linear_sum_assignment(cost)
    mapping={src_ids[int(i)]:tgt_ids[int(j)] for i,j in zip(ri,ci)}
    if mapping.get(src_root)!=tgt_root:
        raise QualificationError("MOTION_COMPILE_AUTO_RETARGET_ROOT_MISMATCH")
    for sid in src_ids:
        sp=src_parent[sid]
        tid=mapping[sid]
        tp=tgt_parent[tid]
        expected=None if sp is None else mapping.get(sp)
        if tp!=expected:
            raise QualificationError("MOTION_COMPILE_AUTO_RETARGET_PARENT_RELATION_MISMATCH")
    total=float(sum(cost[i,j] for i,j in zip(ri,ci)))
    if not math.isfinite(total):
        raise QualificationError("MOTION_COMPILE_AUTO_RETARGET_COST_NONFINITE")
    # Deterministic retargeting must abstain on symmetric/ambiguous matches.
    # Geometry/topology is normalized, so this is a subject-free dimensionless margin.
    for i,j in zip(ri,ci):
        row=np.sort(cost[int(i)])
        if len(row)>1 and float(row[1]-row[0])<0.02:
            raise QualificationError("MOTION_COMPILE_AUTO_RETARGET_AMBIGUOUS")
    return mapping


def _external_tracks(*,asset,payload,skeleton,envelope,root_mode,retarget_map)->tuple[CanonicalJointTrackIR,...]:
    if str(payload.get("schema") or payload.get("schema_version") or "")!="RealSaS.MotionSourceClip.v1":
        raise QualificationError("MOTION_COMPILE_EXTERNAL_CLIP_SCHEMA_INVALID")
    checks={
        "clip_id":asset.clip_id,
        "clip_kind":asset.clip_kind,
        "source_space":asset.source_space,
    }
    for key,expected in checks.items():
        if key in payload and str(payload[key])!=str(expected):
            raise QualificationError(f"MOTION_COMPILE_EXTERNAL_CLIP_IDENTITY_DRIFT:{key}")
    if "duration_seconds" in payload and abs(float(payload["duration_seconds"])-asset.duration_seconds)>1e-9:
        raise QualificationError("MOTION_COMPILE_EXTERNAL_CLIP_DURATION_DRIFT")
    if "loop" in payload and bool(payload["loop"])!=asset.loop:
        raise QualificationError("MOTION_COMPILE_EXTERNAL_CLIP_LOOP_DRIFT")
    if "channel_contract" in payload and tuple(map(str,payload["channel_contract"]))!=asset.channel_contract:
        raise QualificationError("MOTION_COMPILE_EXTERNAL_CLIP_CHANNEL_DRIFT")

    known={j.canonical_joint_id for j in skeleton.joints}
    tracks=[]
    used_target=set()
    used_source=set()
    for raw_track in tuple(payload.get("tracks") or ()):
        row=dict(raw_track)
        if asset.source_space=="CANONICAL_JOINT_TRACKS_V1":
            unknown=set(row)-{"canonical_joint_id","keyframes","metadata"}
            if unknown:
                raise QualificationError("MOTION_COMPILE_TRACK_FIELD_UNSUPPORTED")
            source_id=str(row.get("canonical_joint_id") or "")
            target_id=source_id
            if retarget_map:
                raise QualificationError("MOTION_COMPILE_CANONICAL_TRACK_RETARGET_FORBIDDEN")
        elif asset.source_space=="SOURCE_RIG_TRACKS_V1":
            unknown=set(row)-{"source_joint_id","keyframes","metadata"}
            if unknown:
                raise QualificationError("MOTION_COMPILE_TRACK_FIELD_UNSUPPORTED")
            source_id=str(row.get("source_joint_id") or "")
            if not source_id or source_id not in retarget_map:
                raise QualificationError("MOTION_COMPILE_SOURCE_RIG_RETARGET_MISSING")
            target_id=str(retarget_map[source_id])
        else:
            raise QualificationError("MOTION_COMPILE_EXTERNAL_SOURCE_SPACE_UNSUPPORTED")
        if not source_id or source_id in used_source or target_id in used_target or target_id not in known:
            raise QualificationError("MOTION_COMPILE_TRACK_IDENTITY_INVALID")
        used_source.add(source_id); used_target.add(target_id)
        keys=tuple(_keyframe_from_payload(dict(k),asset.channel_contract) for k in tuple(row.get("keyframes") or ()))
        if not keys:
            raise QualificationError("MOTION_COMPILE_TRACK_KEYFRAMES_EMPTY")
        if any(b.time_seconds<=a.time_seconds for a,b in zip(keys,keys[1:])):
            raise QualificationError("MOTION_COMPILE_TRACK_TIMES_NOT_STRICT")
        for key in keys:
            _validate_keyframe(
                key,channels=asset.channel_contract,joint_id=target_id,root_id=skeleton.root_id,
                root_mode=root_mode,envelope=envelope,duration_seconds=asset.duration_seconds,
            )
        tracks.append(CanonicalJointTrackIR(
            canonical_joint_id=target_id,
            source_joint_id=source_id,
            channel_contract=asset.channel_contract,
            keyframes=keys,
            metadata={
                "retargeted":asset.source_space=="SOURCE_RIG_TRACKS_V1",
                "delta_from_qualified_rest":True,
                "interpolation":"LINEAR_KEYFRAMES_V1",
            },
        ))
    if not tracks:
        raise QualificationError("MOTION_COMPILE_EXTERNAL_TRACKS_EMPTY")
    if asset.source_space=="SOURCE_RIG_TRACKS_V1" and set(retarget_map)!=used_source:
        raise QualificationError("MOTION_COMPILE_RETARGET_MAP_TRACK_ACCOUNTING_DRIFT")
    return tuple(sorted(tracks,key=lambda x:x.canonical_joint_id))


def _procedural_tracks(*,asset,payload,skeleton,envelope,root_mode)->tuple[CanonicalJointTrackIR,...]:
    if set(asset.channel_contract)!={"ROTATION_DEG"}:
        raise QualificationError("MOTION_COMPILE_PROCEDURAL_CHANNEL_UNSUPPORTED")
    if str(payload.get("preset_id") or "")!="MECHANICAL_SWAY_V1":
        raise QualificationError("MOTION_COMPILE_PRESET_UNSUPPORTED")
    target=str(payload.get("target_joint_id") or skeleton.root_id)
    known={j.canonical_joint_id for j in skeleton.joints}
    if target not in known:
        raise QualificationError("MOTION_COMPILE_PRESET_TARGET_UNKNOWN")
    amp=float(payload.get("amplitude_deg",0.0))
    center=float(payload.get("center_deg",0.0))
    if not _finite(amp) or not _finite(center) or amp<0.0:
        raise QualificationError("MOTION_COMPILE_PRESET_PARAMETER_INVALID")
    d=float(asset.duration_seconds)
    values=((0.0,center),(0.25*d,center+amp),(0.5*d,center),(0.75*d,center-amp),(d,center))
    keys=tuple(MotionKeyframeIR(t,rotation_deg=v) for t,v in values)
    for key in keys:
        _validate_keyframe(
            key,channels=asset.channel_contract,joint_id=target,root_id=skeleton.root_id,
            root_mode=root_mode,envelope=envelope,duration_seconds=d,
        )
    return (CanonicalJointTrackIR(
        target,target,asset.channel_contract,keys,
        metadata={"retargeted":False,"delta_from_qualified_rest":True,"interpolation":"LINEAR_KEYFRAMES_V1","procedural_preset":"MECHANICAL_SWAY_V1"},
    ),)


def _contacts_for_clip(*,clip_id:str,rows,skeleton,duration_seconds:float)->tuple[MotionContactConstraintIR,...]:
    known={j.canonical_joint_id for j in skeleton.joints}
    out=[]; seen=set()
    for raw in tuple(rows or ()):
        row=dict(raw); cid=str(row.get("contact_id") or ""); jid=str(row.get("canonical_joint_id") or "")
        start=float(row.get("start_time_seconds",-1.0)); end=float(row.get("end_time_seconds",-1.0)); mode=str(row.get("mode") or "")
        if not cid or cid in seen or jid not in known or mode not in CONTACT_MODES:
            raise QualificationError("MOTION_COMPILE_CONTACT_IDENTITY_INVALID")
        if not _finite(start) or not _finite(end) or start<0.0 or end<=start or end>duration_seconds+1e-9:
            raise QualificationError("MOTION_COMPILE_CONTACT_TIME_INVALID")
        seen.add(cid)
        out.append(MotionContactConstraintIR(cid,clip_id,jid,start,end,mode,metadata={"satisfaction_proved":False}))
    return tuple(out)


def build_motion_compile_constraint_set(
    *,
    source_set:MotionSourceSetIR,
    product_state:CanonicalPuppetStateIR,
    skeleton:QualifiedSkeletonIR,
    envelope:DeformationCapabilityEnvelopeIR,
    presentation:QualifiedPresentationGraphIR,
    compiler_config:Mapping[str,Any]|None=None,
    automatic_retarget_maps:Mapping[str,Mapping[str,str]]|None=None,
)->MotionCompileConstraintSetIR:
    cfg=dict(compiler_config or {})
    unknown=set(cfg)-{"root_trajectory_modes","contacts"}
    if unknown:
        raise QualificationError("MOTION_COMPILE_CONFIG_UNSUPPORTED")
    retarget_maps={str(k):dict(v) for k,v in dict(automatic_retarget_maps or {}).items()}
    root_modes=dict(cfg.get("root_trajectory_modes") or {})
    contacts_cfg=dict(cfg.get("contacts") or {})
    clip_ids={a.clip_id for a in source_set.assets}
    if set(retarget_maps)-clip_ids or set(root_modes)-clip_ids or set(contacts_cfg)-clip_ids:
        raise QualificationError("MOTION_COMPILE_CONFIG_UNKNOWN_CLIP")

    root_rows=[]; map_rows=[]; contacts=[]
    for asset in source_set.assets:
        mode=str(root_modes.get(asset.clip_id) or "IN_PLACE")
        if mode not in ROOT_TRAJECTORY_MODES:
            raise QualificationError("MOTION_COMPILE_ROOT_TRAJECTORY_MODE_INVALID")
        root_rows.append((asset.clip_id,mode))
        raw_map=dict(retarget_maps.get(asset.clip_id) or {})
        normalized_map=tuple(sorted((str(a),str(b)) for a,b in raw_map.items()))
        if len({a for a,_ in normalized_map})!=len(normalized_map) or len({b for _,b in normalized_map})!=len(normalized_map):
            raise QualificationError("MOTION_COMPILE_RETARGET_MAP_NOT_ONE_TO_ONE")
        map_rows.append((asset.clip_id,normalized_map))
        contacts.extend(_contacts_for_clip(
            clip_id=asset.clip_id,rows=contacts_cfg.get(asset.clip_id) or (),
            skeleton=skeleton,duration_seconds=asset.duration_seconds,
        ))

    value=MotionCompileConstraintSetIR(
        product_state_binding_hash=product_state.product_state_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        envelope_binding_hash=envelope.envelope_lineage_hash,
        presentation_binding_hash=presentation.presentation_lineage_hash,
        root_trajectory_modes=tuple(sorted(root_rows)),
        retarget_maps=tuple(sorted(map_rows)),
        contact_constraints=tuple(sorted(contacts,key=lambda x:(x.clip_id,x.contact_id))),
        dynamic_attachment_policy="STATIC_QUALIFIED_PRESENTATION_ONLY_V1",
        constraint_set_hash="",
        metadata={
            "joint_limit_authority":"DEFORMATION_CAPABILITY_ENVELOPE_V1",
            "contact_constraints_are_stage35_obligations":True,
            "historical_contact_solver_promoted":False,
            "dynamic_attachment_edits_allowed":False,
        },
    )
    return replace(value,constraint_set_hash=motion_constraint_set_hash(value))


def validate_motion_compile_constraint_set(
    value:MotionCompileConstraintSetIR,
    *,
    source_set:MotionSourceSetIR,
    product_state:CanonicalPuppetStateIR,
    skeleton:QualifiedSkeletonIR,
    envelope:DeformationCapabilityEnvelopeIR,
    presentation:QualifiedPresentationGraphIR,
)->None:
    expected={
        "product_state_binding_hash":product_state.product_state_hash,
        "skeleton_binding_hash":skeleton.skeleton_lineage_hash,
        "envelope_binding_hash":envelope.envelope_lineage_hash,
        "presentation_binding_hash":presentation.presentation_lineage_hash,
    }
    for field_name,expected_hash in expected.items():
        if getattr(value,field_name)!=expected_hash:
            raise QualificationError(f"MOTION_COMPILE_CONSTRAINT_BINDING_DRIFT:{field_name}")
    if value.dynamic_attachment_policy!="STATIC_QUALIFIED_PRESENTATION_ONLY_V1":
        raise QualificationError("MOTION_COMPILE_DYNAMIC_ATTACHMENT_POLICY_INVALID")
    assets={a.clip_id:a for a in source_set.assets}
    root_rows=dict(value.root_trajectory_modes)
    if len(root_rows)!=len(value.root_trajectory_modes) or set(root_rows)!=set(assets):
        raise QualificationError("MOTION_COMPILE_ROOT_POLICY_ACCOUNTING_INVALID")
    if any(mode not in ROOT_TRAJECTORY_MODES for mode in root_rows.values()):
        raise QualificationError("MOTION_COMPILE_ROOT_TRAJECTORY_MODE_INVALID")
    map_rows=dict(value.retarget_maps)
    if len(map_rows)!=len(value.retarget_maps) or set(map_rows)!=set(assets):
        raise QualificationError("MOTION_COMPILE_RETARGET_ACCOUNTING_INVALID")
    known={j.canonical_joint_id for j in skeleton.joints}
    for clip_id,mapping_rows in map_rows.items():
        asset=assets[clip_id]
        mapping=tuple((str(a),str(b)) for a,b in mapping_rows)
        if asset.source_space=="SOURCE_RIG_TRACKS_V1":
            if not mapping:
                raise QualificationError("MOTION_COMPILE_SOURCE_RIG_RETARGET_MISSING")
            if len({a for a,_ in mapping})!=len(mapping) or len({b for _,b in mapping})!=len(mapping):
                raise QualificationError("MOTION_COMPILE_RETARGET_MAP_NOT_ONE_TO_ONE")
            if any(not a or b not in known for a,b in mapping):
                raise QualificationError("MOTION_COMPILE_RETARGET_MAP_TARGET_INVALID")
        elif mapping:
            raise QualificationError("MOTION_COMPILE_RETARGET_MAP_NOT_APPLICABLE")
    contact_ids=set()
    for row in value.contact_constraints:
        if row.contact_id in contact_ids or row.clip_id not in assets or row.canonical_joint_id not in known or row.mode not in CONTACT_MODES:
            raise QualificationError("MOTION_COMPILE_CONTACT_IDENTITY_INVALID")
        contact_ids.add(row.contact_id)
        duration=assets[row.clip_id].duration_seconds
        if not _finite(row.start_time_seconds) or not _finite(row.end_time_seconds) or row.start_time_seconds<0.0 or row.end_time_seconds<=row.start_time_seconds or row.end_time_seconds>duration+1e-9:
            raise QualificationError("MOTION_COMPILE_CONTACT_TIME_INVALID")
    if value.constraint_set_hash!=motion_constraint_set_hash(value):
        raise QualificationError("MOTION_COMPILE_CONSTRAINT_HASH_MISMATCH")


def validate_qualified_motion(
    value:QualifiedMotionIR,
    *,
    source_set:MotionSourceSetIR,
    source_seal:QualifiedMotionSourceSealIR,
    product_state:CanonicalPuppetStateIR,
    skeleton:QualifiedSkeletonIR,
    envelope:DeformationCapabilityEnvelopeIR,
    presentation:QualifiedPresentationGraphIR,
    constraints:MotionCompileConstraintSetIR,
)->None:
    _check_product_bindings(
        source_set=source_set,source_seal=source_seal,product_state=product_state,
        skeleton=skeleton,envelope=envelope,presentation=presentation,
    )
    expected={
        "motion_source_seal_binding_hash":source_seal.motion_source_seal_hash,
        "source_set_binding_hash":source_set.source_set_hash,
        "product_state_binding_hash":product_state.product_state_hash,
        "skeleton_binding_hash":skeleton.skeleton_lineage_hash,
        "envelope_binding_hash":envelope.envelope_lineage_hash,
        "presentation_binding_hash":presentation.presentation_lineage_hash,
        "constraint_set_binding_hash":constraints.constraint_set_hash,
    }
    for field_name,expected_hash in expected.items():
        if getattr(value,field_name)!=expected_hash:
            raise QualificationError(f"MOTION_QUALIFIED_BINDING_DRIFT:{field_name}")
    validate_motion_compile_constraint_set(
        constraints,source_set=source_set,product_state=product_state,skeleton=skeleton,
        envelope=envelope,presentation=presentation,
    )
    if len(value.clips)!=len(source_set.assets) or {c.clip_id for c in value.clips}!={a.clip_id for a in source_set.assets}:
        raise QualificationError("MOTION_QUALIFIED_CLIP_ACCOUNTING_INVALID")
    assets={a.clip_id:a for a in source_set.assets}
    root_modes=dict(constraints.root_trajectory_modes)
    retarget_maps={clip_id:dict(rows) for clip_id,rows in constraints.retarget_maps}
    envelope_by_joint=_joint_ranges(envelope)
    known={j.canonical_joint_id for j in skeleton.joints}
    for clip in value.clips:
        asset=assets[clip.clip_id]
        expected_class="MECHANICAL_PROBE_ONLY" if asset.source_kind=="INLINE_PRESET_SPEC_V1" else "ARTIST_SOURCE"
        if clip.classification!=expected_class or clip.classification not in MOTION_CLASSIFICATIONS:
            raise QualificationError("MOTION_QUALIFIED_CLIP_CLASSIFICATION_INVALID")
        if clip.root_trajectory_mode!=root_modes[clip.clip_id] or clip.root_trajectory_mode not in ROOT_TRAJECTORY_MODES:
            raise QualificationError("MOTION_QUALIFIED_ROOT_POLICY_DRIFT")
        if clip.source_asset_hash!=asset.source_asset_hash or clip.source_space!=asset.source_space:
            raise QualificationError("MOTION_QUALIFIED_SOURCE_BINDING_DRIFT")
        if clip.clip_kind!=asset.clip_kind or abs(clip.duration_seconds-asset.duration_seconds)>1e-9 or clip.loop!=asset.loop:
            raise QualificationError("MOTION_QUALIFIED_SOURCE_SEMANTICS_DRIFT")
        if not clip.tracks:
            raise QualificationError("MOTION_QUALIFIED_TRACKS_EMPTY")
        targets=set(); sources=set()
        for track in clip.tracks:
            if track.canonical_joint_id not in known or track.canonical_joint_id in targets or track.source_joint_id in sources:
                raise QualificationError("MOTION_QUALIFIED_TRACK_IDENTITY_INVALID")
            targets.add(track.canonical_joint_id); sources.add(track.source_joint_id)
            if track.channel_contract!=asset.channel_contract or not track.keyframes:
                raise QualificationError("MOTION_QUALIFIED_TRACK_CHANNEL_DRIFT")
            if asset.source_space=="SOURCE_RIG_TRACKS_V1":
                if retarget_maps[clip.clip_id].get(track.source_joint_id)!=track.canonical_joint_id:
                    raise QualificationError("MOTION_QUALIFIED_RETARGET_MAPPING_DRIFT")
            elif track.source_joint_id!=track.canonical_joint_id:
                raise QualificationError("MOTION_QUALIFIED_IDENTITY_MAPPING_DRIFT")
            if any(b.time_seconds<=a.time_seconds for a,b in zip(track.keyframes,track.keyframes[1:])):
                raise QualificationError("MOTION_QUALIFIED_TRACK_TIMES_NOT_STRICT")
            for key in track.keyframes:
                _validate_keyframe(
                    key,channels=track.channel_contract,joint_id=track.canonical_joint_id,
                    root_id=skeleton.root_id,root_mode=clip.root_trajectory_mode,
                    envelope=envelope_by_joint,duration_seconds=clip.duration_seconds,
                )
        if clip.clip_lineage_hash!=compiled_motion_clip_hash(clip):
            raise QualificationError("MOTION_QUALIFIED_CLIP_HASH_MISMATCH")
    report=dict(value.qualification_report or {})
    if report.get("status")!="PASS_COMPILED_MECHANICS_ONLY":
        raise QualificationError("MOTION_QUALIFIED_STATUS_INVALID")
    if bool(report.get("motion_quality_claimed",True)):
        raise QualificationError("MOTION_QUALIFIED_PREMATURE_QUALITY_CLAIM")
    if bool(report.get("dynamic_proof_passed",True)):
        raise QualificationError("MOTION_QUALIFIED_PREMATURE_DYNAMIC_PROOF")
    if value.motion_lineage_hash!=qualified_motion_hash(value):
        raise QualificationError("MOTION_QUALIFIED_HASH_MISMATCH")


def build_qualified_motion(
    *,
    source_set:MotionSourceSetIR,
    source_seal:QualifiedMotionSourceSealIR,
    source_payloads:Mapping[str,Mapping[str,Any]],
    product_state:CanonicalPuppetStateIR,
    skeleton:QualifiedSkeletonIR,
    envelope:DeformationCapabilityEnvelopeIR,
    presentation:QualifiedPresentationGraphIR,
    compiler_config:Mapping[str,Any]|None=None,
)->tuple[MotionCompileConstraintSetIR,QualifiedMotionIR]:
    _check_product_bindings(
        source_set=source_set,source_seal=source_seal,product_state=product_state,
        skeleton=skeleton,envelope=envelope,presentation=presentation,
    )
    cfg=dict(compiler_config or {})
    if cfg.get("retarget_maps"):
        raise QualificationError("MOTION_COMPILE_MANUAL_RETARGET_FORBIDDEN")
    generated_maps={}
    for asset in source_set.assets:
        if asset.source_space=="SOURCE_RIG_TRACKS_V1":
            payload=dict(source_payloads.get(asset.clip_id) or {})
            generated_maps[asset.clip_id]=_automatic_retarget_map(payload,skeleton)
    constraints=build_motion_compile_constraint_set(
        source_set=source_set,product_state=product_state,skeleton=skeleton,envelope=envelope,
        presentation=presentation,compiler_config=cfg,automatic_retarget_maps=generated_maps,
    )
    validate_motion_compile_constraint_set(
        constraints,source_set=source_set,product_state=product_state,skeleton=skeleton,
        envelope=envelope,presentation=presentation,
    )
    envelope_by_joint=_joint_ranges(envelope)
    root_modes=dict(constraints.root_trajectory_modes)
    retarget_maps=generated_maps
    clips=[]
    for asset in source_set.assets:
        if asset.clip_id not in source_payloads:
            raise QualificationError("MOTION_COMPILE_SOURCE_PAYLOAD_MISSING")
        payload=dict(source_payloads[asset.clip_id])
        if content_sha256(payload)!=asset.source_payload_hash:
            raise QualificationError("MOTION_COMPILE_SOURCE_PAYLOAD_HASH_DRIFT")
        unsupported=set(asset.channel_contract)-COMPILE_CHANNELS
        if unsupported:
            raise QualificationError("MOTION_COMPILE_CHANNEL_NOT_YET_SUPPORTED")
        root_mode=root_modes[asset.clip_id]
        if asset.source_kind=="INLINE_PRESET_SPEC_V1":
            tracks=_procedural_tracks(
                asset=asset,payload=payload,skeleton=skeleton,envelope=envelope_by_joint,root_mode=root_mode,
            )
            classification="MECHANICAL_PROBE_ONLY"
        elif asset.source_kind=="EXTERNAL_ARTIST_CLIP_V1":
            tracks=_external_tracks(
                asset=asset,payload=payload,skeleton=skeleton,envelope=envelope_by_joint,root_mode=root_mode,
                retarget_map=dict(retarget_maps.get(asset.clip_id) or {}),
            )
            classification="ARTIST_SOURCE"
        else:
            raise QualificationError("MOTION_COMPILE_SOURCE_KIND_UNSUPPORTED")
        clip=CompiledMotionClipIR(
            clip_id=asset.clip_id,
            clip_kind=asset.clip_kind,
            duration_seconds=asset.duration_seconds,
            loop=asset.loop,
            source_asset_hash=asset.source_asset_hash,
            source_space=asset.source_space,
            classification=classification,
            root_trajectory_mode=root_mode,
            tracks=tracks,
            clip_lineage_hash="",
            metadata={
                "delta_from_qualified_rest":True,
                "rest_geometry_mutated":False,
                "motion_quality_claimed":False,
                "stage35_dynamic_proof_required":True,
            },
        )
        clips.append(replace(clip,clip_lineage_hash=compiled_motion_clip_hash(clip)))
    clips=tuple(sorted(clips,key=lambda x:x.clip_id))
    mechanical=sum(c.classification=="MECHANICAL_PROBE_ONLY" for c in clips)
    artist=sum(c.classification=="ARTIST_SOURCE" for c in clips)
    value=QualifiedMotionIR(
        motion_source_seal_binding_hash=source_seal.motion_source_seal_hash,
        source_set_binding_hash=source_set.source_set_hash,
        product_state_binding_hash=product_state.product_state_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        envelope_binding_hash=envelope.envelope_lineage_hash,
        presentation_binding_hash=presentation.presentation_lineage_hash,
        constraint_set_binding_hash=constraints.constraint_set_hash,
        clips=clips,
        qualification_report={
            "status":"PASS_COMPILED_MECHANICS_ONLY",
            "clip_count":len(clips),
            "artist_source_clip_count":artist,
            "mechanical_probe_clip_count":mechanical,
            "all_compiled_tracks_within_deformation_envelope":True,
            "root_trajectory_policy_enforced":True,
            "static_qualified_presentation_attachment_policy":True,
            "contact_constraint_count":len(constraints.contact_constraints),
            "contact_satisfaction_proved":False,
            "dynamic_proof_passed":False,
            "motion_quality_claimed":False,
            "stage35_dynamic_proof_required":True,
        },
        motion_lineage_hash="",
        metadata={
            "runtime_export_authorized":False,
            "professional_motion_evidence_requires_artist_source_plus_stage35":True,
            "historical_rotation_only_probe_cannot_mint_quality":True,
            "historical_arap_xpbd_contact_authority_used":False,
        },
    )
    value=replace(value,motion_lineage_hash=qualified_motion_hash(value))
    validate_qualified_motion(
        value,source_set=source_set,source_seal=source_seal,product_state=product_state,
        skeleton=skeleton,envelope=envelope,presentation=presentation,constraints=constraints,
    )
    return constraints,value
