from __future__ import annotations

"""Canonical full-3D motion compilation for Stage 40.

Professional motion is represented as local joint-frame quaternion deltas. Source
preset rigs are motion evidence only; the target skeleton remains the exact
QualifiedSkeletonIR. Source clips are canonicalized into REALSAS_OBJECT_FRAME_V1
before this module sees them.

This module deliberately does not make a motion-quality claim. Stage 41 executes
compiled motion on exact M/B and owns dynamic/product proof.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from .canonical_puppet_state_v1 import CanonicalPuppetStateIR, canonical_puppet_state_hash
from .hashing import content_sha256
from .motion_source_v1 import (
    MotionSourceSetIR,
    QualifiedMotionSourceSealIR,
    motion_source_seal_hash,
    validate_motion_source_set,
)
from .product_authority_v1 import (
    DeformationCapabilityEnvelopeIR,
    deformation_envelope_lineage_hash,
)
from .product_state_v2 import (
    QualifiedPresentationGraphV2IR,
    presentation_graph_v2_hash,
)
from .types import QualifiedSkeletonIR, QualificationError

Json=dict[str,Any]
Vec3=tuple[float,float,float]
Quat=tuple[float,float,float,float]

V2_CHANNELS={"LOCAL_ROTATION_QUAT_XYZW","LOCAL_TRANSLATION_XYZ","LOCAL_SCALE_XYZ"}
ROOT_TRAJECTORY_MODES={"IN_PLACE","SOURCE_EXPLICIT"}
CONTACT_MODES={"PLANT_2D"}


@dataclass(frozen=True)
class MotionKeyframe3DIR:
    time_seconds:float
    local_rotation_quat_xyzw:Quat=(0.0,0.0,0.0,1.0)
    local_translation_xyz:Vec3=(0.0,0.0,0.0)
    local_scale_xyz:Vec3=(1.0,1.0,1.0)
    metadata:Json=field(default_factory=dict)
    schema_version:str="RealSaS.MotionKeyframeIR.v2"
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class CanonicalJointTrack3DIR:
    canonical_joint_id:str
    source_joint_id:str
    channel_contract:tuple[str,...]
    keyframes:tuple[MotionKeyframe3DIR,...]
    metadata:Json=field(default_factory=dict)
    schema_version:str="RealSaS.CanonicalJointTrackIR.v2"
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class MotionContactConstraintV2IR:
    contact_id:str
    clip_id:str
    canonical_joint_id:str
    start_time_seconds:float
    end_time_seconds:float
    mode:str
    metadata:Json=field(default_factory=dict)
    schema_version:str="RealSaS.MotionContactConstraintIR.v2"
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class MotionCompileConstraintSetV2IR:
    product_state_binding_hash:str
    skeleton_binding_hash:str
    envelope_binding_hash:str
    presentation_binding_hash:str
    root_trajectory_modes:tuple[tuple[str,str],...]
    retarget_maps:tuple[tuple[str,tuple[tuple[str,str],...]],...]
    contact_constraints:tuple[MotionContactConstraintV2IR,...]
    dynamic_attachment_policy:str
    constraint_set_hash:str
    schema_version:str="RealSaS.MotionCompileConstraintSetIR.v2"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class CompiledMotionClip3DIR:
    clip_id:str
    clip_kind:str
    duration_seconds:float
    loop:bool
    source_asset_hash:str
    source_space:str
    classification:str
    root_trajectory_mode:str
    tracks:tuple[CanonicalJointTrack3DIR,...]
    clip_lineage_hash:str
    schema_version:str="RealSaS.CompiledMotionClipIR.v2"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedMotion3DIR:
    motion_source_seal_binding_hash:str
    source_set_binding_hash:str
    product_state_binding_hash:str
    skeleton_binding_hash:str
    envelope_binding_hash:str
    presentation_binding_hash:str
    constraint_set_binding_hash:str
    clips:tuple[CompiledMotionClip3DIR,...]
    qualification_report:Json
    motion_lineage_hash:str
    schema_version:str="RealSaS.QualifiedMotionIR.v2"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def _hash_without(value,field_name):
    payload=value.to_dict()
    payload.pop(field_name,None)
    return content_sha256(payload)


def motion_constraint_set_v2_hash(value):
    return _hash_without(value,"constraint_set_hash")


def compiled_motion_clip_v2_hash(value):
    return _hash_without(value,"clip_lineage_hash")


def qualified_motion_v2_hash(value):
    return _hash_without(value,"motion_lineage_hash")


def _finite_vec(raw,n,code):
    if not isinstance(raw,(list,tuple)) or len(raw)!=n:
        raise QualificationError(code)
    out=tuple(float(x) for x in raw)
    if not all(math.isfinite(x) for x in out):
        raise QualificationError(code)
    return out


def _unit_quat(raw,code):
    q=np.asarray(_finite_vec(raw,4,code),dtype=np.float64)
    norm=float(np.linalg.norm(q))
    if abs(norm-1.0)>1e-5:
        raise QualificationError(code)
    q=q/norm
    return tuple(map(float,q))


def _canonicalize_quat_sequence(keys):
    out=[]
    prev=None
    for key in keys:
        q=np.asarray(key.local_rotation_quat_xyzw,dtype=np.float64)
        if prev is not None and float(np.dot(prev,q))<0.0:
            q=-q
        prev=q
        out.append(replace(key,local_rotation_quat_xyzw=tuple(map(float,q))))
    return tuple(out)


def _check_bindings(*,source_set,source_seal,product_state,skeleton,envelope,presentation):
    validate_motion_source_set(source_set)
    if source_seal.motion_source_seal_hash!=motion_source_seal_hash(source_seal):
        raise QualificationError("MOTION_V2_SOURCE_SEAL_HASH_DRIFT")
    if product_state.product_state_hash!=canonical_puppet_state_hash(product_state):
        raise QualificationError("MOTION_V2_PRODUCT_STATE_HASH_DRIFT")
    if envelope.envelope_lineage_hash!=deformation_envelope_lineage_hash(envelope):
        raise QualificationError("MOTION_V2_ENVELOPE_HASH_DRIFT")
    if presentation.presentation_lineage_hash!=presentation_graph_v2_hash(presentation):
        raise QualificationError("MOTION_V2_PRESENTATION_HASH_DRIFT")
    checks=(
        (source_seal.source_set_binding_hash,source_set.source_set_hash,"SOURCE_SET"),
        (source_seal.product_state_binding_hash,product_state.product_state_hash,"SOURCE_PRODUCT"),
        (product_state.skeleton_lineage_hash,skeleton.skeleton_lineage_hash,"PRODUCT_SKELETON"),
        (product_state.deformation_envelope_lineage_hash,envelope.envelope_lineage_hash,"PRODUCT_ENVELOPE"),
        (envelope.skeleton_lineage_hash,skeleton.skeleton_lineage_hash,"ENVELOPE_SKELETON"),
        (presentation.mechanical_state_binding_hash,product_state.product_state_hash,"PRESENTATION_MECHANICAL_STATE"),
        (presentation.skeleton_binding_hash,skeleton.skeleton_lineage_hash,"PRESENTATION_SKELETON"),
    )
    for actual,wanted,label in checks:
        if actual!=wanted:
            raise QualificationError("MOTION_V2_BINDING_DRIFT:"+label)
    if source_seal.qualification_report.get("status")!="PASS_SOURCE_IDENTITY_AND_PRECONDITION_ONLY":
        raise QualificationError("MOTION_V2_SOURCE_SEAL_NOT_AUTHORIZED")


def _tree(rows,*,id_key,parent_key,pos_key):
    rows=tuple(dict(r) for r in rows)
    ids=[str(r[id_key]) for r in rows]
    if not ids or len(ids)!=len(set(ids)):
        raise QualificationError("MOTION_V2_SKELETON_ID_INVALID")
    by={str(r[id_key]):r for r in rows}
    parent={}
    children={jid:[] for jid in ids}
    roots=[]
    pos={}
    for jid in ids:
        p=by[jid].get(parent_key)
        p=None if p in (None,"") else str(p)
        if p is not None and p not in by:
            raise QualificationError("MOTION_V2_SKELETON_PARENT_UNKNOWN")
        parent[jid]=p
        if p is None:
            roots.append(jid)
        else:
            children[p].append(jid)
        xyz=np.asarray(by[jid][pos_key],dtype=np.float64)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():
            raise QualificationError("MOTION_V2_SKELETON_POSITION_INVALID")
        pos[jid]=xyz
    if len(roots)!=1:
        raise QualificationError("MOTION_V2_SKELETON_ROOT_INVALID")
    root=roots[0]
    depth={}
    stack=[(root,0)]
    while stack:
        jid,d=stack.pop()
        if jid in depth:
            raise QualificationError("MOTION_V2_SKELETON_CYCLE")
        depth[jid]=d
        for child in children[jid]:
            stack.append((child,d+1))
    if len(depth)!=len(ids):
        raise QualificationError("MOTION_V2_SKELETON_DISCONNECTED")
    root_pos=pos[root]
    scale=max((float(np.linalg.norm(p-root_pos)) for p in pos.values()),default=1.0)
    scale=max(scale,1e-9)
    max_depth=max(depth.values()) or 1
    subtree={}
    def subtree_size(jid):
        if jid in subtree:
            return subtree[jid]
        subtree[jid]=1+sum(subtree_size(c) for c in children[jid])
        return subtree[jid]
    subtree_size(root)
    max_children=max(max((len(v) for v in children.values()),default=1),1)
    feat={}
    for jid in ids:
        q=(pos[jid]-root_pos)/scale
        feat[jid]=np.asarray([
            q[0],q[1],q[2],
            float(depth[jid])/max_depth,
            float(len(children[jid]))/max_children,
            float(subtree[jid])/len(ids),
            1.0 if jid==root else 0.0,
        ],dtype=np.float64)
    return ids,parent,children,pos,feat,root


def _is_ancestor(parent,ancestor,node):
    cur=node
    while cur is not None:
        if cur==ancestor:
            return True
        cur=parent[cur]
    return False


def automatic_retarget_map_v2(payload:Mapping[str,Any],skeleton:QualifiedSkeletonIR):
    if str(payload.get("coordinate_frame") or "")!="REALSAS_OBJECT_FRAME_V1":
        raise QualificationError("MOTION_V2_SOURCE_COORDINATE_FRAME_INVALID")
    src_rows=tuple(payload.get("source_skeleton") or ())
    if not src_rows:
        raise QualificationError("MOTION_V2_SOURCE_SKELETON_MISSING")
    sids,spar,schildren,spos,sfeat,sroot=_tree(
        src_rows,
        id_key="source_joint_id",
        parent_key="parent_source_joint_id",
        pos_key="rest_position",
    )
    tgt_rows=tuple({
        "joint_id":j.canonical_joint_id,
        "parent_id":j.parent_canonical_id,
        "position":j.position,
    } for j in skeleton.joints)
    tids,tpar,tchildren,tpos,tfeat,troot=_tree(
        tgt_rows,id_key="joint_id",parent_key="parent_id",pos_key="position"
    )
    if len(sids)<len(tids):
        raise QualificationError("MOTION_V2_SOURCE_RIG_TOO_SMALL")
    cost=np.zeros((len(tids),len(sids)),dtype=np.float64)
    for i,tid in enumerate(tids):
        tf=tfeat[tid]
        for j,sid in enumerate(sids):
            sf=sfeat[sid]
            c=2.0*float(np.linalg.norm(tf[:3]-sf[:3]))
            c+=0.45*abs(float(tf[3]-sf[3]))
            c+=0.35*abs(float(tf[4]-sf[4]))
            c+=0.35*abs(float(tf[5]-sf[5]))
            if (tid==troot)!=(sid==sroot):
                c+=1000.0
            if (
                abs(float(tf[0]))>0.05
                and abs(float(sf[0]))>0.05
                and math.copysign(1.0,float(tf[0]))!=math.copysign(1.0,float(sf[0]))
            ):
                c+=4.0
            cost[i,j]=c
    target_rows_idx,source_cols=linear_sum_assignment(cost)
    if len(target_rows_idx)!=len(tids):
        raise QualificationError("MOTION_V2_RETARGET_ASSIGNMENT_INCOMPLETE")
    source_to_target={
        sids[int(source_col)]:tids[int(target_row)]
        for target_row,source_col in zip(target_rows_idx,source_cols)
    }
    target_to_source={target:source for source,target in source_to_target.items()}
    if target_to_source.get(troot)!=sroot:
        raise QualificationError("MOTION_V2_RETARGET_ROOT_MISMATCH")
    for tid in tids:
        target_parent=tpar[tid]
        if target_parent is None:
            continue
        source_child=target_to_source[tid]
        source_parent=target_to_source[target_parent]
        if not _is_ancestor(spar,source_parent,source_child):
            raise QualificationError("MOTION_V2_RETARGET_ANCESTRY_MISMATCH")
    for target_row,source_col in zip(target_rows_idx,source_cols):
        row=np.sort(cost[int(target_row)])
        if len(row)>1 and float(row[1]-row[0])<0.01:
            raise QualificationError("MOTION_V2_RETARGET_AMBIGUOUS")
    report={
        "schema":"RealSaS.AutomaticRetargetReport.v2",
        "source_joint_count":len(sids),
        "target_joint_count":len(tids),
        "mapped_joint_count":len(source_to_target),
        "unused_source_joint_count":len(sids)-len(tids),
        "total_assignment_cost":float(
            sum(cost[int(i),int(j)] for i,j in zip(target_rows_idx,source_cols))
        ),
        "source_to_target":tuple(sorted(source_to_target.items())),
        "topology_rule":"TARGET_PARENT_MAPS_TO_SOURCE_ANCESTOR",
        "coordinate_frame":"REALSAS_OBJECT_FRAME_V1",
    }
    report["report_hash"]=content_sha256(report)
    return source_to_target,report


def _target_body_scale(skeleton):
    root=next(j for j in skeleton.joints if j.canonical_joint_id==skeleton.root_id)
    root_pos=np.asarray(root.position,dtype=np.float64)
    return max(
        max(
            (float(np.linalg.norm(np.asarray(j.position,dtype=np.float64)-root_pos))
             for j in skeleton.joints),
            default=1.0,
        ),
        1e-9,
    )


def _parse_key(raw,channels,*,translation_scale):
    allowed={
        "time_seconds",
        "local_rotation_quat_xyzw",
        "local_translation_xyz",
        "local_scale_xyz",
        "metadata",
    }
    if set(raw)-allowed:
        raise QualificationError("MOTION_V2_KEY_FIELD_UNSUPPORTED")
    quat=_unit_quat(
        raw.get("local_rotation_quat_xyzw",(0,0,0,1)),
        "MOTION_V2_QUAT_INVALID",
    )
    translation=_finite_vec(
        raw.get("local_translation_xyz",(0,0,0)),
        3,
        "MOTION_V2_TRANSLATION_INVALID",
    )
    scale=_finite_vec(
        raw.get("local_scale_xyz",(1,1,1)),
        3,
        "MOTION_V2_SCALE_INVALID",
    )
    if (
        "LOCAL_ROTATION_QUAT_XYZW" not in channels
        and np.linalg.norm(np.asarray(quat[:3],dtype=np.float64))>1e-12
    ):
        raise QualificationError("MOTION_V2_UNDECLARED_ROTATION")
    if (
        "LOCAL_TRANSLATION_XYZ" not in channels
        and any(abs(x)>1e-12 for x in translation)
    ):
        raise QualificationError("MOTION_V2_UNDECLARED_TRANSLATION")
    if (
        "LOCAL_SCALE_XYZ" not in channels
        and any(abs(x-1.0)>1e-12 for x in scale)
    ):
        raise QualificationError("MOTION_V2_UNDECLARED_SCALE")
    translation=tuple(float(x)*float(translation_scale) for x in translation)
    return MotionKeyframe3DIR(
        float(raw["time_seconds"]),
        quat,
        translation,
        scale,
        dict(raw.get("metadata") or {}),
    )


def _contacts(clip_id,rows,skeleton,duration):
    known={j.canonical_joint_id for j in skeleton.joints}
    out=[]
    seen=set()
    for raw in tuple(rows or ()):
        row=dict(raw)
        contact_id=str(row.get("contact_id") or "")
        joint_id=str(row.get("canonical_joint_id") or "")
        start=float(row.get("start_time_seconds",-1))
        end=float(row.get("end_time_seconds",-1))
        mode=str(row.get("mode") or "")
        if (
            not contact_id
            or contact_id in seen
            or joint_id not in known
            or mode not in CONTACT_MODES
        ):
            raise QualificationError("MOTION_V2_CONTACT_IDENTITY_INVALID")
        if (
            not (math.isfinite(start) and math.isfinite(end))
            or start<0
            or end<=start
            or end>duration+1e-9
        ):
            raise QualificationError("MOTION_V2_CONTACT_TIME_INVALID")
        seen.add(contact_id)
        out.append(
            MotionContactConstraintV2IR(
                contact_id,clip_id,joint_id,start,end,mode,
                metadata={"satisfaction_proved":False},
            )
        )
    return tuple(out)


def build_qualified_motion_v2(
    *,
    source_set:MotionSourceSetIR,
    source_seal:QualifiedMotionSourceSealIR,
    source_payloads:Mapping[str,Mapping[str,Any]],
    product_state:CanonicalPuppetStateIR,
    skeleton:QualifiedSkeletonIR,
    envelope:DeformationCapabilityEnvelopeIR,
    presentation:QualifiedPresentationGraphV2IR,
    compiler_config:Mapping[str,Any]|None=None,
):
    _check_bindings(
        source_set=source_set,
        source_seal=source_seal,
        product_state=product_state,
        skeleton=skeleton,
        envelope=envelope,
        presentation=presentation,
    )
    config=dict(compiler_config or {})
    if set(config)-{"root_trajectory_modes","contacts"}:
        raise QualificationError("MOTION_V2_CONFIG_UNSUPPORTED")
    root_config=dict(config.get("root_trajectory_modes") or {})
    contact_config=dict(config.get("contacts") or {})

    retarget_maps={}
    retarget_reports={}
    clips=[]
    contact_rows=[]
    root_rows=[]
    target_scale=_target_body_scale(skeleton)
    known={j.canonical_joint_id for j in skeleton.joints}

    for asset in source_set.assets:
        if (
            asset.source_kind!="EXTERNAL_ARTIST_CLIP_V1"
            or asset.source_space!="SOURCE_RIG_TRACKS_V2"
        ):
            raise QualificationError("MOTION_V2_PROFESSIONAL_SOURCE_REQUIRED")
        payload=dict(source_payloads.get(asset.clip_id) or {})
        if content_sha256(payload)!=asset.source_payload_hash:
            raise QualificationError("MOTION_V2_SOURCE_PAYLOAD_HASH_DRIFT")
        if str(payload.get("schema") or "")!="RealSaS.MotionSourceClip.v2":
            raise QualificationError("MOTION_V2_SOURCE_SCHEMA_INVALID")
        if tuple(map(str,payload.get("channel_contract") or ()))!=asset.channel_contract:
            raise QualificationError("MOTION_V2_CHANNEL_CONTRACT_DRIFT")
        if (
            set(asset.channel_contract)-V2_CHANNELS
            or "LOCAL_ROTATION_QUAT_XYZW" not in asset.channel_contract
        ):
            raise QualificationError("MOTION_V2_CHANNEL_CONTRACT_INVALID")

        mapping,report=automatic_retarget_map_v2(payload,skeleton)
        retarget_maps[asset.clip_id]=mapping
        retarget_reports[asset.clip_id]=report

        root_mode=str(root_config.get(asset.clip_id) or "IN_PLACE")
        if root_mode not in ROOT_TRAJECTORY_MODES:
            raise QualificationError("MOTION_V2_ROOT_MODE_INVALID")
        root_rows.append((asset.clip_id,root_mode))

        raw_tracks=tuple(payload.get("tracks") or ())
        source_tracks={
            str(row.get("source_joint_id") or ""):dict(row)
            for row in raw_tracks
        }
        if len(source_tracks)!=len(raw_tracks) or "" in source_tracks:
            raise QualificationError("MOTION_V2_SOURCE_TRACK_ID_INVALID")

        compiled_tracks=[]
        used_targets=set()
        for source_id,target_id in sorted(mapping.items(),key=lambda item:item[1]):
            if target_id not in known or target_id in used_targets:
                raise QualificationError("MOTION_V2_TARGET_TRACK_ID_INVALID")
            row=source_tracks.get(source_id)
            if row is None:
                raise QualificationError("MOTION_V2_MAPPED_SOURCE_TRACK_MISSING")
            if set(row)-{"source_joint_id","keyframes","metadata"}:
                raise QualificationError("MOTION_V2_SOURCE_TRACK_FIELD_UNSUPPORTED")
            keys=tuple(
                _parse_key(
                    dict(raw_key),
                    asset.channel_contract,
                    translation_scale=target_scale,
                )
                for raw_key in tuple(row.get("keyframes") or ())
            )
            if not keys:
                raise QualificationError("MOTION_V2_SOURCE_TRACK_KEYS_EMPTY")
            keys=_canonicalize_quat_sequence(keys)
            if any(
                b.time_seconds<=a.time_seconds
                for a,b in zip(keys,keys[1:])
            ):
                raise QualificationError("MOTION_V2_TRACK_TIMES_NOT_STRICT")
            if (
                keys[0].time_seconds<-1e-9
                or keys[-1].time_seconds>asset.duration_seconds+1e-9
            ):
                raise QualificationError("MOTION_V2_TRACK_TIME_OUT_OF_RANGE")
            if (
                target_id!=skeleton.root_id
                and any(
                    np.linalg.norm(np.asarray(k.local_translation_xyz,dtype=np.float64))>1e-9
                    for k in keys
                )
            ):
                raise QualificationError("MOTION_V2_NONROOT_TRANSLATION_FORBIDDEN")
            if any(
                max(abs(x-1.0) for x in k.local_scale_xyz)>1e-9
                for k in keys
            ):
                raise QualificationError("MOTION_V2_SCALE_ANIMATION_NOT_YET_QUALIFIED")
            if (
                target_id==skeleton.root_id
                and root_mode=="IN_PLACE"
                and any(
                    np.linalg.norm(np.asarray(k.local_translation_xyz,dtype=np.float64))>1e-9
                    for k in keys
                )
            ):
                raise QualificationError("MOTION_V2_IN_PLACE_ROOT_TRANSLATION")
            compiled_tracks.append(
                CanonicalJointTrack3DIR(
                    target_id,
                    source_id,
                    asset.channel_contract,
                    keys,
                    metadata={
                        "retargeted":True,
                        "retarget_report_hash":report["report_hash"],
                        "rotation_semantics":"LOCAL_JOINT_FRAME_DELTA_QUATERNION",
                        "translation_semantics":"LOCAL_JOINT_FRAME_DELTA_TRANSLATION",
                        "translation_units":"TARGET_BODY_SCALE",
                        "interpolation":"SLERP_ROTATION_LINEAR_TRANSLATION_V1",
                    },
                )
            )
            used_targets.add(target_id)

        contact_rows.extend(
            _contacts(
                asset.clip_id,
                contact_config.get(asset.clip_id) or (),
                skeleton,
                asset.duration_seconds,
            )
        )
        clip=CompiledMotionClip3DIR(
            clip_id=asset.clip_id,
            clip_kind=asset.clip_kind,
            duration_seconds=float(asset.duration_seconds),
            loop=bool(asset.loop),
            source_asset_hash=asset.source_asset_hash,
            source_space=asset.source_space,
            classification="ARTIST_SOURCE",
            root_trajectory_mode=root_mode,
            tracks=tuple(sorted(compiled_tracks,key=lambda x:x.canonical_joint_id)),
            clip_lineage_hash="",
            metadata={
                "source_motion_semantics_only":True,
                "target_skeleton_authority":skeleton.skeleton_lineage_hash,
                "retarget_report":report,
                "motion_quality_claimed":False,
                "stage35_dynamic_proof_required":True,
            },
        )
        clips.append(
            replace(
                clip,
                clip_lineage_hash=compiled_motion_clip_v2_hash(clip),
            )
        )

    constraints=MotionCompileConstraintSetV2IR(
        product_state_binding_hash=product_state.product_state_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        envelope_binding_hash=envelope.envelope_lineage_hash,
        presentation_binding_hash=presentation.presentation_lineage_hash,
        root_trajectory_modes=tuple(sorted(root_rows)),
        retarget_maps=tuple(
            sorted(
                (
                    clip_id,
                    tuple(sorted(mapping.items())),
                )
                for clip_id,mapping in retarget_maps.items()
            )
        ),
        contact_constraints=tuple(
            sorted(contact_rows,key=lambda row:(row.clip_id,row.contact_id))
        ),
        dynamic_attachment_policy="STATIC_QUALIFIED_PRESENTATION_ONLY_V1",
        constraint_set_hash="",
        metadata={
            "manual_retarget_forbidden":True,
            "legacy_scalar_envelope_is_not_v2_motion_limit":True,
            "exact_motion_deformation_authority":"STAGE35_EXACT_CLIP_PROOF",
        },
    )
    constraints=replace(
        constraints,
        constraint_set_hash=motion_constraint_set_v2_hash(constraints),
    )

    value=QualifiedMotion3DIR(
        motion_source_seal_binding_hash=source_seal.motion_source_seal_hash,
        source_set_binding_hash=source_set.source_set_hash,
        product_state_binding_hash=product_state.product_state_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        envelope_binding_hash=envelope.envelope_lineage_hash,
        presentation_binding_hash=presentation.presentation_lineage_hash,
        constraint_set_binding_hash=constraints.constraint_set_hash,
        clips=tuple(sorted(clips,key=lambda clip:clip.clip_id)),
        qualification_report={
            "status":"PASS_COMPILED_FULL_3D_MECHANICS_ONLY",
            "clip_count":len(clips),
            "artist_source_clip_count":len(clips),
            "mechanical_probe_clip_count":0,
            "full_3d_local_quaternion_motion":True,
            "manual_retarget_used":False,
            "source_rig_is_final_skeleton_authority":False,
            "target_skeleton_is_exact_qualified_skeleton":True,
            "dynamic_proof_passed":False,
            "motion_quality_claimed":False,
        },
        motion_lineage_hash="",
        metadata={
            "runtime_export_authorized":False,
            "professional_motion_evidence_requires_stage35":True,
            "canonical_motion_ontology":"SINGLE_3D_TARGET_SKELETON__LOCAL_QUATERNION_DELTAS",
        },
    )
    value=replace(
        value,
        motion_lineage_hash=qualified_motion_v2_hash(value),
    )
    return constraints,value
