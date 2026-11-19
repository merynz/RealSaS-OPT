from __future__ import annotations

"""Explicit source-rig motion -> canonical animator-control proposal.

Retargeting is a deterministic compiled operator, not an inference heuristic.
Source joint correspondence and controllable FK/ROOT target identities MUST
be supplied and checked. It uses the pre-existing rest-frame retargeter,
including skipped source ancestors, and never changes the canonical rig,
skin, mesh or Graphics representation.

An imported source clip is an unqualified candidate. A valid retargeted
ControlMotionClipProposal is NOT proof that the input was licensed, accurate,
visually faithful, or suitable as a BUSTED motion library redistribution.
"""

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np

from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .control_motion_ir_v1 import (
    ControlMotionClipProposalIRV1,ControlMotionTrackProposalIRV1,
    ControlMotionKeyProposalIRV1,seal_control_motion_proposal_v1,
    qualify_control_motion_v1,
)
from .hashing import content_sha256
from .motion_retarget_frames_v3 import compile_retargeted_keys
from .types import QualificationError


def retarget_source_motion_to_canonical_controls_v1(
    source_payload: Mapping[str, object],
    *,
    rig: CanonicalRigIRV2,
    mechanics: CanonicalMechanicsIRV1,
    target_joint_to_source: Mapping[str, str],
    target_joint_to_control: Mapping[str, str],
    object_rotation: object,
    target_body_scale: float,
    clip_id: str,
    display_name: str,
    intent: str,
    loop: bool,
) -> ControlMotionClipProposalIRV1:
    if (
        not isinstance(rig,CanonicalRigIRV2)
        or not isinstance(mechanics,CanonicalMechanicsIRV1)
        or mechanics.rig_lineage_hash!=rig.rig_lineage_hash
        or mechanics.control_program_hash!=rig.control_program_hash
        or not isinstance(source_payload,Mapping)
        or not isinstance(target_joint_to_source,Mapping)
        or not isinstance(target_joint_to_control,Mapping)
        or not target_joint_to_source
        or set(target_joint_to_source)!=set(target_joint_to_control)
        or any(type(x) is not str or not x for d in (
            target_joint_to_source,target_joint_to_control,
        ) for x in d.values())
        or type(clip_id) is not str or not clip_id.strip()
        or type(display_name) is not str or not display_name.strip()
        or type(intent) is not str or not intent.strip()
        or type(loop) is not bool
        or type(target_body_scale) not in (float,int)
        or not math.isfinite(target_body_scale) or target_body_scale<=0
    ):
        raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_CONTRACT_INVALID")
    source=source_payload.get("source_skeleton")
    tracks=source_payload.get("tracks")
    if not isinstance(source,(list,tuple)) or not isinstance(tracks,(list,tuple)):
        raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_NATIVE_SOURCE_REQUIRED")
    joints={j.canonical_joint_id:j for j in rig.joints}
    frames={f.canonical_joint_id:f for f in rig.local_frames}
    controls={c.canonical_control_id:c for c in rig.controls}
    if set(frames)!=set(joints):
        raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_FRAME_COVERAGE")
    if not set(target_joint_to_source).issubset(joints):
        raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_TARGET_JOINT_UNKNOWN")
    if len(set(target_joint_to_source.values()))!=len(target_joint_to_source):
        raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_SOURCE_JOINT_DUPLICATE")
    if len(set(target_joint_to_control.values()))!=len(target_joint_to_control):
        raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_CONTROL_DUPLICATE")
    for jid,cid in target_joint_to_control.items():
        control=controls.get(cid)
        if (
            control is None or control.control_kind not in ("ROOT","FK")
            or control.target_canonical_joint_ids!=(jid,)
        ):
            raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_NOT_EXACT_FK_CONTROL")
    target_frames={
        jid:SimpleNamespace(
            parent_id=j.parent_canonical_joint_id,
            rest_position=j.P,
            rotation_matrix=np.asarray(frames[jid].basis,dtype=np.float64),
        ) for jid,j in joints.items()
    }
    try:
        translated=compile_retargeted_keys(
            source_payload,
            target_frames=target_frames,
            target_to_source=dict(target_joint_to_source),
            object_rotation=object_rotation,
            target_body_scale=float(target_body_scale),
        )
    except (KeyError,TypeError,ValueError,IndexError) as exc:
        raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_SOURCE_INVALID") from exc
    if set(translated)!=set(target_joint_to_source):
        raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_OUTPUT_ADDRESS_DRIFT")
    rows=[]
    max_t=0.
    for jid in sorted(translated):
        keys=[]
        for key in translated[jid]:
            time=float(key["time_seconds"])
            xyzw=tuple(map(float,key["local_rotation_quat_xyzw"]))
            xyz=tuple(map(float,key["local_translation_xyz"]))
            scale=tuple(map(float,key["local_scale_xyz"]))
            if not math.isfinite(time) or time < 0 or len(xyzw)!=4:
                raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_KEY_INVALID")
            max_t=max(max_t,time)
            keys.append(ControlMotionKeyProposalIRV1(
                time_seconds=time,
                translation=xyz,
                rotation_quat_wxyz=(xyzw[3],xyzw[0],xyzw[1],xyzw[2]),
                scale=scale,
            ))
        if not keys or keys[0].time_seconds!=0.:
            raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_FIRST_KEY_NOT_ZERO")
        rows.append(ControlMotionTrackProposalIRV1(
            proposal_track_id="RETARGET:"+jid,
            canonical_control_id=target_joint_to_control[jid],
            keys=tuple(keys),
            metadata={
                "explicit_source_joint_id":target_joint_to_source[jid],
                "no_automatic_source_joint_guess":True,
            },
        ))
    if not 0<max_t<=86400:
        raise QualificationError("MOTION_SOURCE_CONTROL_RETARGET_DURATION_INVALID")
    source_digest=content_sha256(dict(source_payload))
    map_digest=content_sha256({
        "target_joint_to_source":dict(target_joint_to_source),
        "target_joint_to_control":dict(target_joint_to_control),
        "object_rotation":np.asarray(object_rotation).tolist(),
        "target_body_scale":target_body_scale,
    })
    proposal=seal_control_motion_proposal_v1(ControlMotionClipProposalIRV1(
        proposal_clip_id=clip_id,
        display_name=display_name,
        intent=intent,
        duration_seconds=max_t,
        loop=loop,
        tracks=tuple(rows),
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
        metadata={
            "retarget_operator":"SOURCE_REST_FRAME_TO_EXPLICIT_FK_CONTROL_V1",
            "source_clip_content_sha256":source_digest,
            "correspondence_content_sha256":map_digest,
            "source_animation_rights_qualified":False,
            "source_motion_independently_verified":False,
            "visual_motion_parity_proven":False,
            "joint_curve_compiler_shortcut":False,
        },
    ))
    # Existing canonical Motion court validates all times, unit quaternions,
    # control identities, hashes and Mechanics lineage.
    qualify_control_motion_v1(proposal,rig=rig,mechanics=mechanics)
    return proposal


__all__=["retarget_source_motion_to_canonical_controls_v1"]
