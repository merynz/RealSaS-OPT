from __future__ import annotations

"""Canonical Motion authority over animator controls, not deform joints."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping
import math

from .control_motion_ir_v1 import QualifiedControlMotionIRV1, qualified_control_motion_hash_v1
from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class CanonicalControlMotionKeyIRV2:
    time_seconds: float
    translation: Vec3
    rotation_quat_wxyz: Quat
    scale: Vec3
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalControlMotionTrackIRV2:
    canonical_motion_track_id: str
    canonical_control_id: str
    keys: tuple[CanonicalControlMotionKeyIRV2, ...]
    source_proposal_track_id: str
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalMotionIRV2:
    canonical_clip_id: str
    display_name: str
    intent: str
    duration_seconds: float
    loop: bool
    control_tracks: tuple[CanonicalControlMotionTrackIRV2, ...]
    rig_lineage_hash: str
    control_program_hash: str
    mechanics_lineage_hash: str
    source_proposal_hash: str
    motion_qualification_hash: str
    motion_lineage_hash: str
    schema_version: str = "RealSaS.CanonicalMotionIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_motion_lineage_hash_v2(value: CanonicalMotionIRV2) -> str:
    payload = value.to_dict()
    payload.pop("motion_lineage_hash", None)
    return content_sha256(payload)


def _stable_id(prefix: str, qualification_hash: str, source_id: str) -> str:
    return prefix + ":" + content_sha256({
        "qualification_hash": qualification_hash,
        "source_id": source_id,
    })[:20]


def mint_canonical_motion_v2(qualified: QualifiedControlMotionIRV1) -> CanonicalMotionIRV2:
    if qualified.qualification_hash != qualified_control_motion_hash_v1(qualified):
        raise QualificationError("CANONICAL_MOTION_V2_QUALIFICATION_HASH_DRIFT")
    report = dict(qualified.qualification_report or {})
    if report.get("status") != "PASS_CONTROL_MOTION_VALIDATED":
        raise QualificationError("CANONICAL_MOTION_V2_NOT_QUALIFIED")
    for forbidden in (
        "control_synthesis_performed",
        "track_retargeting_performed",
        "joint_curve_bake_performed",
        "constraint_solver_executed",
        "keyframe_repair_performed",
    ):
        if report.get(forbidden) is not False:
            raise QualificationError(f"CANONICAL_MOTION_V2_COMPILER_OVERREACH:{forbidden}")

    proposal = qualified.proposal
    qh = qualified.qualification_hash
    tracks = tuple(
        CanonicalControlMotionTrackIRV2(
            canonical_motion_track_id=_stable_id("MT", qh, track.proposal_track_id),
            canonical_control_id=track.canonical_control_id,
            keys=tuple(
                CanonicalControlMotionKeyIRV2(
                    time_seconds=float(key.time_seconds),
                    translation=tuple(map(float, key.translation)),  # type: ignore[arg-type]
                    rotation_quat_wxyz=tuple(map(float, key.rotation_quat_wxyz)),  # type: ignore[arg-type]
                    scale=tuple(map(float, key.scale)),  # type: ignore[arg-type]
                    metadata=dict(key.metadata or {}),
                )
                for key in track.keys
            ),
            source_proposal_track_id=track.proposal_track_id,
            metadata=dict(track.metadata or {}),
        )
        for track in proposal.tracks
    )
    value = CanonicalMotionIRV2(
        canonical_clip_id=_stable_id("MOTION", qh, proposal.proposal_clip_id),
        display_name=proposal.display_name,
        intent=proposal.intent,
        duration_seconds=float(proposal.duration_seconds),
        loop=bool(proposal.loop),
        control_tracks=tracks,
        rig_lineage_hash=proposal.rig_lineage_hash,
        control_program_hash=proposal.control_program_hash,
        mechanics_lineage_hash=proposal.mechanics_lineage_hash,
        source_proposal_hash=proposal.proposal_lineage_hash,
        motion_qualification_hash=qh,
        motion_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_MOTION",
            "editable_motion_space": "ANIMATOR_CONTROLS",
            "direct_joint_track_authority": False,
            "derived_joint_bakes_allowed": True,
            "constraint_execution_owner": "MECHANICS",
            "compiler_control_synthesis_performed": False,
            "compiler_retargeting_performed": False,
        },
    )
    return replace(value, motion_lineage_hash=canonical_motion_lineage_hash_v2(value))


def canonical_motion_from_dict_v2(payload: Mapping[str, Any]) -> CanonicalMotionIRV2:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalMotionIR.v2":
        raise QualificationError("CANONICAL_MOTION_V2_SCHEMA_DRIFT")

    def vec3(raw, code):
        out = tuple(map(float, raw))
        if len(out) != 3 or not all(math.isfinite(x) for x in out):
            raise QualificationError(code)
        return out

    def quat(raw):
        out = tuple(map(float, raw))
        if len(out) != 4 or not all(math.isfinite(x) for x in out):
            raise QualificationError("CANONICAL_MOTION_V2_QUAT_INVALID")
        return out

    tracks = tuple(
        CanonicalControlMotionTrackIRV2(
            canonical_motion_track_id=str(row["canonical_motion_track_id"]),
            canonical_control_id=str(row["canonical_control_id"]),
            keys=tuple(
                CanonicalControlMotionKeyIRV2(
                    time_seconds=float(key["time_seconds"]),
                    translation=vec3(key["translation"], "CANONICAL_MOTION_V2_TRANSLATION_INVALID"),
                    rotation_quat_wxyz=quat(key["rotation_quat_wxyz"]),
                    scale=vec3(key["scale"], "CANONICAL_MOTION_V2_SCALE_INVALID"),
                    metadata=dict(key.get("metadata") or {}),
                )
                for key in row.get("keys") or ()
            ),
            source_proposal_track_id=str(row["source_proposal_track_id"]),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("control_tracks") or ()
    )
    value = CanonicalMotionIRV2(
        canonical_clip_id=str(payload["canonical_clip_id"]),
        display_name=str(payload.get("display_name") or ""),
        intent=str(payload.get("intent") or ""),
        duration_seconds=float(payload["duration_seconds"]),
        loop=bool(payload.get("loop", False)),
        control_tracks=tracks,
        rig_lineage_hash=str(payload["rig_lineage_hash"]),
        control_program_hash=str(payload["control_program_hash"]),
        mechanics_lineage_hash=str(payload["mechanics_lineage_hash"]),
        source_proposal_hash=str(payload["source_proposal_hash"]),
        motion_qualification_hash=str(payload["motion_qualification_hash"]),
        motion_lineage_hash=str(payload["motion_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.motion_lineage_hash != canonical_motion_lineage_hash_v2(value):
        raise QualificationError("CANONICAL_MOTION_V2_LINEAGE_HASH_DRIFT")
    if value.metadata.get("authority") != "COMPILER_CANONICAL_MOTION":
        raise QualificationError("CANONICAL_MOTION_V2_AUTHORITY_DRIFT")
    if value.metadata.get("direct_joint_track_authority") is not False:
        raise QualificationError("CANONICAL_MOTION_V2_JOINT_AUTHORITY_DRIFT")
    return value


__all__ = [
    "CanonicalControlMotionKeyIRV2",
    "CanonicalControlMotionTrackIRV2",
    "CanonicalMotionIRV2",
    "canonical_motion_from_dict_v2",
    "canonical_motion_lineage_hash_v2",
    "mint_canonical_motion_v2",
]
