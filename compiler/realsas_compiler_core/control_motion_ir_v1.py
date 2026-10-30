from __future__ import annotations

"""Typed control-space Motion proposal and fail-closed qualification.

Motion is authored/learned in animator control space.  A proposal may reference
only controls already present in CanonicalRigIR.v2.  Compiler qualification does
not create controls, retarget tracks, bake joint curves, solve constraints, or
repair keyframes; it only proves that the proposed clip is a legal input to the
sealed Rig/Mechanics program.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping
import math

from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class ControlMotionKeyProposalIRV1:
    time_seconds: float
    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation_quat_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class ControlMotionTrackProposalIRV1:
    proposal_track_id: str
    canonical_control_id: str
    keys: tuple[ControlMotionKeyProposalIRV1, ...]
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class ControlMotionClipProposalIRV1:
    proposal_clip_id: str
    display_name: str
    intent: str
    duration_seconds: float
    loop: bool
    tracks: tuple[ControlMotionTrackProposalIRV1, ...]
    rig_lineage_hash: str
    control_program_hash: str
    mechanics_lineage_hash: str
    proposal_lineage_hash: str = ""
    schema_version: str = "RealSaS.ControlMotionClipProposalIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedControlMotionIRV1:
    proposal: ControlMotionClipProposalIRV1
    qualification_hash: str
    qualification_report: Json
    schema_version: str = "RealSaS.QualifiedControlMotionIR.v1"

    def to_dict(self) -> dict:
        return asdict(self)


def control_motion_proposal_hash_v1(value: ControlMotionClipProposalIRV1) -> str:
    payload = value.to_dict()
    payload.pop("proposal_lineage_hash", None)
    return content_sha256(payload)


def qualified_control_motion_hash_v1(value: QualifiedControlMotionIRV1) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def seal_control_motion_proposal_v1(value: ControlMotionClipProposalIRV1) -> ControlMotionClipProposalIRV1:
    return replace(value, proposal_lineage_hash=control_motion_proposal_hash_v1(value))


def _finite_vector(value, length: int, code: str) -> tuple[float, ...]:
    try:
        out = tuple(map(float, value))
    except Exception as exc:
        raise QualificationError(code) from exc
    if len(out) != length or not all(math.isfinite(x) for x in out):
        raise QualificationError(code)
    return out


def qualify_control_motion_v1(
    proposal: ControlMotionClipProposalIRV1,
    *,
    rig: CanonicalRigIRV2,
    mechanics: CanonicalMechanicsIRV1,
    quaternion_norm_tolerance: float = 1.0e-3,
) -> QualifiedControlMotionIRV1:
    if proposal.proposal_lineage_hash != control_motion_proposal_hash_v1(proposal):
        raise QualificationError("MOTION_V2_PROPOSAL_HASH_DRIFT")
    if proposal.rig_lineage_hash != rig.rig_lineage_hash:
        raise QualificationError("MOTION_V2_RIG_LINEAGE_DRIFT")
    if proposal.control_program_hash != rig.control_program_hash:
        raise QualificationError("MOTION_V2_CONTROL_PROGRAM_DRIFT")
    if proposal.mechanics_lineage_hash != mechanics.mechanics_lineage_hash:
        raise QualificationError("MOTION_V2_MECHANICS_LINEAGE_DRIFT")
    if mechanics.rig_lineage_hash != rig.rig_lineage_hash:
        raise QualificationError("MOTION_V2_MECHANICS_RIG_DRIFT")
    if mechanics.control_program_hash != rig.control_program_hash:
        raise QualificationError("MOTION_V2_MECHANICS_CONTROL_DRIFT")

    duration = float(proposal.duration_seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise QualificationError("MOTION_V2_DURATION_INVALID")
    if quaternion_norm_tolerance < 0.0 or not math.isfinite(float(quaternion_norm_tolerance)):
        raise QualificationError("MOTION_V2_QUATERNION_TOLERANCE_INVALID")
    if not proposal.proposal_clip_id.strip():
        raise QualificationError("MOTION_V2_CLIP_ID_EMPTY")

    control_ids = {row.canonical_control_id for row in rig.controls}
    if not control_ids:
        raise QualificationError("MOTION_V2_RIG_HAS_NO_CONTROLS")
    seen_tracks: set[str] = set()
    seen_controls: set[str] = set()
    key_count = 0
    for track in proposal.tracks:
        tid = str(track.proposal_track_id).strip()
        cid = str(track.canonical_control_id).strip()
        if not tid or tid in seen_tracks:
            raise QualificationError("MOTION_V2_TRACK_ID_INVALID")
        seen_tracks.add(tid)
        if cid not in control_ids:
            raise QualificationError(f"MOTION_V2_CONTROL_UNKNOWN:{cid}")
        if cid in seen_controls:
            raise QualificationError(f"MOTION_V2_CONTROL_TRACK_DUPLICATE:{cid}")
        seen_controls.add(cid)
        if not track.keys:
            raise QualificationError(f"MOTION_V2_TRACK_EMPTY:{tid}")
        previous = -math.inf
        for key in track.keys:
            t = float(key.time_seconds)
            if not math.isfinite(t) or t < 0.0 or t > duration or t <= previous:
                raise QualificationError(f"MOTION_V2_KEY_TIME_INVALID:{tid}")
            previous = t
            _finite_vector(key.translation, 3, "MOTION_V2_TRANSLATION_INVALID")
            quat = _finite_vector(key.rotation_quat_wxyz, 4, "MOTION_V2_ROTATION_INVALID")
            qnorm = math.sqrt(sum(x * x for x in quat))
            if qnorm <= 1.0e-12 or abs(qnorm - 1.0) > float(quaternion_norm_tolerance):
                raise QualificationError("MOTION_V2_ROTATION_NOT_UNIT")
            scale = _finite_vector(key.scale, 3, "MOTION_V2_SCALE_INVALID")
            if any(x <= 0.0 for x in scale):
                raise QualificationError("MOTION_V2_SCALE_NONPOSITIVE")
            key_count += 1

    report = {
        "schema": "RealSaS.ControlMotionQualification.v1",
        "status": "PASS_CONTROL_MOTION_VALIDATED",
        "clip_id": proposal.proposal_clip_id,
        "track_count": len(proposal.tracks),
        "key_count": key_count,
        "duration_seconds": duration,
        "loop": bool(proposal.loop),
        "rig_lineage_hash": rig.rig_lineage_hash,
        "control_program_hash": rig.control_program_hash,
        "mechanics_lineage_hash": mechanics.mechanics_lineage_hash,
        "control_synthesis_performed": False,
        "track_retargeting_performed": False,
        "joint_curve_bake_performed": False,
        "constraint_solver_executed": False,
        "keyframe_repair_performed": False,
    }
    value = QualifiedControlMotionIRV1(
        proposal=proposal,
        qualification_hash="",
        qualification_report=report,
    )
    return replace(value, qualification_hash=qualified_control_motion_hash_v1(value))


def _key_from_dict(payload: Mapping[str, Any]) -> ControlMotionKeyProposalIRV1:
    return ControlMotionKeyProposalIRV1(
        time_seconds=float(payload["time_seconds"]),
        translation=tuple(map(float, payload.get("translation") or (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
        rotation_quat_wxyz=tuple(map(float, payload.get("rotation_quat_wxyz") or (1.0, 0.0, 0.0, 0.0))),  # type: ignore[arg-type]
        scale=tuple(map(float, payload.get("scale") or (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
        metadata=dict(payload.get("metadata") or {}),
    )


def control_motion_proposal_from_dict_v1(payload: Mapping[str, Any]) -> ControlMotionClipProposalIRV1:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.ControlMotionClipProposalIR.v1":
        raise QualificationError("MOTION_V2_PROPOSAL_SCHEMA_DRIFT")
    tracks = tuple(
        ControlMotionTrackProposalIRV1(
            proposal_track_id=str(row["proposal_track_id"]),
            canonical_control_id=str(row["canonical_control_id"]),
            keys=tuple(_key_from_dict(key) for key in row.get("keys") or ()),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("tracks") or ()
    )
    value = ControlMotionClipProposalIRV1(
        proposal_clip_id=str(payload["proposal_clip_id"]),
        display_name=str(payload.get("display_name") or payload["proposal_clip_id"]),
        intent=str(payload.get("intent") or ""),
        duration_seconds=float(payload["duration_seconds"]),
        loop=bool(payload.get("loop", False)),
        tracks=tracks,
        rig_lineage_hash=str(payload["rig_lineage_hash"]),
        control_program_hash=str(payload["control_program_hash"]),
        mechanics_lineage_hash=str(payload["mechanics_lineage_hash"]),
        proposal_lineage_hash=str(payload["proposal_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.proposal_lineage_hash != control_motion_proposal_hash_v1(value):
        raise QualificationError("MOTION_V2_PROPOSAL_HASH_DRIFT")
    return value


__all__ = [
    "ControlMotionClipProposalIRV1",
    "ControlMotionKeyProposalIRV1",
    "ControlMotionTrackProposalIRV1",
    "QualifiedControlMotionIRV1",
    "control_motion_proposal_from_dict_v1",
    "control_motion_proposal_hash_v1",
    "qualify_control_motion_v1",
    "qualified_control_motion_hash_v1",
    "seal_control_motion_proposal_v1",
]
