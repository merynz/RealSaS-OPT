from __future__ import annotations

"""Rotation-only qualification projection for the bounded Mage preset lane.

The authored Mage preset remains articulated over the qualified canonical skeleton.
This module removes only nonzero puppet-local translation keys because the current
directional evaluator has no qualified translation-unit contract. Rotations, scale
(identity), depth (zero), clip timing, loop identity, and joint semantics remain
explicit and hash-bound.

The projection deliberately does not depend on the source motion state's passive
track emission policy: after translation removal, only effective joint tracks are
retained. This makes the qualified result invariant to the rejected 44-track passive
emission drift while preserving the actual articulated motion.
"""

from .hashing import content_sha256
from .motion import build_mage_topology_preset_motion
from .v4 import build_joint_track, build_motion_state, validate_motion_against_mechanical
from .v4_types import JointTransformKeyIR, MotionClipIR


OPERATION_SCOPE = "ROTATION_ONLY_CURRENT_PRESET_V1"
PRODUCER = "RealSaS.MotionCompiler.MageRotationOnlyQualified.v1"
SCHEMA = "RealSaS.MageRotationOnlyQualifiedPreset.v1"


def _key_without_translation(key) -> JointTransformKeyIR:
    return JointTransformKeyIR(
        float(key.time_sec),
        (0.0, 0.0),
        float(key.rotation_deg),
        tuple(map(float, key.scale_xy)),
        float(key.depth_offset),
    )


def _effective(keys: tuple[JointTransformKeyIR, ...]) -> bool:
    if len(keys) < 2:
        return False
    first = keys[0]
    baseline = (first.translation_xy, first.rotation_deg, first.scale_xy, first.depth_offset)
    return any(
        (key.translation_xy, key.rotation_deg, key.scale_xy, key.depth_offset) != baseline
        for key in keys[1:]
    )


def _clip_payload(clip_id: str, duration_sec: float, loop: bool, tracks) -> dict:
    return {
        "schema": SCHEMA,
        "clip_id": str(clip_id),
        "duration_sec": float(duration_sec),
        "loop": bool(loop),
        "operation_scope": OPERATION_SCOPE,
        "translation_policy": "ZERO_ALL_JOINT_TRANSLATION_XY",
        "tracks": [
            {
                "canonical_joint_id": str(track.canonical_joint_id),
                "keys": [
                    {
                        "time_sec": float(key.time_sec),
                        "rotation_deg": float(key.rotation_deg),
                        "scale_xy": tuple(map(float, key.scale_xy)),
                        "depth_offset": float(key.depth_offset),
                    }
                    for key in track.keys
                ],
            }
            for track in sorted(tracks, key=lambda row: str(row.canonical_joint_id))
        ],
    }


def build_mage_rotation_only_qualified_motion(mechanical, *, sample_count: int = 17):
    """Build the typed-evaluator-qualified Mage idle/run motion state.

    The upstream preset supplies the articulated rotation curves and semantic role
    resolution. This function is the only bounded-demo projection that removes the
    currently-unqualified translation channel. It does not synthesize new rotations.
    """

    base = build_mage_topology_preset_motion(mechanical, sample_count=sample_count)

    projected_by_clip: dict[str, list] = {str(clip.clip_id): [] for clip in base.clips}
    for track in base.joint_tracks:
        keys = tuple(_key_without_translation(key) for key in track.keys)
        if not _effective(keys):
            continue
        projected = build_joint_track(
            f"TRACK:ROQ:{track.clip_id}:{track.canonical_joint_id}",
            str(track.clip_id),
            str(track.canonical_joint_id),
            keys,
            metadata={
                "producer": PRODUCER,
                "operation_scope": OPERATION_SCOPE,
                "translation_policy": "ZERO_ALL_JOINT_TRANSLATION_XY",
            },
        )
        projected_by_clip.setdefault(str(track.clip_id), []).append(projected)

    clips = []
    tracks = []
    for clip in base.clips:
        clip_tracks = tuple(sorted(projected_by_clip.get(str(clip.clip_id), ()), key=lambda row: row.canonical_joint_id))
        if not clip_tracks:
            raise ValueError(f"qualified Mage clip lost all effective rotation tracks: {clip.clip_id}")
        payload = _clip_payload(str(clip.clip_id), float(clip.duration_sec), bool(clip.loop), clip_tracks)
        payload_hash = content_sha256(payload)
        clips.append(MotionClipIR(
            str(clip.clip_id),
            str(clip.clip_kind),
            payload_hash,
            tuple(clip.required_capabilities),
            source_ref=PRODUCER,
            duration_sec=float(clip.duration_sec),
            loop=bool(clip.loop),
            metadata={
                "intent": "IDLE" if str(clip.clip_id) == "mage_fit1_idle_v2" else "RUN",
                "display_name": "Idle" if str(clip.clip_id) == "mage_fit1_idle_v2" else "Run",
                "spec_hash": payload_hash,
                "qualified_operation_scope": OPERATION_SCOPE,
                "translation_policy": "ZERO_ALL_JOINT_TRANSLATION_XY",
                "nonzero_translation_authored": False,
                "semantic_joint_names_used": False,
                "historical_mesh_authority_used": False,
                "historical_weight_authority_used": False,
                "frame0_identity_authored": True,
                "loop_closure_identity_authored": True,
            },
        ))
        tracks.extend(clip_tracks)

    state = build_motion_state(
        tuple(clips),
        tuple(tracks),
        metadata={
            "producer": PRODUCER,
            "preset_family": "MAGE_FIT1_IDLE_RUN_ROTATION_ONLY_QUALIFIED",
            "qualified_operation_scope": OPERATION_SCOPE,
            "translation_policy": "ZERO_ALL_JOINT_TRANSLATION_XY",
            "nonzero_translation_authored": False,
            "source_motion_projection": "ARTICULATED_ROTATION_CURVES_PRESERVED_TRANSLATION_ZEROED",
            "authored_joint_names_used": False,
            "historical_mesh_authority_used": False,
            "historical_weight_authority_used": False,
            "frame0_identity_authored": True,
            "loop_closure_identity_authored": True,
        },
    )
    validate_motion_against_mechanical(state, mechanical)

    if any(tuple(map(float, key.translation_xy)) != (0.0, 0.0) for track in state.joint_tracks for key in track.keys):
        raise AssertionError("rotation-only qualified motion emitted nonzero translation")
    return state


__all__ = [
    "OPERATION_SCOPE",
    "build_mage_rotation_only_qualified_motion",
]
