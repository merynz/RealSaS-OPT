from __future__ import annotations

"""Authority-safe recovery of the historical SaS motion-quality engine.

This module ports the parts of the historical V18/Mechanical motion engine that
directly improve authored motion quality without reclaiming mesh/rig/skin authority:

- named timing/spacing curves,
- phase-aware retiming,
- cubic-Hermite rotation interpolation,
- minimum-jerk blending,
- exact loop endpoint repair,
- optional impact/recoil/settle shaping for future action clips.

The current product lane remains rotation-only because directional translation
units are not yet qualified. Contact locking, corrective deformation and XPBD
secondary motion are separate stages and MUST NOT be claimed by this module.
"""

from dataclasses import dataclass, field
import math

from .hashing import content_sha256
from .v4 import build_joint_track, build_motion_state, validate_motion_against_mechanical
from .v4_types import JointTransformKeyIR, MotionClipIR, MotionStateIR


SCHEMA = "RealSaS.MotionQualityCompiler.v1"
PRODUCER = "RealSaS.MotionCompiler.HistoricalQualityRecovery.v1"
OPERATION_SCOPE = "ROTATION_ONLY_HISTORICAL_MOTION_QUALITY_V1"

_EPS = 1e-12


@dataclass(frozen=True)
class MotionPhaseSpec:
    name: str
    normalized_time: float
    curve_to_next: str = "SMOOTH"
    requires_contact_lock: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": str(self.name),
            "normalized_time": float(self.normalized_time),
            "curve_to_next": str(self.curve_to_next),
            "requires_contact_lock": bool(self.requires_contact_lock),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MotionQualityProfile:
    profile_id: str
    intent: str
    sample_count: int
    phases: tuple[MotionPhaseSpec, ...]
    minimum_jerk_mix: float = 0.35
    hermite_tension: float = 0.15
    exact_loop_repair: bool = True
    contact_schedule_is_intent_only: bool = True
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "profile_id": str(self.profile_id),
            "intent": str(self.intent),
            "sample_count": int(self.sample_count),
            "phases": [p.to_dict() for p in self.phases],
            "minimum_jerk_mix": float(self.minimum_jerk_mix),
            "hermite_tension": float(self.hermite_tension),
            "exact_loop_repair": bool(self.exact_loop_repair),
            "contact_schedule_is_intent_only": bool(self.contact_schedule_is_intent_only),
            "metadata": dict(self.metadata),
        }


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def minimum_jerk(t: float) -> float:
    """Classic C2 endpoint-flat minimum-jerk timing curve."""
    t = _clamp01(t)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def evaluate_named_curve(name: str, t: float) -> float:
    """Recovered SaS-style timing/spacing vocabulary.

    Values are deterministic and bounded for retiming use. Curves whose historical
    artistic meaning included overshoot are represented here by monotone timing
    variants; spatial overshoot belongs to action/corrective stages, not clock time.
    """
    t = _clamp01(t)
    key = str(name).strip().upper().replace("-", "_")
    if key in {"LINEAR"}:
        return t
    if key in {"SMOOTH", "SMOOTHSTEP"}:
        return t * t * (3.0 - 2.0 * t)
    if key in {"EASE_IN_QUAD", "EASEINQUAD"}:
        return t * t
    if key in {"EASE_OUT", "EASE_OUT_QUAD", "EASEOUTQUAD"}:
        return 1.0 - (1.0 - t) * (1.0 - t)
    if key in {"EASE_IN_OUT_QUAD", "EASEINOUTQUAD"}:
        return 2.0 * t * t if t < 0.5 else 1.0 - 2.0 * (1.0 - t) * (1.0 - t)
    if key in {"SINE_IN_OUT", "SINEINOUT"}:
        return 0.5 - 0.5 * math.cos(math.pi * t)
    if key in {"HEAVY"}:
        return t * t * (2.0 - t)
    if key in {"SNAP"}:
        return 1.0 - (1.0 - t) ** 4
    if key in {"HOLD_THEN_SNAP", "HOLDTHENSNAP"}:
        if t <= 0.28:
            return 0.0
        x = (t - 0.28) / 0.72
        return 1.0 - (1.0 - x) ** 4
    if key in {"RECOIL"}:
        return minimum_jerk(t) ** 0.72
    if key in {"SETTLE"}:
        return 1.0 - (1.0 - minimum_jerk(t)) ** 1.35
    if key in {"ELASTIC_LITE", "ELASTICLITE"}:
        s = 0.5 - 0.5 * math.cos(math.pi * t)
        return 0.85 * s + 0.15 * minimum_jerk(t)
    raise ValueError(f"unknown motion curve preset: {name}")


def _validate_phases(profile: MotionQualityProfile) -> tuple[MotionPhaseSpec, ...]:
    if int(profile.sample_count) < 5 or int(profile.sample_count) % 2 == 0:
        raise ValueError("motion quality sample_count must be odd and >= 5")
    if not (0.0 <= float(profile.minimum_jerk_mix) <= 1.0):
        raise ValueError("minimum_jerk_mix must be in [0,1]")
    if not (0.0 <= float(profile.hermite_tension) < 1.0):
        raise ValueError("hermite_tension must be in [0,1)")
    phases = tuple(sorted(profile.phases, key=lambda p: float(p.normalized_time)))
    if len(phases) < 2:
        raise ValueError("motion quality profile requires at least two phase markers")
    if abs(float(phases[0].normalized_time)) > _EPS:
        raise ValueError("first motion phase must start at normalized time 0")
    if abs(float(phases[-1].normalized_time) - 1.0) > _EPS:
        raise ValueError("last motion phase must end at normalized time 1")
    for a, b in zip(phases, phases[1:]):
        if float(b.normalized_time) - float(a.normalized_time) <= _EPS:
            raise ValueError("motion phase times must be strictly increasing")
        evaluate_named_curve(a.curve_to_next, 0.5)
    return phases


def phase_retime(profile: MotionQualityProfile, normalized_time: float) -> float:
    """Map presentation time to source-curve time using phase-local timing curves."""
    phases = _validate_phases(profile)
    u = _clamp01(normalized_time)
    if u <= 0.0 or u >= 1.0:
        return u
    for a, b in zip(phases, phases[1:]):
        ta, tb = float(a.normalized_time), float(b.normalized_time)
        if u <= tb + _EPS:
            local = (u - ta) / max(tb - ta, _EPS)
            shaped = evaluate_named_curve(a.curve_to_next, local)
            mix = float(profile.minimum_jerk_mix)
            shaped = (1.0 - mix) * shaped + mix * minimum_jerk(local)
            return ta + (tb - ta) * shaped
    return 1.0


def _source_slopes(times: tuple[float, ...], values: tuple[float, ...], tension: float) -> tuple[float, ...]:
    if len(times) != len(values) or len(times) < 2:
        raise ValueError("Hermite source requires >=2 aligned samples")
    out = []
    scale = 1.0 - float(tension)
    for i in range(len(times)):
        if i == 0:
            dt = max(times[1] - times[0], _EPS)
            slope = (values[1] - values[0]) / dt
        elif i == len(times) - 1:
            dt = max(times[-1] - times[-2], _EPS)
            slope = (values[-1] - values[-2]) / dt
        else:
            dt = max(times[i + 1] - times[i - 1], _EPS)
            slope = (values[i + 1] - values[i - 1]) / dt
        out.append(scale * slope)
    return tuple(out)


def hermite_sample(times: tuple[float, ...], values: tuple[float, ...], time_sec: float, *, tension: float = 0.15) -> float:
    """Sample sparse scalar keys with deterministic cubic Hermite interpolation."""
    if len(times) != len(values) or len(times) < 2:
        raise ValueError("Hermite sample requires aligned >=2 source keys")
    if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
        raise ValueError("source key times must be strictly increasing")
    t = float(time_sec)
    if t <= times[0]:
        return float(values[0])
    if t >= times[-1]:
        return float(values[-1])
    slopes = _source_slopes(times, values, tension)
    idx = 0
    while idx + 1 < len(times) and t > times[idx + 1]:
        idx += 1
    t0, t1 = times[idx], times[idx + 1]
    y0, y1 = values[idx], values[idx + 1]
    h = max(t1 - t0, _EPS)
    u = (t - t0) / h
    u2, u3 = u * u, u * u * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return h00 * y0 + h10 * h * slopes[idx] + h01 * y1 + h11 * h * slopes[idx + 1]


def impact_spring_damper_response(normalized_time: float, *, impulse: float = 1.0, damping: float = 7.5, frequency_hz: float = 2.6) -> float:
    """Bounded deterministic recoil/settle primitive for later impact clips."""
    t = _clamp01(normalized_time)
    if damping <= 0.0 or frequency_hz <= 0.0:
        raise ValueError("damping and frequency_hz must be positive")
    return float(impulse) * math.exp(-float(damping) * t) * math.sin(2.0 * math.pi * float(frequency_hz) * t)


def _track_payload(track) -> dict:
    return {
        "canonical_joint_id": str(track.canonical_joint_id),
        "keys": [
            {
                "time_sec": float(key.time_sec),
                "rotation_deg": float(key.rotation_deg),
                "translation_xy": tuple(map(float, key.translation_xy)),
                "scale_xy": tuple(map(float, key.scale_xy)),
                "depth_offset": float(key.depth_offset),
            }
            for key in track.keys
        ],
    }


def _polish_track(track, clip: MotionClipIR, profile: MotionQualityProfile):
    source_times = tuple(float(key.time_sec) for key in track.keys)
    rotations = tuple(float(key.rotation_deg) for key in track.keys)
    if len(source_times) < 2:
        raise ValueError(f"motion track has fewer than two keys: {track.track_id}")
    duration = float(clip.duration_sec)
    if duration <= 0.0:
        raise ValueError(f"motion clip has nonpositive duration: {clip.clip_id}")

    sample_count = int(profile.sample_count)
    target_times = tuple(duration * i / (sample_count - 1) for i in range(sample_count))
    keys = []
    for index, target_t in enumerate(target_times):
        if index in (0, sample_count - 1) and bool(clip.loop) and profile.exact_loop_repair:
            rotation = float(rotations[0])
        else:
            source_u = phase_retime(profile, target_t / duration)
            source_t = source_times[0] + source_u * (source_times[-1] - source_times[0])
            rotation = hermite_sample(source_times, rotations, source_t, tension=float(profile.hermite_tension))
        keys.append(JointTransformKeyIR(float(target_t), (0.0, 0.0), float(rotation), (1.0, 1.0), 0.0))
    if bool(clip.loop) and profile.exact_loop_repair:
        keys[-1] = JointTransformKeyIR(float(duration), tuple(keys[0].translation_xy), float(keys[0].rotation_deg), tuple(keys[0].scale_xy), float(keys[0].depth_offset))
    return build_joint_track(
        f"TRACK:QUALITY:{clip.clip_id}:{track.canonical_joint_id}",
        str(clip.clip_id), str(track.canonical_joint_id), tuple(keys),
        metadata={
            "producer": PRODUCER,
            "operation_scope": OPERATION_SCOPE,
            "quality_profile_id": profile.profile_id,
            "source_track_hash": str(track.track_hash),
            "phase_aware_retime": True,
            "cubic_hermite_rotation": True,
            "minimum_jerk_timing_blend": float(profile.minimum_jerk_mix),
            "exact_loop_repair": bool(profile.exact_loop_repair),
            "translation_policy": "ZERO_ALL_JOINT_TRANSLATION_XY",
        },
    )


def build_mage_quality_profiles() -> dict[str, MotionQualityProfile]:
    """Recovered bounded Mage profile: idle breathe + contact-aware run cadence.

    Contact markers are intent until the directional contact solver proves them.
    They influence timing now but are never reported as contact-lock PASS here.
    """
    return {
        "mage_fit1_idle_v2": MotionQualityProfile(
            profile_id="MAGE_IDLE_HISTORICAL_QUALITY_V1", intent="IDLE", sample_count=33,
            phases=(
                MotionPhaseSpec("LOOP_START", 0.0, "SINE_IN_OUT"),
                MotionPhaseSpec("BREATH_APEX", 0.5, "SINE_IN_OUT"),
                MotionPhaseSpec("LOOP_END", 1.0, "SMOOTH"),
            ),
            minimum_jerk_mix=0.18, hermite_tension=0.05,
            metadata={"historical_lineage": "SaS_v18_motion_curve_and_motion_synthesis"},
        ),
        "mage_fit1_run_v2": MotionQualityProfile(
            profile_id="MAGE_RUN_HISTORICAL_QUALITY_V1", intent="RUN", sample_count=33,
            phases=(
                MotionPhaseSpec("LEFT_PLANT", 0.0, "HEAVY", True),
                MotionPhaseSpec("LEFT_RELEASE", 0.14, "SNAP"),
                MotionPhaseSpec("FLIGHT_A", 0.34, "EASE_OUT"),
                MotionPhaseSpec("RIGHT_PLANT", 0.5, "HEAVY", True),
                MotionPhaseSpec("RIGHT_RELEASE", 0.64, "SNAP"),
                MotionPhaseSpec("FLIGHT_B", 0.84, "EASE_OUT"),
                MotionPhaseSpec("LOOP_END", 1.0, "SMOOTH", True),
            ),
            minimum_jerk_mix=0.32, hermite_tension=0.12, contact_schedule_is_intent_only=True,
            metadata={
                "historical_lineage": "SaS_v18_3_to_v18_6_contact_arc_retime_hermite",
                "contact_lock_claimed": False,
                "foot_sliding_claimed": False,
            },
        ),
    }


def compile_motion_quality(motion: MotionStateIR, mechanical, *, profiles: dict[str, MotionQualityProfile] | None = None) -> MotionStateIR:
    """Compile a qualified rotation-only motion state through recovered quality passes."""
    validate_motion_against_mechanical(motion, mechanical)
    profiles = dict(profiles or build_mage_quality_profiles())
    clip_by_id = {str(c.clip_id): c for c in motion.clips}
    tracks_by_clip: dict[str, list] = {cid: [] for cid in clip_by_id}
    for track in motion.joint_tracks:
        if any(tuple(map(float, key.translation_xy)) != (0.0, 0.0) for key in track.keys):
            raise ValueError("historical motion quality compiler requires rotation-only qualified input")
        tracks_by_clip.setdefault(str(track.clip_id), []).append(track)

    out_clips = []
    out_tracks = []
    quality_reports = []
    for clip in motion.clips:
        cid = str(clip.clip_id)
        profile = profiles.get(cid)
        if profile is None:
            raise ValueError(f"missing motion quality profile for clip: {cid}")
        _validate_phases(profile)
        source_tracks = tuple(sorted(tracks_by_clip.get(cid, ()), key=lambda t: str(t.canonical_joint_id)))
        if not source_tracks:
            raise ValueError(f"clip has no joint tracks: {cid}")
        polished = tuple(_polish_track(track, clip, profile) for track in source_tracks)
        payload = {
            "schema": SCHEMA,
            "producer": PRODUCER,
            "operation_scope": OPERATION_SCOPE,
            "source_clip_payload_hash": str(clip.clip_payload_hash),
            "profile": profile.to_dict(),
            "tracks": [_track_payload(track) for track in polished],
        }
        payload_hash = content_sha256(payload)
        out_clips.append(MotionClipIR(
            cid, str(clip.clip_kind), payload_hash, tuple(clip.required_capabilities),
            source_ref=PRODUCER, duration_sec=float(clip.duration_sec), loop=bool(clip.loop),
            metadata={
                **dict(clip.metadata or {}),
                "spec_hash": payload_hash,
                "source_clip_payload_hash": str(clip.clip_payload_hash),
                "quality_profile_id": profile.profile_id,
                "quality_engine": PRODUCER,
                "phase_aware_retime": True,
                "cubic_hermite_rotation": True,
                "minimum_jerk_timing_blend": float(profile.minimum_jerk_mix),
                "exact_loop_repair": bool(profile.exact_loop_repair),
                "contact_schedule_present": any(p.requires_contact_lock for p in profile.phases),
                "contact_schedule_is_intent_only": bool(profile.contact_schedule_is_intent_only),
                "contact_lock_claimed": False,
                "xpbd_secondary_claimed": False,
                "corrective_deformation_claimed": False,
            },
        ))
        out_tracks.extend(polished)
        quality_reports.append({
            "clip_id": cid,
            "profile_id": profile.profile_id,
            "source_track_count": len(source_tracks),
            "output_track_count": len(polished),
            "sample_count": int(profile.sample_count),
            "phase_count": len(profile.phases),
            "contact_intent_phase_count": sum(1 for p in profile.phases if p.requires_contact_lock),
            "translation_channel_zero": True,
            "loop_repair_applied": bool(clip.loop and profile.exact_loop_repair),
        })

    state = build_motion_state(
        tuple(out_clips), tuple(out_tracks),
        metadata={
            **dict(motion.metadata or {}),
            "producer": PRODUCER,
            "operation_scope": OPERATION_SCOPE,
            "historical_motion_engine_recovered": True,
            "historical_recovery_scope": (
                "PHASES", "NAMED_TIMING_CURVES", "CUBIC_HERMITE_ROTATION",
                "MINIMUM_JERK_TIMING_BLEND", "EXACT_LOOP_REPAIR",
            ),
            "source_motion_state_hash": str(motion.motion_state_hash),
            "quality_reports": quality_reports,
            "translation_policy": "ZERO_ALL_JOINT_TRANSLATION_XY",
            "contact_lock_claimed": False,
            "xpbd_secondary_claimed": False,
            "corrective_deformation_claimed": False,
        },
    )
    validate_motion_against_mechanical(state, mechanical)
    if any(tuple(map(float, key.translation_xy)) != (0.0, 0.0) for t in state.joint_tracks for key in t.keys):
        raise AssertionError("motion quality compiler emitted nonzero translation")
    for clip in state.clips:
        if not clip.loop:
            continue
        for track in [t for t in state.joint_tracks if t.clip_id == clip.clip_id]:
            a, b = track.keys[0], track.keys[-1]
            if (a.translation_xy, a.rotation_deg, a.scale_xy, a.depth_offset) != (b.translation_xy, b.rotation_deg, b.scale_xy, b.depth_offset):
                raise AssertionError("exact loop repair failed")
    return state


__all__ = [
    "MotionPhaseSpec",
    "MotionQualityProfile",
    "OPERATION_SCOPE",
    "PRODUCER",
    "build_mage_quality_profiles",
    "compile_motion_quality",
    "evaluate_named_curve",
    "hermite_sample",
    "impact_spring_damper_response",
    "minimum_jerk",
    "phase_retime",
]
