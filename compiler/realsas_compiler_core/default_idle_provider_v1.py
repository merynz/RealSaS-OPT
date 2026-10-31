from __future__ import annotations

"""Deterministic product-default IDLE proposal in canonical control space.

This is a Motion provider, not Compiler retargeting. It consumes an already
canonical Rig/Mechanics pair and emits one legal control-space proposal when no
external motion provider is present. The provider intentionally uses no humanoid
semantics: it selects a deterministic root control and applies a tiny rigid
translation pulse, so arbitrary morphologies receive a safe first IDLE clip.
"""

import math

from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .control_motion_ir_v1 import (
    ControlMotionClipProposalIRV1,
    ControlMotionKeyProposalIRV1,
    ControlMotionTrackProposalIRV1,
    seal_control_motion_proposal_v1,
)
from .types import QualificationError

DEFAULT_IDLE_PROVIDER_V1 = "RealSaS.DefaultIdleControlProvider.v1"
DEFAULT_IDLE_DURATION_SECONDS_V1 = 2.4
DEFAULT_IDLE_RELATIVE_AMPLITUDE_V1 = 0.006


def _rig_extent(rig: CanonicalRigIRV2) -> float:
    if not rig.joints:
        raise QualificationError("DEFAULT_IDLE_RIG_JOINT_REQUIRED")
    axes = tuple(tuple(float(value) for value in joint.P) for joint in rig.joints)
    if any(len(row) != 3 or not all(math.isfinite(value) for value in row) for row in axes):
        raise QualificationError("DEFAULT_IDLE_RIG_POSITION_INVALID")
    lo = tuple(min(row[axis] for row in axes) for axis in range(3))
    hi = tuple(max(row[axis] for row in axes) for axis in range(3))
    diagonal = math.sqrt(sum((hi[axis] - lo[axis]) ** 2 for axis in range(3)))
    # A one-joint/zero-extent rig can still carry a valid root control. Keep the
    # default pulse finite and small in canonical units rather than failing it.
    return diagonal if diagonal > 1.0e-9 else 1.0


def _root_control_id(rig: CanonicalRigIRV2) -> str:
    if not rig.controls:
        raise QualificationError("DEFAULT_IDLE_RIG_CONTROL_REQUIRED")
    roots = tuple(control for control in rig.controls if control.parent_canonical_control_id is None)
    candidates = roots or tuple(rig.controls)
    # Prefer the control that explicitly drives the broadest joint set; the
    # canonical id tie-break keeps selection replay-stable without semantics.
    selected = sorted(
        candidates,
        key=lambda control: (-len(control.target_canonical_joint_ids), control.canonical_control_id),
    )[0]
    return str(selected.canonical_control_id)


def build_default_idle_proposal_v1(
    rig: CanonicalRigIRV2,
    mechanics: CanonicalMechanicsIRV1,
    *,
    duration_seconds: float = DEFAULT_IDLE_DURATION_SECONDS_V1,
    relative_amplitude: float = DEFAULT_IDLE_RELATIVE_AMPLITUDE_V1,
) -> ControlMotionClipProposalIRV1:
    if mechanics.rig_lineage_hash != rig.rig_lineage_hash:
        raise QualificationError("DEFAULT_IDLE_MECHANICS_RIG_DRIFT")
    if mechanics.control_program_hash != rig.control_program_hash:
        raise QualificationError("DEFAULT_IDLE_MECHANICS_CONTROL_PROGRAM_DRIFT")
    duration = float(duration_seconds)
    relative = float(relative_amplitude)
    if not math.isfinite(duration) or duration <= 0.0:
        raise QualificationError("DEFAULT_IDLE_DURATION_INVALID")
    if not math.isfinite(relative) or relative <= 0.0 or relative > 0.05:
        raise QualificationError("DEFAULT_IDLE_AMPLITUDE_INVALID")

    control_id = _root_control_id(rig)
    amplitude = _rig_extent(rig) * relative
    quarter = duration / 4.0
    keys = tuple(
        ControlMotionKeyProposalIRV1(
            time_seconds=time_seconds,
            translation=(0.0, y, 0.0),
            rotation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            metadata={
                "provider": DEFAULT_IDLE_PROVIDER_V1,
                "channel": "ROOT_IDLE_PULSE",
                "semantic_body_part_assumed": False,
            },
        )
        for time_seconds, y in (
            (0.0, 0.0),
            (quarter, amplitude),
            (quarter * 2.0, 0.0),
            (quarter * 3.0, -amplitude),
            (duration, 0.0),
        )
    )
    proposal = ControlMotionClipProposalIRV1(
        proposal_clip_id="DEFAULT_IDLE",
        display_name="Idle",
        intent="idle",
        duration_seconds=duration,
        loop=True,
        tracks=(
            ControlMotionTrackProposalIRV1(
                proposal_track_id="DEFAULT_IDLE:ROOT",
                canonical_control_id=control_id,
                keys=keys,
                metadata={
                    "provider": DEFAULT_IDLE_PROVIDER_V1,
                    "morphology_semantics_used": False,
                    "retargeting_performed": False,
                },
            ),
        ),
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
        metadata={
            "provider": DEFAULT_IDLE_PROVIDER_V1,
            "product_default": True,
            "morphology_semantics_used": False,
            "learned_inference_performed": False,
            "compiler_retargeting_performed": False,
            "relative_amplitude": relative,
            "canonical_extent": _rig_extent(rig),
        },
    )
    return seal_control_motion_proposal_v1(proposal)


__all__ = [
    "DEFAULT_IDLE_DURATION_SECONDS_V1",
    "DEFAULT_IDLE_PROVIDER_V1",
    "DEFAULT_IDLE_RELATIVE_AMPLITUDE_V1",
    "build_default_idle_proposal_v1",
]
