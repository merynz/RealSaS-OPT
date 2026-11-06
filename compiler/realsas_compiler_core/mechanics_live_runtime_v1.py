from __future__ import annotations

"""Mechanics-first runtime intervention and independent behavioral diagnostics.

This interpreter does NOT learn rig/weights, synthesize control handles or own
Graphics. CanonicalRig and CanonicalSkin are immutable inputs, while Motion
supplies optional editable control tracks. Interactive overrides are ephemeral
and never mutate either control program or animation clips.

All solver paths call the SAME deterministic Mechanics executor. A successful
call does NOT certify genericity, physical simulation, or real-time frame rate.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from .canonical_mesh_ir_v1 import CanonicalMeshIR
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .canonical_motion_ir_v2 import CanonicalMotionIRV2
from .control_motion_runtime_v2 import sample_control_transforms_v2
from .mechanics_control_solver_v1 import (
    ControlTransformIRV1, DeformedMeshIRV1, MechanicsPoseIRV1,
    deform_mesh_lbs_v1, solve_control_pose_v1,
)
from .rig_constraint_schema_v1 import RIG_CONSTRAINT_OPERATOR_IDS_V1
from .types import QualificationError


@dataclass(frozen=True)
class MechanicsInterventionFrameV1:
    pose: MechanicsPoseIRV1
    deformation: DeformedMeshIRV1
    active_control_ids: tuple[str, ...]
    overridden_control_ids: tuple[str, ...]
    max_constraint_residual: float
    max_segment_length_relative_error: float
    max_rest_vertex_displacement: float
    status: str
    failures: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MechanicsInterventionStepV1:
    time_seconds: float
    overrides: tuple[ControlTransformIRV1, ...] = ()


@dataclass(frozen=True)
class MechanicsInterventionSequenceV1:
    frames: tuple[MechanicsInterventionFrameV1, ...]
    max_vertex_speed: float
    max_vertex_acceleration: float
    status: str
    metadata: dict = field(default_factory=dict)


def _strict_lineage(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
) -> None:
    if (mesh.mesh_lineage_hash != rig.mesh_lineage_hash
            or mesh.mesh_lineage_hash != skin.mesh_lineage_hash
            or mesh.mesh_lineage_hash != mechanics.mesh_lineage_hash):
        raise QualificationError("LIVE_MECHANICS_MESH_LINEAGE_DRIFT")
    if (rig.rig_lineage_hash != mechanics.rig_lineage_hash
            or rig.deform_structure_hash != mechanics.deform_structure_hash
            or rig.deform_structure_hash != skin.deform_structure_hash
            or rig.control_program_hash != mechanics.control_program_hash
            or skin.skin_lineage_hash != mechanics.skin_lineage_hash):
        raise QualificationError("LIVE_MECHANICS_AUTHORITY_LINEAGE_DRIFT")
    declared_schedule = tuple(row.canonical_constraint_id for row in rig.constraints)
    if mechanics.constraint_schedule != declared_schedule:
        raise QualificationError("LIVE_MECHANICS_CONSTRAINT_SCHEDULE_DRIFT")
    unsupported = {
        row.operator_id for row in rig.constraints
    } - set(mechanics.executable_operator_ids).intersection(RIG_CONSTRAINT_OPERATOR_IDS_V1)
    if unsupported:
        raise QualificationError(
            "LIVE_MECHANICS_UNEXECUTABLE_OPERATOR:" + ",".join(sorted(unsupported))
        )


def _finite_tolerance(value: float, code: str) -> float:
    try:
        value = float(value)
    except (ValueError, TypeError) as exc:
        raise QualificationError(code) from exc
    if not math.isfinite(value) or value < 0.0:
        raise QualificationError(code)
    return value


def _behavioral_metrics(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    pose: MechanicsPoseIRV1,
    deformation: DeformedMeshIRV1,
) -> tuple[float, float]:
    joint_rest = {r.canonical_joint_id: np.asarray(r.P, dtype=np.float64) for r in rig.joints}
    joint_solved = {r.canonical_joint_id: np.asarray(r.solved_P, dtype=np.float64) for r in pose.joints}
    if set(joint_rest) != set(joint_solved):
        raise QualificationError("LIVE_MECHANICS_JOINT_COVERAGE_DRIFT")

    strain = 0.0
    for joint in rig.joints:
        parent = joint.parent_canonical_joint_id
        if parent is None:
            continue
        if parent not in joint_rest:
            raise QualificationError("LIVE_MECHANICS_JOINT_PARENT_UNKNOWN")
        rest_length = float(np.linalg.norm(joint_rest[joint.canonical_joint_id] - joint_rest[parent]))
        solved_length = float(np.linalg.norm(joint_solved[joint.canonical_joint_id] - joint_solved[parent]))
        if rest_length <= 1.0e-12:
            if solved_length > 1.0e-10:
                raise QualificationError("LIVE_MECHANICS_DEGENERATE_SEGMENT")
            continue
        strain = max(strain, abs(solved_length / rest_length - 1.0))

    rest_vertices = {r.canonical_mesh_vertex_id: np.asarray(r.P, dtype=np.float64)
                     for r in mesh.vertices}
    if (len(rest_vertices) != len(mesh.vertices)
            or len(deformation.positions) != len(deformation.canonical_mesh_vertex_ids)
            or set(rest_vertices) != set(deformation.canonical_mesh_vertex_ids)):
        raise QualificationError("LIVE_MECHANICS_VERTEX_COVERAGE_DRIFT")
    displacement = 0.0
    for vid, p in zip(deformation.canonical_mesh_vertex_ids, deformation.positions):
        current = np.asarray(p, dtype=np.float64)
        if current.shape != (3,) or not np.isfinite(current).all():
            raise QualificationError("LIVE_MECHANICS_VERTEX_NONFINITE")
        displacement = max(displacement, float(np.linalg.norm(current - rest_vertices[vid])))
    return strain, displacement


def evaluate_mechanics_intervention_frame_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    *,
    overrides: tuple[ControlTransformIRV1, ...] = (),
    motion: CanonicalMotionIRV2 | None = None,
    time_seconds: float | None = None,
    max_allowed_constraint_residual: float = 1.0e-4,
    max_allowed_segment_strain: float = 1.0e-3,
) -> MechanicsInterventionFrameV1:
    _strict_lineage(mesh, rig, skin, mechanics)
    limit_residual = _finite_tolerance(
        max_allowed_constraint_residual, "LIVE_MECHANICS_RESIDUAL_POLICY_INVALID",
    )
    limit_strain = _finite_tolerance(
        max_allowed_segment_strain, "LIVE_MECHANICS_STRAIN_POLICY_INVALID",
    )
    if motion is None and time_seconds is not None:
        raise QualificationError("LIVE_MECHANICS_TIME_WITHOUT_MOTION")
    if motion is not None and time_seconds is None:
        raise QualificationError("LIVE_MECHANICS_MOTION_TIME_REQUIRED")

    known_controls = {c.canonical_control_id for c in rig.controls}
    if len(known_controls) != len(rig.controls):
        raise QualificationError("LIVE_MECHANICS_DUPLICATE_RIG_CONTROLS")
    base = (
        sample_control_transforms_v2(
            motion, rig=rig, mechanics=mechanics, time_seconds=float(time_seconds),
        ) if motion is not None else ()
    )
    by_id = {r.canonical_control_id: r for r in base}
    if len(by_id) != len(base):
        raise QualificationError("LIVE_MECHANICS_DUPLICATE_MOTION_CONTROL")
    edited: set[str] = set()
    for edit in overrides:
        cid = edit.canonical_control_id
        if cid not in known_controls:
            raise QualificationError("LIVE_MECHANICS_EDIT_UNKNOWN_CONTROL")
        if cid in edited:
            raise QualificationError("LIVE_MECHANICS_EDIT_DUPLICATE_CONTROL")
        edited.add(cid)
        by_id[cid] = edit
    # Consistent authored-order events, irrespective of mouse event order.
    controls = tuple(by_id[row.canonical_control_id] for row in rig.controls
                     if row.canonical_control_id in by_id)
    pose = solve_control_pose_v1(rig, controls)
    deformed = deform_mesh_lbs_v1(mesh, rig, skin, pose)
    maximum_residual = max((abs(float(v)) for _, v in pose.constraint_residuals), default=0.0)
    if not math.isfinite(maximum_residual):
        raise QualificationError("LIVE_MECHANICS_RESIDUAL_NONFINITE")
    strain, displacement = _behavioral_metrics(mesh, rig, pose, deformed)
    failures: list[str] = []
    if maximum_residual > limit_residual:
        failures.append("CONSTRAINT_RESIDUAL_EXCEEDED")
    if strain > limit_strain:
        failures.append("SEGMENT_STRAIN_EXCEEDED")
    return MechanicsInterventionFrameV1(
        pose=pose, deformation=deformed,
        active_control_ids=tuple(x.canonical_control_id for x in controls),
        overridden_control_ids=tuple(c.canonical_control_id for c in rig.controls
                                     if c.canonical_control_id in edited),
        max_constraint_residual=maximum_residual,
        max_segment_length_relative_error=strain,
        max_rest_vertex_displacement=displacement,
        status=("FAIL_MECHANICS_BEHAVIORAL_COURT" if failures
                else "PASS_MECHANICS_BEHAVIORAL_COURT"),
        failures=tuple(failures),
        metadata={
            "authority": "MECHANICS_RUNTIME_DIAGNOSTIC_ONLY",
            "control_synthesis_performed": False,
            "rig_or_skin_mutated": False,
            "model_inference_performed": False,
            "graphics_authority_claimed": False,
            "time_seconds": time_seconds,
        },
    )


def evaluate_mechanics_intervention_sequence_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    steps: tuple[MechanicsInterventionStepV1, ...],
    *,
    motion: CanonicalMotionIRV2 | None = None,
    max_allowed_constraint_residual: float = 1.0e-4,
    max_allowed_segment_strain: float = 1.0e-3,
) -> MechanicsInterventionSequenceV1:
    if not steps:
        raise QualificationError("LIVE_MECHANICS_SEQUENCE_EMPTY")
    if any(not math.isfinite(float(s.time_seconds)) for s in steps):
        raise QualificationError("LIVE_MECHANICS_SEQUENCE_TIME_NONFINITE")
    if any(b.time_seconds <= a.time_seconds for a, b in zip(steps, steps[1:])):
        raise QualificationError("LIVE_MECHANICS_SEQUENCE_TIME_NOT_STRICTLY_INCREASING")
    frames: list[MechanicsInterventionFrameV1] = []
    maximum_speed = 0.0
    maximum_acceleration = 0.0
    prev_velocity: dict[str, np.ndarray] | None = None
    prev_dt: float | None = None
    for index, step in enumerate(steps):
        frame = evaluate_mechanics_intervention_frame_v1(
            mesh, rig, skin, mechanics, overrides=step.overrides,
            motion=motion, time_seconds=(step.time_seconds if motion is not None else None),
            max_allowed_constraint_residual=max_allowed_constraint_residual,
            max_allowed_segment_strain=max_allowed_segment_strain,
        )
        if frames:
            dt = float(step.time_seconds - steps[index - 1].time_seconds)
            old = {vid: np.asarray(p, dtype=np.float64) for vid, p in zip(
                frames[-1].deformation.canonical_mesh_vertex_ids,
                frames[-1].deformation.positions,
            )}
            new = {vid: np.asarray(p, dtype=np.float64) for vid, p in zip(
                frame.deformation.canonical_mesh_vertex_ids, frame.deformation.positions,
            )}
            if set(old) != set(new):
                raise QualificationError("LIVE_MECHANICS_SEQUENCE_VERTEX_ID_DRIFT")
            velocity = {vid: (new[vid] - old[vid]) / dt for vid in old}
            maximum_speed = max(maximum_speed, *(float(np.linalg.norm(v)) for v in velocity.values()))
            if prev_velocity is not None and prev_dt is not None:
                interval = 0.5 * (dt + prev_dt)
                maximum_acceleration = max(
                    maximum_acceleration, *(
                        float(np.linalg.norm((velocity[vid] - prev_velocity[vid]) / interval))
                        for vid in velocity
                    ),
                )
            prev_velocity, prev_dt = velocity, dt
        frames.append(frame)
    return MechanicsInterventionSequenceV1(
        frames=tuple(frames),
        max_vertex_speed=maximum_speed,
        max_vertex_acceleration=maximum_acceleration,
        status=("PASS_MECHANICS_SEQUENCE_DIAGNOSTIC" if all(
            f.status == "PASS_MECHANICS_BEHAVIORAL_COURT" for f in frames
        ) else "FAIL_MECHANICS_SEQUENCE_DIAGNOSTIC"),
        metadata={
            "authority": "MECHANICS_RUNTIME_DIAGNOSTIC_ONLY",
            "all_samples_are_stateless_overrides": True,
            "motion_authority_unchanged": True,
            "training_authorized": False,
        },
    )


__all__ = [
    "MechanicsInterventionFrameV1", "MechanicsInterventionStepV1",
    "MechanicsInterventionSequenceV1",
    "evaluate_mechanics_intervention_frame_v1",
    "evaluate_mechanics_intervention_sequence_v1",
]
