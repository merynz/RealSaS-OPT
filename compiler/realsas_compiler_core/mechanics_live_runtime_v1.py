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
from typing import Mapping, Sequence
import math

import numpy as np

from .canonical_mesh_ir_v1 import CanonicalMeshIR
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .canonical_motion_ir_v2 import CanonicalMotionIRV2
from .mechanics_prepared_skin_v1 import (
    PreparedMechanicsSkinDeformerV1,
    deform_prepared_mechanics_skin_v1,
)
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
    joint_parameter_solution_rad: tuple[tuple[str, str, float], ...] = ()
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
    *,
    prepared_skin: PreparedMechanicsSkinDeformerV1 | None = None,
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

    # Complete, finite vertex-displacement court. Vectorize the numerical
    # operation; do not relax coverage or topology authority for speed.
    ids = tuple(deformation.canonical_mesh_vertex_ids)
    if len(ids) != len(deformation.positions) or len(ids) != len(mesh.vertices):
        raise QualificationError("LIVE_MECHANICS_VERTEX_COVERAGE_DRIFT")
    positions = np.asarray(deformation.positions, dtype=np.float64)
    if positions.shape != (len(ids), 3) or not np.isfinite(positions).all():
        raise QualificationError("LIVE_MECHANICS_VERTEX_NONFINITE")
    if prepared_skin is not None:
        if (ids != prepared_skin.canonical_mesh_vertex_ids
                or len(mesh.vertices) != prepared_skin.vertex_count):
            raise QualificationError("LIVE_MECHANICS_PREPARED_VERTEX_ORDER_DRIFT")
        rest = prepared_skin.rest_vertex_positions
    else:
        mesh_ids = tuple(v.canonical_mesh_vertex_id for v in mesh.vertices)
        if len(set(mesh_ids)) != len(mesh_ids) or len(set(ids)) != len(ids):
            raise QualificationError("LIVE_MECHANICS_VERTEX_COVERAGE_DRIFT")
        rest = np.asarray([r.P for r in mesh.vertices], dtype=np.float64)
        if mesh_ids != ids:
            id_to_index = {vid: i for i, vid in enumerate(mesh_ids)}
            try:
                rest = rest[[id_to_index[vid] for vid in ids]]
            except KeyError as exc:
                raise QualificationError("LIVE_MECHANICS_VERTEX_COVERAGE_DRIFT") from exc
    if rest.shape != positions.shape or not np.isfinite(rest).all():
        raise QualificationError("LIVE_MECHANICS_VERTEX_COVERAGE_DRIFT")
    displacement = float(np.max(np.linalg.norm(positions - rest, axis=1))) if len(ids) else 0.0
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
    solver_mode: str = "FABRIK_V1",
    pinned_contact_control_ids: tuple[str, ...] = (),
    pinned_world_targets: Mapping[str, Sequence[float]] | None = None,
    initial_parameters_rad: tuple[tuple[str, str, float], ...] | None = None,
    active_parameter_keys: tuple[tuple[str, str], ...] | None = None,
    active_parameter_policy: str = "EXPLICIT_OR_ALL",
    prepared_skin: PreparedMechanicsSkinDeformerV1 | None = None,
    multi_goal_jacobian_mode: str = "NUMERIC_V1",
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
    multi_status: str | None = None
    solved_parameters: tuple[tuple[str, str, float], ...] = ()
    if solver_mode == "FABRIK_V1":
        if pinned_contact_control_ids or pinned_world_targets is not None:
            raise QualificationError("LIVE_MECHANICS_PINS_REQUIRE_MULTI_GOAL")
        if initial_parameters_rad is not None:
            raise QualificationError("LIVE_MECHANICS_WARM_START_REQUIRES_MULTI_GOAL")
        if multi_goal_jacobian_mode != "NUMERIC_V1":
            raise QualificationError("LIVE_MECHANICS_JACOBIAN_MODE_REQUIRES_MULTI_GOAL")
        if (active_parameter_keys is not None
                or active_parameter_policy != "EXPLICIT_OR_ALL"):
            raise QualificationError("LIVE_MECHANICS_DOF_POLICY_REQUIRES_MULTI_GOAL")
        pose = solve_control_pose_v1(rig, controls)
    elif solver_mode == "DECLARED_MULTI_GOAL_V1":
        from .mechanics_multigoal_solver_v1 import solve_declared_multi_goal_v1
        solved = solve_declared_multi_goal_v1(
            rig, controls, pinned_contact_control_ids=pinned_contact_control_ids,
            pinned_world_targets=pinned_world_targets,
            goal_tolerance=limit_residual,
            initial_parameters_rad=initial_parameters_rad,
            active_parameter_keys=active_parameter_keys,
            active_parameter_policy=active_parameter_policy,
            jacobian_mode=multi_goal_jacobian_mode,
        )
        pose = solved.pose
        multi_status = solved.status
        solved_parameters = solved.parameters_rad
    else:
        raise QualificationError("LIVE_MECHANICS_SOLVER_MODE_UNSUPPORTED")
    deformed = (
        deform_prepared_mechanics_skin_v1(
            prepared_skin, pose,
            expected_mesh_lineage_hash=mesh.mesh_lineage_hash,
            expected_skin_lineage_hash=skin.skin_lineage_hash,
            expected_deform_structure_hash=rig.deform_structure_hash,
        )
        if prepared_skin is not None
        else deform_mesh_lbs_v1(mesh, rig, skin, pose)
    )
    maximum_residual = max((abs(float(v)) for _, v in pose.constraint_residuals), default=0.0)
    if not math.isfinite(maximum_residual):
        raise QualificationError("LIVE_MECHANICS_RESIDUAL_NONFINITE")
    strain, displacement = _behavioral_metrics(mesh, rig, pose, deformed, prepared_skin=prepared_skin)
    failures: list[str] = []
    if maximum_residual > limit_residual:
        failures.append("CONSTRAINT_RESIDUAL_EXCEEDED")
    if strain > limit_strain:
        failures.append("SEGMENT_STRAIN_EXCEEDED")
    if multi_status is not None and multi_status != "PASS_MULTIGOAL_BEHAVIORAL_COURT":
        failures.append("MULTIGOAL_SOLVER_NOT_QUALIFIED")
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
        joint_parameter_solution_rad=solved_parameters,
        metadata={
            "authority": "MECHANICS_RUNTIME_DIAGNOSTIC_ONLY",
            "control_synthesis_performed": False,
            "rig_or_skin_mutated": False,
            "model_inference_performed": False,
            "graphics_authority_claimed": False,
            "time_seconds": time_seconds,
            "solver_mode": solver_mode,
            "pinned_contact_control_ids": tuple(pinned_contact_control_ids),
            "explicit_contact_world_anchor_count": (
                len(pinned_world_targets) if pinned_world_targets is not None else 0
            ),
            "multi_goal_status": multi_status,
            "warm_start_used": initial_parameters_rad is not None,
            "multi_goal_jacobian_mode": multi_goal_jacobian_mode,
            "active_dof_is_runtime_policy_not_rig_truth": (
                active_parameter_keys is not None
                or active_parameter_policy != "EXPLICIT_OR_ALL"
            ),
            "active_parameter_policy": active_parameter_policy,
            "deformer_kernel": (
                "PREPARED_SPARSE_NUMPY_RESEARCH_ONLY"
                if prepared_skin is not None else "LEGACY_LBS_V1"
            ),
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
    solver_mode: str = "FABRIK_V1",
    pinned_contact_control_ids: tuple[str, ...] = (),
    pinned_world_targets: Mapping[str, Sequence[float]] | None = None,
    prepared_skin: PreparedMechanicsSkinDeformerV1 | None = None,
    multi_goal_jacobian_mode: str = "NUMERIC_V1",
    active_parameter_keys: tuple[tuple[str, str], ...] | None = None,
    active_parameter_policy: str = "EXPLICIT_OR_ALL",
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
    prev_solution: tuple[tuple[str, str, float], ...] | None = None
    warm_start_frame_count = 0
    for index, step in enumerate(steps):
        frame = evaluate_mechanics_intervention_frame_v1(
            mesh, rig, skin, mechanics, overrides=step.overrides,
            motion=motion, time_seconds=(step.time_seconds if motion is not None else None),
            max_allowed_constraint_residual=max_allowed_constraint_residual,
            max_allowed_segment_strain=max_allowed_segment_strain,
            solver_mode=solver_mode,
            pinned_contact_control_ids=pinned_contact_control_ids,
            pinned_world_targets=pinned_world_targets,
            initial_parameters_rad=prev_solution if solver_mode == "DECLARED_MULTI_GOAL_V1" else None,
            prepared_skin=prepared_skin,
            multi_goal_jacobian_mode=multi_goal_jacobian_mode,
            active_parameter_keys=active_parameter_keys,
            active_parameter_policy=active_parameter_policy,
        )
        if prev_solution is not None and solver_mode == "DECLARED_MULTI_GOAL_V1":
            warm_start_frame_count += 1
        prev_solution = (
            frame.joint_parameter_solution_rad
            if frame.status == "PASS_MECHANICS_BEHAVIORAL_COURT"
               and solver_mode == "DECLARED_MULTI_GOAL_V1"
            else None
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
            "all_samples_are_stateless_overrides": solver_mode != "DECLARED_MULTI_GOAL_V1",
            "control_overrides_are_transient": True,
            "multi_goal_warm_start_frame_count": warm_start_frame_count,
            "deformer_kernel": (
                "PREPARED_SPARSE_NUMPY_RESEARCH_ONLY"
                if prepared_skin is not None else "LEGACY_LBS_V1"
            ),
            "motion_authority_unchanged": True,
            "training_authorized": False,
            "multi_goal_jacobian_mode": multi_goal_jacobian_mode,
            "active_dof_is_runtime_policy_not_rig_truth": (
                active_parameter_keys is not None
                or active_parameter_policy != "EXPLICIT_OR_ALL"
            ),
            "active_parameter_policy": active_parameter_policy,
            "solver_mode": solver_mode,
            "pinned_contact_control_ids": tuple(pinned_contact_control_ids),
            "explicit_contact_world_anchor_count": (
                len(pinned_world_targets) if pinned_world_targets is not None else 0
            ),
        },
    )


__all__ = [
    "MechanicsInterventionFrameV1", "MechanicsInterventionStepV1",
    "MechanicsInterventionSequenceV1",
    "evaluate_mechanics_intervention_frame_v1",
    "evaluate_mechanics_intervention_sequence_v1",
]
