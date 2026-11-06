from __future__ import annotations

"""Mechanics multi-goal articulation research (V1, non-shipping).

A deterministic, small/medium-rig least-squares reference court for simultaneous
declared IK effectors and planted-contact controls. It is deliberately separate
from the established FABRIK product solver, so no existing runtime behavior or
Compiler ModelRelease authority can silently change.

Rotation DOF is derived ONLY from explicitly declared CanonicalJointFrame
RX/RY/RZ labels. Optional metadata.rotation_limits_rad may supply *explicit*
limits; absent limits mean a numerical [-pi, pi] rotation chart, NOT a claim
about authored/physical joint range.

Not supported in this version: translational/scale joint DOFs, indirect pole
constraints, unrestricted FK overlays, collision, force or hard contact
dynamics. Every unsupported operator/DOF fails closed. Outcomes include
per-target WORLD-space residuals, not solver-success-only heuristics.
"""

from dataclasses import dataclass, field
from math import pi
from typing import Mapping, Sequence
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import qr

from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .mechanics_attachment_frames_v1 import _matrix
from .mechanics_attachment_feasibility_v1 import evaluate_declared_attachment_rigidity_v1
from .mechanics_control_solver_v1 import (
    ControlTransformIRV1, MechanicsPoseIRV1, SolvedJointTransformIRV1,
    _control_world_transforms,
)
from .types import QualificationError


@dataclass(frozen=True)
class MechanicsGoalResidualV1:
    canonical_constraint_id: str
    canonical_control_id: str
    canonical_tip_joint_id: str
    target_P: tuple[float, float, float]
    solved_P: tuple[float, float, float]
    residual_world: float
    goal_kind: str  # DRAG | PIN


@dataclass(frozen=True)
class MechanicsAttachmentPositionResidualV1:
    canonical_anchor_id: str
    parent_canonical_joint_id: str
    target_world_P: tuple[float, float, float]
    solved_world_P: tuple[float, float, float]
    residual_world: float


@dataclass(frozen=True)
class MechanicsOrientationGoalResidualV1:
    canonical_anchor_id: str
    parent_canonical_joint_id: str
    target_world_basis: tuple[tuple[float, float, float], ...]
    solved_world_basis: tuple[tuple[float, float, float], ...]
    residual_rad: float


@dataclass(frozen=True)
class MechanicsMultiGoalResultV1:
    pose: MechanicsPoseIRV1
    goals: tuple[MechanicsGoalResidualV1, ...]
    status: str
    max_drag_residual: float
    max_pin_residual: float
    parameters_rad: tuple[tuple[str, str, float], ...]
    solver_evaluations: int
    orientation_goals: tuple[MechanicsOrientationGoalResidualV1, ...] = ()
    attachment_position_goals: tuple[MechanicsAttachmentPositionResidualV1, ...] = ()
    max_attachment_position_residual_world: float = 0.
    max_orientation_residual_rad: float = 0.
    metadata: dict = field(default_factory=dict)


def _basis3(rows: Sequence[Sequence[float]], jid: str) -> np.ndarray:
    try:
        basis = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QualificationError("MULTIGOAL_JOINT_FRAME_INVALID:" + jid) from exc
    if basis.shape != (3, 3) or not np.isfinite(basis).all():
        raise QualificationError("MULTIGOAL_JOINT_FRAME_INVALID:" + jid)
    if (np.max(np.abs(basis @ basis.T - np.eye(3))) > 1e-4
            or abs(np.linalg.det(basis) - 1.0) > 1e-4):
        raise QualificationError("MULTIGOAL_JOINT_FRAME_NOT_ORTHONORMAL:" + jid)
    return basis.T  # local coordinates -> canonical rest coordinates


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    co, si = math.cos(angle), math.sin(angle)
    if axis == "RX":
        return np.array(((1., 0., 0.), (0., co, -si), (0., si, co)))
    if axis == "RY":
        return np.array(((co, 0., si), (0., 1., 0.), (-si, 0., co)))
    if axis == "RZ":
        return np.array(((co, -si, 0.), (si, co, 0.), (0., 0., 1.)))
    raise QualificationError("MULTIGOAL_DOF_UNSUPPORTED:" + axis)


def solve_declared_multi_goal_v1(
    rig: CanonicalRigIRV2,
    control_transforms: Sequence[ControlTransformIRV1] = (),
    *,
    pinned_contact_control_ids: Sequence[str] = (),
    pinned_world_targets: Mapping[str, Sequence[float]] | None = None,
    attachment_orientation_targets: Mapping[str, Sequence[Sequence[float]]] | None = None,
    attachment_position_targets: Mapping[str, Sequence[float]] | None = None,
    orientation_tolerance_rad: float = 1e-3,
    attachment_preflight_mode: str = "DISABLED",
    goal_tolerance: float = 1e-3,
    regularization: float = 1e-4,
    max_nfev: int = 200,
    maximum_joints: int = 128,
    maximum_dofs: int = 72,
    maximum_goals: int = 16,
    initial_parameters_rad: Sequence[tuple[str, str, float]] | None = None,
    active_parameter_keys: Sequence[tuple[str, str]] | None = None,
    active_parameter_policy: str = "EXPLICIT_OR_ALL",
    preferred_parameter_keys: Sequence[tuple[str, str]] | None = None,
    singularity_recovery_mode: str = "DISABLED",
    jacobian_mode: str = "NUMERIC_V1",
    verify_analytic_jacobian: bool = False,
) -> MechanicsMultiGoalResultV1:
    for v, code, allow_zero in (
        (goal_tolerance, "MULTIGOAL_TOLERANCE_INVALID", False),
        (regularization, "MULTIGOAL_REGULARIZATION_INVALID", True),
        (orientation_tolerance_rad, "MULTIGOAL_ORIENTATION_TOLERANCE_INVALID", False),
    ):
        if not math.isfinite(float(v)) or (float(v) < 0.0 if allow_zero else float(v) <= 0.0):
            raise QualificationError(code)
    if orientation_tolerance_rad>math.pi:
        raise QualificationError("MULTIGOAL_ORIENTATION_TOLERANCE_INVALID")
    if attachment_preflight_mode not in {"DISABLED","RIGID_HARD_NECESSARY_V1"}:
        raise QualificationError("MULTIGOAL_ATTACHMENT_PREFLIGHT_MODE_UNSUPPORTED")
    if attachment_orientation_targets is not None and jacobian_mode != "ANALYTIC_RESEARCH_V1":
        raise QualificationError("MULTIGOAL_ORIENTATION_REQUIRES_ANALYTIC")
    if attachment_position_targets is not None and jacobian_mode != "ANALYTIC_RESEARCH_V1":
        raise QualificationError("MULTIGOAL_ATTACHMENT_POSITION_REQUIRES_ANALYTIC")
    if min(max_nfev, maximum_joints, maximum_dofs, maximum_goals) < 1:
        raise QualificationError("MULTIGOAL_RESOURCE_POLICY_INVALID")

    if jacobian_mode not in {"NUMERIC_V1", "ANALYTIC_RESEARCH_V1"}:
        raise QualificationError("MULTIGOAL_JACOBIAN_MODE_UNSUPPORTED")
    if verify_analytic_jacobian and jacobian_mode != "ANALYTIC_RESEARCH_V1":
        raise QualificationError("MULTIGOAL_JACOBIAN_COURT_REQUIRES_ANALYTIC")
    if active_parameter_policy not in {"EXPLICIT_OR_ALL", "RANK_REVEALING_RESEARCH_V1"}:
        raise QualificationError("MULTIGOAL_ACTIVE_DOF_POLICY_MODE_UNSUPPORTED")
    if active_parameter_policy == "RANK_REVEALING_RESEARCH_V1" and (
        active_parameter_keys is not None or jacobian_mode != "ANALYTIC_RESEARCH_V1"
    ):
        raise QualificationError("MULTIGOAL_RANK_POLICY_REQUIRES_AUTO_ANALYTIC")
    if preferred_parameter_keys is not None and active_parameter_policy != "RANK_REVEALING_RESEARCH_V1":
        raise QualificationError("MULTIGOAL_HYSTERESIS_REQUIRES_RANK_POLICY")
    if singularity_recovery_mode not in {"DISABLED", "BOUNDED_DETERMINISTIC_SEEDS_V1"}:
        raise QualificationError("MULTIGOAL_SINGULARITY_RECOVERY_MODE_UNSUPPORTED")
    if singularity_recovery_mode != "DISABLED" and active_parameter_keys is not None:
        raise QualificationError("MULTIGOAL_SINGULARITY_RECOVERY_REQUIRES_FULL_DOF")
    if (singularity_recovery_mode != "DISABLED"
            and jacobian_mode != "ANALYTIC_RESEARCH_V1"):
        raise QualificationError("MULTIGOAL_SINGULARITY_RECOVERY_REQUIRES_ANALYTIC")

    joint_by = {j.canonical_joint_id: j for j in rig.joints}
    if not joint_by or len(joint_by) != len(rig.joints) or len(joint_by) > maximum_joints:
        raise QualificationError("MULTIGOAL_JOINT_DOMAIN_INVALID")
    frame_by = {row.canonical_joint_id: row for row in rig.local_frames}
    if len(frame_by) != len(rig.local_frames) or set(frame_by) != set(joint_by):
        raise QualificationError("MULTIGOAL_FRAME_COVERAGE_DRIFT")
    controls = {row.canonical_control_id: row for row in rig.controls}
    if len(controls) != len(rig.controls):
        raise QualificationError("MULTIGOAL_CONTROL_DOMAIN_INVALID")
    authored = {row.canonical_control_id: row for row in control_transforms}
    if len(authored) != len(control_transforms) or set(authored) - set(controls):
        raise QualificationError("MULTIGOAL_CONTROL_OVERRIDE_UNKNOWN_OR_DUPLICATE")
    pins = tuple(map(str, pinned_contact_control_ids))
    if len(set(pins)) != len(pins) or any(cid not in controls for cid in pins):
        raise QualificationError("MULTIGOAL_PIN_UNKNOWN_OR_DUPLICATE")
    if set(pins) & set(authored):
        raise QualificationError("MULTIGOAL_PIN_HAS_OVERRIDE")
    # A declared contact may be held at a previously observed WORLD position,
    # rather than silently snapping to a control's rest position each frame.
    # Exact pin-set binding avoids treating a visual pick as an implicit pin.
    if pinned_world_targets is None:
        anchor_map: dict[str, np.ndarray] = {}
    else:
        if not isinstance(pinned_world_targets, Mapping) or set(pinned_world_targets) != set(pins):
            raise QualificationError("MULTIGOAL_PIN_WORLD_TARGET_BINDING_DRIFT")
        anchor_map = {}
        for cid in pins:
            try:
                p = np.asarray(pinned_world_targets[cid], dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise QualificationError("MULTIGOAL_PIN_WORLD_TARGET_INVALID") from exc
            if p.shape != (3,) or not np.isfinite(p).all():
                raise QualificationError("MULTIGOAL_PIN_WORLD_TARGET_INVALID")
            anchor_map[cid] = p.copy()
    if any(controls[cid].control_kind != "IK_EFFECTOR" for cid in pins):
        raise QualificationError("MULTIGOAL_PIN_NOT_DECLARED_IK_EFFECTOR")
    for cid, row in authored.items():
        if controls[cid].control_kind != "IK_EFFECTOR":
            raise QualificationError("MULTIGOAL_OVERRIDE_NOT_DECLARED_IK_EFFECTOR:" + cid)
        q = np.asarray(row.rotation_quat_wxyz, dtype=np.float64)
        s = np.asarray(row.scale, dtype=np.float64)
        if (q.shape != (4,) or not np.isfinite(q).all() or
                abs(float(np.linalg.norm(q)) - 1.0) > 1e-6 or
                np.max(np.abs(q - (1., 0., 0., 0.))) > 1e-6 or
                s.shape != (3,) or not np.isfinite(s).all() or
                np.max(np.abs(s - 1.0)) > 1e-6):
            raise QualificationError("MULTIGOAL_TARGET_ONLY_TRANSLATION:" + cid)

    # Explicit causal parent graph is input authority; derive only traversal.
    topological: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(jid: str) -> None:
        if jid in visited:
            return
        if jid in visiting:
            raise QualificationError("MULTIGOAL_JOINT_PARENT_CYCLE")
        visiting.add(jid)
        p = joint_by[jid].parent_canonical_joint_id
        if p is not None:
            if p not in joint_by:
                raise QualificationError("MULTIGOAL_JOINT_PARENT_UNKNOWN")
            visit(p)
        visiting.remove(jid)
        visited.add(jid)
        topological.append(jid)

    for jid in joint_by:
        visit(jid)

    rest: dict[str, np.ndarray] = {}
    frames: dict[str, np.ndarray] = {}
    param_schema: list[tuple[str, str]] = []
    lower: list[float] = []
    upper: list[float] = []
    limits_present = 0
    for jid in topological:
        p = np.asarray(joint_by[jid].P, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise QualificationError("MULTIGOAL_REST_POSITION_INVALID")
        rest[jid] = p
        frame = frame_by[jid]
        frames[jid] = _basis3(frame.basis, jid)
        labels = tuple(map(str, frame.dof_labels))
        if len(set(labels)) != len(labels):
            raise QualificationError("MULTIGOAL_DUPLICATE_DOF:" + jid)
        if any(label not in {"RX", "RY", "RZ"} for label in labels):
            raise QualificationError("MULTIGOAL_UNSUPPORTED_DOF:" + jid)
        limits = frame.metadata.get("rotation_limits_rad", {})
        if not isinstance(limits, Mapping) or set(limits) - set(labels):
            raise QualificationError("MULTIGOAL_LIMIT_METADATA_INVALID:" + jid)
        for label in labels:
            if label in limits:
                raw = np.asarray(limits[label], dtype=np.float64)
                if (raw.shape != (2,) or not np.isfinite(raw).all()
                        or not -pi <= raw[0] < raw[1] <= pi):
                    raise QualificationError("MULTIGOAL_ROTATION_LIMIT_INVALID:" + jid)
                lo, hi = float(raw[0]), float(raw[1])
                limits_present += 1
            else:
                lo, hi = -pi, pi  # chart only: never claimed physical limits
            param_schema.append((jid, label))
            lower.append(lo)
            upper.append(hi)
    if not param_schema or len(param_schema) > maximum_dofs:
        raise QualificationError("MULTIGOAL_DOF_RESOURCE_BOUND")
    params_by_joint: dict[str, list[tuple[int, str]]] = {jid: [] for jid in joint_by}
    for index, (jid, label) in enumerate(param_schema):
        params_by_joint[jid].append((index, label))

    control_world = _control_world_transforms(rig, authored)
    chains = {row.canonical_chain_id: row for row in rig.chains}
    if len(chains) != len(rig.chains):
        raise QualificationError("MULTIGOAL_CHAIN_DUPLICATE")
    goals: list[tuple[str, str, str, np.ndarray, str]] = []
    for constraint in rig.constraints:
        if constraint.operator_id != "IK":
            raise QualificationError("MULTIGOAL_OPERATOR_UNSUPPORTED:" + constraint.operator_id)
        if len(constraint.input_canonical_control_ids) != 1:
            raise QualificationError("MULTIGOAL_POLE_NOT_YET_EXECUTABLE")
        if constraint.canonical_chain_id not in chains:
            raise QualificationError("MULTIGOAL_CHAIN_UNKNOWN")
        chain = chains[constraint.canonical_chain_id].canonical_joint_ids
        if len(chain) < 2 or any(jid not in joint_by for jid in chain):
            raise QualificationError("MULTIGOAL_CHAIN_INVALID")
        if any(joint_by[child].parent_canonical_joint_id != parent
               for parent, child in zip(chain, chain[1:])):
            raise QualificationError("MULTIGOAL_CHAIN_NOT_CONTIGUOUS")
        if (len(constraint.output_canonical_joint_ids) not in (0, 1)
                or (constraint.output_canonical_joint_ids
                    and constraint.output_canonical_joint_ids[0] != chain[-1])):
            raise QualificationError("MULTIGOAL_EFFECTOR_TIP_MISMATCH")
        if float(constraint.parameters.get("p0", 1.0)) != 1.0:
            raise QualificationError("MULTIGOAL_PARTIAL_CONSTRAINT_UNSUPPORTED")
        if float(constraint.parameters.get("p2", 1.0)) > 1.0:
            raise QualificationError("MULTIGOAL_STRETCH_UNSUPPORTED")
        cid = constraint.input_canonical_control_ids[0]
        if cid not in controls or controls[cid].control_kind != "IK_EFFECTOR":
            raise QualificationError("MULTIGOAL_IK_CONTROL_NOT_DECLARED")
        kind = "PIN" if cid in pins else "DRAG"
        target = (
            anchor_map[cid]
            if cid in anchor_map else np.asarray(control_world[cid][0], dtype=np.float64)
        )
        if target.shape != (3,) or not np.isfinite(target).all():
            raise QualificationError("MULTIGOAL_GOAL_WORLD_TARGET_INVALID")
        goals.append((constraint.canonical_constraint_id, cid, chain[-1], target, kind))
    if not goals or len(goals) > maximum_goals:
        raise QualificationError("MULTIGOAL_GOAL_RESOURCE_BOUND")
    goal_control_ids = {g[1] for g in goals}
    if set(authored) - goal_control_ids or set(pins) - goal_control_ids:
        raise QualificationError("MULTIGOAL_ORPHANED_CONTROL_OR_PIN")

    # Orientation targets may constrain ONLY an explicit declared local
    # 3D attachment frame, never a synthesized hand/grip frame.
    orientation_goals: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    attachment_position_goals: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    requested=dict(attachment_orientation_targets or {})
    requested_positions=dict(attachment_position_targets or {})
    anchors={row.canonical_anchor_id: row for row in rig.attachment_anchors}
    if len(anchors)!=len(rig.attachment_anchors):
        raise QualificationError("MULTIGOAL_ORIENTATION_DUPLICATE_ANCHOR")
    if set(requested)-set(anchors):
        raise QualificationError("MULTIGOAL_ORIENTATION_ANCHOR_UNKNOWN")
    if set(requested_positions)-set(anchors):
        raise QualificationError("MULTIGOAL_ATTACHMENT_POSITION_ANCHOR_UNKNOWN")
    if len(requested)+len(requested_positions)>maximum_goals:
        raise QualificationError("MULTIGOAL_ATTACHMENT_GOAL_RESOURCE_BOUND")
    for aid in sorted(requested_positions):
        anchor=anchors[aid]
        parent=anchor.parent_canonical_joint_id
        if (parent not in joint_by or
                anchor.parent_canonical_control_id is not None):
            raise QualificationError("MULTIGOAL_ATTACHMENT_POSITION_PARENT_NOT_EXECUTABLE")
        try:
            local=np.asarray(anchor.local_P,dtype=np.float64)
            target=np.asarray(requested_positions[aid],dtype=np.float64)
        except (TypeError,ValueError) as exc:
            raise QualificationError("MULTIGOAL_ATTACHMENT_POSITION_INVALID") from exc
        if (local.shape!=(3,) or target.shape!=(3,)
                or not np.isfinite(local).all() or not np.isfinite(target).all()):
            raise QualificationError("MULTIGOAL_ATTACHMENT_POSITION_INVALID")
        attachment_position_goals.append((aid,parent,local,target))
    for aid in sorted(requested):
        anchor=anchors[aid]
        parent=anchor.parent_canonical_joint_id
        if (parent not in joint_by or
                anchor.parent_canonical_control_id is not None):
            raise QualificationError("MULTIGOAL_ORIENTATION_PARENT_NOT_EXECUTABLE")
        local_basis=anchor.metadata.get("local_basis")
        if local_basis is None:
            raise QualificationError("MULTIGOAL_ORIENTATION_UNDECLARED_GRIP_BASIS")
        local=_matrix(local_basis,"MULTIGOAL_ORIENTATION_LOCAL_BASIS_INVALID")
        target=_matrix(requested[aid],"MULTIGOAL_ORIENTATION_TARGET_BASIS_INVALID")
        orientation_goals.append((aid,parent,local,target))

    preflight_witness_count=0
    if attachment_preflight_mode=="RIGID_HARD_NECESSARY_V1":
        preflight=evaluate_declared_attachment_rigidity_v1(
            rig,
            required_world_positions=attachment_position_targets,
            required_world_orientations=attachment_orientation_targets,
            position_tolerance_world=goal_tolerance,
            orientation_tolerance_rad=orientation_tolerance_rad,
        )
        preflight_witness_count=len(preflight.witnesses)
        if preflight_witness_count:
            first=preflight.witnesses[0]
            raise QualificationError(
                "MULTIGOAL_RIGID_ATTACHMENT_CONFLICT:"
                +first.constraint_kind+":"+",".join(first.anchor_ids)
            )

    parameter_axes = {
        "RX": np.array((1., 0., 0.)),
        "RY": np.array((0., 1., 0.)),
        "RZ": np.array((0., 0., 1.)),
    }
    ancestor_sets: dict[str, set[str]] = {}
    for jid in joint_by:
        ancestors: set[str] = set()
        current: str | None = jid
        while current is not None:
            ancestors.add(current)
            current = joint_by[current].parent_canonical_joint_id
        ancestor_sets[jid] = ancestors

    def forward(q: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        p_world: dict[str, np.ndarray] = {}
        rot_world: dict[str, np.ndarray] = {}
        world_axes = np.zeros((len(param_schema), 3), dtype=np.float64)
        for jid in topological:
            parent = joint_by[jid].parent_canonical_joint_id
            parent_rotation = rot_world[parent] if parent is not None else np.eye(3)
            local = np.eye(3)
            basis = frames[jid]
            for index, label in params_by_joint[jid]:
                if jacobian_mode == "ANALYTIC_RESEARCH_V1":
                    # World axis BEFORE this local DOF rotates.
                    world_axes[index] = parent_rotation @ basis @ local @ parameter_axes[label]
                local = local @ _axis_rotation(label, float(q[index]))
            local_world = basis @ local @ basis.T
            if parent is None:
                p_world[jid] = rest[jid].copy()
            else:
                p_world[jid] = p_world[parent] + parent_rotation @ (rest[jid] - rest[parent])
            rot_world[jid] = parent_rotation @ local_world
        return p_world, rot_world, world_axes

    def residual(q: np.ndarray) -> np.ndarray:
        p, world_rot, _ = forward(q)
        blocks = [
            (10.0 if kind == "PIN" else 1.0) * (p[jid] - target)
            for _, _, jid, target, kind in goals
        ]
        blocks.extend(
            p[parent]+world_rot[parent]@local-target
            for _,parent,local,target in attachment_position_goals
        )
        blocks.extend(
            (world_rot[parent]@local-target).ravel()
            for _,parent,local,target in orientation_goals
        )
        if regularization:
            blocks.append(math.sqrt(regularization) * q)
        return np.concatenate(blocks)

    def analytic_jacobian(q: np.ndarray) -> np.ndarray:
        p, forward_rot, world_axes = forward(q)
        task_rows=(3*len(goals)+3*len(attachment_position_goals)
                   +9*len(orientation_goals))
        rows = task_rows + (len(param_schema) if regularization else 0)
        out = np.zeros((rows, len(param_schema)), dtype=np.float64)
        for gi, (_, _, tip, _, kind) in enumerate(goals):
            weight = 10.0 if kind == "PIN" else 1.0
            for index, (pivot, _) in enumerate(param_schema):
                if pivot not in ancestor_sets[tip]:
                    continue
                out[gi * 3:gi * 3 + 3, index] = (
                    weight * np.cross(world_axes[index], p[tip] - p[pivot])
                )
        for gi,(_,parent,local,_) in enumerate(attachment_position_goals):
            grip_position=p[parent]+forward_rot[parent]@local
            start=3*len(goals)+3*gi
            for index,(pivot,_) in enumerate(param_schema):
                if pivot not in ancestor_sets[parent]:
                    continue
                out[start:start+3,index]=np.cross(
                    world_axes[index],grip_position-p[pivot]
                )
        for gi,(_, parent, local, _) in enumerate(orientation_goals):
            world_basis=forward_rot[parent]@local
            rowstart=3*len(goals)+3*len(attachment_position_goals)+9*gi
            for index,(pivot,_) in enumerate(param_schema):
                if pivot not in ancestor_sets[parent]:
                    continue
                # Exact derivative of all world rotation basis columns:
                # d(R @ local)/dtheta = [axis_world]x @ (R @ local).
                out[rowstart:rowstart+9,index]=np.cross(
                    world_axes[index],world_basis.T
                ).T.ravel()
        if regularization:
            out[task_rows:, :] = math.sqrt(regularization) * np.eye(len(param_schema))
        return out

    lower_np, upper_np = np.asarray(lower), np.asarray(upper)
    if initial_parameters_rad is None:
        x0 = np.minimum(np.maximum(np.zeros(len(param_schema)), lower_np), upper_np)
    else:
        if (len(initial_parameters_rad) != len(param_schema)
                or tuple((str(jid),str(label)) for jid,label,_ in initial_parameters_rad)
                    != tuple(param_schema)):
            raise QualificationError("MULTIGOAL_WARM_START_SCHEMA_DRIFT")
        x0 = np.asarray([float(value) for _,_,value in initial_parameters_rad],
                        dtype=np.float64)
        if (not np.isfinite(x0).all() or
                np.any(x0 < lower_np-1e-12) or np.any(x0 > upper_np+1e-12)):
            raise QualificationError("MULTIGOAL_WARM_START_OUT_OF_BOUNDS")
        x0 = np.minimum(np.maximum(x0, lower_np), upper_np)
    # Rank-revealing task-space policy considers only goal-position
    # observations. Never include diagonal regularization rows in the
    # observability rank: they are policy penalties, not evidence of effects.
    policy_candidate_rank: int | None = None
    policy_fallback = False
    policy_candidate_keys: tuple[tuple[str, str], ...] = ()
    policy_preferred_reused = False
    policy_preferred_span_error: float | None = None
    if active_parameter_policy == "RANK_REVEALING_RESEARCH_V1":
        task_jac = analytic_jacobian(x0)[
            :3*len(goals)+3*len(attachment_position_goals)+9*len(orientation_goals), :
        ]
        if not np.isfinite(task_jac).all():
            raise QualificationError("MULTIGOAL_RANK_POLICY_JACOBIAN_NONFINITE")
        _, triangular, pivot_columns = qr(task_jac, pivoting=True, mode="economic")
        diagonal = np.abs(np.diag(triangular))
        pivot_floor = max(
            1e-9,
            float(np.max(diagonal))*1e-6 if diagonal.size else 0.0,
        )
        policy_candidate_rank = int(np.count_nonzero(diagonal > pivot_floor))
        if policy_candidate_rank:
            selected_indices = sorted(
                int(i) for i in pivot_columns[:policy_candidate_rank]
            )
        else:
            # Zero linearized effects do NOT prove the nonlinear target is
            # unreachable. Preserve every DOF rather than hallucinating a
            # zero-control solution.
            selected_indices = list(range(len(param_schema)))
        # Temporal hysteresis: retain a previous valid, *independently*
        # spanning actuator basis instead of pivoting to another equally
        # capable basis merely because column norms crossed. No source
        # identity inference; this is solely runtime numerical policy.
        if preferred_parameter_keys is not None:
            try:
                prior = tuple((str(jid), str(axis)) for jid, axis in preferred_parameter_keys)
            except (TypeError, ValueError) as exc:
                raise QualificationError("MULTIGOAL_HYSTERESIS_SCHEMA_INVALID") from exc
            if (not prior or len(set(prior)) != len(prior)
                    or not set(prior).issubset(set(param_schema))):
                raise QualificationError("MULTIGOAL_HYSTERESIS_SCHEMA_INVALID")
            if policy_candidate_rank and len(prior) == policy_candidate_rank:
                prior_set=set(prior)
                prior_idx=[i for i,key in enumerate(param_schema) if key in prior_set]
                preferred_columns=task_jac[:, prior_idx]
                orthogonal, triangular_prior=np.linalg.qr(
                    preferred_columns,mode="reduced")
                preferred_rank=int(np.count_nonzero(
                    np.abs(np.diag(triangular_prior))>pivot_floor))
                if preferred_rank == policy_candidate_rank:
                    error=float(np.linalg.norm(
                        task_jac-orthogonal@(orthogonal.T@task_jac)
                    )/max(float(np.linalg.norm(task_jac)),1e-12))
                    policy_preferred_span_error=error
                    if error<=1e-4:
                        selected_indices=prior_idx
                        policy_preferred_reused=True
        active_idx = np.asarray(selected_indices, dtype=np.int64)
        policy_candidate_keys = tuple(param_schema[i] for i in active_idx)
    elif active_parameter_keys is None:
        active_idx = np.arange(len(param_schema), dtype=np.int64)
    else:
        try:
            selected = tuple((str(jid), str(axis)) for jid, axis in active_parameter_keys)
        except (ValueError, TypeError) as exc:
            raise QualificationError("MULTIGOAL_ACTIVE_DOF_POLICY_INVALID") from exc
        if (not selected or len(set(selected)) != len(selected)
                or not set(selected).issubset(set(param_schema))):
            raise QualificationError("MULTIGOAL_ACTIVE_DOF_POLICY_INVALID")
        selected_set = set(selected)
        active_idx = np.asarray(
            [i for i, key in enumerate(param_schema) if key in selected_set],
            dtype=np.int64,
        )

    def optimize_subspace(indices: np.ndarray, seed: np.ndarray):
        def expand_active(active_q: np.ndarray) -> np.ndarray:
            full = seed.copy()
            full[indices] = active_q
            return full

        def residual_active(active_q: np.ndarray) -> np.ndarray:
            return residual(expand_active(active_q))

        def jacobian_active(active_q: np.ndarray) -> np.ndarray:
            return analytic_jacobian(expand_active(active_q))[:, indices]

        optimized = least_squares(
            residual_active, seed[indices],
            bounds=(lower_np[indices], upper_np[indices]),
            jac=jacobian_active if jacobian_mode == "ANALYTIC_RESEARCH_V1" else "2-point",
            method="trf", max_nfev=int(max_nfev),
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        return optimized, expand_active(optimized.x)

    def goals_satisfied(q: np.ndarray) -> bool:
        positions, rotations, _=forward(q)
        if any(np.linalg.norm(positions[tip]-target) > goal_tolerance
               for _,_,tip,target,_ in goals):
            return False
        if any(
            np.linalg.norm(positions[parent]+rotations[parent]@local-target)
            > goal_tolerance
            for _,parent,local,target in attachment_position_goals
        ):
            return False
        return all(
            math.acos(float(np.clip(
                (np.trace(target.T@(rotations[parent]@local))-1.)/2.,
                -1.,1.,
            ))) <= orientation_tolerance_rad
            for _,parent,local,target in orientation_goals
        )

    opt, solution_q = optimize_subspace(active_idx, x0)
    candidate_nfev = int(opt.nfev)
    if active_parameter_policy == "RANK_REVEALING_RESEARCH_V1":
        candidate_failed = not bool(opt.success) or not goals_satisfied(solution_q)
        if candidate_failed:
            # Candidate policy was too restrictive for this goal/pose.
            # Retry with the entire DECLARED DOF set. This is an explicit
            # representation-policy fallback, never a change to source truth.
            policy_fallback = True
            active_idx = np.arange(len(param_schema), dtype=np.int64)
            opt, solution_q = optimize_subspace(active_idx, solution_q)
    recovery_trials=0
    recovery_rescued=False
    if singularity_recovery_mode == "BOUNDED_DETERMINISTIC_SEEDS_V1":
        first_failed = not bool(opt.success) or not goals_satisfied(solution_q)
        if first_failed:
            # Search a *finite*, documented set of perturbations, never a
            # fabricated DOF/limit or stochastic/learned prior. Straight
            # rest poses often have zero radial Jacobian, although a
            # nonlinear bent-chain solution is reachable.
            active_idx=np.arange(len(param_schema),dtype=np.int64)
            task_jac=analytic_jacobian(solution_q)[:3*len(goals)]
            drag_rows=np.concatenate([
                np.arange(3*i,3*i+3,dtype=np.int64)
                for i,g in enumerate(goals) if g[-1]=="DRAG"
            ]) if any(g[-1]=="DRAG" for g in goals) else np.arange(task_jac.shape[0])
            ranks=np.linalg.norm(task_jac[drag_rows],axis=0)
            candidate_columns=sorted(
                range(len(param_schema)), key=lambda i:(-float(ranks[i]),i)
            )[:3]
            for idx in candidate_columns:
                # Restart from the last VALID supplied pose, not from an
                # already compromised local-minimum solution. Larger angles
                # reveal second-order radial reachability at straight chains.
                for signed in (0.70,-0.70,1.30,-1.30):
                    seed=x0.copy()
                    seed[idx]=min(max(float(seed[idx]+signed),
                                      float(lower_np[idx])+1e-10),
                                  float(upper_np[idx])-1e-10)
                    if abs(seed[idx]-solution_q[idx])<1e-9:
                        continue
                    recovered, recovered_q=optimize_subspace(active_idx,seed)
                    recovery_trials+=1
                    if bool(recovered.success) and goals_satisfied(recovered_q):
                        opt,solution_q=recovered,recovered_q
                        recovery_rescued=True
                        break
                if recovery_rescued:
                    break
    p, rotations, _ = forward(solution_q)
    jacobian_debug_error = None
    if verify_analytic_jacobian:
        # Independent central difference at the actual solved joint angles.
        # The diagnostic is opt-in; source teacher labels never enter here.
        h = 1e-6
        numeric_columns = []
        for index in range(len(param_schema)):
            delta = np.zeros(len(param_schema), dtype=np.float64)
            delta[index] = h
            numeric_columns.append((residual(solution_q + delta) - residual(solution_q - delta)) / (2 * h))
        numeric = np.column_stack(numeric_columns)
        jacobian_debug_error = float(np.max(np.abs(numeric - analytic_jacobian(solution_q))))
        if (not math.isfinite(jacobian_debug_error)
                or jacobian_debug_error > 1e-5):
            raise QualificationError("MULTIGOAL_ANALYTIC_JACOBIAN_DIFFERENTIAL_FAILURE")
    reported_goals: list[MechanicsGoalResidualV1] = []
    for constraint_id, cid, jid, target, kind in goals:
        reported_goals.append(MechanicsGoalResidualV1(
            canonical_constraint_id=constraint_id,
            canonical_control_id=cid,
            canonical_tip_joint_id=jid,
            target_P=tuple(map(float, target)),
            solved_P=tuple(map(float, p[jid])),
            residual_world=float(np.linalg.norm(p[jid] - target)),
            goal_kind=kind,
        ))
    reported_attachment_positions=tuple(
        MechanicsAttachmentPositionResidualV1(
            canonical_anchor_id=aid,
            parent_canonical_joint_id=parent,
            target_world_P=tuple(map(float,target)),
            solved_world_P=tuple(map(float,p[parent]+rotations[parent]@local)),
            residual_world=float(np.linalg.norm(
                p[parent]+rotations[parent]@local-target
            )),
        ) for aid,parent,local,target in attachment_position_goals
    )
    attachment_position_max=max(
        (r.residual_world for r in reported_attachment_positions),default=0.
    )
    reported_orientations=tuple(
        MechanicsOrientationGoalResidualV1(
            canonical_anchor_id=aid,
            parent_canonical_joint_id=parent,
            target_world_basis=tuple(tuple(map(float,row)) for row in target),
            solved_world_basis=tuple(
                tuple(map(float,row)) for row in (rotations[parent]@local)
            ),
            residual_rad=math.acos(float(np.clip(
                (np.trace(target.T@(rotations[parent]@local))-1.)/2.,
                -1.,1.,
            ))),
        ) for aid,parent,local,target in orientation_goals
    )
    orientation_max=max((r.residual_rad for r in reported_orientations),default=0.)
    drag_max = max((g.residual_world for g in reported_goals if g.goal_kind == "DRAG"), default=0.0)
    pin_max = max((g.residual_world for g in reported_goals if g.goal_kind == "PIN"), default=0.0)
    passes = bool(
        opt.success and drag_max<=goal_tolerance and pin_max<=goal_tolerance
        and orientation_max<=orientation_tolerance_rad
        and attachment_position_max<=goal_tolerance
    )
    rows = tuple(SolvedJointTransformIRV1(
        canonical_joint_id=jid,
        rest_P=tuple(map(float, rest[jid])),
        solved_P=tuple(map(float, p[jid])),
        rotation_basis=tuple(tuple(float(v) for v in row) for row in rotations[jid]),
    ) for jid in joint_by)
    pose = MechanicsPoseIRV1(
        joints=rows, rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        constraint_residuals=tuple(
            (g.canonical_constraint_id, g.residual_world) for g in reported_goals
        ),
        metadata={
            "authority": "MECHANICS_RESEARCH_MULTI_GOAL_ONLY",
            "solver_family": "SCIPY_ROTATIONAL_FK_LEAST_SQUARES_V1",
            "model_inference_performed": False,
            "graphics_authority_claimed": False,
            "rig_program_mutated": False,
            "contact_is_world_position_pin_not_force_dynamics": True,
        },
    )
    return MechanicsMultiGoalResultV1(
        pose=pose, goals=tuple(reported_goals),
        status=("PASS_MULTIGOAL_BEHAVIORAL_COURT" if passes
                else "FAIL_MULTIGOAL_BEHAVIORAL_COURT"),
        max_drag_residual=drag_max, max_pin_residual=pin_max,
        parameters_rad=tuple(
            (jid, label, float(value))
            for (jid, label), value in zip(param_schema, solution_q)
        ),
        solver_evaluations=int(opt.nfev),
        orientation_goals=reported_orientations,
        attachment_position_goals=reported_attachment_positions,
        max_orientation_residual_rad=orientation_max,
        max_attachment_position_residual_world=attachment_position_max,
        metadata={
            "solver_termination": str(opt.message),
            "jacobian_mode": jacobian_mode,
            "analytic_jacobian_finite_difference_max_error": jacobian_debug_error,
            "optimization_converged": bool(opt.success),
            "declared_angle_limit_count": limits_present,
            "unbounded_rotation_chart_count": len(param_schema) - limits_present,
            "pinned_contact_count": len(pins),
            "declared_attachment_orientation_goal_count": len(orientation_goals),
            "declared_attachment_position_goal_count": len(attachment_position_goals),
            "attachment_preflight_mode": attachment_preflight_mode,
            "attachment_preflight_pairwise_witness_count": preflight_witness_count,
            "preflight_is_necessary_not_global_feasibility": True,
            "attachment_orientation_chordal_objective_geodesic_final_gate": True,
            "explicit_world_contact_anchor_count": len(anchor_map),
            "contact_anchors_are_declared_control_references": True,
            "warm_start_supplied": initial_parameters_rad is not None,
            "active_solver_parameter_count": int(len(active_idx)),
            "frozen_solver_parameter_count": int(len(param_schema) - len(active_idx)),
            "active_dof_selection_is_representation_policy_not_source_truth": True,
            "active_parameter_policy": active_parameter_policy,
            "rank_revealing_candidate_task_rank": policy_candidate_rank,
            "rank_revealing_candidate_parameter_keys": policy_candidate_keys,
            "rank_revealing_prior_basis_reused": policy_preferred_reused,
            "rank_revealing_prior_span_relative_error": policy_preferred_span_error,
            "rank_revealing_fallback_to_all": policy_fallback,
            "rank_revealing_candidate_solver_nfev": candidate_nfev,
            "singularity_recovery_mode": singularity_recovery_mode,
            "singularity_recovery_trials": recovery_trials,
            "singularity_recovery_rescued": recovery_rescued,
            "singularity_recovery_is_bounded_local_multistart_not_global_proof": True,
            "compiler_rig_synthesis_performed": False,
            "training_authority_minted": False,
            "product_mechanics_release_minted": False,
        },
    )


__all__ = [
    "MechanicsGoalResidualV1", "MechanicsOrientationGoalResidualV1",
    "MechanicsAttachmentPositionResidualV1", "MechanicsMultiGoalResultV1",
    "solve_declared_multi_goal_v1",
]
