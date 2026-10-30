from __future__ import annotations

"""Deterministic Mechanics execution for CanonicalRigIR.v2 control programs.

AXIS decides *what* the rig/control/constraint program is. Compiler qualification
checks that the program is legal. This module is the execution semantics: given a
runtime control pose, solve the deform-joint pose and apply CanonicalSkin via LBS.
It contains no learned inference and never changes the canonical rig topology.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
import math

import numpy as np

from .canonical_mesh_ir_v1 import CanonicalMeshIR
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .rig_constraint_schema_v1 import validate_rig_constraint_operator_v1
from .types import QualificationError

Json = dict[str, Any]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w,x,y,z
Mat3 = tuple[Vec3, Vec3, Vec3]


@dataclass(frozen=True)
class ControlTransformIRV1:
    canonical_control_id: str
    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation_quat_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class SolvedJointTransformIRV1:
    canonical_joint_id: str
    rest_P: Vec3
    solved_P: Vec3
    rotation_basis: Mat3
    scale: Vec3 = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class MechanicsPoseIRV1:
    joints: tuple[SolvedJointTransformIRV1, ...]
    rig_lineage_hash: str
    control_program_hash: str
    constraint_residuals: tuple[tuple[str, float], ...] = ()
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DeformedMeshIRV1:
    canonical_mesh_vertex_ids: tuple[str, ...]
    positions: tuple[Vec3, ...]
    mesh_lineage_hash: str
    skin_lineage_hash: str
    rig_lineage_hash: str
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _v3(value: Sequence[float], code: str) -> np.ndarray:
    out = np.asarray(tuple(map(float, value)), dtype=np.float64)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise QualificationError(code)
    return out


def _quat_matrix(value: Sequence[float]) -> np.ndarray:
    q = np.asarray(tuple(map(float, value)), dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise QualificationError("MECHANICS_CONTROL_QUAT_INVALID")
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        raise QualificationError("MECHANICS_CONTROL_QUAT_DEGENERATE")
    w, x, y, z = q / n
    return np.asarray([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        if fallback is None:
            raise QualificationError("MECHANICS_VECTOR_DEGENERATE")
        f = np.asarray(fallback, dtype=np.float64)
        fn = float(np.linalg.norm(f))
        if fn <= 1e-12:
            raise QualificationError("MECHANICS_FALLBACK_VECTOR_DEGENERATE")
        return f / fn
    return v / n


def _minimal_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalize(a)
    b = _normalize(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-10:
        trial = np.asarray((1.0, 0.0, 0.0))
        if abs(float(np.dot(a, trial))) > 0.9:
            trial = np.asarray((0.0, 1.0, 0.0))
        axis = _normalize(np.cross(a, trial))
        return _axis_angle(axis, math.pi)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    vx = np.asarray([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = axis
    c, s = math.cos(float(angle)), math.sin(float(angle))
    C = 1.0 - c
    return np.asarray([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)


def _control_world_transforms(rig: CanonicalRigIRV2, authored: Mapping[str, ControlTransformIRV1]):
    controls = {c.canonical_control_id: c for c in rig.controls}
    unknown = set(authored) - set(controls)
    if unknown:
        raise QualificationError("MECHANICS_CONTROL_STATE_UNKNOWN:" + ",".join(sorted(unknown)))
    memo: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    visiting: set[str] = set()

    def solve(cid: str):
        if cid in memo:
            return memo[cid]
        if cid in visiting:
            raise QualificationError("MECHANICS_CONTROL_PARENT_CYCLE")
        visiting.add(cid)
        c = controls[cid]
        state = authored.get(cid, ControlTransformIRV1(cid))
        t = _v3(state.translation, "MECHANICS_CONTROL_TRANSLATION_INVALID")
        r = _quat_matrix(state.rotation_quat_wxyz)
        s = _v3(state.scale, "MECHANICS_CONTROL_SCALE_INVALID")
        if np.any(s <= 0.0):
            raise QualificationError("MECHANICS_CONTROL_SCALE_NONPOSITIVE")
        local_rest = _v3(c.rest_P, "MECHANICS_CONTROL_REST_INVALID")
        local_p = local_rest + t
        parent = c.parent_canonical_control_id
        if parent is None:
            world_p, world_r, world_s = local_p, r, s
        else:
            if parent not in controls:
                raise QualificationError("MECHANICS_CONTROL_PARENT_UNKNOWN")
            pp, pr, ps = solve(parent)
            parent_rest = _v3(controls[parent].rest_P, "MECHANICS_CONTROL_PARENT_REST_INVALID")
            relative = local_p - parent_rest
            world_p = pp + pr @ (relative * ps)
            world_r = pr @ r
            world_s = ps * s
        visiting.remove(cid)
        memo[cid] = (world_p, world_r, world_s)
        return memo[cid]

    for cid in controls:
        solve(cid)
    return memo


def _fabrik(
    rest: list[np.ndarray],
    current: list[np.ndarray],
    target: np.ndarray,
    *,
    max_stretch: float,
    iterations: int = 24,
    tolerance: float = 1e-5,
) -> tuple[list[np.ndarray], float]:
    if len(current) < 2:
        return current, float(np.linalg.norm(current[-1] - target))
    lengths = [float(np.linalg.norm(rest[i+1] - rest[i])) for i in range(len(rest)-1)]
    if any(x <= 1e-10 for x in lengths):
        raise QualificationError("MECHANICS_IK_ZERO_LENGTH_SEGMENT")
    root = current[0].copy()
    distance = float(np.linalg.norm(target - root))
    total = float(sum(lengths))
    stretch = max(1.0, float(max_stretch))
    if distance > total * stretch:
        direction = _normalize(target - root)
        scale = stretch
        out = [root]
        for length in lengths:
            out.append(out[-1] + direction * length * scale)
        return out, float(np.linalg.norm(out[-1] - target))
    if distance > total:
        # bounded stretch just enough to reach target.
        scale = min(stretch, distance / total)
        lengths = [x * scale for x in lengths]
    out = [p.copy() for p in current]
    for _ in range(max(1, int(iterations))):
        out[-1] = target.copy()
        for i in range(len(out)-2, -1, -1):
            direction = _normalize(out[i] - out[i+1], rest[i] - rest[i+1])
            out[i] = out[i+1] + direction * lengths[i]
        out[0] = root.copy()
        for i in range(len(out)-1):
            direction = _normalize(out[i+1] - out[i], rest[i+1] - rest[i])
            out[i+1] = out[i] + direction * lengths[i]
        if float(np.linalg.norm(out[-1] - target)) <= tolerance:
            break
    return out, float(np.linalg.norm(out[-1] - target))


def _apply_pole(points: list[np.ndarray], pole: np.ndarray, weight: float) -> list[np.ndarray]:
    if len(points) < 3 or weight <= 0.0:
        return points
    root, end = points[0], points[-1]
    axis = end - root
    axis_n = float(np.linalg.norm(axis))
    if axis_n <= 1e-10:
        return points
    axis = axis / axis_n
    pole_vec = pole - root
    pole_plane = pole_vec - axis * float(np.dot(pole_vec, axis))
    if float(np.linalg.norm(pole_plane)) <= 1e-10:
        return points
    pole_dir = _normalize(pole_plane)
    out = [p.copy() for p in points]
    for i in range(1, len(points)-1):
        rel = out[i] - root
        along = axis * float(np.dot(rel, axis))
        radial = rel - along
        radius = float(np.linalg.norm(radial))
        if radius <= 1e-10:
            continue
        current_dir = radial / radius
        cross = float(np.dot(axis, np.cross(current_dir, pole_dir)))
        dot = float(np.clip(np.dot(current_dir, pole_dir), -1.0, 1.0))
        angle = math.atan2(cross, dot) * min(max(float(weight), 0.0), 1.0)
        rotated = _axis_angle(axis, angle) @ radial
        out[i] = root + along + rotated
    return out


def solve_control_pose_v1(
    rig: CanonicalRigIRV2,
    control_transforms: Sequence[ControlTransformIRV1] = (),
) -> MechanicsPoseIRV1:
    joint_by = {j.canonical_joint_id: j for j in rig.joints}
    if len(joint_by) != len(rig.joints) or not joint_by:
        raise QualificationError("MECHANICS_RIG_JOINT_DOMAIN_INVALID")
    authored = {x.canonical_control_id: x for x in control_transforms}
    if len(authored) != len(control_transforms):
        raise QualificationError("MECHANICS_CONTROL_STATE_DUPLICATE")
    cw = _control_world_transforms(rig, authored)

    rest = {jid: _v3(j.P, "MECHANICS_JOINT_REST_INVALID") for jid, j in joint_by.items()}
    pos = {jid: p.copy() for jid, p in rest.items()}
    explicit_rot: dict[str, np.ndarray] = {}

    # Direct animator controls are applied before constraints. ROOT/GLOBAL/FK are
    # transform controls; IK/AIM target controls are consumed by constraints.
    for control in rig.controls:
        cid = control.canonical_control_id
        world_p, world_r, world_s = cw[cid]
        if control.control_kind not in {"GLOBAL", "ROOT", "FK", "SOCKET_CONTROL"}:
            continue
        rest_c = _v3(control.rest_P, "MECHANICS_CONTROL_REST_INVALID")
        for jid in control.target_canonical_joint_ids:
            if jid not in pos:
                raise QualificationError("MECHANICS_CONTROL_TARGET_UNKNOWN")
            rel = rest[jid] - rest_c
            pos[jid] = world_p + world_r @ (rel * world_s)
            explicit_rot[jid] = world_r

    chain_by = {c.canonical_chain_id: c for c in rig.chains}
    residuals: list[tuple[str, float]] = []
    for constraint in rig.constraints:
        validate_rig_constraint_operator_v1(
            operator_id=constraint.operator_id,
            input_control_count=len(constraint.input_canonical_control_ids),
            output_joint_count=len(constraint.output_canonical_joint_ids),
            has_chain=constraint.canonical_chain_id is not None,
            parameters=constraint.parameters,
        )
        inputs = [cw[cid] for cid in constraint.input_canonical_control_ids]
        weight = float(constraint.parameters.get("p0", 1.0))
        weight = min(max(weight, 0.0), 1.0)
        op = constraint.operator_id
        if op == "IK":
            chain = chain_by[str(constraint.canonical_chain_id)]
            ids = list(chain.canonical_joint_ids)
            if len(ids) < 2:
                raise QualificationError("MECHANICS_IK_CHAIN_TOO_SHORT")
            target = inputs[0][0]
            max_stretch = max(1.0, float(constraint.parameters.get("p2", 1.0)))
            solved, residual = _fabrik(
                [rest[jid] for jid in ids], [pos[jid] for jid in ids], target,
                max_stretch=max_stretch,
            )
            if len(inputs) > 1:
                pole_weight = float(constraint.parameters.get("p1", 1.0))
                solved = _apply_pole(solved, inputs[1][0], pole_weight)
                residual = float(np.linalg.norm(solved[-1] - target))
            for jid, value in zip(ids, solved):
                pos[jid] = pos[jid] * (1.0 - weight) + value * weight
            residuals.append((constraint.canonical_constraint_id, residual))
        elif op in {"COPY_TRANSFORM", "FOLLOW"}:
            target_p, target_r, _ = inputs[0]
            for jid in constraint.output_canonical_joint_ids:
                if jid not in pos:
                    raise QualificationError("MECHANICS_CONSTRAINT_OUTPUT_UNKNOWN")
                alpha = weight if op == "FOLLOW" else 1.0
                pos[jid] = pos[jid] * (1.0 - alpha) + target_p * alpha
                explicit_rot[jid] = target_r
            residuals.append((constraint.canonical_constraint_id, 0.0))
        elif op == "AIM":
            target_p = inputs[0][0]
            for jid in constraint.output_canonical_joint_ids:
                direction = target_p - pos[jid]
                frame = next((f for f in rig.local_frames if f.canonical_joint_id == jid), None)
                if frame is None:
                    raise QualificationError("MECHANICS_AIM_FRAME_MISSING")
                rest_axis = np.asarray(frame.basis[0], dtype=np.float64)
                if float(np.linalg.norm(direction)) > 1e-10:
                    explicit_rot[jid] = _minimal_rotation(rest_axis, direction)
            residuals.append((constraint.canonical_constraint_id, 0.0))
        elif op == "STRETCH":
            chain = chain_by[str(constraint.canonical_chain_id)]
            ids = list(chain.canonical_joint_ids)
            target = inputs[0][0]
            root = pos[ids[0]].copy()
            rest_lengths = [float(np.linalg.norm(rest[ids[i+1]] - rest[ids[i]])) for i in range(len(ids)-1)]
            total = sum(rest_lengths)
            if total <= 1e-10:
                raise QualificationError("MECHANICS_STRETCH_CHAIN_DEGENERATE")
            ratio = float(np.linalg.norm(target - root)) / total
            max_ratio = max(1.0, float(constraint.parameters.get("p1", 1.5)))
            ratio = 1.0 + (min(max(ratio, 1.0), max_ratio) - 1.0) * weight
            direction = _normalize(target - root, rest[ids[-1]] - rest[ids[0]])
            cursor = root
            pos[ids[0]] = root
            for i, length in enumerate(rest_lengths):
                cursor = cursor + direction * length * ratio
                pos[ids[i+1]] = cursor.copy()
            residuals.append((constraint.canonical_constraint_id, float(np.linalg.norm(pos[ids[-1]] - target))))
        else:  # schema validator should make this unreachable.
            raise QualificationError(f"MECHANICS_OPERATOR_UNREACHABLE:{op}")

    # Derive a stable rotation for every deform joint from rest->solved child
    # direction unless a direct/constraint rotation already exists.
    children: dict[str, list[str]] = {jid: [] for jid in joint_by}
    for joint in rig.joints:
        if joint.parent_canonical_joint_id is not None:
            children[joint.parent_canonical_joint_id].append(joint.canonical_joint_id)
    solved_rows: list[SolvedJointTransformIRV1] = []
    for joint in rig.joints:
        jid = joint.canonical_joint_id
        rotation = explicit_rot.get(jid)
        if rotation is None and children[jid]:
            child = children[jid][0]
            a = rest[child] - rest[jid]
            b = pos[child] - pos[jid]
            rotation = _minimal_rotation(a, b) if float(np.linalg.norm(b)) > 1e-10 else np.eye(3)
        if rotation is None:
            parent = joint.parent_canonical_joint_id
            rotation = explicit_rot.get(parent, np.eye(3)) if parent is not None else np.eye(3)
        solved_rows.append(SolvedJointTransformIRV1(
            canonical_joint_id=jid,
            rest_P=tuple(map(float, rest[jid])),
            solved_P=tuple(map(float, pos[jid])),
            rotation_basis=tuple(tuple(float(v) for v in row) for row in rotation),
        ))
    return MechanicsPoseIRV1(
        joints=tuple(solved_rows),
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        constraint_residuals=tuple(residuals),
        metadata={
            "authority": "MECHANICS_CONSTRAINT_EXECUTION",
            "rig_program_mutated": False,
            "learned_inference_performed": False,
            "solver_family": "DIRECT_CONTROL_PLUS_FABRIK_V1",
        },
    )


def deform_mesh_lbs_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    pose: MechanicsPoseIRV1,
) -> DeformedMeshIRV1:
    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash or skin.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise QualificationError("MECHANICS_LBS_MESH_LINEAGE_DRIFT")
    if skin.deform_structure_hash != rig.deform_structure_hash:
        raise QualificationError("MECHANICS_LBS_DEFORM_STRUCTURE_DRIFT")
    if pose.rig_lineage_hash != rig.rig_lineage_hash:
        raise QualificationError("MECHANICS_LBS_RIG_LINEAGE_DRIFT")
    joint_pose = {j.canonical_joint_id: j for j in pose.joints}
    rest_joint = {j.canonical_joint_id: _v3(j.P, "MECHANICS_LBS_JOINT_REST_INVALID") for j in rig.joints}
    skin_rows = {r.canonical_mesh_vertex_id: r for r in skin.rows}
    if len(skin_rows) != len(mesh.vertices):
        raise QualificationError("MECHANICS_LBS_SKIN_COVERAGE_DRIFT")
    out: list[Vec3] = []
    ids: list[str] = []
    for vertex in mesh.vertices:
        vid = str(vertex.canonical_mesh_vertex_id)
        row = skin_rows.get(vid)
        if row is None:
            raise QualificationError("MECHANICS_LBS_SKIN_VERTEX_MISSING")
        v = _v3(vertex.P, "MECHANICS_LBS_VERTEX_INVALID")
        result = np.zeros(3, dtype=np.float64)
        total = 0.0
        for jid, weight in row.influences:
            w = float(weight)
            if jid not in joint_pose or jid not in rest_joint:
                raise QualificationError("MECHANICS_LBS_SKIN_JOINT_UNKNOWN")
            jp = joint_pose[jid]
            R = np.asarray(jp.rotation_basis, dtype=np.float64)
            solved_p = _v3(jp.solved_P, "MECHANICS_LBS_SOLVED_JOINT_INVALID")
            transformed = solved_p + R @ (v - rest_joint[jid])
            result += w * transformed
            total += w
        if abs(total - 1.0) > 1e-6 or not np.isfinite(result).all():
            raise QualificationError("MECHANICS_LBS_RESULT_INVALID")
        ids.append(vid)
        out.append(tuple(map(float, result)))
    return DeformedMeshIRV1(
        canonical_mesh_vertex_ids=tuple(ids),
        positions=tuple(out),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        metadata={
            "deformation_method": "LINEAR_BLEND_SKINNING",
            "mechanical_mesh_authority_preserved": True,
            "visual_rebind_performed": False,
        },
    )


__all__ = [
    "ControlTransformIRV1", "DeformedMeshIRV1", "MechanicsPoseIRV1",
    "SolvedJointTransformIRV1", "deform_mesh_lbs_v1", "solve_control_pose_v1",
]
