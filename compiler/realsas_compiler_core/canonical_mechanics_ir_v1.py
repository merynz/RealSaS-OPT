from __future__ import annotations

"""Canonical Mechanics authority for a control-ready Rig + Skin pair.

The Compiler does not invent mechanical behavior here. It qualifies that the
already-canonical AXIS rig program is executable by the declared deterministic
Mechanics semantics, that zero-control rest is stable, and that CanonicalSkin can
be deformed without lineage or numeric failure. The canonical output seals that
execution contract for runtime/Motion consumers.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any
import math

import numpy as np

from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_lineage_hash_v1
from .canonical_rig_ir_v2 import CanonicalRigIRV2, canonical_rig_lineage_hash_v2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2, canonical_skin_lineage_hash_v2
from .hashing import content_sha256
from .mechanics_control_solver_v1 import deform_mesh_lbs_v1, solve_control_pose_v1
from .rig_constraint_schema_v1 import RIG_CONSTRAINT_OPERATOR_IDS_V1, validate_rig_constraint_operator_v1
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class CanonicalMechanicsIRV1:
    mesh_lineage_hash: str
    rig_lineage_hash: str
    deform_structure_hash: str
    control_program_hash: str
    skin_lineage_hash: str
    constraint_schedule: tuple[str, ...]
    executable_operator_ids: tuple[str, ...]
    rest_pose_hash: str
    qualification_hash: str
    mechanics_lineage_hash: str
    schema_version: str = "RealSaS.CanonicalMechanicsIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_mechanics_lineage_hash_v1(value: CanonicalMechanicsIRV1) -> str:
    payload = value.to_dict()
    payload.pop("mechanics_lineage_hash", None)
    return content_sha256(payload)


def _validate_local_frames(rig: CanonicalRigIRV2) -> None:
    by_joint = {f.canonical_joint_id: f for f in rig.local_frames}
    joint_ids = {j.canonical_joint_id for j in rig.joints}
    if set(by_joint) != joint_ids:
        raise QualificationError("MECHANICS_FRAME_COVERAGE_DRIFT")
    for jid, frame in by_joint.items():
        basis = np.asarray(frame.basis, dtype=np.float64)
        if basis.shape != (3, 3) or not np.isfinite(basis).all():
            raise QualificationError("MECHANICS_FRAME_INVALID")
        gram = basis @ basis.T
        if float(np.max(np.abs(gram - np.eye(3)))) > 1e-4:
            raise QualificationError(f"MECHANICS_FRAME_NOT_ORTHONORMAL:{jid}")
        det = float(np.linalg.det(basis))
        if not math.isfinite(det) or abs(det - 1.0) > 1e-4:
            raise QualificationError(f"MECHANICS_FRAME_NOT_RIGHT_HANDED:{jid}")


def mint_canonical_mechanics_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    *,
    rest_position_tolerance: float = 1e-5,
    rest_constraint_tolerance: float = 1e-4,
) -> CanonicalMechanicsIRV1:
    if mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh):
        raise QualificationError("MECHANICS_MESH_LINEAGE_DRIFT")
    if rig.rig_lineage_hash != canonical_rig_lineage_hash_v2(rig):
        raise QualificationError("MECHANICS_RIG_LINEAGE_DRIFT")
    if skin.skin_lineage_hash != canonical_skin_lineage_hash_v2(skin):
        raise QualificationError("MECHANICS_SKIN_LINEAGE_DRIFT")
    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash or skin.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise QualificationError("MECHANICS_PARENT_MESH_DRIFT")
    if skin.deform_structure_hash != rig.deform_structure_hash:
        raise QualificationError("MECHANICS_DEFORM_STRUCTURE_DRIFT")
    if rest_position_tolerance < 0.0 or rest_constraint_tolerance < 0.0:
        raise QualificationError("MECHANICS_TOLERANCE_INVALID")

    _validate_local_frames(rig)
    chain_ids = {c.canonical_chain_id for c in rig.chains}
    control_ids = {c.canonical_control_id for c in rig.controls}
    joint_ids = {j.canonical_joint_id for j in rig.joints}
    for constraint in rig.constraints:
        if any(cid not in control_ids for cid in constraint.input_canonical_control_ids):
            raise QualificationError("MECHANICS_CONSTRAINT_CONTROL_UNKNOWN")
        if any(jid not in joint_ids for jid in constraint.output_canonical_joint_ids):
            raise QualificationError("MECHANICS_CONSTRAINT_JOINT_UNKNOWN")
        if constraint.canonical_chain_id is not None and constraint.canonical_chain_id not in chain_ids:
            raise QualificationError("MECHANICS_CONSTRAINT_CHAIN_UNKNOWN")
        validate_rig_constraint_operator_v1(
            operator_id=constraint.operator_id,
            input_control_count=len(constraint.input_canonical_control_ids),
            output_joint_count=len(constraint.output_canonical_joint_ids),
            has_chain=constraint.canonical_chain_id is not None,
            parameters=constraint.parameters,
        )

    rest_pose = solve_control_pose_v1(rig, ())
    deformed = deform_mesh_lbs_v1(mesh, rig, skin, rest_pose)
    mesh_by_id = {str(v.canonical_mesh_vertex_id): np.asarray(v.P, dtype=np.float64) for v in mesh.vertices}
    max_rest_drift = 0.0
    for vid, value in zip(deformed.canonical_mesh_vertex_ids, deformed.positions):
        max_rest_drift = max(max_rest_drift, float(np.linalg.norm(np.asarray(value, dtype=np.float64) - mesh_by_id[vid])))
    max_constraint_residual = max((float(v) for _, v in rest_pose.constraint_residuals), default=0.0)
    if max_rest_drift > float(rest_position_tolerance):
        raise QualificationError(f"MECHANICS_REST_POSE_DRIFT:{max_rest_drift:.9g}")
    if max_constraint_residual > float(rest_constraint_tolerance):
        raise QualificationError(f"MECHANICS_REST_CONSTRAINT_RESIDUAL:{max_constraint_residual:.9g}")

    rest_pose_hash = content_sha256(rest_pose.to_dict())
    qualification = {
        "schema": "RealSaS.MechanicsQualification.v1",
        "status": "PASS_MECHANICS_EXECUTABLE",
        "frame_coverage": len(rig.local_frames),
        "control_count": len(rig.controls),
        "chain_count": len(rig.chains),
        "constraint_count": len(rig.constraints),
        "max_rest_position_drift": max_rest_drift,
        "max_rest_constraint_residual": max_constraint_residual,
        "rest_position_tolerance": float(rest_position_tolerance),
        "rest_constraint_tolerance": float(rest_constraint_tolerance),
        "constraint_solver_executed": True,
        "rig_program_mutated": False,
        "skin_mutated": False,
        "learned_inference_performed": False,
    }
    qualification_hash = content_sha256(qualification)
    value = CanonicalMechanicsIRV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        constraint_schedule=tuple(c.canonical_constraint_id for c in rig.constraints),
        executable_operator_ids=RIG_CONSTRAINT_OPERATOR_IDS_V1,
        rest_pose_hash=rest_pose_hash,
        qualification_hash=qualification_hash,
        mechanics_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_MECHANICS",
            "constraint_execution_semantics": "MECHANICS_CONTROL_SOLVER_V1",
            "post_bind_frame_observability_owner": "MECHANICS",
            "canonical_constraint_solution_rest_pose": True,
            "runtime_control_pose_solution_uses_same_solver": True,
            "compiler_rig_synthesis_performed": False,
            "motion_authority_claimed": False,
            "graphics_authority_claimed": False,
        },
    )
    return replace(value, mechanics_lineage_hash=canonical_mechanics_lineage_hash_v1(value))


def canonical_mechanics_from_dict_v1(payload: dict[str, Any]) -> CanonicalMechanicsIRV1:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalMechanicsIR.v1":
        raise QualificationError("MECHANICS_SCHEMA_DRIFT")
    value = CanonicalMechanicsIRV1(
        mesh_lineage_hash=str(payload["mesh_lineage_hash"]),
        rig_lineage_hash=str(payload["rig_lineage_hash"]),
        deform_structure_hash=str(payload["deform_structure_hash"]),
        control_program_hash=str(payload["control_program_hash"]),
        skin_lineage_hash=str(payload["skin_lineage_hash"]),
        constraint_schedule=tuple(map(str, payload.get("constraint_schedule") or ())),
        executable_operator_ids=tuple(map(str, payload.get("executable_operator_ids") or ())),
        rest_pose_hash=str(payload["rest_pose_hash"]),
        qualification_hash=str(payload["qualification_hash"]),
        mechanics_lineage_hash=str(payload["mechanics_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.mechanics_lineage_hash != canonical_mechanics_lineage_hash_v1(value):
        raise QualificationError("MECHANICS_LINEAGE_HASH_DRIFT")
    if value.metadata.get("authority") != "COMPILER_CANONICAL_MECHANICS":
        raise QualificationError("MECHANICS_AUTHORITY_DRIFT")
    return value


__all__ = [
    "CanonicalMechanicsIRV1", "canonical_mechanics_from_dict_v1",
    "canonical_mechanics_lineage_hash_v1", "mint_canonical_mechanics_v1",
]
