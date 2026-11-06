from __future__ import annotations

"""Mechanics-only implicit control selection for a picked carrier triangle.

This is a read-only runtime selection of ALREADY DECLARED controls; it never
synthesizes controls, animates graphics, or mints CanonicalRig authority.
The caller supplies the hit triangle and barycentrics from a presentation hit
test. Returned control IDs are opaque; only Mechanics interprets their action.
"""

from dataclasses import dataclass
import math

from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .types import QualificationError


@dataclass(frozen=True)
class MechanicsPickCandidateV1:
    canonical_control_id: str
    control_kind: str
    influence_score: float
    supporting_joint_ids: tuple[str, ...]


@dataclass(frozen=True)
class MechanicsSurfacePickV1:
    triangle_vertex_ids: tuple[str, str, str]
    barycentric: tuple[float, float, float]
    candidates: tuple[MechanicsPickCandidateV1, ...]
    rig_lineage_hash: str
    skin_lineage_hash: str
    status: str  # CANDIDATES | ABSTAIN_NO_DECLARED_CONTROL


def select_mechanics_controls_from_surface_v1(
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    *,
    triangle_vertex_ids: tuple[str, str, str],
    barycentric: tuple[float, float, float],
    maximum_candidates: int = 8,
    minimum_support: float = 1e-6,
) -> MechanicsSurfacePickV1:
    if rig.mesh_lineage_hash != skin.mesh_lineage_hash:
        raise QualificationError("MECHANICS_PICK_MESH_LINEAGE_DRIFT")
    if rig.deform_structure_hash != skin.deform_structure_hash:
        raise QualificationError("MECHANICS_PICK_DEFORM_LINEAGE_DRIFT")
    if maximum_candidates < 1 or not (0.0 <= minimum_support <= 1.0):
        raise QualificationError("MECHANICS_PICK_POLICY_INVALID")
    if len(triangle_vertex_ids) != 3 or len(set(triangle_vertex_ids)) != 3:
        raise QualificationError("MECHANICS_PICK_TRIANGLE_INVALID")
    if len(barycentric) != 3 or not all(math.isfinite(float(x)) and float(x) >= 0.0 for x in barycentric):
        raise QualificationError("MECHANICS_PICK_BARYCENTRIC_INVALID")
    mass = sum(barycentric)
    if abs(mass - 1.0) > 1e-5:
        raise QualificationError("MECHANICS_PICK_BARYCENTRIC_NOT_SIMPLEX")

    by_vertex = {row.canonical_mesh_vertex_id: row for row in skin.rows}
    if len(by_vertex) != len(skin.rows):
        raise QualificationError("MECHANICS_PICK_DUPLICATE_SKIN_VERTEX")
    if any(vid not in by_vertex for vid in triangle_vertex_ids):
        raise QualificationError("MECHANICS_PICK_SKIN_VERTEX_UNKNOWN")

    influence: dict[str, float] = {}
    for vid, bary in zip(triangle_vertex_ids, barycentric):
        for joint_id, weight in by_vertex[vid].influences:
            value = float(weight)
            if not math.isfinite(value) or value < 0.0:
                raise QualificationError("MECHANICS_PICK_SKIN_WEIGHT_INVALID")
            influence[joint_id] = influence.get(joint_id, 0.0) + float(bary) * value

    joint_ids = {j.canonical_joint_id for j in rig.joints}
    if not set(influence).issubset(joint_ids):
        raise QualificationError("MECHANICS_PICK_JOINT_UNKNOWN")
    chains = {row.canonical_chain_id: row for row in rig.chains}
    # IK effectors are recognized only through their DECLARED constraint inputs.
    ik_targets: dict[str, set[str]] = {}
    for constraint in rig.constraints:
        if constraint.operator_id != "IK" or constraint.canonical_chain_id is None:
            continue
        chain = chains.get(constraint.canonical_chain_id)
        if chain is None or not chain.canonical_joint_ids:
            raise QualificationError("MECHANICS_PICK_CHAIN_UNKNOWN")
        tip = chain.canonical_joint_ids[-1]
        for control_id in constraint.input_canonical_control_ids[:1]:
            ik_targets.setdefault(control_id, set()).add(tip)

    candidates: list[MechanicsPickCandidateV1] = []
    seen: set[str] = set()
    for control in rig.controls:
        cid = control.canonical_control_id
        if cid in seen:
            raise QualificationError("MECHANICS_PICK_DUPLICATE_CONTROL")
        seen.add(cid)
        if control.control_kind not in {"IK_EFFECTOR", "FK", "ROOT", "SPLINE", "AIM_TARGET", "SOCKET_CONTROL"}:
            continue
        targets = set(control.target_canonical_joint_ids)
        if control.control_kind == "IK_EFFECTOR":
            targets.update(ik_targets.get(cid, set()))
        if not targets or not targets.issubset(joint_ids):
            continue
        support = sum(influence.get(jid, 0.0) for jid in targets)
        if support < minimum_support:
            continue
        candidates.append(MechanicsPickCandidateV1(
            canonical_control_id=cid,
            control_kind=control.control_kind,
            influence_score=float(support),
            supporting_joint_ids=tuple(sorted(targets)),
        ))
    candidates.sort(key=lambda x: (-x.influence_score, x.canonical_control_id))
    return MechanicsSurfacePickV1(
        triangle_vertex_ids=triangle_vertex_ids,
        barycentric=tuple(map(float, barycentric)),
        candidates=tuple(candidates[:maximum_candidates]),
        rig_lineage_hash=rig.rig_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        status="CANDIDATES" if candidates else "ABSTAIN_NO_DECLARED_CONTROL",
    )


__all__ = [
    "MechanicsPickCandidateV1", "MechanicsSurfacePickV1",
    "select_mechanics_controls_from_surface_v1",
]
