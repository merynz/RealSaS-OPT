from __future__ import annotations

"""Mechanics surface-to-declared-control effect reachability, version 1.

Hit testing belongs to presentation and is passed here as three CanonicalMesh
vertex IDs plus barycentric coordinates. Mechanics only computes which *already
declared* control programs could mechanically influence the hit surface.

A zero direct skin weight is NOT evidence of an inert control: ROOT/FK may
propagate to descendants; IK pole and effector inputs may affect chain joints.
Conversely, an unbound control does NOT receive an invented mechanical role.

This is causal reachability from rig/constraints, not learned semantics, not
intent inference, not auto handle synthesis and not Graphics draw authority.
It gives evidence and ambiguity rather than asserting a unique correct handle.
"""

from dataclasses import dataclass, field
import math

from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_lineage_hash_v1
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .rig_constraint_schema_v1 import validate_rig_constraint_operator_v1
from .types import QualificationError


@dataclass(frozen=True)
class MechanicsSurfaceControlCandidateV1:
    canonical_control_id: str
    control_kind: str
    support_mass: float
    influence_joint_ids: tuple[str, ...]
    causal_constraint_ids: tuple[str, ...]
    effect_paths: tuple[str, ...]


@dataclass(frozen=True)
class MechanicsSurfaceControlEvidenceV1:
    triangle_vertex_ids: tuple[str, str, str]
    barycentric: tuple[float, float, float]
    candidates: tuple[MechanicsSurfaceControlCandidateV1, ...]
    abstained_control_count: int
    status: str  # CANDIDATES | AMBIGUOUS | ABSTAIN_NO_EXECUTABLE_EFFECT
    rig_lineage_hash: str
    skin_lineage_hash: str
    metadata: dict = field(default_factory=dict)


def surface_control_effect_evidence_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    *,
    triangle_vertex_ids: tuple[str, str, str],
    barycentric: tuple[float, float, float],
    minimum_support: float = 1e-8,
    ambiguous_score_delta: float = 0.05,
    maximum_candidates: int = 12,
) -> MechanicsSurfaceControlEvidenceV1:
    if (mesh.mesh_lineage_hash != rig.mesh_lineage_hash or
            mesh.mesh_lineage_hash != skin.mesh_lineage_hash or
            rig.deform_structure_hash != skin.deform_structure_hash or
            mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh) or
            mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH"):

        raise QualificationError("MECHANICS_EFFECT_LINEAGE_DRIFT")
    if (len(triangle_vertex_ids) != 3 or len(set(triangle_vertex_ids)) != 3 or
            len(barycentric) != 3 or
            not all(math.isfinite(float(v)) and v >= 0.0 for v in barycentric) or
            abs(sum(map(float, barycentric))-1.0) > 1e-5):
        raise QualificationError("MECHANICS_EFFECT_SURFACE_BARYCENTRIC_INVALID")
    known_vertex_ids = {v.canonical_mesh_vertex_id for v in mesh.vertices}
    if len(known_vertex_ids) != len(mesh.vertices):
        raise QualificationError("MECHANICS_EFFECT_MESH_DUPLICATE_VERTEX_ID")
    if any(vid not in known_vertex_ids for vid in triangle_vertex_ids):
        raise QualificationError("MECHANICS_EFFECT_SURFACE_VERTEX_UNKNOWN")
    # A collection of three vertex IDs is not yet a surface triangle. Exact
    # canonical topology membership must be proved before control evidence.
    if frozenset(triangle_vertex_ids) not in {frozenset(f) for f in mesh.faces}:
        raise QualificationError("MECHANICS_EFFECT_HIT_NOT_CANONICAL_FACE")
    if (maximum_candidates < 1 or not math.isfinite(minimum_support)
            or not 0.0 <= minimum_support <= 1.0 or
            not math.isfinite(ambiguous_score_delta) or
            not 0.0 <= ambiguous_score_delta <= 1.0):
        raise QualificationError("MECHANICS_EFFECT_SELECTION_POLICY_INVALID")
    joints = {j.canonical_joint_id: j for j in rig.joints}
    controls = {c.canonical_control_id: c for c in rig.controls}
    chains = {c.canonical_chain_id: c for c in rig.chains}
    rows = {r.canonical_mesh_vertex_id: r for r in skin.rows}
    if (len(joints) != len(rig.joints) or len(controls) != len(rig.controls) or
            len(chains) != len(rig.chains) or len(rows) != len(skin.rows)):
        raise QualificationError("MECHANICS_EFFECT_DUPLICATE_ID")
    if any(vid not in rows for vid in triangle_vertex_ids):
        raise QualificationError("MECHANICS_EFFECT_SURFACE_VERTEX_UNKNOWN")
    children: dict[str,list[str]] = {jid:[] for jid in joints}
    for j in rig.joints:
        p=j.parent_canonical_joint_id
        if p is not None:
            if p not in joints:
                raise QualificationError("MECHANICS_EFFECT_PARENT_MISSING")
            children[p].append(j.canonical_joint_id)
    subtree_cache: dict[str,tuple[str,...]] = {}
    def subtree(root: str) -> tuple[str,...]:
        if root in subtree_cache:
            return subtree_cache[root]
        if root not in joints:
            raise QualificationError("MECHANICS_EFFECT_TARGET_JOINT_UNKNOWN")
        pending=[root];visited=set()
        while pending:
            jid=pending.pop()
            if jid in visited:
                raise QualificationError("MECHANICS_EFFECT_JOINT_CYCLE")
            visited.add(jid)
            pending.extend(children[jid])
        result=tuple(sorted(visited))
        subtree_cache[root]=result
        return result
    for jid in joints:
        subtree(jid)

    # Skin-space evidence is linear in barycentrics; explicit simplex checks
    # prevent a broken skin row from inflating candidate scores.
    joint_mass:dict[str,float]={}
    for vid,bary in zip(triangle_vertex_ids,barycentric):
        row=rows[vid]
        total=0.0
        seen_ids:set[str]=set()
        for jid,weight in row.influences:
            if jid in seen_ids:
                raise QualificationError("MECHANICS_EFFECT_DUPLICATE_SKIN_INFLUENCE")
            seen_ids.add(jid)
            w=float(weight)
            if jid not in joints or not math.isfinite(w) or w < 0.0:
                raise QualificationError("MECHANICS_EFFECT_SKIN_ROW_INVALID")
            total+=w
            joint_mass[jid]=joint_mass.get(jid,0.0)+float(bary)*w
        if abs(total-1.0)>1e-6:
            raise QualificationError("MECHANICS_EFFECT_SKIN_SIMPLEX_INVALID")
    possible:dict[str,set[str]]={cid:set() for cid in controls}
    evidence:dict[str,set[str]]={cid:set() for cid in controls}
    source_constraints:dict[str,set[str]]={cid:set() for cid in controls}
    for control in rig.controls:
        cid=control.canonical_control_id
        if control.control_kind in {"GLOBAL","ROOT","FK","SOCKET_CONTROL"}:
            for jid in control.target_canonical_joint_ids:
                possible[cid].update(subtree(jid))
                evidence[cid].add("DECLARED_DIRECT_SUBTREE")
        # An IK_EFFECTOR handle that is not referenced by an executable
        # constraint is mechanically inert in V1, regardless of a declared
        # target joint. The tip field alone is not a solver operator.
    for constraint in rig.constraints:
        validate_rig_constraint_operator_v1(
            operator_id=constraint.operator_id,
            input_control_count=len(constraint.input_canonical_control_ids),
            output_joint_count=len(constraint.output_canonical_joint_ids),
            has_chain=constraint.canonical_chain_id is not None,
            parameters=constraint.parameters,
        )
        out:set[str]=set()
        cid_list=constraint.input_canonical_control_ids
        for cid in cid_list:
            if cid not in controls:
                raise QualificationError("MECHANICS_EFFECT_CONSTRAINT_CONTROL_UNKNOWN")
        if constraint.canonical_chain_id is not None:
            chain=chains.get(constraint.canonical_chain_id)
            if chain is None:
                raise QualificationError("MECHANICS_EFFECT_CONSTRAINT_CHAIN_UNKNOWN")
            if constraint.operator_id in {"IK","STRETCH"}:
                # Execution changes only declared chain joints; it does not
                # automatically edit arbitrary sibling subtrees.
                out.update(chain.canonical_joint_ids)
        for jid in constraint.output_canonical_joint_ids:
            if jid not in joints:
                raise QualificationError("MECHANICS_EFFECT_CONSTRAINT_JOINT_UNKNOWN")
            out.add(jid)
        for cid in cid_list:
            # For mechanical interpretation, no semantic body-part names.
            possible[cid].update(out)
            source_constraints[cid].add(constraint.canonical_constraint_id)
            evidence[cid].add("DECLARED_CONSTRAINT_REACHABILITY")
    scored=[]
    abstained=0
    for control in rig.controls:
        cid=control.canonical_control_id
        affected=possible[cid]
        if not affected:
            abstained+=1
            continue
        score=float(sum(joint_mass.get(jid,0.0) for jid in affected))
        if score<minimum_support:
            abstained+=1
            continue
        scored.append(MechanicsSurfaceControlCandidateV1(
            canonical_control_id=cid,control_kind=control.control_kind,
            support_mass=score,
            influence_joint_ids=tuple(sorted(affected)),
            causal_constraint_ids=tuple(sorted(source_constraints[cid])),
            effect_paths=tuple(sorted(evidence[cid])),
        ))
    scored.sort(key=lambda c:(-c.support_mass,c.canonical_control_id))
    ambiguous=(
        len(scored)>=2
        and scored[0].support_mass-scored[1].support_mass<=ambiguous_score_delta
    )
    return MechanicsSurfaceControlEvidenceV1(
        triangle_vertex_ids=triangle_vertex_ids,
        barycentric=tuple(map(float,barycentric)),
        candidates=tuple(scored[:maximum_candidates]),
        abstained_control_count=abstained,
        status=("ABSTAIN_NO_EXECUTABLE_EFFECT" if not scored else
                "AMBIGUOUS" if ambiguous else "CANDIDATES"),
        rig_lineage_hash=rig.rig_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        metadata={
            "selection_is_not_controller_execution":True,
            "effect_comes_from_declared_rig_or_constraint_only":True,
            "source_bone_name_student_target":False,
            "new_control_synthesis_performed":False,
            "graphics_or_layer_authority_claimed":False,
            "unweighted_controls_can_have_declared_causal_effect":True,
            "canonical_face_membership_verified":True,
            "unreferenced_ik_control_not_claimed_executable":True,
            "effect_is_graph_reachability_not_observed_response":True,
        },
    )


__all__ = [
    "MechanicsSurfaceControlCandidateV1","MechanicsSurfaceControlEvidenceV1",
    "surface_control_effect_evidence_v1",
]
