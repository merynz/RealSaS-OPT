from __future__ import annotations

"""Role-free rigid surface-pair candidate discovery on canonical product parents.

This module is deliberately *not* attachment or constraint authority. It performs
an exhaustive, deterministic classification of the mechanical components already
present in ``CanonicalMeshIR`` under one explicit policy:

- a component may become a rigid candidate only when every one of its canonical
  vertices is controlled by the same canonical Rig joint above the sealed weight
  threshold;
- holder surfaces are restricted to faces controlled by a bounded ancestor set
  of that rigid joint;
- the exact closest held/holder triangle-surface pair is represented as canonical
  face-index + barycentric addresses.

The result is candidate evidence for the Mechanics causal authorization court.
``NO_*`` and ``DEFORMABLE_*`` rows are *not* negative physical truth, and a
candidate is not detachability/attachment authority. The existing held-out
relative-transport gate remains the authority for authorizing/rejecting a fixed
candidate.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np

from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_lineage_hash_v1
from .canonical_rig_ir_v1 import CanonicalRigIR, canonical_rig_lineage_hash_v1
from .canonical_skin_ir_v1 import CanonicalSkinIR, canonical_skin_lineage_hash_v1
from .hashing import content_sha256
from .rigid_attachment_surface_anchor_v1 import derive_surface_binding
from .types import QualificationError

Json = dict[str, Any]
SCHEMA = "RealSaS.MechanicsRigidCandidateDiscovery.v1"

CandidateStatus = Literal[
    "CANDIDATE_REQUIRES_CAUSAL_AUTHORIZATION",
    "DEFORMABLE_OR_MULTICONTROL",
    "NO_ANCESTOR_HOLDER_DOMAIN",
    "NO_HOLDER_SURFACE_DOMAIN",
]


@dataclass(frozen=True)
class MechanicsRigidCandidatePolicyV1:
    minimum_rigid_owner_weight: float = 0.999
    maximum_other_weight_mass: float = 0.001
    ancestor_hops: int = 2
    schema_version: str = "RealSaS.MechanicsRigidCandidatePolicy.v1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalSurfaceAddressCandidateV1:
    face_index: int
    barycentric: tuple[float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MechanicsRigidCandidateRowV1:
    component_id: str
    status: CandidateStatus
    canonical_vertex_ids: tuple[str, ...]
    rigid_owner_joint_id: str = ""
    holder_ancestor_joint_ids: tuple[str, ...] = ()
    held: CanonicalSurfaceAddressCandidateV1 | None = None
    holder: CanonicalSurfaceAddressCandidateV1 | None = None
    setup_surface_gap: float | None = None
    candidate_hash: str = ""
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MechanicsRigidCandidateDiscoveryV1:
    rows: tuple[MechanicsRigidCandidateRowV1, ...]
    mesh_lineage_hash: str
    rig_lineage_hash: str
    skin_lineage_hash: str
    policy_hash: str
    discovery_lineage_hash: str
    schema_version: str = SCHEMA
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def mechanics_rigid_candidate_policy_hash_v1(policy: MechanicsRigidCandidatePolicyV1) -> str:
    return content_sha256(policy.to_dict())


def mechanics_rigid_candidate_discovery_hash_v1(value: MechanicsRigidCandidateDiscoveryV1) -> str:
    payload = value.to_dict()
    payload.pop("discovery_lineage_hash", None)
    return content_sha256(payload)


def _validate_policy(policy: MechanicsRigidCandidatePolicyV1) -> None:
    lo = float(policy.minimum_rigid_owner_weight)
    hi = float(policy.maximum_other_weight_mass)
    hops = int(policy.ancestor_hops)
    if not (0.0 < lo <= 1.0):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_OWNER_THRESHOLD_INVALID")
    if not (0.0 <= hi < 1.0):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_OTHER_MASS_INVALID")
    if lo + hi < 1.0 - 1.0e-12:
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_POLICY_GAP")
    if hops < 1 or hops > 32:
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_ANCESTOR_HOPS_INVALID")


def _validate_parents(mesh: CanonicalMeshIR, rig: CanonicalRigIR, skin: CanonicalSkinIR) -> None:
    if not isinstance(mesh, CanonicalMeshIR) or not isinstance(rig, CanonicalRigIR) or not isinstance(skin, CanonicalSkinIR):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_PARENT_TYPE_INVALID")
    if mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_MESH_LINEAGE_DRIFT")
    if rig.rig_lineage_hash != canonical_rig_lineage_hash_v1(rig):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_RIG_LINEAGE_DRIFT")
    if skin.skin_lineage_hash != canonical_skin_lineage_hash_v1(skin):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_SKIN_LINEAGE_DRIFT")
    if mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH":
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_MESH_AUTHORITY_DRIFT")
    if rig.metadata.get("authority") != "COMPILER_CANONICAL_RIG":
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_RIG_AUTHORITY_DRIFT")
    if skin.metadata.get("authority") != "COMPILER_CANONICAL_SKIN":
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_SKIN_AUTHORITY_DRIFT")
    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_RIG_MESH_PARENT_DRIFT")
    if skin.mesh_lineage_hash != mesh.mesh_lineage_hash or skin.rig_lineage_hash != rig.rig_lineage_hash:
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_SKIN_PARENT_DRIFT")


def _canonical_arrays(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIR,
    skin: CanonicalSkinIR,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, dict[str, int]]:
    vertex_ids = tuple(str(row.canonical_mesh_vertex_id) for row in mesh.vertices)
    if not vertex_ids or len(set(vertex_ids)) != len(vertex_ids):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_VERTEX_DOMAIN_INVALID")
    vertex_index = {vertex_id: index for index, vertex_id in enumerate(vertex_ids)}
    rest = np.asarray([row.P for row in mesh.vertices], dtype=np.float64)
    if rest.shape != (len(vertex_ids), 3) or not np.isfinite(rest).all():
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_REST_INVALID")
    try:
        faces = np.asarray(
            [[vertex_index[str(vertex_id)] for vertex_id in face] for face in mesh.faces],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_FACE_VERTEX_UNKNOWN") from exc
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_FACE_DOMAIN_INVALID")

    component_ids = tuple(str(row.component_id) for row in mesh.vertices)
    if any(not value for value in component_ids):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_COMPONENT_ID_INVALID")
    for face in faces:
        if len({component_ids[int(index)] for index in face}) != 1:
            raise QualificationError("MECHANICS_RIGID_CANDIDATE_FACE_CROSSES_COMPONENT")

    joints = tuple(str(row.canonical_joint_id) for row in rig.joints)
    if not joints or len(set(joints)) != len(joints):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_JOINT_DOMAIN_INVALID")
    joint_index = {joint_id: index for index, joint_id in enumerate(joints)}
    parents = np.full(len(joints), -1, dtype=np.int64)
    for index, row in enumerate(rig.joints):
        parent = row.parent_canonical_joint_id
        if parent is not None:
            if str(parent) not in joint_index:
                raise QualificationError("MECHANICS_RIGID_CANDIDATE_PARENT_UNKNOWN")
            parents[index] = joint_index[str(parent)]

    skin_by_vertex = {str(row.canonical_mesh_vertex_id): row for row in skin.rows}
    if set(skin_by_vertex) != set(vertex_ids):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_SKIN_COVERAGE_DRIFT")
    weights = np.zeros((len(vertex_ids), len(joints)), dtype=np.float64)
    for vi, vertex_id in enumerate(vertex_ids):
        for joint_id, raw_weight in skin_by_vertex[vertex_id].influences:
            if str(joint_id) not in joint_index:
                raise QualificationError("MECHANICS_RIGID_CANDIDATE_SKIN_JOINT_UNKNOWN")
            weight = float(raw_weight)
            if not np.isfinite(weight) or weight < 0.0:
                raise QualificationError("MECHANICS_RIGID_CANDIDATE_SKIN_WEIGHT_INVALID")
            weights[vi, joint_index[str(joint_id)]] += weight
        if abs(float(weights[vi].sum()) - 1.0) > 1.0e-6:
            raise QualificationError("MECHANICS_RIGID_CANDIDATE_SKIN_SIMPLEX_DRIFT")
    return rest, faces, vertex_ids, component_ids, weights, parents, joint_index


def _component_rigid_owner(
    ids: np.ndarray,
    weights: np.ndarray,
    *,
    policy: MechanicsRigidCandidatePolicyV1,
) -> tuple[int | None, float, float]:
    if not len(ids):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_COMPONENT_EMPTY")
    rows = weights[ids]
    dominant = rows.argmax(axis=1)
    if not np.all(dominant == dominant[0]):
        return None, float(np.min(rows.max(axis=1))), float(np.max(1.0 - rows.max(axis=1)))
    owner = int(dominant[0])
    owner_weight = rows[:, owner]
    other_mass = rows.sum(axis=1) - owner_weight
    minimum_owner = float(np.min(owner_weight))
    maximum_other = float(np.max(other_mass))
    if minimum_owner < float(policy.minimum_rigid_owner_weight) or maximum_other > float(policy.maximum_other_weight_mass):
        return None, minimum_owner, maximum_other
    return owner, minimum_owner, maximum_other


def _ancestor_indices(owner_joint_index: int, parents: np.ndarray, hops: int) -> tuple[int, ...]:
    out: list[int] = []
    current = int(parents[int(owner_joint_index)])
    for _ in range(int(hops)):
        if current < 0:
            break
        out.append(current)
        current = int(parents[current])
    return tuple(out)


def _candidate_hash(row: MechanicsRigidCandidateRowV1, *, mesh_hash: str, rig_hash: str, skin_hash: str, policy_hash: str) -> str:
    payload = row.to_dict()
    payload.pop("candidate_hash", None)
    return content_sha256(
        {
            "schema": "RealSaS.MechanicsRigidCandidate.v1",
            "mesh_lineage_hash": mesh_hash,
            "rig_lineage_hash": rig_hash,
            "skin_lineage_hash": skin_hash,
            "policy_hash": policy_hash,
            "candidate": payload,
        }
    )


def discover_rigid_surface_candidates_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIR,
    skin: CanonicalSkinIR,
    *,
    policy: MechanicsRigidCandidatePolicyV1 | None = None,
) -> MechanicsRigidCandidateDiscoveryV1:
    policy = policy or MechanicsRigidCandidatePolicyV1()
    _validate_policy(policy)
    _validate_parents(mesh, rig, skin)
    rest, faces, vertex_ids, component_ids, weights, parents, joint_index = _canonical_arrays(mesh, rig, skin)
    policy_hash = mechanics_rigid_candidate_policy_hash_v1(policy)
    joint_ids = tuple(str(row.canonical_joint_id) for row in rig.joints)
    dominant = weights.argmax(axis=1)

    rows: list[MechanicsRigidCandidateRowV1] = []
    for component_id in sorted(set(component_ids)):
        ids = np.asarray([index for index, value in enumerate(component_ids) if value == component_id], dtype=np.int64)
        canonical_ids = tuple(sorted(vertex_ids[int(index)] for index in ids))
        owner_index, minimum_owner, maximum_other = _component_rigid_owner(ids, weights, policy=policy)
        common_meta: Json = {
            "candidate_only": True,
            "attachment_authority_claimed": False,
            "detachability_authority_claimed": False,
            "constraint_authority_claimed": False,
            "absence_is_not_negative_physical_truth": True,
            "minimum_owner_weight": minimum_owner,
            "maximum_other_weight_mass": maximum_other,
        }
        if owner_index is None:
            row = MechanicsRigidCandidateRowV1(
                component_id=component_id,
                status="DEFORMABLE_OR_MULTICONTROL",
                canonical_vertex_ids=canonical_ids,
                metadata=common_meta,
            )
            rows.append(replace(row, candidate_hash=_candidate_hash(row, mesh_hash=mesh.mesh_lineage_hash, rig_hash=rig.rig_lineage_hash, skin_hash=skin.skin_lineage_hash, policy_hash=policy_hash)))
            continue

        ancestors = _ancestor_indices(owner_index, parents, int(policy.ancestor_hops))
        owner_joint_id = joint_ids[owner_index]
        ancestor_joint_ids = tuple(joint_ids[index] for index in ancestors)
        if not ancestors:
            row = MechanicsRigidCandidateRowV1(
                component_id=component_id,
                status="NO_ANCESTOR_HOLDER_DOMAIN",
                canonical_vertex_ids=canonical_ids,
                rigid_owner_joint_id=owner_joint_id,
                metadata=common_meta,
            )
            rows.append(replace(row, candidate_hash=_candidate_hash(row, mesh_hash=mesh.mesh_lineage_hash, rig_hash=rig.rig_lineage_hash, skin_hash=skin.skin_lineage_hash, policy_hash=policy_hash)))
            continue

        component_mask = np.asarray([value == component_id for value in component_ids], dtype=bool)
        held_face_mask = np.all(component_mask[faces], axis=1)
        holder_face_mask = np.all(~component_mask[faces], axis=1) & np.all(np.isin(dominant[faces], np.asarray(ancestors, dtype=np.int64)), axis=1)
        holder_faces = np.flatnonzero(holder_face_mask)
        if not np.any(held_face_mask) or not len(holder_faces):
            row = MechanicsRigidCandidateRowV1(
                component_id=component_id,
                status="NO_HOLDER_SURFACE_DOMAIN",
                canonical_vertex_ids=canonical_ids,
                rigid_owner_joint_id=owner_joint_id,
                holder_ancestor_joint_ids=ancestor_joint_ids,
                metadata=common_meta,
            )
            rows.append(replace(row, candidate_hash=_candidate_hash(row, mesh_hash=mesh.mesh_lineage_hash, rig_hash=rig.rig_lineage_hash, skin_hash=skin.skin_lineage_hash, policy_hash=policy_hash)))
            continue

        owners = np.zeros(len(vertex_ids), dtype=np.int32)
        owners[ids] = 1
        binding = derive_surface_binding(
            rest_xyz=rest,
            faces=faces,
            attachment_vertex_owner=owners,
            attachment_index=1,
            holder_face_indices=holder_faces,
        )
        row = MechanicsRigidCandidateRowV1(
            component_id=component_id,
            status="CANDIDATE_REQUIRES_CAUSAL_AUTHORIZATION",
            canonical_vertex_ids=canonical_ids,
            rigid_owner_joint_id=owner_joint_id,
            holder_ancestor_joint_ids=ancestor_joint_ids,
            held=CanonicalSurfaceAddressCandidateV1(
                face_index=int(binding.held.face_index),
                barycentric=tuple(map(float, binding.held.barycentric)),
            ),
            holder=CanonicalSurfaceAddressCandidateV1(
                face_index=int(binding.holder.face_index),
                barycentric=tuple(map(float, binding.holder.barycentric)),
            ),
            setup_surface_gap=float(binding.setup_surface_gap),
            metadata={
                **common_meta,
                "holder_domain_policy": "DOMINANT_SKIN_ON_BOUNDED_OWNER_ANCESTORS",
                "surface_pair_policy": "EXACT_GLOBAL_CLOSEST_PAIR_WITHIN_DECLARED_DOMAINS",
            },
        )
        rows.append(replace(row, candidate_hash=_candidate_hash(row, mesh_hash=mesh.mesh_lineage_hash, rig_hash=rig.rig_lineage_hash, skin_hash=skin.skin_lineage_hash, policy_hash=policy_hash)))

    value = MechanicsRigidCandidateDiscoveryV1(
        rows=tuple(rows),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        policy_hash=policy_hash,
        discovery_lineage_hash="",
        metadata={
            "authority": "CANDIDATE_EVIDENCE_ONLY",
            "component_domain_exhaustively_classified_under_policy": True,
            "physical_constraint_discovery_complete": False,
            "causal_authorization_required_for_candidates": True,
            "no_candidate_status_is_not_negative_physical_truth": True,
            "categorical_or_subject_identity_used": False,
        },
    )
    return replace(value, discovery_lineage_hash=mechanics_rigid_candidate_discovery_hash_v1(value))


__all__ = [
    "CanonicalSurfaceAddressCandidateV1",
    "MechanicsRigidCandidateDiscoveryV1",
    "MechanicsRigidCandidatePolicyV1",
    "MechanicsRigidCandidateRowV1",
    "discover_rigid_surface_candidates_v1",
    "mechanics_rigid_candidate_discovery_hash_v1",
    "mechanics_rigid_candidate_policy_hash_v1",
]
