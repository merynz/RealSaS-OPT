from __future__ import annotations

"""Product-facing canonical Rig authority.

AXIS or another learned provider may propose structure and geometry, but the
canonical Rig is minted only after Compiler validation against the exact
CanonicalMesh parent. Current AXIS V5.4.1 feature evidence is surface-derived;
that evidence remains replaceable and is deliberately not the canonical parent.

The stable Rig schema permits a non-empty deform-root set. Current AXIS V5.4.1
emits one hard causal tree; that is an implementation fact, not a product-level
single-root invariant.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_lineage_hash_v1
from .hashing import content_sha256
from .types import QualifiedSkeletonIR, QualificationError, SkeletonProposalIR

Json = dict[str, Any]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class CanonicalRigJointIR:
    canonical_joint_id: str
    P: Vec3
    parent_canonical_joint_id: str | None
    source_proposal_id: str
    proposal_support_evidence_ids: tuple[str, ...] = ()
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalRigIR:
    joints: tuple[CanonicalRigJointIR, ...]
    root_joint_ids: tuple[str, ...]
    mesh_lineage_hash: str
    mesh_carrier_evidence_hash: str
    mesh_carrier_topology_hash: str
    mesh_carrier_geometry_hash: str
    feature_evidence_binding_hash: str
    proposal_evidence_hash: str
    structure_qualification_hash: str
    rig_lineage_hash: str
    schema_version: str = "RealSaS.CanonicalRigIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_rig_lineage_hash_v1(value: CanonicalRigIR) -> str:
    payload = value.to_dict()
    payload.pop("rig_lineage_hash", None)
    return content_sha256(payload)


def _proposal_evidence_hash(proposal: SkeletonProposalIR) -> str:
    return content_sha256(
        {
            "schema": "RealSaS.RigStructureProposalEvidence.v1",
            "proposal": proposal.to_dict(),
        }
    )


def _validate_canonical_forest_v1(
    joints: tuple[CanonicalRigJointIR, ...],
    root_joint_ids: tuple[str, ...],
) -> None:
    if not joints:
        raise QualificationError("CANONICAL_RIG_EMPTY")
    joint_by_id = {str(row.canonical_joint_id): row for row in joints}
    if len(joint_by_id) != len(joints) or any(not key for key in joint_by_id):
        raise QualificationError("CANONICAL_RIG_JOINT_ID_INVALID")
    roots = tuple(sorted(map(str, root_joint_ids)))
    if not roots or len(set(roots)) != len(roots):
        raise QualificationError("CANONICAL_RIG_ROOT_SET_INVALID")
    if any(root not in joint_by_id for root in roots):
        raise QualificationError("CANONICAL_RIG_ROOT_UNKNOWN")
    declared_roots = tuple(
        sorted(
            row.canonical_joint_id
            for row in joints
            if row.parent_canonical_joint_id is None
        )
    )
    if declared_roots != roots:
        raise QualificationError("CANONICAL_RIG_ROOT_SET_DRIFT")
    root_set = set(roots)
    for row in joints:
        parent = row.parent_canonical_joint_id
        if parent is not None and parent not in joint_by_id:
            raise QualificationError("CANONICAL_RIG_PARENT_UNKNOWN")
        seen: set[str] = set()
        current = row.canonical_joint_id
        while current not in root_set:
            if current in seen:
                raise QualificationError("CANONICAL_RIG_CYCLE")
            seen.add(current)
            parent_id = joint_by_id[current].parent_canonical_joint_id
            if parent_id is None:
                raise QualificationError("CANONICAL_RIG_DISCONNECTED_FOREST")
            current = parent_id


def mint_canonical_rig_v1(
    mesh: CanonicalMeshIR,
    *,
    proposal: SkeletonProposalIR,
    qualified_skeleton: QualifiedSkeletonIR,
    feature_evidence_binding_hash: str,
) -> CanonicalRigIR:
    if mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh):
        raise QualificationError("CANONICAL_RIG_MESH_LINEAGE_HASH_DRIFT")
    if mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH":
        raise QualificationError("CANONICAL_RIG_MESH_AUTHORITY_DRIFT")

    evidence_hash = str(feature_evidence_binding_hash).strip()
    if not evidence_hash:
        raise QualificationError("CANONICAL_RIG_FEATURE_EVIDENCE_BINDING_REQUIRED")
    if proposal.surface_binding_hash != evidence_hash:
        raise QualificationError("CANONICAL_RIG_PROPOSAL_EVIDENCE_BINDING_DRIFT")
    # Current AXIS adapter consumes the same admitted surface evidence that was
    # used by Mesh qualification. This is a versioned adapter invariant, not a
    # claim that the surface representation is the canonical Rig parent.
    if mesh.source_evidence_binding_hash != evidence_hash:
        raise QualificationError("CANONICAL_RIG_MESH_EVIDENCE_BINDING_DRIFT")

    proposal_meta = dict(proposal.metadata or {})
    if proposal_meta.get("hard_causal_parent_authority") is not True:
        raise QualificationError("CANONICAL_RIG_REQUIRED_TREE_AUTHORITY_MISSING")
    if proposal_meta.get("compiler_parent_reselection_allowed") is not False:
        raise QualificationError("CANONICAL_RIG_PARENT_RESELECTION_MUST_BE_FALSE")
    if str(proposal_meta.get("compiler_role") or "") != "VALIDATE_AND_MATERIALIZE":
        raise QualificationError("CANONICAL_RIG_COMPILER_ROLE_DRIFT")

    report = dict(qualified_skeleton.qualification_report or {})
    if report.get("status") != "PASS_REQUIRED_TREE_VALIDATED":
        raise QualificationError("CANONICAL_RIG_STRUCTURE_NOT_QUALIFIED")
    if report.get("selection_search_performed") is not False:
        raise QualificationError("CANONICAL_RIG_SELECTION_SEARCH_FORBIDDEN")
    if report.get("parent_reselection_performed") is not False:
        raise QualificationError("CANONICAL_RIG_PARENT_RESELECTION_FORBIDDEN")
    bindings = {
        str(key): str(value)
        for key, value in dict(report.get("authority_bindings") or {}).items()
    }
    expected_bindings = {
        "canonical_mesh": mesh.mesh_lineage_hash,
        "mechanical_carrier_evidence": mesh.carrier_evidence_hash,
        "mechanical_carrier_topology": mesh.carrier_topology_hash,
        "mechanical_carrier_geometry": mesh.carrier_geometry_hash,
    }
    for key, expected in expected_bindings.items():
        if bindings.get(key) != expected:
            raise QualificationError(f"CANONICAL_RIG_AUTHORITY_BINDING_DRIFT:{key}")

    proposal_by_id = {str(j.proposal_id): j for j in proposal.joints}
    if len(proposal_by_id) != len(proposal.joints):
        raise QualificationError("CANONICAL_RIG_DUPLICATE_PROPOSAL_ID")
    joints: list[CanonicalRigJointIR] = []
    seen_ids: set[str] = set()
    for joint in qualified_skeleton.joints:
        canonical_id = str(joint.canonical_joint_id)
        source_id = str(joint.source_proposal_id)
        if not canonical_id or canonical_id in seen_ids:
            raise QualificationError("CANONICAL_RIG_JOINT_ID_INVALID")
        seen_ids.add(canonical_id)
        if source_id not in proposal_by_id:
            raise QualificationError("CANONICAL_RIG_SOURCE_PROPOSAL_UNKNOWN")
        P = tuple(map(float, joint.position))
        if len(P) != 3 or not all(math.isfinite(value) for value in P):
            raise QualificationError("CANONICAL_RIG_JOINT_POSITION_INVALID")
        joints.append(
            CanonicalRigJointIR(
                canonical_joint_id=canonical_id,
                P=P,  # type: ignore[arg-type]
                parent_canonical_joint_id=(
                    None
                    if joint.parent_canonical_id is None
                    else str(joint.parent_canonical_id)
                ),
                source_proposal_id=source_id,
                proposal_support_evidence_ids=tuple(
                    map(str, joint.support_surface_ids)
                ),
                metadata={
                    "support_ids_are_feature_evidence_only": True,
                    "source_proposal_identity_is_not_canonical_authority": True,
                },
            )
        )
    roots = tuple(
        sorted(
            row.canonical_joint_id
            for row in joints
            if row.parent_canonical_joint_id is None
        )
    )
    _validate_canonical_forest_v1(tuple(joints), roots)
    # The current AXIS V5.4.1 validator emits one hard required tree and its
    # legacy qualification receipt has one root_id. Do not promote that
    # implementation fact into the CanonicalRig schema.
    if len(roots) != 1:
        raise QualificationError("CANONICAL_RIG_CURRENT_AXIS_TREE_ROOT_COUNT_DRIFT")
    if str(qualified_skeleton.root_id) != roots[0]:
        raise QualificationError("CANONICAL_RIG_ROOT_DRIFT")

    value = CanonicalRigIR(
        joints=tuple(joints),
        root_joint_ids=roots,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        feature_evidence_binding_hash=evidence_hash,
        proposal_evidence_hash=_proposal_evidence_hash(proposal),
        structure_qualification_hash=str(qualified_skeleton.skeleton_lineage_hash),
        rig_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_RIG",
            "canonical_parent": "CANONICAL_MESH",
            "canonical_root_semantics": "NON_EMPTY_DEFORM_ROOT_SET",
            "current_axis_root_semantics": "SINGLE_HARD_CAUSAL_TREE_IMPLEMENTATION_DETAIL",
            "learned_family_embedded_in_schema": False,
            "learned_structure_is_proposal_evidence": True,
            "hard_required_parent_authority_preserved": True,
            "compiler_parent_reselection_performed": False,
            "feature_evidence_is_canonical_parent": False,
            "frame_observability_claimed": False,
            "post_bind_frame_observability_required": True,
            "skin_authority_claimed": False,
            "motion_authority_claimed": False,
        },
    )
    return replace(value, rig_lineage_hash=canonical_rig_lineage_hash_v1(value))


def canonical_rig_from_dict_v1(payload: Mapping[str, Any]) -> CanonicalRigIR:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalRigIR.v1":
        raise QualificationError("CANONICAL_RIG_SCHEMA_DRIFT")
    joints = tuple(
        CanonicalRigJointIR(
            canonical_joint_id=str(row["canonical_joint_id"]),
            P=tuple(map(float, row["P"])),  # type: ignore[arg-type]
            parent_canonical_joint_id=(
                None
                if row.get("parent_canonical_joint_id") is None
                else str(row["parent_canonical_joint_id"])
            ),
            source_proposal_id=str(row["source_proposal_id"]),
            proposal_support_evidence_ids=tuple(
                map(str, row.get("proposal_support_evidence_ids") or ())
            ),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("joints") or ()
    )
    value = CanonicalRigIR(
        joints=joints,
        root_joint_ids=tuple(map(str, payload.get("root_joint_ids") or ())),
        mesh_lineage_hash=str(payload["mesh_lineage_hash"]),
        mesh_carrier_evidence_hash=str(payload["mesh_carrier_evidence_hash"]),
        mesh_carrier_topology_hash=str(payload["mesh_carrier_topology_hash"]),
        mesh_carrier_geometry_hash=str(payload["mesh_carrier_geometry_hash"]),
        feature_evidence_binding_hash=str(payload["feature_evidence_binding_hash"]),
        proposal_evidence_hash=str(payload["proposal_evidence_hash"]),
        structure_qualification_hash=str(payload["structure_qualification_hash"]),
        rig_lineage_hash=str(payload["rig_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    _validate_canonical_forest_v1(value.joints, value.root_joint_ids)
    if value.rig_lineage_hash != canonical_rig_lineage_hash_v1(value):
        raise QualificationError("CANONICAL_RIG_LINEAGE_HASH_DRIFT")
    if value.metadata.get("authority") != "COMPILER_CANONICAL_RIG":
        raise QualificationError("CANONICAL_RIG_AUTHORITY_DRIFT")
    if value.metadata.get("learned_family_embedded_in_schema") is not False:
        raise QualificationError("CANONICAL_RIG_LEARNED_FAMILY_SCHEMA_FORBIDDEN")
    if value.metadata.get("frame_observability_claimed") is not False:
        raise QualificationError("CANONICAL_RIG_PREMATURE_FRAME_OBSERVABILITY")
    if str(value.metadata.get("canonical_root_semantics") or "") != "NON_EMPTY_DEFORM_ROOT_SET":
        raise QualificationError("CANONICAL_RIG_ROOT_SEMANTICS_DRIFT")
    return value


__all__ = [
    "CanonicalRigIR",
    "CanonicalRigJointIR",
    "canonical_rig_from_dict_v1",
    "canonical_rig_lineage_hash_v1",
    "mint_canonical_rig_v1",
]
