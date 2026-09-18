from __future__ import annotations

"""Compiler-owned mechanical component partition derived only from qualified S/G/W."""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .hashing import content_sha256
from .semantic_joint_roles import infer_humanoid_joint_roles
from .types import QualificationError


_ROLE_COMPONENT_IDS = {
    "HEAD": "RIGID_HEAD",
    "CHEST": "RIGID_CHEST",
    "HAND_LEFT": "RIGID_HAND_LEFT",
    "HAND_RIGHT": "RIGID_HAND_RIGHT",
}
_BODY_COMPONENT_ID = "BODY_UNDERLAY"


@dataclass(frozen=True)
class MechanicalSurfaceAssignmentIR:
    surface_id: str
    component_id: str
    mechanical_class: str
    owner_joint_id: str
    owner_weight: float
    other_mass: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MechanicalComponentPartitionIR:
    surface_binding_hash: str
    skeleton_binding_hash: str
    skin_binding_hash: str
    assignments: tuple[MechanicalSurfaceAssignmentIR, ...]
    component_surface_ids: Mapping[str, tuple[str, ...]]
    component_parent_joint_ids: Mapping[str, str]
    component_mechanical_classes: Mapping[str, str]
    qualification_report: Mapping[str, Any]
    partition_hash: str
    schema_version: str = "RealSaS.MechanicalComponentPartitionIR.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mechanical_component_partition_hash(value: MechanicalComponentPartitionIR) -> str:
    payload = value.to_dict()
    payload.pop("partition_hash", None)
    return content_sha256(payload)


def _default_role_joints(skeleton) -> dict[str, str]:
    roles = infer_humanoid_joint_roles(skeleton)
    return {
        "HEAD": str(roles.head_joint_id),
        "CHEST": str(roles.chest_joint_id),
        "HAND_LEFT": str(roles.left_hand_slot_joint_id),
        "HAND_RIGHT": str(roles.right_hand_slot_joint_id),
    }


def derive_mechanical_component_partition(
    *,
    surface,
    skeleton,
    skin,
    attachment_role_joints: Mapping[str, str] | None = None,
    min_rigid_owner_weight: float = 0.999,
    max_rigid_other_mass: float = 0.001,
    require_attachment_roles: tuple[str, ...] = ("HEAD", "CHEST", "HAND_LEFT", "HAND_RIGHT"),
) -> MechanicalComponentPartitionIR:
    """Partition visible S rows using only qualified skin mechanics and canonical roles.

    One-hot skin does not by itself imply detachability.  Only canonical attachment-role
    joints are eligible for rigid component ownership; every other qualified surface row
    stays in BODY.  Source component names, source rig IDs and teacher partitions are not
    inputs to this operator.
    """
    if skin.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_SURFACE_LINEAGE_MISMATCH")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_SKELETON_LINEAGE_MISMATCH")
    if not (0.0 < float(min_rigid_owner_weight) <= 1.0):
        raise ValueError("min_rigid_owner_weight must be within (0,1]")
    if not (0.0 <= float(max_rigid_other_mass) < 1.0):
        raise ValueError("max_rigid_other_mass must be within [0,1)")

    role_joints = dict(attachment_role_joints or _default_role_joints(skeleton))
    unknown_roles = set(role_joints) - set(_ROLE_COMPONENT_IDS)
    if unknown_roles:
        raise QualificationError(
            f"MECHANICAL_COMPONENT_PARTITION_UNKNOWN_ATTACHMENT_ROLE:{sorted(unknown_roles)}"
        )
    joint_ids = {str(j.canonical_joint_id) for j in skeleton.joints}
    if any(not str(jid) or str(jid) not in joint_ids for jid in role_joints.values()):
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_ATTACHMENT_JOINT_UNKNOWN")
    if len(set(map(str, role_joints.values()))) != len(role_joints):
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_ATTACHMENT_JOINT_ALIAS")

    surface_ids = tuple(str(node.surface_id) for node in surface.surface_nodes)
    if len(surface_ids) != len(set(surface_ids)):
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_DUPLICATE_SURFACE_ID")
    visible_ids = {
        str(node.surface_id)
        for node in surface.surface_nodes
        if tuple(getattr(node, "support_views", ()) or ())
    }
    rows = {str(row.surface_id): row for row in skin.rows}
    if len(rows) != len(skin.rows):
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_DUPLICATE_SKIN_ROW")
    if set(rows) != set(surface_ids):
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_SKIN_SURFACE_ACCOUNTING_MISMATCH")

    joint_to_role = {str(jid): str(role) for role, jid in role_joints.items()}
    assignments: list[MechanicalSurfaceAssignmentIR] = []
    members: dict[str, list[str]] = {_BODY_COMPONENT_ID: []}
    parent_by_component: dict[str, str] = {_BODY_COMPONENT_ID: ""}
    class_by_component: dict[str, str] = {_BODY_COMPONENT_ID: "DEFORMABLE_COMPONENT"}
    for role, jid in sorted(role_joints.items()):
        cid = _ROLE_COMPONENT_IDS[role]
        members[cid] = []
        parent_by_component[cid] = str(jid)
        class_by_component[cid] = "RIGID_SKINNED_COMPONENT"

    for sid in surface_ids:
        row = rows[sid]
        influences = tuple((str(jid), float(weight)) for jid, weight in row.influences)
        if not influences:
            raise QualificationError(f"MECHANICAL_COMPONENT_PARTITION_EMPTY_SKIN_ROW:{sid}")
        ranked = sorted(influences, key=lambda item: (-item[1], item[0]))
        owner_joint, owner_weight = ranked[0]
        other_mass = float(sum(weight for _jid, weight in ranked[1:]))
        role = joint_to_role.get(owner_joint)
        rigid = (
            role is not None
            and float(owner_weight) >= float(min_rigid_owner_weight)
            and float(other_mass) <= float(max_rigid_other_mass)
        )
        if rigid:
            component_id = _ROLE_COMPONENT_IDS[role]
            mechanical_class = "RIGID_SKINNED_COMPONENT"
            parent_joint = owner_joint
        else:
            component_id = _BODY_COMPONENT_ID
            mechanical_class = "DEFORMABLE_COMPONENT"
            parent_joint = ""
        members[component_id].append(sid)
        assignments.append(
            MechanicalSurfaceAssignmentIR(
                surface_id=sid,
                component_id=component_id,
                mechanical_class=mechanical_class,
                owner_joint_id=parent_joint,
                owner_weight=float(owner_weight),
                other_mass=float(other_mass),
            )
        )

    required = tuple(str(role) for role in require_attachment_roles)
    missing_roles = tuple(
        role
        for role in required
        if role not in role_joints or not members[_ROLE_COMPONENT_IDS[role]]
    )
    if missing_roles:
        raise QualificationError(
            f"MECHANICAL_COMPONENT_PARTITION_REQUIRED_ATTACHMENT_EMPTY:{missing_roles}"
        )
    if not members[_BODY_COMPONENT_ID]:
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_BODY_EMPTY")

    accounted = set().union(*(set(values) for values in members.values()))
    overlap_count = sum(len(values) for values in members.values()) - len(accounted)
    if accounted != set(surface_ids) or overlap_count:
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_SURFACE_ACCOUNTING_INVALID")
    if visible_ids - accounted:
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_VISIBLE_SURFACE_MISSING")

    component_surface_ids = {
        cid: tuple(sorted(values))
        for cid, values in sorted(members.items())
        if values
    }
    component_parent_joint_ids = {
        cid: parent_by_component[cid] for cid in component_surface_ids
    }
    component_mechanical_classes = {
        cid: class_by_component[cid] for cid in component_surface_ids
    }
    provisional = MechanicalComponentPartitionIR(
        surface_binding_hash=str(surface.geometry_lineage_hash),
        skeleton_binding_hash=str(skeleton.skeleton_lineage_hash),
        skin_binding_hash=str(skin.skin_lineage_hash),
        assignments=tuple(assignments),
        component_surface_ids=component_surface_ids,
        component_parent_joint_ids=component_parent_joint_ids,
        component_mechanical_classes=component_mechanical_classes,
        qualification_report={
            "passed": True,
            "visible_surface_accounting_fraction": 1.0,
            "surface_count": len(surface_ids),
            "body_surface_count": len(component_surface_ids[_BODY_COMPONENT_ID]),
            "rigid_surface_count": int(
                sum(
                    len(values)
                    for cid, values in component_surface_ids.items()
                    if cid != _BODY_COMPONENT_ID
                )
            ),
            "min_rigid_owner_weight": float(min_rigid_owner_weight),
            "max_rigid_other_mass": float(max_rigid_other_mass),
            "required_attachment_roles": required,
            "teacher_truth_used": False,
            "source_component_name_used": False,
            "source_rig_identity_used": False,
        },
        partition_hash="",
        metadata={
            "authority": "QUALIFIED_SURFACE_SKELETON_SKIN_ONLY",
            "component_ids_are_mechanical_roles_not_source_semantic_labels": True,
            "rigidity_does_not_imply_detachability": True,
        },
    )
    value = MechanicalComponentPartitionIR(
        **{**provisional.__dict__, "partition_hash": mechanical_component_partition_hash(provisional)}
    )
    validate_mechanical_component_partition(value, surface=surface, skeleton=skeleton, skin=skin)
    return value


def validate_mechanical_component_partition(value, *, surface, skeleton, skin) -> None:
    if value.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_SURFACE_DRIFT")
    if value.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_SKELETON_DRIFT")
    if value.skin_binding_hash != skin.skin_lineage_hash:
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_SKIN_DRIFT")
    if value.partition_hash != mechanical_component_partition_hash(value):
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_HASH_MISMATCH")
    if not bool(value.qualification_report.get("passed", False)):
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_REPORT_NOT_PASS")
    rows = {row.surface_id: row for row in value.assignments}
    surface_ids = {str(node.surface_id) for node in surface.surface_nodes}
    if set(rows) != surface_ids:
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_ASSIGNMENT_ACCOUNTING_DRIFT")
    reverse: dict[str, str] = {}
    for cid, ids in value.component_surface_ids.items():
        for sid in ids:
            if sid in reverse:
                raise QualificationError("MECHANICAL_COMPONENT_PARTITION_ASSIGNMENT_OVERLAP")
            reverse[sid] = cid
    if set(reverse) != surface_ids:
        raise QualificationError("MECHANICAL_COMPONENT_PARTITION_COMPONENT_ACCOUNTING_DRIFT")
    for sid, row in rows.items():
        if reverse[sid] != row.component_id:
            raise QualificationError("MECHANICAL_COMPONENT_PARTITION_ROW_COMPONENT_DRIFT")


__all__ = [
    "MechanicalSurfaceAssignmentIR",
    "MechanicalComponentPartitionIR",
    "mechanical_component_partition_hash",
    "derive_mechanical_component_partition",
    "validate_mechanical_component_partition",
]
