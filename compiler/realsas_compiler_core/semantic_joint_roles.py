from __future__ import annotations

"""Deterministic canonical joint-role binding from qualified rest topology.

This module intentionally does *not* use source proposal indices, source joint names,
skin-weight maxima, filenames, or component names to infer canonical attachment roles.
Typed source evidence may ask for a semantic role (for example HEAD or HAND_SLOT_LEFT),
but the role is resolved against the current qualified skeleton from topology and rest
geometry only. The output is hash-bound so component assembly and motion can share one
canonical role authority.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError


@dataclass(frozen=True)
class HumanoidJointRoleBindingIR:
    root_joint_id: str
    spine_chain: tuple[str, ...]
    chest_joint_id: str
    head_joint_id: str
    left_arm_chain: tuple[str, ...]
    right_arm_chain: tuple[str, ...]
    left_leg_chain: tuple[str, ...]
    right_leg_chain: tuple[str, ...]
    left_hand_slot_joint_id: str
    right_hand_slot_joint_id: str
    binding_hash: str
    schema_version: str = "RealSaS.HumanoidJointRoleBindingIR.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def humanoid_joint_role_binding_hash(value: HumanoidJointRoleBindingIR) -> str:
    payload = value.to_dict()
    payload.pop("binding_hash", None)
    return content_sha256(payload)


def _children_and_by_id(skeleton):
    joints = tuple(getattr(skeleton, "joints", ()))
    if len(joints) < 7:
        raise QualificationError("HUMANOID_ROLE_REQUIRES_HUMANOID_TOPOLOGY")
    by_id = {str(j.canonical_joint_id): j for j in joints}
    if len(by_id) != len(joints) or any(not jid for jid in by_id):
        raise QualificationError("HUMANOID_ROLE_REQUIRES_UNIQUE_CANONICAL_JOINT_IDS")
    children = {jid: [] for jid in by_id}
    for joint in joints:
        parent = joint.parent_canonical_id
        if parent is None:
            continue
        parent = str(parent)
        if parent not in by_id:
            raise QualificationError("HUMANOID_ROLE_PARENT_JOINT_MISSING")
        children[parent].append(str(joint.canonical_joint_id))
    for rows in children.values():
        rows.sort()
    return joints, children, by_id


def _chain_from(start: str, children: Mapping[str, list[str]]) -> tuple[str, ...]:
    chain = [start]
    current = start
    seen = {start}
    while len(children[current]) == 1:
        current = children[current][0]
        if current in seen:
            raise QualificationError("HUMANOID_ROLE_TOPOLOGY_CYCLE")
        seen.add(current)
        chain.append(current)
    return tuple(chain)


def infer_humanoid_joint_roles(skeleton) -> HumanoidJointRoleBindingIR:
    """Infer bounded humanoid roles from canonical topology/rest geometry only."""
    joints, children, by_id = _children_and_by_id(skeleton)
    roots = tuple(sorted(str(j.canonical_joint_id) for j in joints if j.parent_canonical_id is None))
    declared = tuple(str(x) for x in tuple(getattr(skeleton, "deform_root_ids", ())) if str(x) in by_id)
    root_candidates = declared or roots
    if len(root_candidates) != 1:
        raise QualificationError("HUMANOID_ROLE_REQUIRES_SINGLE_DEFORM_ROOT")
    root = root_candidates[0]
    root_pos = by_id[root].position
    root_children = tuple(children[root])
    if len(root_children) < 3:
        raise QualificationError("HUMANOID_ROLE_REQUIRES_SPINE_AND_TWO_LEGS")

    def central_up_score(jid: str):
        p = by_id[jid].position
        dz = float(p[2]) - float(root_pos[2])
        return (0 if dz > 0.0 else 1, abs(float(p[0]) - float(root_pos[0])), -dz, jid)

    spine_start = min(root_children, key=central_up_score)
    if float(by_id[spine_start].position[2]) <= float(root_pos[2]):
        raise QualificationError("HUMANOID_ROLE_CANNOT_INFER_UPWARD_SPINE")

    leg_starts = sorted(
        (jid for jid in root_children if jid != spine_start),
        key=lambda jid: (float(by_id[jid].position[0]), jid),
    )
    if len(leg_starts) < 2:
        raise QualificationError("HUMANOID_ROLE_REQUIRES_TWO_LEG_BRANCHES")
    leg_starts = leg_starts[:2]

    spine = [spine_start]
    current = spine_start
    while len(children[current]) == 1:
        nxt = children[current][0]
        if float(by_id[nxt].position[2]) <= float(by_id[current].position[2]):
            break
        spine.append(nxt)
        current = nxt
    chest = spine[-1]
    chest_pos = by_id[chest].position
    chest_children = tuple(children[chest])
    if len(chest_children) < 3:
        raise QualificationError("HUMANOID_ROLE_REQUIRES_HEAD_AND_TWO_ARM_BRANCHES")

    head = min(
        chest_children,
        key=lambda jid: (
            abs(float(by_id[jid].position[0]) - float(chest_pos[0])),
            -float(by_id[jid].position[2]),
            jid,
        ),
    )
    if float(by_id[head].position[2]) <= float(chest_pos[2]):
        raise QualificationError("HUMANOID_ROLE_HEAD_NOT_ABOVE_CHEST")

    arm_starts = sorted(
        (jid for jid in chest_children if jid != head),
        key=lambda jid: (float(by_id[jid].position[0]), jid),
    )
    if len(arm_starts) < 2:
        raise QualificationError("HUMANOID_ROLE_REQUIRES_TWO_ARM_BRANCHES")
    arm_starts = arm_starts[:2]

    left_arm = _chain_from(arm_starts[0], children)
    right_arm = _chain_from(arm_starts[1], children)
    left_leg = _chain_from(leg_starts[0], children)
    right_leg = _chain_from(leg_starts[1], children)
    left_hand = left_arm[-1]
    right_hand = right_arm[-1]

    cx = float(chest_pos[0])
    if not (float(by_id[left_hand].position[0]) < cx < float(by_id[right_hand].position[0])):
        raise QualificationError("HUMANOID_ROLE_BILATERAL_HAND_ORDER_AMBIGUOUS")

    provisional = HumanoidJointRoleBindingIR(
        root_joint_id=root,
        spine_chain=tuple(spine),
        chest_joint_id=chest,
        head_joint_id=head,
        left_arm_chain=left_arm,
        right_arm_chain=right_arm,
        left_leg_chain=left_leg,
        right_leg_chain=right_leg,
        left_hand_slot_joint_id=left_hand,
        right_hand_slot_joint_id=right_hand,
        binding_hash="",
        metadata={
            "authority": "CANONICAL_TOPOLOGY_AND_REST_GEOMETRY_ONLY",
            "source_proposal_index_used": False,
            "source_joint_name_used": False,
            "skin_weight_role_inference_used": False,
            "filename_or_component_name_used": False,
            "bilateral_axis": "CANONICAL_REST_X",
        },
    )
    value = HumanoidJointRoleBindingIR(
        **{**provisional.to_dict(), "binding_hash": humanoid_joint_role_binding_hash(provisional)}
    )
    if value.binding_hash != humanoid_joint_role_binding_hash(value):
        raise QualificationError("HUMANOID_ROLE_BINDING_HASH_MISMATCH")
    return value


_SEMANTIC_ALIASES = {
    "HEAD": "HEAD",
    "CHEST": "CHEST",
    "HAND_SLOT_LEFT": "HAND_SLOT_LEFT",
    "HAND_SLOT_RIGHT": "HAND_SLOT_RIGHT",
    "handslot.l": "HAND_SLOT_LEFT",
    "handslot.r": "HAND_SLOT_RIGHT",
}


def canonical_joint_for_attachment_semantic(
    binding: HumanoidJointRoleBindingIR,
    semantic: str,
) -> str:
    normalized = _SEMANTIC_ALIASES.get(str(semantic))
    if normalized is None:
        raise QualificationError(f"HUMANOID_ROLE_UNSUPPORTED_ATTACHMENT_SEMANTIC:{semantic}")
    if normalized == "HEAD":
        return binding.head_joint_id
    if normalized == "CHEST":
        return binding.chest_joint_id
    if normalized == "HAND_SLOT_LEFT":
        return binding.left_hand_slot_joint_id
    if normalized == "HAND_SLOT_RIGHT":
        return binding.right_hand_slot_joint_id
    raise AssertionError(normalized)


__all__ = [
    "HumanoidJointRoleBindingIR",
    "humanoid_joint_role_binding_hash",
    "infer_humanoid_joint_roles",
    "canonical_joint_for_attachment_semantic",
]
