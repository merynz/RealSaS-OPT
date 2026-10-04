from __future__ import annotations

"""Subject-agnostic control-basis pruning primitives.

These operators do not decide *whether* a control should be removed. That decision
belongs to proof-guided control-basis selection. They only materialize one legal
single-control reduction deterministically once a court has selected a candidate.
"""

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import (
    QualificationError,
    QualifiedJoint,
    QualifiedSkeletonIR,
)


@dataclass(frozen=True)
class ControlPruneReceiptV1:
    source_skeleton_hash: str
    pruned_skeleton_hash: str
    removed_control_id: str
    replacement_parent_id: str
    reparented_child_ids: tuple[str, ...]
    source_control_count: int
    pruned_control_count: int
    semantic_version: str = "RealSaS.ControlPruneReceipt.v1"

    def to_dict(self) -> dict:
        return {
            "source_skeleton_hash": self.source_skeleton_hash,
            "pruned_skeleton_hash": self.pruned_skeleton_hash,
            "removed_control_id": self.removed_control_id,
            "replacement_parent_id": self.replacement_parent_id,
            "reparented_child_ids": list(self.reparented_child_ids),
            "source_control_count": self.source_control_count,
            "pruned_control_count": self.pruned_control_count,
            "semantic_version": self.semantic_version,
        }


def prune_qualified_skeleton_control_v1(
    skeleton: QualifiedSkeletonIR,
    control_id: str,
) -> tuple[QualifiedSkeletonIR, ControlPruneReceiptV1]:
    """Remove one non-root control and reparent its children to its parent.

    Rest positions remain unchanged in object space. This preserves the rest pose
    while removing one independent control DOF from the hierarchy.
    """
    cid = str(control_id)
    by = {str(j.canonical_joint_id): j for j in skeleton.joints}
    if len(by) != len(skeleton.joints):
        raise QualificationError("CONTROL_PRUNE_DUPLICATE_JOINT_ID")
    if cid not in by:
        raise QualificationError("CONTROL_PRUNE_CONTROL_UNKNOWN")
    if cid == str(skeleton.root_id):
        raise QualificationError("CONTROL_PRUNE_ROOT_FORBIDDEN")

    removed = by[cid]
    parent_id = removed.parent_canonical_id
    if parent_id is None or str(parent_id) not in by:
        raise QualificationError("CONTROL_PRUNE_PARENT_INVALID")
    parent_id = str(parent_id)

    child_ids = tuple(
        sorted(
            str(j.canonical_joint_id)
            for j in skeleton.joints
            if j.parent_canonical_id is not None
            and str(j.parent_canonical_id) == cid
        )
    )

    joints = []
    for joint in skeleton.joints:
        jid = str(joint.canonical_joint_id)
        if jid == cid:
            continue
        if joint.parent_canonical_id is not None and str(joint.parent_canonical_id) == cid:
            joint = replace(joint, parent_canonical_id=parent_id)
        joints.append(joint)

    joint_ids = {str(j.canonical_joint_id) for j in joints}
    if str(skeleton.root_id) not in joint_ids:
        raise QualificationError("CONTROL_PRUNE_ROOT_LOST")
    for joint in joints:
        if joint.parent_canonical_id is not None and str(joint.parent_canonical_id) not in joint_ids:
            raise QualificationError("CONTROL_PRUNE_DANGLING_PARENT")

    report = {
        **dict(skeleton.qualification_report or {}),
        "control_basis_operation": "PRUNE_SINGLE_REDUNDANT_CONTROL_V1",
        "source_skeleton_hash": str(skeleton.skeleton_lineage_hash),
        "removed_control_id": cid,
        "replacement_parent_id": parent_id,
        "reparented_child_ids": child_ids,
        "source_control_count": int(len(skeleton.joints)),
        "pruned_control_count": int(len(joints)),
        "product_authority_minted": False,
    }
    lineage = content_sha256(
        {
            "schema": "RealSaS.QualifiedSkeletonIR.v1",
            "operation": "PRUNE_SINGLE_REDUNDANT_CONTROL_V1",
            "source_skeleton_hash": skeleton.skeleton_lineage_hash,
            "root_id": skeleton.root_id,
            "joints": [j.to_dict() for j in joints],
            "removed_control_id": cid,
            "replacement_parent_id": parent_id,
        }
    )
    value = QualifiedSkeletonIR(
        joints=tuple(joints),
        root_id=str(skeleton.root_id),
        qualification_report=report,
        skeleton_lineage_hash=lineage,
    )
    receipt = ControlPruneReceiptV1(
        source_skeleton_hash=str(skeleton.skeleton_lineage_hash),
        pruned_skeleton_hash=lineage,
        removed_control_id=cid,
        replacement_parent_id=parent_id,
        reparented_child_ids=child_ids,
        source_control_count=len(skeleton.joints),
        pruned_control_count=len(joints),
    )
    return value, receipt


def collapse_weight_axis_to_parent_v1(
    weights,
    joint_ids: Sequence[str],
    *,
    removed_control_id: str,
    replacement_parent_id: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Remove one joint column while conserving its full weight mass at the parent."""
    W = np.asarray(weights, dtype=np.float64)
    ids = tuple(map(str, joint_ids))
    if W.ndim != 2 or W.shape[1] != len(ids) or W.shape[0] < 1:
        raise QualificationError("CONTROL_PRUNE_WEIGHT_AXIS_INVALID")
    if len(set(ids)) != len(ids):
        raise QualificationError("CONTROL_PRUNE_WEIGHT_JOINT_DUPLICATE")
    removed = str(removed_control_id)
    parent = str(replacement_parent_id)
    if removed not in ids or parent not in ids or removed == parent:
        raise QualificationError("CONTROL_PRUNE_WEIGHT_BINDING_INVALID")
    if not np.isfinite(W).all() or np.any(W < -1e-10):
        raise QualificationError("CONTROL_PRUNE_WEIGHT_NONFINITE_OR_NEGATIVE")
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise QualificationError("CONTROL_PRUNE_WEIGHT_SIMPLEX_INVALID")

    ri = ids.index(removed)
    pi = ids.index(parent)
    out = W.copy()
    out[:, pi] += out[:, ri]
    keep = [i for i in range(len(ids)) if i != ri]
    out = out[:, keep]
    out_ids = tuple(ids[i] for i in keep)

    if not np.isfinite(out).all() or np.any(out < -1e-10):
        raise QualificationError("CONTROL_PRUNE_WEIGHT_OUTPUT_INVALID")
    if not np.allclose(out.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise QualificationError("CONTROL_PRUNE_WEIGHT_MASS_DRIFT")
    return out, out_ids


__all__ = [
    "ControlPruneReceiptV1",
    "prune_qualified_skeleton_control_v1",
    "collapse_weight_axis_to_parent_v1",
]
