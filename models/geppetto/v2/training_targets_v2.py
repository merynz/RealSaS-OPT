from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geppetto_conditioning_v2 import GeppettoConditioningBatchV2
from .training_targets_v1 import GeppettoTeacherTargetV1


def project_geppetto_teacher_target_v2(
    teacher_projection,
    conditioning: GeppettoConditioningBatchV2,
    batch_index: int,
) -> GeppettoTeacherTargetV1:
    """Project anonymous deform-only teacher controls into Geppetto V2 coordinates.

    This function is training/evaluation-only. Teacher control IDs are used only to
    recover parent indices inside the target; they never become proposal or product
    identity.
    """
    teacher_projection.validate()
    controls = tuple(teacher_projection.controls)
    if not controls:
        raise ValueError("teacher target has no controls")
    if batch_index < 0 or batch_index >= len(conditioning.surface_ids):
        raise IndexError("batch_index out of range")
    id_to_index = {c.control_id: i for i, c in enumerate(controls)}
    if len(id_to_index) != len(controls):
        raise ValueError("duplicate teacher control id")
    positions = np.asarray([c.position for c in controls], dtype=np.float32)
    positions_normalized = conditioning.normalizations[batch_index].normalize(positions)
    parent_indices = np.asarray(
        [-1 if c.parent_control_id is None else id_to_index[c.parent_control_id] for c in controls],
        dtype=np.int64,
    )
    root_mask = np.asarray([c.parent_control_id is None for c in controls], dtype=bool)
    if not root_mask.any():
        raise ValueError("teacher projection contains no deform root")
    return GeppettoTeacherTargetV1(
        positions_normalized=positions_normalized,
        parent_indices=parent_indices,
        root_mask=root_mask,
        valid=True,
    )


def permute_geppetto_teacher_target_v2(
    target: GeppettoTeacherTargetV1,
    permutation: np.ndarray,
) -> GeppettoTeacherTargetV1:
    """Re-serialize a target under a control permutation without changing topology."""
    permutation = np.asarray(permutation, dtype=np.int64)
    j = len(target.positions_normalized)
    if permutation.shape != (j,) or set(permutation.tolist()) != set(range(j)):
        raise ValueError("permutation must contain each target index exactly once")
    inverse = np.empty(j, dtype=np.int64)
    inverse[permutation] = np.arange(j, dtype=np.int64)
    old_parent = np.asarray(target.parent_indices, dtype=np.int64)
    new_parent = np.asarray(
        [-1 if old_parent[old_i] < 0 else inverse[old_parent[old_i]] for old_i in permutation],
        dtype=np.int64,
    )
    return GeppettoTeacherTargetV1(
        positions_normalized=np.asarray(target.positions_normalized, dtype=np.float32)[permutation],
        parent_indices=new_parent,
        root_mask=np.asarray(target.root_mask, dtype=bool)[permutation],
        valid=target.valid,
    )
