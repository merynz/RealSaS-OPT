from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .conditioning_v1 import GeppettoConditioningBatchV1, ArachneConditioningBatchV1


@dataclass(frozen=True)
class GeppettoTeacherTargetV1:
    positions_normalized: np.ndarray
    parent_indices: np.ndarray
    root_mask: np.ndarray
    valid: bool = True
    schema_version: str = "RealSaS.GeppettoTeacherTarget.v1"


def project_geppetto_teacher_target(teacher_projection, conditioning: GeppettoConditioningBatchV1, batch_index: int) -> GeppettoTeacherTargetV1:
    """Teacher-only target projection. Never emits a product/proposal object."""
    teacher_projection.validate()
    controls = tuple(teacher_projection.controls)
    if not controls:
        raise ValueError("teacher target has no controls")
    id_to_index = {c.control_id: i for i, c in enumerate(controls)}
    pos = np.asarray([c.position for c in controls], np.float32)
    pn = conditioning.normalizations[batch_index].normalize(pos)
    parent = np.asarray([-1 if c.parent_control_id is None else id_to_index[c.parent_control_id] for c in controls], np.int64)
    roots = np.asarray([c.parent_control_id is None for c in controls], bool)
    return GeppettoTeacherTargetV1(pn, parent, roots, True)


@dataclass(frozen=True)
class SkinFieldTeacherTargetV1:
    surface_ids: tuple[str, ...]
    joint_ids: tuple[str, ...]
    weights: np.ndarray
    schema_version: str = "RealSaS.SkinFieldTeacherTarget.v1"

    def validate(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64)
        if w.shape != (len(self.surface_ids), len(self.joint_ids)):
            raise ValueError("skin teacher shape mismatch")
        if not np.isfinite(w).all() or (w < -1e-8).any():
            raise ValueError("skin teacher contains invalid weights")
        if np.max(np.abs(w.sum(axis=1) - 1.0)) > 1e-5:
            raise ValueError("skin teacher rows must sum to one")
        if len(set(self.surface_ids)) != len(self.surface_ids) or len(set(self.joint_ids)) != len(self.joint_ids):
            raise ValueError("duplicate skin teacher ids")


def align_skin_teacher_to_conditioning(target: SkinFieldTeacherTargetV1, conditioning: ArachneConditioningBatchV1, batch_index: int) -> np.ndarray:
    target.validate()
    s_map = {sid: i for i, sid in enumerate(target.surface_ids)}
    j_map = {jid: i for i, jid in enumerate(target.joint_ids)}
    desired_s = conditioning.surface_ids[batch_index]
    desired_j = conditioning.joint_ids[batch_index]
    if not set(desired_s).issubset(s_map) or not set(desired_j).issubset(j_map):
        raise ValueError("teacher field does not cover conditioning ids")
    w = np.asarray(target.weights, np.float32)
    return w[np.ix_([s_map[s] for s in desired_s], [j_map[j] for j in desired_j])]
