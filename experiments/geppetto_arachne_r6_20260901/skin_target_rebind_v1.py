from __future__ import annotations

import numpy as np

from experiments.single_family_e2e_v1.mechanical_truth_adapter_v1 import MechanicalSkinTruthV1
from .training_targets_v1 import SkinFieldTeacherTargetV1


def rebind_skin_target_v1(
    truth: MechanicalSkinTruthV1,
    *,
    surface_ids: tuple[str, ...],
    canonical_joint_ids: tuple[str, ...],
    surface_binding_hash: str,
    skeleton_binding_hash: str,
) -> SkinFieldTeacherTargetV1:
    """Reorder current mechanical truth onto an Arachne/codec conditioning axis.

    The target must already be bound to the exact current S and qualified G lineage.
    No source-bone identity is accepted here.
    """
    truth.validate()
    if truth.surface_binding_hash != surface_binding_hash:
        raise ValueError("skin target surface lineage mismatch")
    if truth.skeleton_binding_hash != skeleton_binding_hash:
        raise ValueError("skin target skeleton lineage mismatch")
    if len(set(surface_ids)) != len(surface_ids) or len(set(canonical_joint_ids)) != len(canonical_joint_ids):
        raise ValueError("conditioning axes contain duplicate ids")
    s_index = {sid: i for i, sid in enumerate(truth.surface_ids)}
    j_index = {jid: i for i, jid in enumerate(truth.canonical_joint_ids)}
    missing_s = [sid for sid in surface_ids if sid not in s_index]
    missing_j = [jid for jid in canonical_joint_ids if jid not in j_index]
    if missing_s or missing_j:
        raise ValueError(f"skin target coverage mismatch:surface={missing_s[:8]} joint={missing_j[:8]}")
    aligned = np.asarray(truth.weights, dtype=np.float32)[np.ix_(
        [s_index[sid] for sid in surface_ids],
        [j_index[jid] for jid in canonical_joint_ids],
    )]
    if not np.isfinite(aligned).all() or (aligned < -1e-7).any():
        raise ValueError("aligned skin target invalid")
    if len(aligned) and np.max(np.abs(aligned.sum(axis=1) - 1.0)) > 1e-5:
        raise ValueError("aligned skin target simplex mismatch")
    target = SkinFieldTeacherTargetV1(tuple(surface_ids), tuple(canonical_joint_ids), aligned)
    target.validate()
    return target
