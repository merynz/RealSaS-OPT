from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Sequence

import numpy as np

try:
    from compiler.realsas_compiler_core.types import RiggingSurfaceIR
    from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
except ImportError:
    from realsas_compiler_core.types import RiggingSurfaceIR
    from realsas_compiler_core.v4_types import QualifiedSkeletonIRV2


def _hash(payload: object) -> str:
    def normalize(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (tuple, list)):
            return [normalize(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value
    return sha256(json.dumps(normalize(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MechanicalSkinTruthV1:
    surface_ids: tuple[str, ...]
    canonical_joint_ids: tuple[str, ...]
    weights: np.ndarray
    surface_binding_hash: str
    skeleton_binding_hash: str
    truth_lineage_hash: str
    schema_version: str = "RealSaS.MechanicalSkinTruth.v1"

    def validate(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64)
        if w.shape != (len(self.surface_ids), len(self.canonical_joint_ids)):
            raise ValueError("mechanical truth shape mismatch")
        if not np.isfinite(w).all() or (w < -1e-8).any():
            raise ValueError("mechanical truth contains invalid weights")
        if len(w) and np.max(np.abs(w.sum(axis=1) - 1.0)) > 1e-5:
            raise ValueError("mechanical truth simplex mismatch")
        if len(set(self.surface_ids)) != len(self.surface_ids):
            raise ValueError("duplicate surface ids")
        if len(set(self.canonical_joint_ids)) != len(self.canonical_joint_ids):
            raise ValueError("duplicate canonical joint ids")
        payload = {
            "surface_ids": self.surface_ids,
            "canonical_joint_ids": self.canonical_joint_ids,
            "weights": w,
            "surface_binding_hash": self.surface_binding_hash,
            "skeleton_binding_hash": self.skeleton_binding_hash,
            "schema_version": self.schema_version,
        }
        if _hash(payload) != self.truth_lineage_hash:
            raise ValueError("mechanical truth lineage hash mismatch")


def align_mechanical_skin_truth_v1(
    surface: RiggingSurfaceIR,
    skeleton: QualifiedSkeletonIRV2,
    *,
    teacher_surface_ids: Sequence[str],
    anonymous_control_ids: Sequence[str],
    dense_weights: np.ndarray,
) -> MechanicalSkinTruthV1:
    """Align training-only dense skin truth to current S and Compiler-qualified G.

    The only control bridge is QualifiedJoint.source_proposal_id -> anonymous teacher
    control ID. Source bone names/indices/tails are neither accepted nor emitted.
    """
    if not surface.geometry_lineage_hash:
        raise ValueError("surface geometry lineage hash required")
    if not skeleton.skeleton_lineage_hash:
        raise ValueError("qualified skeleton lineage hash required")
    s_teacher = tuple(map(str, teacher_surface_ids))
    c_teacher = tuple(map(str, anonymous_control_ids))
    if len(set(s_teacher)) != len(s_teacher) or len(set(c_teacher)) != len(c_teacher):
        raise ValueError("teacher axes contain duplicate ids")
    w = np.asarray(dense_weights, dtype=np.float64)
    if w.shape != (len(s_teacher), len(c_teacher)):
        raise ValueError("dense teacher weight shape mismatch")
    if not np.isfinite(w).all() or (w < -1e-8).any():
        raise ValueError("dense teacher weights invalid")
    if len(w) and np.max(np.abs(w.sum(axis=1) - 1.0)) > 1e-5:
        raise ValueError("dense teacher rows must sum to one")

    current_surface_ids = tuple(sorted(node.surface_id for node in surface.surface_nodes))
    if len(current_surface_ids) != len(surface.surface_nodes):
        raise ValueError("surface contains duplicate surface ids")
    s_index = {surface_id: i for i, surface_id in enumerate(s_teacher)}
    missing_surface = [surface_id for surface_id in current_surface_ids if surface_id not in s_index]
    if missing_surface:
        raise ValueError(f"teacher truth does not cover current surface:{missing_surface[:8]}")

    teacher_control_index = {control_id: i for i, control_id in enumerate(c_teacher)}
    ordered_joints = tuple(sorted(skeleton.joints, key=lambda joint: joint.canonical_joint_id))
    if not ordered_joints:
        raise ValueError("qualified skeleton contains no joints")
    proposal_ids = [joint.source_proposal_id for joint in ordered_joints]
    if any(not proposal_id for proposal_id in proposal_ids):
        raise ValueError("qualified joint missing anonymous source_proposal_id")
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ValueError("qualified skeleton repeats source_proposal_id")
    missing_controls = [proposal_id for proposal_id in proposal_ids if proposal_id not in teacher_control_index]
    if missing_controls:
        raise ValueError(f"teacher truth does not cover qualified anonymous controls:{missing_controls[:8]}")

    surface_axis = [s_index[surface_id] for surface_id in current_surface_ids]
    control_axis = [teacher_control_index[proposal_id] for proposal_id in proposal_ids]
    aligned = w[np.ix_(surface_axis, control_axis)].astype(np.float32)
    row_sum = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 1e-12):
        raise ValueError("qualified control subset removes all influence from a surface row")
    # Conditioning on a qualified deform subset is a training-only rebind. Re-normalize
    # after exact subset selection and record the resulting current-G field, never the
    # omitted teacher controls.
    aligned = aligned / row_sum
    canonical_joint_ids = tuple(joint.canonical_joint_id for joint in ordered_joints)
    payload = {
        "surface_ids": current_surface_ids,
        "canonical_joint_ids": canonical_joint_ids,
        "weights": aligned,
        "surface_binding_hash": surface.geometry_lineage_hash,
        "skeleton_binding_hash": skeleton.skeleton_lineage_hash,
        "schema_version": "RealSaS.MechanicalSkinTruth.v1",
    }
    result = MechanicalSkinTruthV1(
        surface_ids=current_surface_ids,
        canonical_joint_ids=canonical_joint_ids,
        weights=aligned,
        surface_binding_hash=surface.geometry_lineage_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        truth_lineage_hash=_hash(payload),
    )
    result.validate()
    return result
