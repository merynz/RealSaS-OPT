from __future__ import annotations

"""Anonymous Geppetto target-policy challengers for articulation-capacity courts.

This module is teacher-only training/evaluation machinery. It never runs at
product inference and never mints skeleton authority.

K0 preserves the existing mechanical-core target exactly.
K1 admits every teacher deform control plus structural bridges between admitted
controls.
K2 additionally admits controls independently proven deformation-necessary by a
separate court.
K3 is a full legal source-skeleton diagnostic ceiling.

None of the policies target a requested joint count.
"""

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from .mechanical_core_target_v1 import (
    MechanicalCoreTargetV1,
    _content_serialization,
    _project_parents,
    _validate,
    build_mechanical_core_target_v1,
)


SCHEMA = "RealSaS.GeppettoMechanicalCapacityTarget.v2"
POLICY_K0 = "K0_CURRENT_CORE"
POLICY_K1 = "K1_ALL_DEFORM"
POLICY_K2 = "K2_DEFORMATION_NECESSARY"
POLICY_K3 = "K3_FULL_LEGAL_DIAGNOSTIC"
POLICIES = (POLICY_K0, POLICY_K1, POLICY_K2, POLICY_K3)


@dataclass(frozen=True)
class MechanicalCapacityTargetV2:
    positions_world: np.ndarray
    parent_indices: np.ndarray
    root_mask: np.ndarray
    source_indices_provenance_only: np.ndarray
    skin_mass: np.ndarray
    source_deform_mask: np.ndarray
    policy: str
    policy_receipt_hash: str
    schema: str = SCHEMA

    @property
    def count(self) -> int:
        return int(len(self.positions_world))


def _select_seed_bridges(parents: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    seed = set(np.flatnonzero(np.asarray(seed_mask, dtype=bool)).tolist())
    if not seed:
        raise ValueError("capacity target has no seed controls")
    selected = set(seed)
    for child in tuple(sorted(seed)):
        chain = []
        parent = int(parents[child])
        while parent >= 0:
            if parent in seed:
                selected.update(chain)
                break
            chain.append(parent)
            parent = int(parents[parent])
    return np.asarray(sorted(selected), dtype=np.int64)


def _from_selected(
    *,
    parents: np.ndarray,
    deform_mask: np.ndarray,
    skin: np.ndarray,
    bone_heads_world: np.ndarray,
    selected: np.ndarray,
    policy: str,
) -> MechanicalCapacityTargetV2:
    positions = np.asarray(bone_heads_world, dtype=np.float64)[selected].copy()
    projected_parents = _project_parents(parents, selected)
    permutation = _content_serialization(positions, projected_parents)
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation), dtype=np.int64)
    serialized_parents = np.asarray(
        [
            -1
            if projected_parents[old] < 0
            else inverse[int(projected_parents[old])]
            for old in permutation.tolist()
        ],
        dtype=np.int64,
    )
    if any(
        parent >= child
        for child, parent in enumerate(serialized_parents.tolist())
        if parent >= 0
    ):
        raise RuntimeError("capacity target serialization violated parent-before-child")

    source = selected[permutation]
    mass = np.asarray(skin, dtype=np.float64).sum(axis=0)[source]
    source_deform = np.asarray(deform_mask, dtype=bool)[source]
    payload = {
        "schema": SCHEMA,
        "policy": policy,
        "positions_world": np.asarray(positions[permutation], np.float32).tolist(),
        "parents": serialized_parents.tolist(),
        "root_mask": (serialized_parents < 0).astype(int).tolist(),
        # Source row indices are intentionally excluded from the semantic receipt.
        "skin_mass": np.asarray(mass, np.float64).tolist(),
        "source_deform_mask": source_deform.astype(int).tolist(),
    }
    receipt = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return MechanicalCapacityTargetV2(
        positions_world=np.asarray(positions[permutation], dtype=np.float32),
        parent_indices=serialized_parents,
        root_mask=(serialized_parents < 0),
        source_indices_provenance_only=np.asarray(source, dtype=np.int64),
        skin_mass=np.asarray(mass, dtype=np.float64),
        source_deform_mask=np.asarray(source_deform, dtype=bool),
        policy=policy,
        policy_receipt_hash=receipt,
    )


def build_mechanical_capacity_target_v2(
    *,
    parents: np.ndarray,
    deform_mask: np.ndarray,
    skin: np.ndarray,
    bone_heads_world: np.ndarray,
    policy: str,
    necessary_control_mask: np.ndarray | None = None,
    support_epsilon: float = 1e-8,
) -> MechanicalCapacityTargetV2:
    parents, deform_mask, skin, bone_heads_world = _validate(
        parents, deform_mask, skin, bone_heads_world
    )
    if policy not in POLICIES:
        raise ValueError(f"unknown capacity target policy:{policy}")
    if support_epsilon < 0:
        raise ValueError("support_epsilon must be non-negative")

    if policy == POLICY_K0:
        base: MechanicalCoreTargetV1 = build_mechanical_core_target_v1(
            parents=parents,
            deform_mask=deform_mask,
            skin=skin,
            bone_heads_world=bone_heads_world,
            support_epsilon=support_epsilon,
        )
        return _from_selected(
            parents=parents,
            deform_mask=deform_mask,
            skin=skin,
            bone_heads_world=bone_heads_world,
            selected=np.asarray(base.source_indices_provenance_only, dtype=np.int64),
            policy=policy,
        )

    if policy == POLICY_K1:
        selected = _select_seed_bridges(parents, deform_mask)
    elif policy == POLICY_K2:
        if necessary_control_mask is None:
            raise ValueError("K2 requires independently measured necessary_control_mask")
        necessary = np.asarray(necessary_control_mask, dtype=bool)
        if necessary.shape != deform_mask.shape:
            raise ValueError("necessary_control_mask shape drift")
        # K2 is deliberately an intermediate target between K0 and K1:
        # start from the K0 skin-supported deform seeds, then add only controls
        # independently proven deformation-necessary, plus structural bridges
        # required to connect those admitted seeds. Unioning with deform_mask
        # would collapse K2 to K1 whenever necessity is measured on deform
        # controls, defeating the purpose of the court.
        skin_mass = np.asarray(skin, dtype=np.float64).sum(axis=0)
        k0_seed = deform_mask & (skin_mass > support_epsilon)
        selected = _select_seed_bridges(parents, k0_seed | necessary)
    else:
        # K3 is intentionally only a diagnostic ceiling.
        selected = np.arange(len(parents), dtype=np.int64)

    return _from_selected(
        parents=parents,
        deform_mask=deform_mask,
        skin=skin,
        bone_heads_world=bone_heads_world,
        selected=selected,
        policy=policy,
    )


__all__ = [
    "MechanicalCapacityTargetV2",
    "POLICY_K0",
    "POLICY_K1",
    "POLICY_K2",
    "POLICY_K3",
    "POLICIES",
    "build_mechanical_capacity_target_v2",
]
