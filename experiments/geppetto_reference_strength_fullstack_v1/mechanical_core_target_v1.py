from __future__ import annotations

"""Teacher-only mechanical-core target construction for Geppetto FIT.

This module is experiment-only and may read teacher skeleton + skin data during
training/evaluation target construction. None of these values are learner input.

The projection is deliberately anonymous:
- no family names;
- no bone/control names;
- no fixed Mage source-index list;
- no fixed product control count.

Position authority is explicitly WORLD/SOURCE frame. Corpus arrays named merely
``bone_heads`` may be canonical-normalized and therefore MUST NOT be passed as
``bone_heads_world`` without an explicit transform. For the current normalized
corpus, ``rest_world_source[:, :3, 3]`` is the accepted source/world head locus.

Mechanical rule:
1. seed controls are deform controls with non-trivial skin mass;
2. an unsupported control is retained only when it is structurally required as
   a bridge between a selected descendant and a selected ancestor;
3. unsupported assembly-only root chains and helper/IK-only leaves are excluded;
4. projected parent is the nearest retained ancestor;
5. teacher rows are serialized parent-before-child using geometry/topology
   content rather than source row identity.
"""

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np


SCHEMA = "RealSaS.GeppettoMechanicalCoreTarget.v1"
RULE = "SKIN_SUPPORTED_PLUS_SUPPORTED_BRIDGES__ASSEMBLY_ONLY_ROOT_EXCLUDED"
POSITION_AUTHORITY = "WORLD_SOURCE_FRAME_EXPLICIT"


@dataclass(frozen=True)
class MechanicalCoreTargetV1:
    positions_world: np.ndarray
    parent_indices: np.ndarray
    root_mask: np.ndarray
    source_indices_provenance_only: np.ndarray
    skin_mass: np.ndarray
    rule: str = RULE
    position_authority: str = POSITION_AUTHORITY
    schema: str = SCHEMA

    @property
    def count(self) -> int:
        return int(len(self.positions_world))


def _validate(
    parents: np.ndarray,
    deform_mask: np.ndarray,
    skin: np.ndarray,
    bone_heads_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(parents, dtype=np.int64)
    d = np.asarray(deform_mask, dtype=bool)
    w = np.asarray(skin, dtype=np.float64)
    h = np.asarray(bone_heads_world, dtype=np.float64)
    j = int(len(p))
    if j < 1 or d.shape != (j,) or h.shape != (j, 3):
        raise ValueError("teacher skeleton shape mismatch")
    if w.ndim != 2 or w.shape[1] != j:
        raise ValueError("teacher skin shape mismatch")
    if not np.isfinite(h).all() or not np.isfinite(w).all():
        raise ValueError("teacher target contains non-finite values")
    if np.any(w < 0):
        raise ValueError("teacher skin contains negative weights")
    if np.any((p < -1) | (p >= j)) or np.any(p == np.arange(j)):
        raise ValueError("teacher parent array is invalid")
    for start in range(j):
        seen: set[int] = set()
        node = start
        while node >= 0:
            if node in seen:
                raise ValueError("teacher parent graph contains a cycle")
            seen.add(node)
            node = int(p[node])
    return p, d, w, h


def world_heads_from_rest_world_source_v1(rest_world_source: np.ndarray) -> np.ndarray:
    """Extract world/source joint-head loci from homogeneous rest transforms."""
    a = np.asarray(rest_world_source, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (4, 4) or len(a) < 1:
        raise ValueError("rest_world_source must be [J,4,4]")
    if not np.isfinite(a).all():
        raise ValueError("rest_world_source contains non-finite values")
    bottom = a[:, 3, :]
    expected = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(bottom, expected[None, :], atol=1e-6, rtol=0.0):
        raise ValueError("rest_world_source is not homogeneous rigid-transform form")
    return a[:, :3, 3].copy()


def _select_mechanical_core(
    parents: np.ndarray,
    deform_mask: np.ndarray,
    skin_mass: np.ndarray,
    *,
    support_epsilon: float,
) -> np.ndarray:
    if support_epsilon < 0:
        raise ValueError("support_epsilon must be non-negative")
    seed = set(np.flatnonzero(deform_mask & (skin_mass > support_epsilon)).tolist())
    if not seed:
        raise ValueError("no skin-supported deform controls")
    selected = set(seed)
    for child in tuple(sorted(seed)):
        chain: list[int] = []
        parent = int(parents[child])
        while parent >= 0:
            if parent in seed:
                selected.update(chain)
                break
            chain.append(parent)
            parent = int(parents[parent])
    return np.asarray(sorted(selected), dtype=np.int64)


def _project_parents(parents: np.ndarray, selected: np.ndarray) -> np.ndarray:
    row = {int(old): i for i, old in enumerate(selected.tolist())}
    out = np.empty(len(selected), dtype=np.int64)
    for i, old in enumerate(selected.tolist()):
        parent = int(parents[old])
        while parent >= 0 and parent not in row:
            parent = int(parents[parent])
        out[i] = -1 if parent < 0 else int(row[parent])
    return out


def _position_key(x: np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("invalid target position")
    return tuple(0.0 if float(v) == 0.0 else float(v) for v in a)


def _content_serialization(positions: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Parent-before-child content serialization without source-row authority."""
    n = int(len(positions))
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for i, parent in enumerate(parents.tolist()):
        if parent < 0:
            roots.append(i)
        else:
            children[int(parent)].append(i)
    state = np.zeros(n, dtype=np.int8)
    signatures: list[tuple | None] = [None] * n

    def signature(node: int) -> tuple:
        if state[node] == 1:
            raise ValueError("projected target contains cycle")
        if state[node] == 2:
            result = signatures[node]
            assert result is not None
            return result
        state[node] = 1
        child_sigs = tuple(sorted(signature(c) for c in children[node]))
        result = (_position_key(positions[node]), child_sigs)
        signatures[node] = result
        state[node] = 2
        return result

    for root in roots:
        signature(root)
    if np.any(state != 2):
        raise ValueError("projected target contains unreachable nodes")

    order: list[int] = []

    def emit(node: int) -> None:
        order.append(node)
        groups: dict[tuple, list[int]] = {}
        for child in children[node]:
            sig = signatures[child]
            assert sig is not None
            groups.setdefault(sig, []).append(child)
        for sig in sorted(groups):
            for child in groups[sig]:
                emit(child)

    root_groups: dict[tuple, list[int]] = {}
    for root in roots:
        sig = signatures[root]
        assert sig is not None
        root_groups.setdefault(sig, []).append(root)
    for sig in sorted(root_groups):
        for root in root_groups[sig]:
            emit(root)

    permutation = np.asarray(order, dtype=np.int64)
    if permutation.shape != (n,) or set(permutation.tolist()) != set(range(n)):
        raise RuntimeError("target serialization is not an exact permutation")
    return permutation


def build_mechanical_core_target_v1(
    *,
    parents: np.ndarray,
    deform_mask: np.ndarray,
    skin: np.ndarray,
    bone_heads_world: np.ndarray,
    support_epsilon: float = 1e-8,
) -> MechanicalCoreTargetV1:
    parents, deform_mask, skin, bone_heads_world = _validate(
        parents, deform_mask, skin, bone_heads_world
    )
    mass = skin.sum(axis=0)
    selected = _select_mechanical_core(
        parents, deform_mask, mass, support_epsilon=support_epsilon
    )
    positions = bone_heads_world[selected].copy()
    projected_parents = _project_parents(parents, selected)
    permutation = _content_serialization(positions, projected_parents)
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation), dtype=np.int64)
    serialized_parents = np.asarray(
        [-1 if projected_parents[old] < 0 else inverse[int(projected_parents[old])]
         for old in permutation.tolist()],
        dtype=np.int64,
    )
    if any(parent >= child for child, parent in enumerate(serialized_parents.tolist()) if parent >= 0):
        raise RuntimeError("target serialization violated parent-before-child invariant")
    return MechanicalCoreTargetV1(
        positions_world=positions[permutation].astype(np.float32),
        parent_indices=serialized_parents,
        root_mask=(serialized_parents < 0),
        source_indices_provenance_only=selected[permutation],
        skin_mass=mass[selected][permutation].astype(np.float64),
    )


def target_content_sha256_v1(target: MechanicalCoreTargetV1) -> str:
    payload = {
        "schema": target.schema,
        "rule": target.rule,
        "position_authority": target.position_authority,
        "positions_world": np.asarray(target.positions_world, np.float32).tolist(),
        "parents": np.asarray(target.parent_indices, np.int64).tolist(),
        "root_mask": np.asarray(target.root_mask, bool).astype(int).tolist(),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SCHEMA",
    "RULE",
    "POSITION_AUTHORITY",
    "MechanicalCoreTargetV1",
    "world_heads_from_rest_world_source_v1",
    "build_mechanical_core_target_v1",
    "target_content_sha256_v1",
]
