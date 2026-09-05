from __future__ import annotations

"""Teacher-only deterministic structural serialization for Geppetto diagnostics.

This module is deliberately experimental. It canonicalizes an anonymous
``GeppettoTeacherTargetV1`` into a parent-before-child preorder using only
teacher geometry + topology. Teacher control IDs and original row indices are
never sorting authority.

The key property is *content invariance* under arbitrary teacher-row
permutations. Exact automorphic duplicates are allowed: when two subtrees are
indistinguishable under the available geometry/topology evidence, the
serialized content contains repeated identical entries rather than inventing a
row identity.
"""

from dataclasses import dataclass

import numpy as np

from models.geppetto.v2.training_targets_v1 import GeppettoTeacherTargetV1


SCHEMA = "RealSaS.GeppettoTeacherStructuralSerialization.v1"


@dataclass(frozen=True)
class StructuralSerializationResultV1:
    target: GeppettoTeacherTargetV1
    source_permutation: np.ndarray
    automorphic_sibling_class_sizes: tuple[int, ...]
    schema_version: str = SCHEMA


def _position_key(position: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(position, dtype=np.float32)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("structural serialization requires finite 3D positions")
    # Collapse signed zero so -0.0 never becomes accidental ordering authority.
    return tuple(0.0 if float(x) == 0.0 else float(x) for x in values)


def _validate_target(target: GeppettoTeacherTargetV1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(target.positions_normalized, dtype=np.float32)
    parents = np.asarray(target.parent_indices, dtype=np.int64)
    roots = np.asarray(target.root_mask, dtype=bool)
    n = int(len(positions))
    if not target.valid or n < 1:
        raise ValueError("structural serialization requires a valid non-empty target")
    if positions.shape != (n, 3) or parents.shape != (n,) or roots.shape != (n,):
        raise ValueError("structural serialization target shape mismatch")
    if not np.isfinite(positions).all():
        raise ValueError("structural serialization positions must be finite")
    if np.any((parents < -1) | (parents >= n)):
        raise ValueError("structural serialization parent index out of range")
    if np.any(parents == np.arange(n, dtype=np.int64)):
        raise ValueError("structural serialization self-parent")
    if not np.array_equal(roots, parents < 0):
        raise ValueError("root_mask must exactly match parent_indices < 0")
    if not roots.any():
        raise ValueError("structural serialization requires at least one root")
    return positions, parents, roots


def canonical_structural_serialization_v1(
    target: GeppettoTeacherTargetV1,
) -> StructuralSerializationResultV1:
    """Canonicalize a teacher forest without using teacher row identity.

    Ordering:
      1. parent before descendant (preorder);
      2. roots/siblings ordered by normalized XYZ;
      3. exact-XYZ ties ordered by the complete recursively canonicalized
         descendant signature;
      4. exact automorphic ties remain interchangeable. Their serialized
         *content* is identical under row permutation, so no source-row
         identity is introduced merely to name indistinguishable copies.

    The output target is suitable for slot-wise teacher supervision during
    training diagnostics. This function is teacher-only and confers no
    proposal/product identity authority.
    """
    positions, parents, roots = _validate_target(target)
    n = int(len(positions))
    children: list[list[int]] = [[] for _ in range(n)]
    root_ids: list[int] = []
    for i, parent in enumerate(parents.tolist()):
        if parent < 0:
            root_ids.append(i)
        else:
            children[int(parent)].append(i)

    state = np.zeros(n, dtype=np.int8)
    signatures: list[tuple | None] = [None] * n

    def signature(node: int) -> tuple:
        if state[node] == 1:
            raise ValueError("structural serialization teacher topology contains a cycle")
        if state[node] == 2:
            result = signatures[node]
            assert result is not None
            return result
        state[node] = 1
        child_signatures = tuple(sorted(signature(child) for child in children[node]))
        result = (_position_key(positions[node]), child_signatures)
        signatures[node] = result
        state[node] = 2
        return result

    for root in root_ids:
        signature(root)
    if np.any(state != 2):
        raise ValueError("structural serialization contains nodes unreachable from roots")

    automorphic_sizes: list[int] = []

    def ordered_children(node: int) -> list[int]:
        groups: dict[tuple, list[int]] = {}
        for child in children[node]:
            child_signature = signatures[child]
            assert child_signature is not None
            groups.setdefault(child_signature, []).append(child)
        for members in groups.values():
            if len(members) > 1:
                automorphic_sizes.append(len(members))
        # The order inside an exact tie is intentionally irrelevant: equal
        # signatures imply equal serialized subtree content. Sorting the
        # signature groups provides all non-row-identity authority.
        ordered: list[int] = []
        for key in sorted(groups):
            ordered.extend(groups[key])
        return ordered

    root_groups: dict[tuple, list[int]] = {}
    for root in root_ids:
        root_signature = signatures[root]
        assert root_signature is not None
        root_groups.setdefault(root_signature, []).append(root)
    for members in root_groups.values():
        if len(members) > 1:
            automorphic_sizes.append(len(members))

    source_order: list[int] = []

    def emit(node: int) -> None:
        source_order.append(node)
        for child in ordered_children(node):
            emit(child)

    for key in sorted(root_groups):
        for root in root_groups[key]:
            emit(root)

    permutation = np.asarray(source_order, dtype=np.int64)
    if permutation.shape != (n,) or set(permutation.tolist()) != set(range(n)):
        raise RuntimeError("structural serialization did not emit an exact permutation")

    inverse = np.empty(n, dtype=np.int64)
    inverse[permutation] = np.arange(n, dtype=np.int64)
    serialized_parents = np.asarray(
        [-1 if parents[old] < 0 else inverse[int(parents[old])] for old in permutation],
        dtype=np.int64,
    )
    serialized_positions = positions[permutation].copy()
    serialized_roots = serialized_parents < 0

    # Fail closed on the defining parent-before-child invariant.
    for child, parent in enumerate(serialized_parents.tolist()):
        if parent >= child:
            raise RuntimeError("structural serialization violated parent-before-child order")

    serialized = GeppettoTeacherTargetV1(
        positions_normalized=serialized_positions,
        parent_indices=serialized_parents,
        root_mask=serialized_roots,
        valid=True,
    )
    return StructuralSerializationResultV1(
        target=serialized,
        source_permutation=permutation,
        automorphic_sibling_class_sizes=tuple(sorted(automorphic_sizes, reverse=True)),
    )


__all__ = [
    "SCHEMA",
    "StructuralSerializationResultV1",
    "canonical_structural_serialization_v1",
]
