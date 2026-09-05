from __future__ import annotations

import numpy as np

from models.geppetto.v2.training_targets_v1 import GeppettoTeacherTargetV1
from experiments.mage_scene_first_e2e_v1.geppetto_structural_serialization_v1 import (
    canonical_structural_serialization_v1,
)


def _target(positions, parents):
    parents = np.asarray(parents, np.int64)
    return GeppettoTeacherTargetV1(
        positions_normalized=np.asarray(positions, np.float32),
        parent_indices=parents,
        root_mask=parents < 0,
        valid=True,
    )


def _permute(target, permutation):
    permutation = np.asarray(permutation, np.int64)
    inverse = np.empty(len(permutation), np.int64)
    inverse[permutation] = np.arange(len(permutation), dtype=np.int64)
    old_parent = np.asarray(target.parent_indices, np.int64)
    new_parent = np.asarray(
        [-1 if old_parent[old] < 0 else inverse[old_parent[old]] for old in permutation],
        np.int64,
    )
    return GeppettoTeacherTargetV1(
        positions_normalized=np.asarray(target.positions_normalized, np.float32)[permutation],
        parent_indices=new_parent,
        root_mask=new_parent < 0,
        valid=True,
    )


def _content(result):
    target = result.target
    return (
        np.asarray(target.positions_normalized, np.float32),
        np.asarray(target.parent_indices, np.int64),
        np.asarray(target.root_mask, bool),
    )


def test_parent_before_child_and_same_xyz_topology_secondary_key():
    # Two root children share exact XYZ. One is a leaf, the other owns a child.
    # Geometry cannot distinguish them; the recursive topology signature must.
    target = _target(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        [-1, 0, 0, 2],
    )
    result = canonical_structural_serialization_v1(target)
    _, parents, _ = _content(result)
    assert parents.tolist() == [-1, 0, 0, 2]
    assert all(parent < child for child, parent in enumerate(parents.tolist()) if parent >= 0)


def test_exact_automorphic_duplicate_is_multiplicity_not_row_identity():
    target = _target(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        [-1, 0, 0],
    )
    a = canonical_structural_serialization_v1(target)
    b = canonical_structural_serialization_v1(_permute(target, [0, 2, 1]))
    for xa, xb in zip(_content(a), _content(b)):
        np.testing.assert_array_equal(xa, xb)
    assert a.automorphic_sibling_class_sizes == (2,)
    assert b.automorphic_sibling_class_sizes == (2,)


def test_serialized_content_is_invariant_under_many_source_row_permutations():
    # Mixed depth, coincident roles, and an exact automorphic leaf pair.
    target = _target(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ],
        [-1, 0, 0, 2, 0, 4, 4],
    )
    reference = _content(canonical_structural_serialization_v1(target))
    rng = np.random.default_rng(20260905)
    for _ in range(256):
        perm = rng.permutation(len(target.positions_normalized))
        candidate = _content(canonical_structural_serialization_v1(_permute(target, perm)))
        for expected, actual in zip(reference, candidate):
            np.testing.assert_array_equal(expected, actual)


def test_invalid_cycle_fails_closed():
    target = _target([[0, 0, 0], [1, 0, 0]], [1, 0])
    try:
        canonical_structural_serialization_v1(target)
    except ValueError:
        pass
    else:
        raise AssertionError("cycle must fail closed")
