from __future__ import annotations

import numpy as np

from experiments.geppetto_reference_strength_fullstack_v1.mechanical_core_target_v1 import (
    RULE,
    build_mechanical_core_target_v1,
    target_content_sha256_v1,
    world_heads_from_rest_world_source_v1,
)


def _skin(rows: int, cols: int, supported: tuple[int, ...]) -> np.ndarray:
    w = np.zeros((rows, cols), dtype=np.float32)
    for r, c in enumerate(supported):
        w[r % rows, c] = 1.0
    return w


def test_assembly_root_and_helper_leaves_are_not_promoted() -> None:
    parents = np.asarray([-1, 0, 1, 2, 0, 0], np.int64)
    deform = np.ones(6, dtype=bool)
    skin = _skin(3, 6, (1, 2, 3))
    heads = np.asarray([[0, 0, i] for i in range(6)], dtype=np.float32)
    target = build_mechanical_core_target_v1(
        parents=parents, deform_mask=deform, skin=skin, bone_heads_world=heads
    )
    assert target.count == 3
    assert target.rule == RULE
    assert set(target.source_indices_provenance_only.tolist()) == {1, 2, 3}
    assert int(target.root_mask.sum()) == 1
    assert target.parent_indices.tolist() == [-1, 0, 1]


def test_unsupported_required_bridge_is_retained() -> None:
    parents = np.asarray([-1, 0, 1, 2], np.int64)
    deform = np.ones(4, dtype=bool)
    skin = _skin(2, 4, (1, 3))
    heads = np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=np.float32)
    target = build_mechanical_core_target_v1(
        parents=parents, deform_mask=deform, skin=skin, bone_heads_world=heads
    )
    assert set(target.source_indices_provenance_only.tolist()) == {1, 2, 3}
    assert target.parent_indices.tolist() == [-1, 0, 1]


def test_unconnected_unsupported_chain_is_not_promoted() -> None:
    parents = np.asarray([-1, 0, 1], np.int64)
    deform = np.ones(3, dtype=bool)
    skin = _skin(1, 3, (2,))
    heads = np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=np.float32)
    target = build_mechanical_core_target_v1(
        parents=parents, deform_mask=deform, skin=skin, bone_heads_world=heads
    )
    assert target.count == 1
    assert target.source_indices_provenance_only.tolist() == [2]
    assert target.parent_indices.tolist() == [-1]


def test_target_content_hash_does_not_depend_on_source_indices() -> None:
    parents = np.asarray([-1, 0, 1], np.int64)
    deform = np.ones(3, dtype=bool)
    skin = _skin(2, 3, (1, 2))
    heads = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    a = build_mechanical_core_target_v1(
        parents=parents, deform_mask=deform, skin=skin, bone_heads_world=heads
    )
    h = target_content_sha256_v1(a)
    b = type(a)(
        positions_world=a.positions_world.copy(),
        parent_indices=a.parent_indices.copy(),
        root_mask=a.root_mask.copy(),
        source_indices_provenance_only=a.source_indices_provenance_only[::-1].copy(),
        skin_mass=a.skin_mass.copy(),
    )
    assert target_content_sha256_v1(b) == h


def test_rest_world_source_extracts_translation_authority() -> None:
    rest = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    rest[0, :3, 3] = [1.0, 2.0, 3.0]
    rest[1, :3, 3] = [-1.0, 0.5, 4.0]
    got = world_heads_from_rest_world_source_v1(rest)
    assert np.array_equal(got, np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]]))


def test_invalid_cycle_fails_closed() -> None:
    parents = np.asarray([1, 0], np.int64)
    deform = np.ones(2, dtype=bool)
    skin = _skin(1, 2, (0,))
    heads = np.asarray([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
    try:
        build_mechanical_core_target_v1(
            parents=parents, deform_mask=deform, skin=skin, bone_heads_world=heads
        )
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cycle must fail closed")
