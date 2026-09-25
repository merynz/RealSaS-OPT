from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    mesh_connected_component_labels_v1,
)


def _reference_labels(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    parent = np.arange(int(vertex_count), dtype=np.int64)
    rank = np.zeros(int(vertex_count), dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for a, b, c in np.asarray(faces, dtype=np.int64):
        union(int(a), int(b))
        union(int(b), int(c))
        union(int(c), int(a))
    roots = np.asarray([find(i) for i in range(int(vertex_count))], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def _partition_pairs(labels: np.ndarray) -> np.ndarray:
    a = np.asarray(labels, dtype=np.int64)
    return a[:, None] == a[None, :]


def test_vectorized_component_labels_match_reference_partition():
    faces = np.asarray(
        [
            [0, 1, 2],
            [2, 3, 0],
            [4, 5, 6],
            [6, 7, 4],
            [9, 10, 11],
        ],
        dtype=np.int64,
    )
    reference = _reference_labels(13, faces)
    actual = mesh_connected_component_labels_v1(13, faces, face_chunk_size=2)
    assert np.array_equal(_partition_pairs(actual), _partition_pairs(reference))


def test_component_partition_is_face_order_invariant():
    rng = np.random.default_rng(260925)
    faces = rng.integers(0, 80, size=(300, 3), dtype=np.int64)
    first = mesh_connected_component_labels_v1(80, faces, face_chunk_size=17)
    shuffled = mesh_connected_component_labels_v1(
        80,
        faces[rng.permutation(len(faces))],
        face_chunk_size=23,
    )
    assert np.array_equal(_partition_pairs(first), _partition_pairs(shuffled))
