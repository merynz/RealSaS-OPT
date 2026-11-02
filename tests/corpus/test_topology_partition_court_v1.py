from __future__ import annotations

import numpy as np

from tools.corpus.topology_partition_court_v1 import (
    canonicalize_partition,
    edge_connected_face_partition,
    partitions_equivalent,
    topology_partition_diagnostics,
    vertex_connected_face_partition,
)


def test_component_numeric_labels_are_not_topology_truth():
    # Two disconnected triangles. Producer labels by minimum vertex identity while
    # an independent court may label by first face occurrence.
    faces = np.asarray([[3, 4, 5], [0, 1, 2]], dtype=np.int32)
    producer = np.asarray([1, 0], dtype=np.int32)
    court = edge_connected_face_partition(faces)
    assert np.array_equal(court, np.asarray([0, 1], dtype=np.int32))
    assert not np.array_equal(producer, court)
    assert partitions_equivalent(producer, court)
    assert np.array_equal(canonicalize_partition(producer), court)


def test_point_only_contact_is_vertex_connected_but_not_edge_connected():
    faces = np.asarray([[0, 1, 2], [0, 3, 4]], dtype=np.int32)
    vertex = vertex_connected_face_partition(faces)
    edge = edge_connected_face_partition(faces)
    assert len(np.unique(vertex)) == 1
    assert len(np.unique(edge)) == 2
    assert not partitions_equivalent(vertex, edge)


def test_nonmanifold_edge_connects_surface_partition_but_not_manifold_partition():
    # Three faces share edge (0, 1). Geometrically they meet along a full edge,
    # but that edge cannot support a two-face manifold adjacency/sign claim.
    faces = np.asarray(
        [[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int32
    )
    surface = edge_connected_face_partition(faces, manifold_edges_only=False)
    manifold = edge_connected_face_partition(faces, manifold_edges_only=True)
    assert len(np.unique(surface)) == 1
    assert len(np.unique(manifold)) == 3
    diagnostics = topology_partition_diagnostics(faces)
    assert diagnostics["surface_edge_connected_component_count"] == 1
    assert diagnostics["manifold_edge_connected_component_count"] == 3
    assert diagnostics["surface_edge_vs_manifold_edge_partition_equivalent"] is False
    assert diagnostics["manifold_edge_refines_surface_edge"] is True


def test_producer_partition_comparison_is_label_invariant():
    faces = np.asarray(
        [[3, 4, 5], [0, 1, 2], [3, 5, 6], [0, 2, 7]], dtype=np.int32
    )
    producer = np.asarray([9, 4, 9, 4], dtype=np.int32)
    diagnostics = topology_partition_diagnostics(faces, producer)
    assert diagnostics["producer_component_count"] == 2
    assert diagnostics["producer_vs_surface_edge_partition_equivalent"] is True
    assert diagnostics["producer_raw_ids_equal_surface_edge_ids"] is False
