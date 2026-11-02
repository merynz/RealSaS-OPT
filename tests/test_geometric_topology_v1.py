from __future__ import annotations

import numpy as np

from models.iris.truth.geometric_topology_v1 import (
    build_geometric_topology_v1,
    face_adjacency_from_geometric_topology_v1,
)


def _split_tetra(offset=(0.0, 0.0, 0.0), owner=0):
    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ) + np.asarray(offset, dtype=np.float64)
    # Consistently oriented tetrahedron faces. Every face owns fresh raw vertex
    # ids, exactly like a maximally seam-split render transport.
    canonical_faces = (
        (0, 2, 1),
        (0, 1, 3),
        (0, 3, 2),
        (1, 2, 3),
    )
    vertices = []
    faces = []
    for tri in canonical_faces:
        start = len(vertices)
        vertices.extend(base[list(tri)])
        faces.append((start, start + 1, start + 2))
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
        np.full(4, owner, dtype=np.int32),
    )


def test_render_seam_splits_are_not_topology_boundaries():
    vertices, faces, owner = _split_tetra()
    topology = build_geometric_topology_v1(vertices, faces, owner)

    assert len(vertices) == 12
    assert len(topology.topology_vertices) == 4
    assert topology.exact_duplicate_collapse_count == 8
    assert len(np.unique(topology.face_component_id)) == 1
    assert np.count_nonzero(topology.boundary_edge) == 0
    assert np.count_nonzero(topology.nonmanifold_edge) == 0
    assert np.all(
        topology.edge_orientation_consistent[
            topology.edge_face_incidence_count == 2
        ]
    )
    assert topology.closed_vertex_manifold is True
    assert topology.sign_topology_valid is True

    v = len(topology.topology_vertices)
    e = len(topology.unique_edge_index)
    f = len(topology.topology_faces)
    assert v - e + f == 2

    adjacency = face_adjacency_from_geometric_topology_v1(vertices, faces, owner)
    assert adjacency.shape == (6, 2)


def test_owner_boundary_prevents_false_cross_object_weld():
    a_v, a_f, a_o = _split_tetra(owner=0)
    b_v, b_f, b_o = _split_tetra(owner=1)
    b_f = b_f + len(a_v)
    vertices = np.concatenate((a_v, b_v), axis=0)
    faces = np.concatenate((a_f, b_f), axis=0)
    owner = np.concatenate((a_o, b_o), axis=0)

    topology = build_geometric_topology_v1(vertices, faces, owner)

    # The two artist objects occupy identical positions but are distinct topology.
    assert len(topology.topology_vertices) == 8
    assert len(np.unique(topology.face_component_id)) == 2
    assert topology.sign_topology_valid is True


def test_point_touching_closed_shells_fail_vertex_link_manifold_court():
    a_v, a_f, a_o = _split_tetra(owner=0)
    # Second tetra shares only the origin geometric position with the first one.
    b_v, b_f, b_o = _split_tetra(offset=(0.0, 0.0, 0.0), owner=0)
    # Move the three non-origin unique positions so only one point coincides.
    for raw_index, point in enumerate(b_v):
        if not np.array_equal(point, np.asarray([0.0, 0.0, 0.0])):
            b_v[raw_index] = point + np.asarray([-2.0, -2.0, -2.0])
    b_f = b_f + len(a_v)

    vertices = np.concatenate((a_v, b_v), axis=0)
    faces = np.concatenate((a_f, b_f), axis=0)
    owner = np.concatenate((a_o, b_o), axis=0)
    topology = build_geometric_topology_v1(vertices, faces, owner)

    # Edge incidence alone still looks closed; the vertex-link court catches the
    # non-manifold point junction created by the exact-position quotient.
    assert np.count_nonzero(topology.boundary_edge) == 0
    assert np.count_nonzero(topology.nonmanifold_edge) == 0
    assert topology.closed_edge_manifold is True
    assert topology.closed_vertex_manifold is False
    assert topology.sign_topology_valid is False
