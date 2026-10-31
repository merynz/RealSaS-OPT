from __future__ import annotations

import numpy as np

from models.iris.truth.component_geometry_authority_v1 import (
    build_component_geometry_authority_v1,
)


def tetra(offset=(0.0, 0.0, 0.0)):
    o = np.asarray(offset, dtype=np.float64)
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    ) + o
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64
    )
    return vertices, faces


def test_closed_component_survives_beside_open_component():
    v0, f0 = tetra()
    v1 = np.asarray([[3, 0, 0], [4, 0, 0], [3, 1, 0]], dtype=np.float64)
    vertices = np.concatenate((v0, v1), axis=0)
    faces = np.concatenate((f0, np.asarray([[4, 5, 6]], dtype=np.int64)), axis=0)
    authority = build_component_geometry_authority_v1(
        vertices, faces, np.zeros(len(faces), dtype=np.int32)
    )
    assert len(set(authority.surface_component_id.tolist())) == 2
    assert np.all(authority.sign_valid_for_face(np.arange(4)))
    assert not bool(authority.sign_valid_for_face(np.asarray([4]))[0])
    assert np.count_nonzero(authority.sign_component_valid) == 1


def test_cross_owner_exact_coincidence_is_not_an_implicit_weld():
    vertices = np.asarray(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 0, 0], [1, 1, 0], [0, 1, 0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    owner = np.asarray([0, 1], dtype=np.int32)
    authority = build_component_geometry_authority_v1(vertices, faces, owner)
    assert len(authority.topology_vertices) == 6
    assert len(set(authority.surface_component_id.tolist())) == 2
    assert not np.any(authority.sign_component_valid)


def test_degenerate_face_is_masked_without_erasing_closed_neighbor():
    v0, f0 = tetra()
    vertices = np.concatenate(
        (v0, np.asarray([[3, 0, 0], [4, 0, 0], [5, 0, 0]], dtype=np.float64)),
        axis=0,
    )
    faces = np.concatenate((f0, np.asarray([[4, 5, 6]], dtype=np.int64)), axis=0)
    authority = build_component_geometry_authority_v1(
        vertices, faces, np.zeros(len(faces), dtype=np.int32)
    )
    assert authority.face_valid.tolist() == [True, True, True, True, False]
    assert np.all(authority.sign_valid_for_face(np.arange(4)))
    assert not bool(authority.sign_valid_for_face(np.asarray([4]))[0])


def test_point_only_contact_does_not_merge_surface_components():
    v0, f0 = tetra()
    # Second tetra reuses only vertex 0; no full edge is shared.
    v1 = np.asarray([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float64)
    vertices = np.concatenate((v0, v1), axis=0)
    f1 = np.asarray(
        [[0, 5, 4], [0, 4, 6], [0, 6, 5], [4, 5, 6]], dtype=np.int64
    )
    faces = np.concatenate((f0, f1), axis=0)
    authority = build_component_geometry_authority_v1(
        vertices, faces, np.zeros(len(faces), dtype=np.int32)
    )
    assert len(set(authority.surface_component_id.tolist())) == 2
    assert np.count_nonzero(authority.sign_component_valid) == 2
    assert np.all(authority.sign_valid_for_face(np.arange(8)))


def test_nonmanifold_edge_prevents_false_sign_authority():
    vertices, faces = tetra()
    vertices = np.concatenate(
        (vertices, np.asarray([[0.5, 0.0, -1.0]], dtype=np.float64)), axis=0
    )
    # Extra triangle shares tetra edge (0,1), making global incidence 3.
    faces = np.concatenate((faces, np.asarray([[0, 1, 4]], dtype=np.int64)), axis=0)
    authority = build_component_geometry_authority_v1(
        vertices, faces, np.zeros(len(faces), dtype=np.int32)
    )
    assert not np.any(authority.sign_component_valid)
    assert not np.any(authority.sign_valid_for_face(np.arange(len(faces))))


def test_render_seam_duplicates_collapse_only_within_same_owner():
    # Four oriented tetra faces, but every face owns its own transport/render
    # vertex copies. The exact same-owner quotient must recover four topology
    # vertices without treating the transport duplicates as distinct geometry.
    vertices = np.asarray(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 0], [0, 0, 1], [0, 1, 0],
            [0, 0, 0], [0, 0, 1], [1, 0, 0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1],    # A,C,B
            [3, 6, 5],    # A,B,D
            [9, 10, 8],   # A,D,C
            [11, 4, 7],   # B,C,D
        ],
        dtype=np.int64,
    )
    authority = build_component_geometry_authority_v1(
        vertices, faces, np.zeros(4, dtype=np.int32)
    )
    assert np.all(authority.raw_to_topology_vertex >= 0)
    assert len(authority.topology_vertices) == 4
    assert authority.exact_duplicate_collapse_count == 8
    assert np.count_nonzero(authority.sign_component_valid) == 1
