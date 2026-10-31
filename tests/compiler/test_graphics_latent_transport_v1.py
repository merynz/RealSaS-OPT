from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_core.graphics_latent_transport_v1 import (
    transport_shape_h_to_canonical_mesh_v1,
)


def _shape():
    return SimpleNamespace(
        metadata={
            "schema": "RealSaS.ShapeProposal.v2",
            "canonical_mesh_authority": False,
        },
        decoded_vertices_tessa=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        surface_xyz_tessa=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        surface_h=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def _binding():
    return SimpleNamespace(
        rows=(
            SimpleNamespace(
                candidate_vertex_id="TCV:A",
                coefficients=(("SHAPE_V2:00000000", 1.0),),
            ),
            SimpleNamespace(
                candidate_vertex_id="TCV:B",
                coefficients=(
                    ("SHAPE_V2:00000001", 0.25),
                    ("SHAPE_V2:00000002", 0.75),
                ),
            ),
        )
    )


def _mesh():
    return SimpleNamespace(
        vertices=(
            SimpleNamespace(source_candidate_vertex_id="TCV:A"),
            SimpleNamespace(source_candidate_vertex_id="TCV:B"),
        )
    )


def test_shared_h_survives_repair_by_explicit_reference_coefficients():
    value = transport_shape_h_to_canonical_mesh_v1(
        shape=_shape(),
        canonical_mesh=_mesh(),
        reference_binding=_binding(),
    )

    np.testing.assert_allclose(value.canonical_h[0], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(value.canonical_h[1], [0.0, 0.25, 0.75, 0.0])
    np.testing.assert_allclose(value.canonical_xyz_tessa[1], [0.25, 0.75, 0.0])
    assert value.diagnostics.repaired_interpolation_row_count == 1
    assert value.diagnostics.exact_identity_row_count == 1
    assert value.diagnostics.nearest_distance_max == 0.0
    assert len(value.diagnostics.transport_hash) == 64


def test_transport_is_replay_stable_and_does_not_mutate_mesh_geometry():
    mesh = _mesh()
    left = transport_shape_h_to_canonical_mesh_v1(
        shape=_shape(), canonical_mesh=mesh, reference_binding=_binding()
    )
    right = transport_shape_h_to_canonical_mesh_v1(
        shape=_shape(), canonical_mesh=mesh, reference_binding=_binding()
    )
    assert left.diagnostics.transport_hash == right.diagnostics.transport_hash
    np.testing.assert_array_equal(left.canonical_h, right.canonical_h)
    assert tuple(v.source_candidate_vertex_id for v in mesh.vertices) == ("TCV:A", "TCV:B")
