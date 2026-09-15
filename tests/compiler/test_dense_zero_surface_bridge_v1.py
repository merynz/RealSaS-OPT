from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.mesh.dense_zero_surface_bridge import (
    DENSE_ZERO_SURFACE_TOPOLOGY_METHOD,
    build_dense_zero_surface_candidate,
    qualify_dense_zero_surface_mesh,
    replay_dense_zero_surface_compaction,
    validate_dense_zero_surface_mesh,
)
from compiler.realsas_compiler_core.types import QualificationError, RiggingSurfaceIR, SurfaceNode


def _fixture():
    dense = np.asarray([
        [-0.80, 0.00, -0.40],
        [-0.76, 0.00, -0.28],
        [ 0.75, 0.00,  0.70],
        [ 0.65, 0.00, -0.70],
    ], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    divisions = 2
    lo = dense.min(axis=0)
    span = np.maximum(dense.max(axis=0) - lo, 1.0e-12)
    keys = np.floor((dense - lo) / span * divisions).astype(np.int64)
    keys = np.clip(keys, 0, divisions - 1)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=count).astype(np.float64)
    compact = np.zeros((count, 3), dtype=np.float64)
    np.add.at(compact, inverse, dense)
    compact /= counts[:, None]
    nodes = tuple(
        SurfaceNode(
            surface_id=f"S:{i}",
            P=tuple(map(float, compact[i])),
            support_views=(0,),
            provenance_refs=("TEST",),
            source_observation_ids=("OBS0",),
            raster_bindings=(),
        )
        for i in range(count)
    )
    surface = RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=(),
        geometry_lineage_hash="SURFACE_TEST_LINEAGE",
        metadata={
            "compact_voxel_divisions": divisions,
            "source_dense_vertex_count": len(dense),
            "source_zero_surface_sha256": "ZERO_SHA",
        },
    )
    camera = {
        "view_index": 0,
        "origin": [0.0, 5.0, 0.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 0.0, 1.0],
        "forward": [0.0, -1.0, 0.0],
        "half_extent": 1.0,
        "resolution": 128,
    }
    return surface, dense, faces, camera, inverse


def _qualified(surface, dense, faces, camera):
    replay = replay_dense_zero_surface_compaction(surface, dense, source_zero_surface_sha256="ZERO_SHA")
    candidate = build_dense_zero_surface_candidate(
        surface, dense, faces, replay,
        selected_face_indices=(0,), camera=camera, view_index=0,
        camera_binding_hash="CAMERA_HASH", source_zero_surface_sha256="ZERO_SHA",
        dynamic_witness_hash="DYNAMIC_HASH", coverage_witness_hash="COVERAGE_HASH",
    )
    mesh, report = qualify_dense_zero_surface_mesh(
        surface, candidate, dense, faces, replay,
        camera=camera, source_zero_surface_sha256="ZERO_SHA",
    )
    return replay, mesh, report


def test_dense_zero_surface_bridge_replays_compaction_and_preserves_source_face():
    surface, dense, faces, camera, inverse = _fixture()
    replay, mesh, report = _qualified(surface, dense, faces, camera)
    assert replay.voxel_divisions == 2
    assert replay.dense_vertex_count == 4
    assert replay.max_compact_point_error <= 1.0e-12
    assert np.array_equal(replay.dense_vertex_to_compact_index, inverse)
    assert report["passed"] is True
    assert report["method"] == DENSE_ZERO_SURFACE_TOPOLOGY_METHOD
    assert report["selected_dense_face_count"] == 1
    assert report["original_dense_zero_surface_faces_only"] is True
    assert report["new_adjacency_created"] is False

    by_dense = {int(v.metadata["source_dense_vertex_index"]): v for v in mesh.vertices}
    if inverse[0] == inverse[1]:
        assert by_dense[0].support_binding.coefficients == by_dense[1].support_binding.coefficients
        assert by_dense[0].metadata["raster_xy"] != by_dense[1].metadata["raster_xy"]


def test_dense_zero_surface_bridge_rejects_compaction_point_drift():
    surface, dense, _faces, _camera, _inverse = _fixture()
    nodes = list(surface.surface_nodes)
    nodes[0] = replace(nodes[0], P=(nodes[0].P[0] + 1.0e-3, *nodes[0].P[1:]))
    with pytest.raises(QualificationError, match="COMPACT_POINT_REPLAY_DRIFT"):
        replay_dense_zero_surface_compaction(
            replace(surface, surface_nodes=tuple(nodes)), dense,
            source_zero_surface_sha256="ZERO_SHA",
        )


def test_dense_zero_surface_bridge_rejects_non_source_face_relabel():
    surface, dense, faces, camera, _inverse = _fixture()
    replay, mesh, _report = _qualified(surface, dense, faces, camera)
    tampered_faces = faces.copy(); tampered_faces[0] = np.asarray([0, 1, 3])
    with pytest.raises(QualificationError, match="FACE_NOT_SOURCE_TOPOLOGY"):
        validate_dense_zero_surface_mesh(
            mesh, surface, dense, tampered_faces, replay,
            camera=camera, source_zero_surface_sha256="ZERO_SHA",
        )


def test_dense_zero_surface_bridge_rejects_tampered_vertex_compaction_witness():
    surface, dense, faces, camera, _inverse = _fixture()
    replay, mesh, _report = _qualified(surface, dense, faces, camera)
    vertices = list(mesh.vertices)
    md = dict(vertices[0].metadata)
    md["compact_surface_index"] = (int(md["compact_surface_index"]) + 1) % len(surface.surface_nodes)
    vertices[0] = replace(vertices[0], metadata=md)
    with pytest.raises(QualificationError):
        validate_dense_zero_surface_mesh(
            replace(mesh, vertices=tuple(vertices)), surface, dense, faces, replay,
            camera=camera, source_zero_surface_sha256="ZERO_SHA",
        )
