from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_core.camera_geometry_v2 import (
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    rasterize_triangles_half_integer_top_left,
)
from compiler.realsas_compiler_services.platform_worker.input_capability_v1 import (
    SourceObservationV1,
)
from compiler.realsas_compiler_services.platform_worker.mesh_capability_v1 import (
    compile_mesh_v1,
)
from compiler.realsas_compiler_services.platform_worker.shape_capability_v2 import (
    DecodedShapeProposalV2,
)


TESSA_VERTICES = np.asarray(
    [[-0.01, -0.01, 0.0], [0.01, -0.01, 0.0], [0.0, 0.01, 0.0]],
    dtype=np.float64,
)
# Deliberately negative relative to +Z normal evidence; Compiler must orient it.
FACES = np.asarray([[0, 2, 1]], dtype=np.int64)


def _camera_bundle() -> dict:
    rows = [
        {
            "origin": [0.0, 0.0, -2.0],
            "right": [1.0, 0.0, 0.0],
            "screen_up": [0.0, 1.0, 0.0],
            "forward": [0.0, 0.0, 1.0],
            "half_extent": 1.0,
        }
        for _ in range(8)
    ]
    return {
        "schema": "RealSaS.CameraProjectionBundle.v1",
        "common_center_xyz": [0.0, 0.0, 0.0],
        "common_half_extent": 1.0,
        "view_order": [f"V{i}" for i in range(8)],
        "per_view_recenter": False,
        "per_view_pose_change": False,
        "cameras": rows,
    }


def _source(*, remove_one_foreground_pixel: bool = False) -> SourceObservationV1:
    object_vertices = TESSA_VERTICES * 2.0
    camera = qualify_camera_v3(
        {
            "schema_version": "RealSaS.FullSurfaceCameraProjection.v3",
            "view_index": 0,
            "origin": [0.0, 0.0, -2.0],
            "right": [1.0, 0.0, 0.0],
            "screen_up": [0.0, 1.0, 0.0],
            "forward": [0.0, 0.0, 1.0],
            "half_extent": 1.0,
            "resolution": 1024,
        },
        view_id="V0",
        view_index=0,
    )
    projected = project_points_xyz_v3(object_vertices, camera)
    triangles = (
        tuple(
            (float(projected[int(i), 0]), float(projected[int(i), 1]))
            for i in FACES[0]
        ),
    )
    mask = bytearray(
        rasterize_triangles_half_integer_top_left(
            triangles,
            width=1024,
            height=1024,
        )
    )
    assert sum(mask) > 20
    rgba = torch.zeros((1, 8, 4, 1024, 1024), dtype=torch.uint8)
    base = np.frombuffer(bytes(mask), dtype=np.uint8).reshape(1024, 1024)
    for view in range(8):
        row = base.copy()
        if remove_one_foreground_pixel and view == 0:
            ys, xs = np.nonzero(row)
            row[int(ys[0]), int(xs[0])] = 0
        rgba[0, view, 3] = torch.from_numpy(row.copy())
    source = SourceObservationV1(native_rgba=rgba, camera_bundle=_camera_bundle())
    source.validate()
    return source


def _shape() -> DecodedShapeProposalV2:
    count = len(TESSA_VERTICES)
    topology = {
        "schema": "RealSaS.TESSADecodedTopologyCourt.v2",
        "grammar_valid": True,
        "decode_failure": None,
        "vertex_count": 3,
        "face_count": 1,
        "declared_component_count": 1,
        "geometric_component_count": 1,
        "restart_count": 1,
        "unique_edge_count": 3,
        "boundary_edge_count": 3,
        "boundary_loop_count": 1,
        "open_boundary_chain_count": 0,
        "nonmanifold_edge_count": 0,
        "zero_area_face_count": 0,
        "finite_geometry": True,
        "within_scale_policy": True,
    }
    return DecodedShapeProposalV2(
        metadata={
            "schema": "RealSaS.ShapeProposal.v2",
            "authority": "LEARNED_SHAPE_PROPOSAL_ONLY",
            "canonical_mesh_authority": False,
            "mesh_authority": False,
            "mesh_object_frame_transform": "REALSAS_OBJECT_FRAME_XYZ_EQUALS_TESSA_XYZ_TIMES_2",
            "decoded_geometry_sha256": "d" * 64,
            "topology_court": topology,
        },
        token_ids=np.asarray([1, 2, 3], dtype=np.int64),
        decoded_vertices_tessa=TESSA_VERTICES.copy(),
        decoded_faces=FACES.copy(),
        decoded_vertex_component_indices=np.asarray([0, 0, 0], dtype=np.int64),
        decoded_face_component_indices=np.asarray([0], dtype=np.int64),
        surface_xyz_tessa=TESSA_VERTICES.reshape(1, count, 3).astype(np.float32),
        surface_normal=np.asarray([[[0.0, 0.0, 1.0]] * count], dtype=np.float32),
        surface_h=np.zeros((1, count, 8), dtype=np.float32),
        mechanical={
            "stretch_major": np.ones((1, count), dtype=np.float32),
            "stretch_minor": np.zeros((1, count), dtype=np.float32),
            "tangent_direction_2theta": np.zeros((1, count, 2), dtype=np.float32),
            "bend_demand": np.zeros((1, count), dtype=np.float32),
            "rigidity_probability": np.zeros((1, count), dtype=np.float32),
            "contact_persistence_probability": np.zeros((1, count), dtype=np.float32),
        },
        contact_persistence_valid=np.zeros((1, count), dtype=np.uint8),
    )


def test_mesh_compiles_exact_source_fidelity_and_mints_static_authority():
    result = compile_mesh_v1(
        _shape(),
        _source(),
        source_shape_content_sha256="a" * 64,
        source_observation_content_sha256="b" * 64,
    )
    mesh = canonical_mesh_from_dict_v1(json.loads(result.payload))
    assert mesh.metadata["authority"] == "COMPILER_STATIC_CANONICAL_MESH"
    assert mesh.metadata["dynamic_qualification_claimed"] is False
    assert len(mesh.vertices) == 3
    assert len(mesh.faces) == 1
    assert len(mesh.edges) == 3
    assert result.diagnostics["component_count"] == 1

    by_id = {vertex.canonical_mesh_vertex_id: np.asarray(vertex.P) for vertex in mesh.vertices}
    a, b, c = (by_id[vertex_id] for vertex_id in mesh.faces[0])
    assert float(np.cross(b - a, c - a)[2]) > 0.0


def test_mesh_fails_closed_when_source_alpha_fidelity_drifts():
    with pytest.raises(RuntimeError, match="MESH_SOURCE_FIDELITY_COURT_FAILED"):
        compile_mesh_v1(
            _shape(),
            _source(remove_one_foreground_pixel=True),
            source_shape_content_sha256="a" * 64,
            source_observation_content_sha256="b" * 64,
        )
