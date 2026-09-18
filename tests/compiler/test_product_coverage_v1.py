from __future__ import annotations

from dataclasses import replace

from types import SimpleNamespace

from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    ComponentObservationRasterIR,
    _mesh_component_triangles,
    camera_projection_binding_hash,
    coverage_metrics,
    mask_sha256,
    rasterize_triangles_half_integer_top_left,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.playback_runtime_v3 import ReferenceRasterContractV1


def test_half_integer_top_left_rasterizer_matches_expected_square_split_without_double_holes():
    triangles = (
        ((0.0,0.0),(2.0,0.0),(2.0,2.0)),
        ((0.0,0.0),(2.0,2.0),(0.0,2.0)),
    )
    mask = rasterize_triangles_half_integer_top_left(triangles,width=3,height=3)
    # The 2x2 square covers the four centers at (0.5/1.5, 0.5/1.5).
    assert sum(mask) == 4
    assert mask[0] == 1 and mask[1] == 1
    assert mask[3] == 1 and mask[4] == 1


def test_coverage_metrics_exposes_coherent_hole_and_interior_peppering_separately():
    width=5; height=5
    authority=bytes([1]*(width*height))
    predicted=bytearray(authority)
    predicted[2*width+2]=0
    metrics=coverage_metrics(authority,bytes(predicted),width=width,height=height)
    assert metrics["recall"] == 24/25
    assert metrics["precision"] == 1.0
    assert metrics["largest_coherent_hole_pixels"] == 1
    assert metrics["largest_coherent_hole_fraction"] == 1/25
    # 5x5 eroded by one reference pixel leaves a 3x3 interior; center is missing.
    assert metrics["interior_foreground_pixel_count"] == 9
    assert metrics["interior_uncovered_fraction"] == 1/9


def test_component_observation_hash_is_exact_mask_bytes():
    contract=ReferenceRasterContractV1()
    mask=bytes([0,1,1,0])
    row=ComponentObservationRasterIR(
        0,"c0","MESH",2,2,mask,mask_sha256(mask),
        "source-observation-hash","camera-hash",contract.contract_hash,
    )
    assert row.mask_sha256 == mask_sha256(mask)
    assert row.to_dict()["mask_bytes_count"] == 4


def test_canonical_g5_projection_does_not_require_per_view_surface_raster_bindings():
    camera=CameraProjectionV3(
        "V0",0,
        origin=(0.0,0.0,-2.0),
        right=(1.0,0.0,0.0),
        screen_up=(0.0,1.0,0.0),
        forward=(0.0,0.0,1.0),
        half_extent=1.0,
        resolution=4,
    )
    vertices=(
        SimpleNamespace(canonical_mesh_vertex_id="v0",P=(-0.5,-0.5,0.0),component_id="c0"),
        SimpleNamespace(canonical_mesh_vertex_id="v1",P=(0.5,-0.5,0.0),component_id="c0"),
        SimpleNamespace(canonical_mesh_vertex_id="v2",P=(0.0,0.5,0.0),component_id="c0"),
    )
    mesh=SimpleNamespace(vertices=vertices,faces=(("v0","v1","v2"),))
    triangles=_mesh_component_triangles(mesh,camera,component_id="c0")
    assert len(triangles)==1
    # Exact orthographic projection from canonical 3D: no SurfaceNode.raster_bindings involved.
    assert triangles[0][0] == (1.0,3.0)
    assert triangles[0][1] == (3.0,3.0)
    assert triangles[0][2] == (2.0,1.0)
    assert len(camera_projection_binding_hash(camera)) == 64
