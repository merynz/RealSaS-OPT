from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    CAACompileArtifactIR,
    caa_compile_hash,
    validate_caa_compile_artifact,
)
from compiler.realsas_compiler_core.appearance_bake_v2 import (
    bake_direction_adaptive_atlas_pages,
    bake_direction_adaptive_source_view_atlas_pages,
)
from compiler.realsas_compiler_core.appearance_compile_v2 import (
    adaptive_face_atlas_plan,
    adaptive_face_sample_offsets,
)
from compiler.realsas_compiler_core.appearance_completion_v2 import (
    SurfaceSampleGraph,
    surface_sample_neighbors,
)
from compiler.realsas_compiler_core.types import QualificationError


def test_adaptive_sample_offsets_are_exact_for_mixed_triangular_lattices():
    resolutions = np.asarray([4, 8, 32], dtype=np.int32)
    offsets = adaptive_face_sample_offsets(resolutions)
    # T(r) = r(r+1)/2.
    assert tuple(map(int, offsets)) == (0, 10, 46, 574)
    assert int(offsets[-1]) == 10 + 36 + 528


def test_adaptive_atlas_plan_is_deterministic_and_preserves_face_resolution():
    resolutions = np.asarray([4, 32, 4, 8, 4], dtype=np.int32)
    first = adaptive_face_atlas_plan(
        resolutions,
        bleed_px=2,
        max_page_resolution=48,
    )
    second = adaptive_face_atlas_plan(
        resolutions,
        bleed_px=2,
        max_page_resolution=48,
    )
    assert first["layout"] == second["layout"]
    assert np.array_equal(first["face_uv"], second["face_uv"])
    assert np.array_equal(first["face_page_index"], second["face_page_index"])
    assert np.array_equal(first["face_tile_resolution"], resolutions)
    assert first["layout"]["selected_resolution_histogram"] == {
        "4": 3,
        "8": 1,
        "32": 1,
    }
    assert first["layout"]["placement_hash"]
    assert first["layout"]["page_count"] >= 1
    assert np.all((first["face_uv"] >= 0.0) & (first["face_uv"] <= 1.0))


def test_adaptive_rgba_provenance_and_source_view_share_one_texel_mapping():
    resolutions = np.asarray([4, 8], dtype=np.int32)
    offsets = adaptive_face_sample_offsets(resolutions)
    total = int(offsets[-1])
    sample_ids = np.arange(total, dtype=np.int64)

    rgba = np.zeros((total, 4), dtype=np.uint8)
    rgba[:, 0] = (sample_ids % 251).astype(np.uint8)
    rgba[:, 1] = ((sample_ids // 251) % 251).astype(np.uint8)
    rgba[:, 3] = 255
    provenance = (sample_ids % 3).astype(np.uint8)
    source_view = (sample_ids % 8).astype(np.int16)

    pages, prov_pages, uv, page_index, layout = (
        bake_direction_adaptive_atlas_pages(
            face_sample_rgba=rgba,
            face_sample_provenance=provenance,
            face_tile_resolutions=resolutions,
            face_sample_offsets=offsets,
            bleed_px=2,
            max_page_resolution=24,
        )
    )
    donor_pages = bake_direction_adaptive_source_view_atlas_pages(
        face_sample_source_view=source_view,
        face_tile_resolutions=resolutions,
        face_sample_offsets=offsets,
        bleed_px=2,
        max_page_resolution=24,
    )
    assert pages.shape[:3] == prov_pages.shape
    assert donor_pages.shape == prov_pages.shape
    assert uv.shape == (2, 3, 2)
    assert page_index.shape == (2,)
    assert layout["maximum_tile_resolution"] == 8

    allocated = prov_pages != 255
    reconstructed_ids = (
        pages[..., 0].astype(np.int64)
        + 251 * pages[..., 1].astype(np.int64)
    )
    assert np.array_equal(
        prov_pages[allocated],
        (reconstructed_ids[allocated] % 3).astype(np.uint8),
    )
    assert np.array_equal(
        donor_pages[allocated],
        (reconstructed_ids[allocated] % 8).astype(np.int16),
    )


def test_adaptive_surface_graph_keeps_local_lattices_and_shared_vertices():
    # Two triangles share the edge from (1,0,0) to (0,1,0), but use different
    # face resolutions. Product coupling is topology-parametric: interior edge
    # fractions need not coincide geometrically across the two lattices.
    r0, r1 = 4, 8
    offsets = adaptive_face_sample_offsets(np.asarray([r0, r1], dtype=np.int32))

    def bary(resolution: int):
        rows = []
        d = float(resolution - 1)
        for j in range(resolution):
            for i in range(resolution - j):
                rows.append((1.0 - i / d - j / d, i / d, j / d))
        return np.asarray(rows, dtype=np.float64)

    tri0 = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    tri1 = np.asarray(
        ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    positions = np.concatenate((bary(r0) @ tri0, bary(r1) @ tri1), axis=0)
    neighbors = surface_sample_neighbors(
        positions=positions,
        face_count=2,
        face_sample_offsets=offsets,
        face_tile_resolutions=np.asarray([r0, r1], dtype=np.int32),
        face_vertex_ids=(
            ("a0", "shared0", "shared1"),
            ("shared0", "b1", "shared1"),
        ),
    )
    assert isinstance(neighbors, SurfaceSampleGraph)
    assert len(neighbors) == int(offsets[-1])
    assert neighbors.edge_count > 0
    assert neighbors.storage_nbytes > 0
    # Every sample has a local or shared canonical neighbour.
    assert all(len(row) > 0 for row in neighbors)
    # Every sample on the true shared edge must couple across the face
    # boundary, even though r=4 (thirds) and r=8 (sevenths) have no compatible
    # interior sample fractions.
    split = int(offsets[1])
    cross = {
        (left, right)
        for left, row in enumerate(neighbors[:split])
        for right in row
        if right >= split
    }
    assert len(cross) >= 2
    edge_mask = np.isclose(positions[:, 0] + positions[:, 1], 1.0, atol=1e-12)
    for sample_index in np.flatnonzero(edge_mask):
        row = neighbors[int(sample_index)]
        if int(sample_index) < split:
            assert any(int(neighbor) >= split for neighbor in row)
        else:
            assert any(int(neighbor) < split for neighbor in row)



def test_compact_surface_graph_storage_scales_linearly_without_python_sets():
    face_count = 128
    resolution = 32
    per_face = resolution * (resolution + 1) // 2
    sample_count = face_count * per_face
    positions = np.zeros((sample_count, 3), dtype=np.float64)
    topology = tuple(
        (f"f{face}_v0", f"f{face}_v1", f"f{face}_v2")
        for face in range(face_count)
    )
    graph = surface_sample_neighbors(
        positions=positions,
        face_count=face_count,
        tile_resolution=resolution,
        face_vertex_ids=topology,
    )
    assert isinstance(graph, SurfaceSampleGraph)
    assert len(graph) == sample_count
    assert graph.edge_count > sample_count
    # A Python set/list graph at this scale costs many hundreds of bytes per
    # sample. CSR should remain comfortably below 96 bytes/sample including
    # the undirected edge index retained for Stage24 seam evaluation.
    assert graph.storage_nbytes < sample_count * 96
    assert int(graph.offsets[-1]) == 2 * graph.edge_count



def _adaptive_artifact() -> CAACompileArtifactIR:
    face_count = 3
    resolutions = np.asarray([4, 8, 4], dtype=np.int32)
    per_direction = int(adaptive_face_sample_offsets(resolutions)[-1])
    value = CAACompileArtifactIR(
        backend_id="DETERMINISTIC_V1",
        preregistration_binding_hash="1" * 64,
        candidate_mesh_binding_hash="2" * 64,
        surface_addressing_binding_hash="3" * 64,
        appearance_domain_binding_hash="4" * 64,
        output_direction_set_binding_hash="5" * 64,
        compile_npz_path="/tmp/caa.npz",
        compile_npz_sha256="6" * 64,
        face_count=face_count,
        direction_count=8,
        tile_resolution=8,
        sample_count_per_face=36,
        total_sample_count=8 * per_direction,
        direct_source_sample_count=8 * per_direction,
        other_view_source_sample_count=0,
        compiled_local_harmonic_sample_count=0,
        compile_hash="",
        metadata={
            "sample_count_mode": "PER_FACE_ADAPTIVE_V1",
            "sample_count_per_direction": per_direction,
            "maximum_tile_resolution": 8,
            "selected_resolution_histogram": {"4": 2, "8": 1},
        },
    )
    return replace(value, compile_hash=caa_compile_hash(value))


def test_adaptive_compile_artifact_accepts_actual_total_not_uniform_worst_case():
    value = _adaptive_artifact()
    validate_caa_compile_artifact(value)
    uniform_worst = value.face_count * value.direction_count * value.sample_count_per_face
    assert value.total_sample_count < uniform_worst


def test_adaptive_compile_artifact_rejects_sample_accounting_drift():
    value = _adaptive_artifact()
    tampered = replace(
        value,
        total_sample_count=value.total_sample_count + 8,
        direct_source_sample_count=value.direct_source_sample_count + 8,
        compile_hash="",
    )
    tampered = replace(tampered, compile_hash=caa_compile_hash(tampered))
    with pytest.raises(
        QualificationError,
        match="CAA_COMPILE_ADAPTIVE_TOTALITY_ACCOUNTING_DRIFT",
    ):
        validate_caa_compile_artifact(tampered)


def test_adaptive_surface_graph_does_not_cross_coincident_nonadjacent_sheets():
    resolutions = np.asarray([4, 4], dtype=np.int32)
    offsets = adaptive_face_sample_offsets(resolutions)

    tri = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )

    def bary(resolution: int):
        rows = []
        d = float(resolution - 1)
        for j in range(resolution):
            for i in range(resolution - j):
                rows.append((1.0 - i / d - j / d, i / d, j / d))
        return np.asarray(rows, dtype=np.float64)

    one = bary(4) @ tri
    positions = np.concatenate((one, one.copy()), axis=0)
    split = int(offsets[1])

    topology_bound = surface_sample_neighbors(
        positions=positions,
        face_count=2,
        face_sample_offsets=offsets,
        face_tile_resolutions=resolutions,
        face_vertex_ids=(
            ("a0", "a1", "a2"),
            ("b0", "b1", "b2"),
        ),
    )
    assert not any(
        neighbor >= split
        for row in topology_bound[:split]
        for neighbor in row
    )

    shared_edge = surface_sample_neighbors(
        positions=positions,
        face_count=2,
        face_sample_offsets=offsets,
        face_tile_resolutions=resolutions,
        face_vertex_ids=(
            ("shared0", "shared1", "a2"),
            ("shared0", "shared1", "b2"),
        ),
    )
    assert any(
        neighbor >= split
        for row in shared_edge[:split]
        for neighbor in row
    )
