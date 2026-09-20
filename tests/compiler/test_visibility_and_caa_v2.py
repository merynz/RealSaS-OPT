from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_core.appearance_bake_v2 import (
    bake_direction_atlas,
    bilinear_premultiplied_rgba,
    conservative_bilinear_provenance,
)
from compiler.realsas_compiler_core.appearance_compile_v2 import (
    bilinear_rgba_u8,
    compile_deterministic_caa,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.visibility_v2 import (
    VISIBILITY_CONTRACT_V2,
    rasterize_visible_owner,
)


def _camera(view: int = 0, resolution: int = 32):
    return CameraProjectionV3(
        view_id=f"V{view}",
        view_index=view,
        origin=(0.0, 0.0, -2.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        half_extent=1.0,
        resolution=resolution,
    )


def _overlap_mesh(equal_depth: bool = False):
    far_z = 0.0 if equal_depth else 0.25
    vertices = (
        SimpleNamespace(canonical_mesh_vertex_id="a0", P=(-0.8, -0.8, 0.0)),
        SimpleNamespace(canonical_mesh_vertex_id="a1", P=(0.8, -0.8, 0.0)),
        SimpleNamespace(canonical_mesh_vertex_id="a2", P=(0.0, 0.8, 0.0)),
        SimpleNamespace(canonical_mesh_vertex_id="b0", P=(-0.8, -0.8, far_z)),
        SimpleNamespace(canonical_mesh_vertex_id="b1", P=(0.8, -0.8, far_z)),
        SimpleNamespace(canonical_mesh_vertex_id="b2", P=(0.0, 0.8, far_z)),
    )
    return SimpleNamespace(
        vertices=vertices,
        faces=(("a0", "a1", "a2"), ("b0", "b1", "b2")),
    )


def test_visibility_is_geometry_depth_only_and_nearer_face_wins():
    assert VISIBILITY_CONTRACT_V2["appearance_input_forbidden"] is True
    assert VISIBILITY_CONTRACT_V2["texture_alpha_selects_front_surface"] is False
    result = rasterize_visible_owner(_overlap_mesh(), _camera())
    visible = result.owner_face_index[result.owner_face_index >= 0]
    assert len(visible) > 0
    assert set(map(int, np.unique(visible))) == {0}


def test_equal_depth_tie_is_sealed_face_index_not_appearance():
    result = rasterize_visible_owner(_overlap_mesh(equal_depth=True), _camera())
    visible = result.owner_face_index[result.owner_face_index >= 0]
    assert len(visible) > 0
    assert set(map(int, np.unique(visible))) == {0}
    assert VISIBILITY_CONTRACT_V2["exact_depth_tie"] == "SEALED_FACE_INDEX_ONLY"


def _candidate():
    vertices = (
        SimpleNamespace(candidate_vertex_id="v0", P=(-0.95, -0.95, 0.0), component_id="c0"),
        SimpleNamespace(candidate_vertex_id="v1", P=(0.95, -0.95, 0.0), component_id="c0"),
        SimpleNamespace(candidate_vertex_id="v2", P=(0.0, 0.95, 0.0), component_id="c0"),
    )
    return SimpleNamespace(vertices=vertices, faces=(("v0", "v1", "v2"),))


def test_safe_transparent_source_background_is_defined_direct_source_not_unseen():
    candidate = _candidate()
    cameras = tuple(_camera(view, 32) for view in range(8))
    images = {}
    masks = {}
    for view in range(8):
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:22, 10:22] = True
        rgba[10:22, 10:22] = (120, 90, 50, 255)
        images[view] = rgba
        masks[view] = mask

    result = compile_deterministic_caa(
        candidate=candidate,
        cameras=cameras,
        source_rgba_by_view=images,
        foreground_mask_by_view=masks,
        tile_resolution=8,
        source_lock_policy={
            "min_abs_normal_camera_cos": 0.17,
            "boundary_safe_erosion_px": 1,
            "min_source_alpha_u8": 1,
        },
    )
    assert not np.any(result["provenance"] == 255)
    direct = result["provenance"] == 0
    assert np.any(direct)
    direct_alpha = result["rgba"][:, :, 3][direct]
    assert np.any(direct_alpha == 0)
    assert np.any(direct_alpha == 255)


def test_source_transport_bilinear_is_premultiplied_safe():
    image = np.asarray([[[255, 0, 0, 255], [0, 0, 255, 0]]], dtype=np.uint8)
    sampled = bilinear_rgba_u8(
        image,
        np.asarray([[0.5, 0.0]], dtype=np.float64),
    )[0]
    assert 126 <= int(sampled[3]) <= 129
    assert int(sampled[0]) >= 250
    assert int(sampled[1]) == 0
    assert int(sampled[2]) == 0


def test_premultiplied_bilinear_filtering_does_not_bleed_transparent_rgb():
    image = np.asarray([[[255, 0, 0, 255], [0, 0, 255, 0]]], dtype=np.uint8)
    sampled = bilinear_premultiplied_rgba(
        image,
        np.asarray([[0.5, 0.0]], dtype=np.float64),
    )[0]
    assert 0.49 <= sampled[3] <= 0.51
    assert 0.49 <= sampled[0] <= 0.51
    assert sampled[1] == 0.0
    assert sampled[2] == 0.0


def test_bilinear_provenance_is_conservative_over_color_footprint():
    provenance = np.asarray([[0, 2]], dtype=np.uint8)
    sampled = conservative_bilinear_provenance(
        provenance,
        np.asarray([[0.49, 0.0], [0.0, 0.0]], dtype=np.float64),
    )
    assert int(sampled[0]) == 2
    assert int(sampled[1]) == 0


def test_face_atlas_bleed_leaves_no_undefined_texel():
    # tile_resolution=4 => 10 triangle samples for one face.
    rgba = np.tile(np.asarray([[40, 80, 120, 255]], dtype=np.uint8), (10, 1))
    provenance = np.zeros(10, dtype=np.uint8)
    atlas, prov, uv, layout = bake_direction_atlas(
        face_sample_rgba=rgba,
        face_sample_provenance=provenance,
        face_count=1,
        tile_resolution=4,
        bleed_px=2,
    )
    assert atlas.shape[:2] == (layout["height"], layout["width"])
    assert uv.shape == (1, 3, 2)
    assert not np.any(prov == 255)


def test_face_atlas_allows_unallocated_grid_padding_but_not_surface_undefinedness():
    # Three faces require a 2x2 grid; the fourth tile is intentional non-surface padding.
    per_face = 4 * (4 + 1) // 2
    rgba = np.tile(
        np.asarray([[10, 20, 30, 255]], dtype=np.uint8),
        (3 * per_face, 1),
    )
    provenance = np.zeros(3 * per_face, dtype=np.uint8)
    atlas, prov, uv, layout = bake_direction_atlas(
        face_sample_rgba=rgba,
        face_sample_provenance=provenance,
        face_count=3,
        tile_resolution=4,
        bleed_px=2,
    )
    assert layout["columns"] == 2 and layout["rows"] == 2
    assert uv.shape == (3, 3, 2)
    stride = layout["tile_stride"]
    unused = prov[stride : 2 * stride, stride : 2 * stride]
    assert np.all(unused == 255)
    for face in range(3):
        tx = (face % layout["columns"]) * stride
        ty = (face // layout["columns"]) * stride
        assert not np.any(prov[ty : ty + stride, tx : tx + stride] == 255)
