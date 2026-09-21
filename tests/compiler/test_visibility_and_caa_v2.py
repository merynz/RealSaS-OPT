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
from compiler.realsas_compiler_core.appearance_render_v2 import render_caa_reference
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.visibility_v2 import (
    VISIBILITY_CONTRACT_V2,
    VISIBILITY_DEPTH_EQUIVALENCE_EPSILON,
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


def test_equal_depth_tie_is_sealed_face_index_but_exposed_as_ambiguity_evidence():
    result = rasterize_visible_owner(_overlap_mesh(equal_depth=True), _camera())
    visible_mask = result.owner_face_index >= 0
    visible = result.owner_face_index[visible_mask]
    assert len(visible) > 0
    assert set(map(int, np.unique(visible))) == {0}
    assert VISIBILITY_CONTRACT_V2["exact_depth_tie"] == "SEALED_FACE_INDEX_ONLY__AMBIGUITY_MUST_BE_QUALIFIED"
    assert np.all(result.second_owner_face_index[visible_mask] == 1)
    assert np.allclose(result.depth_margin[visible_mask], 0.0, atol=1e-12)


def _overlap_mesh_with_depth_delta(delta: float):
    vertices = (
        SimpleNamespace(canonical_mesh_vertex_id="a0", P=(-0.8, -0.8, 0.0)),
        SimpleNamespace(canonical_mesh_vertex_id="a1", P=(0.8, -0.8, 0.0)),
        SimpleNamespace(canonical_mesh_vertex_id="a2", P=(0.0, 0.8, 0.0)),
        SimpleNamespace(canonical_mesh_vertex_id="b0", P=(-0.8, -0.8, float(delta))),
        SimpleNamespace(canonical_mesh_vertex_id="b1", P=(0.8, -0.8, float(delta))),
        SimpleNamespace(canonical_mesh_vertex_id="b2", P=(0.0, 0.8, float(delta))),
    )
    return SimpleNamespace(
        vertices=vertices,
        faces=(("a0", "a1", "a2"), ("b0", "b1", "b2")),
    )


def test_depth_equivalence_epsilon_is_explicit_and_fail_closed_at_boundary():
    eps = VISIBILITY_DEPTH_EQUIVALENCE_EPSILON
    assert eps == 1.0e-12
    assert (
        VISIBILITY_CONTRACT_V2["depth_buffer"]
        == "IEEE754_FLOAT64_SOFTWARE_SORT__NO_HARDWARE_Z_QUANTIZATION"
    )
    assert VISIBILITY_CONTRACT_V2["depth_equivalence_epsilon_camera_z"] == eps

    inside = rasterize_visible_owner(
        _overlap_mesh_with_depth_delta(0.5 * eps),
        _camera(),
    )
    inside_visible = inside.owner_face_index >= 0
    assert np.any(inside_visible)
    assert np.all(
        np.abs(inside.depth_margin[inside_visible]) <= eps + 1.0e-15
    )

    outside_far = rasterize_visible_owner(
        _overlap_mesh_with_depth_delta(2.0 * eps),
        _camera(),
    )
    outside_far_visible = outside_far.owner_face_index >= 0
    assert set(
        map(int, np.unique(outside_far.owner_face_index[outside_far_visible]))
    ) == {0}
    assert np.all(outside_far.depth_margin[outside_far_visible] > eps)

    outside_near = rasterize_visible_owner(
        _overlap_mesh_with_depth_delta(-2.0 * eps),
        _camera(),
    )
    outside_near_visible = outside_near.owner_face_index >= 0
    assert set(
        map(int, np.unique(outside_near.owner_face_index[outside_near_visible]))
    ) == {1}
    assert np.all(outside_near.depth_margin[outside_near_visible] > eps)


def test_separated_overlap_exposes_positive_runner_up_depth_margin():
    result = rasterize_visible_owner(_overlap_mesh(equal_depth=False), _camera())
    visible_mask = result.owner_face_index >= 0
    assert np.any(result.second_owner_face_index[visible_mask] == 1)
    assert np.all(result.depth_margin[visible_mask] > 0.0)


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
            "boundary_safe_erosion_px": 0,
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

def test_linear_light_midpoint_is_not_gamma_space_average():
    image = np.asarray([[[0, 0, 0, 255], [255, 255, 255, 255]]], dtype=np.uint8)
    sampled = bilinear_rgba_u8(
        image,
        np.asarray([[0.5, 0.0]], dtype=np.float64),
    )[0]
    # 0.5 linear light encodes to approximately sRGB 188, not encoded-space 128.
    assert 186 <= int(sampled[0]) <= 189
    assert int(sampled[0]) == int(sampled[1]) == int(sampled[2])
    assert int(sampled[3]) == 255


def test_transparent_front_geometry_reveals_deeper_character_layer():
    mesh = _overlap_mesh(equal_depth=False)
    camera = _camera(resolution=32)
    face_uv = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    texture = np.asarray(
        [[[255, 0, 0, 0], [0, 255, 0, 255]]],
        dtype=np.uint8,
    )
    provenance = np.zeros((1, 2), dtype=np.uint8)
    with np.errstate(invalid="raise"):
        render = render_caa_reference(
            mesh=mesh,
            camera=camera,
            face_uv=face_uv,
            texture_rgba_u8=texture,
            provenance_atlas=provenance,
        )
    center = render.straight_rgba_u8[16, 16]
    assert tuple(map(int, center)) == (0, 255, 0, 255)
    assert int(render.coverage_sample_count) == 4
    assert set(
        map(int, render.coverage_sample_owner_face_index[16, 16])
    ) == {0}
    contributing_faces = render.layer_owner_face_index[16, 16][
        render.contributing_layer_mask[16, 16]
    ]
    assert len(contributing_faces) == 4
    assert set(map(int, contributing_faces)) == {1}
    assert int(render.contributing_layer_count[16, 16]) == 4
    assert not bool(render.layer_overflow[16, 16])


def test_runtime_reference_2x2_coverage_produces_fractional_edge_alpha():
    mesh = SimpleNamespace(
        vertices=(
            SimpleNamespace(
                canonical_mesh_vertex_id="v0",
                P=(-0.73, -0.73, 0.0),
            ),
            SimpleNamespace(
                canonical_mesh_vertex_id="v1",
                P=(0.73, -0.73, 0.0),
            ),
            SimpleNamespace(
                canonical_mesh_vertex_id="v2",
                P=(0.0, 0.73, 0.0),
            ),
        ),
        faces=(("v0", "v1", "v2"),),
    )
    face_uv = np.zeros((1, 3, 2), dtype=np.float64)
    texture = np.asarray([[[220, 80, 40, 255]]], dtype=np.uint8)
    provenance = np.zeros((1, 1), dtype=np.uint8)
    render = render_caa_reference(
        mesh=mesh,
        camera=_camera(resolution=32),
        face_uv=face_uv,
        texture_rgba_u8=texture,
        provenance_atlas=provenance,
    )
    alpha = render.straight_rgba_u8[:, :, 3]
    partial = alpha[(alpha > 0) & (alpha < 255)]
    assert len(partial) > 0
    assert set(map(int, np.unique(partial))).issubset({64, 128, 191})
    assert np.any(render.geometry_visible & (alpha < 255))


def test_visibility_reports_layer_overflow_instead_of_silent_drop():
    vertices = []
    faces = []
    for layer in range(5):
        z = 0.05 * layer
        ids = []
        for corner, point in enumerate(((-0.8, -0.8), (0.8, -0.8), (0.0, 0.8))):
            vertex_id = f"l{layer}_{corner}"
            ids.append(vertex_id)
            vertices.append(
                SimpleNamespace(
                    canonical_mesh_vertex_id=vertex_id,
                    P=(point[0], point[1], z),
                )
            )
        faces.append(tuple(ids))
    mesh = SimpleNamespace(vertices=tuple(vertices), faces=tuple(faces))
    result = rasterize_visible_owner(mesh, _camera())
    assert np.any(result.layer_overflow)


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
