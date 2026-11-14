from __future__ import annotations

import numpy as np
import pytest

from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
    admit_raw_two_view_mesh_v1,
)
from compiler.rss_vnext.workers.rss_vnext_source_rgb_projection_candidate_v1 import (
    build_raw_source_rgb_image_plane_candidates_v1,
)
from compiler.rss_vnext.workers.rss_vnext_uv_texture_atlas_v1 import (
    compile_rss_vnext_uv_texture_atlas_v1,
    reopen_rss_vnext_uv_texture_atlas_v1,
)
from tests.compiler.test_rss_vnext_raw_mesh_admission_v1 import (
    _triangle_source_and_scene,
)


def _fixture():
    source, scene, receipt = _triangle_source_and_scene()
    mesh, _, _ = admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt,
        scene_digest="c" * 64,
    )
    assert mesh is not None
    original, _ = build_raw_source_rgb_image_plane_candidates_v1(
        source=source, mesh=mesh,
    )
    return mesh, original


def test_same_H_graphics_candidate_completes_unknown_uv_in_separate_neural_layer():
    mesh, samples = _fixture()
    generated_logits = np.asarray([
        [5., -5., -4., 0., -2.],
        [5., -5., -4., 0., -2.],
        [5., -5., -4., 0., -2.],
    ], dtype=np.float32)
    body, metadata = compile_rss_vnext_uv_texture_atlas_v1(
        source_rgb_candidates=samples,
        mesh=mesh, graphics_vertex_logits=generated_logits,
        graphics_scene_content_sha256="f" * 64,
    )
    arrays, reopened = reopen_rss_vnext_uv_texture_atlas_v1(body)
    assert metadata == reopened
    assert metadata["neural_graphics_color_texel_count"] > 0
    assert metadata["inferred_surface_texture_completion_performed"] is True
    assert metadata["neural_graphics_rgb_is_inferred_not_observed"] is True
    assert metadata["neural_graphics_scene_content_sha256"] == "f" * 64
    assert metadata["neural_completion_never_overwrites_original_rgb_layers"] is True
    predicted = arrays["neural_rgb_candidate_rgba_u8"][
        arrays["neural_rgb_candidate_mask_u8"] == 1
    ]
    assert len(predicted) == metadata["neural_graphics_color_texel_count"]
    assert (predicted[:, 0] > 250).all()
    assert (predicted[:, 1] < 5).all()
    assert (predicted[:, 2] < 5).all()
    assert (predicted[:, 3] == 255).all()
    assert np.allclose(
        arrays["neural_rgb_log_uncertainty_f16"][
            arrays["neural_rgb_candidate_mask_u8"] == 1
        ], -2., atol=1.e-3,
    )
    source = arrays["texture_rgba_u8"]
    mask = arrays["texture_source_sample_mask_u8"]
    assert (source[mask == 1] == [180, 35, 15, 255]).all()
    assert metadata["source_3d_surface_ownership_verified"] is False


def test_neural_graphics_must_have_exact_tessa_vertex_domain_and_scene_binding():
    mesh, source = _fixture()
    for field, digest in [
        (np.zeros((2, 5), np.float32), "f" * 64),
        (np.zeros((3, 3), np.float32), "f" * 64),
        (np.full((3, 5), np.nan, np.float32), "f" * 64),
        (np.zeros((3, 5), np.float32), "not_sha"),
    ]:
        with pytest.raises(RuntimeError, match="NEURAL_GRAPHICS_FIELD_ABI"):
            compile_rss_vnext_uv_texture_atlas_v1(
                source_rgb_candidates=source, mesh=mesh,
                graphics_vertex_logits=field,
                graphics_scene_content_sha256=digest,
            )


def test_changing_only_neural_graphics_logits_changes_only_generated_texture_layer():
    mesh, source = _fixture()
    a = np.zeros((len(mesh.vertices), 5), np.float32)
    b = a.copy()
    b[:, :3] = [-7., 3., 8.]
    x, _ = compile_rss_vnext_uv_texture_atlas_v1(
        source_rgb_candidates=source, mesh=mesh,
        graphics_vertex_logits=a,
        graphics_scene_content_sha256="b" * 64,
    )
    y, _ = compile_rss_vnext_uv_texture_atlas_v1(
        source_rgb_candidates=source, mesh=mesh,
        graphics_vertex_logits=b,
        graphics_scene_content_sha256="b" * 64,
    )
    xdata, _ = reopen_rss_vnext_uv_texture_atlas_v1(x)
    ydata, _ = reopen_rss_vnext_uv_texture_atlas_v1(y)
    assert x != y
    assert np.array_equal(xdata["texture_rgba_u8"], ydata["texture_rgba_u8"])
    assert np.array_equal(
        xdata["texture_source_sample_mask_u8"],
        ydata["texture_source_sample_mask_u8"],
    )
    assert not np.array_equal(
        xdata["neural_rgb_candidate_rgba_u8"],
        ydata["neural_rgb_candidate_rgba_u8"],
    )
