from __future__ import annotations

from dataclasses import replace
import io
import json

import numpy as np
from PIL import Image
import pytest

from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
    admit_raw_two_view_mesh_v1,
)
from compiler.rss_vnext.workers.rss_vnext_source_rgb_projection_candidate_v1 import (
    build_raw_source_rgb_image_plane_candidates_v1,
)
from compiler.rss_vnext.workers.rss_vnext_uv_texture_atlas_v1 import (
    _archive,
    compile_rss_vnext_uv_texture_atlas_v1,
    reopen_rss_vnext_uv_texture_atlas_v1,
)
from tests.compiler.test_rss_vnext_raw_mesh_admission_v1 import (
    _triangle_source_and_scene,
)


def _source_mesh_and_samples():
    source, scene, receipt = _triangle_source_and_scene()
    mesh, _, witness = admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt,
        scene_digest="c" * 64,
    )
    assert mesh is not None and witness["source_static_silhouette_qualified"]
    samples, _ = build_raw_source_rgb_image_plane_candidates_v1(
        source=source, mesh=mesh,
    )
    return source, mesh, samples


def test_canonical_mesh_faces_have_stable_uv_charts_and_original_texture_layers():
    _, mesh, samples = _source_mesh_and_samples()
    body, manifest = compile_rss_vnext_uv_texture_atlas_v1(
        source_rgb_candidates=samples, mesh=mesh,
    )
    body2, manifest2 = compile_rss_vnext_uv_texture_atlas_v1(
        source_rgb_candidates=samples, mesh=mesh,
    )
    assert body == body2 and manifest == manifest2
    arrays, reopened = reopen_rss_vnext_uv_texture_atlas_v1(body)
    assert manifest == reopened
    assert manifest["canonical_mesh_lineage_hash"] == mesh.mesh_lineage_hash
    assert manifest["face_count"] == len(mesh.faces) == 1
    assert arrays["face_chart_xywh_i32"].shape == (1, 4)
    assert arrays["face_vertex_uv_f32"].shape == (1, 3, 2)
    assert arrays["texture_rgba_u8"].shape[0] == 2
    assert arrays["texture_source_sample_mask_u8"].sum() > 30
    asserted_colors = arrays["texture_rgba_u8"][
        arrays["texture_source_sample_mask_u8"] == 1
    ]
    assert (asserted_colors == [180, 35, 15, 255]).all()
    assert not arrays["source_view_conflict_u8"].any()
    assert manifest["source_view_color_texel_counts"][0] > 0
    assert manifest["source_view_color_texel_counts"][1] > 0
    assert manifest["source_3d_surface_ownership_verified"] is False
    assert manifest["pose_conditioned_occlusion_qualified"] is False
    assert manifest["draw_order_is_not_texture_atlas_authority"] is True
    assert manifest["unfilled_texture_texels_are_unknown"] is True


def test_opposite_original_view_colors_are_preserved_without_silent_blend():
    source, mesh, _ = _source_mesh_and_samples()
    with Image.open(io.BytesIO(source.original_files[1])) as image:
        back = np.asarray(image.convert("RGBA")).copy()
    back[:, :, :3] = [25, 65, 195]
    out = io.BytesIO()
    Image.fromarray(back, "RGBA").save(out, "PNG")
    source = replace(source, original_files=(source.original_files[0], out.getvalue()))
    sample_data, _ = build_raw_source_rgb_image_plane_candidates_v1(
        source=source, mesh=mesh,
    )
    atlas, meta = compile_rss_vnext_uv_texture_atlas_v1(
        source_rgb_candidates=sample_data, mesh=mesh,
    )
    arrays, _ = reopen_rss_vnext_uv_texture_atlas_v1(atlas)
    texture = arrays["texture_rgba_u8"]
    masks = arrays["texture_source_sample_mask_u8"]
    assert (texture[0, masks[0] == 1] == [180, 35, 15, 255]).all()
    assert (texture[1, masks[1] == 1] == [25, 65, 195, 255]).all()
    assert meta["source_view_conflict_texel_count"] > 0
    assert (arrays["source_view_conflict_u8"] == 1).sum() > 0
    assert meta["inferred_surface_texture_completion_performed"] is False


def test_uv_reopen_does_not_accept_rehashed_fake_source_ownership():
    _, mesh, samples = _source_mesh_and_samples()
    body, _ = compile_rss_vnext_uv_texture_atlas_v1(
        source_rgb_candidates=samples, mesh=mesh,
    )
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    manifest = json.loads(arrays["manifest_utf8"].tobytes().decode("utf-8"))
    manifest["source_3d_surface_ownership_verified"] = True
    arrays["manifest_utf8"] = np.frombuffer(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
        dtype=np.uint8,
    ).copy()
    with pytest.raises(RuntimeError, match="FALSE_AUTHORITY"):
        reopen_rss_vnext_uv_texture_atlas_v1(_archive(arrays))


def test_texture_atlas_refuses_same_source_samples_with_different_mesh_hash():
    _, mesh, samples = _source_mesh_and_samples()
    mesh = replace(mesh, mesh_lineage_hash="f" * 64)
    with pytest.raises(RuntimeError, match="CANONICAL_MESH_HASH_DRIFT"):
        compile_rss_vnext_uv_texture_atlas_v1(
            source_rgb_candidates=samples, mesh=mesh,
        )



def test_large_generic_mesh_uv_atlas_degrades_texel_resolution_not_product_availability():
    from compiler.rss_vnext.workers.rss_vnext_uv_texture_atlas_v1 import (
        _packing, MAX_DIMENSION,
    )
    high_sample_count = 96 * 96
    tile, columns, width, height = _packing(4096, high_sample_count)
    assert tile == 32
    assert columns == 64
    assert (width, height) == (2048, 2048)
    assert width <= MAX_DIMENSION and height <= MAX_DIMENSION
    assert _packing(1, high_sample_count)[0] == 96
    assert _packing(100, 24 * 24)[0] == 24
