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


def test_same_scene_graphics_and_semantic_heads_have_distinct_visible_consequences():
    """Changing ONLY Semantic changes candidate face-layer selection, never
    original RGB, neural RGB, UV topology, or source ownership authority."""
    _, mesh, samples = _source_mesh_and_samples()
    n = len(mesh.vertices)
    graphics = np.tile(np.array([0.1, -0.4, 0.7, -1.0, 0.2]), (n, 1))
    semantic0 = np.tile(np.array([8.0, -8.0, -8.0, -8.0]), (n, 1))
    semantic2 = np.tile(np.array([-8.0, -8.0, 8.0, -8.0]), (n, 1))
    kwargs = {
        "source_rgb_candidates": samples,
        "mesh": mesh,
        "graphics_vertex_logits": graphics,
        "graphics_scene_content_sha256": "a" * 64,
        "presentation_role_schema_id":
            "RSS.vNext.SemanticPresentationRoles.FourWayResearch.v1",
    }
    old_atlas,old_meta = compile_rss_vnext_uv_texture_atlas_v1(
        **kwargs, presentation_role_vertex_logits=semantic0,
    )
    new_atlas,new_meta = compile_rss_vnext_uv_texture_atlas_v1(
        **kwargs, presentation_role_vertex_logits=semantic2,
    )
    old,old_receipt = reopen_rss_vnext_uv_texture_atlas_v1(old_atlas)
    new,new_receipt = reopen_rss_vnext_uv_texture_atlas_v1(new_atlas)
    assert old_meta == old_receipt and new_meta == new_receipt
    assert old_meta["schema"].endswith("v2")
    assert old["face_presentation_role_primary_u8"].tolist() == [0]
    assert new["face_presentation_role_primary_u8"].tolist() == [2]
    assert old_meta["semantic_role_sublayer_is_model_candidate_not_source_truth"]
    assert old_meta["source_3d_surface_ownership_verified"] is False
    assert old_meta["pose_conditioned_occlusion_qualified"] is False
    assert old_meta["neural_graphics_rgb_is_inferred_not_observed"]
    for key in (
        "texture_rgba_u8", "texture_source_sample_mask_u8",
        "source_view_conflict_u8", "neural_rgb_candidate_rgba_u8",
        "neural_rgb_candidate_mask_u8", "neural_rgb_log_uncertainty_f16",
        "face_vertex_uv_f32", "face_chart_xywh_i32",
    ):
        assert np.array_equal(old[key],new[key]),key
    assert old_meta["neural_graphics_logits_sha256"] == new_meta[
        "neural_graphics_logits_sha256"
    ]
    assert old_meta["semantic_face_probability_sha256"] != new_meta[
        "semantic_face_probability_sha256"
    ]
    assert old_atlas != new_atlas


def test_semantic_source_unknown_abstention_and_rehashed_face_label_tamper():
    _, mesh, samples = _source_mesh_and_samples()
    n = len(mesh.vertices)
    graphics = np.zeros((n,5),dtype=np.float64)
    # Uniform role likelihood cannot justify any *single* drawable role.
    uniform = np.zeros((n,4),dtype=np.float64)
    atlas,meta = compile_rss_vnext_uv_texture_atlas_v1(
        source_rgb_candidates=samples,mesh=mesh,
        graphics_vertex_logits=graphics,
        graphics_scene_content_sha256="a"*64,
        presentation_role_vertex_logits=uniform,
        presentation_role_schema_id="RSS.FourWayResearch.v1",
    )
    arrays, receipt = reopen_rss_vnext_uv_texture_atlas_v1(atlas)
    assert arrays["face_presentation_role_primary_u8"].tolist()==[255]
    assert meta["semantic_face_abstention_count"]==1
    assert meta["semantic_role_sublayer_never_overrides_original_rgba"] is True
    # The tamperer rearchives and recomputes the archive CAS, but the
    # renderer-facing role must be causally determined by the K probabilities.
    forged={key:value.copy() for key,value in arrays.items()}
    forged["face_presentation_role_primary_u8"][0]=0
    with pytest.raises(RuntimeError,match="SEMANTIC_ROLE_SELECTION_DRIFT"):
        reopen_rss_vnext_uv_texture_atlas_v1(_archive(forged))
    forged={key:value.copy() for key,value in arrays.items()}
    receipt=dict(receipt, semantic_role_sublayer_never_overrides_original_rgba=False)
    forged["manifest_utf8"]=np.frombuffer(
        json.dumps(receipt,sort_keys=True,separators=(",",":")).encode(),dtype=np.uint8,
    ).copy()
    with pytest.raises(RuntimeError,match="FALSE_AUTHORITY"):
        reopen_rss_vnext_uv_texture_atlas_v1(_archive(forged))


def test_semantic_cannot_be_unbound_from_graphics_scene_or_role_schema():
    _, mesh, samples = _source_mesh_and_samples()
    n=len(mesh.vertices)
    with pytest.raises(RuntimeError,match="SEMANTIC_GRAPHICS_SHARED_H_REQUIRED"):
        compile_rss_vnext_uv_texture_atlas_v1(
            source_rgb_candidates=samples,mesh=mesh,
            presentation_role_vertex_logits=np.zeros((n,4)),
            presentation_role_schema_id="roles.v1",
        )
    with pytest.raises(RuntimeError,match="SEMANTIC_VERTEX_ABI"):
        compile_rss_vnext_uv_texture_atlas_v1(
            source_rgb_candidates=samples,mesh=mesh,
            graphics_vertex_logits=np.zeros((n,5)),
            graphics_scene_content_sha256="a"*64,
            presentation_role_vertex_logits=np.full((n,4),np.nan),
            presentation_role_schema_id="roles.v1",
        )


def test_same_h_semantic_roles_change_face_sublayer_without_changing_rgb():
    _, mesh, samples = _source_mesh_and_samples()
    n = len(mesh.vertices)
    graphics = np.zeros((n, 5), dtype=np.float32)
    graphics[:, :3] = [2., -1., 0.25]
    role0 = np.tile(
        np.asarray([[9., -9., -9., -9.]], dtype=np.float32), (n, 1)
    )
    role2 = np.tile(
        np.asarray([[-9., -9., 9., -9.]], dtype=np.float32), (n, 1)
    )
    kwargs = {
        "source_rgb_candidates":samples,
        "mesh":mesh,
        "graphics_vertex_logits":graphics,
        "graphics_scene_content_sha256":"a" * 64,
        "presentation_role_schema_id":
            "RSS.vNext.SemanticPresentationRoles.FourWayResearch.v1",
    }
    atlas0, meta0 = compile_rss_vnext_uv_texture_atlas_v1(
        **kwargs, presentation_role_vertex_logits=role0,
    )
    atlas2, meta2 = compile_rss_vnext_uv_texture_atlas_v1(
        **kwargs, presentation_role_vertex_logits=role2,
    )
    a, _ = reopen_rss_vnext_uv_texture_atlas_v1(atlas0)
    b, _ = reopen_rss_vnext_uv_texture_atlas_v1(atlas2)
    assert not np.array_equal(
        a["face_presentation_role_primary_u8"],
        b["face_presentation_role_primary_u8"],
    )
    assert a["face_presentation_role_primary_u8"].tolist() == [0]
    assert b["face_presentation_role_primary_u8"].tolist() == [2]
    assert not np.array_equal(
        a["face_presentation_role_probabilities_f32"],
        b["face_presentation_role_probabilities_f32"],
    )
    # Semantic controls candidate strata, NEVER source pixels, inferred RGB,
    # atlas chart topology, alpha, appearance or source 3D ownership.
    for field in (
        "texture_rgba_u8", "texture_source_sample_mask_u8",
        "source_view_conflict_u8", "neural_rgb_candidate_rgba_u8",
        "neural_rgb_candidate_mask_u8", "face_vertex_uv_f32",
    ):
        assert np.array_equal(a[field], b[field]), field
    assert meta0["semantic_role_sublayer_is_model_candidate_not_source_truth"]
    assert meta2["source_3d_surface_ownership_verified"] is False
    assert meta2["pose_conditioned_occlusion_qualified"] is False
    assert meta2["product_revision_minted"] is False
    assert atlas0 != atlas2
    assert atlas0 == compile_rss_vnext_uv_texture_atlas_v1(
        **kwargs, presentation_role_vertex_logits=role0,
    )[0]


def test_same_h_semantic_role_abstention_cannot_be_promoted_by_rehashed_cas():
    _, mesh, samples = _source_mesh_and_samples()
    n=len(mesh.vertices)
    graphics=np.zeros((n, 5),dtype=np.float32)
    uniform=np.zeros((n, 4),dtype=np.float32)
    atlas, meta=compile_rss_vnext_uv_texture_atlas_v1(
        source_rgb_candidates=samples, mesh=mesh,
        graphics_vertex_logits=graphics,
        graphics_scene_content_sha256="b"*64,
        presentation_role_vertex_logits=uniform,
        presentation_role_schema_id=
            "RSS.vNext.SemanticPresentationRoles.FourWayResearch.v1",
    )
    arrays, reopened=reopen_rss_vnext_uv_texture_atlas_v1(atlas)
    assert arrays["face_presentation_role_primary_u8"].tolist()==[255]
    assert reopened["semantic_face_abstention_count"]==1
    # Tamper with the role assignment while preserving all binary/receipt
    # framing: the decoder must independently recompute the candidate.
    arrays["face_presentation_role_primary_u8"][0]=1
    with pytest.raises(RuntimeError,match="SEMANTIC_ROLE_SELECTION_DRIFT"):
        reopen_rss_vnext_uv_texture_atlas_v1(_archive(arrays))
    # Even a rehashed manifest cannot manufacture source ownership.
    arrays["face_presentation_role_primary_u8"][0]=255
    fake=json.loads(arrays["manifest_utf8"].tobytes())
    fake["source_3d_surface_ownership_verified"]=True
    arrays["manifest_utf8"]=np.frombuffer(
        json.dumps(fake,sort_keys=True,separators=(",",":")).encode(),
        dtype=np.uint8,
    ).copy()
    with pytest.raises(RuntimeError,match="FALSE_AUTHORITY"):
        reopen_rss_vnext_uv_texture_atlas_v1(_archive(arrays))


def test_semantic_role_field_requires_same_h_graphics_and_source_lineage():
    _, mesh, samples=_source_mesh_and_samples()
    roles=np.ones((len(mesh.vertices),4),dtype=np.float32)
    with pytest.raises(RuntimeError,match="SEMANTIC_GRAPHICS_SHARED_H_REQUIRED"):
        compile_rss_vnext_uv_texture_atlas_v1(
            source_rgb_candidates=samples,mesh=mesh,
            presentation_role_vertex_logits=roles,
            presentation_role_schema_id="role-v1",
        )
    with pytest.raises(RuntimeError,match="SEMANTIC_VERTEX_ABI"):
        compile_rss_vnext_uv_texture_atlas_v1(
            source_rgb_candidates=samples,mesh=mesh,
            graphics_vertex_logits=np.ones((len(mesh.vertices),5)),
            graphics_scene_content_sha256="a"*64,
            presentation_role_vertex_logits=np.ones(
                (len(mesh.vertices)-1,4),dtype=np.float32,
            ),
            presentation_role_schema_id="role-v1",
        )
