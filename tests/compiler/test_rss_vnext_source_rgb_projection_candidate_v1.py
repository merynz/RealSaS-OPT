from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
    admit_raw_two_view_mesh_v1,
)
from compiler.rss_vnext.workers.rss_vnext_source_rgb_projection_candidate_v1 import (
    _owner_map,
    build_raw_source_rgb_image_plane_candidates_v1,
    decode_raw_source_rgb_image_plane_candidates_v1,
)
from tests.compiler.test_rss_vnext_raw_mesh_admission_v1 import (
    _triangle_source_and_scene,
)


def test_original_rgb_is_byte_exact_on_unambiguous_fitted_face_interior():
    source, scene, receipt = _triangle_source_and_scene()
    mesh, _, court = admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt,
        scene_digest="c" * 64,
    )
    assert mesh is not None and court["source_static_silhouette_qualified"]
    first, manifest = build_raw_source_rgb_image_plane_candidates_v1(
        source=source, mesh=mesh,
    )
    repeat, repeated_manifest = build_raw_source_rgb_image_plane_candidates_v1(
        source=source, mesh=mesh,
    )
    assert first == repeat
    assert manifest == repeated_manifest
    arrays, opened = decode_raw_source_rgb_image_plane_candidates_v1(first)
    assert opened == manifest
    assert manifest["sample_count"] > 50
    assert manifest["sample_count"] == len(arrays["source_rgba_u8"])
    assert (arrays["source_rgba_u8"] == [180, 35, 15, 255]).all()
    assert (arrays["face_index_i32"] == 0).all()
    assert set(arrays["view_index_u8"]) == {0, 1}
    assert np.allclose(arrays["barycentric_f32"].sum(axis=1), 1, atol=5e-5)
    assert manifest["source_original_rgb_observation"]
    assert manifest["rgb_values_are_nearest_resampled_original_pixels"]
    assert manifest["source_rgb_surface_ownership_qualified"] is False
    assert manifest["pose_conditioned_source_visibility_qualified"] is False
    assert manifest["source_camera_observed"] is False
    assert manifest["native_source_rgb_render_parity_qualified"] is False
    assert all(
        view["fitted_overlap_ambiguous_pixel_count"] == 0
        for view in manifest["view_summaries"]
    )


def test_two_projected_faces_share_rays_and_have_no_source_color_owner():
    points = np.asarray([[4, 4], [25, 4], [4, 25]], dtype=np.float64)
    single = np.asarray([[0, 1, 2]], dtype=np.int32)
    doubled = np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int32)
    one, bary = _owner_map(points, single, width=32, height=32)
    two, _ = _owner_map(points, doubled, width=32, height=32)
    assert (one == 0).sum() > 10
    assert (two == -2).sum() == (one == 0).sum()
    assert (two >= 0).sum() == 0
    assert np.isfinite(bary).all()


def test_reopen_refuses_rehashed_false_source_authority():
    source, scene, receipt = _triangle_source_and_scene()
    mesh, _, _ = admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt,
        scene_digest="c" * 64,
    )
    raw, manifest = build_raw_source_rgb_image_plane_candidates_v1(
        source=source, mesh=mesh,
    )
    import io
    from compiler.rss_vnext.workers.rss_vnext_source_rgb_projection_candidate_v1 import (
        _encode_npz_deterministic,
    )
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    import json
    manifest["source_rgb_surface_ownership_qualified"] = True
    arrays["manifest_utf8"] = np.frombuffer(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    ).copy()
    forged = _encode_npz_deterministic(arrays)
    with pytest.raises(RuntimeError, match="SOURCE_RGB_REOPEN_TYPED_OR_AUTHORITY_INVALID"):
        decode_raw_source_rgb_image_plane_candidates_v1(forged)


def test_missing_real_alpha_is_not_silently_promoted_to_rgb_surface_support():
    source, scene, receipt = _triangle_source_and_scene()
    mesh, _, _ = admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt,
        scene_digest="c" * 64,
    )
    changed = replace(source, view_claims=("UNKNOWN", "UNKNOWN"))
    with pytest.raises(RuntimeError, match="FRONT_BACK_CLAIMS_REQUIRED"):
        build_raw_source_rgb_image_plane_candidates_v1(
            source=changed, mesh=mesh,
        )
