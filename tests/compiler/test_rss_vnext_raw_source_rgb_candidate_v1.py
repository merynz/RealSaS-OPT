from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest

from compiler.rss_vnext.workers.rss_vnext_raw_source_rgb_candidate_v1 import (
    SCHEMA, MAX_SAMPLES_PER_VIEW, _sample_inferred_surface_pixels,
    build_raw_original_rgb_candidate_v1,
)
from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
    _observed_alpha, _project_fitted, admit_raw_two_view_mesh_v1,
)
from tests.compiler.test_rss_vnext_raw_mesh_admission_v1 import (
    _triangle_source_and_scene,
)


def _fixture():
    source, scene, receipt = _triangle_source_and_scene()
    digest = "c" * 64
    mesh, _, court = admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt, scene_digest=digest,
    )
    assert mesh is not None and court["source_static_silhouette_qualified"]
    return source, scene, receipt, digest, mesh


def test_both_original_rgb_images_preserved_as_observed_pixels_not_face_truth():
    source, scene, receipt, digest, mesh = _fixture()
    actual = build_raw_original_rgb_candidate_v1(
        source=source, scene=scene, receipt=receipt,
        scene_digest=digest, canonical_mesh=mesh,
    )
    assert actual["schema"] == SCHEMA
    assert actual["source_rgb_pixels_observed"] is True
    assert actual["source_rgb_to_canonical_face_ownership_qualified"] is False
    assert actual["original_camera_observed"] is False
    assert actual["texture_uv_qualified"] is False
    assert actual["source_appearance_release_qualified"] is False
    assert actual["product_revision_minted"] is False
    assert actual["sample_count"] > 0
    assert actual["sample_count"] <= 2 * MAX_SAMPLES_PER_VIEW
    assert len(actual["views"]) == 2
    for view, raw in zip(actual["views"], source.original_files):
        assert view["original_image_bytes_sha256"] == sha256(raw).hexdigest()
        assert view["raster_overlap_ambiguous_foreground_pixels"] == 0
        assert view["sampled_source_opaque_interior_count"] > 0
        assert view["original_source_rgb_pixels_observed"] is True
        assert view["projected_face_correspondence_is_original_source_truth"] is False
        assert view["camera_hypothesis"]["source_camera_was_observed"] is False
        for sample in view["samples"]:
            assert sample["rgba8_observed"] == [180, 35, 15, 255]
            assert sample["candidate_face_index"] == 0
            assert sample["candidate_face_owner_is_source_truth"] is False
            assert sum(sample["hypothesized_face_barycentric"]) == pytest.approx(1.)
            assert min(sample["hypothesized_face_barycentric"]) >= .06
    from compiler.realsas_compiler_core.hashing import content_sha256
    obj = dict(actual)
    claim = obj.pop("candidate_hash")
    assert claim == content_sha256(obj)


def test_overlapping_coplanar_faces_abstain_from_source_pixel_owner():
    source, scene, _, _, _ = _fixture()
    observed, blocker = _observed_alpha(source, 0)
    assert blocker is None
    xyz = scene["tessa_candidate_rest_vertices"][0]
    projected, _ = _project_fitted(xyz, observed, 0)
    rgba = np.zeros((*observed.shape, 4), dtype=np.uint8)
    rgba[..., 3] = observed.astype(np.uint8) * 255
    faces = np.repeat(scene["tessa_candidate_faces"], 2, axis=0)
    measured = _sample_inferred_surface_pixels(
        rgba, observed, projected, faces,
    )
    assert measured["raster_overlap_ambiguous_foreground_pixels"] > 0
    assert measured["sampled_source_opaque_interior_count"] == 0
    assert not measured["samples"]


def test_source_rgb_candidate_rejects_different_canonical_mesh_and_scene_lineage():
    source, scene, receipt, digest, mesh = _fixture()
    from dataclasses import replace
    with pytest.raises(RuntimeError, match="CANONICAL_STATIC_REPLAY_REQUIRED"):
        build_raw_original_rgb_candidate_v1(
            source=source, scene=scene, receipt=receipt,
            scene_digest=digest,
            canonical_mesh=replace(mesh, mesh_lineage_hash="f" * 64),
        )
    with pytest.raises(RuntimeError, match="SOURCE_OR_SCENE_BINDING"):
        build_raw_original_rgb_candidate_v1(
            source=source, scene=scene, receipt={
                **receipt, "source_binding_sha256": "e" * 64,
            },
            scene_digest=digest, canonical_mesh=mesh,
        )
