from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256
from tools.corpus.build_truth139_source_component_masks_v1 import build
from tools.corpus.validate_truth139_source_component_masks_v1 import validate


def _npz(path: Path, **arrays) -> None:
    np.savez_compressed(path, **arrays)


def test_subject_local_component_masks_preserve_visible_and_hidden_support(tmp_path: Path):
    geometry = tmp_path / "geometry.npz"
    _npz(
        geometry,
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        face_owner_privileged=np.asarray([10, 20], dtype=np.int32),
    )

    views = []
    # Three pixels: p=1 sees component 10 only; p=2 sees 10 then hidden 20;
    # p=3 sees component 20 only.  K=2 is enough and every stack is complete.
    for view in range(8):
        path = tmp_path / f"V{view}_peel.npz"
        _npz(
            path,
            pixel_linear_index=np.asarray([1, 2, 3], dtype=np.int32),
            hit_valid=np.asarray([[True, False], [True, True], [True, False]], dtype=np.bool_),
            hit_triangle_id=np.asarray([[0, -1], [0, 1], [1, -1]], dtype=np.int32),
            stack_complete=np.asarray([True, True, True], dtype=np.bool_),
        )
        views.append({"view_index": view, "path": str(path), "sha256": sha256(path)})

    layer_manifest = tmp_path / "layer.json"
    layer_value = {
        "schema": "RealSaS.Truth139GraphicsLayerPeelTeacher.v1",
        "status": "PASS_K_HIT_GEOMETRIC_LAYER_PEEL_CANDIDATE__CPU_PREFIX_COURT_REQUIRED__RGBA_COMPLETION_SEPARATE",
        "subject_id": "fixture-winged",
        "layer_peel_teacher_identity_sha256": "a" * 64,
        "parents": {
            "geometry_npz": {"path": str(geometry), "sha256": sha256(geometry)},
        },
        "views": views,
    }
    layer_manifest.write_text(json.dumps(layer_value, sort_keys=True) + "\n")

    result = build(layer_peel_teacher_manifest=layer_manifest, output_dir=tmp_path / "masks")
    assert result["component_count"] == 2
    assert result["component_identity_policy"].startswith("SUBJECT_LOCAL_PERMUTATION_INVARIANT")
    assert result["policy"]["source_owner_is_semantic_class"] is False
    assert result["policy"]["source_owner_is_cross_subject_identity"] is False
    assert result["aggregate"]["visible_pixel_count_by_component"] == [16, 8]
    assert result["aggregate"]["any_depth_pixel_membership_count_by_component"] == [16, 16]
    assert result["aggregate"]["hidden_only_pixel_membership_count_by_component"] == [0, 8]

    court = validate(result)
    assert court["status"] == "PASS_SOURCE_COMPONENT_MASKS_REBIND_EXACT_FACE_OWNER_AND_PEEL_TRUTH"
    assert court["component_count"] == 2
    assert court["counts"]["hidden_only_pixel_memberships"] == 8


def test_full_k_stack_is_preserved_as_overflow_unknown(tmp_path: Path):
    geometry = tmp_path / "geometry.npz"
    _npz(
        geometry,
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        face_owner_privileged=np.asarray([1, 2], dtype=np.int32),
    )
    views = []
    for view in range(8):
        path = tmp_path / f"V{view}_peel.npz"
        _npz(
            path,
            pixel_linear_index=np.asarray([5], dtype=np.int32),
            hit_valid=np.asarray([[True, True]], dtype=np.bool_),
            hit_triangle_id=np.asarray([[0, 1]], dtype=np.int32),
            stack_complete=np.asarray([False], dtype=np.bool_),
        )
        views.append({"view_index": view, "path": str(path), "sha256": sha256(path)})
    layer_manifest = tmp_path / "layer.json"
    layer_manifest.write_text(json.dumps({
        "schema": "RealSaS.Truth139GraphicsLayerPeelTeacher.v1",
        "status": "PASS_K_HIT_GEOMETRIC_LAYER_PEEL_CANDIDATE__CPU_PREFIX_COURT_REQUIRED__RGBA_COMPLETION_SEPARATE",
        "subject_id": "fixture-overflow",
        "layer_peel_teacher_identity_sha256": "b" * 64,
        "parents": {"geometry_npz": {"path": str(geometry), "sha256": sha256(geometry)}},
        "views": views,
    }, sort_keys=True) + "\n")

    result = build(layer_peel_teacher_manifest=layer_manifest, output_dir=tmp_path / "masks")
    assert all(view["overflow_unknown_pixel_count"] == 1 for view in result["views"])
    assert result["policy"]["absence_beyond_measured_k_prefix_is_negative_label"] is False
    court = validate(result)
    assert all(view["overflow_unknown_pixel_count"] == 1 for view in court["view_courts"])
