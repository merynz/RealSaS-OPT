from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from compiler.realsas_compiler_core.types import QualifiedJoint, RiggingSurfaceIR, SurfaceNode
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
from experiments.single_family_e2e_v1.data_manifest_v1 import build_master_family_manifest_v1
from experiments.single_family_e2e_v1.mechanical_truth_adapter_v1 import align_mechanical_skin_truth_v1


def _camera(view: int) -> dict:
    t = math.radians(45.0 * view)
    return {
        "contract": "realsas.level_orthographic_z_orbit.v1",
        "yaw_deg": 45.0 * view,
        "half_extent": 0.54,
        "forward": [math.sin(t), math.cos(t), 0.0],
        "right": [math.cos(t), -math.sin(t), 0.0],
        "screen_up": [0.0, 0.0, 1.0],
    }


def _write_master_asset(root: Path, image_filename: str = "observation.png") -> Path:
    asset = root / "asset_test"
    asset.mkdir(parents=True)
    np.savez(asset / "primary_geometry.npz", vertices=np.asarray([[0,0,0],[1,0,0],[0,1,0]], np.float32), faces=np.asarray([[0,1,2]], np.int64))
    for view in range(8):
        vd = asset / "renders" / f"V{view}"
        vd.mkdir(parents=True)
        Image.new("RGBA", (1024, 1024), (view, 2 * view, 3 * view, 255)).save(vd / image_filename)
        (vd / "camera.json").write_text(json.dumps(_camera(view), sort_keys=True), encoding="utf-8")
        np.savez(
            vd / "raster_authority.npz",
            pixel_linear_index=np.asarray([view], np.int64),
            triangle_id=np.asarray([0], np.int64),
            barycentric_uv=np.asarray([[0.2, 0.3]], np.float32),
            resolution=np.asarray([1024, 1024], np.int64),
        )
    return asset


def test_master_manifest_is_explicit_hash_bound_and_detects_dirty_member(tmp_path):
    asset = _write_master_asset(tmp_path)
    manifest = build_master_family_manifest_v1(asset, image_filename="observation.png")
    assert manifest.asset_id == "asset_test"
    assert len(manifest.views) == 8
    assert len(manifest.file_hashes) == 25
    assert all(view.resolution == (1024, 1024) for view in manifest.views)
    expected = dict(manifest.file_hashes)
    again = build_master_family_manifest_v1(
        asset,
        image_filename="observation.png",
        expected_file_hashes=expected,
        expected_manifest_hash=manifest.manifest_hash,
    )
    assert again.manifest_hash == manifest.manifest_hash
    camera = asset / "renders" / "V3" / "camera.json"
    camera.write_text(camera.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MASTER_MEMBER_HASH_DRIFT"):
        build_master_family_manifest_v1(asset, image_filename="observation.png", expected_file_hashes=expected)


def test_manifest_never_silently_chooses_image_authority(tmp_path):
    asset = _write_master_asset(tmp_path, image_filename="textured.png")
    with pytest.raises(FileNotFoundError, match="MASTER_MEMBER_MISSING"):
        build_master_family_manifest_v1(asset, image_filename="cel_clean.png")
    manifest = build_master_family_manifest_v1(asset, image_filename="textured.png")
    assert manifest.image_filename == "textured.png"


def _surface() -> RiggingSurfaceIR:
    return RiggingSurfaceIR(
        (
            SurfaceNode("S:0", (0.0,0.0,0.0), (0,), ("o0",), ("o0",)),
            SurfaceNode("S:1", (1.0,0.0,0.0), (1,), ("o1",), ("o1",)),
        ),
        geometry_lineage_hash="surface-hash",
    )


def _skeleton(lineage: str = "skeleton-hash") -> QualifiedSkeletonIRV2:
    joints = (
        QualifiedJoint("J:A", (0.0,0.0,0.0), None, ("S:0",), "P:anon:0"),
        QualifiedJoint("J:B", (1.0,0.0,0.0), "J:A", ("S:1",), "P:anon:1"),
    )
    return QualifiedSkeletonIRV2(joints, ("J:A",), {"deforming": False}, {"status": "PASS"}, lineage)


def test_mechanical_truth_rebind_uses_only_current_s_and_qualified_g_lineage():
    truth = align_mechanical_skin_truth_v1(
        _surface(),
        _skeleton(),
        teacher_surface_ids=("S:1", "S:0"),
        anonymous_control_ids=("P:extra", "P:anon:1", "P:anon:0"),
        dense_weights=np.asarray([
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
        ], dtype=np.float32),
    )
    truth.validate()
    assert truth.surface_ids == ("S:0", "S:1")
    assert truth.canonical_joint_ids == ("J:A", "J:B")
    np.testing.assert_allclose(truth.weights.sum(axis=1), 1.0, atol=1e-6)
    assert truth.surface_binding_hash == "surface-hash"
    assert truth.skeleton_binding_hash == "skeleton-hash"
    assert not hasattr(truth, "anonymous_control_ids")
    assert "bone" not in repr(truth).lower()
    changed = align_mechanical_skin_truth_v1(
        _surface(),
        _skeleton("other-skeleton-hash"),
        teacher_surface_ids=("S:1", "S:0"),
        anonymous_control_ids=("P:extra", "P:anon:1", "P:anon:0"),
        dense_weights=np.asarray([[0.20,0.60,0.20],[0.10,0.20,0.70]], dtype=np.float32),
    )
    assert changed.truth_lineage_hash != truth.truth_lineage_hash


def test_mechanical_truth_rebind_fails_closed_on_unmapped_qualified_joint():
    joints = (
        QualifiedJoint("J:A", (0.0,0.0,0.0), None, ("S:0",), "P:missing"),
    )
    skeleton = QualifiedSkeletonIRV2(joints, ("J:A",), {"deforming": False}, {"status": "PASS"}, "g")
    with pytest.raises(ValueError, match="does not cover qualified anonymous controls"):
        align_mechanical_skin_truth_v1(
            _surface(),
            skeleton,
            teacher_surface_ids=("S:0", "S:1"),
            anonymous_control_ids=("P:anon:0",),
            dense_weights=np.asarray([[1.0],[1.0]], dtype=np.float32),
        )
