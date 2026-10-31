from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
    decode_source_observation_arrays_v1,
)
from tools.start_exact8_product_compile_v1 import (
    Exact8ProductError,
    VIEW_ORDER,
    materialize_exact8_source_observation_v1,
    validate_controlled_camera_bundle_v1,
)


def controlled_camera_bundle(*, half: float = 1.25, center=(0.2, -0.1, 0.8)) -> dict:
    center = np.asarray(center, dtype=np.float64)
    distance = max(4.0 * half, 1.0)
    cameras = []
    for index, yaw in enumerate(range(0, 360, 45)):
        theta = math.radians(float(yaw))
        radial = np.asarray([math.sin(theta), -math.cos(theta), 0.0], dtype=np.float64)
        forward = -radial
        right = np.asarray([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
        cameras.append({
            "schema_version": "RealSaS.FullSurfaceCameraProjection.v3",
            "view_id": f"V{index}",
            "view_index": index,
            "origin": (center + radial * distance).tolist(),
            "right": right.tolist(),
            "screen_up": [0.0, 0.0, 1.0],
            "forward": forward.tolist(),
            "half_extent": half,
            "resolution": 1024,
            "clip_start": 0.001,
            "clip_end": 1000.0,
        })
    return {
        "schema": "RealSaS.CameraProjectionBundle.v1",
        "view_order": list(VIEW_ORDER),
        "per_view_recenter": False,
        "per_view_pose_change": False,
        "common_center_xyz": center.tolist(),
        "common_half_extent": half,
        "cameras": cameras,
    }


def write_views(root: Path) -> None:
    for index in range(8):
        rgba = np.zeros((1024, 1024, 4), dtype=np.uint8)
        rgba[64:960, 96:928, 0] = 30 + index
        rgba[64:960, 96:928, 1] = 90
        rgba[64:960, 96:928, 2] = 150
        rgba[64:960, 96:928, 3] = 255
        Image.fromarray(rgba, mode="RGBA").save(root / f"V{index}.png", format="PNG")


class Exact8ProductCompileToolTest(unittest.TestCase):
    def test_materializes_codec_valid_observation_without_synthesis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_views(root)
            camera_path = root / "camera_bundle.json"
            camera_path.write_text(json.dumps(controlled_camera_bundle()), encoding="utf-8")

            payload, receipt = materialize_exact8_source_observation_v1(
                views_dir=root,
                camera_bundle_path=camera_path,
            )
            source = decode_source_observation_arrays_v1(payload)
            self.assertEqual(source.native_rgba.shape, (1, 8, 4, 1024, 1024))
            self.assertEqual(tuple(source.camera_bundle["view_order"]), VIEW_ORDER)
            self.assertEqual(len(receipt["views"]), 8)
            self.assertFalse(receipt["missing_views_synthesized"])
            self.assertFalse(receipt["camera_inferred"])
            self.assertTrue(all(row["border_margin_px"] >= 8 for row in receipt["views"]))

    def test_missing_view_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_views(root)
            (root / "V7.png").unlink()
            camera_path = root / "camera_bundle.json"
            camera_path.write_text(json.dumps(controlled_camera_bundle()), encoding="utf-8")
            with self.assertRaisesRegex(Exact8ProductError, "VIEW_MISSING:V7"):
                materialize_exact8_source_observation_v1(
                    views_dir=root,
                    camera_bundle_path=camera_path,
                )

    def test_camera_basis_drift_fails_closed(self):
        camera = controlled_camera_bundle()
        camera["cameras"][3]["right"] = [1.0, 0.0, 0.0]
        with self.assertRaisesRegex(Exact8ProductError, "CAMERA_RIGHT_DRIFT:V3"):
            validate_controlled_camera_bundle_v1(camera)

    def test_per_view_recenter_is_forbidden(self):
        camera = controlled_camera_bundle()
        camera["per_view_recenter"] = True
        with self.assertRaisesRegex(Exact8ProductError, "PER_VIEW_RECENTER_FORBIDDEN"):
            validate_controlled_camera_bundle_v1(camera)


if __name__ == "__main__":
    unittest.main()
