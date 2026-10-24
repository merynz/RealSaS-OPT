from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
    decode_source_observation_arrays_v1,
)
from compiler.realsas_compiler_services.platform_worker.source_observation_materializer_v1 import (
    materialize_source_observation_from_manifest_v1,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _camera_bundle() -> dict:
    cameras = []
    for view in range(8):
        cameras.append(
            {
                "view_index": view,
                "origin": [float(view), 0.0, 2.0],
                "right": [1.0, 0.0, 0.0],
                "screen_up": [0.0, 0.0, 1.0],
                "forward": [0.0, 1.0, 0.0],
                "half_extent": 2.0,
                "resolution": 1024,
            }
        )
    return {
        "schema": "RealSaS.CameraProjectionBundle.v1",
        "view_order": [f"V{view}" for view in range(8)],
        "common_center_xyz": [0.0, 0.0, 0.0],
        "common_half_extent": 2.0,
        "per_view_recenter": False,
        "per_view_pose_change": False,
        "cameras": cameras,
    }


def _manifest(tmp_path: Path) -> dict:
    camera_path = tmp_path / "camera.json"
    camera_path.write_text(json.dumps(_camera_bundle(), sort_keys=True), encoding="utf-8")
    rasters = []
    masks = []
    for view in range(8):
        rgba = np.zeros((1024, 1024, 4), dtype=np.uint8)
        rgba[100 + view : 120 + view, 200:240, :3] = 25 + view
        rgba[100 + view : 120 + view, 200:240, 3] = 255
        raster_path = tmp_path / f"view_{view}.png"
        Image.fromarray(rgba, mode="RGBA").save(raster_path)

        mask = (rgba[:, :, 3] > 0).astype(np.uint8)
        mask_path = tmp_path / f"mask_{view}.bin"
        mask_path.write_bytes(mask.tobytes(order="C"))
        rasters.append(
            {
                "view_index": view,
                "image": {"path": str(raster_path), "sha256": _sha(raster_path)},
            }
        )
        masks.append(
            {
                "view_index": view,
                "mask": {"path": str(mask_path), "sha256": _sha(mask_path)},
            }
        )
    return {
        "run_id": "source-observation-test",
        "observation": {
            "camera_bundle": {"path": str(camera_path), "sha256": _sha(camera_path)},
            "source_rasters": rasters,
            "source_foreground_masks": masks,
        },
    }


def test_materializer_packages_exact_pixels_camera_and_foreground_authority(tmp_path: Path):
    manifest = _manifest(tmp_path)
    result = materialize_source_observation_from_manifest_v1(manifest)
    decoded = decode_source_observation_arrays_v1(result.payload)

    assert decoded.native_rgba.shape == (1, 8, 4, 1024, 1024)
    assert decoded.native_rgba.dtype == np.uint8
    assert decoded.camera_bundle == _camera_bundle()
    assert result.diagnostics["foreground_authority_check"] == "PASS_ALPHA_GT_ZERO_EQUALS_EXACT_SOURCE_MASK"
    assert result.diagnostics["subject_specific_logic"] is False
    assert result.diagnostics["model_inference_executed"] is False
    assert result.diagnostics["qualification_minted"] is False
    assert hashlib.sha256(result.payload).hexdigest() == result.diagnostics["output_content_sha256"]


def test_materializer_fails_closed_when_alpha_and_foreground_authority_disagree(tmp_path: Path):
    manifest = _manifest(tmp_path)
    row = manifest["observation"]["source_foreground_masks"][3]
    mask_path = Path(row["mask"]["path"])
    mask = bytearray(mask_path.read_bytes())
    mask[0] = 1
    mask_path.write_bytes(mask)
    row["mask"]["sha256"] = _sha(mask_path)

    with pytest.raises(RuntimeError, match="SOURCE_OBSERVATION_V1_FOREGROUND_AUTHORITY_DRIFT:3"):
        materialize_source_observation_from_manifest_v1(manifest)
