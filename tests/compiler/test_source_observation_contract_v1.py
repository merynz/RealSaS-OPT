from __future__ import annotations

import hashlib

import numpy as np
import pytest

from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
    decode_source_observation_arrays_v1,
    derive_foreground_masks_v1,
    encode_source_observation_arrays_v1,
)


def _camera_bundle():
    rows = []
    for index in range(8):
        rows.append(
            {
                "origin": [0.0, 0.0, 2.0],
                "right": [1.0, 0.0, 0.0],
                "screen_up": [0.0, 1.0, 0.0],
                "forward": [0.0, 0.0, -1.0],
                "half_extent": 1.0,
            }
        )
    return {
        "schema": "RealSaS.CameraProjectionBundle.v1",
        "view_order": [f"V{index}" for index in range(8)],
        "common_center_xyz": [0.0, 0.0, 0.0],
        "common_half_extent": 1.0,
        "per_view_recenter": False,
        "per_view_pose_change": False,
        "cameras": rows,
    }


def test_source_observation_roundtrip_and_alpha_mask_hash_binding():
    rgba = np.zeros((1, 8, 4, 1024, 1024), dtype=np.uint8)
    for view in range(8):
        rgba[0, view, 3, view : view + 2, view : view + 3] = 255
    payload = encode_source_observation_arrays_v1(
        native_rgba=rgba, camera_bundle=_camera_bundle()
    )
    decoded = decode_source_observation_arrays_v1(payload)
    expected = {}
    for view in range(8):
        raw = (rgba[0, view, 3] > 0).astype(np.uint8).tobytes(order="C")
        expected[view] = hashlib.sha256(raw).hexdigest()
    masks = derive_foreground_masks_v1(decoded, expected_sha256_by_view=expected)
    assert set(masks) == set(range(8))
    assert all(hashlib.sha256(masks[view]).hexdigest() == expected[view] for view in range(8))


def test_source_observation_mask_materialization_fails_closed_on_authority_drift():
    rgba = np.zeros((1, 8, 4, 1024, 1024), dtype=np.uint8)
    payload = encode_source_observation_arrays_v1(
        native_rgba=rgba, camera_bundle=_camera_bundle()
    )
    decoded = decode_source_observation_arrays_v1(payload)
    empty_sha = hashlib.sha256(bytes(1024 * 1024)).hexdigest()
    expected = {view: empty_sha for view in range(8)}
    expected[3] = "f" * 64
    with pytest.raises(RuntimeError, match="FOREGROUND_AUTHORITY_DRIFT:3"):
        derive_foreground_masks_v1(decoded, expected_sha256_by_view=expected)
