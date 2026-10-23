from __future__ import annotations

import numpy as np
import torch

from compiler.realsas_compiler_services.platform_worker.input_capability_v1 import (
    SOURCE_OBSERVATION_SCHEMA_V1,
    decode_source_observation_v1,
    encode_source_observation_v1,
)
from compiler.realsas_compiler_services.platform_worker.scene_input_contract_v1 import (
    SHAPE_INPUT_SCHEMA_V1,
    decode_shape_input_v1,
    encode_shape_input_v1,
)
from models.iris.v2.dinov2_foundation_v2 import DINOv2FoundationAuthorityV2
from models.iris.vnext.inference_input_v1 import IRISVNextSceneInputsV1


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
        "view_order": [f"V{i}" for i in range(8)],
        "common_center_xyz": [0.0, 0.0, 0.0],
        "common_half_extent": 1.0,
        "per_view_recenter": False,
        "per_view_pose_change": False,
        "cameras": rows,
    }


def test_source_observation_v1_roundtrip_is_exact_and_versioned():
    rgba = np.zeros((8, 4, 1024, 1024), dtype=np.uint8)
    rgba[:, 3] = 255
    payload = encode_source_observation_v1(native_rgba=rgba, camera_bundle=_camera_bundle())
    decoded = decode_source_observation_v1(payload)
    assert decoded.schema == SOURCE_OBSERVATION_SCHEMA_V1
    assert tuple(decoded.native_rgba.shape) == (1, 8, 4, 1024, 1024)
    assert decoded.native_rgba.dtype == torch.uint8
    assert torch.equal(decoded.native_rgba[0], torch.from_numpy(rgba))
    assert decoded.camera_bundle["view_order"] == [f"V{i}" for i in range(8)]


def test_shape_input_transport_is_neutral_and_pickle_free():
    authority = DINOv2FoundationAuthorityV2().authority_hash
    value = IRISVNextSceneInputsV1(
        image_tokens=torch.zeros((1, 8, 1369, 384), dtype=torch.float32),
        token_xy=torch.zeros((1, 8, 1369, 2), dtype=torch.float32),
        camera_features=torch.zeros((1, 8, 13), dtype=torch.float32),
        foundation_authority_sha256=authority,
    )
    payload = encode_shape_input_v1(value)
    decoded = decode_shape_input_v1(payload)
    assert SHAPE_INPUT_SCHEMA_V1 == "RealSaS.ShapeInput.v1"
    assert decoded.foundation_authority_sha256 == authority
    assert torch.equal(decoded.image_tokens, value.image_tokens)
    assert torch.equal(decoded.token_xy, value.token_xy)
    assert torch.equal(decoded.camera_features, value.camera_features)


def test_source_v1_view_count_is_schema_local_not_silently_flexible():
    bad = np.zeros((1, 4, 1024, 1024), dtype=np.uint8)
    try:
        encode_source_observation_v1(native_rgba=bad, camera_bundle=_camera_bundle())
    except ValueError as exc:
        assert "PRODUCT_INPUT_V1_RGBA_SHAPE_INVALID" in str(exc)
    else:
        raise AssertionError("v1 source schema must fail closed on wrong view count")
