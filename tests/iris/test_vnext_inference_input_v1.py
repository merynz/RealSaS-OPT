from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.iris.v2.dinov2_foundation_v2 import (
    DINOv2FoundationAuthorityV2,
    DINOv2LoadedAuthorityV2,
)
from models.iris.vnext import inference_input_v1 as bridge
from models.iris.vnext.inference_input_v1 import (
    PATCH_TOKEN_COUNT_V1,
    camera_features_v1,
    dino_patch_map_to_tokens_v1,
    rgba_to_white_composited_rgb_v1,
    token_xy_v1,
)


def _camera_bundle():
    rows = []
    for index in range(8):
        rows.append(
            {
                "origin": [10.0 + index, 20.0, 30.0],
                "right": [1.0, 0.0, 0.0],
                "screen_up": [0.0, 1.0, 0.0],
                "forward": [0.0, 0.0, -1.0],
                "half_extent": 4.0 + index,
            }
        )
    return {
        "common_center_xyz": [10.0, 20.0, 30.0],
        "common_half_extent": 2.0,
        "cameras": rows,
    }


def _loaded_authority() -> DINOv2LoadedAuthorityV2:
    authority = DINOv2FoundationAuthorityV2()
    return DINOv2LoadedAuthorityV2(
        authority=authority,
        source_dir="/verified/dinov2",
        weight_path="/verified/dinov2_vits14.pth",
        source_head=authority.source_revision,
        weight_sha256=authority.weight_sha256,
        feature_contract_hash=authority.feature_contract().contract_hash,
    )


def test_token_xy_matches_fit8_patch_center_contract():
    value = token_xy_v1(2)
    assert value.shape == (2, 8, PATCH_TOKEN_COUNT_V1, 2)
    first = -1.0 + 1.0 / 37.0
    last = 1.0 - 1.0 / 37.0
    assert value[0, 0, 0].tolist() == pytest.approx([first, first])
    assert value[0, 0, 36].tolist() == pytest.approx([last, first])
    assert value[0, 0, 37].tolist() == pytest.approx([first, first + 2.0 / 37.0])
    assert value[0, 0, -1].tolist() == pytest.approx([last, last])
    assert torch.equal(value[0], value[1])


def test_camera_features_match_fit8_normalized_frame_contract():
    value = camera_features_v1(_camera_bundle(), batch_size=2)
    assert value.shape == (2, 8, 13)
    assert value[0, 0].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 2.0]
    )
    assert value[0, 3, :3].tolist() == pytest.approx([1.5, 0.0, 0.0])
    assert float(value[0, 3, 12]) == pytest.approx(3.5)
    assert torch.equal(value[0], value[1])


def test_dino_patch_flatten_order_matches_training_preparation():
    patch_map = torch.zeros((1, 8, 384, 37, 37), dtype=torch.float32)
    for y in range(37):
        for x in range(37):
            patch_map[0, 0, 0, y, x] = y * 100.0 + x
    tokens = dino_patch_map_to_tokens_v1(patch_map)
    assert tokens.shape == (1, 8, 1369, 384)
    assert float(tokens[0, 0, 0, 0]) == 0.0
    assert float(tokens[0, 0, 36, 0]) == 36.0
    assert float(tokens[0, 0, 37, 0]) == 100.0
    assert float(tokens[0, 0, -1, 0]) == 3636.0


def test_rgba_white_composite_matches_admitted_preparation_rounding():
    rgba = torch.zeros((1, 8, 4, 1024, 1024), dtype=torch.uint8)
    rgba[:, :, :3, 0, 0] = torch.tensor([10, 20, 30], dtype=torch.uint8).view(1, 1, 3)
    rgba[:, :, 3, 0, 0] = 255
    rgba[:, :, :3, 0, 1] = 0
    rgba[:, :, 3, 0, 1] = 0
    rgba[:, :, :3, 0, 2] = torch.tensor([0, 100, 200], dtype=torch.uint8).view(1, 1, 3)
    rgba[:, :, 3, 0, 2] = 128

    rgb = rgba_to_white_composited_rgb_v1(rgba)
    assert rgb.shape == (1, 8, 3, 1024, 1024)
    assert rgb[0, 0, :, 0, 0].tolist() == [10, 20, 30]
    assert rgb[0, 0, :, 0, 1].tolist() == [255, 255, 255]
    expected = torch.round(
        (
            (torch.tensor([0.0, 100.0, 200.0]) / 255.0) * (128.0 / 255.0)
            + (1.0 - 128.0 / 255.0)
        )
        * 255.0
    ).to(torch.uint8)
    assert torch.equal(rgb[0, 0, :, 0, 2], expected)


def test_verified_dino_frontend_is_required_and_must_remain_frozen(monkeypatch):
    seal = _loaded_authority()
    monkeypatch.setattr(
        bridge,
        "verify_dinov2_authority_files_v2",
        lambda source_dir, weight_path: seal,
    )
    model = torch.nn.Linear(1, 1).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    assert bridge.assert_loaded_dino_frontend_v1(model, seal) == seal.authority.authority_hash

    model.train()
    with pytest.raises(RuntimeError, match="IRIS_VNEXT_DINO_MODEL_MUST_BE_EVAL"):
        bridge.assert_loaded_dino_frontend_v1(model, seal)


def test_input_bridge_imports_frozen_foundation_not_training_authority():
    authority = DINOv2FoundationAuthorityV2()
    authority.validate()
    source = Path("models/iris/vnext/inference_input_v1.py").read_text(encoding="utf-8")
    assert "models.iris.v2.dinov2_foundation_v2" in source
    assert "models.iris.vnext.training" not in source
    assert "models.iris.vnext.cohort" not in source
    assert "optimizer" not in source
    assert "load_state_dict" not in source
