from __future__ import annotations

import torch

from compiler.realsas_compiler_services.platform_worker import input_capability_v31 as input31
from compiler.realsas_compiler_services.platform_worker.input_capability_v1 import SourceObservationV1
from compiler.realsas_compiler_services.platform_worker.scene_input_contract_v31 import decode_shape_input_v31
from models.iris.v2.dinov2_foundation_v2 import DINOv2FoundationAuthorityV2
from models.iris.v31.alpha_resampling_v1 import policy_sha256_v1
from models.iris.v31.model_v31 import alpha_patches_v31


def _camera_bundle():
    rows = []
    for _ in range(8):
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


def _source():
    rgba = torch.zeros(1, 8, 4, 1024, 1024, dtype=torch.uint8)
    rgba[:, :, :3] = 64
    rgba[:, :, 3] = 255
    return SourceObservationV1(native_rgba=rgba, camera_bundle=_camera_bundle())


def _fake_features(_model, native_rgb, alpha_518, *, view_batch_size):
    assert tuple(native_rgb.shape) == (1, 8, 3, 1024, 1024)
    assert native_rgb.dtype == torch.uint8
    assert tuple(alpha_518.shape) == (1, 8, 1, 518, 518)
    assert alpha_518.device.type == "cpu"
    assert view_batch_size == 2
    return {
        "final": torch.zeros(1, 8, 37, 37, 384, dtype=torch.float32),
        "layer2": torch.ones(1, 8, 37, 37, 384, dtype=torch.float32),
        "alpha": alpha_patches_v31(alpha_518),
    }


def test_v31_input_lowering_binds_exact_source_camera_policy_and_environment(monkeypatch):
    authority = DINOv2FoundationAuthorityV2().authority_hash
    monkeypatch.setattr(input31, "assert_loaded_dino_frontend_v1", lambda model, loaded: authority)
    monkeypatch.setattr(input31, "extract_frozen_features", _fake_features)
    monkeypatch.setattr(input31, "numerical_environment_sha256_v1", lambda device: "4" * 64)

    result = input31.run_input_inference_v31(
        _source(),
        source_content_sha256="1" * 64,
        dino_model=object(),
        loaded_authority=object(),
        device="cpu",
        view_batch_size=2,
    )
    decoded = decode_shape_input_v31(result.payload)

    assert decoded.observation_bundle_sha256 == "1" * 64
    assert decoded.camera_bundle_sha256 == input31.camera_bundle_sha256_v31(_camera_bundle())
    assert decoded.alpha_resampling_policy_sha256 == policy_sha256_v1()
    assert decoded.numerical_environment_sha256 == "4" * 64
    assert decoded.foundation_authority_sha256 == authority
    assert tuple(decoded.final.shape) == (1, 8, 37, 37, 384)
    assert tuple(decoded.alpha.shape) == (1, 8, 37, 37, 196)
    assert torch.equal(decoded.layer2, torch.ones_like(decoded.layer2))
    assert float(decoded.alpha.min()) >= 1.0 - 2.0e-6
    assert float(decoded.alpha.max()) <= 1.0
    assert result.diagnostics["product_shape_activated"] is False
    assert result.diagnostics["model_release_minted"] is False
    assert result.diagnostics["observation_admission_minted"] is False


def test_v31_camera_hash_is_order_stable_but_content_sensitive():
    source = _camera_bundle()
    reordered = {key: source[key] for key in reversed(list(source))}
    assert input31.camera_bundle_sha256_v31(source) == input31.camera_bundle_sha256_v31(reordered)
    changed = _camera_bundle()
    changed["cameras"][0]["half_extent"] = 1.5
    assert input31.camera_bundle_sha256_v31(source) != input31.camera_bundle_sha256_v31(changed)


def test_v31_input_rejects_unbound_source_hash(monkeypatch):
    authority = DINOv2FoundationAuthorityV2().authority_hash
    monkeypatch.setattr(input31, "assert_loaded_dino_frontend_v1", lambda model, loaded: authority)
    try:
        input31.run_input_inference_v31(
            _source(),
            source_content_sha256="not-a-hash",
            dino_model=object(),
            loaded_authority=object(),
        )
    except RuntimeError as exc:
        assert "PRODUCT_INPUT_V31_SOURCE_CONTENT_SHA_INVALID" in str(exc)
    else:
        raise AssertionError("V3.1 input must bind exact source content bytes")


def test_v31_input_plumbing_is_parallel_not_active_provider():
    active = open(
        "compiler/realsas_compiler_services/platform_worker/input_executor_v1.py",
        encoding="utf-8",
    ).read()
    assert "input_capability_v31" not in active
    assert "execute_input_capability_v31" not in active
