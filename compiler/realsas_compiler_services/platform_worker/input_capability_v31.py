from __future__ import annotations

"""Parallel source-observation lowering for the non-activating IRIS V3.1 lane."""

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Any

import torch

from compiler.realsas_compiler_services.platform_worker.input_capability_v1 import (
    SourceObservationV1,
    decode_source_observation_v1,
    load_product_dino_v1,
)
from compiler.realsas_compiler_services.platform_worker.scene_input_contract_v31 import (
    SHAPE_INPUT_ARTIFACT_TYPE_V31,
    encode_shape_input_v31,
)
from models.iris.v2.dinov2_foundation_v2 import DINOv2LoadedAuthorityV2
from models.iris.v31.alpha_resampling_v1 import policy_sha256_v1, resample_source_alpha_v1
from models.iris.v31.frozen_features_v31 import extract_frozen_features
from models.iris.v31.inference_input_v1 import IRISV31SceneInputsV1
from models.iris.v31.numerical_environment_v1 import numerical_environment_sha256_v1
from models.iris.vnext.inference_input_v1 import (
    assert_loaded_dino_frontend_v1,
    camera_features_v1,
    rgba_to_white_composited_rgb_v1,
)


INPUT_EXECUTOR_CONTRACT_V31 = "RealSaS.ProductInputInference.v3_1"
INPUT_OUTPUT_AUTHORITY_V31 = "VERIFIED_SOURCE_DERIVED_SHAPE_INPUT"


def _lower_sha256(value: str, code: str) -> str:
    raw = str(value or "")
    if len(raw) != 64 or raw.lower() != raw or any(ch not in "0123456789abcdef" for ch in raw):
        raise RuntimeError(code)
    return raw


def camera_bundle_sha256_v31(camera_bundle: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(camera_bundle),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("PRODUCT_INPUT_V31_CAMERA_BUNDLE_NOT_CANONICAL_JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class InputInferenceResultV31:
    payload: bytes
    diagnostics: dict[str, Any]
    schema: str = "RealSaS.ProductInputInferenceResult.v3_1"


@torch.no_grad()
def run_input_inference_v31(
    source: SourceObservationV1,
    *,
    source_content_sha256: str,
    dino_model: torch.nn.Module,
    loaded_authority: DINOv2LoadedAuthorityV2,
    device: torch.device | str = "cpu",
    view_batch_size: int = 2,
) -> InputInferenceResultV31:
    """Lower one verified source packet into exact V3.1 frozen-feature Shape input."""

    source.validate()
    source_hash = _lower_sha256(
        source_content_sha256,
        "PRODUCT_INPUT_V31_SOURCE_CONTENT_SHA_INVALID",
    )
    if isinstance(view_batch_size, bool) or int(view_batch_size) <= 0:
        raise ValueError("PRODUCT_INPUT_V31_VIEW_BATCH_SIZE_INVALID")
    device = torch.device(device)
    foundation_hash = assert_loaded_dino_frontend_v1(dino_model, loaded_authority)

    started = perf_counter()
    rgb = rgba_to_white_composited_rgb_v1(source.native_rgba).to(device=device)
    alpha_518 = resample_source_alpha_v1(source.native_rgba[:, :, 3:4])
    features = extract_frozen_features(
        dino_model,
        rgb,
        alpha_518,
        view_batch_size=int(view_batch_size),
    )
    camera_hash = camera_bundle_sha256_v31(source.camera_bundle)
    alpha_policy_hash = policy_sha256_v1()
    environment_hash = numerical_environment_sha256_v1(device)
    inputs = IRISV31SceneInputsV1(
        final=features["final"],
        layer2=features["layer2"],
        alpha=features["alpha"],
        camera_features=camera_features_v1(
            source.camera_bundle,
            batch_size=int(source.native_rgba.shape[0]),
            device="cpu",
        ),
        foundation_authority_sha256=foundation_hash,
        observation_bundle_sha256=source_hash,
        camera_bundle_sha256=camera_hash,
        alpha_resampling_policy_sha256=alpha_policy_hash,
        numerical_environment_sha256=environment_hash,
    )
    inputs.validate()
    payload = encode_shape_input_v31(inputs)
    return InputInferenceResultV31(
        payload=payload,
        diagnostics={
            "schema": "RealSaS.ProductInputInferenceDiagnostics.v3_1",
            "authority": INPUT_OUTPUT_AUTHORITY_V31,
            "output_artifact_type": SHAPE_INPUT_ARTIFACT_TYPE_V31,
            "source_schema": source.schema,
            "source_content_sha256": source_hash,
            "camera_bundle_sha256": camera_hash,
            "alpha_resampling_policy_sha256": alpha_policy_hash,
            "numerical_environment_sha256": environment_hash,
            "foundation_authority_sha256": foundation_hash,
            "view_count": int(source.native_rgba.shape[1]),
            "native_resolution": int(source.native_rgba.shape[-1]),
            "feature_grid": [37, 37],
            "view_batch_size": int(view_batch_size),
            "inference_device": str(device),
            "elapsed_seconds": float(perf_counter() - started),
            "source_truth_copied_into_shape": False,
            "observation_admission_minted": False,
            "model_release_minted": False,
            "product_shape_activated": False,
        },
    )


__all__ = [
    "INPUT_EXECUTOR_CONTRACT_V31",
    "INPUT_OUTPUT_AUTHORITY_V31",
    "InputInferenceResultV31",
    "SourceObservationV1",
    "camera_bundle_sha256_v31",
    "decode_source_observation_v1",
    "load_product_dino_v1",
    "run_input_inference_v31",
]
