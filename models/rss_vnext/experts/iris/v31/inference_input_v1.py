from __future__ import annotations

from dataclasses import dataclass

import torch

from models.iris.v2.dinov2_foundation_v2 import DINOv2FoundationAuthorityV2
from .model_v31 import admit_observations_v31


VIEW_COUNT_V31 = 8
PATCH_GRID_V31 = 37
FINAL_DIM_V31 = 384
ALPHA_PATCH_DIM_V31 = 196
CAMERA_DIM_V31 = 13


def _lower_sha256(value: str, code: str) -> str:
    raw = str(value or "")
    if len(raw) != 64 or raw.lower() != raw or any(ch not in "0123456789abcdef" for ch in raw):
        raise ValueError(code)
    return raw


@dataclass(frozen=True)
class IRISV31SceneInputsV1:
    """Exact frozen-feature input packet for IRIS V3.1 inference.

    This contract starts after observation admission and after the exact alpha
    resampling policy has been executed elsewhere. It deliberately does not guess
    a 1024->518 alpha transform or make an arbitrary source packet authoritative.
    """

    final: torch.Tensor
    layer2: torch.Tensor
    alpha: torch.Tensor
    camera_features: torch.Tensor
    foundation_authority_sha256: str
    observation_bundle_sha256: str
    camera_bundle_sha256: str
    alpha_resampling_policy_sha256: str
    numerical_environment_sha256: str
    schema: str = "RealSaS.IRISV31SceneInputs.v1"

    def validate(self) -> None:
        if self.final.ndim != 5:
            raise ValueError("IRIS_V31_INPUT_FINAL_RANK_INVALID")
        batch, views, h, w, width = self.final.shape
        if (views, h, w, width) != (
            VIEW_COUNT_V31,
            PATCH_GRID_V31,
            PATCH_GRID_V31,
            FINAL_DIM_V31,
        ) or batch <= 0:
            raise ValueError("IRIS_V31_INPUT_FINAL_SHAPE_INVALID")
        if self.layer2.shape != self.final.shape:
            raise ValueError("IRIS_V31_INPUT_LAYER2_SHAPE_INVALID")
        if self.alpha.shape != (
            batch,
            VIEW_COUNT_V31,
            PATCH_GRID_V31,
            PATCH_GRID_V31,
            ALPHA_PATCH_DIM_V31,
        ):
            raise ValueError("IRIS_V31_INPUT_ALPHA_SHAPE_INVALID")
        if self.camera_features.shape != (batch, VIEW_COUNT_V31, CAMERA_DIM_V31):
            raise ValueError("IRIS_V31_INPUT_CAMERA_SHAPE_INVALID")
        for value in (self.final, self.layer2, self.alpha, self.camera_features):
            if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
                raise ValueError("IRIS_V31_INPUT_NONFINITE_OR_NONFLOAT")
        if float(self.alpha.min()) < 0.0 or float(self.alpha.max()) > 1.0:
            raise ValueError("IRIS_V31_INPUT_ALPHA_RANGE_INVALID")
        authority = DINOv2FoundationAuthorityV2()
        if self.foundation_authority_sha256 != authority.authority_hash:
            raise ValueError("IRIS_V31_INPUT_FOUNDATION_AUTHORITY_DRIFT")
        _lower_sha256(self.observation_bundle_sha256, "IRIS_V31_INPUT_OBSERVATION_HASH_INVALID")
        _lower_sha256(self.camera_bundle_sha256, "IRIS_V31_INPUT_CAMERA_HASH_INVALID")
        _lower_sha256(
            self.alpha_resampling_policy_sha256,
            "IRIS_V31_INPUT_ALPHA_POLICY_HASH_INVALID",
        )
        _lower_sha256(
            self.numerical_environment_sha256,
            "IRIS_V31_INPUT_NUMERICAL_ENVIRONMENT_HASH_INVALID",
        )
        admit_observations_v31(self.final, self.layer2, self.alpha, self.camera_features)
