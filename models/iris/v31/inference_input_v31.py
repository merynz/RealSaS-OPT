from __future__ import annotations

"""Typed model-ready input for IRIS V3.1.

The alpha resampling algorithm is intentionally not selected here. The producer
must bind the exact policy by a content hash and provide already patch-aligned
raw 14x14 alpha cells per DINO patch. This keeps the open resampling decision out
of model/runtime authority while making the eventual binding explicit.
"""

from dataclasses import dataclass

import torch

from models.iris.v2.dinov2_foundation_v2 import DINOv2FoundationAuthorityV2
from .model_v31 import admit_observations_v31


VIEW_COUNT_V31 = 8
PATCH_GRID_V31 = 37
FINAL_CHANNELS_V31 = 384
ALPHA_PATCH_CELLS_V31 = 196
CAMERA_FEATURE_DIM_V31 = 13


def _exact_sha256(value: str, code: str) -> str:
    raw = str(value)
    if len(raw) != 64 or any(ch not in "0123456789abcdef" for ch in raw):
        raise ValueError(code)
    return raw


@dataclass(frozen=True)
class IRISV31SceneInputs:
    final: torch.Tensor
    layer2: torch.Tensor
    alpha: torch.Tensor
    camera_features: torch.Tensor
    foundation_authority_sha256: str
    alpha_resampling_policy_sha256: str
    schema: str = "RealSaS.IRIS.SceneInputs.v3_1"

    def validate(self) -> None:
        if self.final.ndim != 5:
            raise ValueError("IRIS_V31_INPUT_FINAL_RANK")
        batch, views, height, width, channels = self.final.shape
        if (
            batch <= 0
            or views != VIEW_COUNT_V31
            or (height, width, channels)
            != (PATCH_GRID_V31, PATCH_GRID_V31, FINAL_CHANNELS_V31)
        ):
            raise ValueError("IRIS_V31_INPUT_FINAL_SHAPE")
        if self.layer2.shape != self.final.shape:
            raise ValueError("IRIS_V31_INPUT_LAYER2_SHAPE")
        if self.alpha.shape != (
            batch,
            VIEW_COUNT_V31,
            PATCH_GRID_V31,
            PATCH_GRID_V31,
            ALPHA_PATCH_CELLS_V31,
        ):
            raise ValueError("IRIS_V31_INPUT_ALPHA_SHAPE")
        if self.camera_features.shape != (
            batch,
            VIEW_COUNT_V31,
            CAMERA_FEATURE_DIM_V31,
        ):
            raise ValueError("IRIS_V31_INPUT_CAMERA_SHAPE")
        authority = DINOv2FoundationAuthorityV2()
        if self.foundation_authority_sha256 != authority.authority_hash:
            raise ValueError("IRIS_V31_INPUT_FOUNDATION_AUTHORITY_DRIFT")
        _exact_sha256(
            self.alpha_resampling_policy_sha256,
            "IRIS_V31_INPUT_ALPHA_RESAMPLING_POLICY_HASH",
        )
        admit_observations_v31(
            self.final,
            self.layer2,
            self.alpha,
            self.camera_features,
        )
