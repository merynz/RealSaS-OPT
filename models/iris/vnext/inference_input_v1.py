from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from models.iris.v2.dinov2_foundation_v2 import (
    DINO_PATCH_GRID_HW,
    DINO_S_EMBED_DIM,
    DINOv2FoundationAuthorityV2,
    extract_dinov2_s_patch_map_v2,
)


VIEW_COUNT_V1 = 8
PATCH_GRID_V1 = 37
PATCH_TOKEN_COUNT_V1 = PATCH_GRID_V1 * PATCH_GRID_V1
CAMERA_FEATURE_DIM_V1 = 13


@dataclass(frozen=True)
class IRISVNextSceneInputsV1:
    """Image-only scene inputs matching the admitted Fit8 preparation contract."""

    image_tokens: torch.Tensor
    token_xy: torch.Tensor
    camera_features: torch.Tensor
    foundation_authority_sha256: str
    schema: str = "RealSaS.IRISVNextSceneInputs.v1"

    def validate(self) -> None:
        if self.image_tokens.ndim != 4:
            raise ValueError("IRIS_VNEXT_INFERENCE_IMAGE_TOKENS_RANK_INVALID")
        b, v, t, c = self.image_tokens.shape
        if (v, t, c) != (VIEW_COUNT_V1, PATCH_TOKEN_COUNT_V1, DINO_S_EMBED_DIM):
            raise ValueError("IRIS_VNEXT_INFERENCE_IMAGE_TOKENS_SHAPE_INVALID")
        if self.token_xy.shape != (b, v, t, 2):
            raise ValueError("IRIS_VNEXT_INFERENCE_TOKEN_XY_SHAPE_INVALID")
        if self.camera_features.shape != (b, v, CAMERA_FEATURE_DIM_V1):
            raise ValueError("IRIS_VNEXT_INFERENCE_CAMERA_FEATURE_SHAPE_INVALID")
        for tensor in (self.image_tokens, self.token_xy, self.camera_features):
            if not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all()):
                raise ValueError("IRIS_VNEXT_INFERENCE_INPUT_NONFINITE_OR_NONFLOAT")
        authority = DINOv2FoundationAuthorityV2()
        if self.foundation_authority_sha256 != authority.authority_hash:
            raise ValueError("IRIS_VNEXT_INFERENCE_FOUNDATION_AUTHORITY_DRIFT")


def _float01_rgba(native_rgba: torch.Tensor) -> torch.Tensor:
    if native_rgba.ndim != 5 or native_rgba.shape[1:3] != (VIEW_COUNT_V1, 4):
        raise ValueError("IRIS_VNEXT_RGBA_INPUT_SHAPE_INVALID")
    if tuple(native_rgba.shape[-2:]) != (1024, 1024):
        raise ValueError("IRIS_VNEXT_RGBA_INPUT_RESOLUTION_INVALID")
    if native_rgba.dtype == torch.uint8:
        return native_rgba.to(torch.float32).div(255.0)
    if not torch.is_floating_point(native_rgba):
        raise ValueError("IRIS_VNEXT_RGBA_INPUT_DTYPE_INVALID")
    value = native_rgba.to(torch.float32)
    if not bool(torch.isfinite(value).all()) or float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError("IRIS_VNEXT_RGBA_INPUT_RANGE_INVALID")
    return value


def rgba_to_white_composited_rgb_v1(native_rgba: torch.Tensor) -> torch.Tensor:
    """Mirror the admitted Fit8 RGBA->white RGB conversion exactly."""

    rgba = _float01_rgba(native_rgba)
    alpha = rgba[:, :, 3:4]
    rgb = rgba[:, :, :3] * alpha + (1.0 - alpha)
    return torch.round(rgb * 255.0).to(torch.uint8).contiguous()


def dino_patch_map_to_tokens_v1(patch_map: torch.Tensor) -> torch.Tensor:
    """Convert [B,8,384,37,37] DINO maps to admitted [B,8,1369,384] order."""

    if patch_map.ndim != 5:
        raise ValueError("IRIS_VNEXT_DINO_PATCH_MAP_RANK_INVALID")
    b, v, c, h, w = patch_map.shape
    if (v, c, h, w) != (VIEW_COUNT_V1, DINO_S_EMBED_DIM, PATCH_GRID_V1, PATCH_GRID_V1):
        raise ValueError("IRIS_VNEXT_DINO_PATCH_MAP_SHAPE_INVALID")
    if not torch.is_floating_point(patch_map) or not bool(torch.isfinite(patch_map).all()):
        raise ValueError("IRIS_VNEXT_DINO_PATCH_MAP_VALUE_INVALID")
    return patch_map.permute(0, 1, 3, 4, 2).reshape(
        b, VIEW_COUNT_V1, PATCH_TOKEN_COUNT_V1, DINO_S_EMBED_DIM
    ).contiguous()


def token_xy_v1(
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Exact 37x37 patch-center coordinates used by Fit8 preparation."""

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("IRIS_VNEXT_TOKEN_XY_BATCH_INVALID")
    if tuple(DINO_PATCH_GRID_HW) != (PATCH_GRID_V1, PATCH_GRID_V1):
        raise RuntimeError("IRIS_VNEXT_DINO_GRID_AUTHORITY_DRIFT")
    axis = (torch.arange(PATCH_GRID_V1, dtype=torch.float32, device=device) + 0.5) * (
        2.0 / PATCH_GRID_V1
    ) - 1.0
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).reshape(1, 1, PATCH_TOKEN_COUNT_V1, 2)
    return grid.expand(int(batch_size), VIEW_COUNT_V1, -1, -1).clone()


def _vector3(row: Mapping[str, Any], key: str, *, view_index: int) -> torch.Tensor:
    raw = row.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"IRIS_VNEXT_CAMERA_VECTOR_INVALID:{view_index}:{key}")
    value = torch.tensor(raw, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"IRIS_VNEXT_CAMERA_VECTOR_NONFINITE:{view_index}:{key}")
    return value


def camera_features_v1(
    camera_bundle: Mapping[str, Any],
    *,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build the exact admitted 13-D normalized orthographic camera features.

    Feature order is:
      [origin_xyz_normalized(3), right(3), screen_up(3), forward(3),
       half_extent_ratio(1)].
    """

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("IRIS_VNEXT_CAMERA_BATCH_INVALID")
    center_raw = camera_bundle.get("common_center_xyz")
    if not isinstance(center_raw, (list, tuple)) or len(center_raw) != 3:
        raise ValueError("IRIS_VNEXT_COMMON_CENTER_INVALID")
    center = torch.tensor(center_raw, dtype=torch.float32)
    half = float(camera_bundle.get("common_half_extent", 0.0))
    if not bool(torch.isfinite(center).all()) or not torch.isfinite(torch.tensor(half)) or half <= 0.0:
        raise ValueError("IRIS_VNEXT_COMMON_FRAME_INVALID")
    rows = camera_bundle.get("cameras")
    if not isinstance(rows, list) or len(rows) != VIEW_COUNT_V1:
        raise ValueError("IRIS_VNEXT_CAMERA_COUNT_INVALID")

    features = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"IRIS_VNEXT_CAMERA_ROW_INVALID:{index}")
        origin = _vector3(raw_row, "origin", view_index=index)
        right = _vector3(raw_row, "right", view_index=index)
        up = _vector3(raw_row, "screen_up", view_index=index)
        forward = _vector3(raw_row, "forward", view_index=index)
        camera_half = float(raw_row.get("half_extent", 0.0))
        if not torch.isfinite(torch.tensor(camera_half)) or camera_half <= 0.0:
            raise ValueError(f"IRIS_VNEXT_CAMERA_HALF_EXTENT_INVALID:{index}")
        features.append(
            torch.cat(
                (
                    (origin - center) / half,
                    right,
                    up,
                    forward,
                    torch.tensor([camera_half / half], dtype=torch.float32),
                )
            )
        )
    value = torch.stack(features, dim=0).to(device=device, dtype=torch.float32)
    return value.unsqueeze(0).expand(int(batch_size), -1, -1).clone()


@torch.no_grad()
def build_scene_inputs_from_rgba_v1(
    *,
    dino_model: torch.nn.Module,
    native_rgba: torch.Tensor,
    camera_bundle: Mapping[str, Any],
) -> IRISVNextSceneInputsV1:
    """Run the pinned DINO frontend and reproduce the IRIS training input frame."""

    rgb = rgba_to_white_composited_rgb_v1(native_rgba)
    patch_map = extract_dinov2_s_patch_map_v2(dino_model, rgb)
    image_tokens = dino_patch_map_to_tokens_v1(patch_map)
    batch = image_tokens.shape[0]
    value = IRISVNextSceneInputsV1(
        image_tokens=image_tokens,
        token_xy=token_xy_v1(batch, device=image_tokens.device),
        camera_features=camera_features_v1(
            camera_bundle, batch_size=batch, device=image_tokens.device
        ),
        foundation_authority_sha256=DINOv2FoundationAuthorityV2().authority_hash,
    )
    value.validate()
    return value
