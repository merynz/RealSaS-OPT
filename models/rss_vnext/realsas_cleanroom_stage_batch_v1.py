from __future__ import annotations

"""Strict materialized-render -> two-stage diffusion *training* tensor bridge.

Stage A: two DIFFERENT source-rendered arbitrary-view images/poses condition
two independently rendered canonical-pose targets (front and back).
Stage B: two canonical-pose source-rendered images condition six RGB and six
matching normal targets. Never substitute the first canonical RGB for BOTH
source image tokens, never infer original cameras from target cameras, and
never use diffusion-generated images as source truth.

A source backbone MUST be independently pretrained/frozen, like the deployed
DINOv2 encoder. It is not trained by this stage's denoiser gradient.
"""
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .realsas_cleanroom_diffusion_training_v1 import (
    STAGE_CANONICAL, STAGE_MULTIVIEW,
    PairedSourceDiffusionConditionerV1,
)
from .realsas_virtual_camera_policy_v1 import (
    target_canonical_and_six_cameras_v1,
)


def _batch_label(value: Any, label: str) -> str:
    if type(value) is str:
        return value
    if type(value) in (list, tuple) and value:
        if all(type(x) is str and x == value[0] for x in value):
            return value[0]
    raise ValueError("RSS_VNEXT_STAGE_BATCH_METADATA_INVALID:" + label)


def _source_camera_flag_is_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, Tensor):
        return (
            value.dtype == torch.bool and value.ndim <= 1
            and value.numel() > 0 and not bool(value.any())
        )
    if type(value) in (list, tuple) and value:
        return all(item is False for item in value)
    return False


def _as_batch(value: Any, rank: int, field: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_TENSOR_MISSING:" + field)
    if value.ndim == rank - 1:
        return value.unsqueeze(0)
    if value.ndim != rank:
        raise ValueError("RSS_VNEXT_STAGE_BATCH_TENSOR_RANK:" + field)
    return value


def prepare_rendered_two_source_diffusion_microstep_v1(
    sample: Mapping[str, Any],
    *,
    source_encoder: nn.Module,
    conditioner: PairedSourceDiffusionConditionerV1,
) -> dict[str, Tensor | str]:
    """Return exact kwargs consumed by cleanroom_diffusion_noise_microstep_v1.

    The caller adds vae/scheduler/denoiser/conditioner (and row attention for
    Stage B). Source and target batches must already be on the device shared
    by the pinned frozen source encoder; no implicit image rescaling/camera
    reinterpretation happens at this ABI boundary.
    """
    if not isinstance(sample, Mapping):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_MAPPING_REQUIRED")
    stage = _batch_label(sample.get("stage"), "stage")
    if stage not in (STAGE_CANONICAL, STAGE_MULTIVIEW):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_STAGE_INVALID")
    expected_authority = (
        "SOURCE_3D_RENDERED_TWO_ARBITRARY_POSES"
        if stage == STAGE_CANONICAL else
        "SOURCE_3D_RENDERED_TWO_CANONICAL_IMAGES"
    )
    if (
        _batch_label(sample.get("source_pair_authority"), "source_pair_authority")
        != expected_authority
        or not _source_camera_flag_is_false(sample.get("original_source_camera_observed"))
        or not isinstance(source_encoder, nn.Module)
        or not isinstance(conditioner, PairedSourceDiffusionConditionerV1)
    ):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_SOURCE_AUTHORITY_INVALID")
    if source_encoder.training or any(p.requires_grad for p in source_encoder.parameters()):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_SOURCE_ENCODER_MUST_BE_FROZEN")

    source = _as_batch(sample.get("source_rgba"), 5, "source_rgba")
    targets = _as_batch(sample.get("target_rgb"), 5, "target_rgb")
    masks = _as_batch(sample.get("target_rgb_valid"), 5, "target_rgb_valid")
    cameras = _as_batch(
        sample.get("target_camera_features"), 3, "target_camera_features"
    )
    modalities = _as_batch(
        sample.get("target_modalities"), 2, "target_modalities"
    )
    b, v, channels, h, w = source.shape
    n = 2 if stage == STAGE_CANONICAL else 6
    if (
        b < 1 or v != 2 or channels != 4
        or targets.shape != (b, n, 3, h, w)
        or masks.shape != (b, n, 1, h, w)
        or cameras.shape != (b, (2 if stage == STAGE_CANONICAL else 12), 13)
        or modalities.shape != (b, (2 if stage == STAGE_CANONICAL else 12))
        or modalities.dtype != torch.long
        or not torch.is_floating_point(source)
        or not torch.is_floating_point(targets)
        or not torch.is_floating_point(masks)
        or not torch.is_floating_point(cameras)
        or any(x.device != source.device for x in (targets, masks, cameras, modalities))
    ):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_TENSOR_DOMAIN")
    if any(
        not bool(torch.isfinite(x).all())
        or bool((x < 0).any())
        or bool((x > 1).any())
        for x in (source, targets, masks)
    ):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_PIXEL_DOMAIN")
    if not bool(torch.isfinite(cameras).all()) or not bool((masks > 0).any()):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_MISSING_TRUTH")
    if stage == STAGE_CANONICAL and bool((modalities != 0).any()):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_CANONICAL_MODALITY_DRIFT")
    if stage == STAGE_MULTIVIEW and (
        bool((modalities[:, :6] != 0).any())
        or bool((modalities[:, 6:] != 1).any())
    ):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_MULTIVIEW_MODALITY_DRIFT")

    virtual_canonical, virtual_six = target_canonical_and_six_cameras_v1(
        b, device=str(source.device), dtype=cameras.dtype,
    )
    expected_cameras = (
        virtual_canonical if stage == STAGE_CANONICAL
        else torch.cat((virtual_six, virtual_six), dim=1)
    )
    if not torch.allclose(cameras, expected_cameras, atol=0., rtol=0.):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_VIRTUAL_CAMERA_DRIFT")

    with torch.no_grad():
        encoded = source_encoder(source)
    if type(encoded) is not tuple or not encoded:
        raise ValueError("RSS_VNEXT_STAGE_BATCH_SOURCE_ENCODER_OUTPUT_INVALID")
    tokens = encoded[0]
    if (
        not isinstance(tokens, Tensor) or tokens.ndim != 4
        or tokens.shape[:2] != (b, 2) or tokens.shape[2] < 1
        or tokens.shape[-1] != conditioner.source_projection.in_features
        or not bool(torch.isfinite(tokens).all())
        or tokens.requires_grad
    ):
        raise ValueError("RSS_VNEXT_STAGE_BATCH_SOURCE_TOKENS_INVALID")

    if stage == STAGE_MULTIVIEW:
        normals = _as_batch(
            sample.get("target_normal_encoded_rgb"), 5,
            "target_normal_encoded_rgb",
        )
        normal_mask = _as_batch(
            sample.get("target_normal_valid"), 5, "target_normal_valid",
        )
        if (
            normals.shape != targets.shape or normal_mask.shape != masks.shape
            or normals.device != source.device or normal_mask.device != source.device
            or not torch.equal(normal_mask, masks)
            or not bool(torch.isfinite(normals).all())
            or bool((normals < 0).any()) or bool((normals > 1).any())
        ):
            raise ValueError("RSS_VNEXT_STAGE_BATCH_NORMAL_TRUTH_DRIFT")
        rendered_targets = torch.cat((targets, normals), dim=1)
        rendered_mask = torch.cat((masks, normal_mask), dim=1)
    else:
        if (
            sample.get("target_normal_encoded_rgb") is not None
            or sample.get("target_normal_valid") is not None
        ):
            raise ValueError("RSS_VNEXT_STAGE_BATCH_CANONICAL_NORMAL_TEACHER_FORBIDDEN")
        rendered_targets = targets
        rendered_mask = masks

    return {
        "source_tokens": tokens,
        "rendered_target_rgb": rendered_targets,
        "rendered_target_validity_mask": rendered_mask,
        "target_cameras": cameras,
        "target_modalities": modalities,
        "stage": stage,
    }


__all__ = ["prepare_rendered_two_source_diffusion_microstep_v1"]
