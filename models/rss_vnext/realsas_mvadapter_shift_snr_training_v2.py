from __future__ import annotations

"""Pinned official SDXL MV-Adapter training noise-schedule contract (ICCV25).

The upstream I2MV SDXL recipe uses DDPM *shifted SNR* with interpolated
schedule, shift_scale=8.0. Training with ordinary unshifted DDPM would silently
change the task and invalidate the pretrained attention comparison.

The implementation calls the SHA-verified upstream ShiftSNRScheduler code
rather than implementing a guessed beta schedule.
"""

from hashlib import sha256
from typing import Any

import torch
from torch import Tensor

from .realsas_sdxl_mvadapter_pretrain_v1 import (
    StrongSDXLMVAdapterPinV1,
    verify_strong_pretrained_snapshots_v1,
    _verified_upstream_module,
)

SCHEMA="RealSaS.RSSvNext.OfficialMVAdapterSDXLInterpolatedShiftSNR8.v1"
SHIFT_MODE="interpolated"
SHIFT_SCALE=8.0


def schedule_alphas_sha256_v1(scheduler:Any)->str:
    alpha=getattr(scheduler,"alphas_cumprod",None)
    if (
        not isinstance(alpha,Tensor)
        or alpha.ndim!=1 or alpha.numel()<2
        or not bool(torch.isfinite(alpha).all())
        or bool((alpha<=0).any()) or bool((alpha>1).any())
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_SCHEDULER_ALPHA_ABI")
    data=alpha.detach().float().contiguous().cpu().numpy().tobytes()
    return sha256(
        SCHEMA.encode()+len(data).to_bytes(8,"little")+data,
    ).hexdigest()


def build_official_mvadapter_training_scheduler_v2(
    *,pipeline:Any,pin:StrongSDXLMVAdapterPinV1,
):
    """Recreate official source-trained DDPM + ShiftSNR, entirely offline."""
    verify_strong_pretrained_snapshots_v1(pin)
    from diffusers import DDPMScheduler
    upstream=_verified_upstream_module(
        "mvadapter.schedulers.scheduling_shift_snr",pin=pin,
    )
    # The pretrained pipeline may already have shifted its inference
    # scheduler. Always reconstruct ONCE from the sealed unshifted base
    # schedule rather than a previously-shifted scheduler's betas.
    base_config=getattr(
        pipeline,"_rss_vnext_unshifted_scheduler_config",None,
    )
    if base_config is None:
        # Raw pipelines only; an already shifted scheduler lacking a
        # recorded ancestor cannot be safely shifted again.
        if getattr(pipeline.scheduler.config,"trained_betas",None) is not None:
            raise RuntimeError(
                "RSS_VNEXT_MVADAPTER_SHIFTED_SCHEDULER_BASELINE_MISSING"
            )
        base_config=pipeline.scheduler.config
    original=DDPMScheduler.from_config(base_config)
    shifted=upstream.ShiftSNRScheduler.from_scheduler(
        original,shift_mode=SHIFT_MODE,shift_scale=SHIFT_SCALE,
        scheduler_class=DDPMScheduler,
    )
    if shifted.config.prediction_type!="epsilon":
        raise RuntimeError("RSS_VNEXT_MVADAPTER_TRAIN_SNR_EPSILON_REQUIRED")
    shifted._rss_vnext_shift_snr_schedule_sha256=(
        schedule_alphas_sha256_v1(shifted)
    )
    shifted._rss_vnext_training_shift_mode=SHIFT_MODE
    shifted._rss_vnext_training_shift_scale=SHIFT_SCALE
    pipeline.scheduler=shifted
    return shifted


def require_pinned_official_mvadapter_schedule_v2(scheduler:Any)->str:
    digest=schedule_alphas_sha256_v1(scheduler)
    if (
        getattr(scheduler,"_rss_vnext_shift_snr_schedule_sha256",None)
        !=digest
        or getattr(scheduler,"_rss_vnext_training_shift_mode",None)
        !=SHIFT_MODE
        or getattr(scheduler,"_rss_vnext_training_shift_scale",None)
        !=SHIFT_SCALE
        or getattr(scheduler.config,"prediction_type",None)!="epsilon"
    ):
        raise RuntimeError("RSS_VNEXT_MVADAPTER_OFFICIAL_TRAIN_NOISE_ABI_REQUIRED")
    return digest


__all__=[
    "SCHEMA","SHIFT_MODE","SHIFT_SCALE","schedule_alphas_sha256_v1",
    "build_official_mvadapter_training_scheduler_v2",
    "require_pinned_official_mvadapter_schedule_v2",
]
