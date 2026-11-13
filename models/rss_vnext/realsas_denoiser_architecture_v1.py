from __future__ import annotations

"""One explicit execution ABI shared by paired diffusion training and inference.

The active RealSaS pipeline learns with a four-channel epsilon denoiser and
separate latent-row cross-view attention. The separately researched eight-
channel image-conditioned UNet with in-feature row attention is NOT weight-
compatible. Releasing one as the other would be a silent product failure.
"""

from typing import Any

from torch import nn

ACTIVE_DENOISER_ABI = "RealSaS.RSSvNext.PairedEpsilon4ChannelLatentRow.v1"


def _config_field(module: nn.Module, name: str) -> Any:
    config = getattr(module, "config", None)
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def validate_paired_denoiser_abi_v1(
    denoiser: nn.Module, *,
    conditioning_width: int,
    latent_row_attention: nn.Module | None,
    require_explicit_config: bool,
) -> str:
    """Reject 8-channel, in-UNet attention and mismatched context widths.

    Untyped microstep test doubles may omit Diffusers config, but a packaged
    production release MUST publish the exact input/output/context contract.
    """
    if (
        not isinstance(denoiser, nn.Module)
        or type(conditioning_width) is not int or conditioning_width < 1
        or type(require_explicit_config) is not bool
    ):
        raise ValueError("RSS_VNEXT_DENOISER_ABI_ARGUMENTS_INVALID")
    if getattr(denoiser, "rowwise_attention_in_unet_features", False):
        raise RuntimeError("RSS_VNEXT_DENOISER_FEATURE_ROW_ABI_MISMATCH")
    if latent_row_attention is not None:
        actual_latent = getattr(latent_row_attention, "latent_channels", None)
        if actual_latent != 4:
            raise RuntimeError("RSS_VNEXT_DENOISER_ROW_LATENT_ABI_MISMATCH")
    channels_in = _config_field(denoiser, "in_channels")
    channels_out = _config_field(denoiser, "out_channels")
    cross_width = _config_field(denoiser, "cross_attention_dim")
    if require_explicit_config and (
        channels_in is None or channels_out is None or cross_width is None
    ):
        raise RuntimeError("RSS_VNEXT_DENOISER_EXPLICIT_ARCHITECTURE_REQUIRED")
    if channels_in is not None and channels_in != 4:
        raise RuntimeError("RSS_VNEXT_DENOISER_INPUT_CHANNEL_ABI_MISMATCH")
    if channels_out is not None and channels_out != 4:
        raise RuntimeError("RSS_VNEXT_DENOISER_OUTPUT_CHANNEL_ABI_MISMATCH")
    if cross_width is not None and (
        type(cross_width) is not int or cross_width != conditioning_width
    ):
        raise RuntimeError("RSS_VNEXT_DENOISER_CROSS_ATTENTION_WIDTH_MISMATCH")
    return ACTIVE_DENOISER_ABI


__all__ = ["ACTIVE_DENOISER_ABI", "validate_paired_denoiser_abi_v1"]
