from types import SimpleNamespace

import pytest
from torch import nn

from models.rss_vnext.realsas_denoiser_architecture_v1 import (
    ACTIVE_DENOISER_ABI,
    validate_paired_denoiser_abi_v1,
)
from models.rss_vnext.realsas_cleanroom_diffusion_training_v1 import (
    RowWiseRGBNormalViewAttentionV1,
)


class _FakeUNet(nn.Module):
    def __init__(self, *, in_channels=4, out_channels=4, cross_attention_dim=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.config = SimpleNamespace(
            in_channels=in_channels, out_channels=out_channels,
            cross_attention_dim=cross_attention_dim,
        )


def _check(model, **kwargs):
    return validate_paired_denoiser_abi_v1(
        model, conditioning_width=32, latent_row_attention=kwargs.pop(
            "row", RowWiseRGBNormalViewAttentionV1(
                latent_channels=4, hidden_width=8, heads=2,
            ),
        ),
        require_explicit_config=kwargs.pop("require_explicit_config", True),
    )


def test_training_inference_contract_is_explicit_for_paired_denoiser():
    assert _check(_FakeUNet()) == ACTIVE_DENOISER_ABI


@pytest.mark.parametrize("config,code", [
    ({"in_channels": 8}, "INPUT_CHANNEL_ABI_MISMATCH"),
    ({"out_channels": 8}, "OUTPUT_CHANNEL_ABI_MISMATCH"),
    ({"cross_attention_dim": 2048}, "CROSS_ATTENTION_WIDTH_MISMATCH"),
])
def test_unrelated_or_incompatible_denoiser_is_not_a_release(config, code):
    with pytest.raises(RuntimeError, match=code):
        _check(_FakeUNet(**config))


def test_unet_feature_row_attention_is_not_external_latent_row_execution():
    unet = _FakeUNet()
    unet.rowwise_attention_in_unet_features = True
    with pytest.raises(RuntimeError, match="FEATURE_ROW_ABI_MISMATCH"):
        _check(unet)


def test_checkpoint_requires_explicit_denoiser_config_and_four_channel_row_mixer():
    no_metadata = nn.Conv2d(4, 4, 1)
    with pytest.raises(RuntimeError, match="EXPLICIT_ARCHITECTURE_REQUIRED"):
        _check(no_metadata)
    assert _check(no_metadata, require_explicit_config=False) == ACTIVE_DENOISER_ABI
    with pytest.raises(RuntimeError, match="ROW_LATENT_ABI_MISMATCH"):
        _check(_FakeUNet(), row=RowWiseRGBNormalViewAttentionV1(
            latent_channels=8, hidden_width=8, heads=2,
        ))
