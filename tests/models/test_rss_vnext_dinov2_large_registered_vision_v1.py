from __future__ import annotations

"""No downloaded weights: exact DINOv2-L/14 Reg4 ABI and 1024->448 conditioning."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.rss_vnext.unified_character_model import RSSvNextFrozenDINOv2
from models.rss_vnext.contracts import RSSvNextConfig


class FakeRegisteredLarge(nn.Module):
    def __init__(self, *, registers: int = 4, hidden: int = 1024):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            model_type="dinov2_with_registers",
            hidden_size=hidden,
            patch_size=14,
            num_register_tokens=registers,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_channels=3,
        )
        self.pixel_values = None

    def forward(self, *, pixel_values):
        self.pixel_values = pixel_values.detach().clone()
        n = (pixel_values.shape[-1] // 14) ** 2
        result = pixel_values.new_ones(
            (pixel_values.shape[0], 1 + self.config.num_register_tokens + n, 1024),
        )
        result[:, :1 + self.config.num_register_tokens] = -999.0
        return SimpleNamespace(last_hidden_state=result)


def install_model(monkeypatch, *, registers=4, hidden=1024):
    fake = FakeRegisteredLarge(registers=registers, hidden=hidden)

    class AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == "/fake/pinned/local/large-reg4"
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return fake

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=AutoModel))
    return fake


def test_default_product_iris_width_is_1024_and_test_tiny_stays_independent():
    assert RSSvNextConfig().iris.image_token_dim == 1024
    assert RSSvNextConfig.tiny_for_test().iris.image_token_dim == 32


def test_registered_large_two_source_views_strip_cls_registers_and_keep_1024_rgba(monkeypatch):
    fake = install_model(monkeypatch)
    vision = RSSvNextFrozenDINOv2("/fake/pinned/local/large-reg4")
    assert vision.architecture_id == "RSS.vNext.Vision.DINOv2-L14-Reg4.TwoView.448.v1"
    assert vision.vision_side == 448
    assert not any(p.requires_grad for p in vision.backbone.parameters())
    vision.train()
    assert not vision.backbone.training

    source = torch.zeros((1, 2, 4, 1024, 1024))
    source[:, 0, :3] = torch.tensor([1.0, 0.0, 0.0])[None, :, None, None]
    # First view is transparent red, second view is opaque black.
    source[:, 1, 3] = 1.0
    before = source.clone()
    tokens, xy = vision(source)
    assert torch.equal(source, before)  # no in-place source mutation
    assert tokens.shape == (1, 2, 32 * 32, 1024)
    assert xy.shape == (1, 2, 32 * 32, 2)
    assert torch.all(tokens == 1.0)  # CLS and register sentinels excluded
    assert fake.pixel_values.shape == (2, 3, 448, 448)
    mean = vision.mean[0, :, 0, 0]
    std = vision.std[0, :, 0, 0]
    torch.testing.assert_close(
        fake.pixel_values[0, :, 100, 100],
        (torch.ones(3) - mean) / std,
    )
    torch.testing.assert_close(
        fake.pixel_values[1, :, 100, 100],
        -mean / std,
    )
    assert float(xy.min()) > -1.0
    assert float(xy.max()) < 1.0


@pytest.mark.parametrize("registers,hidden", [(0, 1024), (4, 384)])
def test_product_rejects_smaller_or_registerless_checkpoint(monkeypatch, registers, hidden):
    install_model(monkeypatch, registers=registers, hidden=hidden)
    with pytest.raises(RuntimeError, match="BACKBONE_ARCHITECTURE_DRIFT"):
        RSSvNextFrozenDINOv2("/fake/pinned/local/large-reg4")


def test_rejects_wrong_feature_width_and_non_two_view_source(monkeypatch):
    install_model(monkeypatch)
    with pytest.raises(ValueError, match="LARGE_REGISTERED_VISION_CONFIG_INVALID"):
        RSSvNextFrozenDINOv2("/fake/pinned/local/large-reg4", expected_dim=384)
    vision = RSSvNextFrozenDINOv2("/fake/pinned/local/large-reg4")
    with pytest.raises(ValueError, match="DINO_EXPECTS_TWO_RGBA"):
        vision(torch.zeros(1, 1, 4, 448, 448))
    with pytest.raises(ValueError, match="DINO_SQUARE_SOURCE_REQUIRED"):
        vision(torch.zeros(1, 2, 4, 448, 512))


def test_six_generated_conditioning_uses_same_frozen_projection_contract(monkeypatch):
    install_model(monkeypatch)
    vision = RSSvNextFrozenDINOv2("/fake/pinned/local/large-reg4")
    frames = torch.zeros((1, 6, 4, 448, 448))
    frames[:, :, 3] = 1.0
    tokens, coords = vision.forward_variable_views(frames)
    assert tokens.shape == (1, 6, 1024, 1024)
    assert coords.shape == (1, 6, 1024, 2)
