from __future__ import annotations

"""Test the REAL packaging/loading functions with small deterministic modules.

The tiny module namespace isolates the ZIP/Torch checkpoint protocol from the
size of upstream diffusion foundation weights. It is not a scientific test
of trained generation quality or permission to load test vision in product.
"""

from dataclasses import replace
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests, RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.realsas_cleanroom_diffusion_training_v1 import (
    STAGE_CANONICAL, STAGE_MULTIVIEW,
    PairedSourceDiffusionConditionerV1,
    RowWiseRGBNormalViewAttentionV1,
)
from models.rss_vnext.realsas_two_stage_sampler_v1 import RSSvNextPinnedDiffusionStageV1
from models.rss_vnext.realsas_slrm_stage1_lora_v1 import install_rss_realsas_stage1_attention_lora_v1
from models.rss_vnext.realsas_product_neural_bundle_v1 import (
    save_product_neural_bundle_v1, load_product_neural_bundle_v1,
)
from models.rss_vnext.inference_checkpoint_v1 import load_inference_checkpoint_v1


class _MapConfig(dict):
    __getattr__ = dict.get


class UNet2DConditionModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = _MapConfig(config or {"in_channels": 4})
        self.conv = nn.Conv2d(4, 4, 1)
        self.peft_config = {}

    @classmethod
    def from_config(cls, value):
        return cls(value)

    def forward(self, sample, timestep, *, encoder_hidden_states):
        return SimpleNamespace(sample=self.conv(sample))


class AutoencoderKL(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = _MapConfig(config or {"scaling_factor": 1.})
        self.projection = nn.Conv2d(4, 4, 1)

    @classmethod
    def from_config(cls, value):
        return cls(value)

    def decode(self, sample):
        latent = self.projection(sample)[:, :3]
        return SimpleNamespace(sample=F.interpolate(
            latent, scale_factor=8, mode="nearest",
        ))


class DDIMScheduler:
    init_noise_sigma = 1.
    def __init__(self, config=None):
        self.config = _MapConfig(config or {
            "prediction_type": "epsilon", "num_train_timesteps": 10,
        })

    @classmethod
    def from_config(cls, value):
        return cls(value)

    def set_timesteps(self, count, *, device=None):
        self.timesteps = torch.arange(count - 1, -1, -1, device=device)

    def step(self, noise, timestep, sample):
        return SimpleNamespace(prev_sample=sample - .1 * noise)


def _stage(kind):
    row = RowWiseRGBNormalViewAttentionV1(
        latent_channels=4, hidden_width=8, heads=2,
    ) if kind == STAGE_MULTIVIEW else None
    return RSSvNextPinnedDiffusionStageV1(
        stage=kind, denoiser=UNet2DConditionModel(), vae=AutoencoderKL(),
        conditioner=PairedSourceDiffusionConditionerV1(
            source_width=32, conditioning_width=32, heads=4,
        ),
        alpha_head=nn.Conv2d(3, 1, 1),
        scheduler=DDIMScheduler(),
        rowwise_attention=row,
    )


def _model():
    cfg = replace(RSSvNextConfig.tiny_for_test(), constructed_total_views=0)
    model = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).eval()
    install_rss_realsas_stage1_attention_lora_v1(model, rank=2, alpha=4.)
    with torch.no_grad():
        model.normal_condition_proj.weight.fill_(.001)
    return model


def test_single_bundle_roundtrip_restores_three_stage_weights_and_adapter(
    tmp_path, monkeypatch,
):
    torch.manual_seed(711)
    torch.set_num_threads(1)
    # Stub only external factory classes; actual bundled checkpoint functions,
    # stage specs, CAS digests, LoRA reconstruction and strict loading execute.
    stub = ModuleType("diffusers")
    stub.UNet2DConditionModel = UNet2DConditionModel
    stub.AutoencoderKL = AutoencoderKL
    stub.DDIMScheduler = DDIMScheduler
    monkeypatch.setitem(sys.modules, "diffusers", stub)
    model = _model()
    a, b = _stage(STAGE_CANONICAL), _stage(STAGE_MULTIVIEW)
    path = tmp_path / "single-neural-checkpoint.pt"
    digest = save_product_neural_bundle_v1(
        path, model=model, canonical=a, multiview=b,
        image_size=64, denoise_steps=4,
    )
    decoded = load_product_neural_bundle_v1(
        path, expected_sha256=digest,
        vision_encoder=RSSvNextPatchEncoderForTests(32),
        product_mode=False,
    )
    assert decoded.checkpoint_sha256 == digest
    assert decoded.model.construction_requires_generated_multiview
    assert decoded.model._realsas_lora_config == (2, 4.)
    for input_stage, restored in ((a, decoded.canonical), (b, decoded.multiview)):
        assert input_stage.stage == restored.stage
        assert set(input_stage._modules()) == set(restored._modules())
        for name, source in input_stage._modules().items():
            expected = source.state_dict()
            actual = restored._modules()[name].state_dict()
            assert set(expected) == set(actual)
            for key in expected:
                assert torch.equal(expected[key], actual[key]), (name, key)
    # The shape/inference entrypoint must recognize the ONE package, not
    # independently select or create IRIS/TESSA/AXIS/MIRA releases.
    returned = load_inference_checkpoint_v1(
        path, expected_sha256=digest,
        vision_encoder=RSSvNextPatchEncoderForTests(32),
        product_mode=False,
    )
    assert returned._pinned_neural_bundle.checkpoint_sha256 == digest
    assert returned.construction_requires_generated_multiview
    with pytest.raises(RuntimeError, match="PIN_OR_BYTES"):
        load_product_neural_bundle_v1(
            path, expected_sha256="0" * 64,
            vision_encoder=RSSvNextPatchEncoderForTests(32),
            product_mode=False,
        )
    with pytest.raises(RuntimeError, match="BUNDLE_PRODUCTION_VISION_REQUIRED"):
        load_inference_checkpoint_v1(
            path, expected_sha256=digest,
            vision_encoder=RSSvNextPatchEncoderForTests(32),
            product_mode=True,
        )
