from __future__ import annotations

import torch

from models.skin_field_codec.v2.config_v2 import (
    EXPECTED_SKIN_FIELD_CODEC_V2_CONFIG_HASH,
    SKIN_FIELD_CODEC_V2,
)
from models.skin_field_codec.v2.skin_field_codec_v2 import SkinFieldCodecV2


EXPECTED_PARAMETER_COUNT = 414_146


def _count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_v2_frozen_config_and_parameter_count() -> None:
    model = SkinFieldCodecV2()
    assert SKIN_FIELD_CODEC_V2.config_hash == EXPECTED_SKIN_FIELD_CODEC_V2_CONFIG_HASH
    assert model.config.architecture_id == "RealSaS.SkinFieldCodec.ContinuousJointField.PairGeometry.v2"
    assert _count_trainable(model) == EXPECTED_PARAMETER_COUNT
    assert model.pair_geometry_embed[0].in_features == 10
    assert model.pair_geometry_embed[0].out_features == 64
    assert model.decoder.net[0].in_features == 512


def test_pair_geometry_is_decoder_only_and_causal() -> None:
    torch.manual_seed(20260908)
    model = SkinFieldCodecV2().eval()
    b, n, j = 1, 7, 4
    sf = torch.randn(b, n, 20)
    jf = torch.randn(b, j, 8)
    sm = torch.ones(b, n, dtype=torch.bool)
    jm = torch.ones(b, j, dtype=torch.bool)
    teacher = torch.rand(b, n, j)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True)

    # The teacher encoder has no pair_geometry argument by contract.
    latent = model.encode_teacher_weights(sf, jf, teacher, sm, jm)
    pg0 = torch.zeros(b, n, j, 10)
    pg1 = pg0.clone()
    pg1[..., 0] = 1.0
    w0, logits0 = model.decode_from_latents(latent, sf, jf, pg0, sm, jm)
    w1, logits1 = model.decode_from_latents(latent, sf, jf, pg1, sm, jm)

    assert latent.shape == (b, j, 64)
    assert w0.shape == w1.shape == (b, n, j)
    assert logits0.shape == logits1.shape == (b, n, j)
    assert not torch.equal(logits0, logits1)
    assert torch.allclose(w0.sum(dim=-1), torch.ones(b, n), atol=1e-6, rtol=0.0)
    assert torch.allclose(w1.sum(dim=-1), torch.ones(b, n), atol=1e-6, rtol=0.0)


def test_pair_geometry_shape_and_finite_guards() -> None:
    model = SkinFieldCodecV2().eval()
    sf = torch.zeros(1, 2, 20)
    jf = torch.zeros(1, 3, 8)
    sm = torch.ones(1, 2, dtype=torch.bool)
    jm = torch.ones(1, 3, dtype=torch.bool)
    latent = torch.zeros(1, 3, 64)

    bad_shape = torch.zeros(1, 2, 3, 9)
    try:
        model.decode_from_latents(latent, sf, jf, bad_shape, sm, jm)
    except ValueError as exc:
        assert "pair_geometry shape mismatch" in str(exc)
    else:
        raise AssertionError("bad pair geometry width was accepted")

    bad_finite = torch.zeros(1, 2, 3, 10)
    bad_finite[0, 0, 0, 0] = float("nan")
    try:
        model.decode_from_latents(latent, sf, jf, bad_finite, sm, jm)
    except ValueError as exc:
        assert "pair_geometry must be finite" in str(exc)
    else:
        raise AssertionError("non-finite pair geometry was accepted")
