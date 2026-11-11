from __future__ import annotations

import pytest
import torch

from models.mira.skin_fsq_codec_v6 import (
    FiniteScalarQuantizer,
    MIRASkinFSQCodecConfig,
    MIRASkinFSQCodecV6,
    MIRA_V6_CODEC_ID,
)


def test_fsq_mixed_radix_token_roundtrip():
    q = FiniteScalarQuantizer((3, 5, 7))
    ids = torch.arange(q.vocab_size, dtype=torch.long)
    indices = q.token_ids_to_indices(ids)
    assert torch.equal(q.indices_to_token_ids(indices), ids)
    codes = q.indices_to_codes(indices)
    assert codes.shape == (q.vocab_size, 3)
    assert torch.all(codes >= -1.0)
    assert torch.all(codes <= 1.0)


def test_mira_v6_codec_emits_per_joint_codes_and_simplex_reconstruction():
    torch.manual_seed(7)
    cfg = MIRASkinFSQCodecConfig(
        vertex_feature_dim=6,
        joint_feature_dim=5,
        hidden_dim=32,
        tokens_per_joint=3,
        levels=(3, 3, 3),
    )
    model = MIRASkinFSQCodecV6(cfg)
    assert model.architecture_id == MIRA_V6_CODEC_ID

    b, n, j = 2, 11, 4
    vertex = torch.randn(b, n, cfg.vertex_feature_dim)
    joint = torch.randn(b, j, cfg.joint_feature_dim)
    weights = torch.softmax(torch.randn(b, n, j), dim=-1)

    result = model(weights, vertex, joint)
    codes = result["token_ids"]
    recon = result["reconstructed_weights"]

    assert codes.shape == (b, j, cfg.tokens_per_joint)
    assert int(codes.min()) >= 0
    assert int(codes.max()) < cfg.vocab_size
    assert recon.shape == weights.shape
    assert torch.all(recon >= 0.0)
    assert torch.allclose(recon.sum(dim=-1), torch.ones(b, n), atol=1e-5)
    assert torch.isfinite(result["reconstruction_loss"])
    assert torch.isfinite(result["quantization_loss"])
    assert torch.isfinite(result["loss"])


def test_mira_v6_quantization_objective_carries_encoder_gradient():
    torch.manual_seed(13)
    cfg = MIRASkinFSQCodecConfig(4, 3, hidden_dim=24, tokens_per_joint=2, levels=(3, 5))
    model = MIRASkinFSQCodecV6(cfg)
    vertex = torch.randn(2, 9, 4)
    joint = torch.randn(2, 3, 3)
    weights = torch.softmax(torch.randn(2, 9, 3), dim=-1)
    result = model(weights, vertex, joint)
    result["loss"].backward()
    grad = model.latent_head.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_mira_v6_decode_is_code_conditioned_and_does_not_require_dense_teacher():
    torch.manual_seed(9)
    cfg = MIRASkinFSQCodecConfig(4, 3, hidden_dim=24, tokens_per_joint=2, levels=(5, 5))
    model = MIRASkinFSQCodecV6(cfg)
    vertex = torch.randn(1, 8, 4)
    joint = torch.randn(1, 3, 3)
    codes = torch.zeros(1, 3, 2, dtype=torch.long)
    decoded = model.decode_tokens(codes, vertex, joint)
    assert decoded.shape == (1, 8, 3)
    assert torch.allclose(decoded.sum(dim=-1), torch.ones(1, 8), atol=1e-5)


def test_mira_v6_rejects_negative_teacher_weight():
    cfg = MIRASkinFSQCodecConfig(2, 2, hidden_dim=16, tokens_per_joint=1, levels=(3, 3))
    model = MIRASkinFSQCodecV6(cfg)
    vertex = torch.randn(1, 3, 2)
    joint = torch.randn(1, 2, 2)
    bad = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [1.1, -0.1]]])
    with pytest.raises(ValueError, match="WEIGHT_NEGATIVE"):
        model.encode_tokens(bad, vertex, joint)


def test_mira_ste_reconstruction_matches_real_discrete_product_decoder_exactly():
    torch.manual_seed(29)
    cfg = MIRASkinFSQCodecConfig(
        vertex_feature_dim=7, joint_feature_dim=25,
        hidden_dim=32, tokens_per_joint=4, levels=(3, 3, 3),
    )
    model = MIRASkinFSQCodecV6(cfg)
    vertex = torch.randn(1, 9, 7)
    joint = torch.randn(1, 4, 25)
    skin = torch.softmax(torch.randn(1, 9, 4), dim=-1)
    trained = model(skin, vertex, joint)
    product = model.decode_tokens(trained["token_ids"], vertex, joint)
    # This is a critical SkinTokens-style STE obligation: no train/product
    # numerical gap while gradients still flow through the discrete codec.
    assert torch.equal(trained["reconstructed_weights"], product)
    assert trained["token_ids"].dtype == torch.long


def test_mira_dense_skin_reconstruction_loss_alone_trains_fsq_encoder():
    torch.manual_seed(31)
    cfg = MIRASkinFSQCodecConfig(
        vertex_feature_dim=7, joint_feature_dim=25,
        hidden_dim=32, tokens_per_joint=3, levels=(3, 5, 3),
    )
    model = MIRASkinFSQCodecV6(cfg)
    vertex = torch.randn(1, 17, 7)
    joint = torch.randn(1, 4, 25)
    target = torch.softmax(torch.randn(1, 17, 4), dim=-1)
    output = model(target, vertex, joint)
    # IMPORTANT: quantization_loss is deliberately omitted; an encoder trained
    # by that loss alone never learns whether its codes reconstruct source skin.
    output["reconstruction_loss"].backward()
    for param in (
        model.vertex_encode[0].weight,
        model.joint_encode[0].weight,
        model.latent_head.weight,
        # With centered FSQ token IDs, decoded code vectors can be all zero:
        # code_decode.weight then correctly has zero gradient while its bias
        # and the straight-through encoder still receive reconstruction signal.
        model.code_decode[0].bias,
    ):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0


def test_fsq_ste_derivative_matches_bounded_latent_not_unbounded_raw_z():
    q = FiniteScalarQuantizer((5,))
    large = torch.tensor([[-9.0], [9.0]], requires_grad=True)
    medium = torch.tensor([[0.1]], requires_grad=True)
    ste_large, ids_large, hard_large = q(large)
    ste_mid, ids_mid, hard_mid = q(medium)
    assert torch.equal(ste_large.detach(), hard_large)
    assert torch.equal(ste_mid.detach(), hard_mid)
    assert ids_large.tolist() == [0, 4]
    outer = torch.autograd.grad(ste_large.sum(), large)[0]
    center = torch.autograd.grad(ste_mid.sum(), medium)[0]
    assert torch.isfinite(outer).all() and torch.isfinite(center).all()
    assert float(outer.abs().max()) < 1e-5
    assert 0.9 < float(center[0, 0]) < 1.0
