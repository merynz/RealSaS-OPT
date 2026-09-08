from __future__ import annotations

from dataclasses import replace
import inspect
import torch

from models.skin_field_codec.v3 import FiniteScalarQuantizerV3, SkinFieldCodecV3, SkinTokensStrengthConfigV3, scalar_field_loss_v3

EXPECTED_CONFIG_HASH = "50a8e4d10f8ab2912c9f8cbb6922046f5b126571935dfc532072e4b5ee89c382"
EXPECTED_PARAM_COUNT = 273_296_903


def tiny_config() -> SkinTokensStrengthConfigV3:
    return replace(SkinTokensStrengthConfigV3(), field_tokens=2, condition_tokens=8, latent_channels=32, encoder_width=32, decoder_width=64, attention_heads=4, encoder_layers=2, decoder_layers=3, ffn_ratio=2, fsq_levels=(4,4,4), architecture_id="TEST_ONLY.RealSaS.SkinFieldCodec.v3", strict_strength_contract=False)


def test_full_strength_static_contract():
    cfg = SkinTokensStrengthConfigV3(); cfg.validate()
    assert cfg.config_hash == EXPECTED_CONFIG_HASH
    assert (cfg.field_tokens, cfg.condition_tokens, cfg.latent_channels) == (4,384,512)
    assert (cfg.encoder_width, cfg.decoder_width, cfg.encoder_layers, cfg.decoder_layers, cfg.attention_heads, cfg.ffn_ratio) == (512,1024,8,16,8,4)
    assert (cfg.xyz_frequency_dim, cfg.geometry_embed_input_dim, cfg.field_embed_input_dim) == (51,55,56)
    assert cfg.fsq_levels == (8,8,8,5,5,5) and cfg.fsq_codebook_size == 64000
    model = SkinFieldCodecV3(cfg)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAM_COUNT
    assert model.field_queries.shape == (4,512)
    assert len(model.field_encoder.blocks) == 8 and len(model.condition_encoder.blocks) == 8 and len(model.decoder.memory_blocks) == 16
    assert (model.decoder.memory_in.in_features, model.decoder.memory_in.out_features) == (512,1024)
    assert (model.fsq.project_in.in_features, model.fsq.project_in.out_features, model.fsq.codebook_size) == (512,6,64000)


def test_codec_decoder_has_no_joint_or_pair_geometry_lane():
    sig = inspect.signature(SkinFieldCodecV3.decode_field)
    assert tuple(sig.parameters) == ("self","quantized_field_tokens","condition_tokens","query_geometry")
    src = inspect.getsource(SkinFieldCodecV3.decode_field)
    assert "joint_features" not in src and "pair_geometry" not in src


def test_teacher_information_firewall_and_tiny_backward():
    torch.manual_seed(20260908); model = SkinFieldCodecV3(tiny_config()); cfg=model.config
    B,N,M=1,24,12; geom=torch.randn(B,N,7); geom[...,6]=1.; qidx=torch.arange(cfg.condition_tokens)[None]
    obs_a=torch.cat([geom[:,:M],torch.zeros(B,M,1)],-1); obs_b=torch.cat([geom[:,:M],torch.ones(B,M,1)],-1)
    cond_a=model.encode_condition(geom,qidx); cond_b=model.encode_condition(geom.clone(),qidx)
    assert torch.equal(cond_a,cond_b)
    cont_a,_,ids_a=model.encode_field(obs_a); cont_b,quant_b,ids_b=model.encode_field(obs_b)
    assert not torch.equal(cont_a,cont_b)
    assert ids_a.min()>=0 and ids_a.max()<model.fsq.codebook_size and ids_b.min()>=0 and ids_b.max()<model.fsq.codebook_size
    pred=model.decode_field(quant_b,cond_b,geom); truth=torch.rand_like(pred)
    loss=scalar_field_loss_v3(pred,truth,torch.ones_like(pred,dtype=torch.bool))["total"]; loss.backward()
    assert torch.isfinite(loss) and sum(p.grad is not None for p in model.parameters())>0


def test_fsq_roundtrip_shape_and_cardinality():
    torch.manual_seed(7); q=FiniteScalarQuantizerV3(32,(4,4,4)); z=torch.randn(2,5,32); zq,ids=q(z)
    assert zq.shape==z.shape and ids.shape==(2,5) and int(ids.min())>=0 and int(ids.max())<64
    assert q.indices_to_codes(ids).shape==z.shape
