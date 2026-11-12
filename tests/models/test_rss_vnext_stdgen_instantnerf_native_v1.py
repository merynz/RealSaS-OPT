from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from models.rss_vnext.stdgen_instantnerf_native_v1 import (
    StdGENNativeTriplaneTransformerV1, _StdGENNativeSynthesizer,
    _StdGENNativeCameraDINO, sample_stdgen_instantnerf_triplanes_v1,
)
from models.rss_vnext.stdgen_instantnerf_pretrain_court_v1 import (
    StdGENInstantNeRFArchitectureV1,
)


def _tiny():
    return replace(
        StdGENInstantNeRFArchitectureV1(),
        encoder_feature_dim=8, transformer_width=16,
        transformer_layers=2, transformer_heads=4,
        triplane_base_size=2, triplane_high_size=4,
        triplane_channels=5,
    )


def test_published_transformer_key_grammar_and_differentiable_planes():
    config = _tiny()
    module = StdGENNativeTriplaneTransformerV1(config)
    keys = module.state_dict()
    assert keys["pos_embed"].shape == (1,12,16)
    assert keys["deconv.weight"].shape == (16,5,2,2)
    assert keys["layers.1.self_attn.in_proj_weight"].shape == (48,16)
    assert keys["layers.1.cross_attn.q_proj_weight"].shape == (16,16)
    assert keys["layers.1.cross_attn.k_proj_weight"].shape == (16,8)
    context = torch.randn(2,7,8,requires_grad=True)
    planes = module(context)
    assert planes.shape == (2,3,5,4,4)
    planes.square().mean().backward()
    assert context.grad is not None and context.grad.abs().sum()>0
    assert module.pos_embed.grad is not None
    with pytest.raises(ValueError,match="IMAGE_FEATURE_CONTRACT"):
        module(torch.zeros((2,7,7)))


def test_pretrained_decoder_layout_and_point_query_have_shared_geometry_gradient():
    synth = _StdGENNativeSynthesizer(_tiny())
    keys = synth.state_dict()
    assert keys["decoder.net.0.weight"].shape == (64,15)
    assert keys["decoder.net.6.weight"].shape == (4,64)
    planes = torch.randn(1,3,5,4,4,requires_grad=True)
    xyz = torch.tensor([[[0.1,-.3,0.2],[0.2,.4,-.1]]],requires_grad=True)
    result = synth.forward_points(planes,xyz)
    assert result["rgb"].shape == (1,2,3)
    assert result["sigma"].shape == (1,2,1)
    assert torch.isfinite(result["rgb"]).all()
    (result["sigma"].square().mean()+result["rgb"].square().mean()).backward()
    assert planes.grad is not None and planes.grad.abs().sum()>0
    assert xyz.grad is not None and xyz.grad.abs().sum()>0
    with pytest.raises(ValueError,match="QUERY_DOMAIN"):
        synth.forward_points(planes,torch.zeros(1,1,2))


def test_camera_dino_has_adaln_at_every_layer_without_replacing_base_model():
    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.ModuleList([torch.nn.Module() for _ in range(12)])
    class FakeDINO(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config=SimpleNamespace(hidden_size=8)
            self.encoder=FakeEncoder()
    dino=FakeDINO()
    wrapper=_StdGENNativeCameraDINO(dino)
    assert wrapper.model is dino
    assert all(x.adaLN_modulation[1].out_features==32 for x in dino.encoder.layer)
    assert wrapper.camera_embedder[0].in_features==16


def test_pretrained_third_plane_is_zy_not_yz():
    planes = torch.zeros((1,3,5,4,4))
    # On the third plane, column position is Z; row position is Y.
    planes[0,2,0] = torch.arange(4).float()[None,:].expand(4,-1)
    points = torch.tensor([[[0.,0.,.6],[0.,.6,0.]]])
    sampled = sample_stdgen_instantnerf_triplanes_v1(planes,points)
    assert sampled.shape == (1,2,15)
    third_plane_first_feature = 2*5
    assert sampled[0,0,third_plane_first_feature] > sampled[0,1,third_plane_first_feature]
    assert sampled[0,0,third_plane_first_feature] > 2.0
    assert abs(float(sampled[0,1,third_plane_first_feature])-1.5) < 1.e-5
