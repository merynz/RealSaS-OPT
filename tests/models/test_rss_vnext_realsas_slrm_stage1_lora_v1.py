from __future__ import annotations

import math
import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig,RSSvNextPatchEncoderForTests,RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.realsas_slrm_stage1_lora_v1 import (
    RealSaSAttentionWeightLoRAV1,install_rss_realsas_stage1_attention_lora_v1,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(16)
    torch.set_num_threads(1)


def _model():
    cfg=RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )


def _scene(net):
    image=torch.rand((1,6,4,16,16))
    camera=torch.zeros((1,6,13))
    for i,deg in enumerate((-90,-45,0,45,90,180)):
        a=math.radians(deg)
        camera[:,i,3]=math.cos(a)
        camera[:,i,5]=-math.sin(a)
        camera[:,i,7]=1.
        camera[:,i,9]=math.sin(a)
        camera[:,i,11]=math.cos(a)
        camera[:,i,12]=1.
    tokens,xy=net.vision.forward_variable_views(image)
    return net.iris.encode_scene(
        net.early_semantics.forward_variable_views(tokens).conditioned_image_tokens,xy,camera,
    )


def test_zero_init_attention_lora_preserves_existing_scene_forward():
    torch.manual_seed(21)
    net=_model().eval()
    torch.manual_seed(41)
    baseline=_scene(net).detach().clone()
    receipt=install_rss_realsas_stage1_attention_lora_v1(net,rank=4,alpha=4.)
    torch.manual_seed(41)
    after=_scene(net)
    assert torch.allclose(baseline,after,atol=1.e-6)
    assert receipt.adapter_architecture=="RSS_IRIS_TRANSFORMER_NOT_INSTANTNERF_COMPATIBLE"
    assert not receipt.pretrained_checkpoint_loaded_and_verified
    assert not receipt.original_instantnerf_weight_compatibility_verified
    assert receipt.stage1_trainable_parameter_count>0
    assert receipt.stage1_frozen_parameter_count>0
    assert any("self_attn" in x for x in receipt.attention_module_paths)
    assert any("multihead_attn" in x for x in receipt.attention_module_paths)
    with pytest.raises(ValueError,match="MODEL_OR_REINSTALL"):
        install_rss_realsas_stage1_attention_lora_v1(net)


def test_true_stage1_lora_semantic_gradients_but_old_geometry_graphics_frozen():
    net=_model().train()
    receipt=install_rss_realsas_stage1_attention_lora_v1(net,rank=4,alpha=4.)
    optimizer=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=1.e-3)
    planes=_scene(net)
    local=net.iris.query(planes,torch.rand((1,6,3)))["H"]
    semantic=net.presentation_role_head(local)
    loss=planes.square().mean()+semantic[...,0].square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    adapted=[
        p for name,p in net.named_parameters()
        if "parametrizations" in name and ".up" in name
    ]
    assert adapted and any(p.grad is not None and p.grad.abs().sum()>0 for p in adapted)
    assert net.presentation_role_head[-1].weight.grad.abs().sum()>0
    assert net.iris.scene.scene_queries.grad is not None
    assert net.iris.scene.upsample.weight.grad is not None
    assert all(p.grad is None for p in net.graphics.parameters())
    assert all(not p.requires_grad for p in net.graphics.parameters())
    assert all(p.grad is None for p in net.iris.field.parameters())
    optimizer.step()
    assert receipt.rank==4


def test_invalid_rank_or_double_install_fail_closed():
    with pytest.raises(ValueError,match="WEIGHT_SHAPE_OR_RANK"):
        RealSaSAttentionWeightLoRAV1((4,4),rank=16,alpha=16.)
