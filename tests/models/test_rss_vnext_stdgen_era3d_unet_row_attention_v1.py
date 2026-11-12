from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch
from torch import nn

from models.rss_vnext.stdgen_era3d_unet_row_attention_v1 import (
    Era3DUNetFeatureRowAttentionV1,
    install_era3d_cleanroom_unet_row_attention_v1,
)


class Transformer2DModel(nn.Module):
    def __init__(self,ch=8):
        super().__init__()
        self.config=SimpleNamespace(in_channels=ch)
        self.proj=nn.Conv2d(ch,ch,1)
    def forward(self,hidden_states,return_dict=True):
        x=self.proj(hidden_states)
        if not return_dict: return (x,)
        return SimpleNamespace(sample=x)


class _MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(in_channels=4)
        self.conv_in=nn.Conv2d(4,8,3,padding=1)
        self.down=nn.ModuleList([Transformer2DModel(8)])
        self.mid=Transformer2DModel(8)
        self.up=nn.ModuleList([Transformer2DModel(8)])
        self.output=nn.Conv2d(8,4,1)
    def register_to_config(self,**kwargs):
        for key,value in kwargs.items(): setattr(self.config,key,value)
    def forward(self,x,return_dict=True):
        x=self.conv_in(x)
        x=self.down[0](x,return_dict=return_dict)
        x=x[0] if isinstance(x,tuple) else x.sample
        x=self.mid(x,return_dict=return_dict)
        x=x[0] if isinstance(x,tuple) else x.sample
        x=self.up[0](x,return_dict=return_dict)
        x=x[0] if isinstance(x,tuple) else x.sample
        return self.output(x)


def test_install_all_unet_feature_blocks_and_preserve_pretrained_logits_exactly():
    torch.manual_seed(18)
    net=_MiniUNet()
    x=torch.randn((12,4,2,3))
    expected=net(x).detach().clone()
    base_weight=net.conv_in.weight.detach().clone()
    installed=install_era3d_cleanroom_unet_row_attention_v1(
        net,hidden_width=8,
    )
    assert installed==("down.0","mid","up.0")
    assert net.rowwise_attention_in_unet_features is True
    assert net.config.in_channels==8
    assert torch.equal(net.conv_in.weight[:,:4],base_weight)
    assert torch.count_nonzero(net.conv_in.weight[:,4:])==0
    upgraded=net(torch.cat((x,torch.zeros_like(x)),dim=1))
    assert torch.allclose(upgraded,expected,atol=1e-6)
    with pytest.raises(ValueError,match="ALREADY_INSTALLED"):
        install_era3d_cleanroom_unet_row_attention_v1(net)


def test_rowwise_attention_transmits_between_views_and_normals_but_not_rows():
    torch.manual_seed(14)
    row=Era3DUNetFeatureRowAttentionV1(
        feature_channels=8,hidden_width=8,row_chunk=2,
    )
    original=torch.randn((12,8,4,3))
    assert torch.equal(row(original),original)
    with torch.no_grad():
        row.to_feature.weight[0,0]=1.
    before=row(original)
    edited=original.clone()
    edited[0,:,1,:]+=3.
    after=row(edited)
    assert not torch.allclose(before[7,:,1,:],after[7,:,1,:],atol=1e-7)
    assert torch.allclose(before[:,:,0,:],after[:,:,0,:],atol=1e-7)
    assert torch.allclose(before[:,:,2,:],after[:,:,2,:],atol=1e-7)


def test_rowwise_attention_gradients_and_memory_budget_guard():
    row=Era3DUNetFeatureRowAttentionV1(
        feature_channels=8,hidden_width=8,
    )
    value=torch.randn((12,8,2,3),requires_grad=True)
    loss=row(value).pow(2).mean()
    loss.backward()
    assert row.to_feature.weight.grad.abs().sum()>0
    assert value.grad.abs().sum()>0
    restricted=Era3DUNetFeatureRowAttentionV1(
        feature_channels=8,hidden_width=8,max_row_tokens=12,
    )
    with pytest.raises(ValueError,match="RESOURCE_OR_CAMERA_ALIGNMENT"):
        restricted(value)
    with pytest.raises(ValueError,match="RESOURCE_OR_CAMERA_ALIGNMENT"):
        row(value[:6])


def test_install_fail_closed_without_pretrained_2d_unet():
    class _Empty(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in=nn.Conv2d(4,8,3)
            self.config=SimpleNamespace(in_channels=4)
        def register_to_config(self,**kw): pass
    with pytest.raises(ValueError,match="NO_VALID_UNET_FEATURE_TRANSFORMERS"):
        install_era3d_cleanroom_unet_row_attention_v1(_Empty())
