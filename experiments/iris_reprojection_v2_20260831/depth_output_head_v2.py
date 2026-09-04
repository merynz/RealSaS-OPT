from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .ray_modes_v2 import RayModesV2

@dataclass
class DepthOutputV2:
    support_logits:torch.Tensor
    log_sigma:torch.Tensor
    support_probability:torch.Tensor
    conditional_support_probability:torch.Tensor|None=None
    mode_probability:torch.Tensor|None=None

class DepthSupportUncertaintyHeadV2(nn.Module):
    def __init__(self,hidden_dim:int=192):
        super().__init__(); self.head=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim//2),nn.GELU(),nn.Linear(hidden_dim//2,2))
    def forward(self,hidden:torch.Tensor,modes:RayModesV2)->DepthOutputV2:
        idx=modes.mode_indices.clamp_min(0); H=hidden.shape[-1]; gathered=torch.gather(hidden,2,idx[...,None].expand(-1,-1,-1,H)); raw=self.head(gathered); sl=raw[...,0]; ls=raw[...,1].clamp(-8.0,3.0)
        conditional=torch.sigmoid(sl); mode_probability=modes.mode_probabilities; sp=conditional*mode_probability
        sl=torch.where(modes.mode_valid,sl,torch.full_like(sl,-1e4)); ls=torch.where(modes.mode_valid,ls,torch.zeros_like(ls)); conditional=torch.where(modes.mode_valid,conditional,torch.zeros_like(conditional)); mode_probability=torch.where(modes.mode_valid,mode_probability,torch.zeros_like(mode_probability)); sp=torch.where(modes.mode_valid,sp,torch.zeros_like(sp)); return DepthOutputV2(sl,ls,sp,conditional,mode_probability)
