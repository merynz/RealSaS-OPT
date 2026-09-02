from __future__ import annotations
import torch
from torch import nn
from .ray_modes_v2 import RayModesV2

class LocalDepthRefinementV2(nn.Module):
    def __init__(self,hidden_dim:int=192):
        super().__init__(); self.offset=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim//2),nn.GELU(),nn.Linear(hidden_dim//2,1))
    def forward(self,hidden:torch.Tensor,depth_values:torch.Tensor,modes:RayModesV2)->torch.Tensor:
        if hidden.shape[:3]!=depth_values.shape: raise ValueError("hidden/depth shape mismatch")
        idx=modes.mode_indices.clamp_min(0); H=hidden.shape[-1]; gh=torch.gather(hidden,2,idx[...,None].expand(-1,-1,-1,H)); base=torch.gather(depth_values,2,idx); D=depth_values.shape[-1]
        if D==1: width=torch.ones_like(base)*1e-3
        else:
            left=torch.gather(depth_values,2,(idx-1).clamp_min(0)); right=torch.gather(depth_values,2,(idx+1).clamp_max(D-1)); width=(right-left).abs().clamp_min(1e-6)*0.5
        refined=base+torch.tanh(self.offset(gh).squeeze(-1))*width*0.5
        return torch.where(modes.mode_valid,refined,torch.zeros_like(refined))
