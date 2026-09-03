from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class RayModesV2:
    mode_indices:torch.Tensor
    mode_scores:torch.Tensor
    mode_probabilities:torch.Tensor
    mode_valid:torch.Tensor
    ambiguous:torch.Tensor

def extract_ray_modes_v2(score_logits:torch.Tensor,valid_depth:torch.Tensor|None=None,*,max_modes:int=3,min_probability:float=0.02,ambiguity_ratio:float=0.55)->RayModesV2:
    if score_logits.ndim!=3 or max_modes<1: raise ValueError("score logits must be [B,Q,D]")
    B,Q,D=score_logits.shape; valid=torch.ones_like(score_logits,dtype=torch.bool) if valid_depth is None else valid_depth.bool()
    if valid.shape!=score_logits.shape: raise ValueError("valid depth shape mismatch")
    masked=score_logits.masked_fill(~valid,-1e4); probs=torch.softmax(masked,dim=-1)*valid.to(score_logits.dtype); left=torch.full_like(masked,-1e4); right=torch.full_like(masked,-1e4)
    if D>1: left[...,1:]=masked[...,:-1]; right[...,:-1]=masked[...,1:]
    maxima=valid&(masked>=left)&(masked>=right)&(probs>=float(min_probability)); candidate=probs.masked_fill(~maxima,-1.0); K=min(max_modes,D); values,indices=torch.topk(candidate,k=K,dim=-1); mv=values>=0.0
    indices=torch.where(mv,indices,torch.full_like(indices,-1)); values=torch.where(mv,values,torch.zeros_like(values)); scores=torch.gather(masked,-1,indices.clamp_min(0)); scores=torch.where(mv,scores,torch.full_like(scores,-1e4))
    ambiguous=mv[...,1]&(values[...,1]>=float(ambiguity_ratio)*values[...,0].clamp_min(1e-8)) if K>=2 else torch.zeros((B,Q),dtype=torch.bool,device=score_logits.device)
    return RayModesV2(indices,scores,values,mv,ambiguous)
