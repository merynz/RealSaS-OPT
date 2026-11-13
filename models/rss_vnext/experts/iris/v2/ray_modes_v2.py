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

def extract_ray_modes_v2(
    score_logits:torch.Tensor,
    valid_depth:torch.Tensor|None=None,
    *,
    max_modes:int=3,
    min_probability:float=0.02,
    ambiguity_ratio:float=0.55,
    ambiguity_primary_probability_min:float=0.05,
)->RayModesV2:
    if score_logits.ndim!=3 or max_modes<1: raise ValueError("score logits must be [B,Q,D]")
    if not (0.0<=float(min_probability)<1.0): raise ValueError("min_probability must be in [0,1)")
    if not (0.0<=float(ambiguity_ratio)<=1.0): raise ValueError("ambiguity_ratio must be in [0,1]")
    if not (0.0<=float(ambiguity_primary_probability_min)<1.0): raise ValueError("ambiguity_primary_probability_min must be in [0,1)")
    B,Q,D=score_logits.shape
    valid=torch.ones_like(score_logits,dtype=torch.bool) if valid_depth is None else valid_depth.bool()
    if valid.shape!=score_logits.shape: raise ValueError("valid depth shape mismatch")
    masked=score_logits.masked_fill(~valid,-1e4)
    probs=torch.softmax(masked,dim=-1)*valid.to(score_logits.dtype)
    left=torch.full_like(masked,-1e4); right=torch.full_like(masked,-1e4)
    equal_left=torch.zeros_like(valid)
    if D>1:
        left[...,1:]=masked[...,:-1]
        right[...,:-1]=masked[...,1:]
        equal_left[...,1:]=valid[...,1:]&valid[...,:-1]&(masked[...,1:]==masked[...,:-1])
    # A flat plateau is one mode, not several. The leftmost depth sample is the
    # canonical representative because depth index is an analytic monotone axis.
    maxima=valid&(masked>=left)&(masked>=right)&(~equal_left)&(probs>=float(min_probability))
    candidate=probs.masked_fill(~maxima,-1.0)
    K=min(max_modes,D)
    # Stable sort gives a backend-independent depth-index tie break for equal
    # probabilities; torch.topk does not define an equal-value ordering contract.
    order=torch.argsort(candidate,dim=-1,descending=True,stable=True)
    indices=order[...,:K]
    values=torch.gather(candidate,-1,indices)
    mv=values>=0.0
    indices=torch.where(mv,indices,torch.full_like(indices,-1))
    values=torch.where(mv,values,torch.zeros_like(values))
    scores=torch.gather(masked,-1,indices.clamp_min(0))
    scores=torch.where(mv,scores,torch.full_like(scores,-1e4))
    if K>=2:
        primary_strong=values[...,0]>=float(ambiguity_primary_probability_min)
        ambiguous=mv[...,1]&primary_strong&(values[...,1]>=float(ambiguity_ratio)*values[...,0])
    else:
        ambiguous=torch.zeros((B,Q),dtype=torch.bool,device=score_logits.device)
    return RayModesV2(indices,scores,values,mv,ambiguous)
