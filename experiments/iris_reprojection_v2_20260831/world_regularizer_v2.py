from __future__ import annotations
import torch

def isotropic_world_regularizer_v2(q_points:torch.Tensor,score_logits:torch.Tensor,neighbor_pairs:torch.Tensor|None=None,valid_mask:torch.Tensor|None=None)->torch.Tensor:
    if q_points.ndim!=4 or score_logits.shape!=q_points.shape[:3]: raise ValueError("q/score shape mismatch")
    B,Q,D,_=q_points.shape
    if valid_mask is None: valid_mask=torch.ones_like(score_logits,dtype=torch.bool)
    if valid_mask.shape!=score_logits.shape: raise ValueError("valid_mask shape mismatch")
    if Q<2: return score_logits.sum()*0.0
    if neighbor_pairs is None: neighbor_pairs=torch.stack([torch.arange(Q-1,device=q_points.device),torch.arange(1,Q,device=q_points.device)],dim=-1)
    if neighbor_pairs.ndim!=2 or neighbor_pairs.shape[-1]!=2: raise ValueError("neighbor_pairs must be [E,2]")
    a,b=neighbor_pairs[:,0].long(),neighbor_pairs[:,1].long()
    if (a<0).any() or (b<0).any() or (a>=Q).any() or (b>=Q).any(): raise ValueError("neighbor index out of range")
    dist=torch.linalg.norm(q_points[:,a]-q_points[:,b],dim=-1).clamp_min(1e-8); pv=valid_mask[:,a]&valid_mask[:,b]
    if not pv.any(): return score_logits.sum()*0.0
    scale=dist[pv].median().clamp_min(1e-8); nd=(dist/scale).clamp_min(1e-4); delta=score_logits[:,a]-score_logits[:,b]; value=delta.square()/nd
    return value[pv].mean()
