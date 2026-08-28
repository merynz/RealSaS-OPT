from __future__ import annotations
import torch
import torch.nn.functional as F
from iris_losses import sample_pair_field


def cosine_topk(query: torch.Tensor, candidates: torch.Tensor, k: int):
    """query [N,D], candidates [M,D] -> scores/indices [N,k]."""
    q=F.normalize(query.float(),dim=-1); c=F.normalize(candidates.float(),dim=-1)
    scores=q@c.T
    return torch.topk(scores,k=min(k,c.shape[0]),dim=-1)


def reciprocal_mask(ab_index: torch.Tensor, ba_index: torch.Tensor) -> torch.Tensor:
    """ab_index[i]=j and ba_index[j]=i -> reciprocal."""
    i=torch.arange(ab_index.numel(),device=ab_index.device)
    valid=(ab_index>=0)&(ab_index<ba_index.numel())
    out=torch.zeros_like(valid)
    out[valid]=ba_index[ab_index[valid]]==i[valid]
    return out


def ambiguity_set(scores: torch.Tensor, indices: torch.Tensor, margin: float=0.035,
                  min_score: float=0.15, max_keep: int=4):
    """Set-valued policy: keep all near-best legal candidates; never force singleton."""
    if scores.numel()==0: return []
    best=float(scores[0])
    if best<min_score: return []
    keep=[]
    for s,i in zip(scores[:max_keep],indices[:max_keep]):
        if best-float(s)<=margin: keep.append((int(i),float(s)))
    return keep


def build_pair_scores(outputs, src_view, tgt_view, src_xy, tgt_xy,
                      coarse_weight=.60, fine_weight=.25, p_weight=.15):
    """Deterministic evidence combination for an already retained candidate set.

    No mechanical/owner truth enters here. P is common-frame observable geometry;
    Z is learned persistence evidence. This is a reranking primitive, not global identity authority.
    """
    zcs=sample_pair_field(outputs['Z_coarse'],src_view,src_xy)
    zct=sample_pair_field(outputs['Z_coarse'],tgt_view,tgt_xy)
    zfs=sample_pair_field(outputs['Z_fine'],src_view,src_xy)
    zft=sample_pair_field(outputs['Z_fine'],tgt_view,tgt_xy)
    ps=sample_pair_field(outputs['P'],src_view,src_xy)
    pt=sample_pair_field(outputs['P'],tgt_view,tgt_xy)
    sc=(F.normalize(zcs,dim=-1)*F.normalize(zct,dim=-1)).sum(-1)
    sf=(F.normalize(zfs,dim=-1)*F.normalize(zft,dim=-1)).sum(-1)
    sp=torch.exp(-torch.linalg.norm(ps-pt,dim=-1)/.025)
    return coarse_weight*sc+fine_weight*sf+p_weight*sp


def support_from_tracks(track_visible: torch.Tensor) -> torch.Tensor:
    return track_visible.to(torch.int32).sum(-1)
