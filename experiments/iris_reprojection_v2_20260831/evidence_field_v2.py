from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .q_domain_v2 import RayHypothesisDomainV2
from .q_evidence_encoder_v2 import QEvidenceEncodingV2

@dataclass
class EvidenceFieldOutputV2:
    score_logits: torch.Tensor
    hidden: torch.Tensor
    support_fraction: torch.Tensor

class EvidenceFieldV2(nn.Module):
    def __init__(self,hidden_dim:int=192):
        super().__init__(); self.fuse=nn.Sequential(nn.Linear(hidden_dim+4,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim),nn.GELU()); self.score=nn.Linear(hidden_dim,1)
    def forward(self,evidence:QEvidenceEncodingV2,domain:RayHypothesisDomainV2)->EvidenceFieldOutputV2:
        sf=evidence.valid_views.float().mean(dim=-1); q=domain.q_points.to(evidence.pooled.dtype); radius=torch.linalg.norm(q,dim=-1,keepdim=True); h=self.fuse(torch.cat([evidence.pooled,q,radius],dim=-1)); logits=self.score(h).squeeze(-1).masked_fill(~evidence.valid_views.any(dim=-1),-1e4); return EvidenceFieldOutputV2(logits,h,sf)
