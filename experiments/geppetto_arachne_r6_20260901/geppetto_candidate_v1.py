from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

try:
    from compiler.realsas_compiler_core.types import SkeletonProposalEdge, SkeletonProposalIR, SkeletonProposalJoint
except ImportError:
    from realsas_compiler_core.types import SkeletonProposalEdge, SkeletonProposalIR, SkeletonProposalJoint

from .candidate_config_v1 import GEPPETTO_V1, GeppettoCandidateConfigV1
from .conditioning_v1 import GeppettoConditioningBatchV1


@dataclass
class GeppettoRawOutputV1:
    existence_logits: torch.Tensor
    count_logits: torch.Tensor
    abstain_logits: torch.Tensor
    positions_normalized: torch.Tensor
    root_logits: torch.Tensor
    parent_logits: torch.Tensor
    support_logits: torch.Tensor
    query_embeddings: torch.Tensor


class _FourierPosition(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        if out_dim < 12:
            raise ValueError("Fourier out_dim too small")
        self.register_buffer("freq", torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32), persistent=False)
        self.proj = nn.Linear(3 * 2 * 4, out_dim)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        x = p[..., None] * self.freq
        x = torch.cat([torch.sin(math.pi * x), torch.cos(math.pi * x)], dim=-1).flatten(-2)
        return self.proj(x)


class GeppettoCandidateV1(nn.Module):
    """Unordered skeleton evidence proposer.

    Query index is an internal matching coordinate only. It is never a canonical
    joint identity and never survives Compiler qualification.
    """

    def __init__(self, config: GeppettoCandidateConfigV1 = GEPPETTO_V1):
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim
        self.surface_mlp = nn.Sequential(nn.Linear(config.surface_feature_dim, d), nn.GELU(), nn.Linear(d, d))
        self.position_embed = _FourierPosition(d)
        enc_layer = nn.TransformerEncoderLayer(d, config.attention_heads, config.feedforward_dim, config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, config.encoder_layers, norm=nn.LayerNorm(d))
        dec_layer = nn.TransformerDecoderLayer(d, config.attention_heads, config.feedforward_dim, config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(dec_layer, config.decoder_layers, norm=nn.LayerNorm(d))
        self.queries = nn.Parameter(torch.randn(config.max_joint_queries, d) * 0.02)
        self.existence = nn.Linear(d, 1)
        self.position = nn.Linear(d, 3)
        self.root = nn.Linear(d, 1)
        self.count = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.max_joint_queries + 1))
        self.abstain = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.parent_child = nn.Linear(d, d, bias=False)
        self.parent_parent = nn.Linear(d, d, bias=False)
        self.parent_bias = nn.Parameter(torch.zeros(config.max_joint_queries, config.max_joint_queries))
        self.support_q = nn.Linear(d, d, bias=False)
        self.support_k = nn.Linear(d, d, bias=False)

    def forward(self, features: torch.Tensor, positions_normalized: torch.Tensor, valid_mask: torch.Tensor) -> GeppettoRawOutputV1:
        if features.ndim != 3 or positions_normalized.shape[:2] != features.shape[:2] or positions_normalized.shape[-1] != 3:
            raise ValueError("invalid Geppetto input shapes")
        if valid_mask.shape != features.shape[:2]:
            raise ValueError("valid_mask shape mismatch")
        if features.shape[-1] != self.config.surface_feature_dim:
            raise ValueError("surface feature width mismatch")
        if not valid_mask.any(dim=1).all():
            raise ValueError("each sample requires at least one valid surface token")
        x = self.surface_mlp(features) + self.position_embed(positions_normalized)
        memory = self.encoder(x, src_key_padding_mask=~valid_mask.bool())
        q = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        decoded = self.decoder(q, memory, memory_key_padding_mask=~valid_mask.bool())
        masked = memory.masked_fill(~valid_mask[..., None].bool(), 0.0)
        pooled = masked.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(masked.dtype)
        existence = self.existence(decoded).squeeze(-1)
        positions = torch.tanh(self.position(decoded)) * self.config.position_scale
        roots = self.root(decoded).squeeze(-1)
        child = self.parent_child(decoded)
        parent = self.parent_parent(decoded)
        parent_logits = torch.einsum("bkd,bjd->bkj", child, parent) / math.sqrt(decoded.shape[-1]) + self.parent_bias
        eye = torch.eye(self.config.max_joint_queries, device=parent_logits.device, dtype=torch.bool).unsqueeze(0)
        parent_logits = parent_logits.masked_fill(eye, -1e4)
        support = torch.einsum("bkd,bnd->bkn", self.support_q(decoded), self.support_k(memory)) / math.sqrt(decoded.shape[-1])
        support = support.masked_fill(~valid_mask[:, None, :].bool(), -1e4)
        return GeppettoRawOutputV1(existence, self.count(pooled), self.abstain(pooled).squeeze(-1), positions, roots, parent_logits, support, decoded)

    @torch.no_grad()
    def propose(self, conditioning: GeppettoConditioningBatchV1, *, device: torch.device | str | None = None, min_existence_probability: float = 0.05) -> tuple[SkeletonProposalIR, ...]:
        device = device or next(self.parameters()).device
        f = torch.as_tensor(conditioning.features, device=device, dtype=torch.float32)
        p = torch.as_tensor(conditioning.positions_normalized, device=device, dtype=torch.float32)
        m = torch.as_tensor(conditioning.valid_mask, device=device, dtype=torch.bool)
        self.eval()
        out = self(f, p, m)
        results = []
        for b in range(f.shape[0]):
            count = int(torch.argmax(out.count_logits[b]).item())
            probs = torch.sigmoid(out.existence_logits[b])
            if count <= 0:
                count = 1 if float(probs.max()) >= min_existence_probability else 0
            count = min(count, self.config.max_joint_queries)
            selected = torch.argsort(probs, descending=True)[:count].tolist()
            if not selected:
                results.append(SkeletonProposalIR((), (), conditioning.source_surface_hashes[b], model_provenance=self.config.config_hash, metadata={"status": "ABSTAIN_NO_JOINTS", "abstain_probability": float(torch.sigmoid(out.abstain_logits[b]).item()), "conditioning_hash": conditioning.conditioning_hashes[b]}))
                continue
            q_to_pid = {q: f"P:JQ:{i:03d}" for i, q in enumerate(selected)}
            joints = []
            for q in selected:
                pos_norm = out.positions_normalized[b, q].detach().cpu().numpy()
                pos = conditioning.normalizations[b].denormalize(pos_norm[None])[0]
                n_valid = len(conditioning.surface_ids[b])
                topk = min(self.config.support_topk, n_valid)
                support_indices = torch.topk(out.support_logits[b, q, :n_valid], k=topk).indices.tolist()
                support_ids = tuple(conditioning.surface_ids[b][i] for i in support_indices)
                joints.append(SkeletonProposalJoint(q_to_pid[q], tuple(map(float, pos)), float(torch.sigmoid(out.root_logits[b, q]).item()), float(probs[q].item()), support_ids, metadata={"internal_query_index": int(q), "identity_semantics": "PROPOSAL_LOCAL_ONLY"}))
            edges = []
            for child_q in selected:
                for parent_q in selected:
                    if child_q == parent_q:
                        continue
                    score = float(torch.sigmoid(out.parent_logits[b, child_q, parent_q]).item())
                    edges.append(SkeletonProposalEdge(f"E:{q_to_pid[parent_q]}->{q_to_pid[child_q]}", q_to_pid[parent_q], q_to_pid[child_q], score, confidence=min(float(probs[parent_q]), float(probs[child_q])), hard_required=False, hard_forbidden=False, reason="GEPPETTO_SOFT_RELATIONAL_EVIDENCE_V1"))
            results.append(SkeletonProposalIR(tuple(joints), tuple(edges), conditioning.source_surface_hashes[b], model_provenance=self.config.config_hash, metadata={"conditioning_hash": conditioning.conditioning_hashes[b], "candidate_architecture": self.config.architecture_id, "query_indices_are_not_identity": True, "count_argmax": int(torch.argmax(out.count_logits[b]).item()), "abstain_probability": float(torch.sigmoid(out.abstain_logits[b]).item()), "full_3d_reconstruction_claim": False}))
        return tuple(results)
