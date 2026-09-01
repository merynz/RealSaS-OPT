from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from realsas_compiler_core.types import (
    RiggingSurfaceIR,
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
)


@dataclass(frozen=True)
class GeppettoDemoConfig:
    max_joints: int = 64
    token_dim: int = 128
    hidden_dim: int = 256
    encoder_layers: int = 3
    query_layers: int = 3
    heads: int = 8
    support_k: int = 8
    existence_threshold: float = 0.5


@dataclass
class GeppettoRawOutput:
    position_norm: torch.Tensor
    existence_logits: torch.Tensor
    root_logits: torch.Tensor
    edge_logits: torch.Tensor


def _surface_tensor(surface: RiggingSurfaceIR, *, device: torch.device) -> tuple[torch.Tensor, tuple[str, ...]]:
    if not surface.surface_nodes:
        raise ValueError("Geppetto requires a non-empty RiggingSurfaceIR")
    ids = tuple(node.surface_id for node in surface.surface_nodes)
    p = torch.tensor([node.P for node in surface.surface_nodes], dtype=torch.float32, device=device)
    if not torch.isfinite(p).all():
        raise ValueError("surface contains non-finite positions")
    return p, ids


def _normalize_points(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pmin = p.amin(dim=-2, keepdim=True)
    pmax = p.amax(dim=-2, keepdim=True)
    center = 0.5 * (pmin + pmax)
    scale = (pmax - pmin).amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return (p - center) / scale, center, scale


class SurfacePointEncoder(nn.Module):
    def __init__(self, cfg: GeppettoDemoConfig):
        super().__init__()
        d = cfg.token_dim
        self.in_proj = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.encoder_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, p_norm: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(self.in_proj(p_norm)))


class GeppettoDemoV1(nn.Module):
    """Generic demo-only learned skeleton evidence proposer.

    Canonical joint IDs and the final tree remain Compiler-owned.
    """

    def __init__(self, cfg: GeppettoDemoConfig = GeppettoDemoConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.token_dim
        self.surface_encoder = SurfacePointEncoder(cfg)
        self.queries = nn.Parameter(torch.empty(cfg.max_joints, d))
        nn.init.trunc_normal_(self.queries, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=cfg.heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.query_layers)
        self.out_norm = nn.LayerNorm(d)
        self.position_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        self.existence_head = nn.Linear(d, 1)
        self.root_head = nn.Linear(d, 1)
        self.edge_parent = nn.Linear(d, d, bias=False)
        self.edge_child = nn.Linear(d, d, bias=False)
        self.edge_bias = nn.Parameter(torch.zeros(()))

    def forward(self, surface_points: torch.Tensor) -> GeppettoRawOutput:
        if surface_points.ndim != 3 or surface_points.shape[-1] != 3:
            raise ValueError("surface_points must be [B,N,3]")
        p_norm, _, _ = _normalize_points(surface_points)
        memory = self.surface_encoder(p_norm)
        q = self.queries.unsqueeze(0).expand(surface_points.shape[0], -1, -1)
        z = self.out_norm(self.decoder(q, memory))
        pos = 0.75 * torch.tanh(self.position_head(z))
        existence = self.existence_head(z).squeeze(-1)
        root = self.root_head(z).squeeze(-1)
        pfeat = self.edge_parent(z)
        cfeat = self.edge_child(z)
        edge = torch.einsum("bqd,bkd->bqk", pfeat, cfeat) / (z.shape[-1] ** 0.5) + self.edge_bias
        eye = torch.eye(edge.shape[-1], dtype=torch.bool, device=edge.device).unsqueeze(0)
        edge = edge.masked_fill(eye, -30.0)
        return GeppettoRawOutput(pos, existence, root, edge)

    @torch.no_grad()
    def propose(self, surface: RiggingSurfaceIR, *, device: torch.device | str = "cpu") -> SkeletonProposalIR:
        device = torch.device(device)
        self.eval()
        p, surface_ids = _surface_tensor(surface, device=device)
        _, center, scale = _normalize_points(p.unsqueeze(0))
        raw = self(p.unsqueeze(0))
        exist_prob = torch.sigmoid(raw.existence_logits[0])
        active = torch.nonzero(exist_prob >= self.cfg.existence_threshold, as_tuple=False).flatten().tolist()
        if not active:
            active = [int(torch.argmax(exist_prob).item())]
        active = sorted(active)

        position = raw.position_norm[0] * scale[0] + center[0]
        joints: list[SkeletonProposalJoint] = []
        for qi in active:
            jp = position[qi]
            d2 = torch.sum((p - jp) ** 2, dim=-1)
            k = min(self.cfg.support_k, len(surface_ids))
            nearest = torch.topk(d2, k=k, largest=False).indices.tolist()
            joints.append(
                SkeletonProposalJoint(
                    proposal_id=f"GQ:{qi:03d}",
                    position=tuple(float(x) for x in jp.cpu().tolist()),
                    root_score=float(torch.sigmoid(raw.root_logits[0, qi]).item()),
                    confidence=float(exist_prob[qi].item()),
                    support_surface_ids=tuple(surface_ids[j] for j in nearest),
                    metadata={"query_index": qi, "demo_model": "GeppettoDemoV1"},
                )
            )

        edges: list[SkeletonProposalEdge] = []
        for parent_q in active:
            for child_q in active:
                if parent_q == child_q:
                    continue
                score = float(torch.sigmoid(raw.edge_logits[0, parent_q, child_q]).item())
                edges.append(
                    SkeletonProposalEdge(
                        edge_id=f"GE:{parent_q:03d}>{child_q:03d}",
                        parent_proposal_id=f"GQ:{parent_q:03d}",
                        child_proposal_id=f"GQ:{child_q:03d}",
                        score=score,
                        confidence=max(0.0, min(1.0, abs(score - 0.5) * 2.0)),
                        reason="GEPPETTO_DEMO_LEARNED_PARENT_EVIDENCE",
                        metadata={"parent_query": parent_q, "child_query": child_q},
                    )
                )

        return SkeletonProposalIR(
            joints=tuple(joints),
            edges=tuple(edges),
            surface_binding_hash=surface.geometry_lineage_hash,
            model_provenance="RealSaS.GeppettoDemoV1.experimental",
            metadata={
                "generalization_claim": False,
                "canonical_ids_predicted": False,
                "final_tree_owned_by_compiler": True,
                "active_query_indices": tuple(active),
            },
        )
