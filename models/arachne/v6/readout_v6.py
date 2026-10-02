"""Inference-only home of the exact sealed V6 readout; no training/teacher imports.

Class body copied unchanged from SHA256
39ff4f7a2051f0d9c1251a5999f472b9926e83deacbedcedc8e92eea424dce01.
The inference runner verifies AST identity against that sealed source.
"""
import math
import torch
from torch import nn
RELATION_DIM = 256

class ArachneV6RawReadout(nn.Module):
    """V6 readout selected by the Knight causal oracle."""

    def __init__(
        self,
        surface_dim: int,
        token_dim: int,
        pair_dim: int,
        geom_dim: int = 7,
        relation_dim: int = RELATION_DIM,
    ):
        super().__init__()
        d = int(relation_dim)
        self.surface_enc = nn.Sequential(
            nn.LayerNorm(surface_dim), nn.Linear(surface_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        self.geom_enc = nn.Sequential(
            nn.LayerNorm(geom_dim), nn.Linear(geom_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        self.token_enc = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        self.pair_enc = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        self.query_norm = nn.LayerNorm(d)
        self.score = nn.Sequential(
            nn.LayerNorm(d * 7),
            nn.Linear(d * 7, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward_rows(self, idx, surface_memory, geom, pair_geometry, field_tokens, legal):
        s = self.surface_enc(surface_memory[idx].float())
        g = self.geom_enc(geom[idx].float())
        q = self.query_norm(s + g)

        te = self.token_enc(field_tokens.float())
        scores = torch.einsum("rd,jkd->rjk", q, te) / math.sqrt(float(q.shape[-1]))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("rjk,jkd->rjd", attn, te)

        jm = te.mean(1)[None].expand(len(idx), -1, -1)
        pe = self.pair_enc(pair_geometry[idx].float())
        qe = q[:, None, :].expand_as(ctx)
        features = torch.cat([qe, ctx, jm, pe, qe * ctx, qe * pe, ctx * pe], dim=-1)
        logits = self.score(features).squeeze(-1)

        lm = legal[idx].bool()
        logits = logits.masked_fill(~lm, -1e4)
        pred = torch.softmax(logits, dim=-1) * lm.to(logits.dtype)
        pred = pred / pred.sum(-1, keepdim=True).clamp_min(1e-8)
        return logits, pred

    def decode_all(self, surface_memory, geom, pair_geometry, field_tokens, legal, chunk: int = 1024):
        zs, ps = [], []
        n = int(surface_memory.shape[0])
        for start in range(0, n, int(chunk)):
            stop = min(start + int(chunk), n)
            idx = torch.arange(start, stop, device=surface_memory.device, dtype=torch.long)
            z, p = self.forward_rows(idx, surface_memory, geom, pair_geometry, field_tokens, legal)
            zs.append(z.float())
            ps.append(p.float())
        return torch.cat(zs, 0), torch.cat(ps, 0)
