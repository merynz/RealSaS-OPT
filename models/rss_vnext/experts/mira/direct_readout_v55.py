from __future__ import annotations

"""Exact promoted MIRA V5.5 direct-carrier readout.

The module structure and forward equations are the productionized equivalent of
the sealed V6 `ArachneV6RawReadout` used by the MIRA-M V5.5 Run-All. Keeping the
readout in a stable model module removes runtime dependence on a historical
notebook/runner while preserving state-dict identity.
"""

import math

from torch import nn
import torch


MIRA_V55_RELATION_DIM = 256
MIRA_V55_DECODER_PARAMETER_COUNT = 1_619_491


class MIRADirectCarrierReadoutV55(nn.Module):
    def __init__(
        self,
        surface_dim: int,
        token_dim: int,
        pair_dim: int,
        geom_dim: int = 7,
        relation_dim: int = MIRA_V55_RELATION_DIM,
    ):
        super().__init__()
        d = int(relation_dim)
        self.surface_enc = nn.Sequential(
            nn.LayerNorm(surface_dim),
            nn.Linear(surface_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.geom_enc = nn.Sequential(
            nn.LayerNorm(geom_dim),
            nn.Linear(geom_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.token_enc = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.pair_enc = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
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

    def forward_rows(
        self,
        idx,
        surface_memory,
        geom,
        pair_geometry,
        field_tokens,
        legal,
    ):
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
        features = torch.cat(
            [qe, ctx, jm, pe, qe * ctx, qe * pe, ctx * pe],
            dim=-1,
        )
        logits = self.score(features).squeeze(-1)
        lm = legal[idx].bool()
        logits = logits.masked_fill(~lm, -1e4)
        pred = torch.softmax(logits, dim=-1) * lm.to(logits.dtype)
        pred = pred / pred.sum(-1, keepdim=True).clamp_min(1e-8)
        return logits, pred

    def decode_all(
        self,
        surface_memory,
        geom,
        pair_geometry,
        field_tokens,
        legal,
        *,
        chunk: int = 512,
    ):
        if int(chunk) <= 0:
            raise ValueError("MIRA_V55_DECODE_CHUNK_INVALID")
        logits = []
        pred = []
        for start in range(0, len(surface_memory), int(chunk)):
            idx = torch.arange(
                start,
                min(start + int(chunk), len(surface_memory)),
                device=surface_memory.device,
            )
            row_logits, row_pred = self.forward_rows(
                idx,
                surface_memory,
                geom,
                pair_geometry,
                field_tokens,
                legal,
            )
            logits.append(row_logits)
            pred.append(row_pred)
        if not logits:
            raise ValueError("MIRA_V55_EMPTY_CARRIER")
        return torch.cat(logits, dim=0), torch.cat(pred, dim=0)


def make_mira_v55_readout(*, surface_dim: int, token_dim: int, pair_dim: int):
    model = MIRADirectCarrierReadoutV55(
        surface_dim=surface_dim,
        token_dim=token_dim,
        pair_dim=pair_dim,
        geom_dim=7,
        relation_dim=MIRA_V55_RELATION_DIM,
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != MIRA_V55_DECODER_PARAMETER_COUNT:
        raise RuntimeError(f"MIRA_V55_DECODER_PARAMETER_COUNT_DRIFT:{count}")
    return model


__all__ = [
    "MIRA_V55_DECODER_PARAMETER_COUNT",
    "MIRA_V55_RELATION_DIM",
    "MIRADirectCarrierReadoutV55",
    "make_mira_v55_readout",
]
