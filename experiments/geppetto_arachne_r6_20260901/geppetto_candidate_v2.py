from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math

import torch
from torch import nn

try:
    from compiler.realsas_compiler_core.types import SkeletonProposalEdge, SkeletonProposalIR, SkeletonProposalJoint
except ImportError:
    from realsas_compiler_core.types import SkeletonProposalEdge, SkeletonProposalIR, SkeletonProposalJoint

from .geppetto_conditioning_v2 import GeppettoConditioningBatchV2


def _hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class GeppettoCandidateConfigV2:
    surface_feature_dim: int = 24
    model_dim: int = 192
    knn_k: int = 16
    local_layers: int = 2
    global_layers: int = 2
    decoder_layers: int = 2
    attention_heads: int = 6
    feedforward_dim: int = 576
    support_topk: int = 8
    position_modes: int = 3
    parent_pair_chunk: int = 64
    dropout: float = 0.0
    position_scale: float = 1.25
    stop_probability: float = 0.5
    support_presence_probability: float = 0.2
    resource_policy: str = "SURFACE_TOKEN_CARDINALITY_GUARD_V2"
    architecture_id: str = "RealSaS.GeppettoCandidate.LatentAutoregressiveSetProposal.v3"

    def validate(self) -> None:
        if self.surface_feature_dim != 24:
            raise ValueError("Geppetto V2 requires audited 24D conditioning")
        if self.model_dim <= 0 or self.model_dim % self.attention_heads:
            raise ValueError("model_dim/head mismatch")
        if min(self.knn_k, self.local_layers, self.global_layers, self.decoder_layers, self.feedforward_dim, self.support_topk, self.position_modes, self.parent_pair_chunk) <= 0:
            raise ValueError("invalid Geppetto V2 architecture cardinality")
        if self.position_modes < 2:
            raise ValueError("generic-strength Geppetto V2 requires multimodal locus capacity")
        if not (0.0 < self.stop_probability < 1.0) or not (0.0 <= self.support_presence_probability < 1.0):
            raise ValueError("invalid probability threshold")
        if not (0.0 <= self.dropout < 1.0) or self.position_scale <= 0:
            raise ValueError("invalid dropout/position scale")
        if self.resource_policy != "SURFACE_TOKEN_CARDINALITY_GUARD_V2":
            raise ValueError("Geppetto V2 resource policy drift")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


class _LocalGeometryAttention(nn.Module):
    def __init__(self, dim: int, k: int, dropout: float):
        super().__init__()
        self.k = int(k)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.rel = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.proj = nn.Linear(dim, dim)
        self.n1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 3), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 3, dim))
        self.n2 = nn.LayerNorm(dim)

    @staticmethod
    def _gather(x, idx):
        b = torch.arange(x.shape[0], device=x.device)[:, None, None]
        return x[b, idx]

    def forward(self, h, xyz, mask):
        B, N, D = h.shape
        d2 = torch.cdist(xyz.float(), xyz.float(), p=2).square()
        d2 = d2.masked_fill(~mask[:, None, :].bool(), float("inf"))
        k = min(max(1, self.k), N)
        idx = torch.topk(d2, k=k, dim=-1, largest=False, sorted=True).indices
        hn = self._gather(h, idx)
        xn = self._gather(xyz, idx)
        delta = xn - xyz[:, :, None, :]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        r = self.rel(torch.cat([delta, dist], dim=-1))
        q = self.q(h)[:, :, None, :]
        kk, vv = self.kv(hn).chunk(2, dim=-1)
        score = (q * (kk + r)).sum(-1) / math.sqrt(D)
        valid = self._gather(mask[..., None].float(), idx).squeeze(-1).bool()
        score = score.masked_fill(~valid, -1e4)
        attn = torch.softmax(score, dim=-1)
        msg = (attn[..., None] * (vv + r)).sum(2)
        h = self.n1(h + self.proj(msg))
        h = self.n2(h + self.ff(h))
        return h * mask[..., None].to(h.dtype)


class SurfaceSetEncoderV2(nn.Module):
    def __init__(self, cfg: GeppettoCandidateConfigV2):
        super().__init__()
        d = cfg.model_dim
        self.stem = nn.Sequential(nn.Linear(cfg.surface_feature_dim, d), nn.GELU(), nn.Linear(d, d))
        self.local = nn.ModuleList([_LocalGeometryAttention(d, cfg.knn_k, cfg.dropout) for _ in range(cfg.local_layers)])
        layer = nn.TransformerEncoderLayer(d, cfg.attention_heads, cfg.feedforward_dim, cfg.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.global_encoder = nn.TransformerEncoder(layer, cfg.global_layers, norm=nn.LayerNorm(d))

    def forward(self, features, positions, mask):
        h = self.stem(features) * mask[..., None].to(features.dtype)
        for layer in self.local:
            h = layer(h, positions, mask)
        h = self.global_encoder(h, src_key_padding_mask=~mask.bool()) * mask[..., None].to(h.dtype)
        denom = mask.sum(1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = h.sum(1) / denom
        return h, pooled


@dataclass
class GeppettoRawOutputV2:
    positions_normalized: torch.Tensor
    position_log_sigma: torch.Tensor
    existence_logits: torch.Tensor
    stop_logits: torch.Tensor
    root_logits: torch.Tensor
    support_presence_logits: torch.Tensor
    parent_logits: torch.Tensor
    support_logits: torch.Tensor
    control_states: torch.Tensor
    abstain_logits: torch.Tensor
    position_modes_normalized: torch.Tensor
    position_mode_log_sigma: torch.Tensor
    position_mode_logits: torch.Tensor


class GeppettoCandidateV2(nn.Module):
    """Dynamic-cardinality anonymous, multimodal control proposer.

    Sequence position is an internal generation coordinate, never product identity.
    There is no learned/frozen product max-K. Execution is bounded only by admitted
    surface support and explicit chunking. Each control carries multiple continuous
    locus hypotheses; the Compiler still owns canonical IDs, root and legal tree.

    Crucially, the emitted hard-MAP locus is not fed back into the autoregressive
    state. Recurrence is latent-state-only, so a discrete multimodal hypothesis
    crossover cannot rewrite every later anonymous control state.
    """

    def __init__(self, config: GeppettoCandidateConfigV2 = GeppettoCandidateConfigV2()):
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim
        M = config.position_modes
        self.encoder = SurfaceSetEncoderV2(config)
        self.start = nn.Parameter(torch.zeros(d))
        self.rel_query = nn.Linear(d * 2, d)
        self.cells = nn.ModuleList([nn.GRUCell(d * 3, d) for _ in range(config.decoder_layers)])
        self.position = nn.Linear(d, M * 3)
        self.log_sigma = nn.Linear(d, M * 3)
        self.position_mode_logits = nn.Linear(d, M)
        self.existence = nn.Linear(d, 1)
        self.stop = nn.Linear(d, 1)
        self.root = nn.Linear(d, 1)
        self.support_presence = nn.Linear(d, 1)
        self.abstain = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.parent_pair = nn.Sequential(nn.Linear(d * 2 + 4, d), nn.GELU(), nn.Linear(d, 1))
        self.support_q = nn.Linear(d, d, bias=False)
        self.support_k = nn.Linear(d, d, bias=False)

    def resource_limits(self, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.ndim != 2 or not valid_mask.any(dim=1).all():
            raise ValueError("each sample requires admitted surface support")
        return valid_mask.sum(dim=1).long()

    @staticmethod
    def _map_representative(modes: torch.Tensor, mode_log_sigma: torch.Tensor, mode_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a real hypothesis, never the mean between separated modes."""
        if modes.ndim != 3 or mode_log_sigma.shape != modes.shape or mode_logits.shape != modes.shape[:2]:
            raise ValueError("multimodal representative contract drift")
        B, M, C = modes.shape
        if C != 3 or M < 2:
            raise ValueError("expected at least two 3D locus modes")
        idx = torch.argmax(mode_logits, dim=-1)
        gather3 = idx[:, None, None].expand(B, 1, 3)
        pos = torch.gather(modes, 1, gather3).squeeze(1)
        log_sigma = torch.gather(mode_log_sigma, 1, gather3).squeeze(1)
        return pos, log_sigma, idx

    def _parent_logits_chunked(self, h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B, K, D = h.shape
        rows = []
        chunk = int(self.config.parent_pair_chunk)
        for start in range(0, K, chunk):
            end = min(K, start + chunk)
            C = end - start
            child_h = h[:, start:end, None, :].expand(B, C, K, D)
            parent_h = h[:, None, :, :].expand(B, C, K, D)
            delta = pos[:, None, :, :] - pos[:, start:end, None, :]
            dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
            rows.append(self.parent_pair(torch.cat([child_h, parent_h, delta, dist], dim=-1)).squeeze(-1))
        parent = torch.cat(rows, dim=1)
        parent = parent.masked_fill(torch.eye(K, device=h.device, dtype=torch.bool)[None], -1e4)
        return parent

    def _decode(self, memory, pooled, positions, mask, steps: int):
        B, N, D = memory.shape
        M = int(self.config.position_modes)
        limits = self.resource_limits(mask)
        if steps < 1 or int(steps) > int(limits.max().item()):
            raise ValueError("decode_steps exceeds surface-cardinality resource guard")
        states = [pooled.new_zeros((B, D)) for _ in self.cells]
        previous_states = []
        out = []
        for t in range(int(steps)):
            # Internal generation recurrence must not depend on whichever hard-MAP
            # locus happened to win a discrete multimodal confidence crossover.
            prev = self.start[None].expand(B, -1) if t == 0 else previous_states[-1]
            query = self.rel_query(torch.cat([pooled, prev], dim=-1))
            if previous_states:
                hist = torch.stack(previous_states, dim=1)
                score = torch.einsum("bd,btd->bt", query, hist) / math.sqrt(D)
                rel = torch.einsum("bt,btd->bd", torch.softmax(score, dim=-1), hist)
            else:
                rel = torch.zeros_like(pooled)
            x = torch.cat([pooled, prev, rel], dim=-1)
            for li, cell in enumerate(self.cells):
                states[li] = cell(x, states[li])
                x = torch.cat([pooled, states[li], rel], dim=-1)
            h = states[-1]
            modes = torch.tanh(self.position(h).reshape(B, M, 3)) * self.config.position_scale
            # Uncertainty calibration owns only the sigma head. The NLL may update
            # log_sigma parameters, but it must not rewrite the shared latent state
            # that also drives locus, STOP, root, parent and support evidence.
            mode_ls = self.log_sigma(h.detach()).reshape(B, M, 3).clamp(-8.0, 4.0)
            mode_logits = self.position_mode_logits(h)
            pos, rep_ls, _ = self._map_representative(modes, mode_ls, mode_logits)
            previous_states.append(h)
            out.append((h, pos, rep_ls, modes, mode_ls, mode_logits, self.existence(h).squeeze(-1), self.stop(h).squeeze(-1), self.root(h).squeeze(-1), self.support_presence(h).squeeze(-1)))
        fields = list(zip(*out))
        h, pos, ls, modes, mode_ls, mode_logits, ex, stop, root, sp = (torch.stack(x, 1) for x in fields)
        parent = self._parent_logits_chunked(h, pos)
        support = torch.einsum("bkd,bnd->bkn", self.support_q(h), self.support_k(memory)) / math.sqrt(D)
        support = support.masked_fill(~mask[:, None, :].bool(), -1e4)
        return GeppettoRawOutputV2(
            pos,
            ls,
            ex,
            stop,
            root,
            sp,
            parent,
            support,
            h,
            self.abstain(pooled).squeeze(-1),
            modes,
            mode_ls,
            mode_logits,
        )

    def forward(self, features, positions_normalized, valid_mask, *, decode_steps: int):
        if features.ndim != 3 or features.shape[-1] != self.config.surface_feature_dim or positions_normalized.shape != (*features.shape[:2], 3) or valid_mask.shape != features.shape[:2]:
            raise ValueError("Geppetto V2 input shape mismatch")
        memory, pooled = self.encoder(features, positions_normalized, valid_mask.bool())
        return self._decode(memory, pooled, positions_normalized, valid_mask.bool(), int(decode_steps))

    @torch.no_grad()
    def generate(self, features, positions_normalized, valid_mask, *, resource_step_limit: int | None = None):
        limits = self.resource_limits(valid_mask.bool())
        max_steps = int(limits.max().item()) if resource_step_limit is None else int(resource_step_limit)
        if max_steps < 1 or max_steps > int(limits.max().item()):
            raise ValueError("invalid resource_step_limit")
        out = self.forward(features, positions_normalized, valid_mask, decode_steps=max_steps)
        stop = torch.sigmoid(out.stop_logits)
        counts = []
        for b in range(features.shape[0]):
            limit = min(max_steps, int(limits[b].item()))
            hits = torch.nonzero(stop[b, :limit] >= self.config.stop_probability, as_tuple=False).flatten()
            counts.append(int(hits[0].item() + 1) if len(hits) else limit)
        return out, tuple(counts)

    @torch.no_grad()
    def propose(self, conditioning: GeppettoConditioningBatchV2, *, device=None, resource_step_limit: int | None = None):
        device = device or next(self.parameters()).device
        f = torch.as_tensor(conditioning.features, device=device, dtype=torch.float32)
        p = torch.as_tensor(conditioning.positions_normalized, device=device, dtype=torch.float32)
        m = torch.as_tensor(conditioning.valid_mask, device=device, dtype=torch.bool)
        self.eval()
        out, counts = self.generate(f, p, m, resource_step_limit=resource_step_limit)
        proposals = []
        for b, K in enumerate(counts):
            ids = {i: f"P:G2:{i:04d}" for i in range(K)}
            joints = []
            nvalid = len(conditioning.surface_ids[b])
            sp = torch.sigmoid(out.support_presence_logits[b, :K])
            ex = torch.sigmoid(out.existence_logits[b, :K])
            for i in range(K):
                pn = out.positions_normalized[b, i].cpu().numpy()
                pos = conditioning.normalizations[b].denormalize(pn[None])[0]
                sigma = torch.exp(out.position_log_sigma[b, i]).mean()
                support_ids = ()
                if float(sp[i]) >= self.config.support_presence_probability:
                    top = min(self.config.support_topk, nvalid)
                    idx = torch.topk(out.support_logits[b, i, :nvalid], k=top).indices.tolist()
                    support_ids = tuple(conditioning.surface_ids[b][j] for j in idx)
                mode_prob = torch.softmax(out.position_mode_logits[b, i], dim=-1)
                map_mode_index = int(torch.argmax(out.position_mode_logits[b, i]).item())
                hypotheses = []
                for mi in range(self.config.position_modes):
                    mpn = out.position_modes_normalized[b, i, mi].cpu().numpy()
                    mpos = conditioning.normalizations[b].denormalize(mpn[None])[0]
                    hypotheses.append({
                        "position": tuple(map(float, mpos)),
                        "probability": float(mode_prob[mi]),
                        "sigma_normalized": tuple(map(float, torch.exp(out.position_mode_log_sigma[b, i, mi]).cpu().tolist())),
                        "is_representative_map_mode": bool(mi == map_mode_index),
                    })
                confidence = float((ex[i] * torch.exp(-sigma)).clamp(0.0, 1.0))
                joints.append(SkeletonProposalJoint(
                    ids[i],
                    tuple(map(float, pos)),
                    float(torch.sigmoid(out.root_logits[b, i])),
                    confidence,
                    support_ids,
                    metadata={
                        "generation_index_internal_only": i,
                        "position_sigma": tuple(map(float, torch.exp(out.position_log_sigma[b, i]).cpu().tolist())),
                        "position_representative": "MAP_MODE",
                        "position_representative_mode_index": map_mode_index,
                        "position_hypotheses": tuple(hypotheses),
                        "position_hypotheses_schema": "RealSaS.GeppettoPositionHypotheses.v1",
                        "support_presence_probability": float(sp[i]),
                        "canonical_authority": False,
                    },
                ))
            edges = []
            for child in range(K):
                for parent in range(K):
                    if child == parent:
                        continue
                    score = float(torch.sigmoid(out.parent_logits[b, child, parent]))
                    edges.append(SkeletonProposalEdge(f"E:{ids[parent]}->{ids[child]}", ids[parent], ids[child], score, confidence=max(score, 1.0 - score), hard_required=False, hard_forbidden=False, reason="GEPPETTO_V2_SOFT_PARENT_EVIDENCE"))
            proposals.append(SkeletonProposalIR(
                tuple(joints),
                tuple(edges),
                conditioning.source_surface_hashes[b],
                model_provenance=self.config.config_hash,
                metadata={
                    "conditioning_hash": conditioning.conditioning_hashes[b],
                    "candidate_architecture": self.config.architecture_id,
                    "dynamic_cardinality": True,
                    "multimodal_loci": True,
                    "position_modes": self.config.position_modes,
                    "representative_locus_policy": "MAP_MODE",
                    "parent_pair_chunk": self.config.parent_pair_chunk,
                    "resource_policy": self.config.resource_policy,
                    "generated_count": K,
                    "abstain_probability": float(torch.sigmoid(out.abstain_logits[b])),
                    "generation_index_is_not_identity": True,
                    "compiler_owns_root_tree_ids": True,
                    "full_3d_reconstruction_claim": False,
                },
            ))
        return tuple(proposals)