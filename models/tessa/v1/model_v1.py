from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .conditioning_v1 import TESSA_SURFACE_FEATURE_DIM_V1
from .contracts_v1 import TESSAScalingPolicyV1


@dataclass(frozen=True)
class TESSAConfigV1:
    input_dim: int = TESSA_SURFACE_FEATURE_DIM_V1
    d_model: int = 768
    n_heads: int = 12
    surface_layers: int = 6
    decoder_layers: int = 12
    mlp_ratio: int = 4
    dropout: float = 0.0
    coordinate_bins: int = 1024
    surface_latent_count: int = 384
    local_attention_window: int = 2048
    query_chunk_size: int = 256
    max_faces: int = 32768
    max_vertices: int = 32768
    max_faces_per_chart: int = 4096

    @property
    def special_token_count(self) -> int:
        return 8

    @property
    def vocab_size(self) -> int:
        return self.coordinate_bins + self.special_token_count

    @property
    def BOS(self) -> int:
        return self.coordinate_bins

    @property
    def COMPONENT_BEGIN(self) -> int:
        return self.coordinate_bins + 1

    @property
    def CHART_BEGIN(self) -> int:
        return self.coordinate_bins + 2

    @property
    def FACE_BREAK(self) -> int:
        return self.coordinate_bins + 3

    @property
    def CHART_END(self) -> int:
        return self.coordinate_bins + 4

    @property
    def COMPONENT_END(self) -> int:
        return self.coordinate_bins + 5

    @property
    def EOS(self) -> int:
        return self.coordinate_bins + 6

    @property
    def PAD(self) -> int:
        return self.coordinate_bins + 7

    def scaling_policy(self) -> TESSAScalingPolicyV1:
        policy = TESSAScalingPolicyV1(
            max_faces=self.max_faces,
            max_vertices=self.max_vertices,
            local_attention_window=self.local_attention_window,
            query_chunk_size=self.query_chunk_size,
            surface_latent_count=self.surface_latent_count,
            coordinate_bins=self.coordinate_bins,
        )
        policy.validate()
        if self.max_faces_per_chart < 256 or self.max_faces_per_chart > self.max_faces:
            raise ValueError("TESSA_MAX_FACES_PER_CHART_INVALID")
        return policy


@dataclass
class TESSAOutputV1:
    logits: torch.Tensor
    surface_latents: torch.Tensor
    loss: torch.Tensor | None = None


class TESSASurfaceEncoderV1(nn.Module):
    """Permutation-tolerant GSA surface encoder with fixed latent bottleneck.

    Complexity of the expensive latent stack is independent of raw GSA point
    count.  Cross-attention is O(N * K), where K is the fixed latent count.
    """

    def __init__(self, cfg: TESSAConfigV1):
        super().__init__()
        self.cfg = cfg
        self.point_stem = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.latents = nn.Parameter(
            torch.randn(cfg.surface_latent_count, cfg.d_model) / math.sqrt(cfg.d_model)
        )
        self.cross_norm = nn.LayerNorm(cfg.d_model)
        self.cross = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.mlp_ratio,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_stack = nn.TransformerEncoder(layer, num_layers=cfg.surface_layers)
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        surface_features: torch.Tensor,
        surface_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if surface_features.ndim != 3 or surface_features.shape[-1] != self.cfg.input_dim:
            raise ValueError("TESSA_SURFACE_FEATURE_SHAPE_INVALID")
        points = self.point_stem(surface_features)
        batch = points.shape[0]
        q = self.latents.unsqueeze(0).expand(batch, -1, -1)
        key_padding_mask = None
        if surface_valid_mask is not None:
            if surface_valid_mask.shape != surface_features.shape[:2]:
                raise ValueError("TESSA_SURFACE_MASK_SHAPE_INVALID")
            key_padding_mask = ~surface_valid_mask.bool()
        update, _ = self.cross(
            query=self.cross_norm(q),
            key=points,
            value=points,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q = q + update
        q = self.latent_stack(q)
        return self.out_norm(q)


class WindowedCausalSelfAttentionV1(nn.Module):
    """Exact causal attention inside a bounded trailing window.

    Training memory/time is O(L*W), not O(L^2).  This is the scaling invariant
    that prevents a MeshAnything-V2-like short-sequence ceiling from becoming a
    TESSA product contract.
    """

    def __init__(self, cfg: TESSAConfigV1):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("TESSA_D_MODEL_NOT_DIVISIBLE_BY_HEADS")
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model * 3, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, dim = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.cfg.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        chunks = []
        chunk = self.cfg.query_chunk_size
        window = self.cfg.local_attention_window
        for start in range(0, length, chunk):
            end = min(length, start + chunk)
            key_start = max(0, end - window)
            q_chunk = q[:, :, start:end]
            k_chunk = k[:, :, key_start:end]
            v_chunk = v[:, :, key_start:end]
            q_pos = torch.arange(start, end, device=x.device)[:, None]
            k_pos = torch.arange(key_start, end, device=x.device)[None, :]
            allow = k_pos <= q_pos
            mask = torch.zeros(
                (end - start, end - key_start),
                dtype=x.dtype,
                device=x.device,
            )
            mask = mask.masked_fill(~allow, float("-inf"))
            y = F.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                attn_mask=mask[None, None],
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
            chunks.append(y)
        y = torch.cat(chunks, dim=2).transpose(1, 2).contiguous().view(b, length, dim)
        return self.out(y)


class TESSADecoderBlockV1(nn.Module):
    def __init__(self, cfg: TESSAConfigV1):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = WindowedCausalSelfAttentionV1(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.cross = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.mlp_ratio),
            nn.GELU(),
            nn.Linear(cfg.d_model * cfg.mlp_ratio, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        q = self.norm2(x)
        cross, _ = self.cross(q, memory, memory, need_weights=False)
        x = x + cross
        x = x + self.mlp(self.norm3(x))
        return x


class TESSAV1(nn.Module):
    """Topology Estimation for Stable Surface Animation, V1.

    Input: GSA-derived XYZ/N/evidence features for one admitted component/chart.
    Output: learned mesh token proposal only.  The Compiler must decode, project,
    support-bind, qualify and seal before ATLAS or MIRA may consume the carrier.
    """

    def __init__(self, cfg: TESSAConfigV1 | None = None):
        super().__init__()
        self.cfg = cfg or TESSAConfigV1()
        self.cfg.scaling_policy()
        self.surface_encoder = TESSASurfaceEncoderV1(self.cfg)
        self.token_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.position_embedding = nn.Embedding(
            self.cfg.local_attention_window,
            self.cfg.d_model,
        )
        self.decoder = nn.ModuleList(
            [TESSADecoderBlockV1(self.cfg) for _ in range(self.cfg.decoder_layers)]
        )
        self.final_norm = nn.LayerNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

    def _position_ids(self, length: int, device: torch.device) -> torch.Tensor:
        # Relative/sliding positions intentionally wrap at W; long-range identity
        # is carried by chart/component structure and surface cross-attention.
        return torch.arange(length, device=device) % self.cfg.local_attention_window

    def forward(
        self,
        *,
        surface_features: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        surface_valid_mask: torch.Tensor | None = None,
    ) -> TESSAOutputV1:
        if input_ids.ndim != 2:
            raise ValueError("TESSA_INPUT_IDS_SHAPE_INVALID")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.cfg.vocab_size
        ):
            raise ValueError("TESSA_INPUT_TOKEN_OUT_OF_RANGE")
        memory = self.surface_encoder(surface_features, surface_valid_mask)
        pos = self._position_ids(input_ids.shape[1], input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)[None]
        for block in self.decoder:
            x = block(x, memory)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("TESSA_LABEL_SHAPE_INVALID")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=self.cfg.PAD,
            )
        return TESSAOutputV1(logits=logits, surface_latents=memory, loss=loss)


def tessa_attention_work_upper_bound_v1(sequence_length: int, cfg: TESSAConfigV1) -> int:
    """Number of q-k pairs per layer, excluding fixed surface cross-attention."""
    if sequence_length < 0:
        raise ValueError("TESSA_SEQUENCE_LENGTH_NEGATIVE")
    return int(sequence_length * min(sequence_length, cfg.local_attention_window))
