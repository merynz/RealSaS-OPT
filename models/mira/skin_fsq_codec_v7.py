from __future__ import annotations

"""RealSaS MIRA V7: large-capacity, carrier-native discrete skin field codec.

Research implementation; NOT a trained ModelRelease. No SkinTokens code is copied.
The finite-scalar representation is independently implemented using the
existing RealSaS quantizer. The critical V6 reconstruction-gradient severance is
fixed: reconstruction decodes straight-through continuous quantized latents,
while inference decodes exactly the same hard codes from integer token IDs.

Complexity of encoding is O(B*(N+J)*H + B*N*J), NOT O(B*N*J*H).
This matters for dense Corpus370 carrier meshes and nonhumanoid skeletons.
"""

from dataclasses import asdict, dataclass
from math import prod, sqrt

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.mira.skin_fsq_codec_v6 import FiniteScalarQuantizer


MIRA_V7_ARCHITECTURE_ID = "RealSaS.MIRA.CarrierNativeFSQ.STGradient.ActiveField.v7_0"


@dataclass(frozen=True)
class MIRACodecConfigV7:
    vertex_feature_dim: int = 7
    joint_feature_dim: int = 25
    hidden_dim: int = 384
    tokens_per_joint: int = 4
    levels: tuple[int, ...] = (8, 8, 8, 5, 5, 5)
    max_vertices: int = 131072
    max_joints: int = 256
    importance_strength: float = 4.0
    active_threshold: float = 0.02
    eps: float = 1e-8

    def __post_init__(self):
        if min(self.vertex_feature_dim, self.joint_feature_dim, self.hidden_dim,
               self.tokens_per_joint, self.max_vertices, self.max_joints) < 1:
            raise ValueError("MIRA_V7_CONFIG_NONPOSITIVE")
        if not self.levels or any(x < 2 for x in self.levels):
            raise ValueError("MIRA_V7_FSQ_LEVELS_INVALID")
        if not (0 <= self.active_threshold < 1) or self.importance_strength < 0:
            raise ValueError("MIRA_V7_IMPORTANCE_CONFIG_INVALID")

    @property
    def vocab_size(self) -> int:
        return prod(self.levels)

    @property
    def latent_dim(self) -> int:
        return len(self.levels)

    def to_dict(self) -> dict:
        return asdict(self)


class MIRASkinCodecV7(nn.Module):
    architecture_id = MIRA_V7_ARCHITECTURE_ID

    def __init__(self, config: MIRACodecConfigV7 = MIRACodecConfigV7()):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.quantizer = FiniteScalarQuantizer(config.levels)
        self.vertex_encoder = nn.Sequential(
            nn.Linear(config.vertex_feature_dim, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(config.joint_feature_dim, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(h * 3, h * 2), nn.GELU(),
            nn.Linear(h * 2, h),
            nn.GELU(), nn.Linear(h, config.tokens_per_joint * config.latent_dim),
        )
        self.vertex_decoder = nn.Sequential(
            nn.Linear(config.vertex_feature_dim, h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h),
        )
        self.joint_decoder = nn.Sequential(
            nn.Linear(config.joint_feature_dim, h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h),
        )
        self.code_decoder = nn.Sequential(
            nn.Linear(config.tokens_per_joint * config.latent_dim, h * 2),
            nn.GELU(), nn.Linear(h * 2, h),
        )
        self.joint_bias = nn.Linear(h, 1)

    def _shape(self, vertex_features: Tensor, joint_features: Tensor) -> tuple[int, int, int]:
        if vertex_features.ndim != 3 or joint_features.ndim != 3:
            raise ValueError("MIRA_V7_FEATURE_RANK")
        b, n, fv = vertex_features.shape
        bj, j, fj = joint_features.shape
        if b != bj or fv != self.config.vertex_feature_dim or fj != self.config.joint_feature_dim:
            raise ValueError("MIRA_V7_FEATURE_SHAPE")
        if b < 1 or n < 1 or j < 1 or n > self.config.max_vertices or j > self.config.max_joints:
            raise ValueError("MIRA_V7_CARRIER_RESOURCE_BOUND")
        if not torch.isfinite(vertex_features).all() or not torch.isfinite(joint_features).all():
            raise ValueError("MIRA_V7_FEATURE_NONFINITE")
        return b, n, j

    def _target(self, weights: Tensor, b: int, n: int, j: int) -> Tensor:
        if weights.shape != (b, n, j) or not torch.isfinite(weights).all():
            raise ValueError("MIRA_V7_WEIGHT_SHAPE_OR_NONFINITE")
        if torch.any(weights < 0):
            raise ValueError("MIRA_V7_WEIGHT_NEGATIVE")
        mass = weights.sum(dim=-1, keepdim=True)
        if torch.any(mass <= self.config.eps):
            raise ValueError("MIRA_V7_WEIGHT_UNCOVERED_VERTEX")
        return weights / mass

    def _latent(self, target: Tensor, vertices: Tensor, joints: Tensor) -> Tensor:
        # No B*J*N*H tensor: only B*N*H, B*J*H and B*N*J.
        encoded_v = self.vertex_encoder(vertices)
        encoded_j = self.joint_encoder(joints)
        support_mass = target.sum(dim=1).clamp_min(self.config.eps)  # [B,J]
        regional = torch.einsum("bnj,bnh->bjh", target, encoded_v)
        regional = regional / support_mass.unsqueeze(-1)
        global_state = encoded_v.mean(dim=1, keepdim=True).expand_as(encoded_j)
        latents = self.latent_encoder(torch.cat((regional, encoded_j, global_state), dim=-1))
        return latents.reshape(
            joints.shape[0], joints.shape[1],
            self.config.tokens_per_joint, self.config.latent_dim,
        )

    def _hard_and_straight_through(self, latent: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Straight-through reconstruction gradient must reach the latent encoder.
        bounded = torch.tanh(latent)
        indices = self.quantizer._quantize_indices(latent)
        hard = self.quantizer.indices_to_codes(indices).to(dtype=latent.dtype)
        straight_through = bounded + (hard - bounded).detach()
        ids = self.quantizer.indices_to_token_ids(indices)
        return straight_through, hard, ids

    def _decode_codes(self, codes: Tensor, vertices: Tensor, joints: Tensor) -> Tensor:
        b, n, j = self._shape(vertices, joints)
        if codes.shape != (b, j, self.config.tokens_per_joint, self.config.latent_dim):
            raise ValueError("MIRA_V7_CODE_SHAPE")
        latent = codes.flatten(start_dim=2)
        vertex_state = self.vertex_decoder(vertices)
        joint_state = self.joint_decoder(joints) + self.code_decoder(latent)
        logits = torch.einsum("bnh,bjh->bnj", vertex_state, joint_state)
        logits = logits / sqrt(float(self.config.hidden_dim))
        logits = logits + self.joint_bias(joint_state).transpose(1, 2)
        return torch.softmax(logits, dim=-1)

    def encode_tokens(self, weights: Tensor, vertex_features: Tensor,
                      joint_features: Tensor) -> Tensor:
        b, n, j = self._shape(vertex_features, joint_features)
        target = self._target(weights, b, n, j)
        _, _, ids = self._hard_and_straight_through(
            self._latent(target, vertex_features, joint_features)
        )
        return ids

    def decode_tokens(self, token_ids: Tensor, vertex_features: Tensor,
                      joint_features: Tensor) -> Tensor:
        b, _, j = self._shape(vertex_features, joint_features)
        if token_ids.shape != (b, j, self.config.tokens_per_joint):
            raise ValueError("MIRA_V7_TOKEN_SHAPE")
        hard = self.quantizer.token_ids_to_codes(token_ids).to(dtype=vertex_features.dtype)
        return self._decode_codes(hard, vertex_features, joint_features)

    def forward(self, weights: Tensor, vertex_features: Tensor,
                joint_features: Tensor, *, mesh_edge_index: Tensor | None = None) -> dict[str, Tensor]:
        b, n, j = self._shape(vertex_features, joint_features)
        target = self._target(weights, b, n, j)
        latent = self._latent(target, vertex_features, joint_features)
        straight_through, hard, ids = self._hard_and_straight_through(latent)
        decoded = self._decode_codes(straight_through, vertex_features, joint_features)
        abs_error = (decoded - target).abs()
        active = (target > self.config.active_threshold).to(abs_error.dtype)
        weighted = 1.0 + self.config.importance_strength * active
        reconstruction = (abs_error * weighted).sum() / weighted.sum()
        quantization = F.mse_loss(torch.tanh(latent), hard.detach())
        # Encourage sparse local influence without rewarding uncovered vertices;
        # per-vertex simplex is enforced by softmax.
        sparsity = -(decoded.clamp_min(self.config.eps) *
                     decoded.clamp_min(self.config.eps).log()).sum(dim=-1).mean()
        smoothness = decoded.sum() * 0.0
        if mesh_edge_index is not None:
            edges = mesh_edge_index.to(device=decoded.device, dtype=torch.long)
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError("MIRA_V7_EDGE_SHAPE")
            if edges.numel() and (torch.any(edges < 0) or torch.any(edges >= n)):
                raise ValueError("MIRA_V7_EDGE_OUT_OF_RANGE")
            if edges.shape[1] > 0:
                src, dst = edges[0], edges[1]
                predicted_delta = decoded[:, src] - decoded[:, dst]
                truth_delta = target[:, src] - target[:, dst]
                # Teacher-relative gradient consistency, not blind smoothing across seams.
                smoothness = F.smooth_l1_loss(predicted_delta, truth_delta)
        loss = reconstruction + 0.05 * quantization + 0.001 * sparsity + 0.1 * smoothness
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction,
            "quantization_loss": quantization,
            "sparsity_loss": sparsity,
            "edge_gradient_loss": smoothness,
            "mae": abs_error.mean(),
            "max_abs": abs_error.amax(),
            "simplex_max_abs": (decoded.sum(dim=-1) - 1.0).abs().amax(),
            "token_ids": ids,
            "reconstructed_weights": decoded,
        }


__all__ = ["MIRA_V7_ARCHITECTURE_ID", "MIRACodecConfigV7", "MIRASkinCodecV7"]
