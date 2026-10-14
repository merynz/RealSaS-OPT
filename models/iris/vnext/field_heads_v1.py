from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts_v1 import (
    ATTACHMENT_RELATIONS_V1,
    DEFORMATION_MODES_V1,
    UnifiedFieldContractV1,
)


@dataclass(frozen=True)
class UnifiedFieldHeadConfigV1:
    triplane_channels: int = 96
    hidden_dim: int = 256
    layers: int = 4
    owner_embedding_dim: int = 64
    material_embedding_dim: int = 32
    dropout: float = 0.0
    extraction_delta_limit: float = 0.01

    @property
    def sampled_feature_dim(self) -> int:
        return 3 * self.triplane_channels

    def validate(self) -> None:
        floor = UnifiedFieldContractV1()
        if self.triplane_channels < floor.triplane_channels_floor:
            raise ValueError("UNIFIED_IRIS_TRIPLANE_CHANNELS_BELOW_CAPACITY_FLOOR")
        if self.hidden_dim < 128:
            raise ValueError("UNIFIED_IRIS_FIELD_HIDDEN_DIM_TOO_SMALL")
        if self.layers < 3:
            raise ValueError("UNIFIED_IRIS_FIELD_HEAD_TOO_SHALLOW")
        if self.owner_embedding_dim < floor.owner_embedding_dim:
            raise ValueError("UNIFIED_IRIS_OWNER_EMBEDDING_TOO_SMALL")
        if self.material_embedding_dim < floor.material_embedding_dim:
            raise ValueError("UNIFIED_IRIS_MATERIAL_EMBEDDING_TOO_SMALL")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("UNIFIED_IRIS_FIELD_DROPOUT_INVALID")
        if self.extraction_delta_limit <= 0:
            raise ValueError("UNIFIED_IRIS_EXTRACTION_DELTA_LIMIT_INVALID")


class ResidualFieldMLPV1(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(layers - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = h + block(h)
        return h


class GenericDetailAdapterV1(nn.Module):
    """Generic StdGEN++-inspired local specialization.

    StdGEN++ uses a human-face LoRA branch. RealSaS cannot assume a human face,
    so a refinement router selects arbitrary high-detail regions (semantic or
    material boundaries, thin structures, contact/overlap critical regions,
    uncertainty). This adapter adds capacity only to selected point queries.
    """

    def __init__(self, feature_dim: int, rank: int = 64):
        super().__init__()
        if rank <= 0 or rank > feature_dim:
            raise ValueError("UNIFIED_IRIS_DETAIL_ADAPTER_RANK_INVALID")
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor, refinement_weight: torch.Tensor | None) -> torch.Tensor:
        if refinement_weight is None:
            return h
        if refinement_weight.shape != h.shape[:-1]:
            raise ValueError("UNIFIED_IRIS_DETAIL_REFINEMENT_WEIGHT_SHAPE_INVALID")
        delta = self.up(F.gelu(self.down(h)))
        return h + delta * refinement_weight.unsqueeze(-1).to(dtype=h.dtype)


class UnifiedPointFieldHeadsV1(nn.Module):
    """Shared-point decoders for G/A/S/F over one sampled triplane domain.

    Geometry, appearance and semantic ownership are queried at identical spatial
    coordinates. This is the core contract that avoids reconstructing appearance
    on a second visual mesh after canonical mechanical geometry is chosen.
    """

    def __init__(self, config: UnifiedFieldHeadConfigV1 | None = None):
        super().__init__()
        self.config = config or UnifiedFieldHeadConfigV1()
        self.config.validate()
        d = self.config.sampled_feature_dim
        h = self.config.hidden_dim
        l = self.config.layers
        p = self.config.dropout

        self.shared = ResidualFieldMLPV1(d, h, l, p)
        self.detail_adapter = GenericDetailAdapterV1(h, rank=min(64, h))

        self.geometry_adapter = ResidualFieldMLPV1(h, h, 3, p)
        self.appearance_adapter = ResidualFieldMLPV1(h, h, 3, p)
        self.semantic_adapter = ResidualFieldMLPV1(h, h, 4, p)
        self.extraction_adapter = ResidualFieldMLPV1(h, h, 3, p)

        self.sdf_head = nn.Linear(h, 1)
        self.geometry_uncertainty_head = nn.Linear(h, 1)

        # RGBA is canonical appearance evidence. A future view-conditioning
        # adapter can be inserted without changing the CAO evidence contract.
        self.rgba_head = nn.Linear(h, 4)
        self.appearance_uncertainty_head = nn.Linear(h, 1)
        self.source_support_head = nn.Linear(h, 3)

        self.owner_embedding_head = nn.Linear(h, self.config.owner_embedding_dim)
        self.material_embedding_head = nn.Linear(h, self.config.material_embedding_dim)
        self.owner_boundary_head = nn.Linear(h, 1)
        self.material_boundary_head = nn.Linear(h, 1)
        self.amodal_support_head = nn.Linear(h, 1)
        self.deformation_mode_head = nn.Linear(h, len(DEFORMATION_MODES_V1))
        self.attachment_relation_head = nn.Linear(h, len(ATTACHMENT_RELATIONS_V1))
        self.semantic_uncertainty_head = nn.Linear(h, 1)

        self.extraction_delta_head = nn.Linear(h, 3)
        self.extraction_cell_weight_head = nn.Linear(h, 21)

    def forward(
        self,
        sampled_triplane_features: torch.Tensor,
        *,
        refinement_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if sampled_triplane_features.shape[-1] != self.config.sampled_feature_dim:
            raise ValueError("UNIFIED_IRIS_SAMPLED_FEATURE_DIM_INVALID")
        if not torch.isfinite(sampled_triplane_features).all():
            raise ValueError("UNIFIED_IRIS_SAMPLED_FEATURE_NONFINITE")

        shared = self.shared(sampled_triplane_features)
        shared = self.detail_adapter(shared, refinement_weight)

        g = self.geometry_adapter(shared)
        a = self.appearance_adapter(shared)
        s = self.semantic_adapter(shared)
        f = self.extraction_adapter(shared)

        owner_embedding = F.normalize(self.owner_embedding_head(s), dim=-1, eps=1e-6)
        material_embedding = F.normalize(self.material_embedding_head(s), dim=-1, eps=1e-6)

        return {
            "sdf": self.sdf_head(g).squeeze(-1),
            "geometry_log_uncertainty": self.geometry_uncertainty_head(g).squeeze(-1),
            "rgba": torch.sigmoid(self.rgba_head(a)),
            "appearance_log_uncertainty": self.appearance_uncertainty_head(a).squeeze(-1),
            "source_support_logits": self.source_support_head(a),
            "owner_embedding": owner_embedding,
            "material_embedding": material_embedding,
            "owner_boundary_logit": self.owner_boundary_head(s).squeeze(-1),
            "material_boundary_logit": self.material_boundary_head(s).squeeze(-1),
            "amodal_support_logit": self.amodal_support_head(s).squeeze(-1),
            "deformation_mode_logits": self.deformation_mode_head(s),
            "attachment_relation_logits": self.attachment_relation_head(s),
            "semantic_log_uncertainty": self.semantic_uncertainty_head(s).squeeze(-1),
            "extraction_delta": self.config.extraction_delta_limit * torch.tanh(self.extraction_delta_head(f)),
            "extraction_cell_weights": 0.1 * self.extraction_cell_weight_head(f),
            "shared_point_feature": shared,
        }


__all__ = [
    "GenericDetailAdapterV1",
    "ResidualFieldMLPV1",
    "UnifiedFieldHeadConfigV1",
    "UnifiedPointFieldHeadsV1",
]
