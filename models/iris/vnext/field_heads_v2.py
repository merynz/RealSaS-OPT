from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts_v1 import ATTACHMENT_RELATIONS_V1, DEFORMATION_MODES_V1
from .contracts_v2 import UnifiedFieldContractV2, validate_geometry_field_outputs_v2
from .field_heads_v1 import GenericDetailAdapterV1, ResidualFieldMLPV1


@dataclass(frozen=True)
class UnifiedPointFieldHeadConfigV2:
    triplane_channels: int = 96
    hidden_dim: int = 256
    layers: int = 4
    owner_embedding_dim: int = 64
    material_embedding_dim: int = 32
    dropout: float = 0.0

    @property
    def sampled_feature_dim(self) -> int:
        return 3 * self.triplane_channels

    def validate(self) -> None:
        floor = UnifiedFieldContractV2()
        floor.validate()
        if self.triplane_channels < floor.triplane_channels_floor:
            raise ValueError("UNIFIED_IRIS_V2_TRIPLANE_CHANNELS_BELOW_CAPACITY_FLOOR")
        if self.hidden_dim < 128:
            raise ValueError("UNIFIED_IRIS_V2_FIELD_HIDDEN_DIM_TOO_SMALL")
        if self.layers < 3:
            raise ValueError("UNIFIED_IRIS_V2_FIELD_HEAD_TOO_SHALLOW")
        if self.owner_embedding_dim < floor.owner_embedding_dim:
            raise ValueError("UNIFIED_IRIS_V2_OWNER_EMBEDDING_TOO_SMALL")
        if self.material_embedding_dim < floor.material_embedding_dim:
            raise ValueError("UNIFIED_IRIS_V2_MATERIAL_EMBEDDING_TOO_SMALL")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("UNIFIED_IRIS_V2_FIELD_DROPOUT_INVALID")


class UnifiedPointFieldHeadsV2(nn.Module):
    """Continuous G/A/S evidence over one shared triplane domain.

    F extraction is intentionally not produced here. FIT8 courts showed that
    extraction support/topology is scale- and cell-context-dependent, while these
    fields are continuous point evidence. Keeping them separate prevents a chosen
    extraction lattice from becoming the ontology of G/A/S.

    Geometry is open-surface-aware:

    - `unsigned_distance` is globally meaningful proximity evidence;
    - `oriented_level_set` may carry local/signed structure;
    - `orientation_validity` says where that oriented value is authoritative;
    - uncertainty remains explicit.

    No output named `sdf` is exposed as universal geometry authority in V2.
    """

    def __init__(self, config: UnifiedPointFieldHeadConfigV2 | None = None):
        super().__init__()
        self.config = config or UnifiedPointFieldHeadConfigV2()
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

        self.unsigned_distance_head = nn.Linear(h, 1)
        self.oriented_level_set_head = nn.Linear(h, 1)
        self.orientation_validity_head = nn.Linear(h, 1)
        self.geometry_uncertainty_head = nn.Linear(h, 1)

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

    def forward(
        self,
        sampled_triplane_features: torch.Tensor,
        *,
        refinement_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if sampled_triplane_features.shape[-1] != self.config.sampled_feature_dim:
            raise ValueError("UNIFIED_IRIS_V2_SAMPLED_FEATURE_DIM_INVALID")
        if not torch.isfinite(sampled_triplane_features).all():
            raise ValueError("UNIFIED_IRIS_V2_SAMPLED_FEATURE_NONFINITE")

        shared = self.shared(sampled_triplane_features)
        shared = self.detail_adapter(shared, refinement_weight)

        g = self.geometry_adapter(shared)
        a = self.appearance_adapter(shared)
        s = self.semantic_adapter(shared)

        owner_embedding = F.normalize(self.owner_embedding_head(s), dim=-1, eps=1e-6)
        material_embedding = F.normalize(self.material_embedding_head(s), dim=-1, eps=1e-6)

        geometry = {
            "unsigned_distance": F.softplus(self.unsigned_distance_head(g).squeeze(-1)),
            "oriented_level_set": self.oriented_level_set_head(g).squeeze(-1),
            "orientation_validity": torch.sigmoid(
                self.orientation_validity_head(g).squeeze(-1)
            ),
            "geometry_log_uncertainty": self.geometry_uncertainty_head(g).squeeze(-1),
        }
        validate_geometry_field_outputs_v2(geometry)

        return {
            **geometry,
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
            "shared_point_feature": shared,
        }


__all__ = [
    "UnifiedPointFieldHeadConfigV2",
    "UnifiedPointFieldHeadsV2",
]
