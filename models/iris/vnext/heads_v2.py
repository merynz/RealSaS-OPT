from __future__ import annotations

"""Post-V8 relation head with conditioning aligned to Graphics Truth V2.

Historical ``GraphicsRelationHeadV1`` is immutable V8 provenance.  V2 removes
projected boundary predictions from the camera-free static branch and predicts
them only in the camera-conditioned view branch.  A deterministic migration
copies every compatible V1 relation capability and initializes only the two new
projected-boundary logits to the neutral probability 0.5.
"""

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import nn

from .contracts_v1 import IRISVNextConfigV1
from .heads_v1 import GraphicsRelationHeadV1


@dataclass(frozen=True)
class RelationHeadV1ToV2MigrationReceipt:
    schema: str = "RealSaS.IRISRelationHeadV1ToV2MigrationReceipt.v1"
    preserved_static_outputs: tuple[str, ...] = (
        "same_drawable_affinity",
        "cut_probability",
        "attachment_affinity",
        "confidence",
    )
    preserved_view_outputs: tuple[str, ...] = (
        "front_probability",
        "occlusion_probability",
        "visibility_probability_a",
        "visibility_probability_b",
        "confidence",
    )
    new_view_outputs: tuple[str, ...] = (
        "projected_alpha_boundary_probability",
        "projected_appearance_boundary_probability",
    )
    new_boundary_initial_probability: float = 0.5
    historical_v1_mutated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class GraphicsRelationHeadV2(nn.Module):
    """Role-free relation evidence with explicit view-conditioned boundaries."""

    output_semantics = "ROLE_FREE_GRAPHICS_RELATION_EVIDENCE_V2"
    categorical_semantics_used = False
    projected_boundaries_camera_conditioned = True

    def __init__(self, cfg: IRISVNextConfigV1):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Sequential(
            nn.LayerNorm(cfg.local_feature_dim),
            nn.Linear(cfg.local_feature_dim, cfg.relation_embedding_dim),
            nn.GELU(),
            nn.Linear(cfg.relation_embedding_dim, cfg.relation_embedding_dim),
        )
        pair = cfg.relation_embedding_dim * 4
        # same, cut, attachment, confidence.  Projected boundaries do not live here.
        self.static_pair = nn.Sequential(
            nn.LayerNorm(pair),
            nn.Linear(pair, cfg.relation_embedding_dim),
            nn.GELU(),
            nn.Linear(cfg.relation_embedding_dim, 4),
        )
        self.view = nn.Sequential(
            nn.Linear(cfg.camera_feature_dim, cfg.camera_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.camera_embed_dim, cfg.relation_embedding_dim),
        )
        # front, occlusion, vis_a, vis_b, projected alpha, projected appearance, confidence.
        self.view_pair = nn.Sequential(
            nn.LayerNorm(pair + cfg.relation_embedding_dim),
            nn.Linear(pair + cfg.relation_embedding_dim, cfg.relation_embedding_dim),
            nn.GELU(),
            nn.Linear(cfg.relation_embedding_dim, 7),
        )

    @staticmethod
    def _pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat((a, b, a - b, a * b), dim=-1)

    def face_embedding(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embedding(h), dim=-1, eps=1.0e-6)

    def static_relations(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> dict[str, torch.Tensor]:
        a, b = self.face_embedding(feature_a), self.face_embedding(feature_b)
        p = torch.sigmoid(self.static_pair(self._pair(a, b)))
        return {
            "same_drawable_affinity": p[..., 0],
            "cut_probability": p[..., 1],
            "attachment_affinity": p[..., 2],
            "confidence": p[..., 3],
            "fragment_embedding_a": a,
            "fragment_embedding_b": b,
        }

    def view_relations(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        a, b = self.face_embedding(feature_a), self.face_embedding(feature_b)
        if camera_features.shape[:-1] != a.shape[:-1]:
            raise ValueError("IRIS_VNEXT_RELATION_V2_CAMERA_PREFIX_MISMATCH")
        p = torch.sigmoid(
            self.view_pair(
                torch.cat((self._pair(a, b), self.view(camera_features)), dim=-1)
            )
        )
        return {
            "front_probability": p[..., 0],
            "occlusion_probability": p[..., 1],
            "visibility_probability_a": p[..., 2],
            "visibility_probability_b": p[..., 3],
            "projected_alpha_boundary_probability": p[..., 4],
            "projected_appearance_boundary_probability": p[..., 5],
            "confidence": p[..., 6],
        }

    def relation_probabilities(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        static = self.static_relations(feature_a, feature_b)
        viewed = self.view_relations(feature_a, feature_b, camera_features)
        return {
            "same_drawable_affinity": static["same_drawable_affinity"],
            "cut_probability": static["cut_probability"],
            "attachment_affinity": static["attachment_affinity"],
            "front_probability": viewed["front_probability"],
            "occlusion_probability": viewed["occlusion_probability"],
            "visibility_probability_a": viewed["visibility_probability_a"],
            "visibility_probability_b": viewed["visibility_probability_b"],
            "projected_alpha_boundary_probability": viewed[
                "projected_alpha_boundary_probability"
            ],
            "projected_appearance_boundary_probability": viewed[
                "projected_appearance_boundary_probability"
            ],
        }


def _copy_prefix_modules(source: nn.Sequential, target: nn.Sequential) -> None:
    # LayerNorm + hidden Linear are stateful; GELU is stateless.
    target[0].load_state_dict(source[0].state_dict(), strict=True)
    target[1].load_state_dict(source[1].state_dict(), strict=True)


def migrate_graphics_relation_head_v1_to_v2(
    source: GraphicsRelationHeadV1,
) -> tuple[GraphicsRelationHeadV2, RelationHeadV1ToV2MigrationReceipt]:
    """Preserve compatible V1 behavior and add neutral view-boundary outputs."""

    target = GraphicsRelationHeadV2(source.cfg).to(
        device=next(source.parameters()).device,
        dtype=next(source.parameters()).dtype,
    )
    target.embedding.load_state_dict(source.embedding.state_dict(), strict=True)
    target.view.load_state_dict(source.view.state_dict(), strict=True)
    _copy_prefix_modules(source.static_pair, target.static_pair)
    _copy_prefix_modules(source.view_pair, target.view_pair)

    with torch.no_grad():
        # V1 static rows: same, cut, attachment, alpha-boundary, appearance-boundary, confidence.
        target.static_pair[3].weight[0:3].copy_(source.static_pair[3].weight[0:3])
        target.static_pair[3].bias[0:3].copy_(source.static_pair[3].bias[0:3])
        target.static_pair[3].weight[3].copy_(source.static_pair[3].weight[5])
        target.static_pair[3].bias[3].copy_(source.static_pair[3].bias[5])

        # V1 view rows: front, occlusion, vis_a, vis_b, confidence.
        target.view_pair[3].weight[0:4].copy_(source.view_pair[3].weight[0:4])
        target.view_pair[3].bias[0:4].copy_(source.view_pair[3].bias[0:4])
        target.view_pair[3].weight[6].copy_(source.view_pair[3].weight[4])
        target.view_pair[3].bias[6].copy_(source.view_pair[3].bias[4])

        # New projected boundaries start from an explicit neutral prior, not
        # random logits and not a fabricated source label.
        target.view_pair[3].weight[4:6].zero_()
        target.view_pair[3].bias[4:6].zero_()

    return target, RelationHeadV1ToV2MigrationReceipt()
