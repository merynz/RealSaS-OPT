from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .contracts_v1 import IRISVNextConfigV1


@dataclass(frozen=True)
class GraphicsRelationHeadV3Config:
    """Reference-capacity role-free relation configuration.

    V3 deliberately removes the historical 384 -> 64 normalized relation
    bottleneck.  The full local H is retained at native width and paired with
    deterministic camera/geometry features that are already present in the
    admitted input contract.
    """

    feature_width: int = 512
    geometry_width: int = 256
    camera_width: int = 256
    pair_width: int = 1024
    residual_blocks: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        values = (
            self.feature_width,
            self.geometry_width,
            self.camera_width,
            self.pair_width,
            self.residual_blocks,
        )
        if min(values) <= 0:
            raise ValueError("IRIS_RELATION_V3_CONFIG_NONPOSITIVE")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("IRIS_RELATION_V3_DROPOUT_INVALID")


@dataclass(frozen=True)
class RelationCameraGeometryV3:
    """Deterministic relation geometry derived from admitted camera features."""

    values: torch.Tensor
    feature_names: tuple[str, ...]


_RELATION_GEOMETRY_FEATURE_NAMES_V3 = (
    "xyz_a_x",
    "xyz_a_y",
    "xyz_a_z",
    "xyz_b_x",
    "xyz_b_y",
    "xyz_b_z",
    "delta_x",
    "delta_y",
    "delta_z",
    "distance",
    "depth_a",
    "depth_b",
    "delta_depth",
    "u_a",
    "v_a",
    "u_b",
    "v_b",
    "delta_u",
    "delta_v",
    "projected_pair_distance",
    "same_ray_residual",
    "camera_half_extent_ratio",
)


def relation_camera_geometry_v3(
    point_a: torch.Tensor,
    point_b: torch.Tensor,
    camera_features: torch.Tensor,
) -> RelationCameraGeometryV3:
    """Build explicit pair geometry in the exact Fit8 camera coordinate frame.

    The 13-D admitted camera feature contract is:
      [origin_xyz_normalized(3), right(3), screen_up(3), forward(3), half_extent_ratio(1)]

    `point_a`/`point_b` are IRIS normalized object coordinates.  Camera origin is
    expressed in the same center/half-extent frame by preparation, so depth and
    orthographic projected coordinates are deterministic, not learned labels.
    """

    if point_a.shape != point_b.shape or point_a.ndim < 2 or point_a.shape[-1] != 3:
        raise ValueError("IRIS_RELATION_V3_POINT_SHAPE_INVALID")
    if camera_features.shape[:-1] != point_a.shape[:-1] or camera_features.shape[-1] != 13:
        raise ValueError("IRIS_RELATION_V3_CAMERA_SHAPE_INVALID")
    if not all(torch.is_floating_point(x) for x in (point_a, point_b, camera_features)):
        raise ValueError("IRIS_RELATION_V3_FLOAT_INPUT_REQUIRED")
    if not all(bool(torch.isfinite(x).all()) for x in (point_a, point_b, camera_features)):
        raise ValueError("IRIS_RELATION_V3_NONFINITE_INPUT")

    origin = camera_features[..., 0:3]
    right = F.normalize(camera_features[..., 3:6], dim=-1, eps=1.0e-8)
    up = F.normalize(camera_features[..., 6:9], dim=-1, eps=1.0e-8)
    forward = F.normalize(camera_features[..., 9:12], dim=-1, eps=1.0e-8)
    half = camera_features[..., 12:13].abs().clamp_min(1.0e-8)

    rel_a = point_a - origin
    rel_b = point_b - origin
    delta = point_a - point_b
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)

    depth_a = (rel_a * forward).sum(dim=-1, keepdim=True)
    depth_b = (rel_b * forward).sum(dim=-1, keepdim=True)
    delta_depth = depth_a - depth_b

    # Orthographic screen coordinates normalized by the admitted camera half extent.
    u_a = (rel_a * right).sum(dim=-1, keepdim=True) / half
    v_a = (rel_a * up).sum(dim=-1, keepdim=True) / half
    u_b = (rel_b * right).sum(dim=-1, keepdim=True) / half
    v_b = (rel_b * up).sum(dim=-1, keepdim=True) / half
    delta_u = u_a - u_b
    delta_v = v_a - v_b
    projected_pair_distance = torch.sqrt(delta_u.square() + delta_v.square() + 1.0e-12)

    # For an exact orthographic same-ray pair, projected separation is zero.
    # Keep this as a continuous residual rather than inventing a threshold label.
    same_ray_residual = projected_pair_distance

    values = torch.cat(
        (
            point_a,
            point_b,
            delta,
            distance,
            depth_a,
            depth_b,
            delta_depth,
            u_a,
            v_a,
            u_b,
            v_b,
            delta_u,
            delta_v,
            projected_pair_distance,
            same_ray_residual,
            half,
        ),
        dim=-1,
    )
    if values.shape[-1] != len(_RELATION_GEOMETRY_FEATURE_NAMES_V3):
        raise AssertionError("IRIS_RELATION_V3_GEOMETRY_FEATURE_DIM_DRIFT")
    return RelationCameraGeometryV3(
        values=values,
        feature_names=_RELATION_GEOMETRY_FEATURE_NAMES_V3,
    )


class _ResidualMLPBlockV3(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(F.gelu(self.fc1(self.norm(x)))))
        return (x + h) * self.scale


class GraphicsRelationHeadV3(nn.Module):
    """Geometry-explicit pair relations over sealed/raw IRIS H.

    Pair outputs intentionally exclude global visibility and neural confidence.
    Visibility belongs to a complete-ray context or deterministic raster court;
    confidence belongs to calibrated evidence qualification, not an unsupervised
    free scalar that can gate Compiler behavior.
    """

    architecture_id = "RealSaS.IRIS.GraphicsRelationHead.v3"
    output_semantics = "ROLE_FREE_GEOMETRY_EXPLICIT_PAIR_RELATION_EVIDENCE"
    categorical_semantics_used = False
    raw_h_bottlenecked_to_64 = False
    neural_confidence_output = False
    pairwise_visibility_output = False

    STATIC_OUTPUTS = (
        "same_drawable_affinity",
        "cut_probability",
        "attachment_affinity",
    )
    VIEW_OUTPUTS = (
        "front_probability",
        "occlusion_probability",
        "projected_alpha_boundary_probability",
        "projected_appearance_boundary_probability",
    )

    def __init__(
        self,
        iris_config: IRISVNextConfigV1 = IRISVNextConfigV1(),
        relation_config: GraphicsRelationHeadV3Config = GraphicsRelationHeadV3Config(),
    ) -> None:
        super().__init__()
        iris_config.validate()
        relation_config.validate()
        self.iris_config = iris_config
        self.relation_config = relation_config

        # Preserve the full H dimensionality before expansion.  There is no
        # 384->64 normalized face embedding as in V1/V2.
        self.feature = nn.Sequential(
            nn.LayerNorm(iris_config.local_feature_dim),
            nn.Linear(iris_config.local_feature_dim, relation_config.feature_width),
            nn.GELU(),
            nn.Linear(relation_config.feature_width, relation_config.feature_width),
        )
        self.camera = nn.Sequential(
            nn.LayerNorm(iris_config.camera_feature_dim),
            nn.Linear(iris_config.camera_feature_dim, relation_config.camera_width),
            nn.GELU(),
            nn.Linear(relation_config.camera_width, relation_config.camera_width),
        )
        self.geometry = nn.Sequential(
            nn.LayerNorm(len(_RELATION_GEOMETRY_FEATURE_NAMES_V3)),
            nn.Linear(len(_RELATION_GEOMETRY_FEATURE_NAMES_V3), relation_config.geometry_width),
            nn.GELU(),
            nn.Linear(relation_config.geometry_width, relation_config.geometry_width),
        )

        learned_pair_dim = relation_config.feature_width * 4
        static_input_dim = learned_pair_dim
        view_input_dim = learned_pair_dim + relation_config.geometry_width + relation_config.camera_width

        self.static_in = nn.Sequential(
            nn.LayerNorm(static_input_dim),
            nn.Linear(static_input_dim, relation_config.pair_width),
            nn.GELU(),
        )
        self.view_in = nn.Sequential(
            nn.LayerNorm(view_input_dim),
            nn.Linear(view_input_dim, relation_config.pair_width),
            nn.GELU(),
        )
        self.static_blocks = nn.ModuleList(
            [
                _ResidualMLPBlockV3(relation_config.pair_width, relation_config.dropout)
                for _ in range(relation_config.residual_blocks)
            ]
        )
        self.view_blocks = nn.ModuleList(
            [
                _ResidualMLPBlockV3(relation_config.pair_width, relation_config.dropout)
                for _ in range(relation_config.residual_blocks)
            ]
        )
        self.static_out = nn.Linear(relation_config.pair_width, len(self.STATIC_OUTPUTS))
        self.view_out = nn.Linear(relation_config.pair_width, len(self.VIEW_OUTPUTS))

    @staticmethod
    def _pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat((a, b, a - b, a * b), dim=-1)

    def _learned_pair(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        if feature_a.shape != feature_b.shape:
            raise ValueError("IRIS_RELATION_V3_H_PAIR_SHAPE_MISMATCH")
        if feature_a.shape[-1] != self.iris_config.local_feature_dim:
            raise ValueError("IRIS_RELATION_V3_H_DIM_INVALID")
        return self._pair(self.feature(feature_a), self.feature(feature_b))

    def static_relations(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z = self.static_in(self._learned_pair(feature_a, feature_b))
        for block in self.static_blocks:
            z = block(z)
        p = torch.sigmoid(self.static_out(z))
        return {name: p[..., index] for index, name in enumerate(self.STATIC_OUTPUTS)}

    def view_relations(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        point_a: torch.Tensor,
        point_b: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        learned = self._learned_pair(feature_a, feature_b)
        geometry = relation_camera_geometry_v3(point_a, point_b, camera_features).values
        camera = self.camera(camera_features)
        z = self.view_in(torch.cat((learned, self.geometry(geometry), camera), dim=-1))
        for block in self.view_blocks:
            z = block(z)
        p = torch.sigmoid(self.view_out(z))
        return {name: p[..., index] for index, name in enumerate(self.VIEW_OUTPUTS)}

    def relation_probabilities(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        point_a: torch.Tensor,
        point_b: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            **self.static_relations(feature_a, feature_b),
            **self.view_relations(
                feature_a,
                feature_b,
                point_a,
                point_b,
                camera_features,
            ),
        }


class RayStackVisibilityHeadV3(nn.Module):
    """Per-hit global visibility over a complete ordered ray stack.

    This head exists separately from pair relations because visibility of hit B
    cannot be decided from pair (B, C) when another hit A is in front of both.
    It consumes all hits on one ray together.  Runtime may still choose a
    deterministic raster authority instead; this module is evidence-only.
    """

    architecture_id = "RealSaS.IRIS.RayStackVisibilityHead.v3"
    output_semantics = "VIEW_CONDITIONED_GLOBAL_RAY_VISIBILITY_EVIDENCE"
    neural_confidence_output = False

    def __init__(
        self,
        iris_config: IRISVNextConfigV1 = IRISVNextConfigV1(),
        *,
        width: int = 512,
        heads: int = 8,
        layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        iris_config.validate()
        if min(width, heads, layers, mlp_ratio) <= 0 or width % heads:
            raise ValueError("IRIS_RAY_VIS_V3_CONFIG_INVALID")
        self.iris_config = iris_config
        self.hit = nn.Sequential(
            nn.LayerNorm(iris_config.local_feature_dim + 5),
            nn.Linear(iris_config.local_feature_dim + 5, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.camera = nn.Sequential(
            nn.LayerNorm(iris_config.camera_feature_dim),
            nn.Linear(iris_config.camera_feature_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stack = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Linear(width, 1)

    def forward(
        self,
        *,
        hit_h: torch.Tensor,
        hit_xyz: torch.Tensor,
        hit_depth: torch.Tensor,
        camera_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hit_h.ndim != 3 or hit_h.shape[-1] != self.iris_config.local_feature_dim:
            raise ValueError("IRIS_RAY_VIS_V3_H_SHAPE_INVALID")
        if hit_xyz.shape != hit_h.shape[:2] + (3,):
            raise ValueError("IRIS_RAY_VIS_V3_XYZ_SHAPE_INVALID")
        if hit_depth.shape != hit_h.shape[:2]:
            raise ValueError("IRIS_RAY_VIS_V3_DEPTH_SHAPE_INVALID")
        if valid_mask.shape != hit_h.shape[:2]:
            raise ValueError("IRIS_RAY_VIS_V3_MASK_SHAPE_INVALID")
        if camera_features.shape != (hit_h.shape[0], self.iris_config.camera_feature_dim):
            raise ValueError("IRIS_RAY_VIS_V3_CAMERA_SHAPE_INVALID")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("IRIS_RAY_VIS_V3_EMPTY_STACK")

        # normalized rank supplies global stack position without pretending rank
        # alone is semantic visibility.
        count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        rank = torch.arange(hit_h.shape[1], device=hit_h.device)[None, :].expand_as(hit_depth)
        rank = rank.to(dtype=hit_depth.dtype) / (count.to(dtype=hit_depth.dtype) - 1.0).clamp_min(1.0)
        z = self.hit(
            torch.cat(
                (
                    hit_h,
                    hit_xyz,
                    hit_depth[..., None],
                    rank[..., None],
                ),
                dim=-1,
            )
        )
        z = z + self.camera(camera_features)[:, None, :]
        z = self.stack(z, src_key_padding_mask=~valid_mask.bool())
        probability = torch.sigmoid(self.out(z).squeeze(-1))
        return torch.where(valid_mask.bool(), probability, torch.zeros_like(probability))
