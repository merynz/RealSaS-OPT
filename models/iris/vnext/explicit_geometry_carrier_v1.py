from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn

from .contracts_v1 import IRISVNextConfigV1
from .scene_core_v1 import LocalEvidenceFieldV1, SharedSceneTriplaneVNextV1


class GeometryHighResolutionRefinerV1(nn.Module):
    """Geometry-only 64->128 triplane refinement with exact-zero residual init."""

    def __init__(self, cfg: IRISVNextConfigV1):
        super().__init__()
        c = int(cfg.plane_channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.residual_scale = 0.25

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        if planes.ndim != 5 or planes.shape[1] != 3:
            raise ValueError("IRIS_H_GEO_P3_TRIPLANE_SHAPE_INVALID")
        b, three, c, h, w = planes.shape
        x = F.interpolate(planes.reshape(b * three, c, h, w), scale_factor=2.0, mode="bilinear", align_corners=False)
        residual = self.conv2(F.silu(self.conv1(x)))
        y = x + self.residual_scale * residual
        return y.reshape(b, three, c, y.shape[-2], y.shape[-1])


def continuous_xyz_features_v1(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("IRIS_H_GEO_P3_POINTS_SHAPE_INVALID")
    features = [points]
    for frequency in (1.0, 2.0, 4.0):
        phase = math.pi * frequency * points
        features.extend((torch.sin(phase), torch.cos(phase)))
    return torch.cat(features, dim=-1)


class ExplicitGeometryCarrierV1(nn.Module):
    """Dedicated geometry carrier. Shared V8 IRIS remains external and frozen."""

    architecture_id = "RealSaS.IRIS.ExplicitGeometryCarrier.v1"
    output_semantics = "DEDICATED_HIRES_METRIC_GEOMETRY_EVIDENCE"
    shared_h_mutation_allowed = False

    def __init__(self, cfg: IRISVNextConfigV1 = IRISVNextConfigV1()):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.geometry_scene = SharedSceneTriplaneVNextV1(cfg)
        self.highres_refiner = GeometryHighResolutionRefinerV1(cfg)
        self.geometry_field = LocalEvidenceFieldV1(cfg)
        self.position_projection = nn.Linear(21, cfg.local_feature_dim, bias=False)
        nn.init.zeros_(self.position_projection.weight)

    def initialize_from_v8(self, v8_model) -> None:
        self.geometry_scene.load_state_dict(v8_model.scene_core.scene.state_dict(), strict=True)
        self.geometry_field.load_state_dict(v8_model.scene_core.field.state_dict(), strict=True)
        nn.init.zeros_(self.highres_refiner.conv2.weight)
        nn.init.zeros_(self.highres_refiner.conv2.bias)
        nn.init.zeros_(self.position_projection.weight)

    def encode_geometry(self, frozen_view_memory: torch.Tensor) -> torch.Tensor:
        base = self.geometry_scene(frozen_view_memory)
        return self.highres_refiner(base)

    def query_geometry(self, geometry_planes: torch.Tensor, points: torch.Tensor) -> dict[str, torch.Tensor]:
        local = self.geometry_field(geometry_planes, points)
        pos = self.position_projection(continuous_xyz_features_v1(points))
        h_geo = local["H"] + pos
        return {
            "H_geo": h_geo,
            "geometry_field": local["field"],
            "geometry_log_uncertainty": local["log_uncertainty"],
            "geometry_extraction_delta": local["extraction_delta"],
        }
