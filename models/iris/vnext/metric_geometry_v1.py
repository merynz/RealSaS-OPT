from __future__ import annotations

"""Training-only metric geometry readout for H-GEO-P1.

The head is deliberately small. It does not become product authority; its purpose
is to force the existing V8 local carrier H(x) to make exact metric surface
geometry linearly/shallowly decodable while the carrier substrate is retrained.
"""

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MetricGeometryAuxConfigV1:
    local_feature_dim: int = 384
    width: int = 256
    maximum_offset_iris: float = 0.15

    def validate(self) -> None:
        if self.local_feature_dim <= 0 or self.width <= 0:
            raise ValueError("H_GEO_P1_AUX_WIDTH_INVALID")
        if not (0.0 < self.maximum_offset_iris <= 0.25):
            raise ValueError("H_GEO_P1_AUX_OFFSET_BOUND_INVALID")


class IRISMetricGeometryAuxHeadV1(nn.Module):
    architecture_id = "RealSaS.IRISMetricGeometryAuxHead.v1"
    product_authority = False
    inference_contract = False

    def __init__(self, config: MetricGeometryAuxConfigV1 = MetricGeometryAuxConfigV1()):
        super().__init__()
        config.validate()
        self.config = config
        self.body = nn.Sequential(
            nn.LayerNorm(config.local_feature_dim),
            nn.Linear(config.local_feature_dim, config.width),
            nn.SiLU(),
        )
        self.distance = nn.Linear(config.width, 1)
        self.offset = nn.Linear(config.width, 3)
        self.normal = nn.Linear(config.width, 3)
        self.near = nn.Linear(config.width, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        if h.ndim < 2 or h.shape[-1] != self.config.local_feature_dim:
            raise ValueError("H_GEO_P1_AUX_H_SHAPE_INVALID")
        z = self.body(h)
        return {
            "distance_iris": F.softplus(self.distance(z).squeeze(-1)),
            "offset_to_surface_iris": torch.tanh(self.offset(z)) * float(self.config.maximum_offset_iris),
            "normal_unit": F.normalize(self.normal(z), dim=-1, eps=1e-8),
            "near_logit": self.near(z).squeeze(-1),
        }


def metric_geometry_objective_v1(
    prediction: dict[str, torch.Tensor],
    *,
    distance_iris: torch.Tensor,
    offset_to_surface_iris: torch.Tensor,
    normal_unit: torch.Tensor,
    normal_valid: torch.Tensor,
    near_label: torch.Tensor,
    offset_valid_distance_iris: float = 0.12,
) -> dict[str, torch.Tensor]:
    distance = prediction["distance_iris"]
    offset = prediction["offset_to_surface_iris"]
    normal = prediction["normal_unit"]
    near_logit = prediction["near_logit"]
    n = int(distance_iris.shape[0])
    if distance.shape != (n,) or offset.shape != (n, 3) or normal.shape != (n, 3) or near_logit.shape != (n,):
        raise ValueError("H_GEO_P1_AUX_PREDICTION_SHAPE_INVALID")
    distance_loss = F.smooth_l1_loss(distance, distance_iris, beta=0.01)
    offset_valid = distance_iris <= float(offset_valid_distance_iris)
    if bool(offset_valid.any().item()):
        offset_loss = F.smooth_l1_loss(
            offset[offset_valid], offset_to_surface_iris[offset_valid], beta=0.01
        )
    else:
        offset_loss = offset.sum() * 0.0
    if bool(normal_valid.any().item()):
        dot = (normal[normal_valid] * normal_unit[normal_valid]).sum(dim=-1).abs().clamp(0.0, 1.0)
        normal_loss = 1.0 - dot.mean()
    else:
        normal_loss = normal.sum() * 0.0
    near_loss = F.binary_cross_entropy_with_logits(near_logit, near_label.float())
    return {
        "distance": distance_loss,
        "offset": offset_loss,
        "normal": normal_loss,
        "near": near_loss,
    }
