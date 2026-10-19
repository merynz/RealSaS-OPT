from __future__ import annotations

"""Low-capacity point probes for sealed V8 local evidence H.

H-REP-P0 is a representation diagnostic. It never extracts a mesh, never invokes
FlexiCubes/TESSA, and never rewrites V8. All probe arms use the exact same input
envelope and predictor parameter count; only which channels carry information
differs. This separates local H information from XYZ memorization, raw-field
information, and subject-level/nonlocal H statistics without a capacity confound.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

ProbeArmV1 = Literal["H_XYZ", "RAW_FIELD_XYZ", "XYZ_ONLY", "SHUFFLED_H_XYZ"]
PROBE_ARMS_V1: tuple[ProbeArmV1, ...] = (
    "H_XYZ", "RAW_FIELD_XYZ", "XYZ_ONLY", "SHUFFLED_H_XYZ",
)


@dataclass(frozen=True)
class HRepresentationProbeConfigV1:
    local_feature_dim: int = 384
    width: int = 256
    residual_blocks: int = 2
    maximum_offset_iris: float = 0.15

    def validate(self) -> None:
        if min(self.local_feature_dim, self.width, self.residual_blocks) <= 0:
            raise ValueError("H_REP_P0_CONFIG_WIDTH_INVALID")
        if not (0.0 < self.maximum_offset_iris <= 0.25):
            raise ValueError("H_REP_P0_OFFSET_BOUND_INVALID")


class _ResidualBlockV1(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.fc2(F.silu(self.fc1(self.norm(x))))) * (2.0 ** -0.5)


def probe_input_dim_v1(cfg: HRepresentationProbeConfigV1) -> int:
    """Every arm uses an identical H-width envelope plus XYZ."""
    return cfg.local_feature_dim + 3


def build_probe_input_v1(
    arm: ProbeArmV1,
    *,
    h: torch.Tensor,
    xyz_iris: torch.Tensor,
    raw_field: torch.Tensor,
    shuffled_h: torch.Tensor | None = None,
) -> torch.Tensor:
    if h.ndim != 2 or h.shape[-1] <= 0:
        raise ValueError("H_REP_P0_H_SHAPE_INVALID")
    if xyz_iris.shape != (h.shape[0], 3):
        raise ValueError("H_REP_P0_XYZ_SHAPE_INVALID")
    if raw_field.shape != (h.shape[0],):
        raise ValueError("H_REP_P0_FIELD_SHAPE_INVALID")

    if arm == "H_XYZ":
        evidence = h
    elif arm == "SHUFFLED_H_XYZ":
        if shuffled_h is None or shuffled_h.shape != h.shape:
            raise ValueError("H_REP_P0_SHUFFLED_H_SHAPE_INVALID")
        evidence = shuffled_h
    elif arm == "RAW_FIELD_XYZ":
        evidence = torch.zeros_like(h)
        evidence[:, 0] = raw_field
    elif arm == "XYZ_ONLY":
        evidence = torch.zeros_like(h)
    else:
        raise ValueError("H_REP_P0_ARM_INVALID:" + str(arm))
    return torch.cat((evidence, xyz_iris), dim=-1)


class HRepresentationProbeV1(nn.Module):
    architecture_id = "RealSaS.IRIS.HRepresentationProbe.v1"
    product_authority = False
    compiler_qualified = False

    def __init__(
        self,
        arm: ProbeArmV1,
        config: HRepresentationProbeConfigV1 = HRepresentationProbeConfigV1(),
    ):
        super().__init__()
        config.validate()
        if arm not in PROBE_ARMS_V1:
            raise ValueError("H_REP_P0_ARM_INVALID:" + str(arm))
        self.arm = arm
        self.config = config
        d = probe_input_dim_v1(config)
        self.input_norm = nn.LayerNorm(d)
        self.input_projection = nn.Linear(d, config.width)
        self.blocks = nn.ModuleList(
            [_ResidualBlockV1(config.width) for _ in range(config.residual_blocks)]
        )
        self.final_norm = nn.LayerNorm(config.width)
        self.distance_head = nn.Linear(config.width, 1)
        self.offset_head = nn.Linear(config.width, 3)
        self.normal_head = nn.Linear(config.width, 3)
        self.near_head = nn.Linear(config.width, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.input_norm.normalized_shape[0]:
            raise ValueError("H_REP_P0_INPUT_SHAPE_INVALID")
        z = F.silu(self.input_projection(self.input_norm(x)))
        for block in self.blocks:
            z = block(z)
        z = self.final_norm(z)
        return {
            "distance_iris": F.softplus(self.distance_head(z).squeeze(-1)),
            "offset_to_reference_iris": torch.tanh(self.offset_head(z))
            * float(self.config.maximum_offset_iris),
            "normal_unit": F.normalize(self.normal_head(z), dim=-1, eps=1e-8),
            "near_logit": self.near_head(z).squeeze(-1),
        }

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def h_rep_probe_objective_v1(
    prediction: dict[str, torch.Tensor],
    *,
    distance_iris: torch.Tensor,
    offset_to_reference_iris: torch.Tensor,
    normal_unit: torch.Tensor,
    normal_valid: torch.Tensor,
    near_label: torch.Tensor,
    offset_valid_distance_iris: float = 0.12,
) -> dict[str, torch.Tensor]:
    n = int(distance_iris.shape[0])
    pred_distance = prediction["distance_iris"]
    pred_offset = prediction["offset_to_reference_iris"]
    pred_normal = prediction["normal_unit"]
    near_logit = prediction["near_logit"]
    if pred_distance.shape != (n,) or pred_offset.shape != (n, 3) or pred_normal.shape != (n, 3):
        raise ValueError("H_REP_P0_PREDICTION_SHAPE_INVALID")
    if offset_to_reference_iris.shape != (n, 3) or normal_unit.shape != (n, 3):
        raise ValueError("H_REP_P0_TARGET_SHAPE_INVALID")
    if normal_valid.shape != (n,) or near_label.shape != (n,):
        raise ValueError("H_REP_P0_VALIDITY_SHAPE_INVALID")

    distance_loss = F.smooth_l1_loss(pred_distance, distance_iris, beta=0.01)
    offset_valid = distance_iris <= float(offset_valid_distance_iris)
    offset_loss = (
        F.smooth_l1_loss(
            pred_offset[offset_valid], offset_to_reference_iris[offset_valid], beta=0.01
        )
        if bool(offset_valid.any().item())
        else pred_offset.sum() * 0.0
    )
    if bool(normal_valid.any().item()):
        dot = (
            (pred_normal[normal_valid] * normal_unit[normal_valid])
            .sum(dim=-1)
            .abs()
            .clamp(0.0, 1.0)
        )
        normal_loss = 1.0 - dot.mean()
    else:
        normal_loss = pred_normal.sum() * 0.0
    near_loss = F.binary_cross_entropy_with_logits(near_logit, near_label.float())
    return {
        "total": distance_loss + offset_loss + 0.5 * normal_loss + 0.5 * near_loss,
        "distance": distance_loss,
        "offset": offset_loss,
        "normal": normal_loss,
        "near": near_loss,
    }


@torch.no_grad()
def h_rep_probe_metrics_v1(
    prediction: dict[str, torch.Tensor],
    *,
    xyz_iris: torch.Tensor,
    nearest_reference_iris: torch.Tensor,
    distance_iris: torch.Tensor,
    normal_unit: torch.Tensor,
    normal_valid: torch.Tensor,
    near_label: torch.Tensor,
    offset_valid_distance_iris: float = 0.12,
) -> dict[str, float]:
    pred_distance = prediction["distance_iris"].float()
    abs_error = (pred_distance - distance_iris.float()).abs()
    offset_valid = distance_iris <= float(offset_valid_distance_iris)
    endpoint = xyz_iris + prediction["offset_to_reference_iris"].float()
    endpoint_error = torch.linalg.vector_norm(endpoint - nearest_reference_iris, dim=-1)
    if bool(normal_valid.any().item()):
        normal_mean = float(
            (
                prediction["normal_unit"][normal_valid].float()
                * normal_unit[normal_valid].float()
            )
            .sum(dim=-1)
            .abs()
            .clamp(0.0, 1.0)
            .mean()
            .item()
        )
    else:
        normal_mean = float("nan")
    pred_near = prediction["near_logit"] >= 0
    truth = near_label.bool()
    pos = truth
    neg = ~truth
    tpr = (
        float((pred_near[pos] == truth[pos]).float().mean().item())
        if bool(pos.any().item())
        else float("nan")
    )
    tnr = (
        float((pred_near[neg] == truth[neg]).float().mean().item())
        if bool(neg.any().item())
        else float("nan")
    )
    return {
        "distance_mae_iris": float(abs_error.mean().item()),
        "distance_abs_error_p95_iris": float(torch.quantile(abs_error, 0.95).item()),
        "endpoint_error_p95_iris": float(
            torch.quantile(endpoint_error[offset_valid], 0.95).item()
        )
        if bool(offset_valid.any().item())
        else float("nan"),
        "normal_absolute_dot_mean": normal_mean,
        "near_balanced_accuracy": 0.5 * (tpr + tnr),
        "near_tpr": tpr,
        "near_tnr": tnr,
    }
