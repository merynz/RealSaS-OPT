from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import torch

from .codec_deformation_loss_v1 import codec_deformation_loss_v1
from .skin_field_codec_v1 import SkinFieldCodecV1, skin_field_codec_loss_v1


@dataclass(frozen=True)
class CodecA0QualificationTokenV1:
    status: str
    codec_config_hash: str
    source_gate: str
    optimizer_steps: int
    criteria_hash: str
    metrics_hash: str
    schema_version: str = "RealSaS.CodecA0QualificationToken.v1"

    def validate_for(self, codec: SkinFieldCodecV1) -> None:
        if self.status != "PASS":
            raise ValueError("codec A0 qualification token is not PASS")
        if self.codec_config_hash != codec.config.config_hash:
            raise ValueError("codec A0 token config hash mismatch")
        if not self.source_gate or not self.criteria_hash or not self.metrics_hash:
            raise ValueError("codec A0 token missing qualification provenance")
        if self.optimizer_steps < 0:
            raise ValueError("codec A0 token optimizer_steps invalid")


def canonical_metrics_hash_v1(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def train_codec_r6_a0_step_v1(
    codec: SkinFieldCodecV1,
    optimizer,
    surface_features: torch.Tensor,
    joint_features: torch.Tensor,
    teacher_weights: torch.Tensor,
    surface_mask: torch.Tensor,
    joint_mask: torch.Tensor,
    rest_points: torch.Tensor,
    probe_transforms: torch.Tensor,
    *,
    reconstruction_weight: float = 1.0,
    deformation_weight: float = 1.0,
) -> dict[str, float]:
    if reconstruction_weight <= 0 or deformation_weight < 0:
        raise ValueError("invalid codec loss weights")
    codec.train(); optimizer.zero_grad(set_to_none=True)
    output = codec(surface_features, joint_features, teacher_weights, surface_mask, joint_mask)
    reconstruction = skin_field_codec_loss_v1(output.decoded_weights, teacher_weights, surface_mask, joint_mask)
    deformation = codec_deformation_loss_v1(output.decoded_weights, teacher_weights, rest_points, probe_transforms, surface_mask, joint_mask)
    total = reconstruction_weight * reconstruction["total"] + deformation_weight * deformation["deformation_mse"]
    total.backward(); optimizer.step()
    return {
        "total": float(total.detach().cpu()),
        "reconstruction": float(reconstruction["total"].detach().cpu()),
        "cross_entropy": float(reconstruction["cross_entropy"].detach().cpu()),
        "l1": float(reconstruction["l1"].detach().cpu()),
        "deformation_mse": float(deformation["deformation_mse"].detach().cpu()),
        "deformation_rms": float(deformation["deformation_rms"].detach().cpu()),
    }
