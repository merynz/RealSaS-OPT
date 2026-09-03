from __future__ import annotations

import hashlib
import json
import math
import torch

from .codec_deformation_loss_v1 import codec_deformation_loss_v1
from .skin_field_codec_v1 import SkinFieldCodecV1, skin_field_codec_loss_v1
from .train_codec_r6_a0_v1 import CodecA0QualificationTokenV1, canonical_metrics_hash_v1


def _criteria_hash(criteria: dict) -> str:
    return hashlib.sha256(json.dumps(criteria, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


@torch.no_grad()
def eval_codec_r6_a0_v1(
    codec: SkinFieldCodecV1,
    surface_features: torch.Tensor,
    joint_features: torch.Tensor,
    teacher_weights: torch.Tensor,
    surface_mask: torch.Tensor,
    joint_mask: torch.Tensor,
    rest_points: torch.Tensor,
    probe_transforms: torch.Tensor,
    *,
    source_gate: str,
    optimizer_steps: int,
    max_reconstruction: float,
    max_deformation_rms: float,
    max_simplex_residual: float,
) -> tuple[dict[str, float], CodecA0QualificationTokenV1]:
    criteria = {
        "max_reconstruction": float(max_reconstruction),
        "max_deformation_rms": float(max_deformation_rms),
        "max_simplex_residual": float(max_simplex_residual),
    }
    if any((not math.isfinite(v) or v < 0.0) for v in criteria.values()):
        raise ValueError("codec A0 criteria must be finite nonnegative")
    codec.eval()
    output = codec(surface_features, joint_features, teacher_weights, surface_mask, joint_mask)
    reconstruction = skin_field_codec_loss_v1(output.decoded_weights, teacher_weights, surface_mask, joint_mask)
    deformation = codec_deformation_loss_v1(output.decoded_weights, teacher_weights, rest_points, probe_transforms, surface_mask, joint_mask)
    valid_pair = surface_mask[:, :, None].bool() & joint_mask[:, None, :].bool()
    simplex = output.decoded_weights.sum(dim=-1)
    max_simplex = float((simplex[surface_mask.bool()] - 1.0).abs().max().cpu()) if surface_mask.any() else 0.0
    metrics = {
        "reconstruction": float(reconstruction["total"].cpu()),
        "l1": float(reconstruction["l1"].cpu()),
        "deformation_mse": float(deformation["deformation_mse"].cpu()),
        "deformation_rms": float(deformation["deformation_rms"].cpu()),
        "max_simplex_residual": max_simplex,
        "negative_weight_count": int((output.decoded_weights[valid_pair] < 0).sum().cpu()),
    }
    if any(not math.isfinite(float(v)) for v in metrics.values()):
        raise ValueError("non-finite codec A0 metrics")
    passed = (
        metrics["reconstruction"] <= criteria["max_reconstruction"]
        and metrics["deformation_rms"] <= criteria["max_deformation_rms"]
        and metrics["max_simplex_residual"] <= criteria["max_simplex_residual"]
        and metrics["negative_weight_count"] == 0
    )
    token = CodecA0QualificationTokenV1(
        status="PASS" if passed else "FAIL",
        codec_config_hash=codec.config.config_hash,
        source_gate=str(source_gate),
        optimizer_steps=int(optimizer_steps),
        criteria_hash=_criteria_hash(criteria),
        metrics_hash=canonical_metrics_hash_v1(metrics),
    )
    return metrics, token
