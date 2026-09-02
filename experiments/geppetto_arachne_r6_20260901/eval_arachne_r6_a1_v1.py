from __future__ import annotations

import math
import torch

from .arachne_candidate_v2 import ArachneCandidateV2
from .arachne_tail_objective_v1 import row_l1_error_v1, top_fraction_row_l1_tail_v1
from .codec_deformation_loss_v1 import codec_deformation_loss_v1
from .skin_field_codec_v1 import skin_field_codec_loss_v1
from .train_arachne_r6_a1_v1 import _require_lineage
from .train_codec_r6_a0_v1 import CodecA0QualificationTokenV1


@torch.no_grad()
def eval_arachne_r6_a1_v1(
    model: ArachneCandidateV2,
    conditioning,
    teacher_weights: torch.Tensor,
    rest_points: torch.Tensor,
    probe_transforms: torch.Tensor,
    *,
    a0_token: CodecA0QualificationTokenV1,
    expected_surface_hashes: tuple[str, ...],
    expected_skeleton_hashes: tuple[str, ...],
    tail_fraction: float = 0.10,
) -> dict[str, float]:
    a0_token.validate_for(model.codec)
    _require_lineage(conditioning, expected_surface_hashes, expected_skeleton_hashes)
    if any(parameter.requires_grad for parameter in model.codec.parameters()):
        raise ValueError("Arachne A1 evaluation requires frozen qualified codec")
    if not (0.0 < float(tail_fraction) <= 1.0):
        raise ValueError("invalid Arachne tail fraction")
    device = next(model.parameters()).device
    sf = torch.as_tensor(conditioning.surface_features, device=device, dtype=torch.float32)
    jf = torch.as_tensor(conditioning.joint_features, device=device, dtype=torch.float32)
    sm = torch.as_tensor(conditioning.surface_mask, device=device, dtype=torch.bool)
    jm = torch.as_tensor(conditioning.joint_mask, device=device, dtype=torch.bool)
    pi = torch.as_tensor(conditioning.parent_indices, device=device, dtype=torch.long)
    pg = torch.as_tensor(conditioning.pair_geometry, device=device, dtype=torch.float32)
    pm = torch.as_tensor(conditioning.pair_mask, device=device, dtype=torch.bool)
    teacher_weights = teacher_weights.to(device=device, dtype=torch.float32)
    rest_points = rest_points.to(device=device, dtype=torch.float32)
    probe_transforms = probe_transforms.to(device=device, dtype=torch.float32)
    model.eval()
    out = model(sf, jf, sm, jm, pi, pg, pm)
    teacher_latents = model.codec.encode_teacher_weights(sf, jf, teacher_weights, sm, jm)
    valid = jm[..., None].expand_as(out.joint_latent_mean)
    residual = out.joint_latent_mean - teacher_latents
    ls = out.joint_latent_log_sigma
    latent_nll = (0.5 * torch.exp(-2 * ls) * residual.square() + ls)[valid].mean()
    reconstruction = skin_field_codec_loss_v1(out.decoded_weights, teacher_weights, sm, jm)
    deformation = codec_deformation_loss_v1(out.decoded_weights, teacher_weights, rest_points, probe_transforms, sm, jm)
    rows = row_l1_error_v1(out.decoded_weights, teacher_weights, sm)
    tail = top_fraction_row_l1_tail_v1(out.decoded_weights, teacher_weights, sm, fraction=tail_fraction)
    simplex = out.decoded_weights.sum(-1)
    metrics = {
        "latent_nll": float(latent_nll.cpu()),
        "reconstruction": float(reconstruction["total"].cpu()),
        "row_l1_mean": float(rows.mean().cpu()),
        "row_l1_p95": float(torch.quantile(rows, 0.95).cpu()),
        "tail_row_l1": float(tail.cpu()),
        "deformation_rms": float(deformation["deformation_rms"].cpu()),
        "uncertainty_mean": float(torch.exp(ls[valid]).mean().cpu()),
        "max_simplex_residual": float((simplex[sm] - 1.0).abs().max().cpu()),
        "negative_weight_count": int((out.decoded_weights[(sm[:, :, None] & jm[:, None, :])] < 0).sum().cpu()),
    }
    if any(not math.isfinite(float(v)) for v in metrics.values()):
        raise ValueError("non-finite Arachne A1 metrics")
    return metrics
