from __future__ import annotations

import math
import torch

from .arachne_candidate_v2 import ArachneCandidateV2
from .codec_deformation_loss_v1 import codec_deformation_loss_v1
from .skin_field_codec_v1 import skin_field_codec_loss_v1
from .train_codec_r6_a0_v1 import CodecA0QualificationTokenV1


def _require_lineage(conditioning, expected_surface_hashes, expected_skeleton_hashes) -> None:
    if tuple(conditioning.source_surface_hashes) != tuple(expected_surface_hashes):
        raise ValueError("Arachne A1 surface lineage mismatch")
    if tuple(conditioning.source_skeleton_hashes) != tuple(expected_skeleton_hashes):
        raise ValueError("Arachne A1 skeleton lineage mismatch")


def train_arachne_r6_a1_step_v1(
    model: ArachneCandidateV2,
    optimizer,
    conditioning,
    teacher_weights: torch.Tensor,
    rest_points: torch.Tensor,
    probe_transforms: torch.Tensor,
    *,
    a0_token: CodecA0QualificationTokenV1,
    expected_surface_hashes: tuple[str, ...],
    expected_skeleton_hashes: tuple[str, ...],
    latent_nll_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    deformation_weight: float = 1.0,
) -> dict[str, float]:
    a0_token.validate_for(model.codec)
    _require_lineage(conditioning, expected_surface_hashes, expected_skeleton_hashes)
    if any(parameter.requires_grad for parameter in model.codec.parameters()):
        raise ValueError("Arachne A1 requires frozen qualified codec")
    if min(latent_nll_weight, reconstruction_weight, deformation_weight) < 0 or latent_nll_weight == 0:
        raise ValueError("invalid Arachne A1 loss weights")
    device = next(p for p in model.parameters() if p.requires_grad).device
    sf=torch.as_tensor(conditioning.surface_features,device=device,dtype=torch.float32); jf=torch.as_tensor(conditioning.joint_features,device=device,dtype=torch.float32); sm=torch.as_tensor(conditioning.surface_mask,device=device,dtype=torch.bool); jm=torch.as_tensor(conditioning.joint_mask,device=device,dtype=torch.bool); pi=torch.as_tensor(conditioning.parent_indices,device=device,dtype=torch.long); pg=torch.as_tensor(conditioning.pair_geometry,device=device,dtype=torch.float32); pm=torch.as_tensor(conditioning.pair_mask,device=device,dtype=torch.bool); teacher_weights=teacher_weights.to(device=device,dtype=torch.float32); rest_points=rest_points.to(device=device,dtype=torch.float32); probe_transforms=probe_transforms.to(device=device,dtype=torch.float32)
    with torch.no_grad():
        teacher_latents=model.codec.encode_teacher_weights(sf,jf,teacher_weights,sm,jm)
    model.train(); optimizer.zero_grad(set_to_none=True); out=model(sf,jf,sm,jm,pi,pg,pm)
    valid_latent=jm[...,None].expand_as(out.joint_latent_mean)
    residual=out.joint_latent_mean-teacher_latents; ls=out.joint_latent_log_sigma; nll=(0.5*torch.exp(-2.0*ls)*residual.square()+ls); latent_nll=nll[valid_latent].mean()
    reconstruction=skin_field_codec_loss_v1(out.decoded_weights,teacher_weights,sm,jm)
    deformation=codec_deformation_loss_v1(out.decoded_weights,teacher_weights,rest_points,probe_transforms,sm,jm)
    total=latent_nll_weight*latent_nll+reconstruction_weight*reconstruction["total"]+deformation_weight*deformation["deformation_mse"]
    if not torch.isfinite(total): raise ValueError("non-finite Arachne A1 loss")
    total.backward(); optimizer.step()
    return {"total":float(total.detach().cpu()),"latent_nll":float(latent_nll.detach().cpu()),"reconstruction":float(reconstruction["total"].detach().cpu()),"deformation_rms":float(deformation["deformation_rms"].detach().cpu()),"uncertainty_mean":float(torch.exp(ls[valid_latent]).mean().detach().cpu())}
