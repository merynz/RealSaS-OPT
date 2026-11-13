from __future__ import annotations

"""Research-only SkinTokens hard-code and source-relative LBS consequence court.

This court measures the *actual MIRA discrete FSQ decoder*, not merely
the expected-code weights used during causal training. Pose transforms and
source dense weights are independent evaluation teachers; they are NEVER
conditions of RSS student inference. It does not certify rig correctness,
source animation parity, skin quality or a release.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from models.mira.skin_fsq_codec_v6 import MIRASkinFSQCodecV6
from .animation_ready_mesh import posed_lbs_from_joint_matrices


@dataclass(frozen=True)
class RSSvNextSkinPoseDiagnosticV1:
    source_skin_protocol_sha256: str
    teacher_pose_protocol_sha256: str
    codec_checkpoint_sha256: str
    valid_vertex_count: int
    valid_pose_vertex_count: int
    joint_count: int
    active_threshold: float
    expected_weight_mae: float
    hard_weight_mae: float
    expected_to_hard_weight_mae: float
    expected_to_hard_weight_max_abs: float
    hard_support_precision: float
    hard_support_recall: float
    hard_false_influence_mass: float
    hard_active_influence_mass_recall: float
    expected_to_teacher_lbs_mean_error: float
    hard_to_teacher_lbs_mean_error: float
    hard_to_teacher_lbs_max_error: float
    expected_to_hard_lbs_mean_error: float
    expected_to_hard_lbs_max_error: float
    authority: str = "RESEARCH_ONLY_MIRA_HARD_FSQ_TEACHER_RELATIVE_LBS_NO_PASS"


def _sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value
    )


def _mean(x: Tensor, mask: Tensor) -> float:
    selected = x[mask]
    if selected.numel() == 0:
        raise ValueError("RSS_VNEXT_SKIN_POSE_EMPTY_QUALIFIED_METRIC")
    return float(selected.double().mean().item())


@torch.no_grad()
def evaluate_mira_skin_pose_diagnostic_v1(
    *, codec: MIRASkinFSQCodecV6,
    hard_token_ids: Tensor,
    vertex_features: Tensor,
    joint_features: Tensor,
    rest_xyz: Tensor,
    source_dense_weights: Tensor,
    training_expected_weights: Tensor,
    source_skin_matrices: Tensor,
    qualified_vertex_mask: Tensor,
    qualified_pose_mask: Tensor,
    source_skin_protocol_sha256: str,
    teacher_pose_protocol_sha256: str,
    codec_checkpoint_sha256: str,
    active_threshold: float = 0.025,
) -> RSSvNextSkinPoseDiagnosticV1:
    """Decode exact hard FSQ IDs and measure sparse influence and LBS drift.

    All inputs are indexed to the same source mesh and joint order, confirmed
    separately by source and decoder receipts. The source LBS is the baseline
    implied by source weights + supplied qualified matrices; this does not
    itself prove observed posed-surface truth or collision-free motion.
    """
    if not isinstance(codec, MIRASkinFSQCodecV6) or codec.training:
        raise ValueError("RSS_VNEXT_SKIN_POSE_FROZEN_EVAL_CODEC_REQUIRED")
    if not all(_sha(s) for s in (
        source_skin_protocol_sha256, teacher_pose_protocol_sha256,
        codec_checkpoint_sha256,
    )):
        raise ValueError("RSS_VNEXT_SKIN_POSE_EXTERNAL_RECEIPTS_REQUIRED")
    if not 0 < active_threshold < 0.5:
        raise ValueError("RSS_VNEXT_SKIN_POSE_ACTIVE_THRESHOLD_INVALID")
    if rest_xyz.ndim != 3 or rest_xyz.shape[-1] != 3:
        raise ValueError("RSS_VNEXT_SKIN_POSE_REST_ADDRESS")
    b,v,_ = rest_xyz.shape
    if source_dense_weights.ndim != 3 or source_dense_weights.shape[:2] != (b,v):
        raise ValueError("RSS_VNEXT_SKIN_POSE_SOURCE_ADDRESS")
    j=source_dense_weights.shape[-1]
    if b<1 or v<1 or j<2:
        raise ValueError("RSS_VNEXT_SKIN_POSE_EMPTY_OR_SINGLE_JOINT")
    if (training_expected_weights.shape != (b,v,j)
        or vertex_features.shape != (b,v,codec.config.vertex_feature_dim)
        or joint_features.shape != (b,j,codec.config.joint_feature_dim)
        or hard_token_ids.shape != (b,j,codec.config.tokens_per_joint)
        or hard_token_ids.dtype != torch.long
        or source_skin_matrices.ndim != 5
        or source_skin_matrices.shape[0] != b
        or source_skin_matrices.shape[2:] != (j,4,4)):
        raise ValueError("RSS_VNEXT_SKIN_POSE_DECODER_OR_MATRIX_ADDRESS")
    p=source_skin_matrices.shape[1]
    if (p<1 or qualified_vertex_mask.shape!=(b,v)
        or qualified_vertex_mask.dtype != torch.bool
        or qualified_pose_mask.shape!=(b,p)
        or qualified_pose_mask.dtype != torch.bool
        or not qualified_vertex_mask.any()
        or not qualified_pose_mask.any()
        or not bool((qualified_vertex_mask.unsqueeze(1)
            & qualified_pose_mask.unsqueeze(-1)).any())):
        raise ValueError("RSS_VNEXT_SKIN_POSE_TEACHER_MASK_INVALID")

    payloads=(rest_xyz,vertex_features,joint_features,hard_token_ids,
              source_dense_weights,training_expected_weights,source_skin_matrices,
              qualified_vertex_mask,qualified_pose_mask)
    if len({t.device for t in payloads})!=1:
        raise ValueError("RSS_VNEXT_SKIN_POSE_DEVICE_DRIFT")
    if (not torch.is_floating_point(rest_xyz)
        or not all(t.dtype == rest_xyz.dtype for t in (
            vertex_features,joint_features,source_dense_weights,
            training_expected_weights,source_skin_matrices))):
        raise ValueError("RSS_VNEXT_SKIN_POSE_FLOAT_DTYPE_DRIFT")
    if (not bool(torch.isfinite(rest_xyz).all())
        or not bool(torch.isfinite(vertex_features).all())
        or not bool(torch.isfinite(joint_features).all())
        or not bool(torch.isfinite(training_expected_weights).all())
        or not bool(torch.isfinite(source_dense_weights[qualified_vertex_mask]).all())
        or not bool(torch.isfinite(source_skin_matrices[
            qualified_pose_mask]).all())):
        raise ValueError("RSS_VNEXT_SKIN_POSE_NONFINITE_QUALIFIED_INPUT")

    hard=codec.decode_tokens(hard_token_ids,vertex_features,joint_features)
    if (hard.shape!=(b,v,j) or not bool(torch.isfinite(hard).all())):
        raise ValueError("RSS_VNEXT_SKIN_POSE_HARD_DECODER_INVALID")
    if bool(((training_expected_weights<0)|(training_expected_weights>1)).any()
            or ((hard<0)|(hard>1)).any()):
        raise ValueError("RSS_VNEXT_SKIN_POSE_WEIGHT_RANGE")
    for w in (source_dense_weights[qualified_vertex_mask],
              training_expected_weights,hard):
        if not torch.allclose(
            w.sum(-1),torch.ones_like(w.sum(-1)),atol=1.e-5,rtol=0
        ):
            raise ValueError("RSS_VNEXT_SKIN_POSE_SIMPLEX_INVALID")

    vm=qualified_vertex_mask
    active=(source_dense_weights[vm] >= active_threshold)
    estimated=(hard[vm] >= active_threshold)
    true_pos=(active & estimated).sum()
    estimated_count=estimated.sum()
    active_count=active.sum()
    precision=float((true_pos / estimated_count.clamp_min(1)).item())
    recall=float((true_pos / active_count.clamp_min(1)).item())
    observed=source_dense_weights[vm]
    inferred=hard[vm]
    false_mass=float((inferred*~active).sum(dim=-1).mean().item())
    active_mass_recall=float(
        torch.minimum(inferred,observed)[active].sum().div(
            observed[active].sum().clamp_min(1.e-8)
        ).item()
    )

    # Fill *masked* unknown GT with an arbitrary well-formed prediction
    # because they must not enter a source-dependent valid metric.
    safe_source=torch.where(vm[...,None],source_dense_weights,hard)
    # Unknown pose slots may contain NaNs: never hand them to LBS.
    ident=torch.eye(4,device=rest_xyz.device,dtype=rest_xyz.dtype)
    safe_pose=torch.where(
        qualified_pose_mask[...,None,None,None],
        source_skin_matrices,ident[None,None,None],
    )
    teacher_xyz=posed_lbs_from_joint_matrices(rest_xyz,safe_source,safe_pose)
    soft_xyz=posed_lbs_from_joint_matrices(
        rest_xyz,training_expected_weights,safe_pose
    )
    hard_xyz=posed_lbs_from_joint_matrices(rest_xyz,hard,safe_pose)
    mask=qualified_pose_mask[:,:,None] & vm[:,None,:]
    source_soft=(soft_xyz-teacher_xyz).norm(dim=-1)
    source_hard=(hard_xyz-teacher_xyz).norm(dim=-1)
    soft_hard=(soft_xyz-hard_xyz).norm(dim=-1)

    return RSSvNextSkinPoseDiagnosticV1(
        source_skin_protocol_sha256=source_skin_protocol_sha256,
        teacher_pose_protocol_sha256=teacher_pose_protocol_sha256,
        codec_checkpoint_sha256=codec_checkpoint_sha256,
        valid_vertex_count=int(vm.sum().item()),
        valid_pose_vertex_count=int(mask.sum().item()),
        joint_count=j,
        active_threshold=active_threshold,
        expected_weight_mae=_mean(
            (training_expected_weights-source_dense_weights).abs().mean(-1),vm,
        ),
        hard_weight_mae=_mean((hard-source_dense_weights).abs().mean(-1),vm),
        expected_to_hard_weight_mae=_mean(
            (hard-training_expected_weights).abs().mean(-1),vm,
        ),
        expected_to_hard_weight_max_abs=float(
            (hard-training_expected_weights).abs()[vm].amax().item()
        ),
        hard_support_precision=precision,
        hard_support_recall=recall,
        hard_false_influence_mass=false_mass,
        hard_active_influence_mass_recall=active_mass_recall,
        expected_to_teacher_lbs_mean_error=_mean(source_soft,mask),
        hard_to_teacher_lbs_mean_error=_mean(source_hard,mask),
        hard_to_teacher_lbs_max_error=float(source_hard[mask].max().item()),
        expected_to_hard_lbs_mean_error=_mean(soft_hard,mask),
        expected_to_hard_lbs_max_error=float(soft_hard[mask].max().item()),
    )


__all__=["RSSvNextSkinPoseDiagnosticV1",
         "evaluate_mira_skin_pose_diagnostic_v1"]
