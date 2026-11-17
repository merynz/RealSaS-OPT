from __future__ import annotations

"""Pose-conditioned differentiable mechanical supervision on exact TESSA M*.

This is NOT a G3/G3B acceptance gate. It compares ALL three edges and
triangle deformation metrics in every independently qualified pose, then
adds a maximum local G3/G3B-style barrier. A tiny collapsed face cannot
vanish in a mean pose/mesh reconstruction loss. The actual Compiler alone
qualifies G3/G3B and retargeted motion clips after AXIS/MIRA.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from models.tessa.v1.mechanical_objective_v1 import (
    triangle_deformation_metrics_v1,
)
from .seeded_dynamic_full_face_court_v1 import SeededDynamicLimitsV1


def exact_m_qualified_pose_strain_v1(
    *,rest_xyz:Tensor,faces:Tensor,
    predicted_poses:Tensor,source_native_posed:Tensor,
    source_pose_mask:Tensor,source_candidate_skin:Tensor,
    scale:Tensor,
    limits:SeededDynamicLimitsV1=SeededDynamicLimitsV1(),
) -> dict[str,Tensor]:
    """Differentiable all-face source-relative pose fidelity + hard risk.

    Requires source-native barycentrically supported pose targets already
    validated and sealed by NativeCandidatePoseTransport before entry.
    No geometry supervision inferred from student or generated views.
    """
    limits.validate()
    if (
        rest_xyz.ndim!=3 or rest_xyz.shape[0]!=1
        or rest_xyz.shape[-1]!=3
        or faces.ndim!=2 or faces.shape[-1]!=3
        or faces.dtype!=torch.long or not len(faces)
        or len(faces)>limits.max_faces
        or predicted_poses.ndim!=4
        or predicted_poses.shape!=source_native_posed.shape
        or predicted_poses.shape[:1]!=(1,)
        or predicted_poses.shape[-2:]!=rest_xyz.shape[-2:]
        or source_pose_mask.shape!=predicted_poses.shape[:3]
        or source_pose_mask.dtype!=torch.bool
        or not bool(source_pose_mask.any())
        or source_candidate_skin.shape[:2]!=rest_xyz.shape[:2]
        or scale.ndim!=0 or not bool(torch.isfinite(scale))
        or float(scale.detach())<=0
        or any(x.device!=rest_xyz.device for x in (
            faces,predicted_poses,source_native_posed,
            source_pose_mask,source_candidate_skin,scale,
        ))
        or any(not bool(torch.isfinite(x).all()) for x in (
            rest_xyz,predicted_poses,source_native_posed,
            source_candidate_skin,
        ))
        or bool((faces<0).any()) or int(faces.max())>=rest_xyz.shape[1]
    ):
        raise ValueError("RSS_VNEXT_TESSA_POSE_CONSEQUENCE_DOMAIN")
    violations=[]; relative=[]; eligible=0
    skin=source_candidate_skin[0,faces]
    discontinuity=torch.stack((
        (skin[:,0]-skin[:,1]).abs().sum(-1),
        (skin[:,1]-skin[:,2]).abs().sum(-1),
        (skin[:,2]-skin[:,0]).abs().sum(-1),
    )).amax(dim=0)
    risky=discontinuity>limits.g3b_skin_l1_risk_threshold

    for p in range(predicted_poses.shape[1]):
        valid=source_pose_mask[0,p,faces].all(dim=1)
        if not bool(valid.any()):
            raise ValueError("RSS_VNEXT_TESSA_POSE_NO_FULLY_QUALIFIED_FACE")
        eligible+=int(valid.sum().item())
        f=faces[valid]
        r=rest_xyz
        moving=predicted_poses[:,p]
        reference=source_native_posed[:,p]
        pred_tri=moving[:,f]
        ref_tri=reference[:,f]
        def edges(tri):
            return torch.stack((
                (tri[:,:,1]-tri[:,:,0]).norm(dim=-1),
                (tri[:,:,2]-tri[:,:,1]).norm(dim=-1),
                (tri[:,:,0]-tri[:,:,2]).norm(dim=-1),
            ),dim=-1)
        pose_edge=edges(pred_tri)
        truth_edge=edges(ref_tri)
        edge_fidelity=F.smooth_l1_loss(
            pose_edge/scale,truth_edge/scale,
        )
        metrics=triangle_deformation_metrics_v1(
            r,moving,f,
        )
        oracle=triangle_deformation_metrics_v1(
            r.detach(),reference.detach(),f,
        )
        # The oracle is source 3D motion on the same verified barycentric
        # addresses, never a second student surface or guessed joint order.
        area_fidelity=F.smooth_l1_loss(
            metrics.area_ratio,oracle.area_ratio,
        )
        condition_fidelity=F.smooth_l1_loss(
            torch.log(metrics.condition_number.clamp_min(1.)),
            torch.log(oracle.condition_number.clamp_min(1.)),
        )
        relative.append(
            edge_fidelity+.25*area_fidelity+.10*condition_fidelity
        )
        area=metrics.area_ratio
        cond=metrics.condition_number
        max_edge=metrics.max_edge_ratio
        # All-face WORST-CASE normalized barrier, not mean. A bad source
        # reference is never relabelled "animation ready" by this loss.
        limit_risk=torch.stack((
            F.relu(limits.min_area_ratio-area)/limits.min_area_ratio,
            F.relu(area-limits.max_area_ratio)/limits.max_area_ratio,
            F.relu(cond-limits.max_condition_number)/limits.max_condition_number,
            F.relu(max_edge-limits.max_edge_ratio)/limits.max_edge_ratio,
        ),dim=-1).amax(dim=-1)
        worst=limit_risk.amax()
        risky_f=risky[valid]
        worst_skin=(
            limit_risk[:,risky_f].amax()
            if bool(risky_f.any()) else worst.new_zeros(())
        )
        violations.append(worst+.5*worst_skin)
    rel=torch.stack(relative).mean()
    worst=torch.stack(violations).amax()
    total=rel+0.20*worst
    if not bool(torch.isfinite(total)):
        raise ValueError("RSS_VNEXT_TESSA_POSE_CONSEQUENCE_NONFINITE")
    return {
        "total":total,
        "source_relative":rel,
        "g3_g3b_worst_barrier":worst,
        "qualified_face_pose_count":total.new_tensor(eligible),
    }


__all__=["exact_m_qualified_pose_strain_v1"]
