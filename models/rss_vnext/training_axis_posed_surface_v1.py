from __future__ import annotations

"""Source-qualified posed-surface consequence for the *unified* RSS.vNext trainer.

Unlike a SkinTokens sequence CE or codec-only posed-LBS loss, this applies
source-native joint transforms to the student's *TESSA mesh + hard-token MIRA
skin proposal*. It is deliberately opt-in and teacher-mesh-conditioned.
Neither a generated camera, an Idle, nor an unqualified animation can supply
its targets. It does NOT establish teacher-free topology/rigging parity.
"""

from typing import Callable, Mapping

import torch
from torch import Tensor

from .animation_ready_mesh import (
    RSSvNextStructuralMeshCandidate,
    source_pose_surface_consequence_loss,
)


def qualified_unified_posed_surface_objective_v1(
    *,
    mesh: RSSvNextStructuralMeshCandidate,
    predicted_skin: Tensor,
    joint_skin_matrices: Tensor,
    source_rest_surface_samples: Tensor,
    source_posed_surface_samples: Tensor,
    valid_pose_mask: Tensor,
    receipt: Mapping[str, object] | None,
    expected_receipt: Mapping[str, object],
) -> tuple[Tensor, int, Tensor]:
    """Return (static+posed surface loss, eligible poses, posed-only loss).

    The caller MUST bind expected_receipt to the trusted launcher registry,
    exact source observation, dataset, TESSA mesh address and AXIS teacher.
    Byte hashes cannot grant source truth by themselves; the external foundry
    must independently issue the qualified source-motion admission.
    """
    if (
        receipt is None
        or any(receipt.get(k) != v for k, v in expected_receipt.items())
        or receipt.get("admission_status")
        != "QUALIFIED_SOURCE_NATIVE_AXIS_TESSA_POSED_SURFACE"
        or receipt.get("pose_witness_source")
        != "INDEPENDENT_SOURCE_NATIVE_LBS_MESH"
        or receipt.get("generated_view_is_motion_teacher") is not False
        or receipt.get("reindexed_tessa_vertices_and_axis_joints") is not True
        or receipt.get("normalized_geometry_domain")
        != "IRIS_NORMALIZED_SOURCE_XYZ"
    ):
        raise ValueError("RSS_VNEXT_UNIFIED_POSE_RECEIPT_UNQUALIFIED")

    mesh.validate()
    if (
        mesh.source_authority != "TEACHER_FORCED_TRAINING_ONLY"
        or predicted_skin.ndim != 3
        or predicted_skin.shape[:2] != mesh.rest_vertices_iris.shape[:2]
        or joint_skin_matrices.ndim != 5
        or joint_skin_matrices.shape[0] != 1
        or joint_skin_matrices.shape[2:] != (predicted_skin.shape[-1], 4, 4)
        or not 1 <= joint_skin_matrices.shape[1] <= 16
        or predicted_skin.shape[-1] < 2
        or source_rest_surface_samples.ndim != 2
        or source_rest_surface_samples.shape[-1] != 3
        or not 3 <= source_rest_surface_samples.shape[0] <= 2048
        or source_posed_surface_samples.ndim != 3
        or source_posed_surface_samples.shape[:2]
        != (joint_skin_matrices.shape[1], source_rest_surface_samples.shape[0])
        or source_posed_surface_samples.shape[-1] != 3
        or valid_pose_mask.shape != (joint_skin_matrices.shape[1],)
        or valid_pose_mask.dtype != torch.bool
        or not bool(valid_pose_mask.any())
    ):
        raise ValueError("RSS_VNEXT_UNIFIED_POSE_DOMAIN_INVALID")

    source = (
        joint_skin_matrices,
        source_rest_surface_samples,
        source_posed_surface_samples,
        valid_pose_mask,
    )
    if (
        any(x.device != predicted_skin.device for x in source)
        or any(x.requires_grad for x in source)
        or any(not bool(torch.isfinite(x).all()) for x in source[:3])
        or not bool(torch.allclose(
            joint_skin_matrices[..., 3, :],
            joint_skin_matrices.new_tensor((0., 0., 0., 1.)),
            rtol=0., atol=1e-5,
        ))
        or not bool(torch.isfinite(predicted_skin).all())
    ):
        raise ValueError("RSS_VNEXT_UNIFIED_POSE_NUMERIC_OR_TEACHER_GRAD_INVALID")

    # An Idle/global rigid transform is not evidence of skin articulation.
    # Require different joint transports at at least one admitted source pose;
    # unexcited poses can still appear, but cannot qualify the whole batch.
    with torch.no_grad():
        rest = source_rest_surface_samples
        homogeneous = torch.cat((rest, torch.ones_like(rest[:, :1])), dim=-1)
        per_joint = torch.einsum(
            "bpjkc,vc->bpvjk", joint_skin_matrices, homogeneous,
        )[..., :3]
        spread = (per_joint.amax(dim=3) - per_joint.amin(dim=3)).abs().amax(dim=-1)
        if not bool((spread[0, valid_pose_mask] > 1e-5).any()):
            raise ValueError("RSS_VNEXT_UNIFIED_POSE_NO_DIFFERENTIAL_JOINT_MOTION")

    losses = source_pose_surface_consequence_loss(
        mesh,
        predicted_skin,
        joint_skin_matrices,
        source_rest_surface_samples,
        source_posed_surface_samples,
        valid_pose_mask=valid_pose_mask,
    )
    if not bool(torch.isfinite(losses["total"])):
        raise ValueError("RSS_VNEXT_UNIFIED_POSE_LOSS_NONFINITE")
    return (
        losses["total"],
        int(losses["qualified_pose_count"].item()),
        losses["posed_surface"],
    )


__all__ = ["qualified_unified_posed_surface_objective_v1"]
