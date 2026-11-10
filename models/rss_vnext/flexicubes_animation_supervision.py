from __future__ import annotations

"""Animation-sequence supervision for the SAME differentiable FlexiCubes surface.

Provenance: NVIDIA FlexiCubes paper supplement §4.2:
  (a) learn reasonable rest geometry;
  (b) EACH animation optimization step re-skins the *current extracted mesh*
      by inverse-distance kNN weights from source rigged rest vertices;
  (c) evaluate a random/full qualified pose and view, backpropagate through
      LBS → extracted surface XYZ → FlexiCubes field/alpha/beta/gamma → Z/H.

Unlike NVIDIA's per-subject reconstruction experiment, RSS learns a general
two-observed-image network and may not read the source rig/skin at inference.
All source asset/rig/pose data enters only the training loss. These teacher
identifiers are NOT proof of source rights or correspondence; those checks
belong to the external admission/qualification producer.

Caveats deliberately enforced:
  - source domain is ONE admitted closed component; cross-component kNN
    transfer is unsafe and requires a separate component-aware method;
  - nearest-neighbor INDEX selection is discrete, but interpolation weights
    and skinning are differentiable between neighbor-selection changes;
  - FC connectivity can still change and is NOT TESSA final connectivity;
  - no absolute minimum-strain prior punishes legitimate source articulation.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .animation_ready_mesh import (
    _area_weighted_surface_witnesses,
    _two_way_surface_chamfer,
    posed_lbs_from_joint_matrices,
)
from .flexicubes_surface_evidence import RSSvNextFlexiCubesSurfaceEvidence


@dataclass(frozen=True)
class RSSvNextFCAnimationTeacher:
    """External-court-qualified training-only source evidence.

    The source arrays are in the canonical IRIS coordinate system and the
    matrices are REST_TO_POSE skinning matrices (including inverse bind).
    The hashes are an external qualification *reference*, not a self-cert.
    """

    source_asset_sha256: str
    source_rig_binding_sha256: str
    source_pose_suite_sha256: str
    source_admission_witness_id: str
    source_component_domain: str   # SINGLE_QUALIFIED_CLOSED_COMPONENT
    rest_vertices: Tensor          # [R,3]
    rest_skin_weights: Tensor       # [R,J]
    posed_skin_matrices: Tensor     # [P,J,4,4]
    rest_surface_samples: Tensor   # [S0,3]
    pose_surface_samples: Tensor   # [P,S,3]
    pose_valid_mask: Tensor        # [P], bool

    def validate(self, *, device: torch.device, dtype: torch.dtype) -> None:
        for token in (
            self.source_asset_sha256, self.source_rig_binding_sha256,
            self.source_pose_suite_sha256, self.source_admission_witness_id,
        ):
            if not token:
                raise ValueError("RSS_VNEXT_FC_ANIM_MISSING_SOURCE_PROVENANCE")
        if self.source_component_domain != "SINGLE_QUALIFIED_CLOSED_COMPONENT":
            raise ValueError("RSS_VNEXT_FC_ANIM_COMPONENT_TRANSFER_UNSAFE")
        x, w, mat = self.rest_vertices, self.rest_skin_weights, self.posed_skin_matrices
        if x.ndim != 2 or x.shape[-1] != 3 or x.shape[0] < 3:
            raise ValueError("RSS_VNEXT_FC_ANIM_SOURCE_VERTEX_DOMAIN")
        if w.ndim != 2 or w.shape[0] != x.shape[0] or w.shape[1] < 1:
            raise ValueError("RSS_VNEXT_FC_ANIM_SOURCE_SKIN_DOMAIN")
        if mat.ndim != 4 or mat.shape[1:] != (w.shape[1], 4, 4) or mat.shape[0] < 1:
            raise ValueError("RSS_VNEXT_FC_ANIM_SOURCE_MATRICES_DOMAIN")
        if (self.rest_surface_samples.ndim != 2 or self.rest_surface_samples.shape[-1] != 3
                or self.rest_surface_samples.shape[0] < 3):
            raise ValueError("RSS_VNEXT_FC_ANIM_REST_SAMPLES_DOMAIN")
        if (self.pose_surface_samples.ndim != 3
                or self.pose_surface_samples.shape[:1] != (mat.shape[0],)
                or self.pose_surface_samples.shape[-1] != 3
                or self.pose_surface_samples.shape[1] < 3):
            raise ValueError("RSS_VNEXT_FC_ANIM_POSE_SAMPLES_DOMAIN")
        if (self.pose_valid_mask.shape != (mat.shape[0],)
                or self.pose_valid_mask.dtype != torch.bool
                or not bool(self.pose_valid_mask.any())):
            raise ValueError("RSS_VNEXT_FC_ANIM_NO_QUALIFIED_POSE")
        for obj in (x, w, mat, self.rest_surface_samples,
                    self.pose_surface_samples):
            if obj.device != device or obj.dtype != dtype:
                raise ValueError("RSS_VNEXT_FC_ANIM_DEVICE_OR_DTYPE_DRIFT")
            if not bool(torch.isfinite(obj).all()):
                raise ValueError("RSS_VNEXT_FC_ANIM_NONFINITE_TEACHER")
        if self.pose_valid_mask.device != device:
            raise ValueError("RSS_VNEXT_FC_ANIM_POSE_MASK_DEVICE")
        if bool((w < -1e-6).any() or (w.sum(-1) - 1).abs().max() > 1e-4):
            raise ValueError("RSS_VNEXT_FC_ANIM_SOURCE_WEIGHT_SIMPLEX")
        # Only valid teacher motion frames are required to be finite, rather
        # than allowing missing/unqualified poses to produce fabricated truth.
        # This initial implementation requires dense finite arrays for safety.

@dataclass(frozen=True)
class RSSvNextFCAnimationLossConfig:
    knn: int = 10
    rest_weight: float = 1.0
    pose_weight: float = 1.0
    max_sampled_faces: int = 128
    max_reference_vertices: int = 10000

    def validate(self) -> None:
        if not (1 <= self.knn <= 32 and 2 <= self.max_sampled_faces <= 4096
                and self.max_reference_vertices >= self.knn):
            raise ValueError("RSS_VNEXT_FC_ANIM_RESOURCE_BUDGET")
        if self.rest_weight < 0 or self.pose_weight <= 0:
            raise ValueError("RSS_VNEXT_FC_ANIM_LOSS_WEIGHTS")


@dataclass
class RSSvNextFCAnimationLossResult:
    total: Tensor
    rest_surface: Tensor
    posed_surface: Tensor
    transferred_skin: Tensor
    posed_vertices: Tensor
    pose_count: int
    authority: str = "TRAINING_SOURCE_RELATIVE_FC_ANIMATION_ONLY"


def differentiable_source_skin_transfer(
    extracted_xyz: Tensor,
    reference_xyz: Tensor,
    reference_weights: Tensor,
    *,
    k: int = 10,
    epsilon: float = 1e-3,
    max_reference_vertices: int = 10000,
) -> Tensor:
    """NVIDIA-supplement inverse-distance kNN transfer, with autograd.

    The discrete topk indices are *not* differentiable; weights computed from
    the selected distances are. A reference source mesh may have a different
    face/vertex count than the currently extracted FC surface at each step.
    """
    if (extracted_xyz.ndim != 3 or extracted_xyz.shape[0] != 1
            or extracted_xyz.shape[-1] != 3 or extracted_xyz.shape[1] < 3):
        raise ValueError("RSS_VNEXT_FC_SKIN_EXTRACTED_XYZ_SHAPE")
    if reference_xyz.ndim != 2 or reference_xyz.shape[-1] != 3:
        raise ValueError("RSS_VNEXT_FC_SKIN_REFERENCE_XYZ_SHAPE")
    if (reference_weights.ndim != 2
            or reference_weights.shape[0] != reference_xyz.shape[0]):
        raise ValueError("RSS_VNEXT_FC_SKIN_REFERENCE_WEIGHT_SHAPE")
    if (not 1 <= k <= reference_xyz.shape[0]
            or reference_xyz.shape[0] > max_reference_vertices):
        raise ValueError("RSS_VNEXT_FC_SKIN_KNN_REFERENCE_BUDGET")
    if epsilon <= 0:
        raise ValueError("RSS_VNEXT_FC_SKIN_EPSILON_INVALID")
    if (extracted_xyz.device != reference_xyz.device
            or reference_xyz.device != reference_weights.device
            or extracted_xyz.dtype != reference_xyz.dtype
            or reference_xyz.dtype != reference_weights.dtype):
        raise ValueError("RSS_VNEXT_FC_SKIN_COORDINATE_DEVICE_DRIFT")
    if not all(bool(torch.isfinite(t).all())
               for t in (extracted_xyz, reference_xyz, reference_weights)):
        raise ValueError("RSS_VNEXT_FC_SKIN_NONFINITE_INPUT")
    if bool((reference_weights < -1e-6).any()
            or (reference_weights.sum(-1) - 1).abs().max() > 1e-4):
        raise ValueError("RSS_VNEXT_FC_SKIN_INVALID_REFERENCE_WEIGHTS")
    distance = torch.cdist(extracted_xyz.float(), reference_xyz[None].float())[0]
    nearest_distance, nearest_idx = distance.topk(k, dim=-1, largest=False)
    closeness = nearest_distance.clamp_min(epsilon).reciprocal()
    alpha = closeness / closeness.sum(dim=-1, keepdim=True)
    result = (alpha[..., None] * reference_weights[nearest_idx].float()).sum(dim=1)
    return result.to(dtype=extracted_xyz.dtype)[None]


def source_animation_flexicubes_loss(
    surface: RSSvNextFlexiCubesSurfaceEvidence,
    teacher: RSSvNextFCAnimationTeacher,
    *,
    config: RSSvNextFCAnimationLossConfig = RSSvNextFCAnimationLossConfig(),
) -> RSSvNextFCAnimationLossResult:
    """One real extracted FC mesh, all admitted poses, one differentiable loss.

    The caller must already establish paired mesh/rig/pose source authority.
    The loss cannot certify that authority or silently infer skeletal motion
    from 2D images. Production inference must never invoke this operation.
    """
    config.validate()
    if surface.evidence_authority != "UNQUALIFIED_FLEXICUBES_SURFACE_WITNESS":
        raise ValueError("RSS_VNEXT_FC_ANIM_SURFACE_AUTHORITY")
    if surface.source_policy != "SIGNED_HYPOTHESIS_ONLY":
        raise ValueError("RSS_VNEXT_FC_ANIM_SIGNED_SOURCE_POLICY")
    teacher.validate(device=surface.xyz.device, dtype=surface.xyz.dtype)
    weights = differentiable_source_skin_transfer(
        surface.xyz, teacher.rest_vertices, teacher.rest_skin_weights,
        k=config.knn, max_reference_vertices=config.max_reference_vertices,
    )
    matrices = teacher.posed_skin_matrices[None]
    posed = posed_lbs_from_joint_matrices(surface.xyz, weights, matrices)

    rest_surface = _area_weighted_surface_witnesses(
        surface.xyz, surface.faces, face_budget=config.max_sampled_faces,
    )
    rest_error = _two_way_surface_chamfer(
        rest_surface, teacher.rest_surface_samples,
    )
    pose_terms = []
    for p in range(matrices.shape[1]):
        if not bool(teacher.pose_valid_mask[p]):
            continue
        points = _area_weighted_surface_witnesses(
            posed[:, p], surface.faces, face_budget=config.max_sampled_faces,
        )
        pose_terms.append(_two_way_surface_chamfer(
            points, teacher.pose_surface_samples[p],
        ))
    if not pose_terms:
        raise ValueError("RSS_VNEXT_FC_ANIM_NO_VALID_POSE_LOSS")
    posed_error = torch.stack(pose_terms).mean()
    total = config.rest_weight * rest_error + config.pose_weight * posed_error
    if not bool(torch.isfinite(total)):
        raise ValueError("RSS_VNEXT_FC_ANIM_TOTAL_NONFINITE")
    return RSSvNextFCAnimationLossResult(
        total=total,
        rest_surface=rest_error,
        posed_surface=posed_error,
        transferred_skin=weights,
        posed_vertices=posed,
        pose_count=len(pose_terms),
    )


__all__ = [
    "RSSvNextFCAnimationTeacher",
    "RSSvNextFCAnimationLossConfig",
    "RSSvNextFCAnimationLossResult",
    "differentiable_source_skin_transfer",
    "source_animation_flexicubes_loss",
]
