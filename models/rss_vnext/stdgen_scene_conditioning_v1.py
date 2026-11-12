from __future__ import annotations

"""StdGEN-faithful 1/6-view neural scene input, separate from original rays.

The older product camera/SourceObservation ABI intentionally stays EXACTLY
two real calibrated FRONT/BACK images; it must never be populated from a
diffusion model. This module instead handles:
  - one *canonical reference* RGB/A image (StdGEN one-view baseline), or
  - six RGB/A images from a multi-view diffusion model (StdGEN S-LRM input).
The common IRIS 3D triplane and existing geometry/Graphics/semantic heads
are reused, not duplicated. An input produced by diffusion is a hypothesis.

This is not yet product capability for arbitrary real one-image uploads:
source pose/camera normalization, 3D quality and training remain separate.
"""

from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor
from .unified_character_model import RSSvNextUnifiedCharacterModel

CANONICAL_REFERENCE="STDGEN_SINGLE_CANONICAL_REFERENCE_PREDICTION"
SIX_VIEW_PRIOR="STDGEN_SIX_PREDICTED_RGB_VIEWS"
ONE_MEASURED_SOURCE="STDGEN_SINGLE_MEASURED_SOURCE_CAMERA"
_SIX_AZIMUTH=(-90,-45,0,45,90,180)


@dataclass(frozen=True)
class StdGENSceneConditioningV1:
    scene_triplanes: Tensor
    source_binding_sha256: str
    scene_conditioning_authority: str
    conditioning_view_count: int
    original_observed_view_count: int
    source_camera_calibration_minted: bool
    generated_views_became_source_observation: bool
    scene_geometry_is_source_truth: bool
    conditioning_images_are_3d_teacher_targets: bool
    generated_multiview_checkpoint_sha256: str | None
    semantic_image_tokens: Tensor


def _hash(value: str) -> bool:
    return type(value) is str and len(value)==64 and all(x in "0123456789abcdef" for x in value)


def encode_stdgen_conditioning_into_shared_H_v1(
    model: RSSvNextUnifiedCharacterModel, *,
    conditioning_rgba: Tensor, conditioning_cameras: Tensor,
    source_binding_sha256: str,
    original_observed_view_count: int,
    conditioning_authority: str,
    generating_checkpoint_sha256: str | None = None,
) -> StdGENSceneConditioningV1:
    """One canonical view or six aligned diffusion-generated views -> H(x).

    No field becomes original source observations and no generated camera
    receives calibrated-observation authority. The six-view format mirrors
    StdGEN infer_slrm.py's six RGB views; generated normals stay a separate
    refinement prior, not six additional "source views".
    """
    if (
        type(model) is not RSSvNextUnifiedCharacterModel
        or not _hash(source_binding_sha256)
        or type(original_observed_view_count) is not int
        or original_observed_view_count not in (1,2)
        or conditioning_rgba.ndim!=5
        or conditioning_rgba.shape[1] not in (1,6)
        or conditioning_rgba.shape[2]!=4
    ):
        raise ValueError("RSS_VNEXT_STDGEN_CONDITIONING_SOURCE_OR_VIEW_DOMAIN")
    b,v,_,h,w=conditioning_rgba.shape
    if (
        b<1 or b>8 or max(h,w)>2048 or min(h,w)<14
        or conditioning_cameras.shape!=(b,v,13)
        or conditioning_cameras.device!=conditioning_rgba.device
        or conditioning_cameras.dtype!=conditioning_rgba.dtype
        or not bool(torch.isfinite(conditioning_rgba).all())
        or not bool(torch.isfinite(conditioning_cameras).all())
        or bool((conditioning_rgba<0).any() or (conditioning_rgba>1).any())
        or bool((conditioning_cameras[:,:,12]<=0).any())
    ):
        raise ValueError("RSS_VNEXT_STDGEN_CONDITIONING_IMAGE_OR_CAMERA_DOMAIN")
    camera_axes=torch.stack((
        conditioning_cameras[:,:,3:6],
        conditioning_cameras[:,:,6:9],
        conditioning_cameras[:,:,9:12],
    ),dim=-2)
    identity=torch.eye(3,device=camera_axes.device,dtype=camera_axes.dtype)
    if (
        not bool(torch.allclose(
            camera_axes@camera_axes.transpose(-1,-2),identity,atol=1.e-3,rtol=0.
        ))
        or bool((torch.linalg.det(camera_axes)<.99).any())
    ):
        raise ValueError("RSS_VNEXT_STDGEN_CANONICAL_CAMERA_BASIS_INVALID")
    if v==6:
        if (
            conditioning_authority != SIX_VIEW_PRIOR
            or not _hash(generating_checkpoint_sha256)
        ):
            raise ValueError("RSS_VNEXT_STDGEN_SIX_VIEWS_MUST_BE_GENERATED_PRIOR")
        cameras=conditioning_cameras
        axes=torch.stack((cameras[:,:,3:6],cameras[:,:,6:9],cameras[:,:,9:12]),dim=-2)
        gram=axes@axes.transpose(-1,-2)
        eye=torch.eye(3,device=axes.device,dtype=axes.dtype)
        if (
            not bool(torch.allclose(gram,eye,atol=1.e-3,rtol=0))
            or bool((torch.linalg.det(axes)<.99).any())
            or not bool(torch.allclose(cameras[:,:,6:9],cameras[:,0:1,6:9].expand(-1,v,-1),atol=1.e-3,rtol=0))
            or not bool(torch.allclose(cameras[:,:,12],cameras[:,0:1,12].expand(-1,v),atol=1.e-3,rtol=0))
        ):
            raise ValueError("RSS_VNEXT_STDGEN_SIX_VIEW_ORTHOGRAPHIC_ALIGNMENT")
        # Reject six repeated cameras: equivalent to source duplication,
        # not a multi-view reconstruction input.
        for i in range(v):
            for j in range(i+1,v):
                if bool(torch.allclose(cameras[:,i,9:12],cameras[:,j,9:12],atol=.001,rtol=0)):
                    raise ValueError("RSS_VNEXT_STDGEN_DUPLICATE_PREDICTED_CAMERA")
    elif (
        conditioning_authority not in (CANONICAL_REFERENCE,ONE_MEASURED_SOURCE)
        or (conditioning_authority==ONE_MEASURED_SOURCE and (
            generating_checkpoint_sha256 is not None or original_observed_view_count!=1
        ))
        or (conditioning_authority==CANONICAL_REFERENCE and not _hash(generating_checkpoint_sha256))
    ):
        raise ValueError("RSS_VNEXT_STDGEN_ONE_VIEW_PROVENANCE_INVALID")

    if not hasattr(model.vision,"forward_variable_views"):
        raise ValueError("RSS_VNEXT_STDGEN_BACKBONE_HAS_NO_VARIABLE_VIEW_ADAPTER")
    # IMPORTANT: encode_observations() only accepts 2 ORIGINAL views.
    # This bypass does not change those original rays or their cameras.
    tokens,xy=model.vision.forward_variable_views(conditioning_rgba)
    early=model.early_semantics.forward_variable_views(tokens)
    planes=model.iris.encode_scene(
        early.conditioned_image_tokens,xy,conditioning_cameras,
    )
    if planes.ndim!=5 or planes.shape[:2]!=(b,3):
        raise ValueError("RSS_VNEXT_STDGEN_SHARED_SCENE_CARRIER_SHAPE")
    return StdGENSceneConditioningV1(
        scene_triplanes=planes,
        source_binding_sha256=source_binding_sha256,
        scene_conditioning_authority=conditioning_authority,
        conditioning_view_count=v,
        original_observed_view_count=original_observed_view_count,
        source_camera_calibration_minted=False,
        generated_views_became_source_observation=False,
        scene_geometry_is_source_truth=False,
        conditioning_images_are_3d_teacher_targets=False,
        generated_multiview_checkpoint_sha256=generating_checkpoint_sha256,
        semantic_image_tokens=early.conditioned_image_tokens,
    )


__all__=[
    "CANONICAL_REFERENCE","SIX_VIEW_PRIOR","ONE_MEASURED_SOURCE",
    "StdGENSceneConditioningV1","encode_stdgen_conditioning_into_shared_H_v1",
]
