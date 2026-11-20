from __future__ import annotations

"""Exact RSS.vNext 13D -> native MV-Adapter orthographic 6-channel control.

The official MV-Adapter geometry helper names this a 'Plucker embedding', but
its ORTHOGRAPHIC implementation actually emits spatially constant:
    [view_dir(3), normalized_camera_position(3)].
No line moments, no per-pixel ray origins, and no hidden Z-up Blender transform.

RSS.vNext's 13D virtual camera contract already stores the world-space
inward direction in [9:12] and the camera position in [0:3]. Deriving the
six controls directly from THESE authoritative coordinates eliminates the
pretrained demo's unrelated Z-up azimuth convention. It keeps the exact
published conditioner feature ordering and (-1,+1)->(0,1) scaling.
"""

import torch
from torch import Tensor
import torch.nn.functional as F

from .realsas_virtual_camera_policy_v1 import (
    target_canonical_and_six_cameras_v1,
)

SCHEMA="RealSaS.RSSvNext.MVAdapterNativeOrtho13DCameraBridge.v2"


def mv_adapter_controls_from_13d_virtual_cameras_v2(
    cameras:Tensor,*,height:int,width:int,
    require_fixed_six_bank:bool=True,
)->Tensor:
    if (
        not isinstance(cameras,Tensor)
        or cameras.ndim!=3 or cameras.shape[1:]!=(6,13)
        or cameras.shape[0]<1 or cameras.shape[0]>8
        or not torch.is_floating_point(cameras)
        or not bool(torch.isfinite(cameras).all())
        or type(height) is not int or type(width) is not int
        or height!=width or height%8 or height<64
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_CAMERA_BRIDGE_ABI")
    if require_fixed_six_bank:
        _,expected=target_canonical_and_six_cameras_v1(
            cameras.shape[0],device=str(cameras.device),dtype=cameras.dtype,
        )
        if not torch.allclose(cameras,expected,atol=1e-6,rtol=0):
            raise ValueError("RSS_VNEXT_MVADAPTER_CAMERA_BANK_NOT_V1")
    position=cameras[:,:,:3]
    inward=cameras[:,:,9:12]
    if (
        bool((position.norm(dim=-1)<1e-6).any())
        or bool((inward.norm(dim=-1)<1e-6).any())
        or not torch.allclose(
            inward.norm(dim=-1),torch.ones_like(inward[...,0]),
            atol=1e-5,rtol=0,
        )
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_CAMERA_DIRECTION_INVALID")
    # Native MV-Adapter's get_plucker_embeds_from_cameras_ortho:
    # view_dir = camera optical forward, normalized cam_pos = radial.
    # For the v1 RSS virtual camera this is exactly the 13D forward
    # and position fields, even though Blender demo cameras are Z-up.
    native=torch.cat(
        (inward,F.normalize(position,dim=-1)),dim=-1,
    )
    controls=((native+1.)*.5).clamp(0,1)
    return controls[:,:,:,None,None].expand(
        cameras.shape[0],6,6,height,width,
    ).contiguous()


def fixed_sdxl_mvadapter_two_plus_six_cameras_v2(
    batch:int,*,device:str,dtype:torch.dtype,image_size:int,
)->tuple[Tensor,Tensor,Tensor]:
    canonical,six=target_canonical_and_six_cameras_v1(
        batch,device=device,dtype=dtype,
    )
    control=mv_adapter_controls_from_13d_virtual_cameras_v2(
        six,height=image_size,width=image_size,
    )
    return canonical,six,control


__all__=[
    "SCHEMA","mv_adapter_controls_from_13d_virtual_cameras_v2",
    "fixed_sdxl_mvadapter_two_plus_six_cameras_v2",
]
