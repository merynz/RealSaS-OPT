from __future__ import annotations

"""Research-only exact InstantNeRF XY/XZ/ZY -> IRIS XY/XZ/YZ address court.

The published InstantNeRF triplane convention is XY, XZ, ZY with
padding_mode='zeros'. RSS.vNext IRIS is XY, XZ, YZ with
padding_mode='border'. Their third planes can be reoriented via spatial
transpose; that does NOT change 80 channels into 96, align feature
semantics, match out-of-bounds behavior or authorize pretrained transfer.

The court only establishes *mathematical* interpolation equivalence within
the no-padding coordinate interior. Original checkpoint load, color/semantic
equivalence and license compatibility are separate, still-open gates.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from models.iris.vnext.scene_core_v1 import LocalEvidenceFieldV1
from .stdgen_instantnerf_native_v1 import (
    sample_stdgen_instantnerf_triplanes_v1,
)


@dataclass(frozen=True)
class RSSvNextInstantNeRFPlaneAddressCourtV1:
    source_channels: int
    rss_target_channels: int
    query_count: int
    max_absolute_interpolation_difference: float
    third_plane_transform: str = "TRANSPOSE_YZ_FROM_PRETRAIN_ZY"
    source_padding_mode: str = "zeros"
    rss_padding_mode: str = "border"
    addressed_interior_equivalence_qualified: bool = True
    full_volume_interpolation_equivalence_qualified: bool = False
    width_feature_transfer_qualified: bool = False
    learned_appearance_equivalence_qualified: bool = False
    pretrained_checkpoint_loaded: bool = False
    authority: str = "MATH_INTERIOR_ADDRESS_ONLY_NOT_CHECKPOINT_OR_PRODUCT"


def reorient_instantnerf_zy_to_iris_yz_v1(native_planes: Tensor) -> Tensor:
    """Pure spatial permutation: no trainable projection or weight mutation."""
    if (
        native_planes.ndim != 5 or native_planes.shape[1] != 3
        or native_planes.shape[0] < 1
        or native_planes.shape[2] < 1
        or native_planes.shape[-1] < 2
        or native_planes.shape[-1] != native_planes.shape[-2]
        or not native_planes.is_floating_point()
        or not bool(torch.isfinite(native_planes).all())
    ):
        raise ValueError("RSS_VNEXT_PRETRAIN_PLANE_REORIENTATION_DOMAIN")
    return torch.stack((
        native_planes[:,0],
        native_planes[:,1],
        native_planes[:,2].transpose(-2,-1),
    ),dim=1).contiguous()


def check_instantnerf_iris_address_equivalence_v1(
    *, native_planes: Tensor, xyz: Tensor,
    rss_target_channels: int,
    tolerance: float = 2.e-5,
) -> RSSvNextInstantNeRFPlaneAddressCourtV1:
    """Refuse boundary pixels, where zero vs border padding differ.

    Source channels may differ from RSS; this court never resolves that.
    The input is the *unadapted* native triplane proposal.
    """
    converted = reorient_instantnerf_zy_to_iris_yz_v1(native_planes)
    if (
        xyz.ndim != 3 or xyz.shape[0] != native_planes.shape[0]
        or xyz.shape[-1] != 3
        or xyz.shape[1] < 1 or xyz.shape[1] > 65536
        or xyz.device != native_planes.device or xyz.dtype != native_planes.dtype
        or not bool(torch.isfinite(xyz).all())
        or type(rss_target_channels) is not int
        or rss_target_channels < 1
        or not 0. < tolerance <= 1.e-3
    ):
        raise ValueError("RSS_VNEXT_PRETRAIN_PLANE_ADDRESS_COURT_DOMAIN")
    side = native_planes.shape[-1]
    # align_corners=False means coordinates +-1 interpolate from outside
    # the first/last pixel; only the centre-supported interior is equivalent.
    interior = 1. - 1./side
    if bool((xyz.abs() > interior + 1.e-7).any()):
        raise ValueError("RSS_VNEXT_PRETRAIN_PLANE_PADDING_POLICY_NOT_EQUIVALENT")
    native = sample_stdgen_instantnerf_triplanes_v1(native_planes,xyz)
    rss = LocalEvidenceFieldV1.sample_triplanes(converted,xyz)
    if native.shape != rss.shape:
        raise RuntimeError("RSS_VNEXT_PRETRAIN_PLANE_SAMPLE_SHAPE_DRIFT")
    discrepancy = float((native-rss).abs().amax().detach().cpu())
    if discrepancy > tolerance:
        raise RuntimeError("RSS_VNEXT_PRETRAIN_PLANE_INTERIOR_ADDRESS_MISMATCH")
    return RSSvNextInstantNeRFPlaneAddressCourtV1(
        source_channels=native_planes.shape[2],
        rss_target_channels=rss_target_channels,
        query_count=xyz.shape[1],
        max_absolute_interpolation_difference=discrepancy,
    )


__all__=[
    "RSSvNextInstantNeRFPlaneAddressCourtV1",
    "reorient_instantnerf_zy_to_iris_yz_v1",
    "check_instantnerf_iris_address_equivalence_v1",
]
