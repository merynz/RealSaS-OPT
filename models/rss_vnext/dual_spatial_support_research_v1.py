from __future__ import annotations

"""Non-product AniGen/Rigel3D-inspired spatial-support ablation over REAL IRIS H(x).

Research hypothesis: TESSA/AXIS may benefit from a second independently
addressed interior-support sequence. Keeping the original triplanes intact
isolates the *spatial support and cross-communication* decision from an
uncontrolled backbone replacement.

Both arms use the SAME two-view scene, K support tokens, direct IRIS field,
surface/interior cross-attention and memory dimensions. The control has
deterministic farthest-point samples from the existing unlabelled scene grid.
The alternative predicts bounded 3D query displacements from this grid's H;
never source skeleton/joint/mesh at forward. Zero-init displacement means
both arms are initially equivalent with the same weights. Extra trainable
coordinate predictor means matched downstream capacity, not matched active
parameter count: report this distinction in any future learning comparison.

Bone-centerline source coordinates are ONLY accepted by the separate,
receipt-matched train/evaluation loss. Nothing here is canonical geometry,
a skeleton generator, an admitted joint, product authority or a learned PASS.
"""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.iris.vnext.scene_core_v1 import IRISVNextSceneCoreV1


_MODES = ("FIXED_VOLUME_SUPPORT", "LEARNED_INTERIOR_SUPPORT")


@dataclass(frozen=True)
class RSSvNextDualSpatialSupportCandidateV1:
    support_xyz: Tensor               # [B,K,3], student only, normalised
    interior_tokens: Tensor            # [B,K,D], same shared IRIS H field
    surface_tokens: Tensor             # [B,N,D], original grid addresses
    base_query_indices: Tensor         # [B,K], exact source-grid addresses
    mode: str
    authority: str = "RESEARCH_ONLY_PREDICTED_SPATIAL_SUPPORT_NOT_SKELETON_TRUTH"


@dataclass(frozen=True)
class RSSvNextNativeBoneSupportObjectiveV1:
    loss: Tensor
    bone_to_proposal_mean_distance: Tensor
    proposal_to_bone_mean_distance: Tensor
    eligible_bone_samples: int
    source_asset_sha256: str
    receipt_key: str
    authority: str = "SOURCE_QUALIFIED_TRAINING_ONLY_NOT_INFERENCE_INPUT"


def _fps_indices(xyz: Tensor, k: int) -> Tensor:
    """Deterministic geometry-only seeds; never consult privileged joints.

    The first seed is closest to origin; after that maximize Euclidean
    distance to the nearest chosen seed with stable PyTorch argmax ties.
    """
    with torch.no_grad():
        rows = []
        for sample in xyz.detach().float():
            first = int(sample.square().sum(dim=-1).argmin().item())
            selected = [first]
            nearest = (sample - sample[first]).square().sum(dim=-1)
            for _ in range(1, k):
                nearest[selected] = -1.
                index = int(nearest.argmax().item())
                selected.append(index)
                candidate = (sample - sample[index]).square().sum(dim=-1)
                nearest = torch.minimum(nearest, candidate)
            rows.append(torch.tensor(selected, dtype=torch.long, device=xyz.device))
        return torch.stack(rows, dim=0)


class RSSvNextDualSpatialSupportResearchV1(nn.Module):
    """Same-field dual spatial memory; optional extra learned interior support."""

    def __init__(
        self, field_width: int, *, support_count: int = 16,
        heads: int = 4, displacement_scale: float = 0.35,
    ) -> None:
        super().__init__()
        if (field_width < 4 or support_count < 2 or support_count > 128
            or heads < 1 or field_width % heads or
            not 0. < displacement_scale <= .75):
            raise ValueError("RSS_VNEXT_INTERIOR_SUPPORT_ABLATION_CONFIG_INVALID")
        self.field_width = field_width
        self.support_count = support_count
        self.displacement_scale = displacement_scale

        self.support_queries = nn.Parameter(
            torch.randn(1, support_count, field_width) / field_width**.5
        )
        self.support_context = nn.MultiheadAttention(
            field_width, heads, dropout=0., batch_first=True,
        )
        self.support_query_norm = nn.LayerNorm(field_width)
        self.position_head = nn.Sequential(
            nn.LayerNorm(field_width),
            nn.Linear(field_width, field_width),
            nn.GELU(),
            nn.Linear(field_width, 3),
        )
        # Initially identical to the control's locations. No arbitrary
        # pretrained skeleton position can leak through initialization.
        nn.init.zeros_(self.position_head[-1].weight)
        nn.init.zeros_(self.position_head[-1].bias)

        self.position_embedding = nn.Linear(3, field_width, bias=False)
        self.interior_from_surface = nn.MultiheadAttention(
            field_width, heads, dropout=0., batch_first=True,
        )
        self.surface_from_interior = nn.MultiheadAttention(
            field_width, heads, dropout=0., batch_first=True,
        )
        self.interior_norm = nn.LayerNorm(field_width)
        self.surface_norm = nn.LayerNorm(field_width)

    def forward(
        self, *, iris: IRISVNextSceneCoreV1, scene_planes: Tensor,
        grid_xyz: Tensor, grid_h: Tensor, mode: str,
    ) -> RSSvNextDualSpatialSupportCandidateV1:
        if mode not in _MODES:
            raise ValueError("RSS_VNEXT_INTERIOR_SUPPORT_ABLATION_MODE_INVALID")
        if not isinstance(iris, IRISVNextSceneCoreV1):
            raise ValueError("RSS_VNEXT_INTERIOR_SUPPORT_IRIS_CORE_REQUIRED")
        if grid_xyz.ndim != 3 or grid_xyz.shape[-1] != 3:
            raise ValueError("RSS_VNEXT_INTERIOR_SUPPORT_QUERY_DOMAIN")
        b,n,_ = grid_xyz.shape
        if (
            b < 1 or n < self.support_count or n > 32768
            or grid_h.shape != (b,n,self.field_width)
            or scene_planes.ndim != 5
            or scene_planes.shape[:2] != (b,3)
            or scene_planes.shape[2] != iris.config.plane_channels
            or iris.config.local_feature_dim != self.field_width
            or len({x.device for x in (grid_xyz,grid_h,scene_planes)}) != 1
            or len({x.dtype for x in (grid_xyz,grid_h,scene_planes)}) != 1
            or not all(torch.is_floating_point(x) and bool(torch.isfinite(x).all())
                       for x in (grid_xyz,grid_h,scene_planes))
            or bool((grid_xyz.abs()>1.000001).any())
        ):
            raise ValueError("RSS_VNEXT_INTERIOR_SUPPORT_CARRIER_DOMAIN")
        chosen = _fps_indices(grid_xyz, self.support_count)
        anchors = grid_xyz.gather(1, chosen[...,None].expand(-1,-1,3))
        if mode == "FIXED_VOLUME_SUPPORT":
            coords = anchors
        else:
            queries = self.support_queries.expand(b,-1,-1)
            context,_ = self.support_context(
                queries,grid_h,grid_h,need_weights=False,
            )
            context = self.support_query_norm(queries+context)
            displacement = self.displacement_scale * torch.tanh(
                self.position_head(context)
            )
            coords = (anchors+displacement).clamp(-1.,1.)

        # Critically query the ORIGINAL IRIS field at NEW coordinates.
        # No detached second scene or proprietary sparse-voxel generator.
        bone_h = iris.query(scene_planes, coords)["H"]
        bone_query = bone_h+self.position_embedding(coords)
        grid_query = grid_h+self.position_embedding(grid_xyz)
        interior_update,_ = self.interior_from_surface(
            bone_query,grid_query,grid_query,need_weights=False,
        )
        bone_out = self.interior_norm(bone_query+interior_update)
        surface_update,_ = self.surface_from_interior(
            grid_query,bone_out,bone_out,need_weights=False,
        )
        surface_out = self.surface_norm(grid_query+surface_update)
        return RSSvNextDualSpatialSupportCandidateV1(
            support_xyz=coords,
            interior_tokens=bone_out,
            surface_tokens=surface_out,
            base_query_indices=chosen,
            mode=mode,
        )


def source_qualified_bone_support_loss_v1(
    candidate: RSSvNextDualSpatialSupportCandidateV1,
    *, source_bone_xyz: Tensor, source_bone_valid: Tensor,
    source_asset_sha256: str, source_admission_key: str,
    dataset_manifest_sha256: str,
    trusted_source_bone_registry: Mapping[str, Mapping[str, object]],
) -> RSSvNextNativeBoneSupportObjectiveV1:
    """Source-native bone path ONLY as a masked, receipt-bound training label.

    The registry MUST be independently assembled by the data-qualification
    launcher. A hash supplied by a model, adapter or synthesized view alone
    cannot create source-truth status. Pairwise Chamfer supervises support
    placement, not correct joint count, topology, rig or deformable skin.
    """
    from .training_engine import source_surface_tensor_sha256

    def sha(x: object) -> bool:
        return type(x) is str and len(x)==64 and all(
            c in "0123456789abcdef" for c in x
        )
    if (
        type(candidate) is not RSSvNextDualSpatialSupportCandidateV1
        or candidate.mode not in _MODES
        or not sha(source_asset_sha256)
        or not sha(dataset_manifest_sha256)
        or type(source_admission_key) is not str
        or not source_admission_key
        or source_admission_key not in trusted_source_bone_registry
    ):
        raise ValueError("RSS_VNEXT_INTERIOR_SUPPORT_SOURCE_ADMISSION_REQUIRED")
    b,k,_ = candidate.support_xyz.shape
    if (
        source_bone_xyz.ndim != 3
        or source_bone_xyz.shape[0] != b
        or source_bone_xyz.shape[-1] != 3
        or source_bone_valid.shape != source_bone_xyz.shape[:2]
        or source_bone_valid.dtype != torch.bool
        or min(source_bone_xyz.shape[1],k) < 1
        or source_bone_xyz.device != candidate.support_xyz.device
        or source_bone_valid.device != candidate.support_xyz.device
        or source_bone_xyz.dtype != candidate.support_xyz.dtype
        or not bool(source_bone_valid.any(dim=1).all())
        or not bool(torch.isfinite(source_bone_xyz[source_bone_valid]).all())
        or bool((source_bone_xyz[source_bone_valid].abs()>1.000001).any())
    ):
        raise ValueError("RSS_VNEXT_INTERIOR_SUPPORT_SOURCE_BONE_DOMAIN")

    receipt = trusted_source_bone_registry[source_admission_key]
    if (
        receipt.get("admission_status") != "QUALIFIED_SOURCE_NATIVE_BONE_PATH"
        or receipt.get("source_asset_sha256") != source_asset_sha256
        or receipt.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or receipt.get("bone_xyz_sha256") != source_surface_tensor_sha256(source_bone_xyz)
        or receipt.get("bone_valid_sha256") != source_surface_tensor_sha256(source_bone_valid)
        or receipt.get("source_semantics") != "NATIVE_RIG_BONE_CENTERLINE_SAMPLES_NOT_INFERENCE"
    ):
        raise ValueError("RSS_VNEXT_INTERIOR_SUPPORT_SOURCE_BONE_RECEIPT_DRIFT")

    # Masked NaNs/unknown labels remain unavailable, never zero-bone truth.
    losses, forwards, backwards = [],[],[]
    for i in range(b):
        gt = source_bone_xyz[i,source_bone_valid[i]].detach().float()
        pred = candidate.support_xyz[i].float()
        d = torch.cdist(pred[None],gt[None])[0]
        s2b = d.amin(-1).mean()
        b2s = d.amin(-2).mean()
        losses.append(s2b+b2s)
        forwards.append(b2s)
        backwards.append(s2b)
    return RSSvNextNativeBoneSupportObjectiveV1(
        loss=torch.stack(losses).mean(),
        bone_to_proposal_mean_distance=torch.stack(forwards).mean(),
        proposal_to_bone_mean_distance=torch.stack(backwards).mean(),
        eligible_bone_samples=int(source_bone_valid.sum().item()),
        source_asset_sha256=source_asset_sha256,
        receipt_key=source_admission_key,
    )


__all__=[
    "RSSvNextDualSpatialSupportCandidateV1",
    "RSSvNextDualSpatialSupportResearchV1",
    "RSSvNextNativeBoneSupportObjectiveV1",
    "source_qualified_bone_support_loss_v1",
]
