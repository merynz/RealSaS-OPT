from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from realsas_compiler_core.types import ObservationEvidenceIR, ObservationSample, PersistenceGroup


@dataclass(frozen=True)
class IrisDemoConfig:
    views: int = 8
    model_resolution: int = 256
    base_width: int = 32
    max_abs_depth: float = 0.75
    support_threshold: float = 0.5
    alpha_threshold: float = 0.02
    evidence_stride: int = 4
    persistence_voxel: float = 0.004
    persistence_min_views: int = 2


@dataclass
class IrisDemoRawOutput:
    depth: torch.Tensor
    support_logits: torch.Tensor
    log_sigma: torch.Tensor


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(min(16, cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(min(16, cout), cout)
        self.skip = nn.Conv2d(cin, cout, 1, stride, bias=False) if cin != cout or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(y + self.skip(x))


class IrisDemoV1(nn.Module):
    """Generic single-specimen demo learner for the current IRIS external contract.

    It predicts camera-forward depth, support and uncertainty. It deliberately does
    not predict common-frame P; P remains analytic from the fixed camera contract.
    """

    def __init__(self, cfg: IrisDemoConfig = IrisDemoConfig()):
        super().__init__()
        self.cfg = cfg
        b = cfg.base_width
        self.e1 = _ConvBlock(4, b, 1)
        self.e2 = _ConvBlock(b, 2 * b, 2)
        self.e3 = _ConvBlock(2 * b, 4 * b, 2)
        self.e4 = _ConvBlock(4 * b, 6 * b, 2)

        self.yaw_mlp = nn.Sequential(nn.Linear(2, 6 * b), nn.GELU(), nn.Linear(6 * b, 6 * b))
        view_layer = nn.TransformerEncoderLayer(
            d_model=6 * b,
            nhead=8,
            dim_feedforward=12 * b,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.view_reasoner = nn.TransformerEncoder(view_layer, num_layers=2)
        self.context_proj = nn.Linear(6 * b, 6 * b)

        self.d3 = _ConvBlock(6 * b + 4 * b, 4 * b)
        self.d2 = _ConvBlock(4 * b + 2 * b, 2 * b)
        self.d1 = _ConvBlock(2 * b + b, b)
        self.depth_head = nn.Conv2d(b, 1, 1)
        self.support_head = nn.Conv2d(b, 1, 1)
        self.sigma_head = nn.Conv2d(b, 1, 1)

    def forward(self, images: torch.Tensor, yaw_deg: torch.Tensor) -> IrisDemoRawOutput:
        if images.ndim != 5 or images.shape[1] != self.cfg.views or images.shape[2] != 4:
            raise ValueError(f"images must be [B,{self.cfg.views},4,H,W]")
        bsz, views = images.shape[:2]
        x = images.reshape(bsz * views, 4, images.shape[-2], images.shape[-1])
        x = F.interpolate(x, size=(self.cfg.model_resolution, self.cfg.model_resolution), mode="bilinear", align_corners=False)
        f1 = self.e1(x)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        f4 = self.e4(f3)

        pooled = f4.mean(dim=(-2, -1)).reshape(bsz, views, -1)
        if yaw_deg.ndim == 1:
            yaw_deg = yaw_deg.unsqueeze(0).expand(bsz, -1)
        if yaw_deg.shape != (bsz, views):
            raise ValueError("yaw_deg must be [V] or [B,V]")
        theta = torch.deg2rad(yaw_deg.float())
        yaw_feature = self.yaw_mlp(torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1))
        reasoned = self.view_reasoner(pooled + yaw_feature)
        global_context = self.context_proj(reasoned.mean(dim=1, keepdim=True)).expand(-1, views, -1)
        context = (reasoned + global_context).reshape(bsz * views, -1, 1, 1).expand_as(f4)
        z4 = f4 + context

        z3 = F.interpolate(z4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        z3 = self.d3(torch.cat([z3, f3], dim=1))
        z2 = F.interpolate(z3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        z2 = self.d2(torch.cat([z2, f2], dim=1))
        z1 = F.interpolate(z2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        z1 = self.d1(torch.cat([z1, f1], dim=1))

        depth = self.cfg.max_abs_depth * torch.tanh(self.depth_head(z1))
        support = self.support_head(z1)
        log_sigma = torch.clamp(self.sigma_head(z1), -8.0, 3.0)

        def restore(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(bsz, views, *t.shape[1:])

        return IrisDemoRawOutput(restore(depth), restore(support), restore(log_sigma))


def _camera_from_dict(camera: dict) -> dict:
    if camera.get("contract") != "realsas.level_orthographic_z_orbit.v1":
        raise ValueError("CAMERA_CONTRACT_DRIFT")
    forward = tuple(float(x) for x in camera["forward"])
    right = tuple(float(x) for x in camera["right"])
    screen_up = tuple(float(x) for x in camera["screen_up"])
    half_extent = float(camera["half_extent"])
    if half_extent <= 0:
        raise ValueError("CAMERA_HALF_EXTENT_NONPOSITIVE")
    for v in (forward, right, screen_up):
        if len(v) != 3 or not all(math.isfinite(x) for x in v):
            raise ValueError("CAMERA_VECTOR_INVALID")
    return {"forward": forward, "right": right, "screen_up": screen_up, "half_extent": half_extent}


def _analytic_point(grid_x: float, grid_y: float, depth: float, camera: dict) -> tuple[float, float, float]:
    c = _camera_from_dict(camera)
    x = grid_x * c["half_extent"]
    up = -grid_y * c["half_extent"]
    return tuple(
        x * c["right"][i] + up * c["screen_up"][i] + depth * c["forward"][i]
        for i in range(3)
    )


def build_observation_evidence_and_persistence(
    raw: IrisDemoRawOutput,
    input_images: torch.Tensor,
    cameras: list[dict],
    *,
    cfg: IrisDemoConfig,
    batch_index: int = 0,
) -> tuple[ObservationEvidenceIR, tuple[PersistenceGroup, ...]]:
    """Convert learned depth/support into current typed evidence plus deterministic persistence.

    Persistence is generic voxel agreement in analytic common-frame P. It consumes no
    source-rig identity, teacher geometry or specimen ID.
    """
    if len(cameras) != cfg.views:
        raise ValueError("camera count mismatch")
    if raw.depth.shape[1] != cfg.views:
        raise ValueError("raw view count mismatch")
    h, w = raw.depth.shape[-2:]
    alpha = input_images[batch_index, :, 3:4]
    alpha = F.interpolate(alpha, size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    support_prob = torch.sigmoid(raw.support_logits[batch_index, :, 0])
    depth = raw.depth[batch_index, :, 0]
    sigma = torch.exp(raw.log_sigma[batch_index, :, 0])

    samples: list[ObservationSample] = []
    voxel_members: dict[tuple[int, int, int], list[tuple[str, int]]] = defaultdict(list)
    voxel = float(cfg.persistence_voxel)
    if voxel <= 0:
        raise ValueError("persistence_voxel must be positive")

    for view in range(cfg.views):
        camera = _camera_from_dict(cameras[view])
        for py in range(0, h, cfg.evidence_stride):
            gy = ((py + 0.5) / h) * 2.0 - 1.0
            for px in range(0, w, cfg.evidence_stride):
                if float(alpha[view, py, px]) < cfg.alpha_threshold:
                    continue
                prob = float(support_prob[view, py, px])
                if prob < cfg.support_threshold:
                    continue
                d = float(depth[view, py, px])
                if not math.isfinite(d):
                    continue
                gx = ((px + 0.5) / w) * 2.0 - 1.0
                p = _analytic_point(gx, gy, d, cameras[view])
                origin = tuple(
                    p[i] - d * camera["forward"][i]
                    for i in range(3)
                )
                oid = f"IRISD:{view}:{py:04d}:{px:04d}"
                samples.append(
                    ObservationSample(
                        observation_id=oid,
                        view_index=view,
                        raster_xy=(float(px), float(py)),
                        ray_origin=origin,
                        ray_forward=camera["forward"],
                        depth=d,
                        support=True,
                        provenance_ref="RealSaS.IrisDemoV1.image_only",
                        validity_flags=(f"SUPPORT_PROB={prob:.6f}", f"SIGMA={float(sigma[view,py,px]):.6g}"),
                    )
                )
                key = tuple(int(round(coord / voxel)) for coord in p)
                voxel_members[key].append((oid, view))

    admitted_groups: list[PersistenceGroup] = []
    for key in sorted(voxel_members):
        members = voxel_members[key]
        by_view: dict[int, str] = {}
        for oid, view in members:
            by_view.setdefault(view, oid)
        if len(by_view) < cfg.persistence_min_views:
            continue
        ids = tuple(by_view[v] for v in sorted(by_view))
        admitted_groups.append(
            PersistenceGroup(
                group_id=f"DPG:{key[0]}:{key[1]}:{key[2]}",
                observation_ids=ids,
                method="DEMO_IMAGE_ONLY_VOXEL_AGREEMENT_V1",
                diagnostics={"voxel_size": voxel, "support_view_count": len(ids)},
            )
        )

    used = {oid for group in admitted_groups for oid in group.observation_ids}
    evidence = ObservationEvidenceIR(
        samples=tuple(sample for sample in samples if sample.observation_id in used),
        metadata={
            "producer": "RealSaS.IrisDemoV1.experimental",
            "generalization_claim": False,
            "P_authority": "ANALYTIC_NOT_LEARNED",
            "raw_supported_sample_count": len(samples),
            "persistence_group_count": len(admitted_groups),
        },
    )
    return evidence, tuple(admitted_groups)
