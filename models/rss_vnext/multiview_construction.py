from __future__ import annotations

"""RSS.vNext canonical FRONT/BACK → camera-grounded *predicted* multiview.

Reference genealogy:
- RealSaS's RGB/normal *diffusion* generator predicts views BEFORE S-LRM.
- This RSS path instead renders shared learned Z/H with a differentiable
  camera-conditioned volume head AFTER the two-source-view scene encoder.
  It is a trainable, testable alternative; it does NOT claim to reproduce
  RealSaS's high-resolution diffusion generator or image fidelity.

Source authority is immutable: novel predictions NEVER enter the observed
view bank as real observations. Source rig/mesh/extra RGB/normal renders
can be supplied only as supervised targets, never input to this module.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RSSvNextCanonicalCameraRing:
    novel_cameras: Tensor     # [B,4 or 6,13], for 6/8 total including source
    novel_angles_rad: Tensor  # [4 or 6], canonical ring around up axis
    source_cameras: Tensor    # [B,2,13], exact original values
    source_authority: tuple[str, str] = ("SOURCE_FRONT", "SOURCE_BACK")
    novel_authority: str = "PREDICTED_CAMERA_TARGETS_ONLY"


def canonical_novel_camera_ring(
    source_cameras: Tensor,
    total_views: int,
    *,
    tolerance: float = 0.035,
) -> RSSvNextCanonicalCameraRing:
    """Construct 6 or 8 TOTAL viewpoints, including the two source views.

    Camera contract: origin xyz, right xyz, up xyz, forward xyz, ortho
    half-extent. Source must be an admitted antipodal *calibrated* pair.
    Invalid camera estimates fail closed; NEVER silently repair camera poses.
    Front=0°, Back=180°. Target indices exclude BOTH source angles.
    """
    if total_views not in (6, 8):
        raise ValueError("RSS_VNEXT_MV_REQUIRE_6_OR_8_TOTAL")
    if (source_cameras.ndim != 3 or source_cameras.shape[1:] != (2, 13)
            or not torch.is_floating_point(source_cameras)
            or not bool(torch.isfinite(source_cameras).all())):
        raise ValueError("RSS_VNEXT_MV_SOURCE_CAMERA_DOMAIN")
    front, back = source_cameras.unbind(dim=1)
    eye = torch.eye(3, device=source_cameras.device, dtype=source_cameras.dtype)
    for cam in (front, back):
        axes = torch.stack(
            (cam[:, 3:6], cam[:, 6:9], cam[:, 9:12]), dim=1,
        )
        gram = axes @ axes.transpose(1, 2)
        if bool((gram - eye).abs().max() > tolerance):
            raise ValueError("RSS_VNEXT_MV_CAMERA_NOT_ORTHONORMAL")
        if bool((cam[:, 12] <= 0).any()):
            raise ValueError("RSS_VNEXT_MV_ORTHO_EXTENT_INVALID")
    middle = (front[:, :3] + back[:, :3]) / 2
    radial = front[:, :3] - middle
    radius = radial.norm(dim=-1, keepdim=True)
    if bool((radius < .2).any()):
        raise ValueError("RSS_VNEXT_MV_CAMERA_RING_RADIUS_UNSTABLE")
    radial_unit = radial / radius
    right, up, forward = front[:, 3:6], front[:, 6:9], front[:, 9:12]
    if (bool((back[:, 3:6] + right).abs().max() > tolerance)
            or bool((back[:, 6:9] - up).abs().max() > tolerance)
            or bool((back[:, 9:12] + forward).abs().max() > tolerance)
            or bool((back[:, 12] - front[:, 12]).abs().max() > tolerance)
            or bool((forward + radial_unit).abs().max() > tolerance)
            or bool((right * radial_unit).sum(-1).abs().max() > tolerance)
            or bool((up * radial_unit).sum(-1).abs().max() > tolerance)):
        raise ValueError("RSS_VNEXT_MV_FRONT_BACK_NOT_ANTIPODAL")
    total = total_views
    novel_count = total_views - 2
    indices = [i for i in range(1, total) if i != total // 2]
    angles = torch.as_tensor(
        [2 * math.pi * i / total for i in indices],
        device=source_cameras.device, dtype=source_cameras.dtype,
    )
    cs, sn = torch.cos(angles)[None, :, None], torch.sin(angles)[None, :, None]
    origin = middle[:, None] + radius[:, None] * (
        radial_unit[:, None] * cs + right[:, None] * sn
    )
    target_right = right[:, None] * cs - radial_unit[:, None] * sn
    target_forward = forward[:, None] * cs - right[:, None] * sn
    target_up = up[:, None].expand(-1, novel_count, -1)
    extent = front[:, 12:13, None].expand(-1, novel_count, -1)
    target = torch.cat(
        (origin, target_right, target_up, target_forward, extent), dim=-1,
    )
    return RSSvNextCanonicalCameraRing(
        novel_cameras=target,
        novel_angles_rad=angles,
        source_cameras=source_cameras,
    )


@dataclass
class RSSvNextConstructedViews:
    camera_features: Tensor         # [B,V,13] where V=4/6 novel (total 6/8)
    rgba: Tensor                    # [B,V,4,H,W], RGB straight, alpha composited
    normal: Tensor                  # [B,V,3,H,W], predicted vector, not source normals
    depth: Tensor                   # [B,V,1,H,W], relative expected ray depth
    log_uncertainty: Tensor         # [B,V,1,H,W]
    novel_angles_rad: Tensor        # [V]
    source_view_count: int = 2
    authority: str = "PREDICTED_FROM_TWO_VIEWS_NO_NEW_SOURCE_EVIDENCE"

    def validate(self) -> None:
        b, v, c, h, w = self.rgba.shape
        if (v not in (4, 6) or c != 4
                or self.camera_features.shape != (b, v, 13)
                or self.normal.shape != (b, v, 3, h, w)
                or self.depth.shape != (b, v, 1, h, w)
                or self.log_uncertainty.shape != (b, v, 1, h, w)
                or self.novel_angles_rad.shape != (v,)):
            raise ValueError("RSS_VNEXT_MV_PREDICTED_VIEW_SHAPE")
        if not all(bool(torch.isfinite(x).all()) for x in (
            self.rgba, self.normal, self.depth, self.log_uncertainty,
        )):
            raise ValueError("RSS_VNEXT_MV_PREDICTED_VIEW_NONFINITE")
        if bool((self.rgba < -1e-6).any()) or bool((self.rgba > 1.000001).any()):
            raise ValueError("RSS_VNEXT_MV_RGBA_RANGE")
        if self.authority != "PREDICTED_FROM_TWO_VIEWS_NO_NEW_SOURCE_EVIDENCE":
            raise ValueError("RSS_VNEXT_MV_SOURCE_AUTHORITY_ESCALATION")


class RSSvNextViewFieldRenderer(nn.Module):
    """Experimental POST-scene novel-view RGB/alpha/normal from H(x).

    Not the published RealSaS pre-scene multi-view diffusion mechanism.

    View-conditioned colors, learned density and normals; depth order is
    analytically calculated through alpha compositing. It is a hypothesis
    renderer, not a ray-hit depth authority, not signed-solid proof and not
    a promise of physically true normals at unobserved viewpoints.

    O(B*V*H*W*S) time, O(B*V*chunk*S) live activations.
    """

    def __init__(
        self,
        carrier_dim: int,
        *,
        hidden_dim: int = 256,
        ray_samples: int = 48,
        ray_chunk: int = 256,
        max_pixels: int = 4096,
    ):
        super().__init__()
        if min(carrier_dim, hidden_dim, ray_samples, ray_chunk, max_pixels) <= 0:
            raise ValueError("RSS_VNEXT_MV_RENDERER_DIMENSION")
        if ray_samples > 128 or ray_chunk > 4096:
            raise ValueError("RSS_VNEXT_MV_RENDERER_RESOURCE_LIMIT")
        self.ray_samples = ray_samples
        self.ray_chunk = ray_chunk
        self.max_pixels = max_pixels
        self.core = nn.Sequential(
            nn.LayerNorm(carrier_dim + 3),
            nn.Linear(carrier_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.density = nn.Linear(hidden_dim, 1)
        self.rgb = nn.Linear(hidden_dim, 3)
        self.normal = nn.Linear(hidden_dim, 3)
        self.uncertainty = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        *,
        iris,
        scene_triplanes: Tensor,
        ring: RSSvNextCanonicalCameraRing,
        height: int,
        width: int,
        graphics_field: nn.Module | None = None,
    ) -> RSSvNextConstructedViews:
        if height < 2 or width < 2 or height * width > self.max_pixels:
            raise ValueError("RSS_VNEXT_MV_RENDER_IMAGE_BUDGET")
        if (scene_triplanes.ndim != 5 or scene_triplanes.shape[1] != 3
                or scene_triplanes.shape[0] != ring.novel_cameras.shape[0]):
            raise ValueError("RSS_VNEXT_MV_RENDER_SHARED_CARRIER_DOMAIN")
        b, v, _ = ring.novel_cameras.shape
        camera = ring.novel_cameras
        dtype, device = scene_triplanes.dtype, scene_triplanes.device
        if camera.device != device or camera.dtype != dtype:
            raise ValueError("RSS_VNEXT_MV_RENDER_CAMERA_DEVICE_DTYPE")
        yy = 1 - (torch.arange(height, device=device, dtype=dtype) + .5) * (2 / height)
        xx = (torch.arange(width, device=device, dtype=dtype) + .5) * (2 / width) - 1
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        image_coords = torch.stack((gx, gy), dim=-1).reshape(-1, 2)
        outputs = []
        for sample_idx in range(b):
            per_view = []
            for vi in range(v):
                cam = camera[sample_idx, vi]
                origin, right, up, forward = cam[:3], cam[3:6], cam[6:9], cam[9:12]
                extent = cam[12]
                center = ring.source_cameras[sample_idx, :, :3].mean(dim=0)
                radius = (cam[:3] - center).norm()
                near = (radius - math.sqrt(3)).clamp_min(.01)
                far = radius + math.sqrt(3)
                step = (far - near) / self.ray_samples
                depths = near + step * (
                    torch.arange(self.ray_samples, device=device, dtype=dtype) + .5
                )
                patches = []
                for start in range(0, height * width, self.ray_chunk):
                    coords = image_coords[start : start + self.ray_chunk]
                    xy = coords[:, :1] * right[None] + coords[:, 1:] * up[None]
                    ro = origin[None] + extent * xy
                    points = ro[:, None] + depths[None, :, None] * forward
                    # Never extend the inferred field with border replication.
                    inside = (points.abs() <= 1.0).all(dim=-1)
                    local = iris.query(
                        scene_triplanes[sample_idx:sample_idx+1],
                        points.reshape(1, -1, 3),
                    )["H"][0]
                    n, s = points.shape[:2]
                    # The five typed Graphics field channels are shared with
                    # TESSA and the same source-bound H(x). A view-dependent
                    # residual may add to them but cannot become observed art.
                    graphics = (
                        torch.zeros((n * s, 5), dtype=dtype, device=device)
                        if graphics_field is None else graphics_field(local)
                    )
                    if graphics.shape != (n * s, 5) or not bool(torch.isfinite(graphics).all()):
                        raise ValueError("RSS_VNEXT_MV_GRAPHICS_FIELD_INVALID")
                    view_dir = forward.expand(n * s, -1)
                    latent = self.core(torch.cat((local, view_dir), dim=-1))
                    sigma = F.softplus(
                        self.density(latent).reshape(n, s)
                        + graphics[:, 3].reshape(n, s)
                    )
                    sigma = sigma * inside.to(sigma.dtype)
                    opacity = -torch.expm1(-sigma * step)
                    survive = torch.cat((
                        torch.ones((n, 1), dtype=dtype, device=device),
                        (1 - opacity + 1e-9).cumprod(-1)[:, :-1],
                    ), dim=-1)
                    weights = opacity * survive
                    alpha = weights.sum(dim=-1, keepdim=True).clamp(0, 1)
                    rgb_samples = torch.sigmoid(
                        self.rgb(latent) + graphics[:, :3]
                    ).reshape(n, s, 3)
                    rgb_premult = (rgb_samples * weights[..., None]).sum(dim=1)
                    rgb_straight = torch.where(
                        alpha > 1e-6,
                        rgb_premult / alpha.clamp_min(1e-6),
                        torch.zeros_like(rgb_premult),
                    )
                    normals = F.normalize(
                        self.normal(latent).reshape(n, s, 3), dim=-1, eps=1e-7,
                    )
                    mean_normal = (normals * weights[..., None]).sum(dim=1)
                    mean_normal = torch.where(
                        alpha > 1e-6,
                        F.normalize(mean_normal, dim=-1, eps=1e-7),
                        torch.zeros_like(mean_normal),
                    )
                    relative_depth = ((depths - near) / (far - near)).expand(n, -1)
                    expected_depth = (weights * relative_depth).sum(-1, keepdim=True)
                    logvar = (
                        (self.uncertainty(latent).reshape(n, s)
                         + graphics[:, 4].reshape(n, s)) * weights
                    ).sum(dim=-1, keepdim=True) / alpha.clamp_min(1e-6)
                    logvar = logvar.clamp(-8, 4)
                    patches.append(torch.cat((
                        rgb_straight, alpha, mean_normal,
                        expected_depth, logvar,
                    ), dim=-1))
                per_view.append(torch.cat(patches, dim=0).reshape(height, width, 9))
            outputs.append(torch.stack(per_view))
        values = torch.stack(outputs).permute(0, 1, 4, 2, 3).contiguous()
        result = RSSvNextConstructedViews(
            camera_features=camera,
            rgba=values[:, :, :4],
            normal=values[:, :, 4:7],
            depth=values[:, :, 7:8],
            log_uncertainty=values[:, :, 8:9],
            novel_angles_rad=ring.novel_angles_rad,
        )
        result.validate()
        return result


def qualified_novel_view_loss(
    predicted: RSSvNextConstructedViews,
    source_render_rgba: Tensor,
    source_render_normal: Tensor,
    *,
    rgb_qualified_mask: Tensor,
    normal_qualified_mask: Tensor,
    source_render_provenance: str,
) -> dict[str, Tensor]:
    """Only teacher-side admitted ORIGINAL SOURCE RENDERS, never generated views.

    Targets/masks have no path to the inference encoder. RGB loss uses
    premultiplied RGBA and only qualified source-render pixels. Normal loss
    excludes transparent/unqualified pixels. Unobserved or generated pixels
    are NOT taught as negative examples.
    """
    predicted.validate()
    if not source_render_provenance.startswith("QUALIFIED_SOURCE_RENDER:"):
        raise ValueError("RSS_VNEXT_MV_NOVEL_TRUTH_PROVENANCE_REQUIRED")
    if source_render_rgba.shape != predicted.rgba.shape or source_render_normal.shape != predicted.normal.shape:
        raise ValueError("RSS_VNEXT_MV_TEACHER_SHAPE")
    expected = predicted.rgba.shape[:2] + predicted.rgba.shape[-2:]
    if (rgb_qualified_mask.shape != expected or normal_qualified_mask.shape != expected
            or rgb_qualified_mask.dtype != torch.bool
            or normal_qualified_mask.dtype != torch.bool):
        raise ValueError("RSS_VNEXT_MV_TEACHER_MASK_SHAPE")
    if not bool(rgb_qualified_mask.any()) or not bool(normal_qualified_mask.any()):
        raise ValueError("RSS_VNEXT_MV_NO_QUALIFIED_TARGET")
    if not all(bool(torch.isfinite(x).all()) for x in (source_render_rgba, source_render_normal)):
        raise ValueError("RSS_VNEXT_MV_TEACHER_NONFINITE")
    if bool((source_render_rgba < 0).any()) or bool((source_render_rgba > 1).any()):
        raise ValueError("RSS_VNEXT_MV_TEACHER_RGBA_RANGE")
    p = predicted.rgba
    g = source_render_rgba
    p_pre = torch.cat((p[:, :, :3] * p[:, :, 3:4], p[:, :, 3:4]), dim=2)
    g_pre = torch.cat((g[:, :, :3] * g[:, :, 3:4], g[:, :, 3:4]), dim=2)
    rgb_per = (p_pre - g_pre).abs().mean(dim=2)
    rgb_loss = rgb_per[rgb_qualified_mask].mean()
    mask = normal_qualified_mask & (g[:, :, 3] > 0.01)
    if not bool(mask.any()):
        raise ValueError("RSS_VNEXT_MV_NO_VISIBLE_NORMAL_TRUTH")
    cosine = 1 - (
        F.normalize(predicted.normal, dim=2, eps=1e-7)
        * F.normalize(source_render_normal, dim=2, eps=1e-7)
    ).sum(dim=2)
    normal_loss = cosine[mask].mean()
    return {
        "rgba_premultiplied": rgb_loss,
        "visible_normal_cosine": normal_loss,
        "total": rgb_loss + .25 * normal_loss,
    }


__all__ = [
    "RSSvNextCanonicalCameraRing", "canonical_novel_camera_ring",
    "RSSvNextConstructedViews", "RSSvNextViewFieldRenderer",
    "qualified_novel_view_loss",
]
