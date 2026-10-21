from __future__ import annotations

"""Differentiable raster apparatus for SURF-A1.

PyTorch3D is imported lazily so CPU/source CI does not depend on a compiled
renderer. Camera projection is first-party and explicit; the A100 preflight must
prove hard coverage parity against the independently materialized pixel-center
teacher coverage before any optimizer step.
"""

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SurfA1RasterConfigV1:
    image_size: int = 256
    silhouette_sigma: float = 1.0e-4
    faces_per_pixel: int = 24

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("SURF_A1_RASTER_IMAGE_SIZE_INVALID")
        if self.silhouette_sigma <= 0.0:
            raise ValueError("SURF_A1_RASTER_SIGMA_INVALID")
        if self.faces_per_pixel <= 0:
            raise ValueError("SURF_A1_RASTER_FACES_PER_PIXEL_INVALID")


def _unit(value: torch.Tensor, code: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 1.0e-12:
        raise ValueError(code)
    return value / norm


def project_iris_to_pytorch3d_ndc_v1(
    vertices_iris: torch.Tensor,
    *,
    common_center_world: torch.Tensor,
    common_half_extent: torch.Tensor | float,
    camera_origin_world: torch.Tensor,
    camera_right_world: torch.Tensor,
    camera_screen_up_world: torch.Tensor,
    camera_forward_world: torch.Tensor,
    camera_half_extent: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map RealSaS orthographic camera semantics to PyTorch3D NDC."""
    if vertices_iris.ndim != 2 or vertices_iris.shape[-1] != 3:
        raise ValueError("SURF_A1_RASTER_VERTEX_SHAPE_INVALID")
    device, dtype = vertices_iris.device, vertices_iris.dtype
    center = torch.as_tensor(common_center_world, device=device, dtype=dtype).reshape(3)
    common_half = torch.as_tensor(common_half_extent, device=device, dtype=dtype).reshape(())
    camera_half = torch.as_tensor(camera_half_extent, device=device, dtype=dtype).reshape(())
    if float(common_half.detach().item()) <= 0.0 or float(camera_half.detach().item()) <= 0.0:
        raise ValueError("SURF_A1_RASTER_CAMERA_SCALE_INVALID")
    origin = torch.as_tensor(camera_origin_world, device=device, dtype=dtype).reshape(3)
    right = _unit(torch.as_tensor(camera_right_world, device=device, dtype=dtype).reshape(3), "SURF_A1_RASTER_RIGHT_INVALID")
    up = _unit(torch.as_tensor(camera_screen_up_world, device=device, dtype=dtype).reshape(3), "SURF_A1_RASTER_UP_INVALID")
    forward = _unit(torch.as_tensor(camera_forward_world, device=device, dtype=dtype).reshape(3), "SURF_A1_RASTER_FORWARD_INVALID")
    vertices_world = center[None] + common_half * vertices_iris
    rel = vertices_world - origin[None]
    # PyTorch3D NDC is +X left / +Y up; RealSaS screen is +X right / +Y down.
    x = -(rel @ right) / camera_half
    y = (rel @ up) / camera_half
    depth = (rel @ forward) / common_half
    return torch.stack((x, y, depth), dim=-1), vertices_world


def _pytorch3d_modules() -> tuple[Any, Any]:
    try:
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        from pytorch3d.structures import Meshes
    except Exception as exc:  # pragma: no cover - A100 runtime gate
        raise RuntimeError("SURF_A1_PYTORCH3D_IMPORT_FAILED") from exc
    return rasterize_meshes, Meshes


def _face_normals_world(vertices_world: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tri = vertices_world[faces]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return F.normalize(normals, dim=-1, eps=1.0e-8)


def rasterize_surface_v1(
    vertices_iris: torch.Tensor,
    faces: torch.Tensor,
    *,
    camera: dict[str, torch.Tensor | float],
    config: SurfA1RasterConfigV1 = SurfA1RasterConfigV1(),
) -> dict[str, torch.Tensor]:
    config.validate()
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.numel() == 0:
        raise ValueError("SURF_A1_RASTER_FACE_SHAPE_INVALID")
    rasterize_meshes, Meshes = _pytorch3d_modules()
    ndc, vertices_world = project_iris_to_pytorch3d_ndc_v1(
        vertices_iris,
        common_center_world=camera["common_center_world"],
        common_half_extent=camera["common_half_extent"],
        camera_origin_world=camera["origin_world"],
        camera_right_world=camera["right_world"],
        camera_screen_up_world=camera["screen_up_world"],
        camera_forward_world=camera["forward_world"],
        camera_half_extent=camera["half_extent"],
    )
    meshes = Meshes(verts=[ndc], faces=[faces.long()])
    hard_face, hard_z, _, _ = rasterize_meshes(
        meshes,
        image_size=int(config.image_size),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=False,
        clip_barycentric_coords=False,
        cull_backfaces=False,
        cull_to_frustum=False,
    )
    hard_face = hard_face[0, ..., 0]
    coverage = hard_face >= 0
    depth = hard_z[0, ..., 0]
    depth = torch.where(coverage, depth, torch.zeros_like(depth))
    face_normals = _face_normals_world(vertices_world, faces.long())
    safe_face = hard_face.clamp_min(0)
    normal = face_normals[safe_face]
    normal = torch.where(coverage[..., None], normal, torch.zeros_like(normal))

    blur_radius = math.log(1.0 / 1.0e-4 - 1.0) * float(config.silhouette_sigma)
    soft_face, _, _, soft_dist = rasterize_meshes(
        meshes,
        image_size=int(config.image_size),
        blur_radius=float(blur_radius),
        faces_per_pixel=int(config.faces_per_pixel),
        perspective_correct=False,
        clip_barycentric_coords=True,
        cull_backfaces=False,
        cull_to_frustum=False,
    )
    valid = soft_face[0] >= 0
    probability = torch.sigmoid(-soft_dist[0] / float(config.silhouette_sigma)) * valid.to(soft_dist.dtype)
    alpha = 1.0 - torch.prod(1.0 - probability.clamp(0.0, 1.0), dim=-1)
    return {
        "soft_alpha": alpha,
        "hard_coverage": coverage,
        "depth_normalized": depth,
        "normal_world": normal,
        "vertices_world": vertices_world,
    }


def binary_precision_recall_v1(predicted: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    p = predicted.bool()
    t = truth.bool()
    tp = int((p & t).sum().item())
    fp = int((p & ~t).sum().item())
    fn = int((~p & t).sum().item())
    return {
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def raster_losses_v1(
    rendered: dict[str, torch.Tensor],
    *,
    source_mask: torch.Tensor,
    teacher_depth_normalized: torch.Tensor,
    teacher_normal_world: torch.Tensor,
    depth_normal_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask = source_mask.to(device=rendered["soft_alpha"].device, dtype=rendered["soft_alpha"].dtype)
    if mask.shape != rendered["soft_alpha"].shape:
        raise ValueError("SURF_A1_RASTER_MASK_SHAPE_DRIFT")
    silhouette = F.binary_cross_entropy(
        rendered["soft_alpha"].clamp(1.0e-5, 1.0 - 1.0e-5), mask
    )
    truth_valid = depth_normal_valid.to(device=mask.device).bool()
    overlap = truth_valid & rendered["hard_coverage"]
    if bool(overlap.any().item()):
        truth_depth = teacher_depth_normalized.to(device=mask.device, dtype=rendered["depth_normalized"].dtype)
        depth = F.smooth_l1_loss(rendered["depth_normalized"][overlap], truth_depth[overlap])
        truth_normal = teacher_normal_world.to(device=mask.device, dtype=rendered["normal_world"].dtype)
        dot = (rendered["normal_world"][overlap] * truth_normal[overlap]).sum(dim=-1).abs().clamp(0.0, 1.0)
        normal = (1.0 - dot).mean()
    else:
        anchor = rendered["soft_alpha"].sum() * 0.0
        depth = anchor
        normal = anchor
    return {
        "silhouette": silhouette,
        "depth": depth,
        "normal": normal,
        "depth_normal_overlap_fraction": overlap.float().mean(),
    }
