from __future__ import annotations

"""BUSTED-owned bounded orthographic triangle raster and differentiable readback.

No nvdiffrast import, source reuse, GPU-specific binary, or external licence
requirement. Face selection is a hard depth/coverage decision, as in normal
rasterization; selected barycentrics, surface position, normals and a smooth
silhouette edge retain PyTorch autograd on CPU and CUDA. This is a reference
backend (bounding-box face walk), NOT an optimized high-throughput GPU raster.
Training/release throughput must be benchmarked separately, not inferred from
small correctness tests.
"""
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class RasterizeContext:
    name: str = "BUSTED_TORCH_REFERENCE_V1"


class BustedTorchRasterV1:
    """A minimal raster backend with the subset of operations used by Stage3.

    Returns the exact 4-channel raster ABI expected by the existing mesh-stage
    consumer: w0, w1, edge coverage, triangle ID + 1 (zero is background).
    The bottom-left raster origin is intentional; the caller flips once.
    """

    @staticmethod
    def RasterizeCudaContext() -> RasterizeContext:
        # Historical caller name; valid for CPU and CUDA, no CUDA claim.
        return RasterizeContext()

    @staticmethod
    def _barycentric(px: Tensor, py: Tensor, a: Tensor, b: Tensor, c: Tensor):
        den = (b[..., 1] - c[..., 1]) * (a[..., 0] - c[..., 0]) + (
            c[..., 0] - b[..., 0]
        ) * (a[..., 1] - c[..., 1])
        tiny = den.abs() < 1.e-12
        den = torch.where(tiny, torch.ones_like(den), den)
        w0 = ((b[..., 1] - c[..., 1]) * (px - c[..., 0]) + (
            c[..., 0] - b[..., 0]
        ) * (py - c[..., 1])) / den
        w1 = ((c[..., 1] - a[..., 1]) * (px - c[..., 0]) + (
            a[..., 0] - c[..., 0]
        ) * (py - c[..., 1])) / den
        return w0, w1, 1. - w0 - w1, tiny

    @staticmethod
    def rasterize(context: RasterizeContext, clip: Tensor, faces: Tensor, resolution: tuple[int, int]):
        if (
            type(context) is not RasterizeContext
            or clip.ndim != 3 or clip.shape[0] != 1 or clip.shape[2] != 4
            or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype not in (torch.int32, torch.int64)
            or clip.device != faces.device
            or not bool(torch.isfinite(clip).all())
            or not bool((clip[..., 3] > 0).all())
            or type(resolution) is not tuple or len(resolution) != 2
        ):
            raise ValueError("BUSTED_RASTER_ABI_INVALID")
        h, width = resolution
        if (
            type(h) is not int or type(width) is not int
            or not 1 <= h <= 2048 or not 1 <= width <= 2048
            or h * width > 2_097_152 or faces.shape[0] > 250_000
            or faces.shape[0] < 1 or clip.shape[1] < 3
            or int(faces.min()) < 0 or int(faces.max()) >= clip.shape[1]
        ):
            raise ValueError("BUSTED_RASTER_RESOURCE_OR_FACE_DOMAIN")
        # Topology selection is deliberately nondifferentiable. Only final
        # per-pixel barycentric attributes and soft edge coverage carry grads.
        with torch.no_grad():
            xy = (clip[0, :, :2] / clip[0, :, 3:4]).detach()
            z = (clip[0, :, 2] / clip[0, :, 3]).detach()
            win_id = torch.full((h, width), -1, dtype=torch.long, device=clip.device)
            win_depth = torch.full((h, width), float("inf"), device=clip.device)
            win_edge = torch.full((h, width), -float("inf"), device=clip.device)
            for fi, ids in enumerate(faces.long()):
                a, b, c = xy[ids]
                den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
                if abs(float(den)) < 1.e-12:
                    continue
                xs = (xy[ids, 0] + 1.) * (width/2.) - .5
                ys = (xy[ids, 1] + 1.) * (h/2.) - .5
                x0 = max(0, int(torch.floor(xs.min()).item()) - 2)
                x1 = min(width, int(torch.ceil(xs.max()).item()) + 3)
                y0 = max(0, int(torch.floor(ys.min()).item()) - 2)
                y1 = min(h, int(torch.ceil(ys.max()).item()) + 3)
                if x0 >= x1 or y0 >= y1:
                    continue
                yy, xx = torch.meshgrid(
                    torch.arange(y0, y1, device=clip.device, dtype=clip.dtype),
                    torch.arange(x0, x1, device=clip.device, dtype=clip.dtype),
                    indexing="ij",
                )
                px = (xx + .5) * (2./width) - 1.
                py = (yy + .5) * (2./h) - 1.
                w0, w1, w2, _ = BustedTorchRasterV1._barycentric(px, py, a, b, c)
                # Signed distance to three directed sides, in pixel units.
                sides = (
                    (b-c)*torch.tensor((width/2., h/2.), device=clip.device),
                    (c-a)*torch.tensor((width/2., h/2.), device=clip.device),
                    (a-b)*torch.tensor((width/2., h/2.), device=clip.device),
                )
                area = abs(den) * (width*h/4.)
                edge = torch.minimum(
                    torch.minimum(w0*area/sides[0].norm().clamp_min(1.e-8),
                                  w1*area/sides[1].norm().clamp_min(1.e-8)),
                    w2*area/sides[2].norm().clamp_min(1.e-8),
                )
                inside = edge >= -1.e-6
                near = edge >= -1.5
                depth = w0*z[ids[0]] + w1*z[ids[1]] + w2*z[ids[2]]
                current_id = win_id[y0:y1, x0:x1]
                current_z = win_depth[y0:y1, x0:x1]
                current_edge = win_edge[y0:y1, x0:x1]
                # Interior always outranks near-edge extensions. Exact depth
                # ties are resolved by the smaller canonical triangle index.
                was_inside = current_edge >= -1.e-6
                select_inside = inside & (
                    ~was_inside | (depth < current_z - 1.e-7)
                    | ((depth-current_z).abs() <= 1.e-7)
                    & ((current_id < 0) | (fi < current_id))
                )
                select_edge = near & ~inside & ~was_inside & (
                    (edge > current_edge + 1.e-7) |
                    ((edge-current_edge).abs() <= 1.e-7)
                    & ((current_id < 0) | (fi < current_id))
                )
                take = select_inside | select_edge
                win_id[y0:y1, x0:x1] = torch.where(take, fi, current_id)
                win_depth[y0:y1, x0:x1] = torch.where(take, depth, current_z)
                win_edge[y0:y1, x0:x1] = torch.where(take, edge, current_edge)

        covered = win_id >= 0
        triangles = faces.long()[win_id.clamp_min(0)]
        vertices = (clip[0, triangles, :2] /
                    clip[0, triangles, 3:4])
        yy, xx = torch.meshgrid(
            torch.arange(h, device=clip.device, dtype=clip.dtype),
            torch.arange(width, device=clip.device, dtype=clip.dtype),
            indexing="ij",
        )
        px = (xx + .5)*(2./width)-1.
        py = (yy + .5)*(2./h)-1.
        w0, w1, w2, _ = BustedTorchRasterV1._barycentric(
            px, py, vertices[..., 0, :], vertices[..., 1, :], vertices[..., 2, :],
        )
        sides = (
            (vertices[..., 1, :]-vertices[..., 2, :])*
                torch.tensor((width/2., h/2.), device=clip.device),
            (vertices[..., 2, :]-vertices[..., 0, :])*
                torch.tensor((width/2., h/2.), device=clip.device),
            (vertices[..., 0, :]-vertices[..., 1, :])*
                torch.tensor((width/2., h/2.), device=clip.device),
        )
        den = ((vertices[..., 1, 1]-vertices[..., 2, 1])*
               (vertices[..., 0, 0]-vertices[..., 2, 0]) +
               (vertices[..., 2, 0]-vertices[..., 1, 0])*
               (vertices[..., 0, 1]-vertices[..., 2, 1])).abs()
        area = den*(width*h/4.)
        edge = torch.minimum(
            torch.minimum(w0*area/sides[0].norm(dim=-1).clamp_min(1.e-8),
                          w1*area/sides[1].norm(dim=-1).clamp_min(1.e-8)),
            w2*area/sides[2].norm(dim=-1).clamp_min(1.e-8),
        )
        # A narrow antialiased boundary provides silhouette gradients even
        # just outside the winning triangle; no depth blurring.
        alpha = torch.sigmoid(edge*4.)
        raster = torch.stack((
            torch.where(covered, w0, torch.zeros_like(w0)),
            torch.where(covered, w1, torch.zeros_like(w1)),
            torch.where(covered, alpha, torch.zeros_like(alpha)),
            (win_id+1).to(clip.dtype),
        ), dim=-1)[None]
        return raster, None

    @staticmethod
    def interpolate(attributes: Tensor, raster: Tensor, faces: Tensor):
        if (
            attributes.ndim != 3 or attributes.shape[0] != 1
            or raster.ndim != 4 or raster.shape[0] != 1 or raster.shape[-1] != 4
            or attributes.device != raster.device or faces.device != raster.device
        ):
            raise ValueError("BUSTED_RASTER_INTERPOLATION_ABI")
        fi = (raster[0, ..., 3].long()-1)
        valid = fi >= 0
        ids = faces.long()[fi.clamp_min(0)]
        corners = attributes[0, ids]
        w0, w1 = raster[0, ..., 0:1], raster[0, ..., 1:2]
        result = (corners[..., 0, :]*w0 + corners[..., 1, :]*w1
                  + corners[..., 2, :]*(1.-w0-w1))
        return (result*valid.unsqueeze(-1)).unsqueeze(0), None

    @staticmethod
    def antialias(rgba: Tensor, raster: Tensor, clip: Tensor, faces: Tensor) -> Tensor:
        if rgba.shape[:3] != raster.shape[:3] or rgba.shape[-1] != 4:
            raise ValueError("BUSTED_RASTER_ANTIALIAS_DOMAIN")
        a = raster[..., 2:3]
        return torch.cat((rgba[..., :3]*a, rgba[..., 3:4]*a), dim=-1)


__all__ = ["RasterizeContext", "BustedTorchRasterV1"]
