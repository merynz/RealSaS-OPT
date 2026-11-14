from __future__ import annotations

"""PyTorch3D GPU rasterization adapter for RSS.vNext Stage 3.

The first-hit triangle search is PyTorch3D's compiled CUDA rasterizer, NOT a
copy of BUSTED's bounded Python face-walk. The adapter preserves the original
4-channel Stage-3 ABI: (w0, w1, smooth alpha, face index + 1), bottom-left row
origin, and zero background. The caller performs one final Y flip into its
top-left PNG/source teacher coordinates.

Only first-hit face selection is discrete. Visible barycentrics and signed
face-edge distances remain differentiable. We compute RGB/roles/normals and
metric depth from the SAME selected TESSA/FlexiCubes triangle downstream.

PyTorch3D is BSD licensed (third-party, not vendored). A missing or
non-CUDA-enabled install MUST fail for the production CUDA path; there is no
unannounced fallback to the slow research rasterizer.
"""

from dataclasses import dataclass
from importlib import import_module

import torch
from torch import Tensor

RASTER_ID = "PYTORCH3D_COMPILED_FIRST_HIT_STAGE3_V1"


@dataclass(frozen=True)
class PyTorch3DRasterContextV1:
    # CPU is for explicitly requested compatibility tests only. Default
    # product GPU path must exercise PyTorch3D's compiled CUDA kernel.
    allow_cpu_test_only: bool = False


def _require_pytorch3d_v1():
    try:
        meshes = import_module("pytorch3d.structures").Meshes
        rasterize = import_module(
            "pytorch3d.renderer.mesh.rasterize_meshes"
        ).rasterize_meshes
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError(
            "RSS_VNEXT_PYTORCH3D_COMPILED_RASTER_REQUIRED"
        ) from exc
    return meshes, rasterize


def _input_check(clip: Tensor, faces: Tensor, resolution: tuple[int, int]) -> None:
    if (
        clip.ndim != 3 or clip.shape[0] != 1 or clip.shape[2] != 4
        or faces.ndim != 2 or faces.shape[-1] != 3
        or faces.dtype not in (torch.int32, torch.int64)
        or clip.device != faces.device
        or not torch.is_floating_point(clip)
        or not bool(torch.isfinite(clip).all())
        or not bool((clip[..., 3] > 0).all())
        or type(resolution) is not tuple or len(resolution) != 2
    ):
        raise ValueError("RSS_VNEXT_PYTORCH3D_INPUT_ABI")
    height, width = resolution
    if (
        type(height) is not int or type(width) is not int
        or not 1 <= height <= 2048 or not 1 <= width <= 2048
        or height * width > 2_097_152
        or not 1 <= faces.shape[0] <= 250_000
        or not 3 <= clip.shape[1] <= 250_000
        or int(faces.min()) < 0 or int(faces.max()) >= clip.shape[1]
    ):
        raise ValueError("RSS_VNEXT_PYTORCH3D_RESOURCE_OR_INDEX_DOMAIN")


class PyTorch3DRasterV1:
    """The existing Stage-3 raster ABI backed by PyTorch3D rasterize_meshes."""

    @staticmethod
    def RasterizeCudaContext() -> PyTorch3DRasterContextV1:
        return PyTorch3DRasterContextV1()

    @staticmethod
    def rasterize(
        context: PyTorch3DRasterContextV1,
        clip: Tensor,
        faces: Tensor,
        resolution: tuple[int, int],
    ) -> tuple[Tensor, None]:
        if type(context) is not PyTorch3DRasterContextV1:
            raise ValueError("RSS_VNEXT_PYTORCH3D_CONTEXT_ABI")
        _input_check(clip, faces, resolution)
        if clip.device.type != "cuda" and not context.allow_cpu_test_only:
            raise RuntimeError("RSS_VNEXT_PYTORCH3D_CUDA_STAGE3_REQUIRED")
        height, width = resolution
        Meshes, rasterize_meshes = _require_pytorch3d_v1()

        # BUSTED camera: +X right, +Y up, NDC [-1,1] spans either whole
        # image axis. PyTorch3D: +X left, +Y up; for rectangular images the
        # *longer* axis extends to +/- max(axis1/axis2,1). Adapt BOTH.
        # PyTorch3D additionally requires strictly positive view-Z. Stage-3
        # original clip Z is [-1,1]; map monotonically to [0.1,0.9] so
        # its depth ordering is identical without clipping minimum-Z faces.
        xyz = clip[0, :, :3].float() / clip[0, :, 3:4].float()
        verts_ndc = torch.stack((
            -xyz[:, 0] * max(1., width / height),
            xyz[:, 1] * max(1., height / width),
            0.5 + 0.4 * xyz[:, 2],
        ), dim=-1).contiguous()
        if not bool(torch.isfinite(verts_ndc).all()):
            raise ValueError("RSS_VNEXT_PYTORCH3D_PROJECTED_NONFINITE")
        # Stage-3 builds orthographic, normalized depth from its mesh; it
        # must not silently introduce near/far clipping or per-face remaps.
        if bool((verts_ndc[:, 2] <= 0).any()) or bool((verts_ndc[:, 2] > 1).any()):
            raise ValueError("RSS_VNEXT_PYTORCH3D_POSITIVE_DEPTH_REQUIRED")

        triangles = faces.long().contiguous()
        mesh = Meshes(verts=[verts_ndc], faces=[triangles])
        try:
            pix_to_face, zbuf, bary, dist = rasterize_meshes(
                mesh, image_size=(height, width),
                blur_radius=0.0, faces_per_pixel=1,
                bin_size=None, perspective_correct=False,
                clip_barycentric_coords=False, cull_backfaces=False,
                cull_to_frustum=False, z_clip_value=None,
            )
        except (RuntimeError, OSError) as exc:
            raise RuntimeError(
                "RSS_VNEXT_PYTORCH3D_CUDA_RASTER_EXECUTION_FAILED"
            ) from exc
        if (
            pix_to_face.shape != (1, height, width, 1)
            or zbuf.shape != (1, height, width, 1)
            or bary.shape != (1, height, width, 1, 3)
            or dist.shape != (1, height, width, 1)
            or pix_to_face.device != clip.device
            or bary.device != clip.device or dist.device != clip.device
            or not bool(torch.isfinite(bary).all())
        ):
            raise RuntimeError("RSS_VNEXT_PYTORCH3D_OUTPUT_ABI")
        # PyTorch3D rows are top-left; Stage-3's existing dr ABI is bottom-
        # left and the caller flips ONCE after constructing final channels.
        face_id = pix_to_face[..., 0].flip(dims=(1,))
        if bool((face_id >= triangles.shape[0]).any()):
            raise RuntimeError("RSS_VNEXT_PYTORCH3D_FACE_ID_OUT_OF_RANGE")
        valid = face_id >= 0
        barycentric = bary[..., 0, :].flip(dims=(1,))
        signed_sq_distance = dist[..., 0].flip(dims=(1,))
        # PyTorch3D returns a NEGATIVE signed squared distance inside the
        # face. Use an interior 1px sigmoid edge so geometry/silhouette
        # gradients exist without allowing a closer *outside* face to replace
        # a physically visible interior first-hit triangle.
        edge_distance_px = (
            -torch.sign(signed_sq_distance)
            * torch.sqrt(signed_sq_distance.abs() + 1.e-12)
            * (min(height, width) / 2.)
        )
        coverage_alpha = torch.where(
            valid, torch.sigmoid(4. * edge_distance_px),
            torch.zeros_like(edge_distance_px),
        )
        output = torch.stack((
            torch.where(valid, barycentric[..., 0], torch.zeros_like(coverage_alpha)),
            torch.where(valid, barycentric[..., 1], torch.zeros_like(coverage_alpha)),
            coverage_alpha,
            (face_id + 1).to(dtype=coverage_alpha.dtype),
        ), dim=-1)
        return output, None

    @staticmethod
    def interpolate(attributes: Tensor, raster: Tensor, faces: Tensor):
        if (
            attributes.ndim != 3 or attributes.shape[0] != 1
            or raster.ndim != 4 or raster.shape[0] != 1 or raster.shape[-1] != 4
            or attributes.device != raster.device or faces.device != raster.device
            or faces.ndim != 2 or faces.shape[-1] != 3
        ):
            raise ValueError("RSS_VNEXT_PYTORCH3D_INTERPOLATION_ABI")
        face_id = raster[0, :, :, 3].long() - 1
        valid = face_id >= 0
        if bool((face_id >= faces.shape[0]).any()):
            raise ValueError("RSS_VNEXT_PYTORCH3D_INTERPOLATION_FACE_ID")
        corners = attributes[0, faces.long()[face_id.clamp_min(0)]]
        w0, w1 = raster[0, ..., 0:1], raster[0, ..., 1:2]
        selected = (
            corners[..., 0, :] * w0
            + corners[..., 1, :] * w1
            + corners[..., 2, :] * (1. - w0 - w1)
        )
        return (selected * valid.unsqueeze(-1)).unsqueeze(0), None

    @staticmethod
    def antialias(rgba: Tensor, raster: Tensor, clip: Tensor, faces: Tensor) -> Tensor:
        if (
            rgba.shape[:3] != raster.shape[:3]
            or rgba.shape[-1] != 4
            or rgba.device != raster.device
        ):
            raise ValueError("RSS_VNEXT_PYTORCH3D_ANTIALIAS_ABI")
        alpha = raster[..., 2:3]
        return torch.cat((rgba[..., :3] * alpha, rgba[..., 3:4] * alpha), dim=-1)


__all__ = ["RASTER_ID", "PyTorch3DRasterContextV1", "PyTorch3DRasterV1"]
