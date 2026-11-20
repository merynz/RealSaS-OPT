from __future__ import annotations

"""CUDA-only P0 M* posed RGB/alpha/SLR/geometry gradient ABI qualification.

Run on a pinned A100 environment with compiled PyTorch3D:
 PYTHONPATH=. python tools/research/rss_vnext_posed_mstar_cuda_qualification_v1.py \\
   --report /tmp/rss_mstar_pose_cuda_qualification.json

CPU test comparison and synthetic geometry NEVER certify a licensed source
corpus, garment motion quality, hidden appearance or product native renderer.
No automatic CPU fallback under CUDA failure.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from models.rss_vnext.product_mstar_pose_graphics_v1 import (
    render_posed_indexed_mstar_v1,
)
from models.rss_vnext.realsas_flexicubes_mesh_render_v1 import PYTORCH3D_RASTER_ID


class _AnalyticH(nn.Module):
    """Synthetic H query for exact pose/rest gradient ABI, not trained IRIS."""
    def query(self, planes, xyz):
        return {"H": xyz}


def _mesh(device):
    x = torch.linspace(-.68, .68, 7, device=device)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    z = (0.08 * torch.sin(3. * xx) * torch.cos(2. * yy))
    rest = torch.stack((xx, yy, z), dim=-1).reshape(1, -1, 3)
    faces = []
    for r in range(6):
        for c in range(6):
            a = r*7+c
            faces.extend(((a, a+1, a+7), (a+1, a+8, a+7)))
    return rest, torch.tensor(faces, dtype=torch.long, device=device)


def _camera(device):
    c = torch.zeros((1,1,13), device=device)
    c[0,0,2] = 2.
    c[0,0,3] = c[0,0,7] = 1.
    c[0,0,11] = -1.
    c[0,0,12] = 1.
    return c


def _render_once(device):
    rest, faces = _mesh(device)
    rest = rest.clone().requires_grad_()
    # Nonzero smooth pose, same vertex/face domain as rest.
    shift = 0.05 * torch.sin(rest[..., 0:1]*4)
    posed = torch.cat((
        rest[..., 0:1] + .03,
        rest[..., 1:2],
        rest[..., 2:3] + shift,
    ), dim=-1)
    head = nn.Linear(3,5).to(device)
    role = nn.Linear(3,3).to(device)
    torch.manual_seed(201)
    with torch.no_grad():
        head.weight.copy_(torch.arange(15,device=device).reshape(5,3)/110)
        head.bias.zero_()
        role.weight.copy_(torch.arange(9,device=device).reshape(3,3)/130)
        role.bias.zero_()
    planar = torch.zeros((1,3,1,2,2),device=device)
    t0 = perf_counter()
    r = render_posed_indexed_mstar_v1(
        rest_xyz=rest, posed_xyz=posed, faces=faces,
        scene_triplanes=planar, iris=_AnalyticH(),
        graphics_head=head, role_head=role,
        cameras=_camera(device), height=64, width=64,
    )
    loss = (
        (r.rgb_white-.4).square().mean()
        + (r.alpha-.6).square().mean()
        + 0.2 * (r.normal_xyz-.2).square().mean()
        + 0.01 * r.depth.square().mean()
    )
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (perf_counter()-t0)*1000
    if (
        rest.grad is None or not bool(torch.isfinite(rest.grad).all())
        or not bool(torch.any(rest.grad.abs() > 1e-9))
        or head.weight.grad is None
        or not bool(torch.any(head.weight.grad.abs() > 1e-9))
    ):
        raise RuntimeError("RSS_VNEXT_MSTAR_POSE_CUDA_GRADIENT_ABI_FAILED")
    return {
        "render":r,
        "loss":float(loss.detach()),
        "rest_gradient_norm":float(torch.linalg.vector_norm(rest.grad)),
        "graphics_gradient_norm":float(torch.linalg.vector_norm(head.weight.grad)),
        "duration_ms":ms,
        "face_count":int(faces.shape[0]),
    }


def run_qualification():
    if not torch.cuda.is_available():
        raise RuntimeError("RSS_VNEXT_MSTAR_POSE_COMPILED_CUDA_REQUIRED")
    from pytorch3d import __version__ as pytorch3d_version
    torch.cuda.reset_peak_memory_stats()
    cuda = _render_once(torch.device("cuda"))
    if cuda["render"].raster_backend_id != PYTORCH3D_RASTER_ID:
        raise RuntimeError("RSS_VNEXT_MSTAR_POSE_CUDA_BACKEND_IDENTITY_DRIFT")
    cpu = _render_once(torch.device("cpu"))
    # Compare visible interior only. The reference AA edge kernel and the
    # compiled PyTorch3D signed-distance edge kernel differ by design.
    a = cuda["render"]
    b = cpu["render"]
    mask = (a.alpha.detach().cpu()[...,0] > .99) & (
        b.alpha.detach()[...,0] > .99
    )
    count = int(mask.sum())
    if count < 12:
        raise RuntimeError("RSS_VNEXT_MSTAR_POSE_NO_COMPARABLE_VISIBLE_INTERIOR")
    rgb_max = float((a.rgb_white.detach().cpu()[mask] -
                     b.rgb_white.detach()[mask]).abs().max())
    depth_max = float((a.depth.detach().cpu()[mask] -
                       b.depth.detach()[mask]).abs().max())
    if rgb_max > 1e-3 or depth_max > 1e-3:
        raise RuntimeError("RSS_VNEXT_MSTAR_POSE_RGB_DEPTH_PARITY_FAILED")
    return {
        "schema":"RealSaS.RSSvNext.PosedMstarGraphicsCUDAQualification.v1",
        "torch_version":torch.__version__,
        "pytorch3d_version":pytorch3d_version,
        "gpu_name":torch.cuda.get_device_name(),
        "compiled_cuda_backend_executed":True,
        "backend_identity":PYTORCH3D_RASTER_ID,
        "synthetic_reference_only":True,
        "source_native_teacher_qualified":False,
        "native_Cpp_renderer_parity_qualified":False,
        "full_model_training_qualified":False,
        "product_release_qualified":False,
        "visible_interior_comparable_pixels":count,
        "rgb_max_abs_difference":rgb_max,
        "depth_max_abs_difference":depth_max,
        "cuda_forward_backward_ms":cuda["duration_ms"],
        "cuda_rest_gradient_norm":cuda["rest_gradient_norm"],
        "cuda_graphics_gradient_norm":cuda["graphics_gradient_norm"],
        "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report",type=Path)
    args = parser.parse_args()
    report = run_qualification()
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,sort_keys=True))


if __name__ == "__main__":
    main()
