from __future__ import annotations

"""Real CUDA-only PyTorch3D Stage-3 raster ABI and throughput probe.

On a production A100 training image:
    PYTHONPATH=. python tools/research/rss_vnext_pytorch3d_cuda_qualification_v1.py \
      --report /tmp/rss_vnext_pytorch3d_cuda_receipt.json

This is a *binary* qualification microcourt. It does not establish overall
source-render fidelity, animation quality or product admission by itself.
No mock, CPU substitute, or fallback is allowed in this script.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from models.rss_vnext.busted_raster_backend_v1 import BustedTorchRasterV1
from models.rss_vnext.pytorch3d_raster_backend_v1 import (
    RASTER_ID, PyTorch3DRasterV1,
)


def _overlapping_triangles(device):
    far = [[-.8, -.8, .5, 1.], [.8, -.8, .5, 1.], [0., .8, .5, 1.]]
    near = [[-.8, -.8, -.5, 1.], [.8, -.8, -.5, 1.], [0., .8, -.5, 1.]]
    clip = torch.tensor([far+near], device=device, dtype=torch.float32)
    faces = torch.tensor([[0,1,2],[3,4,5]], device=device, dtype=torch.int32)
    return clip, faces


def _grid_mesh(device, side=40):
    coords = torch.linspace(-.9, .9, side, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    z = .2 * torch.cos(3. * x) * torch.sin(3. * y)
    vertices = torch.stack((
        x.reshape(-1), y.reshape(-1), z.reshape(-1),
        torch.ones(side*side,device=device),
    ), dim=-1)[None]
    faces = []
    for row in range(side-1):
        for col in range(side-1):
            a = row*side+col
            faces.append((a,a+1,a+side))
            faces.append((a+1,a+side+1,a+side))
    return vertices, torch.tensor(faces,dtype=torch.int32,device=device)


def run_qualification(*, warmup=3, repeats=8):
    if not torch.cuda.is_available():
        raise RuntimeError("RSS_VNEXT_PYTORCH3D_A100_NATIVE_CUDA_REQUIRED")
    device=torch.device("cuda")
    from pytorch3d import __version__ as pytorch3d_version
    result={
        "schema":"RealSaS.RSSvNext.PyTorch3DStage3CUDAQualificationProbe.v1",
        "backend_identity":RASTER_ID,
        "gpu_name":torch.cuda.get_device_name(device),
        "torch_version":torch.__version__,
        "cuda_runtime":torch.version.cuda,
        "pytorch3d_version":pytorch3d_version,
        "CPU_reference_is_product_backend":False,
        "native_cuda_raster_executed":False,
        "full_product_qualified":False,
    }
    probe={}
    for h,w in ((64,64),(32,64),(64,32)):
        cpu_clip,cpu_faces=_overlapping_triangles(torch.device("cpu"))
        torch_raster,_=BustedTorchRasterV1.rasterize(
            BustedTorchRasterV1.RasterizeCudaContext(),
            cpu_clip,cpu_faces,(h,w),
        )
        cuda_clip,cuda_faces=_overlapping_triangles(device)
        cuda_clip.requires_grad_(True)
        native,_=PyTorch3DRasterV1.rasterize(
            PyTorch3DRasterV1.RasterizeCudaContext(),
            cuda_clip,cuda_faces,(h,w),
        )
        torch.cuda.synchronize()
        ref_id=torch_raster[...,3].long()
        nat_id=native[...,3].detach().cpu().long()
        # Compare the entire interior where the reference is not near an AA
        # edge. Distances and AA curves may differ at boundaries.
        ref_interior=torch_raster[...,2]>0.99
        comparable=ref_interior & (nat_id>0)
        if comparable.sum().item()<8:
            raise RuntimeError("RSS_VNEXT_PYTORCH3D_NO_COMPARABLE_INTERIOR")
        mismatch=(ref_id[comparable]!=nat_id[comparable]).sum().item()
        if mismatch:
            raise RuntimeError("RSS_VNEXT_PYTORCH3D_FIRST_HIT_FACE_DISAGREEMENT")
        if int(nat_id[0,h//2,w//2])!=2:
            raise RuntimeError("RSS_VNEXT_PYTORCH3D_NEAR_FACE_DEPTH_WRONG")
        positions=cuda_clip[...,:3]
        sampled,_=PyTorch3DRasterV1.interpolate(positions,native,cuda_faces)
        loss=(sampled[0,h//2,w//2].square().mean()+native[...,2].mean())
        loss.backward()
        if (
            cuda_clip.grad is None
            or not bool(torch.isfinite(cuda_clip.grad).all())
            or not bool((cuda_clip.grad[...,0:2].abs().sum()>0))
        ):
            raise RuntimeError("RSS_VNEXT_PYTORCH3D_VERTEX_GRADIENT_FAILURE")
        probe[f"{h}x{w}"]={
            "comparable_interior_pixels":int(comparable.sum().item()),
            "first_hit_mismatches":int(mismatch),
            "center_face_id_plus_one":2,
            "finite_vertex_gradient":True,
            "nonzero_xy_vertex_gradient":True,
        }
    result["first_hit_and_gradient_probes"]=probe

    mesh,faces=_grid_mesh(device)
    assert faces.numel()>3000
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    def execute():
        raster,_=PyTorch3DRasterV1.rasterize(
            PyTorch3DRasterV1.RasterizeCudaContext(),
            mesh,faces,(256,256),
        )
        return raster
    for _ in range(warmup):
        execute()
    torch.cuda.synchronize()
    elapsed=[]
    for _ in range(repeats):
        start=time.perf_counter()
        execute()
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter()-start)*1000.)
    result["microbenchmark"]={
        "face_count":int(faces.shape[0]),
        "resolution":[256,256],
        "warmup":warmup,
        "repeats":repeats,
        "latency_ms_median":statistics.median(elapsed),
        "latency_ms_p95":sorted(elapsed)[max(0,int(0.95*len(elapsed))-1)],
        "peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device),
        "full_training_gradient_or_six_view_throughput_qualified":False,
    }
    result["native_cuda_raster_executed"]=True
    result["native_cuda_probe_passed"]=True
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=8)
    args=p.parse_args()
    if not 1 <= args.warmup <= 50 or not 2 <= args.repeats <= 200:
        p.error("warmup or repeats outside guarded budget")
    receipt=run_qualification(warmup=args.warmup,repeats=args.repeats)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(
            json.dumps(receipt,sort_keys=True,indent=2)+"\n",
            encoding="utf-8",
        )
    print(json.dumps(receipt,sort_keys=True))


if __name__=="__main__":
    main()
