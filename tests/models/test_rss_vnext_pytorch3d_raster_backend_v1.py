from __future__ import annotations

"""Stage-3 PyTorch3D adapter contract tests.

Mocked PyTorch3D API tests confirm coordinate/shape/identity transport, NOT
compiled GPU quality. Real CPU/GPU raster tests are conditional on the native
extension being installed. A CUDA machine MUST fail when it is missing.
"""

import pytest
import torch

from models.rss_vnext.busted_raster_backend_v1 import BustedTorchRasterV1
from models.rss_vnext import pytorch3d_raster_backend_v1 as adapter
from models.rss_vnext.pytorch3d_raster_backend_v1 import (
    PyTorch3DRasterContextV1, PyTorch3DRasterV1, RASTER_ID,
)
from models.rss_vnext.realsas_flexicubes_mesh_render_v1 import (
    stage3_default_raster_backend_v1,
)


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(1)


def _two_depth_mesh():
    # Near triangle is the SECOND face, and source GL y=+ is up.
    far = [[-.8, -.7, .5, 1.], [.8, -.7, .5, 1.], [0., .8, .5, 1.]]
    near = [[-.8, -.7, -.5, 1.], [.8, -.7, -.5, 1.], [0., .8, -.5, 1.]]
    clip = torch.tensor([far+near], dtype=torch.float32, requires_grad=True)
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    return clip, faces


def _stub_compiled_api(monkeypatch, clip, faces, *, resolution):
    """Fake API for transport tests only; never identify it as a GPU proof."""
    record = {}
    h, w = resolution

    class FakeMeshes:
        def __init__(self, *, verts, faces):
            record["xyz"] = verts[0]
            record["faces"] = faces[0]

    def fake_raster(mesh, **kwargs):
        record["args"] = kwargs
        assert kwargs["image_size"] == (h, w)
        assert kwargs["blur_radius"] == 0.
        assert kwargs["faces_per_pixel"] == 1
        assert kwargs["perspective_correct"] is False
        assert kwargs["cull_backfaces"] is False
        ndc = record["xyz"]
        # The exact BUSTED GL -> PyTorch3D NDC convention is invertible for
        # both wide and tall rectangular images.
        inverse = torch.stack((
            -ndc[:, 0] / max(1., w/h),
            ndc[:, 1] / max(1., h/w),
            (ndc[:, 2] - .5) / .4,
            torch.ones_like(ndc[:, 0]),
        ), dim=-1)[None]
        if clip is not None:
            expected = clip.detach()
            assert torch.allclose(inverse, expected, atol=1.e-6, rtol=0)
        cpu_rast, _ = BustedTorchRasterV1.rasterize(
            BustedTorchRasterV1.RasterizeCudaContext(),
            inverse, faces, resolution=(h, w),
        )
        face_idx = (cpu_rast[..., 3].long()-1).flip(1)[..., None]
        valid = face_idx >= 0
        bary = torch.stack((
            cpu_rast[..., 0],
            cpu_rast[..., 1],
            1. - cpu_rast[..., 0] - cpu_rast[..., 1],
        ), dim=-1).flip(1)[:, :, :, None, :]
        # Fake signed squared distances, matching API shape/sign only.
        distances = torch.where(
            valid, -torch.ones_like(face_idx, dtype=inverse.dtype)*0.01,
            -torch.ones_like(face_idx, dtype=inverse.dtype),
        )
        zbuf = torch.where(
            valid, torch.ones_like(distances)*0.3,
            -torch.ones_like(distances),
        )
        return face_idx,zbuf,bary,distances

    monkeypatch.setattr(
        adapter, "_require_pytorch3d_v1",
        lambda: (FakeMeshes, fake_raster),
    )
    return record


@pytest.mark.parametrize("resolution", [(32,32),(16,32),(32,16)])
def test_adapter_maps_ndc_depth_orientation_and_barycentric_abi(monkeypatch,resolution):
    clip, faces = _two_depth_mesh()
    record = _stub_compiled_api(
        monkeypatch, clip, faces, resolution=resolution,
    )
    raster, _ = PyTorch3DRasterV1.rasterize(
        PyTorch3DRasterContextV1(allow_cpu_test_only=True),
        clip, faces, resolution,
    )
    h,w=resolution
    assert raster.shape == (1,h,w,4)
    assert record["faces"].dtype == torch.long
    assert raster[0,h//2,w//2,3].item()==2
    assert raster[0,0,0,3].item()==0
    assert float(raster[0,h//2,w//2,2])>0
    assert float(raster[...,2].max()) <= 1.
    attrs = torch.tensor([[
        [0.,0.,0.], [1.,0.,0.], [.5,1.,0.],
        [0.,0.,2.], [1.,0.,2.], [.5,1.,2.],
    ]], requires_grad=True)
    interp,_ = PyTorch3DRasterV1.interpolate(attrs, raster, faces)
    assert interp.shape==(1,h,w,3)
    assert float(interp[0,h//2,w//2,2]) == pytest.approx(2.)
    soft=PyTorch3DRasterV1.antialias(
        torch.ones((1,h,w,4)),raster,clip,faces,
    )
    loss=interp[0,h//2,w//2,0]+soft[...,3].mean()
    loss.backward()
    assert attrs.grad is not None and attrs.grad.abs().sum()>0
    assert clip.grad is not None and clip.grad.abs().sum()>0


def test_prod_cuda_selects_pytorch3d_never_cpu_python_implicitly():
    assert stage3_default_raster_backend_v1(torch.device("cuda")) is PyTorch3DRasterV1
    assert stage3_default_raster_backend_v1(torch.device("cpu")) is BustedTorchRasterV1
    assert stage3_default_raster_backend_v1(torch.device("meta")) is None
    assert RASTER_ID.startswith("PYTORCH3D_COMPILED")


def test_missing_compiled_pytorch3d_fails_closed(monkeypatch):
    clip,faces=_two_depth_mesh()
    def unavailable():
        raise RuntimeError("RSS_VNEXT_PYTORCH3D_COMPILED_RASTER_REQUIRED")
    monkeypatch.setattr(adapter,"_require_pytorch3d_v1",unavailable)
    with pytest.raises(RuntimeError,match="COMPILED_RASTER_REQUIRED"):
        PyTorch3DRasterV1.rasterize(
            PyTorch3DRasterContextV1(allow_cpu_test_only=True),
            clip,faces,(16,16),
        )
    with pytest.raises(RuntimeError,match="CUDA_STAGE3_REQUIRED"):
        PyTorch3DRasterV1.rasterize(
            PyTorch3DRasterContextV1(),clip,faces,(16,16),
        )


def test_invalid_faces_and_w_values_refused_before_dispatch(monkeypatch):
    clip,faces=_two_depth_mesh()
    _stub_compiled_api(monkeypatch,clip,faces,resolution=(16,16))
    with pytest.raises(ValueError,match="RESOURCE_OR_INDEX"):
        PyTorch3DRasterV1.rasterize(
            PyTorch3DRasterContextV1(allow_cpu_test_only=True),
            clip,torch.tensor([[0,1,100]]),(16,16),
        )
    broken=clip.detach().clone()
    broken[0,0,3]=0
    with pytest.raises(ValueError,match="INPUT_ABI"):
        PyTorch3DRasterV1.rasterize(
            PyTorch3DRasterContextV1(allow_cpu_test_only=True),
            broken,faces,(16,16),
        )


def test_real_compiled_cpu_first_hit_and_gradients_when_pytorch3d_installed():
    try:
        adapter._require_pytorch3d_v1()
    except RuntimeError:
        pytest.skip("PyTorch3D native extension not installed in CPU CI")
    clip,faces=_two_depth_mesh()
    raster,_=PyTorch3DRasterV1.rasterize(
        PyTorch3DRasterContextV1(allow_cpu_test_only=True),
        clip,faces,(32,32),
    )
    assert raster[0,16,16,3].item()==2
    assert raster[0,0,0,3].item()==0
    assert raster[...,2].max()>0
    interp,_=PyTorch3DRasterV1.interpolate(clip[...,0:3],raster,faces)
    (interp[0,16,16,:].sum()+raster[...,2].sum()).backward()
    assert clip.grad is not None and torch.isfinite(clip.grad).all()
    assert clip.grad.abs().sum()>0


def test_real_compiled_cuda_required_for_product_stage3():
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA CUDA worker; product CUDA qualification still required")
    # On GPU, a missing PyTorch3D binary is a TEST FAILURE (not a skip).
    clip,faces=_two_depth_mesh()
    clip=clip.detach().to("cuda").requires_grad_()
    faces=faces.to("cuda")
    raster,_=PyTorch3DRasterV1.rasterize(
        PyTorch3DRasterV1.RasterizeCudaContext(),clip,faces,(64,64),
    )
    assert raster[0,32,32,3].item()==2
    interp,_=PyTorch3DRasterV1.interpolate(clip[...,:3],raster,faces)
    (interp[0,32,32].sum()+raster[...,2].sum()).backward()
    assert clip.grad is not None and torch.isfinite(clip.grad).all()
    assert clip.grad.abs().sum()>0
