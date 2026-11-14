from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch
from torch import nn

from models.rss_vnext.flexicubes_surface_evidence import RSSvNextFlexiCubesSurfaceEvidence
from models.rss_vnext.realsas_flexicubes_mesh_render_v1 import (
    RASTER_ID, PYTORCH3D_RASTER_ID, INJECTED_RASTER_ID,
    render_realsas_layered_flexicubes_mesh_v1,
)


class _SyntheticDR:
    """CPU MOCK only: not a replacement for native NVIDIA CUDA rasterization."""
    class RasterizeCudaContext: pass
    @staticmethod
    def rasterize(ctx,clip,tri,resolution):
        h,w=resolution
        a=torch.zeros((1,h,w,4))
        a[...,3]=1.
        return a,torch.zeros_like(a)
    @staticmethod
    def interpolate(attrs,rast,tri):
        h,w=rast.shape[1:3]
        x=attrs.mean(dim=1)[:,None,None].expand(-1,h,w,-1)
        return x,torch.zeros((1,h,w,0))
    @staticmethod
    def antialias(x,rast,clip,tri): return x


class _MiniIRIS:
    @staticmethod
    def query(scene,point):
        return {"H":torch.cat((point,point[:,:,:1]),dim=-1)}


def _witness():
    xyz=torch.tensor([[[-.3,-.3,-.1],[.3,-.3,.1],[0.,.3,0.]]],requires_grad=True)
    return RSSvNextFlexiCubesSurfaceEvidence(
        xyz=xyz,faces=torch.tensor([[0,1,2]],dtype=torch.long),
        local_features=torch.zeros((1,3,4)),
        presentation_role_logits=torch.zeros((1,3,2)),
        regularizer=(xyz.square()).mean(),
        sample_grid_xyz=torch.zeros((1,8,3)),
        signed_field_grid=torch.tensor([[-1.,1.,-1.,1.,-1.,1.,-1.,1.]]),
        cube_parameters=torch.zeros((1,21)),
        source_policy="SIGNED_HYPOTHESIS_ONLY",
        layer_name="peeled",
    )


def _camera():
    c=torch.zeros((1,1,13))
    c[:,:,3]=c[:,:,7]=c[:,:,11]=c[:,:,12]=1.
    return c


def test_same_H_and_actual_mesh_channels_flow_to_stage3_raster_contract():
    witness=_witness()
    rgb_head=nn.Linear(4,5)
    role_head=nn.Linear(4,2)
    out=render_realsas_layered_flexicubes_mesh_v1(
        witness=witness,iris=_MiniIRIS(),scene_triplanes=torch.zeros((1,3,1,1,1)),
        graphics_head=rgb_head,semantic_role_head=role_head,
        cameras=_camera(),height=8,width=8,
        source_scene_prediction_sha256="a"*64,
        raster_backend=_SyntheticDR(),
    )
    assert out.rasterizer_identity==INJECTED_RASTER_ID
    assert out.rgb_white_composite.shape==(1,64,3)
    assert out.alpha.shape==(1,64,1)
    assert out.semantic_posterior.shape==(1,64,2)
    assert out.normal_xyz.shape==(1,64,3)
    assert out.depth.shape==(1,64,1)
    assert out.mesh_surface_prediction_sha256!="a"*64
    (out.rgb_white_composite.sum()+out.semantic_posterior[...,0].sum()).backward()
    assert rgb_head.weight.grad.abs().sum()>0
    assert role_head.weight.grad.abs().sum()>0
    assert witness.xyz.grad is not None


def test_owned_reference_default_backend_runs_without_nvidia_and_backpropagates():
    w=_witness()
    color_head=nn.Linear(4,5)
    role_head=nn.Linear(4,2)
    out=render_realsas_layered_flexicubes_mesh_v1(
        witness=w,iris=_MiniIRIS(),scene_triplanes=torch.zeros((1,3,1,1,1)),
        graphics_head=color_head,semantic_role_head=role_head,
        cameras=_camera(),height=16,width=16,
        source_scene_prediction_sha256="a"*64,
    )
    assert out.rasterizer_identity == "BUSTED_TORCH_FIRST_HIT_DIFFERENTIABLE_STAGE3_V1"
    assert (out.alpha > 0).any()
    assert (out.alpha < 1).any()
    (out.rgb_white_composite.sum()+out.alpha.sum()
     +out.semantic_posterior[...,0].sum()+out.depth.sum()).backward()
    assert color_head.weight.grad is not None
    assert role_head.weight.grad is not None
    assert w.xyz.grad is not None and torch.isfinite(w.xyz.grad).all()
    assert w.xyz.grad.abs().sum() > 0


def test_unqualified_source_or_non_gpu_native_backend_refused():
    w=_witness()
    w.source_policy="SOURCE_MEASURED"
    with pytest.raises(ValueError,match="WITNESS_AUTHORITY"):
        render_realsas_layered_flexicubes_mesh_v1(
            witness=w,iris=_MiniIRIS(),scene_triplanes=torch.zeros((1,3,1,1,1)),
            graphics_head=nn.Linear(4,5),semantic_role_head=nn.Linear(4,2),
            cameras=_camera(),height=8,width=8,
            source_scene_prediction_sha256="a"*64,raster_backend=_SyntheticDR(),
        )


def test_stage3_pytorch3d_transport_keeps_live_h_and_render_channels(monkeypatch):
    from models.rss_vnext.pytorch3d_raster_backend_v1 import (
        PyTorch3DRasterV1, PyTorch3DRasterContextV1,
    )
    from tests.models.test_rss_vnext_pytorch3d_raster_backend_v1 import (
        _stub_compiled_api,
    )
    # CPU-only API stub verifies the Stage-3 integration/renderer provenance.
    # The native PyTorch3D compiled GPU proof runs separately on a CUDA worker.
    _stub_compiled_api(
        monkeypatch,None,torch.tensor([[0,1,2]],dtype=torch.int32),
        resolution=(16,32),
    )
    witness=_witness()
    rgb_head=nn.Linear(4,5)
    role_head=nn.Linear(4,2)
    out=render_realsas_layered_flexicubes_mesh_v1(
        witness=witness,iris=_MiniIRIS(),
        scene_triplanes=torch.zeros((1,3,1,1,1)),
        graphics_head=rgb_head,semantic_role_head=role_head,
        cameras=_camera(),height=16,width=32,
        source_scene_prediction_sha256="a"*64,
        raster_backend=PyTorch3DRasterV1,
        raster_context=PyTorch3DRasterContextV1(allow_cpu_test_only=True),
    )
    assert out.rasterizer_identity==PYTORCH3D_RASTER_ID
    assert out.rgb_white_composite.shape==(1,16*32,3)
    assert out.alpha.shape==(1,16*32,1)
    assert out.semantic_posterior.shape==(1,16*32,2)
    assert out.normal_xyz.shape==(1,16*32,3)
    assert out.depth.shape==(1,16*32,1)
    assert (out.alpha>0).any()
    assert torch.isfinite(out.depth).all()
    (out.rgb_white_composite.sum()+out.alpha.sum()
     +out.semantic_posterior[...,0].sum()+out.depth.sum()).backward()
    assert rgb_head.weight.grad is not None and rgb_head.weight.grad.abs().sum()>0
    assert role_head.weight.grad is not None and role_head.weight.grad.abs().sum()>0
    assert witness.xyz.grad is not None and torch.isfinite(witness.xyz.grad).all()
    assert witness.xyz.grad.abs().sum()>0
