from __future__ import annotations

import numpy as np
import pytest
import torch

from models.rss_vnext.stdgen_source_layer_truth_v1 import (
    StdGENSourceOpaqueTriangleSurfaceV1,
)
from models.rss_vnext.stdgen_source_nvdiffrast_v1 import (
    RASTER_ID, render_stdgen_opaque_source_layer_gpu_candidate_v1,
)


class _FakeDR:
    """CPU shape fixture ONLY; never satisfies native CUDA/GPU authority."""
    class RasterizeCudaContext:
        pass

    @staticmethod
    def rasterize(ctx,clip,tri,resolution):
        h,w=resolution
        rast=torch.zeros((1,h,w,4))
        rast[:,h//4:3*h//4,w//4:3*w//4,3]=1.
        return rast,torch.zeros_like(rast)

    @staticmethod
    def interpolate(attrs,rast,tri):
        h,w=rast.shape[1:3]
        point=attrs.mean(dim=1)[:,None,None].expand(-1,h,w,-1)
        return point,torch.zeros((1,h,w,0))


def _source():
    return StdGENSourceOpaqueTriangleSurfaceV1(
        source_mesh_xyz=np.array([[-.8,-.8,.2],[.8,-.8,.2],[0.,.8,.2]],dtype=np.float32),
        source_faces=np.array([[0,1,2]],dtype=np.int64),
        source_face_rgb_linear=np.array([[.2,.4,.6]],dtype=np.float32),
        source_face_role=np.array([1],dtype=np.int64),
        semantic_role_count=2,
        source_mesh_truth_sha256="a"*64,
        source_rights_admission_sha256="b"*64,
        renderer_version_sha256="c"*64,
        source_colors_from_artist_materials=True,
        source_roles_independently_assigned=True,
        source_surface_is_opaque=True,
    )


def _camera():
    cam=np.zeros(13,dtype=np.float32)
    cam[2]=-1.
    cam[3]=cam[7]=cam[11]=cam[12]=1.
    return cam


def test_source_gpu_candidate_channels_are_non_authoritative():
    render=render_stdgen_opaque_source_layer_gpu_candidate_v1(
        source=_source(),camera_13d=_camera(),height=32,width=32,
        role_subset=(1,),raster_backend=_FakeDR(),
    )
    assert render.identity==RASTER_ID
    assert render.authority=="GPU_RENDER_CANDIDATE__NOT_SOURCE_TEACHER_ADMITTED"
    assert render.rgb_white_linear.shape==(32,32,3)
    assert render.normal_xyz.shape==(32,32,3)
    assert render.metric_depth.shape==(32,32,1)
    assert render.visible_hit.shape==(32,32)
    assert render.alpha[16,16,0]==1.
    assert render.alpha[0,0,0]==0.
    assert np.allclose(render.rgb_white_linear[16,16],[.2,.4,.6])
    assert np.allclose(render.rgb_white_linear[0,0],[1.,1.,1.])
    assert render.role_indices[16,16]==1
    assert render.role_indices[0,0]==-1
    assert np.isclose(render.metric_depth[16,16,0],1.2)
    assert len(render.raster_receipt_sha256)==64


def test_wrong_source_role_and_camera_are_not_silently_accepted():
    with pytest.raises(ValueError,match="LAYER_EMPTY"):
        render_stdgen_opaque_source_layer_gpu_candidate_v1(
            source=_source(),camera_13d=_camera(),height=32,width=32,
            role_subset=(0,),raster_backend=_FakeDR(),
        )
    camera=_camera()
    camera[11]=2.
    with pytest.raises(ValueError,match="CAMERA_BASIS"):
        render_stdgen_opaque_source_layer_gpu_candidate_v1(
            source=_source(),camera_13d=camera,height=32,width=32,
            role_subset=(1,),raster_backend=_FakeDR(),
        )


def test_gpu_candidate_never_uses_cpu_in_qualifying_backend_mode():
    if torch.cuda.is_available():
        pytest.skip("Requires CUDA installation independent of native source GPU court")
    with pytest.raises(RuntimeError,match="GPU_SOURCE_CUDA_REQUIRED"):
        render_stdgen_opaque_source_layer_gpu_candidate_v1(
            source=_source(),camera_13d=_camera(),height=32,width=32,
            role_subset=(1,),
        )
