from __future__ import annotations

"""Native CUDA Stage-3 probe: never substitute a CPU/mock raster.

This test may be skipped in ordinary hosted CPU CI. To qualify a GPU worker,
run with REALSAS_REQUIRE_STDGEN_CUDA=1; missing CUDA or nvdiffrast then FAILS.
A green CPU run of other Stage-3 tests is not NVIDIA evidence.
"""
import os

import pytest
import torch
from torch import nn

from models.rss_vnext.flexicubes_surface_evidence import RSSvNextFlexiCubesSurfaceEvidence
from models.rss_vnext.stdgen_flexicubes_mesh_render_v1 import (
    RASTER_ID, render_stdgen_layered_flexicubes_mesh_v1,
)

_REQUIRED = os.environ.get("REALSAS_REQUIRE_STDGEN_CUDA") == "1"


def _require_native_backend():
    if not torch.cuda.is_available():
        if _REQUIRED:
            pytest.fail("RSS_VNEXT_STAGE3_NATIVE_CUDA_REQUIRED_NOT_AVAILABLE")
        pytest.skip("No CUDA device; native Stage-3 proof not executed")
    try:
        import nvdiffrast.torch as dr
    except ImportError:
        if _REQUIRED:
            pytest.fail("RSS_VNEXT_STAGE3_NATIVE_NVDIFFRAST_NOT_INSTALLED")
        pytest.skip("No native nvdiffrast; native Stage-3 proof not executed")
    return dr


class _ActualPointQueries:
    @staticmethod
    def query(scene, positions):
        return {"H": torch.cat([positions, positions[..., :1]], dim=-1)}


def _real_witness():
    # An actual clipped triangle, large enough to occupy central raster
    # pixels. Off-axis vertices make color and metric depth sensitive to XYZ.
    vertices = torch.tensor(
        [[[-0.75, -0.7, 0.05], [0.75, -0.7, 0.20], [0., 0.85, 0.11]]],
        dtype=torch.float32, device="cuda", requires_grad=True,
    )
    return RSSvNextFlexiCubesSurfaceEvidence(
        xyz=vertices,
        faces=torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.long),
        local_features=torch.zeros((1, 3, 4), device="cuda"),
        presentation_role_logits=torch.zeros((1, 3, 2), device="cuda"),
        regularizer=vertices.square().mean(),
        sample_grid_xyz=torch.zeros((1, 8, 3), device="cuda"),
        signed_field_grid=torch.tensor(
            [[-1., 1., -1., 1., -1., 1., -1., 1.]], device="cuda",
        ),
        cube_parameters=torch.zeros((1, 21), device="cuda"),
        source_policy="SIGNED_HYPOTHESIS_ONLY",
        layer_name="peeled",
    )


def test_native_nvidia_stage3_first_hit_and_mesh_shared_field_gradients():
    dr = _require_native_backend()
    assert hasattr(dr, "RasterizeCudaContext")
    torch.manual_seed(7331)
    witness = _real_witness()
    cameras = torch.zeros((1, 1, 13), device="cuda")
    cameras[..., 3] = cameras[..., 7] = cameras[..., 11] = cameras[..., 12] = 1.
    camera = cameras.detach().clone()
    graphics = nn.Linear(4, 5, device="cuda")
    roles = nn.Linear(4, 2, device="cuda")
    scene = torch.zeros((1, 3, 1, 1, 1), device="cuda")

    result = render_stdgen_layered_flexicubes_mesh_v1(
        witness=witness,
        iris=_ActualPointQueries(),
        scene_triplanes=scene,
        graphics_head=graphics, semantic_role_head=roles,
        cameras=cameras, height=96, width=96,
        source_scene_prediction_sha256="d"*64,
    )
    assert result.rasterizer_identity == RASTER_ID
    assert result.rgb_white_composite.shape == (1, 96*96, 3)
    assert torch.isfinite(result.depth).all()
    assert torch.isfinite(result.normal_xyz).all()
    assert torch.isfinite(result.semantic_posterior).all()
    alpha = result.alpha.reshape(96, 96)
    assert alpha[48, 48].item() > 0.90  # genuine center first hit
    assert alpha[0, 0].item() < 0.01  # empty ray stays background
    visible = (result.alpha[..., 0] > 0.99).detach()
    assert visible.sum().item() > 150

    # All heads must affect the *same real mesh raster*; backprop must
    # reach geometry, rather than just a separately rendered NeRF.
    total = (
        result.rgb_white_composite[visible].square().mean()
        + result.semantic_posterior[visible, 0].square().mean()
        + 0.3 * result.depth[visible].square().mean()
        + 0.2 * result.normal_xyz[visible].square().mean()
    )
    total.backward()
    assert graphics.weight.grad is not None and graphics.weight.grad.abs().sum() > 0
    assert roles.weight.grad is not None and roles.weight.grad.abs().sum() > 0
    assert witness.xyz.grad is not None
    assert torch.isfinite(witness.xyz.grad).all()
    assert witness.xyz.grad.abs().sum() > 0
    assert torch.equal(cameras, camera)  # camera not optimized as fake geometry

    # Fail closed if this ran on a fixture rather than native CUDA.
    assert witness.xyz.device.type == "cuda"
    assert result.rasterizer_identity == RASTER_ID
