import torch

from models.iris.v3.scene_first_signed_v3 import SceneFirstSignedGeometryV3
from models.iris.v4.scene_first_signed_v4 import (
    SceneFirstSignedGeometryConfigV4,
    SceneFirstSignedGeometryV4,
)


def _count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default_v4_is_real_32_query_lattice_and_64_final_triplane():
    cfg = SceneFirstSignedGeometryConfigV4()
    model = SceneFirstSignedGeometryV4(cfg)
    assert model.architecture_id == "RealSaS.IRIS.SceneFirstSignedGeometry.v4.demo64"
    assert tuple(model.scene_fusion.scene_queries.shape) == (3, 32, 32, 256)
    assert model.scene_fusion.upsample.in_channels == 256
    assert model.scene_fusion.upsample.out_channels == 64
    assert model.scene_fusion.upsample.kernel_size == (2, 2)
    assert model.scene_fusion.upsample.stride == (2, 2)
    assert len(model.scene_fusion.decoder.layers) == 4
    assert len(model.surface_field.blocks) == 4
    assert 3 * cfg.plane_channels == 192


def test_first_v4_keeps_scene_transformer_width_and_depth_frozen_vs_v3():
    v3 = SceneFirstSignedGeometryV3()
    v4 = SceneFirstSignedGeometryV4()
    assert v3.config.scene_dim == v4.config.scene_dim == 256
    assert v3.config.attention_heads == v4.config.attention_heads == 8
    assert v3.config.scene_layers == v4.config.scene_layers == 4
    assert v3.config.plane_channels == v4.config.plane_channels == 64
    assert v3.config.plane_size == 24
    assert v4.config.plane_size == 32
    assert _count(v4) > _count(v3)


def test_v4_query_path_is_signed_and_differentiable():
    cfg = SceneFirstSignedGeometryConfigV4(
        image_token_dim=8,
        camera_feature_dim=13,
        scene_dim=32,
        camera_embed_dim=16,
        plane_size=4,
        plane_channels=8,
        field_hidden_dim=16,
        field_residual_blocks=4,
        attention_heads=4,
        scene_layers=1,
        feedforward_multiplier=2,
        dropout=0.0,
    )
    torch.manual_seed(4)
    model = SceneFirstSignedGeometryV4(cfg)
    image = torch.randn(1, 2, 5, 8)
    xy = torch.randn(1, 2, 5, 2).clamp(-1, 1)
    camera = torch.randn(1, 2, 13)
    points = torch.randn(1, 17, 3).clamp(-1, 1)
    out = model(image, xy, camera, points)
    assert out["scene_planes"].shape == (1, 3, 8, 8, 8)
    assert out["sdf"].shape == (1, 17)
    assert out["log_uncertainty"].shape == (1, 17)
    out["sdf"].square().mean().backward()
    grad = model.surface_field.sdf_head.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(torch.linalg.vector_norm(grad)) > 0.0
