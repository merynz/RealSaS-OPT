from pathlib import Path

import numpy as np
import pytest
import torch

from models.iris.v3.scene_first_signed_v3 import SceneFirstSignedGeometryConfigV3, SceneFirstSignedGeometryV3
from models.iris.v3.zero_surface_decoder_v3 import extract_zero_surface_mesh_v3


def _tiny_model():
    cfg = SceneFirstSignedGeometryConfigV3(
        image_token_dim=8,
        camera_feature_dim=13,
        scene_dim=24,
        camera_embed_dim=12,
        plane_size=4,
        plane_channels=6,
        field_hidden_dim=16,
        attention_heads=4,
        scene_layers=1,
        feedforward_multiplier=2,
        dropout=0.0,
    )
    torch.manual_seed(7)
    return SceneFirstSignedGeometryV3(cfg).eval()


def test_joint_view_camera_permutation_is_scene_invariant():
    model = _tiny_model()
    torch.manual_seed(11)
    image = torch.randn(1, 4, 9, 8)
    xy = torch.randn(1, 4, 9, 2).clamp(-1, 1)
    camera = torch.randn(1, 4, 13)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        a = model.encode_scene(image, xy, camera)
        b = model.encode_scene(image[:, perm], xy[:, perm], camera[:, perm])
    assert torch.max(torch.abs(a - b)).item() < 2e-5


def test_signed_head_is_not_unsigned_clamped():
    model = _tiny_model()
    assert not any(isinstance(m, torch.nn.Softplus) for m in model.surface_field.modules())
    assert model.decoder_policy == "SIGNED_FIELD_ZERO_LEVEL_SURFACE"
    image = torch.randn(1, 2, 4, 8)
    xy = torch.randn(1, 2, 4, 2).clamp(-1, 1)
    camera = torch.randn(1, 2, 13)
    points = torch.randn(1, 17, 3).clamp(-1, 1)
    out = model(image, xy, camera, points)
    assert out["sdf"].shape == (1, 17)
    assert out["log_uncertainty"].shape == (1, 17)


def test_zero_surface_decoder_extracts_sphere():
    pytest.importorskip("skimage")
    r = 32
    a = np.linspace(-1.0, 1.0, r, dtype=np.float32)
    z, y, x = np.meshgrid(a, a, a, indexing="ij")
    field = np.sqrt(x * x + y * y + z * z) - 0.55
    mesh = extract_zero_surface_mesh_v3(field)
    radius = np.linalg.norm(mesh.vertices_normalized, axis=1)
    assert len(mesh.vertices_normalized) > 500
    assert float(np.quantile(np.abs(radius - 0.55), 0.95)) < 0.04
    assert mesh.field_min < 0 < mesh.field_max


def test_module_has_no_charactergen_runtime_dependency():
    source = Path(__file__).parents[2] / "models" / "iris" / "v3" / "scene_first_signed_v3.py"
    text = source.read_text(encoding="utf-8").lower()
    assert "charactergen" not in text
