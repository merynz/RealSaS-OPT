from __future__ import annotations

import inspect
import unittest

import torch

from experiments.cleanroom_geometry_v1.geometry_field_v1 import (
    CleanroomGeometryFieldV1,
    GeometryFieldConfigV1,
    pack_orthographic_camera_features_v1,
)


def _fixture(batch: int = 2, tokens: int = 12, channels: int = 32):
    views = 8
    image_tokens = torch.randn(batch, views, tokens, channels)
    token_xy = torch.rand(batch, views, tokens, 2) * 2.0 - 1.0
    angles = torch.arange(views, dtype=torch.float32) * (2.0 * torch.pi / views)
    origin = torch.zeros(batch, views, 3)
    forward = torch.stack([torch.sin(angles), torch.cos(angles), torch.zeros_like(angles)], dim=-1)[None].expand(batch, -1, -1)
    right = torch.stack([torch.cos(angles), -torch.sin(angles), torch.zeros_like(angles)], dim=-1)[None].expand(batch, -1, -1)
    up = torch.tensor([0.0, 0.0, 1.0])[None, None].expand(batch, views, -1)
    camera = pack_orthographic_camera_features_v1(origin, right, up, forward, torch.ones(batch, views))
    points = torch.rand(batch, 25, 3) * 2.0 - 1.0
    return image_tokens, token_xy, camera, points


class CleanroomGeometryFieldV1Tests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(7)
        cfg = GeometryFieldConfigV1(
            image_token_dim=32,
            scene_dim=64,
            camera_embed_dim=32,
            num_heads=8,
            num_layers=2,
            plane_size=8,
            plane_channels=16,
            field_hidden_dim=48,
        )
        return CleanroomGeometryFieldV1(cfg).eval()

    def test_continuous_query_contract(self):
        torch.manual_seed(11)
        model = self._model()
        image_tokens, token_xy, camera, points = _fixture()
        with torch.no_grad():
            out = model(image_tokens, token_xy, camera, points)
        self.assertEqual(tuple(out.triplanes.shape), (2, 3, 16, 16, 16))
        self.assertEqual(tuple(out.field.unsigned_distance.shape), (2, 25, 1))
        self.assertEqual(tuple(out.field.log_uncertainty.shape), (2, 25, 1))
        self.assertTrue(torch.isfinite(out.field.unsigned_distance).all())
        self.assertTrue((out.field.unsigned_distance > 0).all())

    def test_view_permutation_equivariance(self):
        torch.manual_seed(13)
        model = self._model()
        image_tokens, token_xy, camera, points = _fixture(batch=1)
        perm = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4])
        with torch.no_grad():
            a = model(image_tokens, token_xy, camera, points)
            b = model(image_tokens[:, perm], token_xy[:, perm], camera[:, perm], points)
        self.assertLess(float((a.triplanes - b.triplanes).abs().max()), 3e-6)
        self.assertLess(float((a.field.unsigned_distance - b.field.unsigned_distance).abs().max()), 3e-6)

    def test_camera_pack_requires_exact_eight_views(self):
        origin = right = up = forward = torch.zeros(1, 7, 3)
        with self.assertRaises(ValueError):
            pack_orthographic_camera_features_v1(origin, right, up, forward, torch.ones(1, 7))

    def test_experimental_module_has_no_mesh_export_dependency(self):
        import experiments.cleanroom_geometry_v1.geometry_field_v1 as module
        source = inspect.getsource(module)
        for forbidden in ("nvdiffrast", "MarchingTetrahedra", "DMTet", "exporter_cls", "view_embedding"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
