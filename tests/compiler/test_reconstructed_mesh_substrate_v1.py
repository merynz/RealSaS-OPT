from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from compiler.realsas_compiler_core.substrate.reconstructed_mesh import (
    load_obj_triangle_mesh,
    rigging_surface_from_obj,
    rigging_surface_from_reconstructed_mesh,
)
from models.geppetto.v2.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2


class ReconstructedMeshSubstrateTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.asarray([
            [-1.0, -1.0, 0.0],
            [ 1.0, -1.0, 0.0],
            [ 1.0,  1.0, 0.0],
            [-1.0,  1.0, 0.0],
            [ 0.0,  0.0, 1.0],
        ], dtype=np.float64)
        self.faces = np.asarray([
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [0, 3, 2],
            [0, 2, 1],
        ], dtype=np.int64)

    def test_complete_mesh_surface_is_deterministic_and_epistemically_typed(self):
        a = rigging_surface_from_reconstructed_mesh(
            self.vertices,
            self.faces,
            sample_count=128,
            source_ref="TEST",
            reconstruction_model="CharacterGen.test",
            reconstruction_checkpoint="ckpt",
        )
        b = rigging_surface_from_reconstructed_mesh(
            self.vertices,
            self.faces,
            sample_count=128,
            source_ref="TEST",
            reconstruction_model="CharacterGen.test",
            reconstruction_checkpoint="ckpt",
        )
        self.assertEqual(a.geometry_lineage_hash, b.geometry_lineage_hash)
        self.assertEqual(len(a.surface_nodes), 128)
        self.assertEqual(a.metadata["source_domain"], "RECONSTRUCTED_COMPLETE_MESH")
        self.assertTrue(a.metadata["full_3d_reconstruction_claim"])
        self.assertFalse(a.metadata["observational_support_authority"])
        self.assertTrue(a.metadata["Nd_operator_sha256"])
        for node in a.surface_nodes:
            self.assertEqual(node.support_views, ())
            self.assertEqual(node.raster_bindings, ())
            self.assertIn("RECONSTRUCTED_COMPLETE_SURFACE", node.validity_flags)
            self.assertIsNotNone(node.derived_normal)
            self.assertAlmostEqual(float(np.linalg.norm(node.derived_normal)), 1.0, places=5)

    def test_geppetto_v2_accepts_reconstructed_mesh_substrate(self):
        surface = rigging_surface_from_reconstructed_mesh(
            self.vertices,
            self.faces,
            sample_count=64,
            source_ref="TEST",
        )
        batch = GeppettoConditioningAdapterV2()((surface,))
        self.assertEqual(batch.features.shape, (1, 64, 24))
        self.assertEqual(batch.positions_normalized.shape, (1, 64, 3))
        self.assertTrue(batch.valid_mask.all())
        self.assertTrue(np.isfinite(batch.features).all())

    def test_obj_loader_triangulates_polygon_and_negative_indices(self):
        text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 1 1 0",
            "v 0 1 0",
            "f -4 -3 -2 -1",
        ]) + "\n"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "quad.obj"
            path.write_text(text, encoding="utf-8")
            v, f = load_obj_triangle_mesh(path)
            self.assertEqual(v.shape, (4, 3))
            self.assertEqual(f.shape, (2, 3))
            surface = rigging_surface_from_obj(path, sample_count=16)
            self.assertEqual(len(surface.surface_nodes), 16)


if __name__ == "__main__":
    unittest.main()
