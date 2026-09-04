from __future__ import annotations

import math
import unittest

import numpy as np

from compiler.realsas_compiler_core.substrate.complete_mesh import (
    mesh_surface_sampler_hash,
    rigging_surface_from_complete_triangle_mesh,
)
from models.geppetto.v2.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2


class CompleteMeshSubstrateV1(unittest.TestCase):
    def _tetra(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        faces = np.asarray(
            [
                [0, 2, 1],
                [0, 1, 3],
                [0, 3, 2],
                [1, 2, 3],
            ],
            dtype=np.int64,
        )
        return vertices, faces

    def test_deterministic_complete_mesh_surface(self):
        v, f = self._tetra()
        kwargs = dict(
            provenance_ref="UNIT:CHARACTERGEN",
            sample_count=128,
            backend_id="CHARACTERGEN_3D_STAGE_PINNED",
            source_asset_id="tetra",
        )
        a = rigging_surface_from_complete_triangle_mesh(v, f, **kwargs)
        b = rigging_surface_from_complete_triangle_mesh(v, f, **kwargs)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(len(a.surface_nodes), 128)
        self.assertEqual(a.metadata["Nd_operator_sha256"], mesh_surface_sampler_hash())
        self.assertFalse(a.metadata["teacher_truth_used"])
        self.assertFalse(a.metadata["observational_support_claim"])
        self.assertTrue(all(not n.support_views for n in a.surface_nodes))
        for node in a.surface_nodes:
            self.assertIsNotNone(node.derived_normal)
            self.assertAlmostEqual(
                math.sqrt(sum(x * x for x in node.derived_normal)), 1.0, places=6
            )

    def test_geppetto_v2_accepts_complete_mesh_surface(self):
        v, f = self._tetra()
        surface = rigging_surface_from_complete_triangle_mesh(
            v,
            f,
            provenance_ref="UNIT:CHARACTERGEN",
            sample_count=64,
            backend_id="CHARACTERGEN_3D_STAGE_PINNED",
        )
        batch = GeppettoConditioningAdapterV2()((surface,))
        self.assertEqual(batch.features.shape, (1, 64, 24))
        self.assertTrue(np.isfinite(batch.features).all())
        self.assertTrue(batch.valid_mask.all())
        # Completion geometry is not mislabeled as observed support.
        self.assertTrue(np.all(batch.features[0, :, 11:19] == 0.0))

    def test_protected_metadata_cannot_be_overridden(self):
        v, f = self._tetra()
        with self.assertRaises(ValueError):
            rigging_surface_from_complete_triangle_mesh(
                v,
                f,
                provenance_ref="UNIT",
                sample_count=8,
                extra_metadata={"teacher_truth_used": True},
            )

    def test_rejects_degenerate_mesh(self):
        vertices = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        with self.assertRaises(Exception):
            rigging_surface_from_complete_triangle_mesh(
                vertices,
                faces,
                provenance_ref="UNIT",
                sample_count=8,
            )


if __name__ == "__main__":
    unittest.main()
