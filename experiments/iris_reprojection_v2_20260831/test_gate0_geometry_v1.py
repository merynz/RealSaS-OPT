from __future__ import annotations

import unittest

import numpy as np

from gate0_geometry_v1 import (
    SurfaceState,
    backproject_grid_depth,
    hull_frontier,
    make_dense_lattice,
    orbit_camera,
    project_world,
    robust_multiview_evidence,
    streaming_chunk_plan,
    thickness_cell_strata,
    visual_hull_membership,
)


class Gate0GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cams = [orbit_camera(45.0 * v, half_extent=0.54) for v in range(8)]

    def test_projection_backprojection_roundtrip(self):
        rng = np.random.default_rng(20260831)
        pts = rng.uniform(low=[-0.35, -0.35, -0.35], high=[0.35, 0.35, 0.35], size=(128, 3))
        for cam in self.cams:
            grid, depth = project_world(pts, cam)
            rec = backproject_grid_depth(grid, depth, cam)
            self.assertLess(float(np.max(np.abs(rec - pts))), 1e-10)

    def test_world_z_maps_to_identical_image_row_across_orbit(self):
        pts = np.asarray([
            [-0.31, 0.22, -0.25],
            [0.11, -0.17, 0.00],
            [0.27, 0.29, 0.21],
        ], dtype=np.float64)
        rows = []
        for cam in self.cams:
            grid, _ = project_world(pts, cam)
            rows.append(grid[:, 1])
        rows = np.stack(rows, axis=0)
        self.assertTrue(np.allclose(rows, rows[0:1], atol=1e-12, rtol=0))

    def _sphere_masks(self, resolution=128, radius=0.30):
        # Orthographic sphere silhouette is a circle in (screen-right, world-Z) for every yaw.
        yy, xx = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="ij")
        gxx = 2.0 * (xx + 0.5) / resolution - 1.0
        gyy = 2.0 * (yy + 0.5) / resolution - 1.0
        world_x = gxx * 0.54
        world_up = -gyy * 0.54
        m = (world_x * world_x + world_up * world_up) <= radius * radius
        return [m.copy() for _ in range(8)]

    def test_visual_hull_contains_sphere_surface_and_reduces_domain(self):
        rng = np.random.default_rng(7)
        # Truth points on a sphere.
        u = rng.normal(size=(4000, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        truth = 0.30 * u
        lattice, _ = make_dense_lattice(((-0.40, 0.40), (-0.40, 0.40), (-0.40, 0.40)), 0.04)
        masks = self._sphere_masks()
        active = visual_hull_membership(lattice, self.cams, masks, pad_px=1)
        contained = visual_hull_membership(truth, self.cams, masks, pad_px=1)
        self.assertGreaterEqual(float(contained.mean()), 0.99)
        self.assertLess(float(active.mean()), 0.75)

    def test_hull_padding_frontier_is_monotone(self):
        rng = np.random.default_rng(9)
        u = rng.normal(size=(1000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        truth = 0.30 * u
        lattice, _ = make_dense_lattice(((-0.40, 0.40), (-0.40, 0.40), (-0.40, 0.40)), 0.05)
        frontier = hull_frontier(truth, lattice, self.cams, self._sphere_masks(96), paddings_px=(0, 1, 2, 4))
        contain = [x["truth_containment"] for x in frontier]
        active = [x["active_lattice_fraction"] for x in frontier]
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(contain, contain[1:])))
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(active, active[1:])))

    def test_thin_structure_strata_exposes_subcell_cases(self):
        r = thickness_cell_strata(np.asarray([0.002, 0.007, 0.012, 0.020, 0.050]), spacing=0.008)
        self.assertEqual(r["bins"]["lt_1"], 2)
        self.assertEqual(r["bins"]["ge_1_lt_2"], 1)
        self.assertEqual(r["bins"]["ge_2_lt_4"], 1)
        self.assertEqual(r["bins"]["ge_4"], 1)

    def test_robust_evidence_is_view_permutation_invariant(self):
        rng = np.random.default_rng(13)
        base = rng.normal(size=(16,))
        x = np.stack([base + 0.01 * rng.normal(size=(16,)) for _ in range(7)] + [-base], axis=0)
        a = robust_multiview_evidence(x, trim_fraction=0.25)
        perm = rng.permutation(8)
        b = robust_multiview_evidence(x[perm], trim_fraction=0.25)
        self.assertAlmostEqual(a["cosine_trimmed_mean"], b["cosine_trimmed_mean"], places=12)
        self.assertAlmostEqual(a["cosine_median"], b["cosine_median"], places=12)
        self.assertAlmostEqual(a["dispersion"], b["dispersion"], places=12)
        self.assertGreater(a["cosine_trimmed_mean"], 0.95)

    def test_robust_evidence_does_not_require_all_views(self):
        x = np.eye(4, dtype=np.float64)
        valid = np.asarray([True, False, True, False])
        r = robust_multiview_evidence(x, valid)
        self.assertEqual(r["support_views"], 2)
        self.assertEqual(r["pair_count"], 1)

    def test_streaming_plan_respects_budget(self):
        r = streaming_chunk_plan(active_candidates=2_000_000, views=8, descriptor_dim=64, max_working_bytes=64 * 1024 * 1024)
        self.assertLessEqual(r["max_raw_descriptor_working_bytes"], 64 * 1024 * 1024)
        self.assertGreater(r["chunk_count"], 1)

    def test_unknown_states_are_explicit(self):
        self.assertNotEqual(SurfaceState.UNKNOWN_OCCLUDED, SurfaceState.UNKNOWN_UNOBSERVED)
        self.assertNotEqual(SurfaceState.RESOLUTION_UNSUPPORTED, SurfaceState.SURFACE_SUPPORTED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
