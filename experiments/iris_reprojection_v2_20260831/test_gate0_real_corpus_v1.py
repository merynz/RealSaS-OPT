from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gate0_real_corpus_v1 import (
    deterministic_truth_row_ids,
    four_neighbor_boundary,
    image_foreground_mask,
    lattice_axis_count,
    normal_ci95,
    observed_medial_diameters_world,
    padded_masks,
    raster_mask_from_pix,
)


class RealCorpusGate0HelperTests(unittest.TestCase):
    def test_foreground_rule_opaque_green(self):
        a = np.zeros((1024, 1024, 4), dtype=np.uint8)
        a[:, :, :3] = np.array([0, 255, 0], np.uint8)
        a[:, :, 3] = 255
        a[100:110, 200:220, :3] = np.array([12, 34, 56], np.uint8)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.png"; Image.fromarray(a, "RGBA").save(p)
            m, mode = image_foreground_mask(p)
        self.assertEqual(mode, "OPAQUE_EXACT_GREEN")
        self.assertEqual(int(m.sum()), 200)

    def test_foreground_rule_alpha_authority(self):
        a = np.zeros((1024, 1024, 4), dtype=np.uint8)
        a[:, :, :3] = 255
        a[:, :, 3] = 0
        a[20:25, 30:40, 3] = 200
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.png"; Image.fromarray(a, "RGBA").save(p)
            m, mode = image_foreground_mask(p)
        self.assertEqual(mode, "ALPHA_THRESHOLD")
        self.assertEqual(int(m.sum()), 50)

    def test_boundary_is_four_neighbor(self):
        m = np.zeros((9, 9), dtype=bool); m[2:7, 2:7] = True
        b = four_neighbor_boundary(m)
        self.assertEqual(int(b.sum()), 16)
        self.assertFalse(bool(b[4, 4]))

    def test_truth_row_sampling_is_deterministic_and_keeps_boundary(self):
        mask = np.zeros((1024, 1024), dtype=bool)
        mask[100:300, 200:500] = True
        pix = np.flatnonzero(mask.reshape(-1)).astype(np.int64)
        a = deterministic_truth_row_ids("asset_x", 3, pix, mask)
        b = deterministic_truth_row_ids("asset_x", 3, pix, mask)
        self.assertTrue(np.array_equal(a, b))
        self.assertGreaterEqual(len(a), 4096)
        boundary = four_neighbor_boundary(mask).reshape(-1)
        selected_boundary = int(boundary[pix[a]].sum())
        self.assertGreater(selected_boundary, 0)

    def test_raster_mask_roundtrip(self):
        pix = np.asarray([0, 1, 1024, 1025, 999999], dtype=np.int64)
        m = raster_mask_from_pix(pix)
        self.assertEqual(int(m.sum()), len(pix))
        self.assertTrue(bool(m.reshape(-1)[999999]))

    def test_medial_proxy_on_five_pixel_bar(self):
        m = np.zeros((1024, 1024), dtype=bool)
        m[100:900, 500:505] = True
        d = observed_medial_diameters_world(m, half_extent=0.54)
        self.assertGreater(len(d), 0)
        px_world = 2 * 0.54 / 1024
        # Five-pixel bar has EDT peak 3 px under the frozen 2*EDT proxy.
        self.assertAlmostEqual(float(d.max()), 6.0 * px_world, places=7)

    def test_padding_is_monotone(self):
        m = np.zeros((32, 32), dtype=bool); m[15, 15] = True
        pm = padded_masks(m)
        counts = [int(pm[p].sum()) for p in (0, 1, 2, 4, 8)]
        self.assertEqual(counts, sorted(counts))

    def test_normal_ci(self):
        lo, hi = normal_ci95(250, 1000)
        self.assertLess(lo, 0.25); self.assertGreater(hi, 0.25)
        self.assertGreaterEqual(lo, 0.0); self.assertLessEqual(hi, 1.0)

    def test_lattice_axis_count_monotone(self):
        a = lattice_axis_count(0.54, 0.016)
        b = lattice_axis_count(0.54, 0.008)
        c = lattice_axis_count(0.54, 0.004)
        self.assertLess(a, b); self.assertLess(b, c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
