from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from tools.ops.inspect_fit8_morphology_scale_census_v1 import (
    _analytic_observability,
    mask_run_lengths,
    nearest_rank_quantiles,
    tessa_units_per_pixel,
)


class FIT8MorphologyScaleCensusTests(unittest.TestCase):
    def test_run_measurement_preserves_thin_foreground_and_internal_gap(self):
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:5, 1:3] = True
        mask[1:5, 5:7] = True
        foreground, gaps = mask_run_lengths(mask)
        self.assertGreater(len(foreground), 0)
        self.assertGreater(len(gaps), 0)
        self.assertEqual(int(foreground.min()), 2)
        self.assertEqual(int(foreground.max()), 4)
        self.assertTrue(bool(np.all(gaps == 2)))

    def test_tessa_pixel_scale_matches_half_iris_domain(self):
        # Camera spans the common canonical half extent, so IRIS screen width is
        # exactly 2 and TESSA screen width is exactly 1.
        self.assertAlmostEqual(tessa_units_per_pixel(3.0, 3.0, 1024), 1.0 / 1024.0)
        self.assertAlmostEqual(tessa_units_per_pixel(1.5, 3.0, 1024), 0.5 / 1024.0)

    def test_nearest_rank_quantiles_are_deterministic(self):
        values = np.asarray([1.0, 2.0, 3.0, 4.0, 100.0], dtype=np.float64)
        q = nearest_rank_quantiles(values)
        self.assertEqual(q["min"], 1.0)
        self.assertEqual(q["p01"], 1.0)
        self.assertEqual(q["p05"], 1.0)
        self.assertEqual(q["p50"], 3.0)
        self.assertEqual(q["p95"], 100.0)
        self.assertEqual(q["max"], 100.0)

    def test_existing_analytic_table_is_read_without_selecting_resolution(self):
        repo_root = Path(__file__).resolve().parents[2]
        result = _analytic_observability(repo_root)
        self.assertEqual(result["status"], "EXTRACTION_OBSERVABILITY_TABLE_ESTABLISHED_V1")
        self.assertEqual(result["resolutions"], [24, 32, 48, 64])
        self.assertFalse(result["selection"]["production_resolution_selected"])
        self.assertIn("cylinder_diameter", result["table"]["24"])
        self.assertIn("parallel_sheet_gap", result["table"]["64"])

    def test_empty_or_invalid_masks_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "FIT8_MORPHOLOGY_MASK_INVALID"):
            mask_run_lengths(np.zeros((4, 4), dtype=bool))
        with self.assertRaisesRegex(ValueError, "FIT8_MORPHOLOGY_CAMERA_SCALE_NONPOSITIVE"):
            tessa_units_per_pixel(0.0, 1.0, 1024)


if __name__ == "__main__":
    unittest.main()
