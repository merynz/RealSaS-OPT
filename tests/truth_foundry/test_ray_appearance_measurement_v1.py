import unittest

import numpy as np

from models.iris.truth.ray_appearance_measurement_v1 import (
    ABSTAIN_FOOTPRINT,
    ABSTAIN_INCOMPLETE,
    ABSTAIN_TIE,
    FAIL_REPLAY,
    PASS,
    qualify_measured_ray_appearance_v1,
)
from models.iris.truth.ray_compositing_v1 import measure_ray


class RayAppearanceMeasurementTests(unittest.TestCase):
    def _measured_fixture(self, depth, alpha, straight):
        depth = np.asarray(depth, np.float64)
        alpha = np.asarray(alpha, np.float64)
        straight = np.asarray(straight, np.float64)
        standalone_premul = alpha[:, None] * straight
        full = measure_ray(depth, alpha, straight, complete=True, tie_epsilon=1e-9)
        removal_alpha = []
        removal_rgb = []
        for index in range(len(depth)):
            keep = np.arange(len(depth)) != index
            row = measure_ray(depth[keep], alpha[keep], straight[keep], complete=True, tie_epsilon=1e-9)
            removal_alpha.append(row.alpha)
            removal_rgb.append(row.premultiplied_rgb)
        return (
            standalone_premul,
            full.alpha,
            full.premultiplied_rgb,
            np.asarray(removal_alpha),
            np.asarray(removal_rgb),
        )

    def test_measured_full_and_removal_interventions_admit_scalar_hit_appearance(self):
        depth = [1.0, 2.0]
        alpha = [.5, .5]
        straight = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        standalone, full_alpha, full_rgb, removal_alpha, removal_rgb = self._measured_fixture(depth, alpha, straight)
        result = qualify_measured_ray_appearance_v1(
            depth, alpha, standalone, full_alpha, full_rgb, removal_alpha, removal_rgb,
            complete=True, tie_epsilon=1e-9, measurement_tolerance=1e-12, footprint_converged=True,
        )
        self.assertEqual(result.status, PASS)
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.effective_alpha, alpha)
        np.testing.assert_allclose(result.straight_linear_rgb, straight)
        np.testing.assert_allclose(result.incoming_transmittance, [1.0, .5])
        np.testing.assert_allclose(result.contribution, [.5, .25])

    def test_correlated_subpixel_coverage_cannot_be_silently_factorized(self):
        # Standalone pixels each report 50% coverage, which a scalar source-over
        # model predicts as 75% total alpha.  A measured 50% full alpha can arise
        # when both surfaces cover the same subpixel samples.  The court must
        # reject this raster footprint rather than invent scalar per-hit truth.
        result = qualify_measured_ray_appearance_v1(
            [1.0, 2.0],
            [.5, .5],
            [[.5, 0.0, 0.0], [0.0, 0.0, .5]],
            .5,
            [.5, 0.0, 0.0],
            [.5, .5],
            [[0.0, 0.0, .5], [.5, 0.0, 0.0]],
            complete=True,
            tie_epsilon=1e-9,
            measurement_tolerance=1e-9,
            footprint_converged=True,
        )
        self.assertEqual(result.status, FAIL_REPLAY)
        self.assertFalse(result.valid)
        self.assertGreater(result.maximum_full_alpha_error, .2)

    def test_same_colour_zero_rgb_removal_delta_does_not_erase_occlusion(self):
        depth = [1.0, 2.0]
        alpha = [1.0, 1.0]
        straight = [[.25, .5, .75], [.25, .5, .75]]
        standalone, full_alpha, full_rgb, removal_alpha, removal_rgb = self._measured_fixture(depth, alpha, straight)
        result = qualify_measured_ray_appearance_v1(
            depth, alpha, standalone, full_alpha, full_rgb, removal_alpha, removal_rgb,
            complete=True, tie_epsilon=1e-9, measurement_tolerance=1e-12, footprint_converged=True,
        )
        self.assertEqual(result.status, PASS)
        np.testing.assert_array_equal(full_rgb, removal_rgb[0])
        self.assertEqual(result.contribution[0], 1.0)
        self.assertEqual(result.incoming_transmittance[1], 0.0)

    def test_incomplete_tie_and_nonconverged_footprint_abstain(self):
        common = dict(
            depth=[1.0, 2.0],
            standalone_alpha=[.5, .5],
            standalone_premultiplied_linear_rgb=[[.5, 0.0, 0.0], [0.0, 0.0, .5]],
            measured_full_alpha=.75,
            measured_full_premultiplied_linear_rgb=[.5, 0.0, .25],
            measured_removal_alpha=[.5, .5],
            measured_removal_premultiplied_linear_rgb=[[0.0, 0.0, .5], [.5, 0.0, 0.0]],
            measurement_tolerance=1e-9,
        )
        incomplete = qualify_measured_ray_appearance_v1(
            **common, complete=False, tie_epsilon=1e-9, footprint_converged=True,
        )
        self.assertEqual(incomplete.status, ABSTAIN_INCOMPLETE)
        tied = qualify_measured_ray_appearance_v1(
            **{**common, "depth": [1.0, 1.0]}, complete=True, tie_epsilon=1e-9, footprint_converged=True,
        )
        self.assertEqual(tied.status, ABSTAIN_TIE)
        unstable = qualify_measured_ray_appearance_v1(
            **common, complete=True, tie_epsilon=1e-9, footprint_converged=False,
        )
        self.assertEqual(unstable.status, ABSTAIN_FOOTPRINT)


if __name__ == "__main__":
    unittest.main()
