from __future__ import annotations

import unittest

import torch

from models.iris.v31.alpha_resampling_v1 import (
    POLICY_V1,
    policy_sha256_v1,
    resample_source_alpha_v1,
)


EXPECTED_POLICY_SHA256 = "82ec820bfb4ac3217b6752c426d1d0b3ea14edc62cc0370b5800ad5dec027532"


class V31AlphaResamplingPolicyTests(unittest.TestCase):
    def test_policy_identity_is_exact_and_non_authorizing(self):
        self.assertEqual(policy_sha256_v1(), EXPECTED_POLICY_SHA256)
        self.assertEqual(POLICY_V1["mode"], "BILINEAR")
        self.assertTrue(POLICY_V1["antialias"])
        self.assertFalse(POLICY_V1["align_corners"])
        self.assertFalse(POLICY_V1["clamp"])
        self.assertEqual(POLICY_V1["court"]["selected_policy"], "BILINEAR_ANTIALIAS")
        self.assertEqual(POLICY_V1["court"]["status"], "V31_ALPHA_RESAMPLING_POLICY_SELECTED_BOUNDED_V1")
        self.assertFalse(POLICY_V1["claims"]["training_authorized"])
        self.assertFalse(POLICY_V1["claims"]["model_release_minted"])
        self.assertFalse(POLICY_V1["claims"]["product_shape_activated"])

    def test_uint8_alpha_is_cpu_float32_bounded_and_exact_shape(self):
        source = torch.zeros(1, 2, 1, 1024, 1024, dtype=torch.uint8)
        source[:, :, :, :, 511] = 255
        target = resample_source_alpha_v1(source)
        self.assertEqual(tuple(target.shape), (1, 2, 1, 518, 518))
        self.assertEqual(target.dtype, torch.float32)
        self.assertEqual(target.device.type, "cpu")
        self.assertGreater(float(target.max()), 0.0)
        self.assertGreaterEqual(float(target.min()), 0.0)
        self.assertLessEqual(float(target.max()), 1.0)

    def test_one_pixel_support_survives_multiple_phases(self):
        for offset in (0, 1, 2, 3, 17, 31, 127, 255, 511, 700, 1000):
            source = torch.zeros(1, 1, 1, 1024, 1024, dtype=torch.float32)
            source[:, :, :, :, offset] = 1.0
            target = resample_source_alpha_v1(source)
            self.assertGreater(float(target.max()), 1.0e-8, msg=f"offset={offset}")
            self.assertLessEqual(abs(float(target.mean()) - float(source.mean())), 2.5e-4)

    def test_out_of_range_source_is_rejected_not_clamped(self):
        source = torch.zeros(1, 1, 1, 1024, 1024, dtype=torch.float32)
        source[..., 0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "IRIS_V31_ALPHA_SOURCE_RANGE_INVALID"):
            resample_source_alpha_v1(source)


if __name__ == "__main__":
    unittest.main()
