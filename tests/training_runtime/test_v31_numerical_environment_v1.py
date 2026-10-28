from __future__ import annotations

import unittest

from models.iris.v31.numerical_environment_v1 import (
    SCHEMA_V1,
    numerical_environment_sha256_v1,
    numerical_environment_v1,
)


class V31NumericalEnvironmentTests(unittest.TestCase):
    def test_cpu_identity_is_versioned_and_stable_shape(self):
        value = numerical_environment_v1("cpu")
        self.assertEqual(value["schema"], SCHEMA_V1)
        self.assertEqual(value["device_type"], "cpu")
        self.assertIsNone(value["device_index"])
        self.assertIsNone(value["device_name"])
        self.assertIsNone(value["device_capability"])
        self.assertEqual(value["authority_class"], "NUMERICAL_LINEAGE_ONLY")
        self.assertTrue(value["torch_version"])
        self.assertTrue(value["torchvision_version"])
        digest = numerical_environment_sha256_v1("cpu")
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, numerical_environment_sha256_v1("cpu"))


if __name__ == "__main__":
    unittest.main()
