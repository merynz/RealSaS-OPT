import unittest

import numpy as np

from models.iris.vnext.relation_truth_v3_adapter_v1 import (
    all_hit_pair_truth_v3,
    all_hit_stack_visibility_truth_v3,
)


class RelationTruthV3AdapterTests(unittest.TestCase):
    def test_complete_stack_not_pair_columns_owns_visibility(self):
        arrays = {
            "stack_offsets": np.asarray([0, 3], dtype=np.int64),
            "stack_face": np.asarray([10, 20, 30], dtype=np.int64),
            "stack_depth": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            "face_a": np.asarray([20, 30], dtype=np.int64),
            "face_b": np.asarray([30, 20], dtype=np.int64),
            "front_probability": np.asarray([1.0, 0.0], dtype=np.float32),
            "occlusion_positive_support": np.asarray([1.0, 1.0], dtype=np.float32),
            # These historical columns are deliberately contradictory; V3 must ignore them.
            "visibility_probability_a": np.asarray([1.0, 1.0], dtype=np.float32),
            "visibility_probability_b": np.asarray([1.0, 1.0], dtype=np.float32),
        }
        stack = all_hit_stack_visibility_truth_v3(
            arrays, view_index=4, provenance_prefix="unit"
        )
        self.assertEqual(len(stack), 1)
        values, valid = stack[0].training_maps()
        self.assertEqual(values, (1.0, 0.0, 0.0))
        self.assertEqual(valid, (True, True, True))

        pairs = all_hit_pair_truth_v3(arrays, view_index=4, provenance_prefix="unit")
        self.assertEqual(len(pairs), 2)
        first_values, first_valid = pairs[0].training_maps()
        self.assertEqual(first_values["front_probability"], 1.0)
        self.assertTrue(first_valid["front_probability"])
        self.assertTrue(first_valid["occlusion_probability"])
        self.assertFalse(first_valid["same_drawable_affinity"])
        self.assertFalse(first_valid["projected_alpha_boundary_probability"])

    def test_empty_rays_are_skipped_but_csr_remains_complete(self):
        arrays = {
            "stack_offsets": np.asarray([0, 0, 2], dtype=np.int64),
            "stack_face": np.asarray([7, 8], dtype=np.int64),
            "stack_depth": np.asarray([0.5, 0.9], dtype=np.float64),
        }
        rows = all_hit_stack_visibility_truth_v3(
            arrays, view_index=0, provenance_prefix="unit"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].ray_index, 1)
        self.assertEqual(rows[0].face_ids, (7, 8))


if __name__ == "__main__":
    unittest.main()
