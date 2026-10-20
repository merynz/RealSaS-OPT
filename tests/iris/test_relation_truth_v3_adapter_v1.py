import unittest

import numpy as np

from models.iris.vnext.relation_truth_v3_adapter_v1 import all_hit_pair_truth_v3, all_hit_stack_visibility_truth_v3


class RelationTruthV3AdapterTests(unittest.TestCase):
    def test_explicit_stack_validity_owns_visibility(self):
        arrays = {
            "stack_offsets": np.asarray([0, 3], np.int64),
            "stack_face": np.asarray([10, 20, 30], np.int64),
            "stack_depth": np.asarray([1.0, 1.0, 3.0], np.float64),
            "stack_visibility_probability": np.asarray([0.0, 0.0, 0.0], np.float32),
            "stack_visibility_valid": np.asarray([False, False, True], bool),
            "stack_tied_minimum": np.asarray([True, True, False], bool),
            "face_a": np.asarray([20, 30], np.int64), "face_b": np.asarray([30, 20], np.int64),
            "front_probability": np.asarray([1.0, 0.0], np.float32), "front_valid": np.asarray([True, True], bool),
            "occlusion_positive_support": np.asarray([1.0, 1.0], np.float32),
        }
        stack = all_hit_stack_visibility_truth_v3(arrays, view_index=4, provenance_prefix="unit")
        values, valid = stack[0].training_maps()
        self.assertEqual(values, (0.0, 0.0, 0.0)); self.assertEqual(valid, (False, False, True))
        pairs = all_hit_pair_truth_v3(arrays, view_index=4, provenance_prefix="unit")
        values, validity = pairs[0].training_maps()
        self.assertEqual(values["front_probability"], 1.0); self.assertTrue(validity["front_probability"])
        self.assertTrue(validity["occlusion_probability"]); self.assertFalse(validity["same_drawable_affinity"])

    def test_equal_depth_front_abstains(self):
        arrays = {
            "face_a": np.asarray([1, 2]), "face_b": np.asarray([2, 1]),
            "front_probability": np.asarray([0.0, 0.0], np.float32), "front_valid": np.asarray([False, False]),
            "occlusion_positive_support": np.asarray([1.0, 1.0], np.float32),
        }
        rows = all_hit_pair_truth_v3(arrays, view_index=0, provenance_prefix="tie")
        for row in rows:
            values, valid = row.training_maps(); self.assertEqual(values["front_probability"], 0.0); self.assertFalse(valid["front_probability"])

    def test_v1_style_arrays_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "front_valid"):
            all_hit_pair_truth_v3({"face_a":np.asarray([1]),"face_b":np.asarray([2]),"front_probability":np.asarray([1.]),"occlusion_positive_support":np.asarray([1.])}, view_index=0, provenance_prefix="old")
        with self.assertRaisesRegex(ValueError, "stack_visibility_probability"):
            all_hit_stack_visibility_truth_v3({"stack_offsets":np.asarray([0,2]),"stack_face":np.asarray([1,2]),"stack_depth":np.asarray([1.,2.])}, view_index=0, provenance_prefix="old")

    def test_invalid_values_must_be_zero(self):
        arrays={"face_a":np.asarray([1]),"face_b":np.asarray([2]),"front_probability":np.asarray([1.]),"front_valid":np.asarray([False]),"occlusion_positive_support":np.asarray([1.])}
        with self.assertRaisesRegex(ValueError, "INVALID_FRONT_NONZERO"):
            all_hit_pair_truth_v3(arrays, view_index=0, provenance_prefix="bad")


if __name__ == "__main__": unittest.main()
