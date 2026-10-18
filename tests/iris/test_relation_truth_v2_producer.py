import unittest

import numpy as np

from models.iris.vnext.graphics_truth_v2 import (
    STATIC_RELATION_KEYS_V2,
    VIEW_RELATION_KEYS_V2,
)
from tools.iris_vnext.build_fit8_relation_truth_v2 import (
    _binary_stats,
    _empty_static_targets,
    _orient_two_hit_rows_v2,
)


class RelationTruthV2ProducerTests(unittest.TestCase):
    def test_static_targets_abstain_without_independent_authority(self):
        values, validity = _empty_static_targets(7)
        self.assertEqual(set(values), set(STATIC_RELATION_KEYS_V2))
        self.assertEqual(set(validity), set(STATIC_RELATION_KEYS_V2))
        for name in STATIC_RELATION_KEYS_V2:
            self.assertTrue(np.array_equal(values[name], np.zeros(7, dtype=np.float32)))
            self.assertFalse(validity[name].any())

    def test_two_hit_rows_are_emitted_in_both_orientations(self):
        front = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        back = np.asarray([[1.5, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np.float32)
        row = _orient_two_hit_rows_v2(
            point_front=front,
            point_back=back,
            alpha_boundary=np.asarray([1.0, 0.0], dtype=np.float32),
            appearance_boundary=np.asarray([0.0, 1.0], dtype=np.float32),
            max_oriented_pairs=4,
        )
        self.assertEqual(row["point_a"].shape, (4, 3))
        self.assertTrue(np.array_equal(row["point_a"][0], front[0]))
        self.assertTrue(np.array_equal(row["point_b"][0], back[0]))
        self.assertTrue(np.array_equal(row["point_a"][1], back[0]))
        self.assertTrue(np.array_equal(row["point_b"][1], front[0]))
        self.assertTrue(np.array_equal(row["front_probability"], [1.0, 0.0, 1.0, 0.0]))
        self.assertTrue(np.array_equal(row["occlusion_probability"], [1.0, 0.0, 1.0, 0.0]))
        self.assertTrue(np.array_equal(row["visibility_probability_a"], [1.0, 0.0, 1.0, 0.0]))
        self.assertTrue(np.array_equal(row["visibility_probability_b"], [0.0, 1.0, 0.0, 1.0]))

    def test_projected_boundaries_repeat_for_pair_orientation(self):
        row = _orient_two_hit_rows_v2(
            point_front=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            point_back=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            alpha_boundary=np.asarray([1.0], dtype=np.float32),
            appearance_boundary=np.asarray([0.0], dtype=np.float32),
            max_oriented_pairs=2,
        )
        self.assertTrue(
            np.array_equal(row["projected_alpha_boundary_probability"], [1.0, 1.0])
        )
        self.assertTrue(
            np.array_equal(row["projected_appearance_boundary_probability"], [0.0, 0.0])
        )
        for name in VIEW_RELATION_KEYS_V2:
            self.assertTrue(row[f"{name}__valid"].all())

    def test_directional_class_balance_is_exact(self):
        n = 5
        row = _orient_two_hit_rows_v2(
            point_front=np.zeros((n, 3), dtype=np.float32),
            point_back=np.ones((n, 3), dtype=np.float32),
            alpha_boundary=np.zeros(n, dtype=np.float32),
            appearance_boundary=np.zeros(n, dtype=np.float32),
            max_oriented_pairs=10,
        )
        for name in (
            "front_probability",
            "occlusion_probability",
            "visibility_probability_a",
            "visibility_probability_b",
        ):
            stats = _binary_stats(row[name], row[f"{name}__valid"])
            self.assertEqual(stats["valid"], 10)
            self.assertEqual(stats["positive"], 5)
            self.assertEqual(stats["negative"], 5)
            self.assertEqual(stats["positive_fraction"], 0.5)

    def test_odd_pair_budget_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "BUDGET_MUST_BE_POSITIVE_EVEN"):
            _orient_two_hit_rows_v2(
                point_front=np.zeros((1, 3), dtype=np.float32),
                point_back=np.ones((1, 3), dtype=np.float32),
                alpha_boundary=np.zeros(1, dtype=np.float32),
                appearance_boundary=np.zeros(1, dtype=np.float32),
                max_oriented_pairs=3,
            )

    def test_empty_two_hit_support_is_explicit_not_fabricated(self):
        row = _orient_two_hit_rows_v2(
            point_front=np.zeros((0, 3), dtype=np.float32),
            point_back=np.zeros((0, 3), dtype=np.float32),
            alpha_boundary=np.zeros(0, dtype=np.float32),
            appearance_boundary=np.zeros(0, dtype=np.float32),
            max_oriented_pairs=2,
        )
        self.assertEqual(len(row["point_a"]), 0)
        for name in VIEW_RELATION_KEYS_V2:
            self.assertEqual(len(row[name]), 0)
            self.assertEqual(len(row[f"{name}__valid"]), 0)


if __name__ == "__main__":
    unittest.main()
