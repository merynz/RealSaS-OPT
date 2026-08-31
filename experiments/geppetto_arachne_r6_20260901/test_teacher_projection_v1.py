from __future__ import annotations

import unittest

import numpy as np

from teacher_projection_v1 import (
    bfs_training_order_v1,
    control_skin_column_map_v1,
    project_skeleton_teacher_v1,
)


def _tails(heads: np.ndarray) -> np.ndarray:
    return heads + np.asarray([0.0, 0.0, 0.25], dtype=np.float32)


def _depths(projection) -> dict[str, int]:
    by_id = projection.control_by_id()
    out: dict[str, int] = {}
    for control_id in by_id:
        depth = 0
        current = by_id[control_id]
        seen = set()
        while current.parent_control_id is not None:
            if current.control_id in seen:
                raise AssertionError("cycle in projected controls")
            seen.add(current.control_id)
            depth += 1
            current = by_id[current.parent_control_id]
        out[control_id] = depth
    return out


class TeacherProjectionR6Tests(unittest.TestCase):
    def test_helper_chain_skips_to_nearest_deform_ancestor(self):
        heads = np.asarray(
            [[0, 0, 0], [0, 0, .25], [0, 0, .5], [0, 0, .75], [0, 0, 1.0]],
            dtype=np.float32,
        )
        projection = project_skeleton_teacher_v1(
            heads,
            _tails(heads),
            np.asarray([-1, 0, 1, 2, 3]),
            np.asarray([1, 0, 0, 1, 1], dtype=bool),
        )
        by_id = projection.control_by_id()
        self.assertEqual(tuple(by_id), ("TC:00000", "TC:00003", "TC:00004"))
        self.assertEqual(by_id["TC:00003"].parent_control_id, "TC:00000")
        self.assertEqual(by_id["TC:00004"].parent_control_id, "TC:00003")
        self.assertEqual(projection.skipped_helper_count, 2)

    def test_multiple_roots_are_preserved_without_super_root(self):
        heads = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float32)
        projection = project_skeleton_teacher_v1(
            heads,
            _tails(heads),
            np.asarray([-1, -1, 0, 1]),
            np.asarray([1, 1, 1, 1], dtype=bool),
        )
        self.assertEqual(projection.root_control_ids, ("TC:00000", "TC:00001"))
        self.assertTrue(projection.metadata["multi_root_preserved"])
        self.assertFalse(projection.metadata["canonical_ids_created"])

    def test_arbitrary_parent_cycle_fails_closed(self):
        heads = np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "parent cycle detected"):
            project_skeleton_teacher_v1(
                heads,
                _tails(heads),
                np.asarray([1, 2, 0]),
                np.asarray([1, 0, 1], dtype=bool),
            )

    def test_deterministic_bfs_is_complete_and_duplicate_free(self):
        heads = np.asarray(
            [[0, 0, 0], [-1, 0, 1], [1, 0, 1], [-1, 0, 2], [1, 0, 2]],
            dtype=np.float32,
        )
        projection = project_skeleton_teacher_v1(
            heads,
            _tails(heads),
            np.asarray([-1, 0, 0, 1, 2]),
            np.ones(5, dtype=bool),
        )
        self.assertEqual(projection.bfs_control_ids, ("TC:00000", "TC:00001", "TC:00002", "TC:00003", "TC:00004"))
        self.assertEqual(len(set(projection.bfs_control_ids)), projection.deform_control_count)

    def test_sibling_randomization_preserves_control_set_and_depth(self):
        heads = np.asarray(
            [[0, 0, 0], [-1, 0, 1], [1, 0, 1], [-2, 0, 2], [2, 0, 2], [0, 0, 2]],
            dtype=np.float32,
        )
        projection = project_skeleton_teacher_v1(
            heads,
            _tails(heads),
            np.asarray([-1, 0, 0, 1, 2, 1]),
            np.ones(6, dtype=bool),
        )
        base_depths = _depths(projection)
        randomized = bfs_training_order_v1(projection, sibling_seed=20260901)
        self.assertEqual(set(randomized), set(projection.bfs_control_ids))
        depth_sequence = [base_depths[control_id] for control_id in randomized]
        self.assertEqual(depth_sequence, sorted(depth_sequence))

    def test_skin_column_provenance_is_exact_source_index(self):
        heads = np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=np.float32)
        projection = project_skeleton_teacher_v1(
            heads,
            _tails(heads),
            np.asarray([-1, 0, 1, 2]),
            np.asarray([1, 0, 1, 1], dtype=bool),
        )
        self.assertEqual(
            control_skin_column_map_v1(projection),
            {"TC:00000": 0, "TC:00002": 2, "TC:00003": 3},
        )
        self.assertTrue(all(not control_id.startswith("J:") for control_id in projection.control_by_id()))


if __name__ == "__main__":
    unittest.main()
