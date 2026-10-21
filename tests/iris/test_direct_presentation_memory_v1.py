import unittest

import numpy as np
import torch

from models.iris.vnext.presentation_evidence_v1 import PRESENTATION_FEATURE_NAMES_V1
from models.iris.vnext.presentation_observations_v1 import (
    SOURCE_OBSERVATION_PRESENTATION_PROVENANCE_V1,
    surface_presentation_from_source_observations_v1,
)
from tools.iris_vnext.build_fit8_direct_presentation_memory_v1 import (
    _pair_part,
    _ray_part,
    _stack_visibility,
    _surface_part,
)
from tools.iris_vnext.build_fit8_direct_surface_presentation_queries_v1 import parser as query_parser


class DirectPresentationMemoryV1Tests(unittest.TestCase):
    def _view(self):
        # ray0: unique single hit; ray1: unique minimum + hidden hit;
        # ray2: tied minima, so front/visibility must abstain there.
        point = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.1],
                [0.1, 0.0, 0.2],
                [-0.1, 0.0, 0.1],
                [-0.1, 0.0, 0.1],
            ],
            dtype=np.float32,
        )
        return {
            "camera_features": np.asarray(
                [0.0, 0.0, -2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            "stack_offsets": np.asarray([0, 1, 3, 5], dtype=np.int64),
            "stack_depth": np.asarray([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float64),
            "stack_point_iris": point,
            "source_rgba": np.asarray(
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            "source_alpha_boundary": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
            "source_appearance_boundary": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "source_support": np.ones(3, dtype=np.float32),
            "pair_point_a_iris": np.asarray([point[1], point[2], point[3], point[4]], dtype=np.float32),
            "pair_point_b_iris": np.asarray([point[2], point[1], point[4], point[3]], dtype=np.float32),
            "pair_rank_a": np.asarray([0, 1, 0, 1], dtype=np.int16),
            "pair_rank_b": np.asarray([1, 0, 1, 0], dtype=np.int16),
            "pair_ray_index": np.asarray([1, 1, 2, 2], dtype=np.int32),
            "pair_base_pair_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        }

    def test_source_observation_provenance_forbids_teacher_geometry(self):
        self.assertFalse(SOURCE_OBSERVATION_PRESENTATION_PROVENANCE_V1["teacher_geometry_allowed"])
        self.assertFalse(SOURCE_OBSERVATION_PRESENTATION_PROVENANCE_V1["teacher_topology_allowed"])
        parser_dests = {action.dest for action in query_parser()._actions}
        self.assertNotIn("teacher_dir", parser_dests)
        self.assertNotIn("teacher_mesh", parser_dests)
        self.assertNotIn("teacher_topology", parser_dests)

    def test_stack_visibility_is_tie_safe(self):
        p, valid = _stack_visibility(np.asarray([1.0, 1.0, 2.0]), 1.0e-10)
        self.assertEqual(p.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(valid.tolist(), [False, False, True])
        p2, valid2 = _stack_visibility(np.asarray([1.0, 2.0, 3.0]), 1.0e-10)
        self.assertEqual(p2.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(valid2.tolist(), [True, True, True])

    def test_surface_source_observations_require_unique_model_front(self):
        part = _surface_part(self._view(), subject_id="Snake", view_index=0, epsilon=1.0e-10)
        self.assertIsNotNone(part)
        self.assertEqual(part.values.shape[1], 2)
        source_valid = PRESENTATION_FEATURE_NAMES_V1.index("source_rgba_valid")
        alpha_valid = PRESENTATION_FEATURE_NAMES_V1.index("alpha_boundary_valid")
        support_valid = PRESENTATION_FEATURE_NAMES_V1.index("source_support_valid")
        self.assertTrue(bool((part.values[..., source_valid] == 1.0).all()))
        self.assertTrue(bool((part.values[..., alpha_valid] == 1.0).all()))
        self.assertTrue(bool((part.values[..., support_valid] == 1.0).all()))

    def test_pair_front_drops_equal_depth_and_keeps_both_orientations(self):
        part = _pair_part(self._view(), subject_id="Snake", view_index=0, epsilon=1.0e-10)
        self.assertIsNotNone(part)
        self.assertEqual(part.values.shape[1], 2)
        front = PRESENTATION_FEATURE_NAMES_V1.index("front_probability")
        front_valid = PRESENTATION_FEATURE_NAMES_V1.index("front_valid")
        self.assertEqual(sorted(part.values[0, :, front].tolist()), [0.0, 1.0])
        self.assertTrue(bool((part.values[..., front_valid] == 1.0).all()))

    def test_ray_tokens_preserve_complete_stack_tied_abstain(self):
        part = _ray_part(self._view(), subject_id="Snake", view_index=0, epsilon=1.0e-10)
        self.assertIsNotNone(part)
        self.assertEqual(part.values.shape[1], 5)
        vis = PRESENTATION_FEATURE_NAMES_V1.index("global_visibility")
        vis_valid = PRESENTATION_FEATURE_NAMES_V1.index("global_visibility_valid")
        self.assertEqual(int((part.values[..., vis] == 1.0).sum()), 2)
        self.assertEqual(int((part.values[..., vis_valid] == 0.0).sum()), 2)
        self.assertTrue(bool(part.valid_mask.all()))

    def test_source_builder_does_not_upgrade_unknown_fields(self):
        xyz = torch.zeros(1, 1, 3)
        camera = torch.tensor(
            [[[0.0, 0.0, -2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]]]
        )
        valid = torch.ones(1, 1, dtype=torch.bool)
        evidence = surface_presentation_from_source_observations_v1(
            xyz=xyz,
            camera_features=camera,
            token_valid_mask=valid,
            source_rgba=torch.ones(1, 1, 4),
            source_rgba_valid=valid,
            alpha_boundary=torch.ones(1, 1),
            appearance_boundary=torch.zeros(1, 1),
            boundary_valid=valid,
            source_support=torch.ones(1, 1),
            source_support_valid=valid,
        )
        for name in (
            "front_valid",
            "occlusion_valid",
            "global_visibility_valid",
            "deformation_coherence_valid",
            "attachment_persistence_valid",
        ):
            self.assertEqual(float(evidence.values[..., PRESENTATION_FEATURE_NAMES_V1.index(name)]), 0.0)


if __name__ == "__main__":
    unittest.main()
