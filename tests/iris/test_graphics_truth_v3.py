import unittest

from models.iris.vnext.graphics_truth_v2 import QualifiedProbabilityTargetV2
from models.iris.vnext.graphics_truth_v3 import (
    PAIR_KEYS_V3,
    GraphicsPairTruthRowV3,
    RayStackVisibilityTruthV1,
)


class GraphicsTruthV3Tests(unittest.TestCase):
    def test_pair_truth_has_no_visibility_family(self):
        self.assertNotIn("visibility_probability_a", PAIR_KEYS_V3)
        self.assertNotIn("visibility_probability_b", PAIR_KEYS_V3)
        self.assertNotIn("global_visibility", PAIR_KEYS_V3)

    def test_complete_three_hit_stack_marks_only_frontmost_visible(self):
        truth = RayStackVisibilityTruthV1.from_complete_depth_order(
            view_index=2,
            ray_index=11,
            face_ids=(10, 20, 30),
            depths=(1.0, 2.0, 3.0),
            provenance="unit:complete-all-hit",
        )
        values, valid = truth.training_maps()
        self.assertEqual(values, (1.0, 0.0, 0.0))
        self.assertEqual(valid, (True, True, True))
        self.assertTrue(truth.complete_stack)

    def test_incomplete_stack_must_abstain(self):
        truth = RayStackVisibilityTruthV1.abstain_incomplete(
            view_index=0,
            ray_index=0,
            face_ids=(1, 2),
            depths=(1.0, 2.0),
            provenance="unit:first-second-only",
        )
        _, valid = truth.training_maps()
        self.assertEqual(valid, (False, False))
        self.assertFalse(truth.complete_stack)

        with self.assertRaisesRegex(ValueError, "INCOMPLETE_STACK_CANNOT_TRAIN"):
            RayStackVisibilityTruthV1(
                view_index=0,
                ray_index=0,
                face_ids=(1, 2),
                depths=(1.0, 2.0),
                visibility=(
                    QualifiedProbabilityTargetV2.abstain(provenance="unit:a"),
                    RayStackVisibilityTruthV1.from_complete_depth_order(
                        view_index=0,
                        ray_index=0,
                        face_ids=(1,),
                        depths=(1.0,),
                        provenance="unit:b",
                    ).visibility[0],
                ),
                complete_stack=False,
                provenance="unit:mixed-invalid",
            )

    def test_pair_row_preserves_unknown_as_invalid(self):
        unknown = QualifiedProbabilityTargetV2.abstain(provenance="unit:unknown")
        row = GraphicsPairTruthRowV3(
            view_index=0,
            face_a=1,
            face_b=2,
            same_drawable_affinity=unknown,
            cut_probability=unknown,
            attachment_affinity=unknown,
            front_probability=unknown,
            occlusion_probability=unknown,
            projected_alpha_boundary_probability=unknown,
            projected_appearance_boundary_probability=unknown,
        )
        _, validity = row.training_maps()
        self.assertTrue(all(value is False for value in validity.values()))


if __name__ == "__main__":
    unittest.main()
