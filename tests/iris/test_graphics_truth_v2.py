import unittest

from models.iris.vnext.boundary_causality_v1 import EvidenceAuthority
from models.iris.vnext.graphics_truth_v2 import (
    GraphicsStaticTruthRowV2,
    GraphicsViewTruthRowV2,
    QualifiedProbabilityTargetV2,
    assert_required_validity_keys_v2,
    static_row_to_training_maps_v2,
    view_row_to_training_maps_v2,
)


class GraphicsTruthV2CurrentContractTests(unittest.TestCase):
    @staticmethod
    def _valid(value: float, provenance: str) -> QualifiedProbabilityTargetV2:
        return QualifiedProbabilityTargetV2(
            value=value,
            authority=EvidenceAuthority.DETERMINISTIC_DERIVED,
            provenance=provenance,
            valid_for_training=True,
        )

    def test_static_truth_abstains_instead_of_minting_negative(self):
        row = GraphicsStaticTruthRowV2(
            face_a=1,
            face_b=2,
            same_drawable_affinity=QualifiedProbabilityTargetV2.abstain(
                provenance="drawable_owner_not_identifiable"
            ),
            cut_probability=QualifiedProbabilityTargetV2.abstain(
                provenance="cut_not_identifiable"
            ),
            attachment_affinity=QualifiedProbabilityTargetV2.abstain(
                provenance="attachment_not_identifiable"
            ),
        )
        values, validity = static_row_to_training_maps_v2(row)
        self.assertEqual(values["cut_probability"], 0.0)
        self.assertFalse(validity["cut_probability"])
        self.assertNotIn("projected_alpha_boundary_probability", values)

    def test_projected_boundary_truth_is_view_conditioned(self):
        row = GraphicsViewTruthRowV2(
            view_index=3,
            face_a=1,
            face_b=2,
            front_probability=self._valid(1.0, "ray_stack:front"),
            occlusion_probability=self._valid(1.0, "ray_stack:occlusion"),
            visibility_probability_a=self._valid(1.0, "ray_stack:vis_a"),
            visibility_probability_b=self._valid(0.0, "ray_stack:vis_b"),
            projected_alpha_boundary_probability=self._valid(1.0, "rgba:alpha"),
            projected_appearance_boundary_probability=self._valid(1.0, "rgba:appearance"),
        )
        values, validity = view_row_to_training_maps_v2(row)
        self.assertEqual(values["projected_alpha_boundary_probability"], 1.0)
        self.assertTrue(validity["projected_appearance_boundary_probability"])
        self.assertNotIn("cut_probability", values)

    def test_diagnostic_only_target_cannot_be_trainable(self):
        with self.assertRaisesRegex(ValueError, "DIAGNOSTIC_TARGET_CANNOT_TRAIN"):
            QualifiedProbabilityTargetV2(
                value=1.0,
                authority=EvidenceAuthority.DIAGNOSTIC_ONLY,
                provenance="proxy",
                valid_for_training=True,
            )

    def test_required_validity_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "VALIDITY_MISSING:occlusion_probability"):
            assert_required_validity_keys_v2(
                {"front_probability": 1.0, "occlusion_probability": 1.0},
                {"front_probability": True},
                required=("front_probability", "occlusion_probability"),
            )


if __name__ == "__main__":
    unittest.main()
