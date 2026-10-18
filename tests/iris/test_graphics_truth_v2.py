import unittest

import torch

from models.iris.vnext.boundary_causality_v1 import EvidenceAuthority
from models.iris.vnext.graphics_truth_v2 import (
    GraphicsStaticTruthRowV2,
    QualifiedProbabilityTargetV2,
    static_row_to_training_maps_v2,
)
from models.iris.vnext.training_v2 import RELATION_KEYS_V2, graphics_relation_loss_v2


class GraphicsTruthV2Tests(unittest.TestCase):
    def test_abstained_cut_is_not_training_truth(self):
        valid = QualifiedProbabilityTargetV2(
            value=1.0,
            authority=EvidenceAuthority.RAW_AUTHORED,
            provenance="authored_material_slot:face=1|2",
            valid_for_training=True,
        )
        unknown_cut = QualifiedProbabilityTargetV2.abstain(
            provenance="drawable_cut_not_identifiable_from_current_source"
        )
        row = GraphicsStaticTruthRowV2(
            face_a=1,
            face_b=2,
            same_drawable_affinity=QualifiedProbabilityTargetV2.abstain(
                provenance="anonymous_object_id_is_not_authoritative_drawable_owner"
            ),
            cut_probability=unknown_cut,
            attachment_affinity=QualifiedProbabilityTargetV2.abstain(
                provenance="no_authoritative_attachment_relation"
            ),
            alpha_boundary_probability=QualifiedProbabilityTargetV2.abstain(
                provenance="no_raster_sample_for_this_static_pair"
            ),
            appearance_boundary_probability=valid,
        )
        values, validity = static_row_to_training_maps_v2(row)
        self.assertEqual(values["cut_probability"], 0.0)
        self.assertFalse(validity["cut_probability"])
        self.assertTrue(validity["appearance_boundary_probability"])

    def test_diagnostic_only_target_cannot_be_marked_trainable(self):
        with self.assertRaisesRegex(ValueError, "DIAGNOSTIC_TARGET_CANNOT_TRAIN"):
            QualifiedProbabilityTargetV2(
                value=1.0,
                authority=EvidenceAuthority.DIAGNOSTIC_ONLY,
                provenance="object_name_proxy",
                valid_for_training=True,
            )

    def test_relation_loss_ignores_abstained_family(self):
        predicted = {name: torch.full((4,), 0.5) for name in RELATION_KEYS_V2}
        truth = {name: torch.zeros(4) for name in RELATION_KEYS_V2}
        validity = {name: torch.zeros(4, dtype=torch.bool) for name in RELATION_KEYS_V2}

        truth["front_probability"] = torch.ones(4)
        validity["front_probability"] = torch.ones(4, dtype=torch.bool)
        # Deliberately terrible cut predictions must not affect the loss when cut abstains.
        predicted["cut_probability"] = torch.full((4,), 0.999999)
        truth["cut_probability"] = torch.zeros(4)

        loss = graphics_relation_loss_v2(predicted, truth, validity)
        expected = torch.nn.functional.binary_cross_entropy(
            predicted["front_probability"], truth["front_probability"]
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_all_abstained_is_fail_closed(self):
        predicted = {name: torch.full((2,), 0.5) for name in RELATION_KEYS_V2}
        truth = {name: torch.zeros(2) for name in RELATION_KEYS_V2}
        validity = {name: torch.zeros(2, dtype=torch.bool) for name in RELATION_KEYS_V2}
        with self.assertRaisesRegex(ValueError, "ALL_TARGETS_ABSTAINED"):
            graphics_relation_loss_v2(predicted, truth, validity)

    def test_shape_mismatch_fails(self):
        predicted = {name: torch.full((2,), 0.5) for name in RELATION_KEYS_V2}
        truth = {name: torch.zeros(2) for name in RELATION_KEYS_V2}
        validity = {name: torch.ones(2, dtype=torch.bool) for name in RELATION_KEYS_V2}
        validity["visibility_probability_b"] = torch.ones(3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "SHAPE_MISMATCH"):
            graphics_relation_loss_v2(predicted, truth, validity)


if __name__ == "__main__":
    unittest.main()
