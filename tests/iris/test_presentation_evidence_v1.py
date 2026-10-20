import unittest

import torch

from models.iris.vnext.presentation_evidence_v1 import (
    PRESENTATION_FEATURE_NAMES_V1,
    PRESENTATION_KIND_PAIR,
    PRESENTATION_KIND_RAY_HIT,
    build_role_free_presentation_evidence_v1,
    null_presentation_evidence_v1,
)


class PresentationEvidenceV1Tests(unittest.TestCase):
    def test_invalid_scalar_is_zeroed_not_interpreted_as_negative_truth(self):
        xyz = torch.zeros(1, 2, 3)
        kind = torch.tensor([[PRESENTATION_KIND_PAIR, PRESENTATION_KIND_RAY_HIT]])
        token_valid = torch.ones(1, 2, dtype=torch.bool)
        front = torch.tensor([[0.9, 0.8]])
        front_valid = torch.tensor([[True, False]])
        evidence = build_role_free_presentation_evidence_v1(
            xyz=xyz,
            kind_id=kind,
            token_valid_mask=token_valid,
            front_probability=front,
            front_valid=front_valid,
        )
        front_index = PRESENTATION_FEATURE_NAMES_V1.index("front_probability")
        valid_index = PRESENTATION_FEATURE_NAMES_V1.index("front_valid")
        self.assertAlmostEqual(float(evidence.values[0, 0, front_index]), 0.9, places=6)
        self.assertEqual(float(evidence.values[0, 1, front_index]), 0.0)
        self.assertEqual(float(evidence.values[0, 0, valid_index]), 1.0)
        self.assertEqual(float(evidence.values[0, 1, valid_index]), 0.0)

    def test_no_named_semantic_taxonomy_in_feature_contract(self):
        lowered = " ".join(PRESENTATION_FEATURE_NAMES_V1).lower()
        for forbidden in ("arm", "leg", "hair", "cape", "body", "cloth"):
            self.assertNotIn(forbidden, lowered)

    def test_null_control_is_admitted_but_contains_no_signal(self):
        evidence = null_presentation_evidence_v1(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(evidence.values.shape, (2, 1, len(PRESENTATION_FEATURE_NAMES_V1)))
        self.assertTrue(torch.equal(evidence.values, torch.zeros_like(evidence.values)))
        self.assertTrue(bool(evidence.valid_mask.all()))

    def test_value_requires_explicit_validity(self):
        xyz = torch.zeros(1, 1, 3)
        with self.assertRaisesRegex(ValueError, "VALUE_VALIDITY_PAIR_REQUIRED"):
            build_role_free_presentation_evidence_v1(
                xyz=xyz,
                kind_id=torch.zeros(1, 1, dtype=torch.long),
                token_valid_mask=torch.ones(1, 1, dtype=torch.bool),
                global_visibility=torch.ones(1, 1),
            )


if __name__ == "__main__":
    unittest.main()
