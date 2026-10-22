import unittest

from models.iris.vnext.boundary_causality_v1 import (
    BoundaryCause,
    BoundaryEvidenceV1,
    EvidenceAuthority,
    boundary_causality_teacher_v1,
)


class BoundaryCausalityV1Tests(unittest.TestCase):
    def setUp(self):
        self.teacher = boundary_causality_teacher_v1()

    def test_no_evidence_abstains_to_unknown(self):
        record = self.teacher.qualify([])
        self.assertTrue(record.abstained)
        self.assertFalse(record.training_target_available)
        self.assertEqual(record.positive_causes, (BoundaryCause.RESIDUAL_UNKNOWN,))

    def test_authored_material_boundary_is_training_target(self):
        record = self.teacher.qualify(
            [
                BoundaryEvidenceV1(
                    cause=BoundaryCause.MATERIAL_BOUNDARY,
                    authority=EvidenceAuthority.RAW_AUTHORED,
                    provenance="source_face_material_id:17|18",
                )
            ]
        )
        self.assertFalse(record.abstained)
        self.assertTrue(record.training_target_available)
        self.assertEqual(record.positive_causes, (BoundaryCause.MATERIAL_BOUNDARY,))

    def test_multilabel_geometric_and_occlusion_causes_are_allowed(self):
        record = self.teacher.qualify(
            [
                BoundaryEvidenceV1(
                    cause=BoundaryCause.GEOMETRIC_TERMINATION,
                    authority=EvidenceAuthority.QUALIFIED_DERIVED,
                    provenance="qualified_surface_boundary:face=41",
                ),
                BoundaryEvidenceV1(
                    cause=BoundaryCause.OCCLUSION_VISIBILITY_BOUNDARY,
                    authority=EvidenceAuthority.DETERMINISTIC_DERIVED,
                    provenance="ray_hit_stack:view=3,pixel=20,21",
                ),
            ]
        )
        self.assertEqual(
            set(record.positive_causes),
            {
                BoundaryCause.GEOMETRIC_TERMINATION,
                BoundaryCause.OCCLUSION_VISIBILITY_BOUNDARY,
            },
        )

    def test_diagnostic_only_evidence_cannot_become_training_truth(self):
        record = self.teacher.qualify(
            [
                BoundaryEvidenceV1(
                    cause=BoundaryCause.SOURCE_PRIMITIVE_COMPONENT_SEAM,
                    authority=EvidenceAuthority.DIAGNOSTIC_ONLY,
                    provenance="diagnostic_position_weld:edge=9",
                )
            ]
        )
        self.assertTrue(record.abstained)
        self.assertEqual(record.positive_causes, (BoundaryCause.RESIDUAL_UNKNOWN,))
        self.assertEqual(len(record.evidence), 1)

    def test_provenance_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, "PROVENANCE_REQUIRED"):
            BoundaryEvidenceV1(
                cause=BoundaryCause.ALPHA_BOUNDARY,
                authority=EvidenceAuthority.DETERMINISTIC_DERIVED,
                provenance="",
            )

    def test_unknown_is_not_allowed_as_positive_evidence(self):
        with self.assertRaisesRegex(ValueError, "ABSTENTION_NOT_POSITIVE_EVIDENCE"):
            BoundaryEvidenceV1(
                cause=BoundaryCause.RESIDUAL_UNKNOWN,
                authority=EvidenceAuthority.QUALIFIED_DERIVED,
                provenance="made-up-hidden-state",
            )

    def test_no_drawable_amodal_or_presentation_labels_exist(self):
        names = {item.name for item in BoundaryCause}
        self.assertNotIn("DRAWABLE_OWNER_BOUNDARY", names)
        self.assertNotIn("AMODAL_SUPPORT_BOUNDARY", names)
        self.assertNotIn("PRESENTATION_OVERRIDE_BOUNDARY", names)

    def test_duplicate_evidence_is_deterministically_deduplicated(self):
        item = BoundaryEvidenceV1(
            cause=BoundaryCause.ALPHA_BOUNDARY,
            authority=EvidenceAuthority.DETERMINISTIC_DERIVED,
            provenance="canonical_rgba:view=1,pixel=8,8",
        )
        record = self.teacher.qualify([item, item])
        self.assertEqual(record.evidence, (item,))
        self.assertEqual(record.positive_causes, (BoundaryCause.ALPHA_BOUNDARY,))


if __name__ == "__main__":
    unittest.main()
