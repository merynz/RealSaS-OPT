import unittest

from models.tessa.v2 import (
    StructuralizationVariantEvidenceV2,
    evaluate_structuralization_ab_v2,
)


_H = "a" * 64


def _provenance():
    return {
        "candidate": _H,
        "static_qualification": _H,
        "multiview_qualification": _H,
        "dynamic_qualification": _H,
        "source_fidelity": _H,
    }


def _direct(**updates):
    value = dict(
        variant_id="DIRECT_IRIS_SURFACE",
        candidate_binding_hash=_H,
        vertex_count=8000,
        face_count=12000,
        compiler_static_qualified=True,
        compiler_multiview_qualified=True,
        compiler_dynamic_qualified=True,
        support_binding_qualified=True,
        source_surface_p95_distance=0.010,
        source_surface_max_distance=0.030,
        minimum_multiview_mask_iou=0.96,
        maximum_multiview_depth_abs_error=0.020,
        minimum_multiview_normal_cosine=0.95,
        geometric_component_count=3,
        boundary_edge_count=12,
        open_boundary_chain_count=0,
        nonmanifold_edge_count=0,
        zero_area_face_count=0,
        g3_maximum_condition_number=2.0,
        g3_minimum_edge_ratio=0.80,
        g3_maximum_edge_ratio=1.20,
        evidence_provenance=_provenance(),
    )
    value.update(updates)
    return StructuralizationVariantEvidenceV2(**value)


def _tessa(**updates):
    value = dict(
        variant_id="TESSA_STRUCTURALIZED",
        candidate_binding_hash="b" * 64,
        vertex_count=3500,
        face_count=6000,
        compiler_static_qualified=True,
        compiler_multiview_qualified=True,
        compiler_dynamic_qualified=True,
        support_binding_qualified=True,
        source_surface_p95_distance=0.0102,
        source_surface_max_distance=0.0305,
        minimum_multiview_mask_iou=0.958,
        maximum_multiview_depth_abs_error=0.0205,
        minimum_multiview_normal_cosine=0.947,
        geometric_component_count=3,
        boundary_edge_count=10,
        open_boundary_chain_count=0,
        nonmanifold_edge_count=0,
        zero_area_face_count=0,
        g3_maximum_condition_number=2.04,
        g3_minimum_edge_ratio=0.795,
        g3_maximum_edge_ratio=1.205,
        evidence_provenance={key: "b" * 64 for key in _provenance()},
    )
    value.update(updates)
    return StructuralizationVariantEvidenceV2(**value)


class StructuralizationABCourtV2Tests(unittest.TestCase):
    def test_tessa_is_promotable_only_when_all_downstream_courts_and_fidelity_hold(self):
        receipt = evaluate_structuralization_ab_v2(_direct(), _tessa())
        self.assertEqual(receipt.status, "TESSA_PROMOTABLE")
        self.assertTrue(receipt.tessa_mandatory_authorized)
        self.assertFalse(receipt.training_authorized)
        self.assertFalse(receipt.scientific_pass_claimed)
        self.assertGreaterEqual(receipt.deltas["face_reduction_fraction"], 0.25)
        self.assertEqual(receipt.blockers, ())

    def test_direct_surface_is_preferred_when_tessa_adds_no_structural_economy(self):
        receipt = evaluate_structuralization_ab_v2(
            _direct(),
            _tessa(face_count=11500, vertex_count=7600),
        )
        self.assertEqual(receipt.status, "DIRECT_SURFACE_PREFERRED")
        self.assertFalse(receipt.tessa_mandatory_authorized)
        self.assertIn("face_reduction_sufficient", receipt.blockers)

    def test_direct_surface_is_preferred_when_tessa_fails_compiler_dynamic_court(self):
        receipt = evaluate_structuralization_ab_v2(
            _direct(),
            _tessa(compiler_dynamic_qualified=False),
        )
        self.assertEqual(receipt.status, "DIRECT_SURFACE_PREFERRED")
        self.assertFalse(receipt.tessa_mandatory_authorized)
        self.assertIn("tessa_all_compiler_courts_pass", receipt.blockers)

    def test_geometry_or_motion_regression_blocks_tessa_even_with_large_face_reduction(self):
        receipt = evaluate_structuralization_ab_v2(
            _direct(),
            _tessa(
                face_count=3000,
                source_surface_p95_distance=0.020,
                g3_maximum_condition_number=3.0,
            ),
        )
        self.assertNotEqual(receipt.status, "TESSA_PROMOTABLE")
        self.assertFalse(receipt.tessa_mandatory_authorized)
        self.assertIn("surface_p95_preserved", receipt.blockers)
        self.assertIn("g3_conditioning_preserved", receipt.blockers)

    def test_exact_provenance_is_required(self):
        broken = _tessa(evidence_provenance={"candidate": "b" * 64})
        with self.assertRaisesRegex(ValueError, "TESSA_AB_EVIDENCE_PROVENANCE_INCOMPLETE"):
            evaluate_structuralization_ab_v2(_direct(), broken)


if __name__ == "__main__":
    unittest.main()
