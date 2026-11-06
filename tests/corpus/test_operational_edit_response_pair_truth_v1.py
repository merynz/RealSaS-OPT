"""Adversarial contracts: source edit-signature truth is not drawable truth."""
from __future__ import annotations

import copy
import unittest

from tools.corpus.derive_operational_edit_response_pair_truth_v1 import derive_pairs


def sample(op, x, y=14, face=9, view=0):
    # Valid source anchor analogue, not real-source PASS.
    return {
        "original_face_id": face, "original_camera_view": view,
        "original_uv_edit_index": op,
        "pose0_pixel_xy": [x, y],
        "pose14_qualified_pixel_xy": [x + 1, y + 1],
        "pose14_transported_barycentric_screen_xy": [x + 1.3, y + 1.1],
        "pose0_center_source_barycentric": [0.25, 0.25 + x * 0.001, 0.5 - x * 0.001],
        "source_pose0_and_pose14_independent_physical_positive": True,
        "same_source_face_UV_anchor": True,
        "original_artist_drawable_owner": "UNKNOWN",
        "exact_Cycles_sample_ray_UV_correspondence": "UNKNOWN_NOT_ASSERTED",
    }


class PhysicalEditSignaturePairTests(unittest.TestCase):
    def test_same_face_has_positive_and_negative_pairs(self):
        anchors = [sample(0, 5), sample(0, 7), sample(1, 25), sample(1, 28)]
        pairs = derive_pairs(anchors, num_operations=3)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(sum(p["measured_edit_signature_agreement"] == 1 for p in pairs), 2)
        self.assertEqual(sum(p["measured_edit_signature_agreement"] == 0 for p in pairs), 1)
        for p in pairs:
            self.assertEqual(p["same_drawable_owner"], "UNKNOWN")
            self.assertEqual(p["proposed_compiler_cut"], "UNDECIDED")
            self.assertTrue(p["validity_source_physical_both_poses"])
            self.assertEqual(p["locus_a_TEACHER_PRIVATE"]["original_face_id_TEACHER_PRIVATE"],
                             p["locus_b_TEACHER_PRIVATE"]["original_face_id_TEACHER_PRIVATE"])
            self.assertNotEqual(p["locus_a_TEACHER_PRIVATE"]["source_barycentric_TEACHER_PRIVATE"],
                                p["locus_b_TEACHER_PRIVATE"]["source_barycentric_TEACHER_PRIVATE"])

    def test_different_faces_are_not_assumed_equal_or_different(self):
        anchors = [sample(0, 5, face=9), sample(1, 12, face=10)]
        self.assertEqual(derive_pairs(anchors, num_operations=3), [])

    def test_different_view_not_silently_joined(self):
        anchors = [sample(0, 5), sample(1, 12, view=2)]
        self.assertEqual(derive_pairs(anchors, num_operations=3), [])

    def test_teacher_op_alias_on_same_pixel_is_rejected(self):
        anchors = [sample(0, 5), sample(1, 5)]
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS_SOURCE_PIXEL"):
            derive_pairs(anchors, num_operations=3)

    def test_false_physical_edit_receipt_rejected(self):
        a = sample(0, 5)
        a["source_pose0_and_pose14_independent_physical_positive"] = False
        with self.assertRaisesRegex(ValueError, "SOURCE_PROOF"):
            derive_pairs([a], num_operations=3)

    def test_fake_artist_owner_rejected(self):
        a = sample(0, 5)
        a["original_artist_drawable_owner"] = "ARM"
        with self.assertRaisesRegex(ValueError, "SOURCE_PROOF"):
            derive_pairs([a], num_operations=3)

    def test_degenerate_barycentric_rejected(self):
        a = sample(0, 5)
        a["pose0_center_source_barycentric"] = [0.5, 0.5, 0.5]
        with self.assertRaisesRegex(ValueError, "SOURCE_BARYCENTRIC"):
            derive_pairs([a], num_operations=3)

    def test_transport_pixel_must_contain_transport_projection(self):
        a = sample(0, 5)
        a["pose14_transported_barycentric_screen_xy"] = [42., 42.]
        with self.assertRaisesRegex(ValueError, "POSE_ADDRESS"):
            derive_pairs([a], num_operations=3)

    def test_invalid_operation_cardinality_or_id(self):
        with self.assertRaisesRegex(ValueError, "CONTROL_CARDINALITY"):
            derive_pairs([], num_operations=1)
        with self.assertRaisesRegex(ValueError, "SOURCE_HANDLE"):
            derive_pairs([sample(4, 5)], num_operations=3)

    def test_no_student_input_features_with_teacher_op_or_face_id(self):
        pair = derive_pairs([sample(0, 5), sample(1, 12)], num_operations=3)[0]
        self.assertNotIn("student_input", pair)
        self.assertEqual(pair["attachment_relation"], "UNKNOWN")
        self.assertIn("TEACHER_ONLY", "observed_operation_indices_TEACHER_ONLY")
        self.assertEqual(pair["measured_edit_signature_agreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
