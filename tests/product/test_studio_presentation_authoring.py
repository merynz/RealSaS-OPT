from __future__ import annotations

import unittest

from product.living_compile.authoring import _validate_binding_edits, _validate_presentation_override
from product.living_compile.common import LivingCompileError


class StudioPresentationAuthoringTests(unittest.TestCase):
    def test_presentation_requires_exact_canonical_permutation(self):
        meshes = {"BACK": {}, "BODY": {}, "FRONT": {}}
        clean = _validate_presentation_override(meshes, {
            "slot_order_back_to_front": ["BACK", "FRONT", "BODY"],
            "hidden_mesh_ids": ["BACK"],
        })
        self.assertEqual(clean["source_graphics_order_back_to_front"], ["BACK", "BODY", "FRONT"])
        self.assertEqual(clean["slot_order_back_to_front"], ["BACK", "FRONT", "BODY"])
        self.assertTrue(clean["graphics_evidence_departure"])

        with self.assertRaisesRegex(LivingCompileError, "exact canonical slot permutation"):
            _validate_presentation_override(meshes, {
                "slot_order_back_to_front": ["BACK", "BODY"],
                "hidden_mesh_ids": [],
            })
        with self.assertRaisesRegex(LivingCompileError, "unique canonical subset"):
            _validate_presentation_override(meshes, {
                "slot_order_back_to_front": ["BACK", "BODY", "FRONT"],
                "hidden_mesh_ids": ["UNKNOWN"],
            })

    def test_binding_transform_is_bounded_and_deform_joint_only(self):
        bindings = {"B:HAND": {"binding_id": "B:HAND"}}
        controls = {"J:HAND", "J:ROOT"}
        clean = _validate_binding_edits(bindings, controls, [{
            "binding_id": "B:HAND",
            "primary_final_control_id": "J:ROOT",
            "translation_xy": [12.0, -3.0],
            "rotation_degrees": 25.0,
            "scale_xy": [1.25, 0.75],
            "visible": False,
        }])
        self.assertEqual(clean[0]["primary_final_control_id"], "J:ROOT")
        self.assertEqual(clean[0]["translation_xy"], [12.0, -3.0])
        self.assertEqual(clean[0]["scale_xy"], [1.25, 0.75])
        self.assertFalse(clean[0]["visible"])

        with self.assertRaisesRegex(LivingCompileError, "unknown deform joint"):
            _validate_binding_edits(bindings, controls, [{
                "binding_id": "B:HAND",
                "primary_final_control_id": "C:ANIMATOR",
            }])
        with self.assertRaisesRegex(LivingCompileError, "scale must be within"):
            _validate_binding_edits(bindings, controls, [{
                "binding_id": "B:HAND",
                "primary_final_control_id": "J:HAND",
                "scale_xy": [0, 1],
            }])


if __name__ == "__main__":
    unittest.main()
