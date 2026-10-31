from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from product.living_compile.authoring import (
    _validate_binding_edits,
    _validate_presentation_override,
    save_user_layer,
)
from product.living_compile.common import LivingCompileError
from product.living_compile.scene import build_scene
from tests.product.test_living_compile_v4 import fixture


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

    def test_save_roundtrip_persists_socket_and_graphics_departure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "bundle"
            root.mkdir()
            fixture(root)
            scene = build_scene(root)
            mesh_id = next(row["mesh_id"] for row in scene["puppet"]["meshes"] if row["view_id"] == "V0")
            binding = next(row for row in scene["binding"]["bindings"] if row["view_id"] == "V0")
            result = save_user_layer(root, {
                "view_id": "V0",
                "binding_edits": [{
                    "binding_id": binding["binding_id"],
                    "primary_final_control_id": binding["primary_final_control_id"],
                    "translation_xy": [5.0, -2.0],
                    "rotation_degrees": 30.0,
                    "scale_xy": [1.1, 0.9],
                    "visible": True,
                }],
                "presentation_override": {
                    "slot_order_back_to_front": [mesh_id],
                    "hidden_mesh_ids": [mesh_id],
                },
            })
            layer = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
            self.assertEqual(layer["binding_edits"][0]["translation_xy"], [5.0, -2.0])
            self.assertEqual(layer["binding_edits"][0]["rotation_degrees"], 30.0)
            self.assertEqual(layer["presentation_override"]["source_graphics_order_back_to_front"], [mesh_id])
            self.assertEqual(layer["presentation_override"]["hidden_mesh_ids"], [mesh_id])
            self.assertTrue(layer["metadata"]["graphics_evidence_departure"])
            self.assertTrue(result["graphics_evidence_departure"])


if __name__ == "__main__":
    unittest.main()
