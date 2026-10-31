from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from product.living_compile.authoring import save_user_layer
from product.living_compile.common import LivingCompileError
from tests.product.test_living_compile_v4 import fixture


class MeshTopologyAuthoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "bundle"
        self.root.mkdir()
        fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_split_edge_is_persisted_as_sibling_authoring_data(self):
        canonical_path = self.root / "puppet/canonical_puppet_graph_v3.json"
        before = canonical_path.read_bytes()
        result = save_user_layer(self.root, {
            "view_id": "V0",
            "mesh_topology_edits": [{
                "operation": "split_edge",
                "mesh_id": "V0:BODY",
                "vertex_a_id": "v:0",
                "vertex_b_id": "v:1",
                "new_vertex_id": "user:v1",
                "t": 0.5,
            }],
        })
        self.assertEqual(result["mesh_topology_edit_count"], 1)
        self.assertEqual(before, canonical_path.read_bytes())
        layer = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertEqual(layer["mesh_topology_edits"][0]["new_vertex_id"], "user:v1")
        self.assertEqual(layer["metadata"]["mesh_topology_edit_count"], 1)
        self.assertTrue(layer["requires_compiler_requalification"])
        self.assertFalse(layer["metadata"]["canonical_bundle_mutated"])

    def test_delete_referenced_vertex_fails_closed(self):
        with self.assertRaisesRegex(LivingCompileError, "still referenced"):
            save_user_layer(self.root, {
                "view_id": "V0",
                "mesh_topology_edits": [{
                    "operation": "delete_vertex",
                    "mesh_id": "V0:BODY",
                    "vertex_id": "v:0",
                }],
            })

    def test_duplicate_and_zero_area_faces_fail_closed(self):
        with self.assertRaisesRegex(LivingCompileError, "duplicate face"):
            save_user_layer(self.root, {
                "view_id": "V0",
                "mesh_topology_edits": [{
                    "operation": "add_face",
                    "mesh_id": "V0:BODY",
                    "face_id": "user:f1",
                    "vertex_ids": ["v:0", "v:1", "v:2"],
                }],
            })
        with self.assertRaisesRegex(LivingCompileError, "zero-area"):
            save_user_layer(self.root, {
                "view_id": "V0",
                "mesh_topology_edits": [
                    {"operation": "add_vertex", "mesh_id": "V0:BODY", "vertex_id": "user:v1", "position_xy": [10, 10]},
                    {"operation": "add_vertex", "mesh_id": "V0:BODY", "vertex_id": "user:v2", "position_xy": [20, 20]},
                    {"operation": "add_vertex", "mesh_id": "V0:BODY", "vertex_id": "user:v3", "position_xy": [30, 30]},
                    {"operation": "add_face", "mesh_id": "V0:BODY", "face_id": "user:f1", "vertex_ids": ["user:v1", "user:v2", "user:v3"]},
                ],
            })

    def test_weld_replays_and_drops_degenerate_incident_face(self):
        result = save_user_layer(self.root, {
            "view_id": "V0",
            "mesh_topology_edits": [{
                "operation": "weld_vertices",
                "mesh_id": "V0:BODY",
                "keep_vertex_id": "v:0",
                "remove_vertex_id": "v:1",
            }],
        })
        self.assertEqual(result["mesh_topology_edit_count"], 1)


if __name__ == "__main__":
    unittest.main()
