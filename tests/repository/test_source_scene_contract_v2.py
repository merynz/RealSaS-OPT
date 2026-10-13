from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from tools.observation.source_scene_contract_v2 import (
    _json_and_chunks, prepare_animation_free_gltf, validate_imported_scene_snapshot,
)


def node(key, parent, world, local, normalization):
    return {"node_id": key, "parent_node_id": parent,
            "node_to_source_world": world.tolist(), "effective_node_to_parent": local.tolist(),
            "node_to_canonical": (normalization @ world).tolist(),
            "active_action_count": 0, "active_nla_track_count": 0}


def snapshot(scale=2.):
    normalization = np.diag([scale, scale, scale, 1.])
    root = np.eye(4); root[:3, 3] = [7, 8, 9]
    local = np.eye(4); local[:3, 3] = [-2, 1, 3]
    return {"schema": "RealSaS.ImportedSourceSceneSnapshot.v2",
            "source_world_to_canonical": normalization.tolist(),
            "canonical_to_source_world": np.linalg.inv(normalization).tolist(),
            "nodes": [node("root", None, root, root, normalization),
                      node("child", "root", root @ local, local, normalization)]}


class SourceSceneContractTests(unittest.TestCase):
    def test_parent_world_and_common_normalization(self):
        result = validate_imported_scene_snapshot(snapshot())
        self.assertEqual(result["max_parent_world_residual"], 0.)
        self.assertFalse(result["independent_source_bind_parity_claimed"])
        self.assertFalse(result["training_admission_claimed"])

    def test_lost_root_or_attachment_transform_fails(self):
        data = snapshot(); data["nodes"][1]["node_to_source_world"][0][3] += .1
        with self.assertRaisesRegex(ValueError, "SCENE_PARENT_WORLD_DRIFT"):
            validate_imported_scene_snapshot(data)

    def test_live_animation_cannot_be_called_rest(self):
        for field in ("active_action_count", "active_nla_track_count"):
            data = snapshot(); data["nodes"][0][field] = 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "LIVE_ANIMATION"):
                validate_imported_scene_snapshot(data)
        data = snapshot(); data["nodes"][0]["driver_count"] = 1
        with self.assertRaisesRegex(ValueError, "LIVE_DRIVERS"):
            validate_imported_scene_snapshot(data)

    def test_parent_cycles_and_missing_nodes_fail(self):
        for parent, error in (("child", "PARENT_CYCLE"), ("missing", "PARENT_MISSING")):
            data = snapshot(); data["nodes"][0]["parent_node_id"] = parent
            with self.subTest(parent=parent), self.assertRaisesRegex(ValueError, error):
                validate_imported_scene_snapshot(data)

    def test_tiny_units_are_valid_but_nonuniform_normalization_is_not(self):
        validate_imported_scene_snapshot(snapshot(1e-8))
        data = snapshot(1e-8); data["source_world_to_canonical"][0][0] *= 2
        data["canonical_to_source_world"][0][0] /= 2
        with self.assertRaisesRegex(ValueError, "NONUNIFORM"):
            validate_imported_scene_snapshot(data)

    def test_canonical_transform_and_inverse_must_match(self):
        for field, error in (("canonical_to_source_world", "NORMALIZATION_INVERSE"),
                             ("node_to_canonical", "CANONICAL_BINDING_DRIFT")):
            data = snapshot()
            target = data[field] if field in data else data["nodes"][1][field]
            target[0][3] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                validate_imported_scene_snapshot(data)

    def test_animation_free_derivative_preserves_scene_skin_material_and_external_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; raw.mkdir()
            (raw / "buffer.bin").write_bytes(b"exact source bytes")
            doc = {"asset": {"version": "2.0"}, "nodes": [{"translation": [3, 5, 8]}],
                   "skins": [{"joints": [0]}], "materials": [{"alphaMode": "MASK"}],
                   "buffers": [{"uri": "buffer.bin", "byteLength": 18}],
                   "animations": [{"name": "ObjectAndRigMotion"}]}
            source = raw / "source.gltf"; source.write_text(json.dumps(doc)); before = source.read_bytes()
            target = root / "derived" / source.name
            receipt = prepare_animation_free_gltf(source, target)
            expected = copy.deepcopy(doc); expected.pop("animations")
            self.assertEqual(json.loads(target.read_bytes()), expected)
            self.assertEqual((target.parent / "buffer.bin").read_bytes(), b"exact source bytes")
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(receipt["removed_animation_count"], 1)
            with self.assertRaisesRegex(ValueError, "OUTSIDE_RAW"):
                prepare_animation_free_gltf(source, raw / "derived" / source.name)

    def test_glb_keeps_binary_chunk_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; raw.mkdir()
            doc = {"asset": {"version": "2.0"}, "nodes": [{"scale": [.2, .2, .2]}],
                   "animations": [{"name": "run"}]}
            encoded = json.dumps(doc).encode(); encoded += b" " * (-len(encoded) % 4)
            binary = b"\x00\x01\x02\x03" * 4
            body = struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
            body += struct.pack("<II", len(binary), 0x004E4942) + binary
            source = raw / "source.glb"
            source.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
            target = root / "derived" / source.name
            prepare_animation_free_gltf(source, target)
            new, chunks = _json_and_chunks(target)
            self.assertNotIn("animations", new)
            self.assertEqual(new["nodes"], doc["nodes"])
            self.assertEqual(chunks, [(0x004E4942, binary)])

    def test_external_resource_cannot_escape_or_overwrite_derivative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; raw.mkdir()
            source = raw / "source.gltf"
            for uri in ("../private.bin", "https://example.org/input.bin", "source.gltf"):
                source.write_text(json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": uri}]}))
                with self.subTest(uri=uri), self.assertRaises(ValueError):
                    prepare_animation_free_gltf(source, root / "derived" / source.name)


if __name__ == "__main__":
    unittest.main()
