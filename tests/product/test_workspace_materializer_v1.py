from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
    encode_source_observation_arrays_v1,
)
from product.living_compile.scene import build_scene
from product.workspace_worker.materialize_workspace_v1 import materialize


AUTHORITY = {
    "mesh:canonical": ("RealSaS.CanonicalMeshIR", "v1"),
    "rig:canonical": ("RealSaS.CanonicalRigIR", "v2"),
    "skin:canonical": ("RealSaS.CanonicalSkinIR", "v2"),
    "mechanics:canonical": ("RealSaS.CanonicalMechanicsIR", "v1"),
    "graphics:canonical": ("RealSaS.CanonicalGraphicsIR", "v1"),
    "motion:canonical": ("RealSaS.CanonicalMotionIR", "v2"),
    "proof:product_pass": ("RealSaS.ProductProofBundleIR", "v1"),
}


def camera_bundle() -> dict:
    cameras = []
    for view in range(8):
        cameras.append({
            "schema_version": "RealSaS.FullSurfaceCameraProjection.v3",
            "view_id": f"V{view}",
            "view_index": view,
            "origin": [0.0, 0.0, -2.0],
            "right": [1.0, 0.0, 0.0],
            "screen_up": [0.0, 1.0, 0.0],
            "forward": [0.0, 0.0, 1.0],
            "half_extent": 2.0,
            "resolution": 1024,
        })
    return {
        "schema": "RealSaS.CameraProjectionBundle.v1",
        "view_order": [f"V{i}" for i in range(8)],
        "per_view_recenter": False,
        "per_view_pose_change": False,
        "cameras": cameras,
    }


def payloads() -> dict[str, dict]:
    mesh_hash = "1" * 64
    rig_hash = "2" * 64
    deform_hash = "3" * 64
    control_hash = "4" * 64
    skin_hash = "5" * 64
    mechanics_hash = "6" * 64
    product_hash = "7" * 64
    vertices = [
        {"canonical_mesh_vertex_id": "MV:A", "P": [-0.5, -0.5, 0.0], "component_id": "BODY"},
        {"canonical_mesh_vertex_id": "MV:B", "P": [0.5, -0.5, 0.0], "component_id": "BODY"},
        {"canonical_mesh_vertex_id": "MV:C", "P": [0.0, 0.5, 0.0], "component_id": "BODY"},
    ]
    mesh = {
        "schema_version": "RealSaS.CanonicalMeshIR.v1",
        "mesh_lineage_hash": mesh_hash,
        "vertices": vertices,
        "faces": [["MV:A", "MV:B", "MV:C"]],
    }
    rig = {
        "schema_version": "RealSaS.CanonicalRigIR.v2",
        "mesh_lineage_hash": mesh_hash,
        "rig_lineage_hash": rig_hash,
        "deform_structure_hash": deform_hash,
        "control_program_hash": control_hash,
        "deform_root_joint_ids": ["J:ROOT"],
        "joints": [{
            "canonical_joint_id": "J:ROOT", "P": [0.0, 0.0, 0.0],
            "parent_canonical_joint_id": None, "support_evidence_ids": [],
        }],
        "controls": [{
            "canonical_control_id": "C:ROOT", "control_kind": "TRANSLATE_ROTATE",
            "target_canonical_joint_ids": ["J:ROOT"], "parent_canonical_control_id": None,
            "rest_P": [0.0, 0.0, 0.0], "metadata": {},
        }],
        "chains": [], "constraints": [], "attachment_anchors": [],
    }
    skin = {
        "schema_version": "RealSaS.CanonicalSkinIR.v2",
        "mesh_lineage_hash": mesh_hash,
        "deform_structure_hash": deform_hash,
        "skin_lineage_hash": skin_hash,
        "rows": [
            {"canonical_mesh_vertex_id": vertex["canonical_mesh_vertex_id"], "influences": [["J:ROOT", 1.0]]}
            for vertex in vertices
        ],
    }
    mechanics = {
        "schema_version": "RealSaS.CanonicalMechanicsIR.v1",
        "mesh_lineage_hash": mesh_hash,
        "rig_lineage_hash": rig_hash,
        "skin_lineage_hash": skin_hash,
        "mechanics_lineage_hash": mechanics_hash,
    }
    graphics = {
        "schema_version": "RealSaS.CanonicalGraphicsIR.v1",
        "mesh_lineage_hash": mesh_hash,
        "skin_lineage_hash": skin_hash,
        "mechanics_lineage_hash": mechanics_hash,
        "presentation": {
            "fragments": [{"fragment_id": "FRAG:BODY", "face_indices": [0]}],
            "views": [
                {"view_index": view, "visible_fragment_ids": ["FRAG:BODY"], "order_edges": []}
                for view in range(8)
            ],
        },
    }
    motion = {
        "schema_version": "RealSaS.CanonicalMotionIR.v2",
        "canonical_clip_id": "MOTION:IDLE",
        "display_name": "Idle",
        "intent": "IDLE",
        "duration_seconds": 1.0,
        "loop": True,
        "rig_lineage_hash": rig_hash,
        "control_program_hash": control_hash,
        "mechanics_lineage_hash": mechanics_hash,
        "motion_lineage_hash": "8" * 64,
        "control_tracks": [{
            "canonical_motion_track_id": "MT:ROOT",
            "canonical_control_id": "C:ROOT",
            "source_proposal_track_id": "P:ROOT",
            "keys": [{
                "time_seconds": 0.0,
                "translation": [0.0, 0.0, 0.0],
                "rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }],
        }],
    }
    proof = {
        "schema_version": "RealSaS.ProductProofBundleIR.v1",
        "source_product_state_hash": product_hash,
        "overall_status": "PASS",
        "proof_bundle_hash": "9" * 64,
        "domain_reports": [{
            "proof_domain": "MOTION", "status": "PASS", "failure_signatures": [],
            "owner_attribution": [], "domain_proof_hash": "a" * 64,
        }],
    }
    return {
        "mesh:canonical": mesh,
        "rig:canonical": rig,
        "skin:canonical": skin,
        "mechanics:canonical": mechanics,
        "graphics:canonical": graphics,
        "motion:canonical": motion,
        "proof:product_pass": proof,
    }


def write_request(root: Path, output: Path) -> Path:
    authority_rows = []
    for index, (role, value) in enumerate(payloads().items()):
        path = root / f"authority_{index}.json"
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        artifact_type, schema_version = AUTHORITY[role]
        authority_rows.append({
            "role": role, "scope": "AUTHORITY", "artifact_id": f"authority-{index}",
            "path": str(path), "artifact_type": artifact_type, "schema_version": schema_version,
        })

    rgba = np.zeros((1, 8, 4, 1024, 1024), dtype=np.uint8)
    rgba[:, :, 0] = 80
    rgba[:, :, 1] = 120
    rgba[:, :, 2] = 160
    rgba[:, :, 3] = 255
    source_path = root / "source_observation.npz"
    source_path.write_bytes(encode_source_observation_arrays_v1(native_rgba=rgba, camera_bundle=camera_bundle()))
    support_rows = [{
        "role": "support:source_observation", "scope": "SUPPORT", "artifact_id": "source-observation",
        "path": str(source_path), "artifact_type": "RealSaS.SourceObservation", "schema_version": "v1",
    }]
    request = {
        "schema_version": "RealSaS.ProductWorkspaceMaterializationRequest.v1",
        "compile_id": "cmp_materializer",
        "product_revision_id": "11111111-1111-1111-1111-111111111111",
        "attempt": 1,
        "artifacts": authority_rows,
        "support_artifacts": support_rows,
        "authority": {
            "source_product_revision_is_immutable": True,
            "workspace_is_derived_not_product_authority": True,
            "support_artifacts_are_non_authoritative": True,
            "proof_must_be_preserved": True,
        },
        "required_output": {"path": str(output)},
    }
    path = root / "request.json"
    path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    return path


class WorkspaceMaterializerV1Tests(unittest.TestCase):
    def test_materialization_is_deterministic_and_living_compile_openable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request_a = write_request(root, root / "a.zip")
            output_a = materialize(request_a)
            bytes_a = output_a.read_bytes()

            request = json.loads(request_a.read_text(encoding="utf-8"))
            request["required_output"]["path"] = str(root / "b.zip")
            request_b = root / "request_b.json"
            request_b.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
            output_b = materialize(request_b)
            self.assertEqual(bytes_a, output_b.read_bytes())

            bundle = root / "bundle"
            bundle.mkdir()
            with zipfile.ZipFile(output_a, "r") as archive:
                archive.extractall(bundle)
            scene = build_scene(bundle)
            self.assertTrue(scene["proof"]["passed"])
            self.assertEqual(scene["counts"]["views"], 8)
            self.assertEqual(scene["counts"]["meshes"], 8)
            self.assertEqual(len(scene["index"]["source_views"]), 8)

            product = json.loads((bundle / "puppet" / "canonical_puppet_graph_v3.json").read_text())
            self.assertEqual(product["metadata"]["authority"], "DERIVED_POST_SEAL_WORKSPACE_PROJECTION")
            self.assertFalse(product["metadata"]["new_mesh_identity_minted"])
            self.assertEqual(product["mechanical_state"]["control_program"]["controls"][0]["canonical_control_id"], "C:ROOT")
            self.assertEqual(product["motion_state"]["clips"][0]["control_tracks"][0]["canonical_control_id"], "C:ROOT")

            manifest = json.loads((bundle / "bundle_manifest.json").read_text())
            self.assertEqual(len(manifest["authority_artifacts"]), 7)
            self.assertEqual(len(manifest["support_artifacts"]), 1)
            self.assertFalse(manifest["authority"]["workspace_projection_is_authority"])
            self.assertFalse(manifest["support_artifacts"][0]["product_authority"])

    def test_lineage_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = write_request(root, root / "bad.zip")
            payload = json.loads(request.read_text())
            rig = next(row for row in payload["artifacts"] if row["role"] == "rig:canonical")
            rig_payload = json.loads(Path(rig["path"]).read_text())
            rig_payload["mesh_lineage_hash"] = "f" * 64
            Path(rig["path"]).write_text(json.dumps(rig_payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "RIG_MESH_LINEAGE_DRIFT"):
                materialize(request)


if __name__ == "__main__":
    unittest.main()
