import base64
import hashlib
import json

import numpy as np
import pytest

from tools.corpus.plan_teacher_truth_revalidation_v1 import plan


def fixture(root):
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="<f4")
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype="<u2")
    raw = vertices.tobytes() + faces.tobytes()
    source = root / "tetra.gltf"
    source.write_text(json.dumps({
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(raw).decode(), "byteLength": len(raw)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": vertices.nbytes},
                        {"buffer": 0, "byteOffset": vertices.nbytes, "byteLength": faces.nbytes}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
                      {"bufferView": 1, "componentType": 5123, "count": 12, "type": "SCALAR"}],
    }))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = root / "source_manifest.json"
    manifest.write_text(json.dumps({"subject_count": 1, "subjects": [{
        "subject_id": "tetra", "corpus_relative_path": source.name,
        "expected_sha256": sha, "actual_sha256": sha, "valid": True,
    }]}))
    readiness = root / "prior.json"
    artifact = {"path": "/historical/geometry.json", "sha256": "a" * 64}
    readiness.write_text(json.dumps({"subject_count": 1, "training_authorized": True,
                                    "split": {"train": ["tetra"]},
                                    "subjects": {"tetra": {"geometry": artifact}}}))
    return manifest, readiness, source, artifact


def test_preserves_historical_bytes_without_inheriting_authority(tmp_path):
    manifest, prior, _, artifact = fixture(tmp_path)
    result = plan(manifest, prior, tmp_path, tmp_path / "audit")
    assert result["verified_source_count"] == 1
    assert result["measurement_counts"]["inspected_subjects"] == 1
    assert result["authority"]["training_authorized"] is False
    assert result["authority"]["historical_split_inherited"] is False
    assert result["authority"]["truth_artifacts_regenerated"] is False
    assert result["authority"]["final_admitted_subject_count"] is None
    subject = result["subjects"][0]
    assert subject["historical_artifact_index"]["geometry"] == artifact
    assert all(not row["reuse_certified"] for row in subject["families"].values())


def test_source_drift_is_not_hidden_by_historical_pass(tmp_path):
    manifest, prior, source, _ = fixture(tmp_path)
    source.write_text(source.read_text() + "\n")
    result = plan(manifest, prior, tmp_path, tmp_path / "audit")
    assert result["verified_source_count"] == 0
    assert result["failures"]["tetra"]["reason"] == "TEACHER_SOURCE_IDENTITY_DRIFT"


def test_inventory_mode_cannot_claim_structural_inspection(tmp_path):
    manifest, prior, _, _ = fixture(tmp_path)
    result = plan(manifest, prior, tmp_path, tmp_path / "audit", inspect=False)
    assert result["status"] == "SOURCE_BYTES_VERIFIED__STRUCTURAL_INSPECTION_PENDING"
    assert result["subjects"][0]["source_inspection"] is None


def test_duplicate_source_identity_rows_fail_before_measurement(tmp_path):
    manifest, prior, _, _ = fixture(tmp_path)
    source_manifest = json.loads(manifest.read_text())
    source_manifest["subjects"] *= 2
    source_manifest["subject_count"] = 2
    manifest.write_text(json.dumps(source_manifest))
    with pytest.raises(ValueError, match="TEACHER_SOURCE_MANIFEST_CARDINALITY"):
        plan(manifest, prior, tmp_path, tmp_path / "audit")
