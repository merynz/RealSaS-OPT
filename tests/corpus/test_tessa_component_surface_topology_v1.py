from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.training.tessa_component_surface_topology_v1 import load_tessa_surface_topology_v1
from tools.training.validate_corpus_mechanical_tessa_teacher_v4 import validate, REQUIRED


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_topology(path: Path, *, face_valid=(True, True), surface=(0, 0), sign=(0, 1)):
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
    )
    # Shared edge (1,2) has the same directed orientation in both faces. Surface
    # connectivity joins them; a sign/manifold relation must not.
    faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    np.savez_compressed(
        path,
        raw_to_topology_vertex=np.arange(4, dtype=np.int32),
        topology_vertices_source_world=vertices,
        topology_faces=faces,
        topology_vertex_owner=np.zeros(4, dtype=np.int32),
        face_valid=np.asarray(face_valid, dtype=np.bool_),
        surface_component_id=np.asarray(surface, dtype=np.int32),
        sign_component_id=np.asarray(sign, dtype=np.int32),
        sign_component_valid=np.asarray([False, False], dtype=np.bool_),
    )
    return vertices, faces


def test_tessa_adapter_uses_surface_component_not_sign_component(tmp_path):
    topology = tmp_path / "topology.npz"
    write_topology(topology)
    truth = load_tessa_surface_topology_v1(topology)
    assert truth.component_count == 1
    assert truth.surface_component_id.tolist() == [0, 0]
    assert not hasattr(truth, "sign_component_id")


def test_invalid_geometry_face_requires_repair_instead_of_silent_drop(tmp_path):
    topology = tmp_path / "topology.npz"
    write_topology(topology, face_valid=(True, False))
    with pytest.raises(ValueError, match="TESSA_SURFACE_TOPOLOGY_INVALID_FACE_REPAIR_REQUIRED"):
        load_tessa_surface_topology_v1(topology)


def write_manifest_fixture(root: Path):
    topology = root / "topology.npz"
    vertices, faces = write_topology(topology)
    geometry_manifest = root / "geometry_manifest.json"
    geometry_manifest.write_text(
        json.dumps(
            {
                "schema": "RealSaS.ComponentGeometryAuthority.v1",
                "status": "PASS_COMPONENT_GEOMETRY_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED",
                "subject_id": "subject",
                "outputs": {"topology": {"path": str(topology), "sha256": sha256(topology)}},
            },
            sort_keys=True,
        )
        + "\n"
    )
    geometry_court = root / "geometry_court.json"
    geometry_court.write_text(
        json.dumps(
            {
                "schema": "RealSaS.ComponentGeometryAuthorityCourt.v1",
                "status": "PASS_COMPONENT_GEOMETRY_AUTHORITY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE",
                "subject_id": "subject",
                "authority": {"independent_court_passed": True},
            },
            sort_keys=True,
        )
        + "\n"
    )
    teacher_mesh = root / "teacher.npz"
    source_triangle_id = np.asarray([0, 0, 0, 1], dtype=np.int32)
    source_barycentric = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32
    )
    np.savez_compressed(
        teacher_mesh,
        vertices_source_world=vertices,
        faces=faces,
        source_triangle_id=source_triangle_id,
        source_barycentric=source_barycentric,
    )
    selected = {
        "ratio": 1.0,
        "kind": "SOURCE_FALLBACK",
        "vertex_count": 4,
        "face_count": 2,
        "mechanical_pass": True,
    }
    selected.update({key: True for key in REQUIRED})
    selected["candidate_evaluation_contract"] = "MEASURED_ALL_KINDS_NO_FALLBACK_OVERRIDE_V1"
    manifest = {
        "schema": "RealSaS.CorpusMechanicalTESSATeacher.v4",
        "status": "PASS_CORPUS_MECHANICAL_TESSA_TEACHER_CANDIDATE__INDEPENDENT_COURT_REQUIRED",
        "subject_id": "subject",
        "parents": {
            "geometry_manifest": {"path": str(geometry_manifest), "sha256": sha256(geometry_manifest)},
            "geometry_court": {"path": str(geometry_court), "sha256": sha256(geometry_court)},
        },
        "source_topology": {
            "surface_component_count": 1,
            "component_relation": "EDGE_CONNECTED_SURFACE_COMPONENT_ID",
            "sign_component_relation_consumed": False,
            "sdf_or_occupancy_truth_consumed": False,
            "render_vertex_identity_used_as_topology": False,
        },
        "candidates": [selected],
        "selected": selected,
        "teacher_mesh": {"path": str(teacher_mesh), "sha256": sha256(teacher_mesh)},
        "family_candidate_outcomes": {
            "SURFACE_TOPOLOGY": "CONSUMER_SEMANTICS_PASS_CANDIDATE",
            "CAUSAL_RESPONSE": "EXISTING_EVIDENCE_CONSUMED__NEW_FAMILY_COURT_STILL_REQUIRED",
        },
        "teacher_identity_sha256": "1" * 64,
    }
    return manifest


def test_tessa_court_passes_when_surface_and_sign_partitions_differ(tmp_path):
    manifest = write_manifest_fixture(tmp_path)
    court = validate(manifest)
    assert court["status"] == "PASS_CORPUS_MECHANICAL_TESSA_SURFACE_TOPOLOGY_CONSUMER_COURT__CAUSAL_RESPONSE_AUTHORITY_SEPARATE"
    assert court["surface_component_count"] == 1
    assert court["surface_and_sign_partitions_equal"] is False
    assert court["sign_component_consumed"] is False
    assert court["sdf_or_occupancy_consumed"] is False
    assert court["family_candidate_outcomes"]["CAUSAL_RESPONSE"] == "PENDING_NEW_FAMILY_COURT"
    assert court["authority"]["training_authorized"] is False


def test_tessa_court_rejects_sign_component_dependency(tmp_path):
    manifest = write_manifest_fixture(tmp_path)
    manifest["source_topology"]["sign_component_relation_consumed"] = True
    with pytest.raises(ValueError, match="TESSA_V4_COURT_SIGN_COMPONENT_CONSUMED"):
        validate(manifest)


def test_tessa_court_rejects_wrong_surface_partition_even_if_sign_matches(tmp_path):
    manifest = write_manifest_fixture(tmp_path)
    topology_path = Path(json.loads(Path(manifest["parents"]["geometry_manifest"]["path"]).read_text())["outputs"]["topology"]["path"])
    write_topology(topology_path, surface=(0, 1), sign=(0, 1))
    geometry_manifest_path = Path(manifest["parents"]["geometry_manifest"]["path"])
    gm = json.loads(geometry_manifest_path.read_text())
    gm["outputs"]["topology"]["sha256"] = sha256(topology_path)
    geometry_manifest_path.write_text(json.dumps(gm, sort_keys=True) + "\n")
    manifest["parents"]["geometry_manifest"]["sha256"] = sha256(geometry_manifest_path)
    with pytest.raises(ValueError, match="TESSA_V4_COURT_SURFACE_COMPONENT_PARTITION"):
        validate(manifest)
