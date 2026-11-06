"""Identity gate tests. Synthetic inputs cannot mint original source authority."""
import hashlib
from pathlib import Path
import numpy as np
import pytest
from tools.corpus import court_original_source_face_uv_alignment_v0 as gate


def fixture(tmp_path, monkeypatch):
    glb = tmp_path / "source.glb"
    teacher = tmp_path / "teacher.npz"
    glb.write_bytes(b"synthetic-glb-no-authority")
    position = np.asarray([
        (0., 0., 0.), (1., 0., 0.), (0., 1., 0.),
        (0., 0., 1.), (1., 0., 1.), (0., 1., 1.),
    ])
    faces = np.asarray([[0, 1, 2], [3, 4, 5]])
    face_uv = np.asarray([
        [(0., 0.), (0.45, 0.), (0., 0.9)],
        [(0.55, 0.), (1., 0.), (1., 0.9)],
    ])
    monkeypatch.setattr(gate, "source_arrays",
                        lambda _: ((position, faces, face_uv), None))
    np.savez_compressed(teacher, faces=faces,
                        rest_vertices_source_world=position * 2 + 7,
                        posed_vertices_source_world=np.stack([position * 2 + 7,
                                                             position * 2 + 8]))
    return glb, teacher, faces, position


def evaluate(glb, teacher):
    return gate.audit_alignment(
        glb=glb, teacher=teacher,
        expected_glb_sha=gate.sha256(glb),
        expected_teacher_sha=gate.sha256(teacher))


def test_exact_original_index_and_global_metric_alignment_is_a_narrow_witness(
        tmp_path, monkeypatch):
    glb, teacher, _, _ = fixture(tmp_path, monkeypatch)
    report, uv = evaluate(glb, teacher)
    assert report["source_to_canonical_binding"] == (
        "INDEX_AND_GLOBAL_REST_EDGE_METRIC_VERIFIED")
    assert uv.shape == (2, 3, 2)
    assert report["teacher_pose_count"] == 2
    assert report["source_artist_owner"] == "UNKNOWN"
    assert not any(report[k] for k in gate.NO_AUTHORITY)
    assert not report["student_inference_qualified"]


def test_order_permutation_is_not_forgiven_by_same_face_set(tmp_path, monkeypatch):
    glb, teacher, faces, position = fixture(tmp_path, monkeypatch)
    np.savez_compressed(teacher, faces=faces[::-1],
                        rest_vertices_source_world=position * 2 + 7,
                        posed_vertices_source_world=np.stack([position * 2 + 7]))
    report, uv = evaluate(glb, teacher)
    assert uv is None
    assert report["source_to_canonical_binding"] == "UNKNOWN"
    assert report["reason"] == "ORIGINAL_FACE_INDEX_ORDER_NOT_VERIFIED"


def test_local_nonuniform_deformation_cannot_masquerade_as_global_scale(
        tmp_path, monkeypatch):
    glb, teacher, faces, position = fixture(tmp_path, monkeypatch)
    warped = position.copy()
    warped[1, 0] *= 1.4
    np.savez_compressed(teacher, faces=faces,
                        rest_vertices_source_world=warped,
                        posed_vertices_source_world=np.stack([warped]))
    report, uv = evaluate(glb, teacher)
    assert uv is None
    assert report["reason"] == "SOURCE_TO_TEACHER_GEOMETRIC_IDENTITY_UNDETERMINED"


def test_source_bytes_sha_mismatch_fails_before_any_mapping(
        tmp_path, monkeypatch):
    glb, teacher, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="PARENT_BYTE_SHA_DRIFT"):
        gate.audit_alignment(glb=glb, teacher=teacher,
                             expected_glb_sha="0" * 64,
                             expected_teacher_sha=gate.sha256(teacher))


def test_multisurface_source_abstains_without_qualified_offset_map(
        tmp_path, monkeypatch):
    glb, teacher, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(gate, "source_arrays",
                        lambda _: (None, "MULTIPRIMITIVE_VERTEX_MAP_UNQUALIFIED"))
    report, uv = evaluate(glb, teacher)
    assert uv is None
    assert report["reason"] == "MULTIPRIMITIVE_VERTEX_MAP_UNQUALIFIED"
