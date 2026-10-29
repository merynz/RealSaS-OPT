from __future__ import annotations

import hashlib

import numpy as np

from tools.training.build_truth139_tessa_teacher_mesh_v1 import build_teacher_mesh
from tools.training.build_truth139_tessa_teacher_manifests_v1 import build_manifests


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_teacher_mesh_uses_exact_iris_to_tessa_half_scale_and_is_target_only():
    vertices = np.asarray(
        [[0.0, 0.0, 0.8], [-0.8, -0.8, -0.8], [0.8, -0.8, -0.8], [0.0, 0.8, -0.8]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
    arrays, receipt = build_teacher_mesh(
        subject_id="tetra",
        source_sha256="a" * 64,
        rest_vertices=vertices,
        faces=faces,
        normalization={
            "schema": "RealSaS.NormalizationDomainIR.v1",
            "center_xyz": [0.0, 0.0, 0.0],
            "half_extent": 1.0,
        },
    )
    assert np.allclose(arrays["vertices_normalized"], vertices * 0.5)
    assert np.array_equal(arrays["faces"], faces)
    assert receipt["teacher_topology_target_only"] is True
    assert receipt["student_conditioning_allowed"] is False
    assert receipt["representability_court"] == "PASS_FULL_SOURCE_TOPOLOGY_SEQUENCE"
    assert receipt["token_count"] > 0


def test_teacher_manifests_physically_separate_train_and_unseen(tmp_path):
    rows = []
    for subject in ("a", "b", "c"):
        mesh = tmp_path / f"{subject}_mesh.npz"
        np.savez(mesh, vertices_normalized=np.zeros((4, 3), np.float32), faces=np.zeros((1, 3), np.int64))
        receipt = tmp_path / f"{subject}_receipt.json"
        receipt.write_text("{}")
        rows.append({
            "subject_id": subject,
            "source_sha256": hashlib.sha256((subject + "source").encode()).hexdigest(),
            "teacher_mesh": {"path": str(mesh), "sha256": _sha(mesh)},
            "teacher_receipt": {"path": str(receipt), "sha256": _sha(receipt)},
        })
    split_identity = "1" * 64
    index = {
        "schema": "RealSaS.Truth139TESSATeacherIndex.v1",
        "split_identity_sha256": split_identity,
        "subjects": rows,
    }
    split = {
        "schema": "RealSaS.LineageSafeSubjectSplit.v1",
        "split_identity_sha256": split_identity,
        "train_subjects": ["a", "b"],
        "unseen_subjects": ["c"],
    }
    train, unseen = build_manifests(index, split)
    assert [row["subject_id"] for row in train["subjects"]] == ["a", "b"]
    assert [row["subject_id"] for row in unseen["subjects"]] == ["c"]
    assert "c_mesh" not in repr(train)
    assert train["student_conditioning_allowed"] is False
    assert unseen["optimizer_updates_allowed"] is False
