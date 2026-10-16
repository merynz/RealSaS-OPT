from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_vnext_fit8_final_preparation_receipt_binds_active_source_chain():
    source = _text("tools/iris_vnext/prepare_fit8_vnext_cohort_v1.py")
    required = (
        "canonical/VNEXT_FIT8_INPUT_BINDING_V1.json",
        "tools/iris_vnext/input_binding_v1.py",
        "tools/iris_vnext/blender_authored_teacher_export_v1.py",
        "tools/iris_vnext/blender_authored_teacher_export_v2.py",
        "tools/iris_vnext/blender_authored_teacher_export_v3.py",
        "tools/iris_vnext/export_fit8_authored_teachers_v1.py",
        "tools/iris_vnext/build_fit8_vnext_shards_v1.py",
        "tools/iris_vnext/build_fit8_vnext_shards_v2.py",
        "tools/iris_vnext/prepare_fit8_vnext_cohort_v1.py",
        "models/iris/vnext/cohort_v1.py",
        "models/iris/vnext/contracts_v1.py",
        "models/iris/vnext/topology_teacher_v1.py",
        "models/iris/vnext/teacher_truth_v1.py",
        "models/tessa/v1/model_v1.py",
        "models/tessa/v1/training_data_v1.py",
    )
    for path in required:
        assert f'"{path}"' in source
    assert "VNEXT_TEACHER_BATCH_EXPORTER_AUTHORITY_DRIFT" in source
    assert "VNEXT_TEACHER_BATCH_EXPORTER_SOURCE_HASH_DRIFT" in source
    assert '"active_preparation_source_provenance": source_provenance' in source
    assert '"source_commit_sha": source_commit_sha' in source
    assert '"source_commit_bound": source_commit_sha is not None' in source


def test_rank_deficient_mechanical_truth_is_bound_into_preparation_seal():
    prepare = _text("tools/iris_vnext/prepare_fit8_vnext_cohort_v1.py")
    truth = _text("models/iris/vnext/teacher_truth_v1.py")
    assert '"models/iris/vnext/teacher_truth_v1.py"' in prepare
    assert "3x2 deformation gradient" in truth
    assert "pose_degenerate_face_samples" in truth
    assert "DEGENERATE_BEND_CONSERVATIVE_PI" in truth
