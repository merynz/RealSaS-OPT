from __future__ import annotations

import json

import pytest

from tools.corpus.build_truth139_materialization_work_orders_v1 import build
from tools.corpus.run_truth139_graphics_materialization_subject_v1 import (
    DENSE_JOB,
    VISIBILITY_JOB,
    build_command_plan,
    validate_request,
)
from tools.corpus.seal_truth139_materialization_receipt_v1 import REPO_ROOT, checked_out_git_sha, file_sha256


def _work(tmp_path, raw_source):
    subject = {
        "subject_id": "truth139_000",
        "subject_identity_signature": "identity-000",
        "source": {
            "corpus_relative_path": "raw/quaternius/pack/glTF/truth139_000.gltf",
            "sha256": file_sha256(raw_source),
            "byte_size": raw_source.stat().st_size,
            "format": "GLTF",
            "creator": "quaternius",
            "pack_id": "quaternius-pack",
            "dependencies": [],
            "dependency_count": 0,
            "source_page_snapshot": {"path": "provenance/source_pages/pack.html", "sha256": "a" * 64},
            "official_url": "https://example.invalid",
            "drive_folder_id": "folder",
            "source_family_signature": ["SRCF:000"],
        },
        "raw_capabilities": {
            "OBSERVATION_FSTAR": True,
            "SURFACE_NORMAL_TRUTH": True,
            "STAGE47_MECHANICS": True,
            "STAGE48_APPEARANCE_PRESENTATION": True,
            "TESSA_TOPOLOGY_TEACHER": True,
        },
        "development_or_regression_tainted": False,
    }
    manifest = {
        "schema": "RealSaS.Truth139CorpusManifest.v2",
        "corpus_manifest_identity_sha256": "b" * 64,
        "historical_reproduction": {"passed": False, "operational_blocker": False},
        "subjects": [subject],
    }
    payload = build(manifest)
    path = tmp_path / "work_orders.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path, payload


def _request(tmp_path):
    raw_source = tmp_path / "source.gltf"
    raw_source.write_text('{"asset":{"version":"2.0"}}')
    geometry = tmp_path / "geometry.npz"
    geometry.write_bytes(b"geometry-placeholder")
    scene = tmp_path / "scene.json"
    scene.write_text("{}")
    camera = tmp_path / "camera.json"
    camera.write_text("{}")
    observations = tmp_path / "observations"
    observations.mkdir()
    work_path, work = _work(tmp_path, raw_source)
    request = validate_request(
        work_orders_path=work_path,
        subject_id="truth139_000",
        raw_source_path=raw_source,
        geometry_npz=geometry,
        source_scene_snapshot=scene,
        camera_bundle=camera,
        observation_dir=observations,
        output_root=tmp_path / "out",
        code_commit_sha=checked_out_git_sha(REPO_ROOT),
    )
    return request, work, raw_source


def test_plan_binds_canonical_dense_visibility_courts_and_receipts(tmp_path):
    request, work, _raw = _request(tmp_path)
    plan = build_command_plan(request, subject_id="truth139_000", device="cuda")
    assert [row["name"] for row in plan] == [
        "DENSE_GRAPHICS_FRONT_SURFACE",
        "DENSE_GRAPHICS_GEOMETRY_COURT",
        "DENSE_GRAPHICS_EXECUTION_RECEIPT",
        "VISIBILITY_OCCLUSION_CROSS_VIEW",
        "VISIBILITY_OCCLUSION_CPU_COURT",
        "VISIBILITY_OCCLUSION_EXECUTION_RECEIPT",
    ]
    dense = work["canonical_implementations"][DENSE_JOB]
    visibility = work["canonical_implementations"][VISIBILITY_JOB]
    assert dense["independent_court"] == "tools/corpus/validate_truth139_dense_graphics_geometry_v1.py"
    assert visibility["producer"] == "tools/corpus/build_truth139_visibility_occlusion_teacher_v2.py"

    dense_command = plan[0]["command"]
    assert dense_command[1].endswith(dense["producer"])
    assert "--observation-provenance" in dense_command
    assert dense_command[dense_command.index("--observation-provenance") + 1] == "SOURCE_OBSERVATION"

    dense_court_command = plan[1]["command"]
    assert dense_court_command[1].endswith(dense["independent_court"])
    assert "--dense-graphics-manifest" in dense_court_command

    visibility_command = plan[3]["command"]
    assert visibility_command[1].endswith(visibility["producer"])
    visibility_court_command = plan[4]["command"]
    assert visibility_court_command[1].endswith(visibility["independent_court"])
    assert "--teacher-manifest" in visibility_court_command

    for receipt_index in (2, 5):
        command = plan[receipt_index]["command"]
        assert command[1].endswith("tools/corpus/seal_truth139_materialization_receipt_v1.py")
        assert command[command.index("--code-commit-sha") + 1] == checked_out_git_sha(REPO_ROOT)


def test_request_rejects_raw_source_byte_drift(tmp_path):
    request, _work, raw_source = _request(tmp_path)
    work_path = request["paths"]["work_orders"]
    raw_source.write_text("mutated-after-seal")
    with pytest.raises(ValueError, match="TRUTH139_GRAPHICS_WORKER_RAW_SOURCE_SHA"):
        validate_request(
            work_orders_path=work_path,
            subject_id="truth139_000",
            raw_source_path=raw_source,
            geometry_npz=request["paths"]["geometry_npz"],
            source_scene_snapshot=request["paths"]["source_scene_snapshot"],
            camera_bundle=request["paths"]["camera_bundle"],
            observation_dir=request["paths"]["observation_dir"],
            output_root=request["paths"]["output_root"],
            code_commit_sha=checked_out_git_sha(REPO_ROOT),
        )
