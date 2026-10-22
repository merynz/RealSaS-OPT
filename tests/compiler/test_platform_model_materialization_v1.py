import hashlib
import json
from pathlib import Path

import pytest

from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.model_inputs import (
    behavior_identity_sha256,
    materialize_model_releases,
)


def _model(family, *, release_id, engine_release_id, artifact_id, content):
    content_sha = hashlib.sha256(content).hexdigest()
    return {
        "family": family,
        "model_release_id": release_id,
        "architecture_id": f"RealSaS.{family}.Test.v1",
        "model_engine_release_id": engine_release_id,
        "source_code_sha": "a" * 40,
        "checkpoint": {
            "id": artifact_id,
            "role": "model:" + family,
            "artifact_type": "RealSaS.ModelCheckpoint",
            "schema_version": "v1",
            "semantic_sha256": "c" * 64,
            "storage_key": "cas/sha256/" + content_sha,
            "content_sha256": content_sha,
            "size_bytes": len(content),
        },
    }


def test_behavior_identity_cross_language_vector():
    model = {
        "family": "IRIS",
        "model_release_id": "11111111-1111-1111-1111-111111111111",
        "architecture_id": "RealSaS.IRIS.Test.v1",
        "model_engine_release_id": "22222222-2222-2222-2222-222222222222",
        "source_code_sha": "a" * 40,
        "checkpoint": {
            "id": "33333333-3333-3333-3333-333333333333",
            "role": "model:IRIS",
            "artifact_type": "RealSaS.ModelCheckpoint",
            "schema_version": "v1",
            "semantic_sha256": "c" * 64,
            "storage_key": "cas/sha256/fixture",
            "content_sha256": "b" * 64,
            "size_bytes": 1,
        },
    }
    assert behavior_identity_sha256(model) == "bc50f4276691df4f83ff2c1336d862e41c9f3b087e5005e2881ebf06ed4a8e89"


def test_materializes_exact_bytes_canonically_without_mutating_disk_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("REALSAS_AUTHORITY_ROOT", str(tmp_path))
    manifest = {"run_id": "MODEL_RUN", "subject_id": "SUBJECT"}
    disk_manifest = tmp_path / "run_manifest.json"
    before = json.dumps(manifest, sort_keys=True).encode()
    disk_manifest.write_bytes(before)

    iris_bytes = b"iris-checkpoint-exact-bytes"
    tessa_bytes = b"tessa-checkpoint-exact-bytes"
    iris = _model(
        "IRIS",
        release_id="11111111-1111-1111-1111-111111111111",
        engine_release_id="21111111-1111-1111-1111-111111111111",
        artifact_id="31111111-1111-1111-1111-111111111111",
        content=iris_bytes,
    )
    tessa = _model(
        "TESSA",
        release_id="12222222-2222-2222-2222-222222222222",
        engine_release_id="22222222-2222-2222-2222-222222222222",
        artifact_id="32222222-2222-2222-2222-222222222222",
        content=tessa_bytes,
    )
    objects = {
        iris["checkpoint"]["content_sha256"]: iris_bytes,
        tessa["checkpoint"]["content_sha256"]: tessa_bytes,
    }

    def read(ref):
        return objects[ref["content_sha256"]]

    first = materialize_model_releases(manifest=manifest, models=[tessa, iris], read_object=read)
    first_snapshot = manifest["_platform_model_snapshot_sha256"]
    assert [row["family"] for row in first] == ["IRIS", "TESSA"]
    assert Path(first[0]["checkpoint"]["path"]).read_bytes() == iris_bytes
    assert Path(first[1]["checkpoint"]["path"]).read_bytes() == tessa_bytes
    assert "/models/IRIS/11111111-1111-1111-1111-111111111111/" in first[0]["checkpoint"]["path"].replace("\\", "/")
    assert disk_manifest.read_bytes() == before

    second_manifest = {"run_id": "MODEL_RUN", "subject_id": "SUBJECT"}
    second = materialize_model_releases(manifest=second_manifest, models=[iris, tessa], read_object=read)
    assert [row["behavior_identity_sha256"] for row in second] == [row["behavior_identity_sha256"] for row in first]
    assert second_manifest["_platform_model_snapshot_sha256"] == first_snapshot


def test_model_materialization_fails_closed_on_duplicate_corrupt_or_unsafe_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("REALSAS_AUTHORITY_ROOT", str(tmp_path))
    data = b"checkpoint"
    base = _model(
        "IRIS",
        release_id="11111111-1111-1111-1111-111111111111",
        engine_release_id="21111111-1111-1111-1111-111111111111",
        artifact_id="31111111-1111-1111-1111-111111111111",
        content=data,
    )

    with pytest.raises(RuntimeError, match="ENGINE_MODEL_DUPLICATE_FAMILY"):
        materialize_model_releases(
            manifest={"run_id": "DUP"},
            models=[base, dict(base)],
            read_object=lambda _: data,
        )

    with pytest.raises(RuntimeError, match="ENGINE_MODEL_CHECKPOINT_CAS_DRIFT"):
        materialize_model_releases(
            manifest={"run_id": "CORRUPT"},
            models=[base],
            read_object=lambda _: b"different-bytes",
        )

    unsafe = dict(base)
    unsafe["family"] = "../IRIS"
    unsafe["checkpoint"] = dict(base["checkpoint"], role="model:../IRIS")
    with pytest.raises(RuntimeError, match="ENGINE_MODEL_FAMILY_INVALID"):
        materialize_model_releases(
            manifest={"run_id": "UNSAFE"},
            models=[unsafe],
            read_object=lambda _: data,
        )


def test_empty_model_snapshot_preserves_legacy_manifest_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("REALSAS_AUTHORITY_ROOT", str(tmp_path))
    manifest = {
        "run_id": "LEGACY",
        "_platform_model_releases": ["stale"],
        "_platform_model_snapshot_sha256": "stale",
    }
    assert materialize_model_releases(manifest=manifest, models=[], read_object=lambda _: b"") == []
    assert "_platform_model_releases" not in manifest
    assert "_platform_model_snapshot_sha256" not in manifest


def _fingerprint(manifest):
    stage = {"id": "MODEL_STAGE", "depends_on": [], "manifest_keys": [], "policy": {"cacheable": True}}
    ledger = {
        "run_id": "MODEL_FINGERPRINT_RUN",
        "pipeline_plan_sha256": "d" * 64,
        "stages": [{"id": "MODEL_STAGE", "outputs": []}],
    }
    return mainline._fingerprint({}, ledger, manifest, stage, "e" * 64)[0]


def test_model_snapshot_changes_engine_stage_fingerprint_without_changing_legacy_fingerprint():
    legacy = _fingerprint({})
    assert _fingerprint({"_platform_model_releases": ["non-semantic-local-state"]}) == legacy

    model_a = _fingerprint({"_platform_model_snapshot_sha256": "1" * 64})
    model_b = _fingerprint({"_platform_model_snapshot_sha256": "2" * 64})
    assert model_a != legacy
    assert model_b != legacy
    assert model_a != model_b

    with pytest.raises(RuntimeError, match="MODEL_SNAPSHOT_SHA_INVALID"):
        _fingerprint({"_platform_model_snapshot_sha256": "not-a-sha"})
