from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from compiler.realsas_compiler_services.platform_worker import model_binding, model_inputs


FAMILY = "IRIS"
ARCHITECTURE = "RealSaS.IRIS.VNextEvidenceModel.v3"
MODEL_RELEASE_ID = "11111111-1111-4111-8111-111111111111"
ENGINE_RELEASE_ID = "22222222-2222-4222-8222-222222222222"
ARTIFACT_ID = "33333333-3333-4333-8333-333333333333"


def _manifest(root: Path, payload: bytes = b"checkpoint") -> dict:
    run_id = "run-model-binding-v1"
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    path = root / "runs" / run_id / "models" / FAMILY / MODEL_RELEASE_ID / f"{checkpoint_sha}.checkpoint"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)

    identity_shape = {
        "family": FAMILY,
        "model_release_id": MODEL_RELEASE_ID,
        "architecture_id": ARCHITECTURE,
        "model_engine_release_id": ENGINE_RELEASE_ID,
        "source_code_sha": "a" * 40,
        "checkpoint": {
            "id": ARTIFACT_ID,
            "content_sha256": checkpoint_sha,
        },
    }
    behavior_sha = model_inputs.behavior_identity_sha256(identity_shape)
    row = {
        "schema": "RealSaS.ModelReleaseBehaviorIdentity.v1",
        "family": FAMILY,
        "model_release_id": MODEL_RELEASE_ID,
        "architecture_id": ARCHITECTURE,
        "model_engine_release_id": ENGINE_RELEASE_ID,
        "source_code_sha": "a" * 40,
        "checkpoint_artifact_id": ARTIFACT_ID,
        "checkpoint_content_sha256": checkpoint_sha,
        "behavior_identity_sha256": behavior_sha,
        "checkpoint": {
            "artifact_type": "RealSaS.ModelCheckpoint",
            "schema_version": "v1",
            "semantic_sha256": "d" * 64,
            "content_sha256": checkpoint_sha,
            "size_bytes": len(payload),
            "path": str(path.resolve()),
        },
    }
    snapshot_sha = model_binding.mainline.content_sha256(
        {
            "schema": "RealSaS.ModelReleaseSnapshot.v1",
            "models": [
                {
                    "family": FAMILY,
                    "behavior_identity_sha256": behavior_sha,
                }
            ],
        }
    )
    return {
        "run_id": run_id,
        "_platform_model_snapshot_sha256": snapshot_sha,
        "_platform_model_releases": [row],
    }


def test_resolves_only_exact_materialized_attempt_model(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    manifest = _manifest(root)

    ref = model_binding.require_materialized_model_release(
        manifest, family=FAMILY, architecture_id=ARCHITECTURE
    )
    assert ref.family == FAMILY
    assert ref.architecture_id == ARCHITECTURE
    assert ref.model_release_id == MODEL_RELEASE_ID
    assert ref.behavior_identity_sha256 == manifest["_platform_model_releases"][0]["behavior_identity_sha256"]
    assert ref.checkpoint_path.is_file()
    assert ref.checkpoint_content_sha256 == hashlib.sha256(b"checkpoint").hexdigest()


def test_wrong_architecture_fails_closed(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    manifest = _manifest(root)
    with pytest.raises(RuntimeError, match="ENGINE_MODEL_BINDING_ARCHITECTURE_DRIFT"):
        model_binding.require_materialized_model_release(
            manifest,
            family=FAMILY,
            architecture_id="RealSaS.IRIS.SomeOtherArchitecture.v1",
        )


def test_duplicate_family_is_ambiguous_even_if_rows_are_otherwise_identical(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    manifest = _manifest(root)
    manifest["_platform_model_releases"].append(dict(manifest["_platform_model_releases"][0]))
    with pytest.raises(RuntimeError, match="ENGINE_MODEL_BINDING_FAMILY_AMBIGUOUS"):
        model_binding.require_materialized_model_release(
            manifest, family=FAMILY, architecture_id=ARCHITECTURE
        )


def test_behavior_identity_is_rederived_not_trusted(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    manifest = _manifest(root)
    manifest["_platform_model_releases"][0]["source_code_sha"] = "e" * 40
    with pytest.raises(RuntimeError, match="ENGINE_MODEL_BINDING_BEHAVIOR_IDENTITY_DRIFT"):
        model_binding.require_materialized_model_release(
            manifest, family=FAMILY, architecture_id=ARCHITECTURE
        )


def test_whole_model_snapshot_hash_is_rederived(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    manifest = _manifest(root)
    manifest["_platform_model_snapshot_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="ENGINE_MODEL_BINDING_SNAPSHOT_DRIFT"):
        model_binding.require_materialized_model_release(
            manifest, family=FAMILY, architecture_id=ARCHITECTURE
        )


def test_checkpoint_bytes_are_reverified_at_consumption(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    manifest = _manifest(root)
    path = Path(manifest["_platform_model_releases"][0]["checkpoint"]["path"])
    path.write_bytes(b"checkpoinx")
    with pytest.raises(RuntimeError, match="ENGINE_MODEL_BINDING_CHECKPOINT_CONTENT_DRIFT"):
        model_binding.require_materialized_model_release(
            manifest, family=FAMILY, architecture_id=ARCHITECTURE
        )


def test_checkpoint_path_cannot_escape_run_local_model_root(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    manifest = _manifest(root)
    outside = tmp_path / "outside.checkpoint"
    outside.write_bytes(b"checkpoint")
    manifest["_platform_model_releases"][0]["checkpoint"]["path"] = str(outside.resolve())
    with pytest.raises(RuntimeError, match="ENGINE_MODEL_BINDING_CHECKPOINT_PATH_ESCAPE"):
        model_binding.require_materialized_model_release(
            manifest, family=FAMILY, architecture_id=ARCHITECTURE
        )


def test_binding_source_consumes_attempt_snapshot_not_live_channel_registry():
    source = Path(
        "compiler/realsas_compiler_services/platform_worker/model_binding.py"
    ).read_text(encoding="utf-8")
    assert "_platform_model_releases" in source
    assert "model_channels" not in source
    assert "GetModelChannel" not in source
    assert "ResolveProduction" not in source
