from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.runtime_checkpoint_v1 import (
    RSSvNextCheckpointRefV1, load_rss_vnext_runtime_checkpoint_v1,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, architecture_digest,
)


@pytest.fixture(autouse=True)
def setup_seed():
    torch.manual_seed(1337)
    torch.set_num_threads(1)


def _model():
    cfg = RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )


def _write(tmp_path, *, step=1, missing_key=False):
    source = _model()
    raw = source.state_dict()
    if missing_key:
        raw.pop("iris.scene.scene_queries")
    payload = {
        "schema": RSSvNextStructuralTrainer.checkpoint_schema,
        "architecture_id": source.architecture_id,
        "architecture_digest": architecture_digest(source.cfg),
        "dataset_manifest_sha256": "c" * 64,
        "model": raw,
        "optimizer": {},
        "global_step": step,
        "sampler_cursor": 12,
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    ref = RSSvNextCheckpointRefV1(
        checkpoint_path=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        architecture_id=source.architecture_id,
        architecture_digest=architecture_digest(source.cfg),
        dataset_manifest_sha256="c" * 64,
        backbone_artifact_sha256="d" * 64,
    )
    return ref


def test_exact_research_checkpoint_loads_strict_and_never_mints_product(tmp_path):
    ref = _write(tmp_path)
    loaded = load_rss_vnext_runtime_checkpoint_v1(
        _model(), ref, actual_backbone_sha256="d" * 64,
    )
    assert loaded.qualification == "UNQUALIFIED_RESEARCH_CHECKPOINT"
    assert not loaded.model.training
    assert loaded.checkpoint_sha256 == ref.checkpoint_sha256
    with pytest.raises(RuntimeError, match="PRODUCT_RELEASE_VERIFICATION_REQUIRED"):
        loaded.require_product_model_release()


def test_wrong_bytes_backbone_config_and_dataset_rejected(tmp_path):
    ref = _write(tmp_path)
    with pytest.raises(ValueError, match="CHECKPOINT_BYTES_DRIFT"):
        load_rss_vnext_runtime_checkpoint_v1(
            _model(), replace(ref, checkpoint_sha256="e" * 64),
            actual_backbone_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="BACKBONE_BYTES_DRIFT"):
        load_rss_vnext_runtime_checkpoint_v1(
            _model(), ref, actual_backbone_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="CONFIG_DIGEST_DRIFT"):
        load_rss_vnext_runtime_checkpoint_v1(
            _model(), replace(ref, architecture_digest="e" * 64),
            actual_backbone_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="TRAINER_DATASET_DRIFT"):
        load_rss_vnext_runtime_checkpoint_v1(
            _model(), replace(ref, dataset_manifest_sha256="e" * 64),
            actual_backbone_sha256="d" * 64,
        )


def test_untrained_and_incompatible_state_dict_cannot_serve(tmp_path):
    ref = _write(tmp_path, step=0)
    with pytest.raises(ValueError, match="UNTRAINED_CHECKPOINT_FORBIDDEN"):
        load_rss_vnext_runtime_checkpoint_v1(
            _model(), ref, actual_backbone_sha256="d" * 64,
        )
    ref = _write(tmp_path, missing_key=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_rss_vnext_runtime_checkpoint_v1(
            _model(), ref, actual_backbone_sha256="d" * 64,
        )


def test_checkpoint_claims_do_not_create_product_model_release(tmp_path):
    ref = _write(tmp_path)
    pretend = replace(ref, model_release_id="MYMODEL", admission_receipt_hash="e"*64)
    with pytest.raises(RuntimeError, match="PRODUCT_TEST_BACKBONE_FORBIDDEN"):
        load_rss_vnext_runtime_checkpoint_v1(
            _model(), pretend, actual_backbone_sha256="d" * 64,
            verify_platform_model_release=lambda _: True,
        )
    with pytest.raises(ValueError, match="ORPHAN_VERIFIER"):
        load_rss_vnext_runtime_checkpoint_v1(
            _model(), ref, actual_backbone_sha256="d" * 64,
            verify_platform_model_release=lambda _: True,
        )
    with pytest.raises(ValueError, match="ADMISSION_PAIR_REQUIRED"):
        replace(ref, admission_receipt_hash="e"*64).validate()


def test_symlink_and_non_file_rejected(tmp_path):
    ref = _write(tmp_path)
    symlink = tmp_path / "moving.pt"
    symlink.symlink_to(tmp_path / "checkpoint.pt")
    with pytest.raises(ValueError, match="REGULAR_FILE_REQUIRED"):
        load_rss_vnext_runtime_checkpoint_v1(
            _model(), replace(ref, checkpoint_path=str(symlink)),
            actual_backbone_sha256="d" * 64,
        )
