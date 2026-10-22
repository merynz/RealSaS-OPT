from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from models.iris.vnext import product_checkpoint_v1 as checkpoint
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v3 import GraphicsRelationHeadV3Config
from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3


def _tiny_model(*, sealed: bool = True) -> IRISVNextEvidenceModelV3:
    cfg = IRISVNextConfigV1(
        scene_dim=64,
        camera_embed_dim=32,
        plane_size=4,
        plane_channels=16,
        field_hidden_dim=64,
        field_residual_blocks=2,
        local_feature_dim=64,
        attention_heads=8,
        scene_layers=2,
        topology_width=64,
        topology_heads=8,
        topology_decoder_layers=1,
        topology_memory_tokens=8,
        topology_refinement_layers=1,
        topology_source_width=32,
        extraction_training_resolution=8,
        surface_witness_count=16,
    )
    relation = GraphicsRelationHeadV3Config(
        feature_width=64,
        geometry_width=32,
        camera_width=32,
        pair_width=96,
        residual_blocks=2,
    )
    torch.manual_seed(909)
    model = IRISVNextEvidenceModelV3(cfg, relation)
    if sealed:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_payload(path: Path, payload: dict) -> str:
    torch.save(payload, path)
    return _sha(path)


def test_round_trip_is_exact_v3_eval_frozen_and_state_equal(tmp_path):
    model = _tiny_model()
    path = tmp_path / "iris-v3-product.pt"
    seal = checkpoint.save_product_checkpoint_v1(model, path)

    loaded = checkpoint.load_product_checkpoint_v1(
        path,
        expected_content_sha256=seal.content_sha256,
    )
    loaded.assert_inference_sealed()
    assert type(loaded.model) is IRISVNextEvidenceModelV3
    assert loaded.model.architecture_id == "RealSaS.IRIS.VNextEvidenceModel.v3"
    assert loaded.model.topology_decoder_embedded is False
    assert loaded.content_sha256 == seal.content_sha256
    assert loaded.state_schema_sha256 == seal.state_schema_sha256
    assert loaded.architecture_contract_sha256 == seal.architecture_contract_sha256

    original = model.state_dict()
    restored = loaded.model.state_dict()
    assert set(restored) == set(original)
    for name in original:
        assert torch.equal(restored[name], original[name]), name


def test_export_refuses_training_or_trainable_model():
    model = _tiny_model(sealed=False)
    with pytest.raises(RuntimeError, match="IRIS_VNEXT_PRODUCT_EXPORT_MODEL_MUST_BE_EVAL"):
        checkpoint.product_checkpoint_payload_v1(model)

    model.eval()
    with pytest.raises(RuntimeError, match="IRIS_VNEXT_PRODUCT_EXPORT_MODEL_MUST_BE_FROZEN"):
        checkpoint.product_checkpoint_payload_v1(model)


def test_product_payload_has_only_behavioral_model_state():
    payload = checkpoint.product_checkpoint_payload_v1(_tiny_model())
    assert set(payload) == {
        "schema",
        "architecture_id",
        "config",
        "relation_config",
        "state_schema_sha256",
        "architecture_contract_sha256",
        "model_state",
    }
    forbidden = {"optimizer", "torch_rng", "cuda_rng", "teacher", "training_scope", "step"}
    assert forbidden.isdisjoint(payload)
    assert payload["schema"] == checkpoint.PRODUCT_CHECKPOINT_SCHEMA_V1
    assert payload["architecture_id"] == IRISVNextEvidenceModelV3.architecture_id


def test_loader_rejects_legacy_v8_and_v9_training_envelopes(tmp_path):
    for name, payload in (
        (
            "v8.pt",
            {
                "schema": "RealSaS.IRISVNextFit8Checkpoint.v8",
                "model": _tiny_model().state_dict(),
                "optimizer": {},
                "step": 3200,
            },
        ),
        (
            "v9.pt",
            {
                "schema": "RealSaS.IRISVNextFocusedRelationCheckpoint.v1",
                "relation_state": _tiny_model().relations.state_dict(),
                "optimizer": {},
                "step": 1,
            },
        ),
    ):
        path = tmp_path / name
        digest = _save_payload(path, payload)
        with pytest.raises(
            RuntimeError,
            match="IRIS_VNEXT_PRODUCT_CHECKPOINT_(FIELDS|SCHEMA)_INVALID",
        ):
            checkpoint.load_product_checkpoint_v1(path, expected_content_sha256=digest)


def test_state_schema_is_rederived_before_strict_load(tmp_path):
    payload = checkpoint.product_checkpoint_payload_v1(_tiny_model())
    first_key = sorted(payload["model_state"])[0]
    del payload["model_state"][first_key]
    path = tmp_path / "missing-state-key.pt"
    digest = _save_payload(path, payload)

    with pytest.raises(RuntimeError, match="IRIS_VNEXT_PRODUCT_STATE_SCHEMA_DRIFT"):
        checkpoint.load_product_checkpoint_v1(path, expected_content_sha256=digest)


def test_architecture_contract_is_rederived_from_configs(tmp_path):
    payload = checkpoint.product_checkpoint_payload_v1(_tiny_model())
    payload["relation_config"] = dict(payload["relation_config"])
    payload["relation_config"]["pair_width"] += 8
    path = tmp_path / "config-drift.pt"
    digest = _save_payload(path, payload)

    with pytest.raises(RuntimeError, match="IRIS_VNEXT_PRODUCT_ARCHITECTURE_CONTRACT_DRIFT"):
        checkpoint.load_product_checkpoint_v1(path, expected_content_sha256=digest)


def test_extra_training_baggage_is_not_admitted(tmp_path):
    payload = checkpoint.product_checkpoint_payload_v1(_tiny_model())
    payload["optimizer"] = {"state": {}}
    path = tmp_path / "baggage.pt"
    digest = _save_payload(path, payload)

    with pytest.raises(RuntimeError, match="IRIS_VNEXT_PRODUCT_CHECKPOINT_FIELDS_INVALID"):
        checkpoint.load_product_checkpoint_v1(path, expected_content_sha256=digest)


def test_content_sha_is_verified_before_payload_admission(tmp_path):
    model = _tiny_model()
    path = tmp_path / "content.pt"
    checkpoint.save_product_checkpoint_v1(model, path)

    with pytest.raises(RuntimeError, match="IRIS_VNEXT_PRODUCT_CHECKPOINT_CONTENT_DRIFT"):
        checkpoint.load_product_checkpoint_v1(
            path,
            expected_content_sha256="0" * 64,
        )


def test_loader_uses_weights_only(monkeypatch, tmp_path):
    model = _tiny_model()
    path = tmp_path / "weights-only.pt"
    seal = checkpoint.save_product_checkpoint_v1(model, path)
    original_load = checkpoint.torch.load
    observed = {}

    def guarded_load(*args, **kwargs):
        observed["weights_only"] = kwargs.get("weights_only")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", guarded_load)
    checkpoint.load_product_checkpoint_v1(
        path,
        expected_content_sha256=seal.content_sha256,
    )
    assert observed == {"weights_only": True}
