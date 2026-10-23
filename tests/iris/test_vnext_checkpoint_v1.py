from __future__ import annotations

import torch
import pytest

from models.iris.vnext.checkpoint_v1 import (
    CHECKPOINT_SCHEMA_V1,
    checkpoint_payload_v1,
    load_checkpoint_v1,
    write_checkpoint_v1,
)
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v3 import GraphicsRelationHeadV3Config
from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3


def _model(seed: int = 17) -> IRISVNextEvidenceModelV3:
    torch.manual_seed(seed)
    config = IRISVNextConfigV1(
        scene_dim=16,
        camera_embed_dim=8,
        plane_size=2,
        plane_channels=4,
        field_hidden_dim=16,
        field_residual_blocks=1,
        local_feature_dim=16,
        attention_heads=4,
        scene_layers=1,
        feedforward_multiplier=2,
        topology_width=16,
        topology_heads=4,
        topology_decoder_layers=1,
        topology_memory_tokens=4,
        topology_refinement_layers=0,
        topology_source_width=8,
        appearance_descriptor_dim=8,
        relation_embedding_dim=8,
        extraction_training_resolution=4,
        surface_witness_count=8,
    )
    relation = GraphicsRelationHeadV3Config(
        feature_width=16,
        geometry_width=8,
        camera_width=8,
        pair_width=24,
        residual_blocks=1,
    )
    return IRISVNextEvidenceModelV3(config, relation)


def test_roundtrip_is_exact_and_inference_frozen(tmp_path):
    source = _model()
    path = write_checkpoint_v1(tmp_path / "iris.pt", source)
    loaded = load_checkpoint_v1(path)

    assert loaded.architecture_id == IRISVNextEvidenceModelV3.architecture_id
    assert loaded.training is False
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert set(source.state_dict()) == set(loaded.state_dict())
    for key, tensor in source.state_dict().items():
        assert torch.equal(tensor.cpu(), loaded.state_dict()[key].cpu())


def test_payload_is_model_state_only_not_training_state():
    payload = checkpoint_payload_v1(_model())
    assert set(payload) == {
        "schema",
        "architecture_id",
        "config",
        "relation_config",
        "model_state_dict",
    }
    assert payload["schema"] == CHECKPOINT_SCHEMA_V1
    assert "optimizer" not in payload
    assert "torch_rng" not in payload
    assert "stage_index" not in payload
    assert "step" not in payload


def test_training_checkpoint_shape_is_rejected(tmp_path):
    path = tmp_path / "legacy-v8.pt"
    torch.save(
        {
            "schema": "RealSaS.IRISVNextFit8Checkpoint.v4",
            "model": _model().state_dict(),
            "optimizer": {},
            "step": 12000,
        },
        path,
    )
    with pytest.raises(RuntimeError, match="IRIS_VNEXT_CHECKPOINT_PAYLOAD_INVALID"):
        load_checkpoint_v1(path)


def test_wrong_architecture_is_rejected(tmp_path):
    payload = checkpoint_payload_v1(_model())
    payload["architecture_id"] = "RealSaS.IRIS.VNext.v1.1"
    path = tmp_path / "wrong-arch.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="IRIS_VNEXT_CHECKPOINT_ARCHITECTURE_DRIFT"):
        load_checkpoint_v1(path)


def test_missing_state_key_is_rejected_strictly(tmp_path):
    payload = checkpoint_payload_v1(_model())
    state = dict(payload["model_state_dict"])
    state.pop(next(iter(state)))
    payload["model_state_dict"] = state
    path = tmp_path / "missing-state.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="IRIS_VNEXT_CHECKPOINT_STATE_DRIFT"):
        load_checkpoint_v1(path)


def test_extra_training_metadata_is_rejected(tmp_path):
    payload = checkpoint_payload_v1(_model())
    payload["optimizer"] = {"forbidden": True}
    path = tmp_path / "extra-training-state.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="IRIS_VNEXT_CHECKPOINT_PAYLOAD_INVALID"):
        load_checkpoint_v1(path)
