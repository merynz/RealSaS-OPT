from __future__ import annotations

import pytest
import torch

from models.iris.v31.checkpoint_v1 import (
    CHECKPOINT_SCHEMA_V1 as IRIS_CHECKPOINT_SCHEMA_V1,
    checkpoint_payload_v1 as iris_checkpoint_payload_v1,
    load_checkpoint_v1 as load_iris_checkpoint_v1,
    write_checkpoint_v1 as write_iris_checkpoint_v1,
)
from models.iris.v31.model_v31 import IRISConfigV31, IRISV31
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v31.attention_v31 import SEMANTICS_ID
from models.tessa.v31.checkpoint_v1 import (
    CHECKPOINT_SCHEMA_V1 as TESSA_CHECKPOINT_SCHEMA_V1,
    checkpoint_payload_v1 as tessa_checkpoint_payload_v1,
    load_checkpoint_v1 as load_tessa_checkpoint_v1,
    write_checkpoint_v1 as write_tessa_checkpoint_v1,
)
from models.tessa.v31.model_v31 import TESSAStructuralizerV31


def _freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def _iris(seed: int = 31) -> IRISV31:
    torch.manual_seed(seed)
    return _freeze(
        IRISV31(
            IRISConfigV31(
                carrier_resolution=2,
                carrier_channels=4,
                mechanical_channels=("stretch_major", "bend_demand"),
                pair_channels=("front_relation",),
                lift_chunk=4,
            )
        )
    )


def _tessa(seed: int = 41, *, backend: str = "SDPA") -> TESSAStructuralizerV31:
    torch.manual_seed(seed)
    config = TESSAStructuralizerConfigV3(
        d_model=16,
        n_heads=4,
        decoder_layers=1,
        mlp_ratio=2,
        dropout=0.0,
        coordinate_bins=128,
        max_faces=32,
        max_vertices=32,
        scene_memory_tokens=2,
        surface_memory_tokens=2,
        presentation_memory_tokens=2,
        conditioning_token_budget=6,
        memory_source_width=8,
        memory_refinement_layers=0,
        local_attention_window=8,
        query_chunk_size=4,
        gradient_checkpointing=False,
    )
    return _freeze(
        TESSAStructuralizerV31(
            scene_input_dim=6,
            surface_input_dim=7,
            presentation_input_dim=8,
            config=config,
            backend=backend,
        )
    )


def _assert_state_equal(source, loaded):
    assert set(source.state_dict()) == set(loaded.state_dict())
    for key, tensor in source.state_dict().items():
        assert torch.equal(tensor.cpu(), loaded.state_dict()[key].cpu())


def test_iris_v31_checkpoint_roundtrip_is_exact_and_inference_frozen(tmp_path):
    source = _iris()
    path = write_iris_checkpoint_v1(tmp_path / "iris-v31.pt", source)
    loaded = load_iris_checkpoint_v1(path)

    assert loaded.architecture_id == IRISV31.architecture_id
    assert loaded.training is False
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    _assert_state_equal(source, loaded)


def test_iris_v31_payload_is_inference_state_only_and_strict(tmp_path):
    payload = iris_checkpoint_payload_v1(_iris())
    assert payload["schema"] == IRIS_CHECKPOINT_SCHEMA_V1
    assert set(payload) == {"schema", "architecture_id", "config", "model_state_dict"}
    assert "optimizer" not in payload and "step" not in payload and "rng" not in payload

    payload["optimizer"] = {}
    path = tmp_path / "iris-extra-training-state.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="IRIS_V31_CHECKPOINT_PAYLOAD_INVALID"):
        load_iris_checkpoint_v1(path)


def test_tessa_v31_checkpoint_roundtrip_binds_attention_semantics_and_backend(tmp_path):
    source = _tessa()
    payload = tessa_checkpoint_payload_v1(source)
    assert payload["schema"] == TESSA_CHECKPOINT_SCHEMA_V1
    assert payload["attention_semantics"] == SEMANTICS_ID
    assert payload["backend"] == "SDPA"
    assert "optimizer" not in payload and "step" not in payload and "rng" not in payload

    path = write_tessa_checkpoint_v1(tmp_path / "tessa-v31.pt", source)
    loaded = load_tessa_checkpoint_v1(path)
    assert loaded.architecture_id == TESSAStructuralizerV31.architecture_id
    assert loaded.attention_semantics == SEMANTICS_ID
    assert {block.self_attn.backend for block in loaded.decoder} == {"SDPA"}
    assert loaded.training is False
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    _assert_state_equal(source, loaded)


def test_tessa_v31_attention_semantics_drift_is_rejected(tmp_path):
    payload = tessa_checkpoint_payload_v1(_tessa())
    payload["attention_semantics"] = "LEGACY_CHUNK_END_ANCHORED"
    path = tmp_path / "tessa-wrong-semantics.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="TESSA_V31_CHECKPOINT_ATTENTION_SEMANTICS_DRIFT"):
        load_tessa_checkpoint_v1(path)


def test_tessa_v31_backend_identity_is_exact(tmp_path):
    payload = tessa_checkpoint_payload_v1(_tessa())
    payload["backend"] = "UNKNOWN"
    path = tmp_path / "tessa-wrong-backend.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="TESSA_V31_CHECKPOINT_BACKEND_INVALID"):
        load_tessa_checkpoint_v1(path)
