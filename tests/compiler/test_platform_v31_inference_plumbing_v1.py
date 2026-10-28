from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from compiler.realsas_compiler_services.platform_worker import model_binding, model_inputs
from compiler.realsas_compiler_services.platform_worker.iris_v31_inference_v1 import (
    encode_dense_scene_v31,
    load_iris_v31_inference_v1,
)
from compiler.realsas_compiler_services.platform_worker.scene_input_contract_v31 import (
    decode_shape_input_v31,
    encode_shape_input_v31,
)
from compiler.realsas_compiler_services.platform_worker.tessa_v31_inference_v1 import (
    TESSAV31InferenceInputsV1,
    encode_tessa_memory_v31,
    load_tessa_v31_inference_v1,
)
from models.iris.v2.dinov2_foundation_v2 import DINOv2FoundationAuthorityV2
from models.iris.v31.checkpoint_v1 import write_checkpoint_v1 as write_iris_checkpoint_v31
from models.iris.v31.inference_input_v1 import IRISV31SceneInputsV1
from models.iris.v31.model_v31 import IRISConfigV31, IRISV31
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v31.checkpoint_v1 import write_checkpoint_v1 as write_tessa_checkpoint_v31
from models.tessa.v31.model_v31 import TESSAStructuralizerV31


ENGINE_RELEASE_ID = "22222222-2222-4222-8222-222222222222"
ARTIFACT_ID = "33333333-3333-4333-8333-333333333333"


def _freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def _manifest(root: Path, *, family: str, architecture: str, model, writer, release_id: str) -> dict:
    run_id = "run-v31-inference-v1-" + family.lower()
    path = root / "runs" / run_id / "models" / family / release_id / "checkpoint.pt"
    writer(path, model)
    payload = path.read_bytes()
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    identity_shape = {
        "family": family,
        "model_release_id": release_id,
        "architecture_id": architecture,
        "model_engine_release_id": ENGINE_RELEASE_ID,
        "source_code_sha": "a" * 40,
        "checkpoint": {"id": ARTIFACT_ID, "content_sha256": checkpoint_sha},
    }
    behavior_sha = model_inputs.behavior_identity_sha256(identity_shape)
    row = {
        "schema": "RealSaS.ModelReleaseBehaviorIdentity.v1",
        "family": family,
        "model_release_id": release_id,
        "architecture_id": architecture,
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
            "models": [{"family": family, "behavior_identity_sha256": behavior_sha}],
        }
    )
    return {
        "run_id": run_id,
        "_platform_model_snapshot_sha256": snapshot_sha,
        "_platform_model_releases": [row],
    }


def _camera_features() -> torch.Tensor:
    row = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ],
        dtype=torch.float32,
    )
    return row.reshape(1, 1, 13).expand(1, 8, 13).clone()


def _iris_inputs() -> IRISV31SceneInputsV1:
    return IRISV31SceneInputsV1(
        final=torch.zeros(1, 8, 37, 37, 384, dtype=torch.float32),
        layer2=torch.zeros(1, 8, 37, 37, 384, dtype=torch.float32),
        alpha=torch.zeros(1, 8, 37, 37, 196, dtype=torch.float32),
        camera_features=_camera_features(),
        foundation_authority_sha256=DINOv2FoundationAuthorityV2().authority_hash,
        observation_bundle_sha256="1" * 64,
        camera_bundle_sha256="2" * 64,
        alpha_resampling_policy_sha256="3" * 64,
        numerical_environment_sha256="4" * 64,
    )


def _iris_model() -> IRISV31:
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


def _tessa_model() -> TESSAStructuralizerV31:
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
            backend="SDPA",
        )
    )


def test_v31_shape_input_transport_is_pickle_free_and_exact():
    source = _iris_inputs()
    payload = encode_shape_input_v31(source)
    decoded = decode_shape_input_v31(payload)
    assert torch.equal(decoded.final, source.final)
    assert torch.equal(decoded.layer2, source.layer2)
    assert torch.equal(decoded.alpha, source.alpha)
    assert torch.equal(decoded.camera_features, source.camera_features)
    assert decoded.alpha_resampling_policy_sha256 == source.alpha_resampling_policy_sha256
    assert decoded.observation_bundle_sha256 == source.observation_bundle_sha256


def test_attempt_bound_iris_v31_checkpoint_executes_dense_carrier(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    model = _iris_model()
    manifest = _manifest(
        root,
        family="IRIS",
        architecture=IRISV31.architecture_id,
        model=model,
        writer=write_iris_checkpoint_v31,
        release_id="11111111-1111-4111-8111-111111111111",
    )
    binding = load_iris_v31_inference_v1(manifest)
    first = encode_dense_scene_v31(binding, _iris_inputs())
    second = encode_dense_scene_v31(binding, _iris_inputs())
    assert tuple(first.shape) == (1, 4, 2, 2, 2)
    assert torch.equal(first, second)
    assert binding.model.training is False
    assert not any(parameter.requires_grad for parameter in binding.model.parameters())


def test_attempt_bound_tessa_v31_checkpoint_encodes_exact_memory(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    model = _tessa_model()
    manifest = _manifest(
        root,
        family="TESSA",
        architecture=TESSAStructuralizerV31.architecture_id,
        model=model,
        writer=write_tessa_checkpoint_v31,
        release_id="55555555-5555-4555-8555-555555555555",
    )
    binding = load_tessa_v31_inference_v1(manifest)
    inputs = TESSAV31InferenceInputsV1(
        scene_source=torch.zeros(1, 3, 6),
        surface_source=torch.zeros(1, 4, 7),
        presentation_source=torch.zeros(1, 2, 8),
        surface_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        presentation_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    memory = encode_tessa_memory_v31(binding, inputs)
    assert tuple(memory.shape) == (1, 6, 16)
    assert torch.isfinite(memory).all()
    assert {block.self_attn.backend for block in binding.model.decoder} == {"SDPA"}


def test_v31_runtime_plumbing_has_no_live_channel_or_training_authority():
    for path in (
        "compiler/realsas_compiler_services/platform_worker/iris_v31_inference_v1.py",
        "compiler/realsas_compiler_services/platform_worker/tessa_v31_inference_v1.py",
        "compiler/realsas_compiler_services/platform_worker/scene_input_contract_v31.py",
    ):
        source = Path(path).read_text(encoding="utf-8")
        assert "model_channels" not in source
        assert "optimizer" not in source
        assert "latest" not in source.lower()
    source = Path(
        "compiler/realsas_compiler_services/platform_worker/scene_input_contract_v31.py"
    ).read_text(encoding="utf-8")
    assert "1024->518" in source
    assert "interpolate(" not in source
