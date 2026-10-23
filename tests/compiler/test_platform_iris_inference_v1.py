from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from compiler.realsas_compiler_services.platform_worker import model_binding, model_inputs
from compiler.realsas_compiler_services.platform_worker.iris_inference_v1 import (
    encode_scene_v1,
    load_iris_vnext_inference_v1,
)
from models.iris.v2.dinov2_foundation_v2 import DINOv2FoundationAuthorityV2
from models.iris.vnext.checkpoint_v1 import write_checkpoint_v1
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v3 import GraphicsRelationHeadV3Config
from models.iris.vnext.inference_input_v1 import IRISVNextSceneInputsV1
from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3


FAMILY = "IRIS"
ARCHITECTURE = IRISVNextEvidenceModelV3.architecture_id
ENGINE_RELEASE_ID = "22222222-2222-4222-8222-222222222222"
ARTIFACT_ID = "33333333-3333-4333-8333-333333333333"


def _model(seed: int) -> IRISVNextEvidenceModelV3:
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


def _manifest(root: Path, model: IRISVNextEvidenceModelV3, model_release_id: str) -> dict:
    run_id = "run-iris-inference-v1"
    path = root / "runs" / run_id / "models" / FAMILY / model_release_id / "iris.checkpoint"
    write_checkpoint_v1(path, model)
    payload = path.read_bytes()
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    identity_shape = {
        "family": FAMILY,
        "model_release_id": model_release_id,
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
        "model_release_id": model_release_id,
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


def _inputs() -> IRISVNextSceneInputsV1:
    generator = torch.Generator().manual_seed(991)
    return IRISVNextSceneInputsV1(
        image_tokens=torch.randn(1, 8, 37 * 37, 384, generator=generator),
        token_xy=torch.zeros(1, 8, 37 * 37, 2, dtype=torch.float32),
        camera_features=torch.zeros(1, 8, 13, dtype=torch.float32),
        foundation_authority_sha256=DINOv2FoundationAuthorityV2().authority_hash,
    )


def test_attempt_bound_checkpoint_executes_real_v3_scene_forward(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    manifest = _manifest(
        root,
        _model(17),
        "11111111-1111-4111-8111-111111111111",
    )
    binding = load_iris_vnext_inference_v1(manifest)
    inputs = _inputs()

    first = encode_scene_v1(binding, inputs)
    second = encode_scene_v1(binding, inputs)

    assert tuple(first.shape) == (1, 3, 4, 4, 4)
    assert torch.equal(first, second)
    assert binding.model.training is False
    assert not any(parameter.requires_grad for parameter in binding.model.parameters())


def test_different_exact_checkpoint_changes_behavior(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    inputs = _inputs()
    a = load_iris_vnext_inference_v1(
        _manifest(root, _model(17), "11111111-1111-4111-8111-111111111111")
    )
    b = load_iris_vnext_inference_v1(
        _manifest(root, _model(23), "44444444-4444-4444-8444-444444444444")
    )

    out_a = encode_scene_v1(a, inputs)
    out_b = encode_scene_v1(b, inputs)

    assert a.model_ref.checkpoint_content_sha256 != b.model_ref.checkpoint_content_sha256
    assert not torch.allclose(out_a, out_b)


def test_inference_bridge_has_no_training_or_live_channel_dependency():
    source = Path(
        "compiler/realsas_compiler_services/platform_worker/iris_inference_v1.py"
    ).read_text(encoding="utf-8")
    assert "cohort_v" not in source
    assert "optimizer" not in source
    assert "model_channels" not in source
    assert "latest" not in source.lower()
    assert "require_materialized_model_release" in source
