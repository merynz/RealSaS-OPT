from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from compiler.realsas_compiler_services.platform_worker import model_binding, model_inputs
from compiler.realsas_compiler_services.platform_worker.tessa_inference_v1 import (
    TESSAInferencePolicyV1,
    TESSA_OUTPUT_AUTHORITY,
    build_inputs_from_iris_v1,
    encode_tessa_memory_v1,
    generate_topology_proposal_v1,
    load_tessa_v3_inference_v1,
)
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v3 import GraphicsRelationHeadV3Config
from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3
from models.iris.vnext.presentation_evidence_v1 import PRESENTATION_FEATURE_NAMES_V1
from models.iris.vnext.structure_v4 import IRISVNextStructuralizationBridgeV4
from models.iris.vnext.surface_v1 import IRISSurfaceEvidenceV1
from models.tessa.v3.checkpoint_v1 import write_checkpoint_v1
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v3.model_v3 import TESSAStructuralizerV3


FAMILY = "TESSA"
ARCHITECTURE = TESSAStructuralizerV3.architecture_id
ENGINE_RELEASE_ID = "22222222-2222-4222-8222-222222222222"
ARTIFACT_ID = "33333333-3333-4333-8333-333333333333"


def _iris() -> IRISVNextEvidenceModelV3:
    torch.manual_seed(901)
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
    model = IRISVNextEvidenceModelV3(config, relation)
    model.eval()
    model.requires_grad_(False)
    return model


def _config() -> TESSAStructuralizerConfigV3:
    return TESSAStructuralizerConfigV3(
        d_model=32,
        n_heads=4,
        decoder_layers=1,
        mlp_ratio=2,
        dropout=0.0,
        coordinate_bins=128,
        max_faces=16,
        max_vertices=32,
        scene_memory_tokens=2,
        surface_memory_tokens=2,
        presentation_memory_tokens=1,
        conditioning_token_budget=5,
        memory_source_width=16,
        memory_refinement_layers=0,
        local_attention_window=8,
        query_chunk_size=4,
        gradient_checkpointing=False,
    )


def _tessa(seed: int, iris: IRISVNextEvidenceModelV3) -> TESSAStructuralizerV3:
    torch.manual_seed(seed)
    model = TESSAStructuralizerV3(
        scene_input_dim=iris.config.plane_channels + 5,
        surface_input_dim=3 + 3 + iris.config.local_feature_dim + 6 + 2,
        presentation_input_dim=len(PRESENTATION_FEATURE_NAMES_V1),
        config=_config(),
    )
    model.eval()
    model.requires_grad_(False)
    return model


def _evidence(iris: IRISVNextEvidenceModelV3) -> IRISSurfaceEvidenceV1:
    generator = torch.Generator().manual_seed(777)
    count = 5
    xyz_iris = torch.randn(1, count, 3, generator=generator).clamp(-0.4, 0.4)
    normal = F.normalize(torch.randn(1, count, 3, generator=generator), dim=-1)
    evidence = IRISSurfaceEvidenceV1(
        scene_planes=torch.randn(
            1, 3, iris.config.plane_channels, 2, 2, generator=generator
        ),
        surface_xyz_iris=xyz_iris,
        surface_xyz_tessa=(xyz_iris * 0.5).clamp(-0.5, 0.5),
        surface_normal=normal,
        surface_h=torch.randn(1, count, iris.config.local_feature_dim, generator=generator),
        mechanical={
            "stretch_major": torch.rand(1, count, generator=generator) + 1.0,
            "stretch_minor": torch.rand(1, count, generator=generator),
            "tangent_direction_2theta": F.normalize(
                torch.randn(1, count, 2, generator=generator), dim=-1
            ),
            "bend_demand": torch.rand(1, count, generator=generator),
            "rigidity_probability": torch.rand(1, count, generator=generator),
            "contact_persistence_probability": torch.rand(1, count, generator=generator),
        },
        contact_persistence_valid=torch.zeros(1, count, dtype=torch.bool),
        extracted_vertex_counts=(12,),
        extracted_face_counts=(20,),
    )
    evidence.validate(iris)
    return evidence


def _manifest(root: Path, model: TESSAStructuralizerV3, model_release_id: str) -> dict:
    run_id = "run-tessa-inference-v1"
    path = root / "runs" / run_id / "models" / FAMILY / model_release_id / "tessa.checkpoint"
    write_checkpoint_v1(path, model)
    payload = path.read_bytes()
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    identity_shape = {
        "family": FAMILY,
        "model_release_id": model_release_id,
        "architecture_id": ARCHITECTURE,
        "model_engine_release_id": ENGINE_RELEASE_ID,
        "source_code_sha": "a" * 40,
        "checkpoint": {"id": ARTIFACT_ID, "content_sha256": checkpoint_sha},
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
            "models": [{"family": FAMILY, "behavior_identity_sha256": behavior_sha}],
        }
    )
    return {
        "run_id": run_id,
        "_platform_model_snapshot_sha256": snapshot_sha,
        "_platform_model_releases": [row],
    }


def test_attempt_bound_tessa_checkpoint_encodes_current_iris_evidence(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    iris = _iris()
    manifest = _manifest(
        root, _tessa(41, iris), "11111111-1111-4111-8111-111111111111"
    )
    binding = load_tessa_v3_inference_v1(manifest)
    inputs = build_inputs_from_iris_v1(iris, _evidence(iris))

    first = encode_tessa_memory_v1(binding, inputs)
    second = encode_tessa_memory_v1(binding, inputs)

    assert tuple(first.shape) == (1, 5, 32)
    assert inputs.presentation_source.shape[-1] == len(PRESENTATION_FEATURE_NAMES_V1)
    assert torch.equal(first, second)
    assert binding.model.training is False
    assert not any(parameter.requires_grad for parameter in binding.model.parameters())


def test_product_input_lowering_matches_current_v4_bridge_contract():
    iris = _iris()
    evidence = _evidence(iris)
    inputs = build_inputs_from_iris_v1(iris, evidence)
    bridge = IRISVNextStructuralizationBridgeV4(
        iris_config=iris.config,
        tessa_config=_config(),
    )
    expected_scene = bridge.scene_source(evidence.scene_planes)
    expected_surface = bridge.surface_source(
        witness_xyz=evidence.surface_xyz_tessa,
        witness_normal=evidence.surface_normal,
        witness_h=evidence.surface_h,
        witness_mechanical=evidence.mechanical,
        contact_persistence=evidence.mechanical["contact_persistence_probability"],
        contact_persistence_valid=evidence.contact_persistence_valid,
    )
    expected_presentation = bridge.presentation_source(
        reference=expected_surface,
        presentation=None,
    )

    assert torch.equal(inputs.scene_source, expected_scene)
    assert torch.equal(inputs.surface_source, expected_surface)
    assert torch.equal(inputs.presentation_source, expected_presentation.values)
    assert torch.equal(inputs.presentation_valid_mask, expected_presentation.valid_mask)


def test_different_exact_tessa_checkpoint_changes_conditioning_memory(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    iris = _iris()
    inputs = build_inputs_from_iris_v1(iris, _evidence(iris))
    a = load_tessa_v3_inference_v1(
        _manifest(root, _tessa(41, iris), "11111111-1111-4111-8111-111111111111")
    )
    b = load_tessa_v3_inference_v1(
        _manifest(root, _tessa(43, iris), "44444444-4444-4444-8444-444444444444")
    )

    out_a = encode_tessa_memory_v1(a, inputs)
    out_b = encode_tessa_memory_v1(b, inputs)

    assert a.model_ref.checkpoint_content_sha256 != b.model_ref.checkpoint_content_sha256
    assert not torch.allclose(out_a, out_b)


def test_incremental_product_generation_is_deterministic_and_never_canonical(monkeypatch, tmp_path):
    root = tmp_path / "authority"
    monkeypatch.setattr(model_binding.mainline, "authority_root", lambda: root)
    iris = _iris()
    binding = load_tessa_v3_inference_v1(
        _manifest(root, _tessa(47, iris), "55555555-5555-4555-8555-555555555555")
    )
    inputs = build_inputs_from_iris_v1(iris, _evidence(iris))
    policy = TESSAInferencePolicyV1(maximum_new_tokens=12)

    first = generate_topology_proposal_v1(binding, inputs, policy=policy)
    second = generate_topology_proposal_v1(binding, inputs, policy=policy)

    assert first.token_ids == second.token_ids
    assert first.output_authority == TESSA_OUTPUT_AUTHORITY
    assert first.canonical_mesh_authority is False
    assert first.model_release_id == binding.model_ref.model_release_id
    assert len(first.token_ids) <= 13


def test_tessa_inference_path_has_no_training_or_live_channel_dependency():
    source = Path(
        "compiler/realsas_compiler_services/platform_worker/tessa_inference_v1.py"
    ).read_text(encoding="utf-8")
    assert "cohort_v" not in source
    assert "optimizer" not in source
    assert "model_channels" not in source
    assert "latest" not in source.lower()
    assert "incremental_decode_step_windowed_v2" in source
    assert "generate_tessa_candidate_v2" not in source
    assert "require_materialized_model_release" in source
