from __future__ import annotations

import torch
import pytest

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v3 import GraphicsRelationHeadV3Config
from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3
from models.iris.vnext.structure_v4 import IRISVNextStructuralizationBridgeV4
from models.iris.vnext.surface_v1 import extract_surface_evidence_v1
from models.tessa.v3 import TESSAStructuralizerConfigV3


class FakeSurfaceExtractor:
    def __init__(self, forced_x: float | None = None):
        self.forced_x = forced_x
        self.training_flags: list[bool] = []

    def construct_iris_grid(self, resolution: int):
        assert resolution >= 2
        vertices = torch.tensor(
            [
                [-0.4, -0.4, -0.4],
                [0.4, -0.4, -0.4],
                [0.4, 0.4, -0.4],
                [-0.4, 0.4, -0.4],
                [-0.4, -0.4, 0.4],
                [0.4, -0.4, 0.4],
                [0.4, 0.4, 0.4],
                [-0.4, 0.4, 0.4],
            ],
            dtype=torch.float32,
        )
        cubes = torch.arange(8, dtype=torch.long).reshape(1, 8)
        return vertices, cubes

    def extract(
        self,
        *,
        vertices_iris,
        field,
        cubes,
        raw_parameters,
        resolution,
        training,
    ):
        assert tuple(raw_parameters.shape) == (1, 21)
        assert tuple(field.shape) == (8,)
        self.training_flags.append(bool(training))
        vertices = vertices_iris[:4].clone()
        if self.forced_x is not None:
            vertices[0, 0] = float(self.forced_x)
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=vertices.device)
        return vertices, faces, vertices.new_zeros(1)


def _config():
    return IRISVNextConfigV1(
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
        maximum_extraction_delta=0.03,
        extraction_training_resolution=4,
        surface_witness_count=8,
    )


def _model():
    torch.manual_seed(71)
    relation = GraphicsRelationHeadV3Config(
        feature_width=16,
        geometry_width=8,
        camera_width=8,
        pair_width=24,
        residual_blocks=1,
    )
    model = IRISVNextEvidenceModelV3(_config(), relation)
    model.eval()
    model.requires_grad_(False)
    return model


def _scene(model):
    torch.manual_seed(72)
    return torch.randn(1, 3, model.config.plane_channels, 4, 4)


def test_surface_is_evidence_not_mesh_and_never_runs_training_extraction():
    model = _model()
    extractor = FakeSurfaceExtractor()
    evidence = extract_surface_evidence_v1(
        model,
        _scene(model),
        extractor,
        resolution=4,
        witness_count=4,
    )

    assert evidence.schema == "RealSaS.IRISSurfaceEvidence.v1"
    assert evidence.mesh_authority is False
    assert evidence.extractor_faces_serialized is False
    assert not hasattr(evidence, "faces")
    assert tuple(evidence.surface_xyz_iris.shape) == (1, 4, 3)
    assert tuple(evidence.surface_h.shape) == (1, 4, 16)
    assert evidence.contact_persistence_valid.sum().item() == 0
    assert extractor.training_flags == [False]


def test_authorized_overshoot_is_verified_then_projected_to_tessa_domain():
    model = _model()
    evidence = extract_surface_evidence_v1(
        model,
        _scene(model),
        FakeSurfaceExtractor(forced_x=1.02),
        resolution=4,
        witness_count=4,
    )
    assert float(evidence.surface_xyz_iris.abs().max()) >= 1.0
    assert float(evidence.surface_xyz_tessa.abs().max()) <= 0.500001
    assert float(evidence.surface_xyz_tessa[0, 0, 0]) == pytest.approx(0.5)


def test_overshoot_beyond_model_authority_fails_closed():
    model = _model()
    with pytest.raises(RuntimeError, match="IRIS_SURFACE_EXCEEDS_AUTHORIZED_EXTRACTION_DELTA"):
        extract_surface_evidence_v1(
            model,
            _scene(model),
            FakeSurfaceExtractor(forced_x=1.04),
            resolution=4,
            witness_count=4,
        )


def test_surface_evidence_is_directly_consumable_by_current_tessa_bridge():
    model = _model()
    evidence = extract_surface_evidence_v1(
        model,
        _scene(model),
        FakeSurfaceExtractor(),
        resolution=4,
        witness_count=4,
    )
    tessa = TESSAStructuralizerConfigV3(
        d_model=16,
        n_heads=4,
        decoder_layers=1,
        coordinate_bins=128,
        scene_memory_tokens=2,
        surface_memory_tokens=2,
        presentation_memory_tokens=2,
        conditioning_token_budget=6,
        memory_source_width=8,
        memory_refinement_layers=1,
        local_attention_window=8,
        query_chunk_size=2,
        gradient_checkpointing=False,
    )
    bridge = IRISVNextStructuralizationBridgeV4(
        iris_config=model.config,
        tessa_config=tessa,
    )
    mechanics = {
        key: value
        for key, value in evidence.mechanical.items()
        if key != "contact_persistence_probability"
    }
    source = bridge.surface_source(
        witness_xyz=evidence.surface_xyz_tessa,
        witness_normal=evidence.surface_normal,
        witness_h=evidence.surface_h,
        witness_mechanical=mechanics,
        contact_persistence=evidence.mechanical["contact_persistence_probability"],
        contact_persistence_valid=evidence.contact_persistence_valid,
    )
    assert tuple(source.shape) == (1, 4, 3 + 3 + 16 + 6 + 2)
    # Unknown contact is represented as explicit abstention, never a fabricated negative.
    assert torch.equal(source[..., -1], torch.zeros_like(source[..., -1]))
    assert torch.equal(source[..., -2], torch.zeros_like(source[..., -2]))
