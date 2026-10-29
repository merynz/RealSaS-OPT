from __future__ import annotations

import torch

from models.iris.v31.evidence_adapter_model_v2 import IRISEvidenceAdapterConfigV2
from models.iris.v31.presentation_objective_v1 import (
    MECHANICAL_CHANNELS_V31_PRESENTATION,
    PAIR_CHANNELS_V31_PRESENTATION,
)
from models.iris.v31.tessa_bridge_v1 import (
    Truth139DenseIRISTESSABridgeV1,
    Truth139TESSABridgeConfigV1,
    dense_cell_center_xyz_v1,
)
from models.iris.v31.truth139_model_v1 import IRISV31Truth139V1
from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2


def _iris():
    cfg = IRISEvidenceAdapterConfigV2(
        carrier_resolution=4,
        carrier_channels=8,
        mechanical_channels=MECHANICAL_CHANNELS_V31_PRESENTATION,
        pair_channels=PAIR_CHANNELS_V31_PRESENTATION,
        query_width=32,
        query_layers=3,
        fourier_bands=4,
        private_bottleneck=8,
        evidence_adapter_bottleneck=8,
        lift_chunk=16,
        delta_bound=0.03,
    )
    return IRISV31Truth139V1(cfg)


def _tessa_config():
    return TESSAStructuralizerConfigV2(
        d_model=32,
        n_heads=4,
        decoder_layers=1,
        mlp_ratio=2,
        dropout=0.0,
        coordinate_bins=128,
        max_faces=128,
        max_vertices=128,
        scene_memory_tokens=8,
        surface_memory_tokens=8,
        memory_source_width=16,
        memory_refinement_layers=1,
        local_attention_window=128,
        query_chunk_size=16,
        gradient_checkpointing=False,
    )


def test_dense_scene_coordinates_preserve_cell_centres_and_tessa_domain():
    xyz = dense_cell_center_xyz_v1(4, device=torch.device("cpu"), dtype=torch.float32)
    axis = torch.unique(xyz[..., 0]).sort().values
    assert torch.allclose(axis, torch.tensor([-0.375, -0.125, 0.125, 0.375]))


def test_bridge_uses_dense_scene_and_inference_available_surface_evidence():
    iris = _iris()
    bridge = Truth139DenseIRISTESSABridgeV1(
        bridge_config=Truth139TESSABridgeConfigV1(
            scene_channels=8,
            iris_query_width=32,
            pooled_scene_resolution=3,
        ),
        tessa_config=_tessa_config(),
    )
    carrier = torch.randn(1, 8, 4, 4, 4, requires_grad=True)
    witness = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.25, -0.5, 0.75], [-0.75, 0.2, -0.1]]],
        dtype=torch.float32,
    )
    scene = bridge.scene_source(carrier)
    surface = bridge.surface_source(iris_model=iris, carrier=carrier, witness_xyz_iris=witness)
    assert scene.shape == (1, 27, 11)
    assert surface.shape == (1, 3, 42)
    assert float(surface[..., :3].abs().max()) <= 0.500001

    input_ids = torch.tensor([[_tessa_config().BOS, 1, 2, 3]], dtype=torch.long)
    output = bridge(
        iris_model=iris,
        carrier=carrier,
        witness_xyz_iris=witness,
        input_ids=input_ids,
        witness_valid_mask=torch.ones((1, 3), dtype=torch.bool),
    )
    assert output.logits.shape == (1, 4, _tessa_config().vocab_size)
    assert bridge.output_authority == "LEARNED_MESH_CANDIDATE_PROPOSAL_ONLY"
    assert bridge.teacher_topology_in_inference is False
    assert bridge.presentation_stream_fabricated is False


def test_topology_backward_cannot_rewrite_iris_carrier_or_model():
    iris = _iris()
    bridge = Truth139DenseIRISTESSABridgeV1(
        bridge_config=Truth139TESSABridgeConfigV1(8, 32, 3),
        tessa_config=_tessa_config(),
    )
    carrier = torch.randn(1, 8, 4, 4, 4, requires_grad=True)
    witness = torch.randn(1, 4, 3).clamp(-1.0, 1.0)
    ids = torch.tensor([[_tessa_config().BOS, 1, 2, 3]], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3, _tessa_config().EOS]], dtype=torch.long)
    output = bridge(
        iris_model=iris,
        carrier=carrier,
        witness_xyz_iris=witness,
        input_ids=ids,
        labels=labels,
        witness_valid_mask=torch.ones((1, 4), dtype=torch.bool),
    )
    assert output.loss is not None
    output.loss.backward()
    assert carrier.grad is None
    assert all(parameter.grad is None for parameter in iris.parameters())
    assert any(parameter.grad is not None for parameter in bridge.structuralizer.parameters())
