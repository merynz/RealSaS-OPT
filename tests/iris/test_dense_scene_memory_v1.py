from __future__ import annotations

import pytest
import torch

from models.iris.topology.dense_scene_memory_v1 import (
    IRISDenseTopologySceneMemoryConfigV1,
    IRISDenseTopologySceneMemoryV1,
    _cell_center_xyz,
)
from models.iris.topology.head_dense_v1 import IRISDenseSceneTopologyHeadV1
from models.tessa.v1.model_v1 import TESSAConfigV1


def _memory_config():
    return IRISDenseTopologySceneMemoryConfigV1(
        scene_channels=8,
        pooled_resolution=4,
        source_width=16,
        topology_width=32,
        memory_tokens=64,
        attention_heads=4,
        refinement_layers=1,
        mlp_ratio=2,
    )


def _tessa_config():
    return TESSAConfigV1(
        d_model=32,
        n_heads=4,
        surface_layers=1,
        decoder_layers=1,
        mlp_ratio=2,
        coordinate_bins=128,
        surface_latent_count=64,
        local_attention_window=128,
        query_chunk_size=16,
    )


def test_dense_memory_is_bounded_and_gradients_reach_carrier():
    module = IRISDenseTopologySceneMemoryV1(_memory_config())
    carrier = torch.randn(2, 8, 6, 7, 8, requires_grad=True)
    memory = module(carrier)
    assert memory.shape == (2, 64, 32)
    memory.square().mean().backward()
    assert carrier.grad is not None
    assert float(carrier.grad.abs().sum()) > 0.0


def test_dense_position_features_use_cell_centres_not_endpoints():
    xyz = _cell_center_xyz(4, device=torch.device("cpu"), dtype=torch.float32)
    axis = torch.unique(xyz[..., 0]).sort().values
    assert torch.allclose(axis, torch.tensor([-0.75, -0.25, 0.25, 0.75]))


def test_dense_memory_rejects_wrong_channels_and_nonfinite_values():
    module = IRISDenseTopologySceneMemoryV1(_memory_config())
    with pytest.raises(ValueError, match="CHANNEL_COUNT"):
        module(torch.randn(1, 7, 4, 4, 4))
    bad = torch.randn(1, 8, 4, 4, 4)
    bad[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="CARRIER_INVALID"):
        module(bad)


def test_dense_topology_head_keeps_proposal_only_authority():
    head = IRISDenseSceneTopologyHeadV1(
        scene_channels=8,
        tessa_config=_tessa_config(),
        memory_config=_memory_config(),
    )
    carrier = torch.randn(1, 8, 4, 4, 4)
    input_ids = torch.tensor([[128, 1, 2, 3]], dtype=torch.long)
    result = head(carrier=carrier, input_ids=input_ids)
    assert result.logits.shape == (1, 4, _tessa_config().vocab_size)
    assert result.surface_latents.shape == (1, 64, 32)
    assert head.output_authority == "LEARNED_TOPOLOGY_PROPOSAL_ONLY"
    assert head.legacy_gsa_17d_used is False
