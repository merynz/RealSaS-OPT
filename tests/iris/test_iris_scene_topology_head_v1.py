from __future__ import annotations

import inspect

import torch

from models.iris.topology import (
    IRISSceneTopologyHeadV1,
    IRISTopologySceneMemoryConfigV1,
    IRISTopologySceneMemoryV1,
)
from models.tessa.v1.model_v1 import TESSAConfigV1


def _tessa_cfg() -> TESSAConfigV1:
    return TESSAConfigV1(
        d_model=64,
        n_heads=4,
        surface_layers=1,
        decoder_layers=2,
        mlp_ratio=2,
        coordinate_bins=128,
        surface_latent_count=64,
        local_attention_window=128,
        query_chunk_size=32,
    )


def _memory_cfg() -> IRISTopologySceneMemoryConfigV1:
    return IRISTopologySceneMemoryConfigV1(
        scene_channels=8,
        source_width=32,
        topology_width=64,
        memory_tokens=64,
        attention_heads=4,
        refinement_layers=1,
        mlp_ratio=2,
        dropout=0.0,
    )


def test_scene_memory_is_bounded_across_spatial_resolution():
    model = IRISTopologySceneMemoryV1(_memory_cfg()).eval()
    with torch.no_grad():
        a = model(torch.randn(2, 3, 8, 8, 8))
        b = model(torch.randn(2, 3, 8, 16, 16))
    assert a.shape == (2, 64, 64)
    assert b.shape == (2, 64, 64)


def test_scene_memory_is_deterministic_and_depends_on_rich_scene_values():
    torch.manual_seed(17)
    model = IRISTopologySceneMemoryV1(_memory_cfg()).eval()
    planes = torch.randn(1, 3, 8, 8, 8)
    changed = planes.clone()
    changed[:, 1, 3, 4, 5] += 2.0
    with torch.no_grad():
        first = model(planes)
        repeat = model(planes)
        other = model(changed)
    torch.testing.assert_close(first, repeat)
    assert not torch.allclose(first, other)


def test_scene_memory_backpropagates_directly_to_iris_scene_evidence():
    torch.manual_seed(23)
    model = IRISTopologySceneMemoryV1(_memory_cfg())
    planes = torch.randn(1, 3, 8, 8, 8, requires_grad=True)
    loss = model(planes).square().mean()
    loss.backward()
    assert planes.grad is not None
    assert torch.isfinite(planes.grad).all()
    assert float(planes.grad.abs().sum()) > 0.0


def test_topology_head_has_no_legacy_surface_features_input_or_surface_encoder():
    signature = inspect.signature(IRISSceneTopologyHeadV1.forward)
    assert "surface_features" not in signature.parameters
    assert "scene_planes" in signature.parameters
    model = IRISSceneTopologyHeadV1(
        tessa_config=_tessa_cfg(),
        memory_config=_memory_cfg(),
    )
    state_keys = tuple(model.state_dict())
    assert not any("surface_encoder" in key for key in state_keys)
    assert model.legacy_gsa_17d_used is False
    assert model.scene_memory.legacy_gsa_17d_used is False


def test_scene_native_topology_head_reuses_tessa_token_contract():
    torch.manual_seed(29)
    cfg = _tessa_cfg()
    model = IRISSceneTopologyHeadV1(
        tessa_config=cfg,
        memory_config=_memory_cfg(),
    ).eval()
    scene = torch.randn(2, 3, 8, 8, 8)
    ids = torch.randint(0, cfg.coordinate_bins, (2, 24))
    labels = ids.clone()
    with torch.no_grad():
        out = model(scene_planes=scene, input_ids=ids, labels=labels)
    assert out.logits.shape == (2, 24, cfg.vocab_size)
    assert out.surface_latents.shape == (2, cfg.surface_latent_count, cfg.d_model)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_explicit_plane_identity_makes_plane_role_semantic():
    torch.manual_seed(31)
    model = IRISTopologySceneMemoryV1(_memory_cfg()).eval()
    scene = torch.randn(1, 3, 8, 8, 8)
    permuted = scene[:, torch.tensor([1, 0, 2])]
    with torch.no_grad():
        a = model(scene)
        b = model(permuted)
    assert not torch.allclose(a, b)
