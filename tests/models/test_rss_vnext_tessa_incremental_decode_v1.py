from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.tessa.v31.model_v31 import TESSAStructuralizerV31
from models.rss_vnext.tessa_incremental_decode_v1 import decode_next_token_v1
import models.rss_vnext.unified_character_model as unified_module


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(7)
    torch.set_num_threads(1)


def _decoder():
    cfg = replace(
        RSSvNextConfig.tiny_for_test().tessa,
        local_attention_window=8, query_chunk_size=4,
    )
    model = TESSAStructuralizerV31(
        scene_input_dim=8, surface_input_dim=8,
        presentation_input_dim=8, config=cfg,
    ).eval()
    return model, torch.randn((1, cfg.conditioning_token_budget, cfg.d_model))


def test_incremental_exact_logits_match_full_prefix_including_window_rollover():
    model, memory = _decoder()
    ids = torch.randint(0, model.config.coordinate_bins, (1, 19))
    ids[0, 0] = model.config.BOS
    cache = None
    for length in range(1, ids.shape[1]+1):
        next_logits, cache = decode_next_token_v1(model, 
            memory=memory, token=ids[:, length-1:length], cache=cache,
        )
        reference = model.decode_tokens(
            memory=memory, input_ids=ids[:, :length],
        ).logits[:, -1]
        assert cache.position == length
        assert cache.keys[0].shape[2] == min(length, model.config.local_attention_window)
        assert torch.allclose(next_logits, reference, atol=2e-5, rtol=1e-5), length


def test_incremental_refuses_foreign_memory_and_training_mode():
    model, memory = _decoder()
    ids = torch.tensor([[model.config.BOS]])
    _, cache = decode_next_token_v1(model, memory=memory, token=ids)
    with pytest.raises(ValueError, match="FOREIGN_CACHE"):
        decode_next_token_v1(model, 
            memory=memory.clone(), token=ids, cache=cache,
        )
    model.train()
    with pytest.raises(RuntimeError, match="INCREMENTAL_REQUIRES_EVAL"):
        decode_next_token_v1(model, memory=memory, token=ids)


def test_teacher_free_product_calls_incremental_decoder_without_prefix_rescans(monkeypatch):
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).eval()
    real_decode = unified_module.decode_next_token_v1

    def observed(tessa, *, memory: torch.Tensor, token: torch.Tensor, cache=None):
        assert tessa is net.tessa
        if cache is not None:
            assert cache.memory is memory
        return real_decode(tessa, memory=memory, token=token, cache=cache)

    # No complete prefix decoder call is permitted in product generation.
    def forbidden(**_kwargs):
        raise RuntimeError("FULL_PREFIX_DECODER_INFER_PATH_FORBIDDEN")

    monkeypatch.setattr(net.tessa, "decode_tokens", forbidden)
    monkeypatch.setattr(net.tessa, "decode_next_token_v1", observed)
    rgba = torch.rand((1, 2, 4, 16, 16))
    cameras = torch.zeros((1, 2, 13))
    for idx in (3, 7, 11, 12):
        cameras[:, :, idx] = 1.
    result = net.propose_teacher_free(
        rgba, cameras, generate_controls=False, max_topology_tokens=16,
    )
    assert result["mesh_status"] in ("ABSTAIN", "UNQUALIFIED_MESH_CANDIDATE")
    assert result["observed_view_count"] == 2
