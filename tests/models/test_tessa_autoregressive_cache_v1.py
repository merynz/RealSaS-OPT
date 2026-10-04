from __future__ import annotations

import torch

from models.tessa.v1 import TESSAConfigV1, TESSAV1
from tools.training.run_tessa_autoregressive_reconstruction_v1 import (
    _decoder_step_cached,
    _precompute_cross_kv,
)


def test_cached_incremental_logits_match_causal_forward_inside_window():
    torch.manual_seed(7)
    cfg = TESSAConfigV1(
        d_model=32,
        n_heads=4,
        surface_layers=1,
        decoder_layers=2,
        mlp_ratio=2,
        surface_latent_count=8,
        local_attention_window=16,
        query_chunk_size=4,
        max_faces=32768,
        max_vertices=32768,
    )
    model = TESSAV1(cfg).eval()
    surface = torch.randn(1, 11, cfg.input_dim)
    memory = model.encode_surface(surface)
    cross_kv = _precompute_cross_kv(model, memory)
    self_kv = [None] * len(model.decoder)
    ids = torch.tensor([[cfg.BOS, 1, 2, 3, cfg.FACE_BREAK, 5, 6, 7]], dtype=torch.long)

    for pos in range(ids.shape[1]):
        cached_logits, self_kv = _decoder_step_cached(
            model=model,
            token_id=int(ids[0, pos]),
            absolute_position=pos,
            self_kv=self_kv,
            cross_kv=cross_kv,
        )
        full = model.decode_tokens(memory=memory, input_ids=ids[:, : pos + 1])
        torch.testing.assert_close(cached_logits, full.logits[:, -1, :], rtol=1e-5, atol=1e-5)
