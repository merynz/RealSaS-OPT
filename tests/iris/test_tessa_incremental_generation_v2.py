from __future__ import annotations

import torch

from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2
from models.tessa.v2.global_attention_v2 import TESSAGlobalHistoryPolicyV2
from models.tessa.v2.incremental_generation_v2 import incremental_prefix_logits_v2
from models.tessa.v2.model_global_v2 import TESSAStructuralizerGlobalV2


def _model():
    torch.manual_seed(7)
    cfg = TESSAStructuralizerConfigV2(
        d_model=32,
        n_heads=4,
        decoder_layers=2,
        mlp_ratio=2,
        coordinate_bins=128,
        max_faces=128,
        max_vertices=128,
        scene_memory_tokens=4,
        surface_memory_tokens=4,
        memory_source_width=24,
        memory_refinement_layers=0,
        local_attention_window=8,
        query_chunk_size=2,
        gradient_checkpointing=False,
    )
    return TESSAStructuralizerGlobalV2(
        scene_input_dim=11,
        surface_input_dim=13,
        config=cfg,
        history_policy=TESSAGlobalHistoryPolicyV2(block_size=2, max_summary_tokens=64),
    ).eval()


def test_incremental_logits_match_autoregressive_prefix_recompute_across_history_boundary():
    model = _model()
    torch.manual_seed(11)
    scene = torch.randn(1, 9, 11)
    surface = torch.randn(1, 10, 13)
    memory = model.encode_memory(scene_source=scene, surface_source=surface)
    ids = torch.randint(0, model.config.vocab_size, (1, 15))
    with torch.no_grad():
        incremental = incremental_prefix_logits_v2(model, memory=memory, input_ids=ids)
        reference_rows = []
        for end in range(1, ids.shape[1] + 1):
            # Reference generation re-evaluates the whole prefix and consumes the
            # final-position logits. That is the exact behavior the cache replaces.
            reference_rows.append(
                model.decode_tokens(memory=memory, input_ids=ids[:, :end]).logits[:, -1]
            )
        reference = torch.stack(reference_rows, dim=1)
    assert torch.allclose(reference, incremental, atol=2e-5, rtol=2e-5), (
        reference - incremental
    ).abs().max().item()


def test_incremental_single_token_shape_contract():
    model = _model()
    memory = model.encode_memory(
        scene_source=torch.randn(1, 3, 11),
        surface_source=torch.randn(1, 4, 13),
    )
    ids = torch.tensor([[model.config.BOS]], dtype=torch.long)
    out = incremental_prefix_logits_v2(model, memory=memory, input_ids=ids)
    assert out.shape == (1, 1, model.config.vocab_size)
