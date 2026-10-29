from __future__ import annotations

import numpy as np
import torch

from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.training_data_v1 import build_teacher_asset_sequence_v1
from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2
from models.tessa.v31.attention_v31 import SEMANTICS_ID
from models.tessa.v31.dual_stream_incremental_v31 import DualStreamIncrementalDecoderV31
from models.tessa.v31.dual_stream_v31 import TESSADualStreamStructuralizerV31
from models.tessa.v31.topology_generation_v1 import decode_generated_topology_v1


def _cfg():
    return TESSAStructuralizerConfigV2(
        d_model=32,
        n_heads=4,
        decoder_layers=2,
        mlp_ratio=2,
        dropout=0.0,
        coordinate_bins=128,
        max_faces=128,
        max_vertices=128,
        scene_memory_tokens=2,
        surface_memory_tokens=2,
        memory_source_width=16,
        memory_refinement_layers=0,
        local_attention_window=8,
        query_chunk_size=3,
        gradient_checkpointing=False,
    )


def _model():
    return TESSADualStreamStructuralizerV31(
        scene_input_dim=5,
        surface_input_dim=7,
        config=_cfg(),
        backend="SDPA",
    )


def test_dual_stream_incremental_matches_full_prefix_beyond_window():
    torch.manual_seed(17)
    model = _model().eval()
    scene = torch.randn(1, 3, 5)
    surface = torch.randn(1, 4, 7)
    with torch.no_grad():
        memory = model.encode_memory(scene_source=scene, surface_source=surface)
    state = DualStreamIncrementalDecoderV31(model, memory, sequence_position_offset=5)
    ids = torch.randint(0, _cfg().vocab_size, (1, 21))
    with torch.no_grad():
        for index in range(ids.shape[1]):
            actual = state.step(int(ids[0, index]))
            expected = model.decode_tokens(
                memory=memory,
                input_ids=ids[:, : index + 1],
                sequence_position_offset=5,
            ).logits[:, -1]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-7)
    assert model.attention_semantics == SEMANTICS_ID
    assert model.presentation_stream_present is False
    assert all(cache.stable_k.shape[2] <= _cfg().local_attention_window for cache in state.layers)


def test_canonical_teacher_sequence_roundtrips_through_free_run_mesh_decoder():
    cfg = _cfg()
    codec = TESSAConfigV1(
        coordinate_bins=cfg.coordinate_bins,
        max_faces=cfg.max_faces,
        max_vertices=cfg.max_vertices,
        local_attention_window=cfg.local_attention_window,
        query_chunk_size=cfg.query_chunk_size,
        surface_latent_count=cfg.surface_memory_tokens,
    )
    vertices = np.asarray(
        [[0.0, 0.0, 0.4], [-0.4, -0.4, -0.4], [0.4, -0.4, -0.4], [0.0, 0.4, -0.4]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
    sequence = build_teacher_asset_sequence_v1(vertices, faces, cfg=codec)
    decoded = decode_generated_topology_v1(sequence.token_ids, config=cfg)
    assert decoded.status == "PASS_CANONICAL_TESSA_SEQUENCE_CODEC"
    assert len(decoded.faces) == len(faces)
    assert decoded.component_count == 1
    assert np.all(decoded.vertices_normalized >= -0.5)
    assert np.all(decoded.vertices_normalized <= 0.5)
    assert len({tuple(sorted(map(int, row))) for row in decoded.faces.tolist()}) == len(faces)
