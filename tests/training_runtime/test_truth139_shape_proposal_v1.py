from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_services.platform_worker.shape_capability_v2 import decode_shape_proposal_v2
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.training_data_v1 import build_teacher_asset_sequence_v1
from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2
from models.tessa.v31.topology_generation_v1 import decode_generated_topology_v1
from models.tessa.v31.truth139_shape_proposal_v1 import (
    SCOPE,
    package_truth139_shape_proposal_v1,
)


def _config():
    return TESSAStructuralizerConfigV2(
        d_model=32,
        n_heads=4,
        decoder_layers=1,
        mlp_ratio=2,
        dropout=0.0,
        coordinate_bins=128,
        max_faces=128,
        max_vertices=128,
        scene_memory_tokens=2,
        surface_memory_tokens=2,
        memory_source_width=16,
        memory_refinement_layers=0,
        local_attention_window=32,
        query_chunk_size=8,
        gradient_checkpointing=False,
    )


def test_static_consequence_shape_is_product_decodable_and_mechanics_abstains():
    cfg = _config()
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
    generated = decode_generated_topology_v1(sequence.token_ids, config=cfg)
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-0.5, 0.5, size=(1, 9, 3)).astype(np.float32)
    normal = rng.normal(size=(1, 9, 3)).astype(np.float32)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    h = rng.normal(size=(1, 9, 16)).astype(np.float32)
    geometry = rng.normal(size=(1, 9, 4)).astype(np.float32)
    cache = {
        "schema": "RealSaS.Truth139TESSAConditioningCache.v1",
        "teacher_topology_used": False,
        "surface_source": np.concatenate((xyz, normal, h, geometry), axis=-1),
    }
    result = package_truth139_shape_proposal_v1(
        generated=generated,
        conditioning_cache=cache,
        iris_checkpoint_sha256="1" * 64,
        tessa_checkpoint_sha256="2" * 64,
    )
    shape = decode_shape_proposal_v2(result.payload)
    assert shape.metadata["truth139_research_scope"] == SCOPE
    assert shape.metadata["mechanics_abstained"] is True
    assert shape.metadata["canonical_mesh_authority"] is False
    assert shape.metadata["source_truth_copied_into_shape"] is False
    assert shape.contact_persistence_valid.sum() == 0
    assert np.array_equal(shape.decoded_faces, generated.faces)
    assert shape.surface_h.shape[-1] == 16
