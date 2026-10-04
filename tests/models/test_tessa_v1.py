import numpy as np
import torch

from compiler.realsas_compiler_core.preproduct_authority_v1 import NormalizationDomainIR
from compiler.realsas_compiler_core.types import (
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)
from models.tessa.v1 import (
    TESSAConfigV1,
    TESSAV1,
    build_tessa_conditioning_v1,
    encode_adjacent_teacher_mesh_v1,
    mechanical_consequence_loss_v1,
    tessa_attention_work_upper_bound_v1,
    triangle_deformation_metrics_v1,
)


def _surface():
    nodes = tuple(
        SurfaceNode(
            surface_id=f"S{i}",
            P=p,
            support_views=(0, 1),
            provenance_refs=(f"P{i}",),
            source_observation_ids=(f"O{i}",),
            derived_normal=(0.0, 0.0, 1.0),
        )
        for i, p in enumerate(
            ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (-0.5, 0.5, 0.0), (0.5, 0.5, 0.0))
        )
    )
    rel = (
        SurfaceRelation("R0", "S0", "S1", "LOCAL"),
        SurfaceRelation("R1", "S1", "S3", "LOCAL"),
    )
    return RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=rel,
        geometry_lineage_hash="gsa-test-lineage",
    )


def _normalization():
    return NormalizationDomainIR(
        observation_set_binding_hash="obs-test",
        camera_set_binding_hash="cam-test",
        center_xyz=(0.0, 0.0, 0.0),
        half_extent=1.0,
        normalized_bounds_min=(-1.0, -1.0, -1.0),
        normalized_bounds_max=(1.0, 1.0, 1.0),
        coordinate_frame="REALSAS_OBJECT_FRAME",
        normalization_hash="norm-test-hash",
    )


def _small_cfg():
    return TESSAConfigV1(
        d_model=64,
        n_heads=8,
        surface_layers=1,
        decoder_layers=1,
        mlp_ratio=2,
        surface_latent_count=64,
        local_attention_window=128,
        query_chunk_size=32,
        max_faces=8192,
        max_vertices=8192,
    )


def test_tessa_default_scale_admits_knight_source_mesh():
    cfg = TESSAConfigV1()
    assert cfg.scaling_policy().admits(vertex_count=3665, face_count=6952)
    assert cfg.max_faces >= 4 * 6952


def test_tessa_attention_contract_is_linear_in_long_sequence_window():
    cfg = _small_cfg()
    length = 10_000
    work = tessa_attention_work_upper_bound_v1(length, cfg)
    assert work == length * cfg.local_attention_window
    assert work < length * length


def test_tessa_gsa_conditioning_uses_stage08_frame_not_gsa_bbox():
    c = build_tessa_conditioning_v1(_surface(), normalization=_normalization())
    assert c.features.shape == (4, 17)
    assert c.surface_ids == ("S0", "S1", "S2", "S3")
    assert c.source_geometry_lineage_hash == "gsa-test-lineage"
    assert c.normalization_hash == "norm-test-hash"
    assert c.coordinate_frame == "REALSAS_OBJECT_FRAME"
    assert c.scale == 2.0
    assert torch.isfinite(c.features).all()
    # Surface bbox is +/-0.5 world units, but Stage08 half_extent=1.0. The
    # TESSA token frame must therefore be +/-0.25, proving it did not refit to
    # the finite GSA cloud bbox.
    assert torch.allclose(c.features[:, :3].abs().amax(), torch.tensor(0.25), atol=1e-6)


def test_tessa_adjacent_teacher_encoding_compresses_shared_edge():
    cfg = _small_cfg()
    vertices = np.asarray(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    seq = encode_adjacent_teacher_mesh_v1(
        vertices,
        faces,
        component_id="C0",
        chart_id="CH0",
        cfg=cfg,
    )
    assert seq.face_count == 2
    assert seq.restart_count == 1
    coordinate_tokens = [t for t in seq.token_ids if 0 <= t < cfg.coordinate_bins]
    assert len(coordinate_tokens) == 12


def test_tessa_identity_deformation_has_unit_metrics_and_zero_penalty():
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    metrics = triangle_deformation_metrics_v1(rest, rest.clone(), faces)
    assert torch.allclose(metrics.area_ratio, torch.ones_like(metrics.area_ratio), atol=1e-5)
    assert torch.allclose(metrics.condition_number, torch.ones_like(metrics.condition_number), atol=1e-5)
    assert torch.allclose(metrics.max_edge_ratio, torch.ones_like(metrics.max_edge_ratio), atol=1e-5)
    losses = mechanical_consequence_loss_v1(
        metrics,
        reference_jacobian=metrics.jacobian.detach(),
        reference_normal=metrics.normal.detach(),
    )
    assert float(losses["total"]) < 1e-6


def test_tessa_forward_uses_surface_memory_and_sequence_offset():
    cfg = _small_cfg()
    model = TESSAV1(cfg)
    surface = torch.randn(2, 32, 17)
    input_ids = torch.randint(0, cfg.coordinate_bins, (2, 24))
    labels = input_ids.clone()
    out = model(
        surface_features=surface,
        input_ids=input_ids,
        labels=labels,
        sequence_position_offset=257,
    )
    assert out.logits.shape == (2, 24, cfg.vocab_size)
    assert out.surface_latents.shape == (2, cfg.surface_latent_count, cfg.d_model)
    assert out.loss is not None and torch.isfinite(out.loss)
    expected = (torch.arange(24) + 257) % cfg.local_attention_window
    assert torch.equal(model._position_ids(24, input_ids.device, offset=257), expected)
