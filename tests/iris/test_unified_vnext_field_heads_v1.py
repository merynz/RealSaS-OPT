import torch

from models.iris.vnext.field_heads_v1 import (
    UnifiedFieldHeadConfigV1,
    UnifiedPointFieldHeadsV1,
)


def test_unified_field_heads_share_one_point_domain_and_keep_expected_shapes():
    cfg = UnifiedFieldHeadConfigV1(
        triplane_channels=80,
        hidden_dim=128,
        layers=3,
        owner_embedding_dim=64,
        material_embedding_dim=32,
        extraction_delta_limit=0.02,
    )
    model = UnifiedPointFieldHeadsV1(cfg)
    x = torch.randn(2, 17, cfg.sampled_feature_dim)
    refinement = torch.rand(2, 17)
    out = model(x, refinement_weight=refinement)

    assert out["sdf"].shape == (2, 17)
    assert out["geometry_log_uncertainty"].shape == (2, 17)
    assert out["rgba"].shape == (2, 17, 4)
    assert out["owner_embedding"].shape == (2, 17, 64)
    assert out["material_embedding"].shape == (2, 17, 32)
    assert out["owner_boundary_logit"].shape == (2, 17)
    assert out["material_boundary_logit"].shape == (2, 17)
    assert out["amodal_support_logit"].shape == (2, 17)
    assert out["deformation_mode_logits"].shape[-1] == 6
    assert out["attachment_relation_logits"].shape[-1] == 5
    assert out["extraction_delta"].shape == (2, 17, 3)
    assert out["extraction_cell_weights"].shape == (2, 17, 21)
    assert torch.all(out["rgba"] >= 0) and torch.all(out["rgba"] <= 1)
    assert float(out["extraction_delta"].abs().max()) <= 0.020001


def test_owner_and_material_descriptors_are_normalized_relational_embeddings():
    cfg = UnifiedFieldHeadConfigV1(
        triplane_channels=80,
        hidden_dim=128,
        layers=3,
        owner_embedding_dim=64,
        material_embedding_dim=32,
    )
    model = UnifiedPointFieldHeadsV1(cfg)
    out = model(torch.randn(1, 9, cfg.sampled_feature_dim))
    owner_norm = torch.linalg.vector_norm(out["owner_embedding"], dim=-1)
    material_norm = torch.linalg.vector_norm(out["material_embedding"], dim=-1)
    assert torch.allclose(owner_norm, torch.ones_like(owner_norm), atol=1e-4, rtol=1e-4)
    assert torch.allclose(material_norm, torch.ones_like(material_norm), atol=1e-4, rtol=1e-4)
