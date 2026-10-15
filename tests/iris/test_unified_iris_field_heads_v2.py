import torch

from models.iris.vnext.field_heads_v2 import (
    UnifiedPointFieldHeadConfigV2,
    UnifiedPointFieldHeadsV2,
)


def test_v2_point_field_is_open_surface_aware_and_f_extraction_free():
    config = UnifiedPointFieldHeadConfigV2(
        triplane_channels=80,
        hidden_dim=128,
        layers=3,
        owner_embedding_dim=64,
        material_embedding_dim=32,
    )
    model = UnifiedPointFieldHeadsV2(config)
    features = torch.randn(2, 5, config.sampled_feature_dim)
    outputs = model(features)

    for name in (
        "unsigned_distance",
        "oriented_level_set",
        "orientation_validity",
        "geometry_log_uncertainty",
    ):
        assert name in outputs
        assert outputs[name].shape == (2, 5)

    assert torch.all(outputs["unsigned_distance"] >= 0.0)
    assert torch.all(outputs["orientation_validity"] >= 0.0)
    assert torch.all(outputs["orientation_validity"] <= 1.0)

    assert "sdf" not in outputs
    assert "extraction_delta" not in outputs
    assert "extraction_cell_weights" not in outputs


def test_v2_shared_point_feature_remains_available_to_downstream_f_queries():
    config = UnifiedPointFieldHeadConfigV2(
        triplane_channels=80,
        hidden_dim=128,
        layers=3,
    )
    model = UnifiedPointFieldHeadsV2(config)
    features = torch.randn(1, 7, config.sampled_feature_dim)
    outputs = model(features)
    assert outputs["shared_point_feature"].shape == (1, 7, 128)
