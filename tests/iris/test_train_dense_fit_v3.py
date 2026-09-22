import pytest
import torch

from models.iris.v3.dense_source_coverage_v3 import DenseSourceCoveragePolicyV3
from models.iris.v3.train_dense_fit_v3 import (
    DenseRayTrainingBatchV3,
    compute_v3_dense_fit_objective,
)


class _ScalarField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.01))

    def query(self, scene_planes, query_points):
        sdf = self.weight + 0.05 * query_points[..., 0]
        return {"sdf": sdf, "log_uncertainty": torch.zeros_like(sdf)}


def _dense_batch(view_index: int, shift: float) -> DenseRayTrainingBatchV3:
    points = torch.tensor(
        [[
            [[-0.5 + shift, 0.0, -0.5], [-0.5 + shift, 0.0, 0.0], [-0.5 + shift, 0.0, 0.5]],
            [[0.5 + shift, 0.0, -0.5], [0.5 + shift, 0.0, 0.0], [0.5 + shift, 0.0, 0.5]],
        ]],
        dtype=torch.float32,
    )
    return DenseRayTrainingBatchV3(
        ray_points_normalized=points,
        target_foreground=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        component_id=torch.tensor([[0, -1]], dtype=torch.long),
        boundary=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        view_index=view_index,
    )


def _sparse_inputs():
    return dict(
        local_points_normalized=torch.tensor([[[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]]),
        local_target_sdf=torch.tensor([[0.02, -0.02]]),
        background_points_normalized=torch.tensor([[[0.8, 0.0, 0.0], [0.9, 0.0, 0.0]]]),
        certified_outside_points_normalized=torch.tensor([[[0.7, 0.0, 0.0], [0.6, 0.0, 0.0]]]),
        teacher_surface_zero_points_normalized=torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
    )


def test_arm_a_compatibility_path_is_sparse_plus_mean_dense_at_frozen_weight():
    model = _ScalarField()
    policy = DenseSourceCoveragePolicyV3()
    out = compute_v3_dense_fit_objective(
        model=model,
        scene_planes=torch.zeros(1, 1, 1, 1),
        dense_view_batches=(_dense_batch(0, 0.0), _dense_batch(4, 0.05)),
        policy=policy,
        enforce_full_ray_background_empty_space=False,
        **_sparse_inputs(),
    )
    assert torch.allclose(
        out["total"],
        out["sparse_total"] + policy.top_level_loss_weight * out["dense_total"],
    )
    assert float(out["background_full_ray_empty_space"]) == 0.0
    assert len(out["dense_views"]) == 2
    assert {row["view_index"] for row in out["dense_views"]} == {0, 4}
    out["total"].backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad)
    assert abs(float(model.weight.grad)) > 0.0


def test_arm_b_objective_adds_full_ray_background_empty_space_exactly_once():
    model = _ScalarField()
    policy = DenseSourceCoveragePolicyV3()
    out = compute_v3_dense_fit_objective(
        model=model,
        scene_planes=torch.zeros(1, 1, 1, 1),
        dense_view_batches=(_dense_batch(0, 0.0), _dense_batch(4, 0.05)),
        policy=policy,
        positive_margin=0.04,
        enforce_full_ray_background_empty_space=True,
        **_sparse_inputs(),
    )
    assert float(out["background_full_ray_empty_space"]) > 0.0
    expected = (
        out["sparse_total"]
        + policy.top_level_loss_weight * out["dense_total"]
        + out["background_full_ray_empty_space"]
    )
    assert torch.allclose(out["total"], expected)
    out["total"].backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad)
    assert abs(float(model.weight.grad)) > 0.0


def test_duplicate_dense_view_is_rejected():
    model = _ScalarField()
    with pytest.raises(ValueError, match="duplicate dense scheduled view"):
        compute_v3_dense_fit_objective(
            model=model,
            scene_planes=torch.zeros(1, 1, 1, 1),
            dense_view_batches=(_dense_batch(0, 0.0), _dense_batch(0, 0.05)),
            **_sparse_inputs(),
        )


def test_dense_batch_requires_exact_ray_metadata_shapes():
    batch = DenseRayTrainingBatchV3(
        ray_points_normalized=torch.zeros(1, 2, 3, 3),
        target_foreground=torch.zeros(1, 3),
        component_id=torch.full((1, 2), -1, dtype=torch.long),
        boundary=torch.zeros(1, 2),
        view_index=0,
    )
    with pytest.raises(ValueError, match="target_foreground must be"):
        batch.validate()
