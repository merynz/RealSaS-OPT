import torch

from models.iris.v3.dense_source_coverage_v3 import DenseSourceCoveragePolicyV3
from models.iris.v3.train_dense_fit_v3 import DenseRayTrainingBatchV3
from models.iris.v3.train_upper_bound_fit_v3 import (
    GradedNegativeSpaceBatchV3,
    compute_v3_owner_separation_upper_bound_objective,
)


class _ScalarField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.01))

    def query(self, scene_planes, query_points):
        sdf = self.weight + 0.04 * query_points[..., 0]
        return {"sdf": sdf, "log_uncertainty": torch.zeros_like(sdf)}


def _dense_batch(view_index: int) -> DenseRayTrainingBatchV3:
    points = torch.tensor(
        [[
            [[-0.4, 0.0, -0.5], [-0.4, 0.0, 0.0], [-0.4, 0.0, 0.5]],
            [[0.4, 0.0, -0.5], [0.4, 0.0, 0.0], [0.4, 0.0, 0.5]],
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


def _negative_batch(view_index: int) -> GradedNegativeSpaceBatchV3:
    points = torch.tensor(
        [[
            [[0.1, 0.0, -0.5], [0.1, 0.0, 0.0], [0.1, 0.0, 0.5]],
            [[0.3, 0.0, -0.5], [0.3, 0.0, 0.0], [0.3, 0.0, 0.5]],
        ]],
        dtype=torch.float32,
    )
    return GradedNegativeSpaceBatchV3(
        ray_points_normalized=points,
        target_margin_normalized=torch.tensor([[0.002, 0.02]], dtype=torch.float32),
        view_index=view_index,
    )


def test_upper_bound_objective_is_structural_sparse_plus_dense_plus_graded_negative_space():
    model = _ScalarField()
    policy = DenseSourceCoveragePolicyV3()
    out = compute_v3_owner_separation_upper_bound_objective(
        model=model,
        scene_planes=torch.zeros(1, 1, 1, 1),
        local_points_normalized=torch.tensor([[[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]]),
        local_target_sdf=torch.tensor([[0.02, -0.02]]),
        teacher_surface_zero_points_normalized=torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
        dense_view_batches=(_dense_batch(0), _dense_batch(4)),
        graded_negative_space_batches=(_negative_batch(0), _negative_batch(4)),
        policy=policy,
    )
    expected = (
        out["structural_sparse_total"]
        + policy.top_level_loss_weight * out["dense_total"]
        + out["graded_negative_space_total"]
    )
    assert torch.allclose(out["total"], expected)
    assert "background" not in out
    assert "certified_outside" not in out
    out["total"].backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad)
    assert abs(float(model.weight.grad)) > 0.0


def test_graded_negative_space_batch_requires_positive_metric_margin():
    bad = GradedNegativeSpaceBatchV3(
        ray_points_normalized=torch.zeros(1, 1, 3, 3),
        target_margin_normalized=torch.zeros(1, 1),
        view_index=0,
    )
    try:
        bad.validate()
    except ValueError as exc:
        assert "margins must be positive" in str(exc)
    else:
        raise AssertionError("zero metric margin must fail closed")
