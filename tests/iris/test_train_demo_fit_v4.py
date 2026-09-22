import pytest
import torch

from models.iris.v3.dense_source_coverage_v3 import DenseSourceCoveragePolicyV3
from models.iris.v4.train_demo_fit_v4 import (
    SourceConstraintRayBatchV4,
    component_balanced_foreground_existence_hinge_v4,
    compute_v4_demo_geometry_objective,
)


class _ScalarField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.01))

    def query(self, scene_planes, query_points):
        sdf = self.weight + 0.05 * query_points[..., 0]
        return {"sdf": sdf, "log_uncertainty": torch.zeros_like(sdf)}


def _view_batch(view_index: int) -> SourceConstraintRayBatchV4:
    points = torch.tensor(
        [[
            [[-0.5, 0.0, -0.5], [-0.5, 0.0, 0.0], [-0.5, 0.0, 0.5]],
            [[0.5, 0.0, -0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5]],
        ]],
        dtype=torch.float32,
    )
    return SourceConstraintRayBatchV4(
        ray_points_normalized=points,
        target_foreground=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        component_id=torch.tensor([[0, -1]], dtype=torch.long),
        boundary=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        distance_to_foreground_px=torch.tensor([[0.0, 10.0]], dtype=torch.float32),
        hard_replay=torch.tensor([[False, True]]),
        camera_half_extent_normalized=1.0,
        raster_width=1024,
        view_index=view_index,
    )


def _inputs():
    return dict(
        local_points_normalized=torch.tensor([[[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]]),
        local_target_sdf=torch.tensor([[0.02, -0.02]]),
        teacher_surface_zero_points_normalized=torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
        source_exterior_points_normalized=torch.tensor([[[0.8, 0.0, 0.0], [0.9, 0.0, 0.0]]]),
        source_exterior_margin_normalized=torch.tensor([[0.02, 0.02]]),
    )


def test_foreground_existence_hinge_pushes_missing_ray_negative_then_saturates():
    policy = DenseSourceCoveragePolicyV3()
    boundary_margin = 0.5 * policy.occupancy_beta_normalized
    interior_margin = policy.occupancy_beta_normalized

    missing = torch.tensor([[[0.02, 0.03, 0.04]]], dtype=torch.float64, requires_grad=True)
    out = component_balanced_foreground_existence_hinge_v4(
        missing,
        torch.ones((1, 1), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.long),
        torch.ones((1, 1), dtype=torch.float64),
        boundary_inside_margin_normalized=boundary_margin,
        interior_inside_margin_normalized=interior_margin,
    )
    out["total"].backward()
    assert float(out["total"]) > 0.0
    assert torch.isfinite(missing.grad).all()
    assert float(missing.grad[0, 0, 0]) > 0.0

    satisfied = torch.tensor(
        [[[-2.0 * boundary_margin, 0.03, 0.04]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    done = component_balanced_foreground_existence_hinge_v4(
        satisfied,
        torch.ones((1, 1), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.long),
        torch.ones((1, 1), dtype=torch.float64),
        boundary_inside_margin_normalized=boundary_margin,
        interior_inside_margin_normalized=interior_margin,
    )
    done["total"].backward()
    assert float(done["total"]) == 0.0
    assert torch.equal(satisfied.grad, torch.zeros_like(satisfied.grad))


def test_foreground_boundary_margin_is_weaker_than_interior_margin():
    policy = DenseSourceCoveragePolicyV3()
    boundary_margin = 0.5 * policy.occupancy_beta_normalized
    interior_margin = policy.occupancy_beta_normalized
    sdf = torch.tensor(
        [[[-0.75 * interior_margin, 0.02], [-0.75 * interior_margin, 0.02]]],
        dtype=torch.float64,
    )
    out = component_balanced_foreground_existence_hinge_v4(
        sdf,
        torch.ones((1, 2), dtype=torch.float64),
        torch.zeros((1, 2), dtype=torch.long),
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        boundary_inside_margin_normalized=boundary_margin,
        interior_inside_margin_normalized=interior_margin,
    )
    assert float(out["total"]) > 0.0
    assert float(out["satisfied_fraction"]) == 0.5


def test_v4_objective_binds_all_eight_views_and_metric_3d_exterior():
    model = _ScalarField()
    out = compute_v4_demo_geometry_objective(
        model=model,
        scene_planes=torch.zeros(1, 1, 1, 1),
        source_view_batches=tuple(_view_batch(v) for v in range(8)),
        **_inputs(),
    )
    assert len(out["source_views"]) == 8
    assert {row["view_index"] for row in out["source_views"]} == set(range(8))
    assert float(out["source_exterior_total"]) >= 0.0
    expected = (
        out["structural_total"]
        + out["foreground_existence_total"]
        + out["negative_space_total"]
        + out["source_exterior_total"]
    )
    assert torch.allclose(out["total"], expected)
    out["total"].backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad)
    assert abs(float(model.weight.grad)) > 0.0


def test_v4_rejects_pair_only_training_to_prevent_arm_c_retention_failure():
    model = _ScalarField()
    with pytest.raises(ValueError, match="all eight source views"):
        compute_v4_demo_geometry_objective(
            model=model,
            scene_planes=torch.zeros(1, 1, 1, 1),
            source_view_batches=(_view_batch(0), _view_batch(4)),
            **_inputs(),
        )


def test_v4_rejects_missing_source_exterior_authority():
    model = _ScalarField()
    bad = _inputs()
    bad["source_exterior_points_normalized"] = torch.zeros(1, 0, 3)
    bad["source_exterior_margin_normalized"] = torch.zeros(1, 0)
    with pytest.raises(ValueError, match="source exterior points"):
        compute_v4_demo_geometry_objective(
            model=model,
            scene_planes=torch.zeros(1, 1, 1, 1),
            source_view_batches=tuple(_view_batch(v) for v in range(8)),
            **bad,
        )
