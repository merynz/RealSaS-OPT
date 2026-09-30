import torch

from models.iris.v4.regularity_v4_1 import NarrowBandEikonalPolicyV41
from models.iris.v4.train_demo_fit_v4 import SourceConstraintRayBatchV4
from models.iris.v4.train_demo_fit_v4_1 import compute_v41_demo_geometry_objective


class _LinearField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.2))

    def query(self, scene_planes, points):
        sdf = self.weight * points[..., 0]
        return {"sdf": sdf, "log_uncertainty": torch.zeros_like(sdf)}


def _view_batch(view_index: int) -> SourceConstraintRayBatchV4:
    points = torch.tensor(
        [[
            [[-0.5, 0.0, -0.5], [-0.5, 0.0, 0.0], [-0.5, 0.0, 0.5]],
            [[0.5, 0.0, -0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5]],
        ]], dtype=torch.float32,
    )
    return SourceConstraintRayBatchV4(
        ray_points_normalized=points,
        target_foreground=torch.tensor([[1.0, 0.0]]),
        component_id=torch.tensor([[0, -1]], dtype=torch.long),
        boundary=torch.tensor([[1.0, 0.0]]),
        distance_to_foreground_px=torch.tensor([[0.0, 10.0]]),
        hard_replay=torch.tensor([[True, True]]),
        camera_half_extent_normalized=1.0,
        raster_width=1024,
        view_index=view_index,
    )


def test_v41_objective_adds_finite_difference_eikonal_without_changing_view_contract():
    model = _LinearField()
    x = torch.linspace(-0.015, 0.015, 32)
    local_points = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1).unsqueeze(0)
    out = compute_v41_demo_geometry_objective(
        model=model,
        scene_planes=torch.zeros(1, 3, 1, 2, 2),
        local_points_normalized=local_points,
        local_target_sdf=x.unsqueeze(0),
        teacher_surface_zero_points_normalized=torch.zeros(1, 8, 3),
        source_exterior_points_normalized=torch.tensor([[[0.8,0,0],[0.9,0,0]]], dtype=torch.float32),
        source_exterior_margin_normalized=torch.tensor([[0.02,0.02]], dtype=torch.float32),
        source_view_batches=tuple(_view_batch(v) for v in range(8)),
        eikonal_policy=NarrowBandEikonalPolicyV41(maximum_points_per_step=32, top_level_weight=0.02),
    )
    assert len(out["source_views"]) == 8
    assert float(out["eikonal_total"]) > 0.0
    assert int(out["eikonal_sample_count"]) == 32
    expected = out["base_total_before_eikonal"] + 0.02 * out["eikonal_total"]
    assert torch.allclose(out["total"], expected)
    out["total"].backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad)
