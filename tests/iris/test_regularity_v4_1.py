import torch

from models.iris.v4.regularity_v4_1 import (
    NarrowBandEikonalPolicyV41,
    finite_difference_narrow_band_eikonal_v41,
    make_checkpoint_score_v41,
)


class _PlaneField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def query(self, scene_planes, points):
        sdf = self.scale * points[..., 0]
        return {"sdf": sdf, "log_uncertainty": torch.zeros_like(sdf)}


class _OscillatoryField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def query(self, scene_planes, points):
        x = points[..., 0]
        sdf = self.scale * (x + 0.08 * torch.sin(35.0 * x))
        return {"sdf": sdf, "log_uncertainty": torch.zeros_like(sdf)}


def _inputs():
    x = torch.linspace(-0.015, 0.015, 64)
    points = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1).unsqueeze(0)
    target = x.unsqueeze(0)
    planes = torch.zeros(1, 3, 1, 2, 2, requires_grad=True)
    return planes, points, target


def test_v41_eikonal_is_near_zero_for_unit_plane_sdf_and_backpropagates():
    model = _PlaneField()
    planes, points, target = _inputs()
    out = finite_difference_narrow_band_eikonal_v41(
        model=model,
        scene_planes=planes,
        local_points_normalized=points,
        local_target_sdf=target,
        policy=NarrowBandEikonalPolicyV41(maximum_points_per_step=64),
    )
    assert int(out["sample_count"]) == 64
    assert float(out["total"]) < 1e-6
    assert abs(float(out["gradient_norm_mean"]) - 1.0) < 1e-3
    out["total"].backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_v41_eikonal_penalizes_oscillatory_near_surface_field():
    model = _OscillatoryField()
    planes, points, target = _inputs()
    out = finite_difference_narrow_band_eikonal_v41(
        model=model,
        scene_planes=planes,
        local_points_normalized=points,
        local_target_sdf=target,
        policy=NarrowBandEikonalPolicyV41(maximum_points_per_step=64),
    )
    assert float(out["total"]) > 0.05
    assert float(out["gradient_norm_abs_error_mean"]) > 0.1


def test_v41_checkpoint_score_prioritizes_hard_source_sign_over_lower_loss():
    hard_better = make_checkpoint_score_v41(
        max_background_zero_crossing_fraction=0.01,
        max_foreground_miss_fraction=0.02,
        source_exterior_negative_fraction=0.0,
        max_background_margin_violation_fraction=0.1,
        eikonal_abs_error_mean=0.2,
        scalar_loss=10.0,
    )
    low_loss_but_hard_worse = make_checkpoint_score_v41(
        max_background_zero_crossing_fraction=0.02,
        max_foreground_miss_fraction=0.0,
        source_exterior_negative_fraction=0.0,
        max_background_margin_violation_fraction=0.0,
        eikonal_abs_error_mean=0.0,
        scalar_loss=0.0,
    )
    assert hard_better < low_loss_but_hard_worse
