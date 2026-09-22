import torch

from models.iris.v3.negative_space_barrier_v3 import (
    background_full_ray_empty_space_barrier,
)


def test_background_full_ray_barrier_pushes_every_violating_depth_sample_positive():
    sdf = torch.tensor([[[-0.20, 0.00, 0.03, 0.08]]], dtype=torch.float64, requires_grad=True)
    target = torch.zeros((1, 1), dtype=torch.float64)
    out = background_full_ray_empty_space_barrier(
        sdf,
        target,
        positive_margin=0.04,
    )
    out["total"].backward()
    assert int(out["background_ray_count"]) == 1
    assert torch.isfinite(sdf.grad).all()
    assert float(sdf.grad[0, 0, 0]) < 0.0
    assert float(sdf.grad[0, 0, 1]) < 0.0
    assert float(sdf.grad[0, 0, 2]) < 0.0
    assert float(sdf.grad[0, 0, 3]) == 0.0


def test_foreground_ray_is_not_forced_empty():
    sdf = torch.tensor([[[-0.20, -0.01, 0.03, 0.08]]], dtype=torch.float64, requires_grad=True)
    target = torch.ones((1, 1), dtype=torch.float64)
    out = background_full_ray_empty_space_barrier(sdf, target, positive_margin=0.04)
    assert float(out["total"]) == 0.0
    out["total"].backward()
    assert torch.equal(sdf.grad, torch.zeros_like(sdf.grad))


def test_fork_gap_background_semantics_remove_multiple_negative_islands_not_only_argmin():
    # Subject-free negative-space fixture: one known-background camera ray crosses two
    # disconnected false-negative SDF islands. Both islands must receive correction.
    sdf = torch.tensor([[[0.08, -0.07, 0.09, -0.02, 0.10]]], dtype=torch.float64, requires_grad=True)
    target = torch.zeros((1, 1), dtype=torch.float64)
    out = background_full_ray_empty_space_barrier(sdf, target, positive_margin=0.04)
    out["total"].backward()
    grad = sdf.grad[0, 0]
    assert float(grad[1]) < 0.0
    assert float(grad[3]) < 0.0
    assert float(grad[0]) == 0.0
    assert float(grad[2]) == 0.0
    assert float(grad[4]) == 0.0


def test_concave_notch_background_ray_keeps_empty_space_authority():
    # A notch/gap is source-background even if neighboring rays are foreground.
    sdf = torch.tensor(
        [[
            [-0.05, 0.02, 0.06],
            [-0.05, -0.02, 0.01],
            [-0.05, 0.02, 0.06],
        ]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    out = background_full_ray_empty_space_barrier(sdf, target, positive_margin=0.04)
    out["total"].backward()
    grad = sdf.grad[0]
    assert torch.equal(grad[0], torch.zeros_like(grad[0]))
    assert float(grad[1, 0]) < 0.0
    assert float(grad[1, 1]) < 0.0
    assert float(grad[1, 2]) < 0.0
    assert torch.equal(grad[2], torch.zeros_like(grad[2]))
