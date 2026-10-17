from __future__ import annotations

import torch

from tools.iris_vnext.a100_preflight_v2 import independent_analytic_sphere_field_v2


def test_a100_preflight_v2_analytic_field_is_independent_leaf_at_origin():
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    field = independent_analytic_sphere_field_v2(vertices)

    assert field.is_leaf
    assert field.requires_grad
    assert bool(torch.isfinite(field).all())
    assert torch.allclose(field.detach(), torch.tensor([-0.55, -0.30]))

    field.sum().backward()
    assert vertices.grad is None
    assert field.grad is not None
    assert bool(torch.isfinite(field.grad).all())
    assert torch.all(field.grad == 1)
