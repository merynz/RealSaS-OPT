import numpy as np
import torch

from models.iris.v4.source_exterior_v4 import (
    SourceExteriorPolicyV4,
    certify_source_exterior_points_v4,
    source_exterior_metric_barrier_v4,
)


def _fixture():
    masks = np.zeros((8, 32, 32), dtype=bool)
    masks[:, 10:22, 10:22] = True
    origins = np.repeat(np.asarray([[0.0, 0.0, -2.0]], dtype=np.float64), 8, axis=0)
    right = np.repeat(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), 8, axis=0)
    up = np.repeat(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64), 8, axis=0)
    forward = np.repeat(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64), 8, axis=0)
    half = np.ones(8, dtype=np.float64)
    return masks, origins, right, up, forward, half


def test_source_exterior_certification_is_source_only_and_out_of_frame_unknown():
    masks, origins, right, up, forward, half = _fixture()
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.80, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    out = certify_source_exterior_points_v4(
        points,
        masks,
        origins,
        right,
        up,
        forward,
        half,
    )
    assert bool(out["certified"][0]) is False
    assert bool(out["certified"][1]) is True
    assert float(out["margin_normalized"][1]) > 0.0
    assert int(out["witness_view"][1]) >= 0
    assert bool(out["certified"][2]) is False
    assert int(out["observed_view_count"][2]) == 0


def test_quantization_guard_makes_near_boundary_certificate_conservative():
    masks, origins, right, up, forward, half = _fixture()
    policy = SourceExteriorPolicyV4(pixel_quantization_guard_px=np.sqrt(2.0))
    points = np.asarray([[0.40, 0.0, 0.0], [0.80, 0.0, 0.0]], dtype=np.float64)
    out = certify_source_exterior_points_v4(
        points,
        masks,
        origins,
        right,
        up,
        forward,
        half,
        policy=policy,
    )
    assert float(out["margin_normalized"][0]) <= float(out["margin_normalized"][1])
    assert float(out["margin_normalized"][1]) <= policy.maximum_margin_normalized + 1e-7


def test_source_exterior_metric_barrier_pushes_violations_positive():
    sdf = torch.tensor([[-0.10, 0.01, 0.08]], dtype=torch.float64, requires_grad=True)
    margin = torch.tensor([[0.02, 0.02, 0.02]], dtype=torch.float64)
    out = source_exterior_metric_barrier_v4(sdf, margin)
    out["total"].backward()
    assert torch.isfinite(sdf.grad).all()
    assert float(sdf.grad[0, 0]) < 0.0
    assert float(sdf.grad[0, 1]) < 0.0
    assert float(sdf.grad[0, 2]) == 0.0
    assert float(out["negative_fraction"]) > 0.0
