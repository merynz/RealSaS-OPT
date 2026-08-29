import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("depth_corruption_v1", HERE / "depth_corruption_v1.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def pixels(res=64):
    yy, xx = np.mgrid[8:56, 10:54]
    return np.sort((yy.reshape(-1) * res + xx.reshape(-1)).astype(np.int64))


def test_zero_epsilon_is_bit_exact_identity():
    pix = pixels()
    P = np.arange(len(pix) * 3, dtype=np.float32).reshape(-1, 3) / 1000
    s = m.DepthCorruptionSpec(0.0, 32.0, "ONE_BAD_HASHED")
    d = m.depth_delta_for_view("asset", 0, pix, resolution=64, spec=s)
    out = m.corrupt_points_along_ray(P, [0, 1, 0], d)
    assert np.array_equal(P, out)


def test_nonzero_corruption_is_ray_aligned_and_rms_calibrated():
    pix = pixels()
    P = np.zeros((len(pix), 3), np.float32)
    eps = 0.003
    s = m.DepthCorruptionSpec(eps, 5.0, "ALL8")
    d = m.depth_delta_for_view("asset", 3, pix, resolution=64, spec=s)
    out = m.corrupt_points_along_ray(P, [1, 2, 0], d)
    diag = m.corruption_diagnostics(P, out, [1, 2, 0], d)
    assert abs(diag["depth_delta_rms"] - eps) < 2e-7
    assert diag["max_ray_perpendicular_residual"] < 5e-9


def test_determinism():
    pix = pixels()
    s = m.DepthCorruptionSpec(0.01, 8.0, "ALL8")
    a = m.depth_delta_for_view("asset-x", 2, pix, resolution=64, spec=s)
    b = m.depth_delta_for_view("asset-x", 2, pix, resolution=64, spec=s)
    assert np.array_equal(a, b)


def test_ell_changes_shape_not_rms():
    pix = pixels()
    eps = 0.004
    a = m.depth_delta_for_view(
        "asset", 1, pix, resolution=64,
        spec=m.DepthCorruptionSpec(eps, 0.0, "ALL8"),
    )
    b = m.depth_delta_for_view(
        "asset", 1, pix, resolution=64,
        spec=m.DepthCorruptionSpec(eps, 10.0, "ALL8"),
    )
    assert not np.allclose(a, b)
    assert abs(float(np.sqrt(np.mean(a.astype(np.float64)**2))) - eps) < 2e-7
    assert abs(float(np.sqrt(np.mean(b.astype(np.float64)**2))) - eps) < 2e-7


def test_asymmetry_cardinalities():
    assert len(m.affected_views("a", "ALL8")) == 8
    assert len(m.affected_views("a", "ONE_BAD_HASHED")) == 1
    vv = m.affected_views("a", "TWO_OPPOSITE_HASHED")
    assert len(vv) == 2 and ((vv[1] - vv[0]) % 4 == 0)
    assert len(m.affected_views("a", "FOUR_ALTERNATING_HASHED")) == 4
