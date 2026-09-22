import numpy as np
import torch

from models.iris.v3.full_screen_negative_space_audit_v3 import audit_full_screen_negative_space_v3


class _XBandField(torch.nn.Module):
    def __init__(self, half_width: float):
        super().__init__()
        self.half_width = float(half_width)

    def query(self, scene_planes, query_points):
        sdf = torch.abs(query_points[..., 0]) - self.half_width
        return {"sdf": sdf, "log_uncertainty": torch.zeros_like(sdf)}


def _inputs():
    foreground = np.zeros((8, 8), dtype=bool)
    foreground[:, 3:5] = True
    admitted = np.ones_like(foreground)
    common = dict(
        scene_planes=torch.zeros(1, 1, 1, 1),
        foreground_mask=foreground,
        admitted_mask=admitted,
        camera_origin_normalized=np.asarray([0.0, 0.0, -2.0]),
        camera_right=np.asarray([1.0, 0.0, 0.0]),
        camera_screen_up=np.asarray([0.0, 1.0, 0.0]),
        camera_forward=np.asarray([0.0, 0.0, 1.0]),
        camera_half_extent_normalized=1.0,
        pixel_chunk=7,
    )
    return foreground, common


def test_full_screen_audit_returns_raster_maps_and_zero_crossings():
    foreground, common = _inputs()
    out = audit_full_screen_negative_space_v3(model=_XBandField(0.5), **common)
    assert out["ray_min_sdf"].shape == foreground.shape
    assert out["zero_crossing_mask"].shape == foreground.shape
    assert out["margin_violation_mask"].shape == foreground.shape
    assert out["background_pixel_count"] == int((~foreground).sum())
    assert 0.0 < out["zero_crossing_fraction"] < 1.0
    assert set(out["distance_binned"]) == {"0_2", "2_8", "8_32", "gt_32"}


def test_full_screen_audit_clean_field_has_no_background_zero_crossing():
    _, common = _inputs()
    out = audit_full_screen_negative_space_v3(model=_XBandField(0.2), **common)
    assert out["zero_crossing_fraction"] == 0.0
