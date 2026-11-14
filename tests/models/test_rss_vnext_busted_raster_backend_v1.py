from __future__ import annotations

import pytest
import torch

from models.rss_vnext.busted_raster_backend_v1 import BustedTorchRasterV1


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(1)


def _scene():
    # Two coplanar screen projections: the second triangle is nearer.
    near = [
        [-.8, -.8, -.5, 1.], [.8, -.8, -.5, 1.],
        [0., .8, -.5, 1.],
    ]
    far = [
        [-.8, -.8, .5, 1.], [.8, -.8, .5, 1.],
        [0., .8, .5, 1.],
    ]
    clip = torch.tensor(far+near, dtype=torch.float32).unsqueeze(0).requires_grad_()
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    return clip, faces


def test_first_hit_depth_is_per_pixel_never_average_face_order():
    clip, faces = _scene()
    r, _ = BustedTorchRasterV1.rasterize(
        BustedTorchRasterV1.RasterizeCudaContext(),
        clip, faces, resolution=(32, 32),
    )
    # Center pixel is inside both: nearer triangle (face id 2) wins.
    assert r[0, 16, 16, 3].item() == 2
    assert r[0, 16, 16, 2].item() > 0.
    assert r[0, 0, 0, 3].item() == 0
    # Reversing the triangle order does not reverse the depth decision.
    flipped, _ = BustedTorchRasterV1.rasterize(
        BustedTorchRasterV1.RasterizeCudaContext(),
        clip, faces.flip(0), resolution=(32, 32),
    )
    assert flipped[0, 16, 16, 3].item() == 1


def test_selected_barycentric_surface_and_silhouette_have_gradients():
    clip, faces = _scene()
    r, _ = BustedTorchRasterV1.rasterize(
        BustedTorchRasterV1.RasterizeCudaContext(),
        clip, faces, resolution=(32, 32),
    )
    attrs = torch.tensor([
        [[0., 0., 0.], [1., 0., 0.], [.5, 1., 0.],
         [0., 0., 2.], [1., 0., 2.], [.5, 1., 2.]]
    ], requires_grad=True)
    value, _ = BustedTorchRasterV1.interpolate(attrs, r, faces)
    assert value.shape == (1, 32, 32, 3)
    assert value[0, 16, 16, 2].item() == pytest.approx(2.)
    antialiased = BustedTorchRasterV1.antialias(
        torch.ones((1, 32, 32, 4)), r, clip, faces,
    )
    assert ((antialiased[..., 3] > 0.) & (antialiased[..., 3] < 1.)).any()
    loss = value[0, 16, 16, 0] + antialiased[..., 3].sum()
    loss.backward()
    assert attrs.grad is not None and attrs.grad.abs().sum() > 0
    assert clip.grad is not None and torch.isfinite(clip.grad).all()
    assert clip.grad[..., :2].abs().sum() > 0


def test_offscreen_face_does_not_forge_coverage():
    clip = torch.tensor([[[3., 3., .0, 1.], [4., 3., .0, 1.],
                          [3., 4., .0, 1.]]])
    faces = torch.tensor([[0, 1, 2]])
    r, _ = BustedTorchRasterV1.rasterize(
        BustedTorchRasterV1.RasterizeCudaContext(), clip, faces,
        resolution=(32, 32),
    )
    assert not torch.any(r[..., 3])
    assert not torch.any(r[..., 2])


def test_invalid_index_and_nonfinite_inputs_fail_closed():
    clip, faces = _scene()
    with pytest.raises(ValueError, match="RESOURCE_OR_FACE_DOMAIN"):
        BustedTorchRasterV1.rasterize(
            BustedTorchRasterV1.RasterizeCudaContext(), clip,
            torch.tensor([[0, 1, 99]]), resolution=(8, 8),
        )
    broken = clip.detach().clone()
    broken[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="ABI_INVALID"):
        BustedTorchRasterV1.rasterize(
            BustedTorchRasterV1.RasterizeCudaContext(), broken, faces,
            resolution=(8, 8),
        )
