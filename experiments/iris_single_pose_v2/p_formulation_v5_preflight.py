from __future__ import annotations

import inspect
import json

import torch
import torch.nn.functional as F

from losses import total_loss
from model import camera_basis_from_yaw, IRISV2Config
from model_pv4 import estimate_sheet_half_extent_from_alpha
from model_pv5 import IRISSinglePoseV2PV5, estimate_native_sheet_half_extent


def tiny_cfg():
    return IRISV2Config(
        widths=(8, 16, 24, 32),
        coarse_dim=16,
        fine_dim=8,
        context_hw=8,
        within_heads=4,
        cross_heads=4,
        cross_layers=1,
    )


def make_native_sheet(target_half_extent: float) -> torch.Tensor:
    res = 1024
    x = torch.zeros(1, 8, 4, res, res, dtype=torch.float32)
    span = max(2, min(res, int(round(res / (2.0 * float(target_half_extent))))))
    span_y = max(2, int(round(0.72 * span)))
    span_v2 = max(2, int(round(0.61 * span)))
    cx = cy = res // 2

    def paint(v, width, height):
        x0 = max(0, cx - width // 2); x1 = min(res, x0 + width)
        y0 = max(0, cy - height // 2); y1 = min(res, y0 + height)
        x[:, v, :3, y0:y1, x0:x1] = 1.0
        x[:, v, 3, y0:y1, x0:x1] = 1.0

    for v in range(8):
        paint(v, span_v2 if v in (2, 6) else span, span_y)
    return x


def resize_sheet(images: torch.Tensor, res: int) -> torch.Tensor:
    if images.shape[-1] == res:
        return images.clone()
    b, v, c, h, w = images.shape
    y = F.interpolate(
        images.reshape(b * v, c, h, w),
        size=(res, res),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return y.reshape(b, v, c, res, res)


def make_batch(images, yaw, native_h, tracks=8, geom=16):
    b, v, _, _, _ = images.shape
    right, up, forward = camera_basis_from_yaw(yaw, batch=b, views=v, dtype=torch.float32)
    geom_xy = torch.rand(b, v, geom, 2) * 1.6 - 0.8
    depth = torch.rand(b, v, geom) * 0.6 - 0.3
    geom_p = (
        native_h[:, None, None, None] * geom_xy[..., 0, None] * right[:, :, None, :]
        - native_h[:, None, None, None] * geom_xy[..., 1, None] * up[:, :, None, :]
        + depth[..., None] * forward[:, :, None, :]
    )
    geom_n = torch.nn.functional.normalize(torch.rand(b, v, geom, 3) - 0.5, dim=-1)
    base = torch.rand(b, tracks, 2) * 1.0 - 0.5
    track_xy = base[:, :, None, :].repeat(1, 1, v, 1)
    track_p = torch.rand(b, tracks, 3) - 0.5
    return {
        "images": images,
        "yaw_deg": yaw,
        "geom_xy": geom_xy,
        "geom_p": geom_p,
        "geom_n": geom_n,
        "geom_mask": torch.ones(b, v, geom, dtype=torch.bool),
        "track_xy": track_xy,
        "track_visible": torch.ones(b, tracks, v, dtype=torch.bool),
        "track_p": track_p,
    }


def expected_shapes(res: int):
    cfg = tiny_cfg()
    return {
        "P": (1, 8, 3, res // 2, res // 2),
        "N": (1, 8, 3, res // 2, res // 2),
        "U_geo": (1, 8, 1, res // 2, res // 2),
        "Z_coarse": (1, 8, cfg.coarse_dim, res // 8, res // 8),
        "Z_fine": (1, 8, cfg.fine_dim, res // 2, res // 2),
    }


def main():
    torch.manual_seed(20260824)
    yaw = torch.arange(8, dtype=torch.float32)[None] * 45.0

    sig = inspect.signature(IRISSinglePoseV2PV5.forward)
    if list(sig.parameters) != ["self", "images", "yaw_deg", "sheet_half_extent"]:
        raise RuntimeError(f"P-V5 forward signature drift: {sig}")
    src = inspect.getsource(IRISSinglePoseV2PV5.forward).lower()
    if "camera" in src or "estimate_sheet_half_extent_from_alpha" in src or "estimate_native_sheet_half_extent" in src:
        raise RuntimeError("P-V5 learner forward must consume transported observable scale, not derive/read camera scale")

    resolution_report = []
    model = IRISSinglePoseV2PV5(tiny_cfg()).eval()
    scale_cases = [0.54, 0.5570941257476807, 0.6172158837318421]
    with torch.no_grad():
        for target in scale_cases:
            native = make_native_sheet(target)
            h_native = estimate_native_sheet_half_extent(native, yaw)
            if h_native.requires_grad or tuple(h_native.shape) != (1,) or not torch.isfinite(h_native).all():
                raise RuntimeError("bad native observable scale authority")
            # Prove helper equality at the native boundary only.
            h_direct = estimate_sheet_half_extent_from_alpha(native, yaw)
            if not torch.equal(h_native.cpu(), h_direct.detach().cpu()):
                raise RuntimeError("native observable scale helper drift")

            used = []
            for res in (1024, 512, 256):
                learner_images = resize_sheet(native, res)
                out = model(learner_images, yaw, h_native)
                for k, shape in expected_shapes(res).items():
                    if tuple(out[k].shape) != shape:
                        raise RuntimeError((target, res, k, tuple(out[k].shape), shape))
                used.append(float(h_native[0]))
                resolution_report.append({
                    "target_fixture": target,
                    "learner_resolution": res,
                    "native_observable_half_extent": float(h_native[0]),
                    "P_shape": list(out["P"].shape),
                })
            if not (used[0] == used[1] == used[2]):
                raise RuntimeError(f"native scale transport drift: {used}")

    # Negative contract: non-native scale estimation must be rejected by the native helper.
    try:
        estimate_native_sheet_half_extent(resize_sheet(make_native_sheet(0.5570941257476807), 512), yaw)
    except ValueError:
        pass
    else:
        raise RuntimeError("native scale helper incorrectly accepted resized 512 learner input")

    # Full mixed-loss/backward with a transported scale at learner R64.
    native = make_native_sheet(0.6172158837318421)
    h_native = estimate_native_sheet_half_extent(native, yaw)
    images = resize_sheet(native, 64)
    model = IRISSinglePoseV2PV5(tiny_cfg()).train()
    batch = make_batch(images, yaw, h_native)
    out = model(batch["images"], batch["yaw_deg"], h_native)
    parts = total_loss(out, batch, epoch=4, warmup_epochs=4)
    if not torch.isfinite(parts["total"]):
        raise RuntimeError("nonfinite P-V5 full loss")
    parts["total"].backward()
    grad = model.p_depth_head.weight.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) <= 0:
        raise RuntimeError("P-V5 depth head missing finite nonzero gradient")

    report = {
        "schema": "RealSaS.IRISSinglePoseV2.PFormulationV5Preflight.v1",
        "status": "PASS",
        "overall_extractor_image_only": True,
        "native_scale_source": "original ordered 1024 RGBA alpha",
        "learner_forward_inputs": ["images", "yaw_deg", "sheet_half_extent"],
        "camera_half_extent_input": False,
        "scale_reestimated_after_resize": False,
        "resolution_report": resolution_report,
        "P_field_resolution": "R/2",
        "full_loss_finite": True,
        "depth_head_gradient_finite_nonzero": True,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
