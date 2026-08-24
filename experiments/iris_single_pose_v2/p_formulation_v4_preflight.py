from __future__ import annotations

import json
import torch

from losses import total_loss
from model import camera_basis_from_yaw, IRISV2Config
from model_pv4 import IRISSinglePoseV2PV4, estimate_sheet_half_extent_from_alpha, reconstruct_p_from_observable_scale


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


def make_sheet(res: int, target_half_extent: float) -> torch.Tensor:
    """Synthetic ordered RGBA sheet whose canonical max bbox extent is one."""
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


def make_batch(images, yaw, tracks=8, geom=16):
    b, v, _, _, _ = images.shape
    he = estimate_sheet_half_extent_from_alpha(images, yaw)
    right, up, forward = camera_basis_from_yaw(yaw, batch=b, views=v, dtype=torch.float32)
    geom_xy = torch.rand(b, v, geom, 2) * 1.6 - 0.8
    depth = torch.rand(b, v, geom) * 0.6 - 0.3
    geom_p = (
        he[:, None, None, None] * geom_xy[..., 0, None] * right[:, :, None, :]
        - he[:, None, None, None] * geom_xy[..., 1, None] * up[:, :, None, :]
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


def expected_model_shapes(res: int):
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
    scale_cases = [0.54, 0.5570941257476807, 0.6172158837318421]
    scale_report = []

    for res in (256, 512, 1024):
        for target in scale_cases:
            images = make_sheet(res, target)
            he = estimate_sheet_half_extent_from_alpha(images, yaw)
            if he.shape != (1,) or not torch.isfinite(he).all() or float(he[0]) <= 0:
                raise RuntimeError((res, target, he))
            alpha = images[0, 0, 3] >= 0.5
            _, xs = torch.nonzero(alpha, as_tuple=True)
            span = int(xs.max() - xs.min() + 1)
            expected = 1.0 / (2.0 * (span / float(res)))
            if abs(float(he[0]) - expected) > 1e-7:
                raise RuntimeError((res, target, float(he[0]), expected))

            depth = torch.zeros(1, 8, 1, res // 2, res // 2)
            p = reconstruct_p_from_observable_scale(depth, yaw, he)
            right, up, _ = camera_basis_from_yaw(yaw, batch=1, views=8, dtype=torch.float32)
            h = p.shape[-1]
            g = 2.0 * (torch.arange(h, dtype=torch.float32) + 0.5) / float(h) - 1.0
            gy, gx = torch.meshgrid(g, g, indexing="ij")
            rp = (p * right[..., :, None, None]).sum(2)
            zp = (p * up[..., :, None, None]).sum(2)
            er = float((rp - he[:, None, None, None] * gx[None, None]).abs().max())
            ez = float((zp + he[:, None, None, None] * gy[None, None]).abs().max())
            if max(er, ez) > 3e-6:
                raise RuntimeError((res, target, er, ez))
            scale_report.append({
                "resolution": res,
                "target_half_extent_fixture": target,
                "observable_half_extent": float(he[0]),
                "screen_plane_max_abs_error": max(er, ez),
            })

    # Execute the actual P-V4 model at all contracted resolutions, not merely the helper formula.
    model = IRISSinglePoseV2PV4(tiny_cfg()).eval()
    model_resolution_shapes = {}
    with torch.no_grad():
        for i, res in enumerate((256, 512, 1024)):
            images = make_sheet(res, scale_cases[i])
            out = model(images, yaw)
            expected = expected_model_shapes(res)
            for k, shape in expected.items():
                if tuple(out[k].shape) != shape:
                    raise RuntimeError((res, k, tuple(out[k].shape), shape))
            model_resolution_shapes[str(res)] = {k: list(v.shape) for k, v in out.items()}

    # Full mixed-loss/backward semantic smoke with P-V4 active.
    model = IRISSinglePoseV2PV4(tiny_cfg()).train()
    images = make_sheet(64, 0.6172158837318421)
    batch = make_batch(images, yaw)
    out = model(batch["images"], batch["yaw_deg"])
    parts = total_loss(out, batch, epoch=4, warmup_epochs=4)
    if not torch.isfinite(parts["total"]):
        raise RuntimeError("nonfinite P-V4 full loss")
    parts["total"].backward()
    grad = model.p_depth_head.weight.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) <= 0:
        raise RuntimeError("P-V4 depth head missing finite nonzero gradient")

    report = {
        "schema": "RealSaS.IRISSinglePoseV2.PFormulationV4Preflight.v1",
        "status": "PASS",
        "image_only_model_input": True,
        "camera_half_extent_model_input": False,
        "observable_scale_authority": "RGBA alpha occupancy + canonical largest bbox extent=1",
        "scale_cases": scale_report,
        "model_resolution_shapes": model_resolution_shapes,
        "P_field_resolution": "R/2",
        "full_loss_finite": True,
        "depth_head_gradient_finite_nonzero": True,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
