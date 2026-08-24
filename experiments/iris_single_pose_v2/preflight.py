from __future__ import annotations

import json
import torch
import numpy as np

from coords import pixel_center_to_grid, grid_to_pixel_center, grid_to_cell_index, cell_index_to_grid, field_cell_centers
from model import IRISSinglePoseV2, IRISV2Config, camera_basis_from_yaw, reconstruct_p_from_view_depth
from losses import total_loss, _positive_set_cycle_loss, sample_field
from matcher import match_query, MatcherConfig
from metrics import native_pixel_error, exact_coarse_cell_hit


ORTHO_HALF_EXTENT = 0.54


def tiny_cfg():
    return IRISV2Config(
        widths=(8, 16, 24, 32),
        coarse_dim=16,
        fine_dim=8,
        context_hw=8,
        within_heads=4,
        cross_heads=4,
        cross_layers=1,
        orthographic_half_extent=ORTHO_HALF_EXTENT,
    )


def check_coords():
    for r in (128, 256, 512, 1024):
        pix = np.asarray([[0.0, 0.0], [r - 1.0, r - 1.0], [r / 2.0 - 0.5, r / 2.0 - 0.5]], np.float32)
        back = grid_to_pixel_center(pixel_center_to_grid(pix, r), r)
        if not np.allclose(pix, back, atol=1e-5):
            raise AssertionError((r, pix, back))
        for i in (0, r // 2, r - 1):
            g = cell_index_to_grid(i, r)
            if int(grid_to_cell_index(g, r)) != i:
                raise AssertionError((r, i, g, grid_to_cell_index(g, r)))


def check_p_formulation():
    yaw = torch.arange(8, dtype=torch.float32)[None] * 45.0
    right, up, forward = camera_basis_from_yaw(yaw, batch=1, views=8, dtype=torch.float32)
    if not torch.allclose(right[0, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6):
        raise AssertionError(right[0, 0])
    if not torch.allclose(forward[0, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6):
        raise AssertionError(forward[0, 0])
    s = 2.0 ** -0.5
    if not torch.allclose(right[0, 1], torch.tensor([s, -s, 0.0]), atol=1e-6):
        raise AssertionError(right[0, 1])
    if not torch.allclose(forward[0, 1], torch.tensor([s, s, 0.0]), atol=1e-6):
        raise AssertionError(forward[0, 1])
    if not torch.allclose(up, torch.tensor([0.0, 0.0, 1.0]).reshape(1, 1, 3).expand_as(up), atol=1e-6):
        raise AssertionError("up basis drift")

    report = {}
    for input_r in (256, 512, 1024):
        h = input_r // 2
        depth = torch.zeros(1, 8, 1, h, h)
        p = reconstruct_p_from_view_depth(depth, yaw, ORTHO_HALF_EXTENT)
        grid = field_cell_centers(h, h)
        gx = grid[..., 0][None, None]
        gy = grid[..., 1][None, None]
        rproj = (p * right[..., :, None, None]).sum(2)
        uproj = (p * up[..., :, None, None]).sum(2)
        fproj = (p * forward[..., :, None, None]).sum(2)
        er = float((rproj - ORTHO_HALF_EXTENT * gx).abs().max())
        eu = float((uproj + ORTHO_HALF_EXTENT * gy).abs().max())
        ef = float(fproj.abs().max())
        if max(er, eu, ef) > 2e-6:
            raise AssertionError((input_r, er, eu, ef))
        # Bilinear sampling must preserve the analytic screen plane at arbitrary interior queries.
        q = torch.linspace(-0.8, 0.8, 17)
        qy, qx = torch.meshgrid(q, q, indexing="ij")
        query = torch.stack([qx.reshape(-1), qy.reshape(-1)], dim=-1)
        query = query[None, None].expand(1, 8, -1, -1).contiguous()
        ps = sample_field(p, query)
        rs = (ps * right[:, :, None, :]).sum(-1)
        us = (ps * up[:, :, None, :]).sum(-1)
        es = max(
            float((rs - ORTHO_HALF_EXTENT * query[..., 0]).abs().max()),
            float((us + ORTHO_HALF_EXTENT * query[..., 1]).abs().max()),
        )
        if es > 3e-6:
            raise AssertionError((input_r, "sampled_screen_plane", es))
        report[input_r] = {"field_hw": h, "screen_plane_max_abs_error": max(er, eu, es)}
    return report


def check_model_resolutions():
    model = IRISSinglePoseV2(tiny_cfg()).eval()
    yaw = torch.arange(8, dtype=torch.float32)[None] * 45.0
    results = {}
    with torch.no_grad():
        for r in (256, 512, 1024):
            out = model(torch.rand(1, 8, 4, r, r), yaw)
            expected = {
                "P": (1, 8, 3, r // 2, r // 2),
                "N": (1, 8, 3, r // 2, r // 2),
                "U_geo": (1, 8, 1, r // 2, r // 2),
                "Z_coarse": (1, 8, tiny_cfg().coarse_dim, r // 8, r // 8),
                "Z_fine": (1, 8, tiny_cfg().fine_dim, r // 2, r // 2),
            }
            for k, shape in expected.items():
                if tuple(out[k].shape) != shape:
                    raise AssertionError((r, k, tuple(out[k].shape), shape))
            results[r] = {k: list(v.shape) for k, v in out.items()}
    return results


def make_batch(tracks=6, geom=8):
    b, v = 1, 8
    yaw = torch.arange(v, dtype=torch.float32)[None] * 45.0
    geom_xy = torch.rand(b, v, geom, 2) * 1.6 - 0.8
    right, up, forward = camera_basis_from_yaw(yaw, batch=b, views=v, dtype=torch.float32)
    depth = torch.rand(b, v, geom) * 0.6 - 0.3
    geom_p = (
        ORTHO_HALF_EXTENT * geom_xy[..., 0, None] * right[:, :, None, :]
        - ORTHO_HALF_EXTENT * geom_xy[..., 1, None] * up[:, :, None, :]
        + depth[..., None] * forward[:, :, None, :]
    )
    geom_n = torch.nn.functional.normalize(torch.rand(b, v, geom, 3) - 0.5, dim=-1)
    geom_mask = torch.ones(b, v, geom, dtype=torch.bool)
    base = torch.rand(b, tracks, 2) * 1.2 - 0.6
    track_xy = base[:, :, None, :].repeat(1, 1, v, 1)
    for view in range(v):
        track_xy[:, :, view, 0] = (base[:, :, 0] + (view - 3.5) * 0.01).clamp(-0.8, 0.8)
    return dict(
        yaw_deg=yaw,
        geom_xy=geom_xy,
        geom_p=geom_p,
        geom_n=geom_n,
        geom_mask=geom_mask,
        track_xy=track_xy,
        track_visible=torch.ones(b, tracks, v, dtype=torch.bool),
        track_p=torch.rand(b, tracks, 3) - 0.5,
    )


def check_losses():
    model = IRISSinglePoseV2(tiny_cfg())
    r = 64
    x = torch.rand(1, 8, 4, r, r)
    yaw = torch.arange(8, dtype=torch.float32)[None] * 45.0
    batch = make_batch()
    parts = total_loss(model(x, yaw), batch, epoch=4, warmup_epochs=4)
    for k, value in parts.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise AssertionError((k, value))
    parts["total"].backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError("no finite gradients")
    dg = model.p_depth_head.weight.grad
    if dg is None or not torch.isfinite(dg).all() or float(dg.abs().sum()) <= 0.0:
        raise AssertionError("P depth head did not receive finite nonzero gradient")
    return {k: float(v.detach()) for k, v in parts.items() if torch.is_tensor(v) and v.numel() == 1}


def check_set_valued_cycle():
    # A->B->A deliberately swaps indices 0 and 1. They are one legal same-locus
    # equivalence set, so a set-valued cycle objective must accept the return.
    swap = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    identity = torch.eye(3)
    pos = torch.tensor([[True, True, False], [True, True, False], [False, False, True]])
    loss = _positive_set_cycle_loss(swap, identity, pos, pos)
    if float(loss) > 1e-6:
        raise AssertionError(f"legal set-valued cycle was penalized: {float(loss)}")
    return float(loss)


def check_matcher_role_separation():
    r = 64
    hc, hf = r // 8, r // 2
    zc = torch.zeros(1, 8, 4, hc, hc)
    zf = torch.zeros(1, 8, 4, hf, hf)
    p = torch.zeros(1, 8, 3, hf, hf)
    u = torch.zeros(1, 8, 1, hf, hf)
    images = torch.ones(1, 8, 4, r, r)
    images[:, :, 3] = 1.0
    q = np.asarray([0.0, 0.0], np.float32)
    zc[:, 0, 0] = 1.0
    zf[:, 0, 0] = 1.0
    zc[:, 1, 1] = 1.0
    zc[0, 1, 1, 4, 4] = 0.0
    zc[0, 1, 0, 4, 4] = 1.0
    zf[:, 1, 1] = 1.0
    for y in range(14, 19):
        for x in range(14, 19):
            zf[0, 1, 1, y, x] = 0.0
            zf[0, 1, 0, y, x] = 0.8
    zf[0, 1, :, 2, 2] = 0.0
    zf[0, 1, 0, 2, 2] = 1.0
    outputs = {
        "Z_coarse": torch.nn.functional.normalize(zc + 1e-6, dim=2),
        "Z_fine": torch.nn.functional.normalize(zf + 1e-6, dim=2),
        "P": p,
        "U_geo": u,
        "N": torch.zeros_like(p),
    }
    cfg = MatcherConfig(
        coarse_keep=1,
        p_rescue_keep=0,
        final_topk=4,
        coarse_row_half_width_cells=8,
        fine_radius_cells=2,
        fine_per_basin=2,
    )
    res = match_query(outputs, images, 0, 1, q, cfg)
    if not res.basins or res.basins[0].coarse_index != (4 * hc + 4):
        raise AssertionError([b.coarse_index for b in res.basins[:3]])
    far = np.asarray([float(cell_index_to_grid(2, hf)), float(cell_index_to_grid(2, hf))], np.float32)
    if len(res.top_coords) and np.min(np.linalg.norm(res.top_coords - far[None], axis=1)) < 1e-6:
        raise AssertionError("global Z_fine leaked into candidate admission")
    return {"basins": len(res.basins), "candidates": len(res.candidates)}


def check_metric_units():
    truth = pixel_center_to_grid(np.asarray([511.5, 511.5], np.float32), 1024)
    pred = pixel_center_to_grid(np.asarray([515.5, 511.5], np.float32), 1024)
    e = float(native_pixel_error(pred, truth, 1024))
    if abs(e - 4.0) > 1e-4:
        raise AssertionError(e)
    coarse = 128
    cx = int(grid_to_cell_index(truth[0], coarse))
    cy = int(grid_to_cell_index(truth[1], coarse))
    cc = np.asarray([[float(cell_index_to_grid(cx, coarse)), float(cell_index_to_grid(cy, coarse))]], np.float32)
    if not exact_coarse_cell_hit(cc, truth, coarse, 1):
        raise AssertionError("exact coarse-cell metric failed")


def main():
    report = {}
    check_coords()
    report["coordinate_roundtrip"] = "PASS"
    report["P_camera_conditioned_formulation"] = check_p_formulation()
    report["model_resolution_shapes"] = check_model_resolutions()
    report["loss_forward_backward"] = check_losses()
    report["set_valued_cycle"] = {"status": "PASS", "loss": check_set_valued_cycle()}
    report["matcher_role_separation"] = check_matcher_role_separation()
    check_metric_units()
    report["native_pixel_metric_units"] = "PASS"
    report["status"] = "PASS"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
