from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


def sample_field(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """field [B,V,C,H,W], grid [B,V,K,2] -> [B,V,K,C], align_corners=False."""
    b, v, c, h, w = field.shape
    if grid.shape[:2] != (b, v) or grid.shape[-1] != 2:
        raise ValueError((tuple(field.shape), tuple(grid.shape)))
    k = grid.shape[2]
    out = F.grid_sample(
        field.reshape(b * v, c, h, w),
        grid.reshape(b * v, k, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return out[..., 0].permute(0, 2, 1).reshape(b, v, k, c)


def smooth_l1_vec(pred, target, beta=0.01):
    return F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(-1)


def geometry_loss(outputs, batch):
    grid = batch["geom_xy"].float()
    gt_p = batch["geom_p"].float()
    gt_n = batch["geom_n"].float()
    mask = batch["geom_mask"].float()
    # AMP boundary: model activations may be FP16, but all supervision/loss numerics are FP32.
    pr_p = sample_field(outputs["P"].float(), grid)
    pr_n = sample_field(outputs["N"].float(), grid)
    p_elem = smooth_l1_vec(pr_p, gt_p, 0.01)
    n_elem = 1.0 - (pr_n * gt_n).sum(-1).clamp(-1.0, 1.0)
    den = mask.sum().clamp_min(1.0)
    lp = (p_elem * mask).sum() / den
    ln = (n_elem * mask).sum() / den
    s = sample_field(outputs["U_geo"].float(), grid)[..., 0]
    euclid_detached = torch.linalg.norm(pr_p.detach() - gt_p, dim=-1)
    lu = ((torch.exp(-s) * euclid_detached + s) * mask).sum() / den
    return {
        "P": lp,
        "N": ln,
        "U_geo": lu,
        "p_euclid": (euclid_detached * mask).sum().detach() / den,
        "n_cos": (n_elem * mask).sum().detach() / den,
    }


def _masked_multi_positive_ce(logits, pos, valid):
    logits = logits.masked_fill(~valid, -1e4)
    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return logits.sum() * 0.0
    den = torch.logsumexp(logits[has_pos], dim=1)
    num = torch.logsumexp(logits[has_pos].masked_fill(~pos[has_pos], -1e4), dim=1)
    return (den - num).mean()


def _positive_set_cycle_loss(pab, pba, source_positive_set, target_positive_set):
    """Reward return to any legal same-locus positive, never one privileged diagonal ID."""
    cyc_source = pab @ pba
    cyc_target = pba @ pab
    src_mass = (cyc_source * source_positive_set.to(cyc_source.dtype)).sum(dim=1)
    tgt_mass = (cyc_target * target_positive_set.to(cyc_target.dtype)).sum(dim=1)
    src_valid = source_positive_set.any(dim=1)
    tgt_valid = target_positive_set.any(dim=1)
    terms = []
    if src_valid.any():
        terms.append(-torch.log(src_mass[src_valid].clamp_min(1e-12)).mean())
    if tgt_valid.any():
        terms.append(-torch.log(tgt_mass[tgt_valid].clamp_min(1e-12)).mean())
    if not terms:
        return pab.sum() * 0.0
    return torch.stack(terms).mean()


def pairwise_coarse_objective(zs, zt, p, temperature=0.08, same_radius=0.003, far_radius=0.025, margin=0.10):
    # Similarity / softmax / margin numerics stay FP32 under autocast.
    zs = F.normalize(zs.float(), dim=-1, eps=1e-8)
    zt = F.normalize(zt.float(), dim=-1, eps=1e-8)
    logits = (zs @ zt.T) / temperature
    dist = torch.cdist(p.float(), p.float())
    pos = dist <= same_radius
    valid = (dist >= far_radius) | pos
    dual = 0.5 * (
        _masked_multi_positive_ce(logits, pos, valid)
        + _masked_multi_positive_ce(logits.T, pos.T, valid.T)
    )
    sim = zs @ zt.T
    neg_mask = dist >= far_radius
    pos_sim = sim.masked_fill(~pos, -1e9).max(dim=1).values
    neg_sim = sim.masked_fill(~neg_mask, -1e9).max(dim=1).values
    vm = (pos_sim > -1e8) & (neg_sim > -1e8)
    hard = F.relu(margin + neg_sim[vm] - pos_sim[vm]).mean() if vm.any() else sim.sum() * 0.0
    pab = F.softmax(logits.masked_fill(~valid, -1e4), dim=1)
    pba = F.softmax(logits.T.masked_fill(~valid.T, -1e4), dim=1)
    reciprocal = _positive_set_cycle_loss(pab, pba, pos, pos.T)
    return dual, hard, reciprocal


def _uniform_observation_subsample(idx: torch.Tensor, max_obs: int) -> torch.Tensor:
    """Deterministically thin a view-major observation list without prefix/view bias."""
    if idx.shape[0] <= max_obs:
        return idx
    take = torch.linspace(0, idx.shape[0] - 1, steps=max_obs, device=idx.device)
    take = take.round().to(torch.long)
    return idx[take]


def observation_supcon(z, p, vis, temperature=0.08, same_radius=0.003, far_radius=0.025, max_obs=1024):
    idx = vis.T.nonzero(as_tuple=False)
    if idx.shape[0] < 3:
        return z.sum() * 0.0
    idx = _uniform_observation_subsample(idx, max_obs)
    f = F.normalize(z[idx[:, 0], idx[:, 1]].float(), dim=-1, eps=1e-8)
    pp = p[idx[:, 1]].float()
    sim = (f @ f.T) / temperature
    dist = torch.cdist(pp, pp)
    eye = torch.eye(len(f), device=f.device, dtype=torch.bool)
    pos = (dist <= same_radius) & ~eye
    valid = ((dist >= far_radius) | pos) & ~eye
    return _masked_multi_positive_ce(sim, pos, valid)


def coarse_correspondence_loss(outputs, batch):
    xy = batch["track_xy"].permute(0, 2, 1, 3).contiguous()
    vis = batch["track_visible"].bool()
    p = batch["track_p"]
    z = sample_field(outputs["Z_coarse"].float(), xy.float())
    duals, hards, recips, globals_ = [], [], [], []
    for b in range(z.shape[0]):
        globals_.append(observation_supcon(z[b], p[b], vis[b]))
        for s in range(z.shape[1]):
            for t in range(s + 1, z.shape[1]):
                m = vis[b, :, s] & vis[b, :, t]
                if int(m.sum()) < 2:
                    continue
                d, h, r = pairwise_coarse_objective(z[b, s, m], z[b, t, m], p[b, m])
                duals.append(d)
                hards.append(h)
                recips.append(r)
    zero = z.sum() * 0.0
    mean = lambda xs: torch.stack(xs).mean() if xs else zero
    return {
        "Zc_multi_positive": mean(globals_),
        "Zc_dual": mean(duals),
        "Zc_hard_margin": mean(hards),
        "Zc_reciprocal_soft": mean(recips),
    }


def _fine_offsets(radius_cells: int, device, dtype):
    vals = torch.arange(-radius_cells, radius_cells + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(vals, vals, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def _uniform_track_subsample(ids: torch.Tensor, max_tracks: int) -> torch.Tensor:
    """Deterministic position-uniform thinning; never take a track-ID prefix."""
    if ids.numel() <= max_tracks:
        return ids
    take = torch.linspace(0, ids.numel() - 1, steps=max_tracks, device=ids.device)
    return ids[take.round().to(torch.long)]


def fine_local_loss(outputs, batch, radius_cells=4, temperature=0.05, max_tracks_per_pair=64):
    # Local correlation / CE numerics stay FP32 under autocast.
    zf = outputs["Z_fine"].float()
    bsz, views, dim, hf, wf = zf.shape
    xy = batch["track_xy"]
    vis = batch["track_visible"].bool()
    offsets = _fine_offsets(radius_cells, zf.device, zf.dtype)
    delta = offsets.clone()
    delta[:, 0] *= 2.0 / float(wf)
    delta[:, 1] *= 2.0 / float(hf)
    zero_idx = ((offsets[:, 0] == 0) & (offsets[:, 1] == 0)).nonzero(as_tuple=False).flatten()
    if zero_idx.numel() != 1:
        raise RuntimeError("fine offset lattice has no unique zero")
    zero_idx = int(zero_idx.item())
    losses, top1s, errs = [], [], []
    for b in range(bsz):
        for s in range(views):
            for t in range(s + 1, views):
                ids = torch.nonzero(vis[b, :, s] & vis[b, :, t], as_tuple=False).flatten()
                if ids.numel() == 0:
                    continue
                ids = _uniform_track_subsample(ids, max_tracks_per_pair)
                src = xy[b, ids, s]
                truth = xy[b, ids, t]
                qs = sample_field(zf[b:b+1, s:s+1], src[None, None])[0, 0]
                cand = truth[:, None, :] + delta[None]
                inb = (cand.abs() <= 1.0).all(-1)
                zcand = sample_field(
                    zf[b:b+1, t:t+1],
                    cand.clamp(-1, 1).reshape(1, 1, -1, 2),
                )[0, 0].reshape(len(ids), len(delta), dim)
                logits = (
                    F.normalize(qs, dim=-1)[:, None, :]
                    * F.normalize(zcand, dim=-1)
                ).sum(-1) / temperature
                logits = logits.masked_fill(~inb, -1e4)
                valid = inb[:, zero_idx]
                if not valid.any():
                    continue
                lv = logits[valid]
                labels = torch.full((int(valid.sum()),), zero_idx, device=zf.device, dtype=torch.long)
                losses.append(F.cross_entropy(lv, labels))
                pred = lv.argmax(-1)
                top1s.append((pred == zero_idx).float().mean())
                errs.append(offsets[pred].float().norm(dim=-1).mean())
    zero = zf.sum() * 0.0
    return {
        "Zf_local": torch.stack(losses).mean() if losses else zero,
        "Zf_local_top1": torch.stack(top1s).mean().detach() if top1s else zero.detach(),
        "Zf_local_error_cells": torch.stack(errs).mean().detach() if errs else zero.detach(),
    }


def track_p_consistency_loss(outputs, batch):
    xy = batch["track_xy"].permute(0, 2, 1, 3).contiguous()
    vis = batch["track_visible"].bool()
    p = sample_field(outputs["P"].float(), xy.float())
    losses = []
    for b in range(p.shape[0]):
        for t in range(p.shape[2]):
            vv = torch.nonzero(vis[b, t], as_tuple=False).flatten()
            if vv.numel() < 2:
                continue
            q = p[b, vv, t]
            center = q.mean(0, keepdim=True)
            losses.append(F.smooth_l1_loss(q, center.expand_as(q), beta=0.01))
    return torch.stack(losses).mean() if losses else p.sum() * 0.0


@dataclass(frozen=True)
class LossWeights:
    P: float = 1.0
    N: float = 0.25
    U_geo: float = 0.05
    Zc_multi_positive: float = 0.10
    Zc_dual: float = 0.10
    Zc_hard_margin: float = 0.05
    Zc_reciprocal_soft: float = 0.02
    P_consistency: float = 0.20
    Zf_local: float = 0.10


def total_loss(outputs, batch, epoch: int, warmup_epochs: int = 4, weights: LossWeights = LossWeights()):
    g = geometry_loss(outputs, batch)
    parts = dict(g)
    total = weights.P * g["P"] + weights.N * g["N"] + weights.U_geo * g["U_geo"]
    if epoch >= warmup_epochs and batch.get("track_visible") is not None:
        c = coarse_correspondence_loss(outputs, batch)
        f = fine_local_loss(outputs, batch)
        pc = track_p_consistency_loss(outputs, batch)
        parts.update(c)
        parts.update(f)
        parts["P_consistency"] = pc
        total = (
            total
            + weights.Zc_multi_positive * c["Zc_multi_positive"]
            + weights.Zc_dual * c["Zc_dual"]
            + weights.Zc_hard_margin * c["Zc_hard_margin"]
            + weights.Zc_reciprocal_soft * c["Zc_reciprocal_soft"]
            + weights.P_consistency * pc
            + weights.Zf_local * f["Zf_local"]
        )
    parts["total"] = total
    return parts
