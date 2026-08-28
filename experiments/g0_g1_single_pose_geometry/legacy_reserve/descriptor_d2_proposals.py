from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from descriptor_d2_spatial import sample_dense_native

IMAGE_NATIVE = 256


def foreground_mask(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB"), np.uint8)
    return ~((a[..., 0] == 0) & (a[..., 1] == 255) & (a[..., 2] == 0))


def foreground_candidates(mask: np.ndarray, stride: int = 4) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return np.empty((0, 2), np.float32)
    keep = (xx % stride == 0) & (yy % stride == 0)
    q = np.stack([xx[keep], yy[keep]], axis=1).astype(np.float32)
    if len(q) >= 8:
        return q
    return np.stack([xx, yy], axis=1).astype(np.float32)


def _sample_single(field: torch.Tensor, xy: np.ndarray, image_size: int = IMAGE_NATIVE) -> torch.Tensor:
    if field.ndim != 3:
        raise ValueError(tuple(field.shape))
    if len(xy) == 0:
        return torch.empty((0, field.shape[0]), device=field.device, dtype=field.dtype)
    q = torch.as_tensor(xy, device=field.device, dtype=torch.float32).view(1, 1, -1, 2)
    return sample_dense_native(field.view(1, 1, *field.shape), q, image_size)[0, 0]


def _consensus(field: torch.Tensor, xy: torch.Tensor, vis: torch.Tensor, image_size: int = IMAGE_NATIVE) -> torch.Tensor:
    # field [1,V,C,H,W], xy [1,V,N,2], vis [1,V,N] -> [N,C]
    z = F.normalize(sample_dense_native(field, xy, image_size).float(), dim=-1, eps=1e-6)[0]
    w = vis[0].float().unsqueeze(-1)
    c = (z * w).sum(dim=0) / w.sum(dim=0).clamp_min(1.0)
    return F.normalize(c, dim=-1, eps=1e-6)


def _refined_candidates(seeds: np.ndarray, mask: np.ndarray, radius: int = 4, step: int = 2) -> np.ndarray:
    pts = set()
    H, W = mask.shape
    for x, y in seeds:
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                xx = int(round(float(x + dx)))
                yy = int(round(float(y + dy)))
                if 0 <= xx < W and 0 <= yy < H and bool(mask[yy, xx]):
                    pts.add((xx, yy))
    if not pts:
        pts = {tuple(map(int, q)) for q in seeds}
    return np.asarray(sorted(pts, key=lambda q: (q[1], q[0])), np.float32)


def _summary_from_raw(raw: Dict[str, List[float]]) -> Dict[str, float]:
    d1 = np.asarray(raw["top1_dist"], np.float64)
    d4 = np.asarray(raw["top4_nearest_dist"], np.float64)
    ds = np.asarray(raw["support_nearest_dist"], np.float64)
    regret = np.asarray(raw["ranking_regret"], np.float64)
    oracle = np.asarray(raw["oracle_hit"], np.float64)
    oracle4 = np.asarray(raw["oracle_top4_recall"], np.float64)
    margin = np.asarray(raw["score_margin"], np.float64)
    if len(d1) == 0:
        return {"query_n": 0}
    out = {
        "query_n": int(len(d1)),
        "proposal_top1_mean_dist_px": float(d1.mean()),
        "proposal_top1_median_dist_px": float(np.median(d1)),
        "proposal_top1_p90_dist_px": float(np.quantile(d1, 0.90)),
        "proposal_top4_nearest_mean_dist_px": float(d4.mean()),
        "proposal_support_nearest_mean_dist_px": float(ds.mean()),
        "proposal_ranking_regret_mean_px": float(regret.mean()),
        "proposal_oracle_hit_rate": float(oracle.mean()),
        "proposal_oracle_top4_recall": float(oracle4.mean()),
        "proposal_score_top1_top2_margin_mean": float(margin.mean()) if len(margin) else float("nan"),
    }
    for thr, tag in [(2.0, "2"), (4.0, "4"), (6.4, "6_4"), (12.8, "12_8")]:
        out[f"proposal_top1_pck{tag}"] = float((d1 <= thr).mean())
        out[f"proposal_top4_pck{tag}"] = float((d4 <= thr).mean())
        out[f"proposal_support_pck{tag}"] = float((ds <= thr).mean())
    return out


@torch.inference_mode()
def proposal_diagnostics_episode(
    outputs: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    b_masks: List[np.ndarray],
    mode: str,
    coarse_stride: int = 4,
    seed_count: int = 8,
    refine_radius: int = 4,
    refine_step: int = 2,
    topk: int = 4,
    image_size: int = IMAGE_NATIVE,
) -> Dict[str, object]:
    if mode not in {"coarse", "fine"}:
        raise ValueError(mode)
    coarse = outputs["descriptor"]
    fine = outputs.get("descriptor_fine")
    if mode == "fine" and fine is None:
        raise ValueError("fine mode requires descriptor_fine")
    if coarse.shape[0] != 1 or len(b_masks) != coarse.shape[2]:
        raise ValueError((tuple(coarse.shape), len(b_masks)))
    xyA = target["XY_A"].to(coarse.device)
    xyB = target["XY_B"].to(coarse.device)
    vA = target["V_A"].bool().to(coarse.device)
    vB = target["V_B"].bool().to(coarse.device)
    zA_coarse = _consensus(coarse[:, 0].float(), xyA, vA, image_size)
    zA_fine = _consensus(fine[:, 0].float(), xyA, vA, image_size) if fine is not None else None

    raw = {k: [] for k in ["top1_dist", "top4_nearest_dist", "support_nearest_dist", "ranking_regret", "oracle_hit", "oracle_top4_recall", "score_margin"]}
    V = coarse.shape[2]
    for v in range(V):
        mask = b_masks[v]
        coarse_xy = foreground_candidates(mask, coarse_stride)
        if len(coarse_xy) == 0:
            continue
        zc = F.normalize(_sample_single(coarse[0, 1, v].float(), coarse_xy, image_size), dim=-1, eps=1e-6)
        ids = (vA[0, v] & vB[0, v]).nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        scores = zA_coarse[ids] @ zc.T
        # stable sort mirrors frozen proposal semantics.
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        seed_n = min(seed_count, order.shape[1])
        seed_ids = order[:, :seed_n].detach().cpu().numpy()
        # Build each carrier's exact deduplicated local set in Python, but sample the
        # descriptor field only ONCE per view. This preserves frozen proposal semantics
        # while avoiding hundreds of tiny CUDA grid_sample launches per episode.
        carriers=[]; refined_list=[]; spans=[]; cursor=0
        for rr, carrier_t in enumerate(ids):
            i = int(carrier_t.item())
            seeds = coarse_xy[seed_ids[rr]]
            refined = _refined_candidates(seeds, mask, refine_radius, refine_step)
            if len(refined) == 0:
                continue
            carriers.append(i); refined_list.append(refined); spans.append((cursor,cursor+len(refined))); cursor += len(refined)
        if not refined_list:
            continue
        all_refined=np.concatenate(refined_list,axis=0)
        rank_field=coarse[0,1,v].float() if mode=="coarse" else fine[0,1,v].float()
        all_z=F.normalize(_sample_single(rank_field,all_refined,image_size),dim=-1,eps=1e-6)
        for i,refined,(lo,hi) in zip(carriers,refined_list,spans):
            zq=zA_coarse[i] if mode=="coarse" else zA_fine[i]
            sr=all_z[lo:hi]@zq
            ord2=torch.argsort(sr,descending=True,stable=True)
            k=min(topk,len(refined)); top_idx=ord2[:k].detach().cpu().numpy()
            truth=xyB[0,v,i].float().detach().cpu().numpy(); dist=np.linalg.norm(refined-truth[None,:],axis=1)
            support_min=float(dist.min()); top1=float(dist[int(top_idx[0])]); top4=float(dist[top_idx].min())
            raw["top1_dist"].append(top1); raw["top4_nearest_dist"].append(top4); raw["support_nearest_dist"].append(support_min)
            raw["ranking_regret"].append(max(0.0,top1-support_min)); raw["oracle_hit"].append(float(top1<=support_min+1e-6)); raw["oracle_top4_recall"].append(float(top4<=support_min+1e-6))
            if len(sr)>=2:
                ss=torch.sort(sr.float(),descending=True).values; raw["score_margin"].append(float((ss[0]-ss[1]).item()))
    return {"summary": _summary_from_raw(raw), "raw": raw}


def merge_raw(parts: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    out = {k: [] for k in ["top1_dist", "top4_nearest_dist", "support_nearest_dist", "ranking_regret", "oracle_hit", "oracle_top4_recall", "score_margin"]}
    for p in parts:
        for k in out:
            out[k].extend(p.get(k, []))
    return out


def summarize_raw(raw: Dict[str, List[float]]) -> Dict[str, float]:
    return _summary_from_raw(raw)
