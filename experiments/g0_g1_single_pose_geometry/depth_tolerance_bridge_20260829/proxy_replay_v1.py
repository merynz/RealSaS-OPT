#!/usr/bin/env python3
"""Exact-semantics frozen E0 D2 proxy replay adapter for the structured depth bridge.

Scientific role:
- reconstruct the historical D2 reciprocal MUTUAL_P003 admission on corrupted
  observation-space geometry;
- expose the exact frozen Geppetto/Arachne information-isolation proxy models
  and evaluation semantics needed by the bridge;
- never train, retune, or reinterpret the historical proxy checkpoints.

Source authority:
The model/evaluator semantics in this module were restored from the embedded
byte-authority inside `RealSaS_E0_DOWNSTREAM_PROXY_V1_2.ipynb`.
The embedded historical sources were SHA-256 verified as:
  e0_downstream_proxy_v1.py
    0f599e8f717d4c5070c04e224d90e52d1dc6e76a6e2068e64ed7c9d9d2b950b4
  run_e0_downstream_proxy_calibration_v1.py
    7fc3021726df64069ba44be42f946db25929372497f39f392898477b374e2ca4
This adapter intentionally copies only the exact inference/evaluation semantics
needed for no-training replay.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from bridge_persistence_v1 import (
    _corrupt_observable_views,
    build_corrupted_persistence_carrier,
)
from depth_corruption_v1 import DepthCorruptionSpec
from e0_geometry import camera_for_view
from surface_builder_e0_v1 import derived_match_row, load_asset_authority

VIEWS = 8
POINT_FEATURE_DIM = 36
ARACHNE_PAIR_FEATURE_DIM = 57
MAX_GEPPETTO_QUERIES = 48
MUTUAL_P003 = 0.003

EXPECTED_GEPPETTO_D2_SHA256 = "f8c6146fc3ad81146ced01805b9be454ad194b86db3a9ab3d72d7b9eb3747b65"
EXPECTED_ARACHNE_D2_SHA256 = "72898a62f23c55aa82047f7bc4b39be787abb97d14f9a3fb973d59b2b5689745"
EXPECTED_CONTRACT_SHA256 = "cf204ad1ef7d8460e402fb6c3db7361d122b3284116aa6e840af032d0fb39a2a"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed64(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:16], 16) & 0x7fffffffffffffff


def deterministic_subsample(n: int, k: int, label: str) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(_seed64(label))
    return np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)


def arachne_pair_features(point_X, point_P, heads, tails):
    import torch
    point_X = torch.as_tensor(point_X)
    point_P = torch.as_tensor(point_P)
    heads = torch.as_tensor(heads)
    tails = torch.as_tensor(tails)
    N, J = point_P.shape[0], heads.shape[0]
    H = heads[None].expand(N, J, 3)
    T = tails[None].expand(N, J, 3)
    P = point_P[:, None, :].expand(N, J, 3)
    M = 0.5 * (H + T)
    RH = H - P
    RT = T - P
    RM = M - P
    DH = torch.linalg.norm(RH, dim=-1, keepdim=True)
    DT = torch.linalg.norm(RT, dim=-1, keepdim=True)
    DM = torch.linalg.norm(RM, dim=-1, keepdim=True)
    axis = T - H
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
    skel = torch.cat([H, T, RH, RT, RM, DH, DT, DM, axis], dim=-1)
    PX = point_X[:, None, :].expand(N, J, POINT_FEATURE_DIM)
    out = torch.cat([PX, skel], dim=-1)
    if out.shape[-1] != ARACHNE_PAIR_FEATURE_DIM:
        raise AssertionError(out.shape)
    return out


def make_arachne_model():
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(ARACHNE_PAIR_FEATURE_DIM, 96),
        nn.GELU(),
        nn.Linear(96, 96),
        nn.GELU(),
        nn.Linear(96, 1),
    )


def make_geppetto_model():
    import torch
    import torch.nn as nn

    class GeppettoIsolation(nn.Module):
        def __init__(self):
            super().__init__()
            self.point = nn.Sequential(
                nn.Linear(POINT_FEATURE_DIM, 48),
                nn.GELU(),
                nn.Linear(48, 48),
            )
            self.queries = nn.Parameter(torch.randn(MAX_GEPPETTO_QUERIES, 48) * 0.02)
            self.attn = nn.MultiheadAttention(48, 4, batch_first=True)
            self.norm1 = nn.LayerNorm(48)
            self.ff = nn.Sequential(nn.Linear(48, 96), nn.GELU(), nn.Linear(96, 48))
            self.norm2 = nn.LayerNorm(48)
            self.xyz = nn.Linear(48, 3)

        def forward(self, x):
            kv = self.point(x)
            q = self.queries[None].expand(x.shape[0], -1, -1)
            a, _ = self.attn(q, kv, kv, need_weights=False)
            q = self.norm1(q + a)
            q = self.norm2(q + self.ff(q))
            return self.xyz(q)

    return GeppettoIsolation()


def hungarian_pairs(pred, target):
    import torch
    from scipy.optimize import linear_sum_assignment
    with torch.no_grad():
        cost = torch.cdist(pred.detach().float(), target.detach().float()).cpu().numpy()
        rr, cc = linear_sum_assignment(cost)
    return (
        torch.as_tensor(rr, dtype=torch.long, device=pred.device),
        torch.as_tensor(cc, dtype=torch.long, device=pred.device),
    )


def _influence_disp_error(pred: np.ndarray, target: np.ndarray, aid: str) -> float:
    J = pred.shape[1]
    rng = np.random.default_rng(int(hashlib.sha256(f"E0-INFLUENCE|{aid}".encode()).hexdigest()[:16], 16))
    delta = rng.normal(size=(J, 3)).astype(np.float32)
    delta /= np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
    delta *= 0.02
    dp = (pred - target) @ delta
    return float(np.linalg.norm(dp, axis=1).mean())


def load_d2_models(geppetto_path: str | Path, arachne_path: str | Path, device: str = "cpu"):
    import torch
    gp = Path(geppetto_path)
    ap = Path(arachne_path)
    if sha256_file(gp) != EXPECTED_GEPPETTO_D2_SHA256:
        raise RuntimeError("GEPPETTO_D2_BEST checkpoint SHA drift")
    if sha256_file(ap) != EXPECTED_ARACHNE_D2_SHA256:
        raise RuntimeError("ARACHNE_D2_BEST checkpoint SHA drift")
    gck = torch.load(gp, map_location="cpu", weights_only=False)
    ack = torch.load(ap, map_location="cpu", weights_only=False)
    for ck, arm, schema in (
        (gck, "D2", "RealSaS.E0.GeppettoIsolationProxy.v1"),
        (ack, "D2", "RealSaS.E0.ArachneProxy.v1"),
    ):
        if ck.get("arm") != arm or ck.get("schema") != schema:
            raise RuntimeError("checkpoint identity drift")
        if ck.get("contract_sha256") != EXPECTED_CONTRACT_SHA256:
            raise RuntimeError("checkpoint contract drift")
    gm = make_geppetto_model().to(device)
    am = make_arachne_model().to(device)
    gm.load_state_dict(gck["model"], strict=True)
    am.load_state_dict(ack["model"], strict=True)
    gm.eval(); am.eval()
    return gm, am


def d2_proxy_x36_from_bridge(
    asset_dir: str | Path,
    spec: DepthCorruptionSpec,
):
    """Return historical-D2-compatible X36 after corrupted forward + reciprocal cycle admission."""
    asset_dir = Path(asset_dir)
    carrier = build_corrupted_persistence_carrier(asset_dir, spec)
    _geom, authority = load_asset_authority(asset_dir)
    clean_obs = [x.observable for x in authority]
    obs, _reports = _corrupt_observable_views(asset_dir.name, clean_obs, spec)

    P = np.asarray(carrier.P, np.float32)
    sv = np.asarray(carrier.source_view, np.int64)
    rows = np.asarray(carrier.matched_row, np.int64)
    base = np.asarray(carrier.support, bool)
    support = np.zeros_like(base)
    support[np.arange(len(P)), sv] = True

    for i in range(len(P)):
        s = int(sv[i])
        for tv in range(VIEWS):
            if tv == s or not base[i, tv]:
                continue
            br = int(rows[i, tv])
            if br < 0:
                continue
            rr, _ = derived_match_row(
                obs[tv].P[br],
                obs[tv].N_derived[br],
                obs[tv].grid[br],
                tv,
                obs[tv].half_extent,
                obs[s],
            )
            if rr < 0:
                continue
            cycle_p = float(np.linalg.norm(obs[s].P[rr] - P[i]))
            if cycle_p <= MUTUAL_P003:
                support[i, tv] = True

    admitted_rows = rows.copy()
    admitted_rows[~support] = -1
    grid = np.zeros((len(P), VIEWS, 2), np.float32)
    for v in range(VIEWS):
        idx = np.flatnonzero(support[:, v] & (admitted_rows[:, v] >= 0))
        if len(idx):
            grid[idx, v] = obs[v].grid[admitted_rows[idx, v]]
    depth = np.stack(
        [
            P @ np.asarray(camera_for_view(v, obs[v].half_extent)["forward"], np.float32)
            for v in range(VIEWS)
        ],
        axis=1,
    ).astype(np.float32)
    X = np.concatenate(
        [
            P,
            support.astype(np.float32),
            grid.reshape(len(P), 16),
            (support.sum(axis=1, keepdims=True) / float(VIEWS)).astype(np.float32),
            depth,
        ],
        axis=1,
    ).astype(np.float32)
    if X.shape != (len(P), POINT_FEATURE_DIM):
        raise AssertionError(X.shape)
    return carrier, support, admitted_rows, X


def eval_geppetto_d2(model, X: np.ndarray, target_pack: dict, asset_id: str, device: str = "cpu") -> dict:
    import torch
    model.eval()
    X = np.asarray(X, np.float32)
    idx = deterministic_subsample(len(X), 256, f"GEPPETTO|{asset_id}|EVAL")
    xx = torch.from_numpy(X[idx]).to(device)[None]
    target = torch.from_numpy(np.asarray(target_pack["probe_joint_heads"], np.float32)).to(device)
    with torch.no_grad():
        pred = model(xx)[0][:len(target)]
        rr, cc = hungarian_pairs(pred, target)
        d = torch.linalg.norm(pred[rr] - target[cc], dim=1).cpu().numpy()
    return {
        "asset_id": asset_id,
        "joint_mean": float(d.mean()),
        "joint_p95": float(np.quantile(d, .95)),
        "pck_005": float(np.mean(d <= .05)),
        "pck_008": float(np.mean(d <= .08)),
        "probe_joint_count": int(len(target)),
        "original_legal_control_count": int(np.asarray(target_pack["original_legal_control_count"]).item()),
    }


def eval_arachne_d2(model, X: np.ndarray, target_pack: dict, asset_id: str, device: str = "cpu") -> dict:
    import torch
    import torch.nn.functional as F
    model.eval()
    X = np.asarray(X, np.float32)
    W0 = np.asarray(target_pack["D2_skin"], np.float32)
    valid = np.asarray(target_pack["D2_skin_valid"]).astype(bool)
    X = X[valid]
    W = W0[valid]
    heads = np.asarray(target_pack["legal_bone_heads"], np.float32)
    tails = np.asarray(target_pack["legal_bone_tails"], np.float32)
    if len(X) == 0:
        raise RuntimeError(f"{asset_id}: no valid D2 Arachne rows")
    pred_chunks = []
    ce_sum = 0.0
    n = 0
    with torch.no_grad():
        for s in range(0, len(X), 128):
            xx = torch.from_numpy(X[s:s+128]).to(device)
            ww = torch.from_numpy(W[s:s+128]).to(device)
            hh = torch.from_numpy(heads).to(device)
            tt = torch.from_numpy(tails).to(device)
            pair = arachne_pair_features(xx, xx[:, :3], hh, tt)
            logits = model(pair).squeeze(-1)
            logp = F.log_softmax(logits, dim=1)
            ce = -(ww * logp).sum(1)
            ce_sum += float(ce.sum().cpu())
            n += len(xx)
            pred_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    pred = np.concatenate(pred_chunks, axis=0)
    mae = float(np.abs(pred - W).mean())
    top1 = float(np.mean(np.argmax(pred, axis=1) == np.argmax(W, axis=1)))
    k = min(4, pred.shape[1])
    idx = np.argpartition(-pred, k-1, axis=1)[:, :k]
    top4_mass = float(np.take_along_axis(W, idx, axis=1).sum(1).mean())
    infl = _influence_disp_error(pred, W, asset_id)
    return {
        "asset_id": asset_id,
        "ce": ce_sum / n,
        "weight_mae": mae,
        "top1": top1,
        "gt_mass_top4": top4_mass,
        "influence_disp_mean": infl,
        "valid_points": int(n),
        "controls": int(W.shape[1]),
    }


def aggregate_geppetto(per: list[dict]) -> dict:
    return {
        "joint_mean": float(np.mean([r["joint_mean"] for r in per])),
        "family_p95": float(np.quantile([r["joint_mean"] for r in per], .95)),
        "joint_p95_mean": float(np.mean([r["joint_p95"] for r in per])),
        "pck_005": float(np.mean([r["pck_005"] for r in per])),
        "pck_008": float(np.mean([r["pck_008"] for r in per])),
    }


def aggregate_arachne(per: list[dict]) -> dict:
    keys = ["ce", "weight_mae", "top1", "gt_mass_top4", "influence_disp_mean"]
    return {k: float(np.mean([r[k] for r in per])) for k in keys}
