from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from surface_builder_e0_v1 import load_asset_authority, derived_match_row

CALIBRATION_IDS = (
    "asset_551ea351b43a1787d0f55536",
    "asset_36fb02305846592b1ecdf3d4",
    "asset_0679fdef64f19a4832a6d521",
    "asset_76313e4bd82b82fcd1659c70",
    "asset_6f086a5b1a66378ffe04d7e4",
    "asset_425122d500ecf5767404f9c0",
    "asset_5a19f8c5254be7bf30c504f5",
    "asset_f8a40d6c5d815fe79c8b5e42",
)
VARIANTS = (
    "BASE",
    "MUTUAL_P006",
    "MUTUAL_P003",
    "MUTUAL_P003_SUPPORT2",
    "MUTUAL_P003_SUPPORT3",
)

def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

def mask_sha(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).hexdigest()

def face_components(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, np.int64)
    n = len(faces)
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, np.uint8)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
    vertex_faces: Dict[int, list[int]] = {}
    for fi, tri in enumerate(faces):
        for v in tri:
            vertex_faces.setdefault(int(v), []).append(fi)
    for fs in vertex_faces.values():
        if len(fs) > 1:
            f0 = fs[0]
            for f in fs[1:]:
                union(f0, f)
    remap = {}
    out = np.empty(n, np.int64)
    for i in range(n):
        r = find(i)
        if r not in remap:
            remap[r] = len(remap)
        out[i] = remap[r]
    return out

def _nonself(source_view: np.ndarray, views: int = 8) -> np.ndarray:
    K = len(source_view)
    out = np.ones((K, views), bool)
    out[np.arange(K), np.asarray(source_view, np.int64)] = False
    return out

def observation_only_admission_masks(asset_dir: Path, b: dict) -> Tuple[dict, dict]:
    _, auth = load_asset_authority(asset_dir)
    obs = [x.observable for x in auth]
    P = np.asarray(b["P"], np.float32)
    sv = np.asarray(b["source_view"], np.int64)
    rows = np.asarray(b["matched_row"], np.int64)
    base = np.asarray(b["support_mask"], bool)
    nonself = _nonself(sv)
    mutual006 = np.zeros_like(base)
    mutual003 = np.zeros_like(base)
    for i in range(len(P)):
        s = int(sv[i])
        for tv in range(8):
            if tv == s:
                continue
            br = int(rows[i, tv])
            if br < 0:
                continue
            rr, _ = derived_match_row(obs[tv].P[br], obs[tv].N_derived[br], obs[tv].grid[br], tv, obs[tv].half_extent, obs[s])
            if rr < 0:
                continue
            cycle_p = float(np.linalg.norm(obs[s].P[rr] - P[i]))
            mutual006[i, tv] = cycle_p <= 0.006
            mutual003[i, tv] = cycle_p <= 0.003
    m006 = base & mutual006 & nonself
    m003 = base & mutual003 & nonself
    c003 = m003.sum(axis=1)
    masks = {
        "BASE": base & nonself,
        "MUTUAL_P006": m006,
        "MUTUAL_P003": m003,
        "MUTUAL_P003_SUPPORT2": m003 & (c003[:, None] >= 2),
        "MUTUAL_P003_SUPPORT3": m003 & (c003[:, None] >= 3),
    }
    freeze = {
        "schema": "RealSaS.E0BObservationOnlyAdmissionFreeze.v1",
        "teacher_identity_consumed": False,
        "variant_mask_sha256": {k: mask_sha(v) for k, v in masks.items()},
        "pair_counts": {k: int(v.sum()) for k, v in masks.items()},
    }
    return masks, freeze

def evaluate_frozen_masks(asset_dir: Path, a: dict, b: dict, masks: dict) -> dict:
    geom, auth = load_asset_authority(asset_dir)
    obs = [x.observable for x in auth]
    comp = face_components(np.asarray(geom["faces"], np.int64))
    P = np.asarray(b["P"], np.float32)
    sv = np.asarray(b["source_view"], np.int64)
    rows = np.asarray(b["matched_row"], np.int64)
    a_support = np.asarray(a["support_mask"], bool)
    source_tri = np.asarray(a["teacher_source_triangle_id"], np.int64)
    nonself = _nonself(sv)
    truth = int((a_support & nonself).sum())
    correct = np.zeros_like(a_support)
    cross = np.zeros_like(a_support)
    same_wrong = np.zeros_like(a_support)
    hidden_bridge = np.zeros_like(a_support)
    visible_wrong = np.zeros_like(a_support)
    for i in range(len(P)):
        scomp = int(comp[int(source_tri[i])])
        for tv in range(8):
            if tv == int(sv[i]):
                continue
            br = int(rows[i, tv])
            if br < 0:
                continue
            physical_error = float(np.linalg.norm(obs[tv].P[br] - P[i]))
            ok = bool(a_support[i, tv] and physical_error <= 0.003)
            correct[i, tv] = ok
            if not ok:
                ttri = int(auth[tv].triangle_id[br])
                tcomp = int(comp[ttri])
                cross[i, tv] = tcomp != scomp
                same_wrong[i, tv] = tcomp == scomp
                hidden_bridge[i, tv] = not bool(a_support[i, tv])
                visible_wrong[i, tv] = bool(a_support[i, tv])
    match_p = np.asarray(b["match_P_error"], np.float32)
    by_variant = {}
    for name, mask0 in masks.items():
        mask = np.asarray(mask0, bool) & nonself
        pred = int(mask.sum())
        tp = int((mask & correct).sum())
        fp = pred - tp
        xcomp = int((mask & cross).sum())
        sw = int((mask & same_wrong).sum())
        hidden = int((mask & hidden_bridge).sum())
        vis_wrong = int((mask & visible_wrong).sum())
        accepted_err = match_p[mask]
        tp_err = match_p[mask & correct]
        fp_err = match_p[mask & (~correct)]
        precision = tp / pred if pred else 1.0
        recall = tp / truth if truth else 1.0
        by_variant[name] = {
            "predicted_pairs": pred,
            "oracle_pairs": truth,
            "true_positive_pairs": tp,
            "false_pairs": fp,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "cross_component_false_pairs": xcomp,
            "same_component_wrong_pairs": sw,
            "oracle_hidden_bridge_pairs": hidden,
            "oracle_visible_wrong_pairs": vis_wrong,
            "unsafe_cross_component_commit_rate": (xcomp / pred) if pred else 0.0,
            "cross_component_share_of_false": (xcomp / fp) if fp else 0.0,
            "all_accepted_match_P_p95": float(np.nanquantile(accepted_err, .95)) if accepted_err.size else None,
            "true_positive_match_P_p95": float(np.nanquantile(tp_err, .95)) if tp_err.size else None,
            "false_match_P_p50": float(np.nanquantile(fp_err, .50)) if fp_err.size else None,
        }
    return {
        "schema": "RealSaS.E0BSafetyAssetEval.v1",
        "asset_id": asset_dir.name,
        "mesh_connected_component_count": int(comp.max() + 1) if len(comp) else 0,
        "teacher_used_only_after_admission_freeze": True,
        "variants": by_variant,
    }

def audit_asset(asset_dir: Path, e0_asset_out: Path) -> tuple[dict, dict]:
    b_path = e0_asset_out / "E0_B_OBSERVABLE_DETERMINISTIC.npz"
    a_path = e0_asset_out / "E0_A_OBSERVABLE_ORACLE.npz"
    if not b_path.is_file() or not a_path.is_file():
        raise FileNotFoundError({"asset": asset_dir.name, "a": str(a_path), "b": str(b_path)})
    with np.load(b_path, allow_pickle=False) as z:
        b = {k: np.asarray(z[k]) for k in z.files}
    masks, freeze = observation_only_admission_masks(asset_dir, b)
    with np.load(a_path, allow_pickle=False) as z:
        a = {k: np.asarray(z[k]) for k in z.files}
    ev = evaluate_frozen_masks(asset_dir, a, b, masks)
    return freeze, ev

def aggregate(rows: list[dict]) -> dict:
    variants = {}
    for name in VARIANTS:
        rs = [r["variants"][name] for r in rows]
        pred = sum(x["predicted_pairs"] for x in rs)
        truth = sum(x["oracle_pairs"] for x in rs)
        tp = sum(x["true_positive_pairs"] for x in rs)
        fp = sum(x["false_pairs"] for x in rs)
        cross = sum(x["cross_component_false_pairs"] for x in rs)
        precision = tp / pred if pred else 1.0
        recall = tp / truth if truth else 1.0
        variants[name] = {
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "macro_precision": float(np.mean([x["precision"] for x in rs])),
            "macro_recall": float(np.mean([x["recall"] for x in rs])),
            "worst_asset_precision": float(np.min([x["precision"] for x in rs])),
            "worst_asset_recall": float(np.min([x["recall"] for x in rs])),
            "predicted_pairs": pred,
            "true_positive_pairs": tp,
            "false_pairs": fp,
            "cross_component_false_pairs": cross,
            "unsafe_cross_component_commit_rate": (cross / pred) if pred else 0.0,
            "cross_component_share_of_false": (cross / fp) if fp else 0.0,
            "oracle_hidden_bridge_pairs": int(sum(x["oracle_hidden_bridge_pairs"] for x in rs)),
            "same_component_wrong_pairs": int(sum(x["same_component_wrong_pairs"] for x in rs)),
            "median_all_accepted_match_P_p95": float(np.median([x["all_accepted_match_P_p95"] for x in rs])),
            "worst_all_accepted_match_P_p95": float(np.max([x["all_accepted_match_P_p95"] for x in rs])),
            "median_true_positive_match_P_p95": float(np.median([x["true_positive_match_P_p95"] for x in rs])),
            "worst_true_positive_match_P_p95": float(np.max([x["true_positive_match_P_p95"] for x in rs])),
        }
    return variants

def print_tables(rows: list[dict], agg: dict) -> None:
    print("\nE0-b SAFETY — FIT CALIBRATION ONLY, NOT GENERALIZATION")
    print("asset      BASE P/R    BASE cross%   M003 P/R    M003 cross%   M003+S2 P/R  M003+S2 cross%")
    print("-" * 102)
    for r in rows:
        v = r["variants"]
        b0, m3, s2 = v["BASE"], v["MUTUAL_P003"], v["MUTUAL_P003_SUPPORT2"]
        aid = r["asset_id"].replace("asset_", "")[:8]
        print(f"{aid:<10} {b0['precision']:.4f}/{b0['recall']:.4f}   {100*b0['unsafe_cross_component_commit_rate']:.3f}%      {m3['precision']:.4f}/{m3['recall']:.4f}   {100*m3['unsafe_cross_component_commit_rate']:.3f}%      {s2['precision']:.4f}/{s2['recall']:.4f}      {100*s2['unsafe_cross_component_commit_rate']:.3f}%")
    print("\nAGGREGATE")
    for name in VARIANTS:
        x = agg[name]
        print(f"{name:<23} micro P/R={x['micro_precision']:.5f}/{x['micro_recall']:.5f} worstP={x['worst_asset_precision']:.5f} crossCommit={100*x['unsafe_cross_component_commit_rate']:.4f}% crossFalseShare={100*x['cross_component_share_of_false']:.2f}%")
    print("\nBASE FALSE-DECOMPOSITION + P95 SEMANTICS")
    print("asset      false  cross  sameWrong  hidden  cross/accepted  allP95    TPonlyP95")
    print("-" * 86)
    for r in rows:
        x = r["variants"]["BASE"]
        aid = r["asset_id"].replace("asset_", "")[:8]
        print(f"{aid:<10} {x['false_pairs']:>5d}  {x['cross_component_false_pairs']:>5d}  {x['same_component_wrong_pairs']:>9d}  {x['oracle_hidden_bridge_pairs']:>6d}  {100*x['unsafe_cross_component_commit_rate']:>12.3f}%  {x['all_accepted_match_P_p95']:.6f}  {x['true_positive_match_P_p95']:.6f}")
    print("\nP95 SEMANTICS: `all_accepted_match_P_p95` includes false matches; `true_positive_match_P_p95` is TP-only.")
    print("IMPORTANT: connected-component crossing is an UNSAFE LOWER BOUND; same-component wrong matches may still hurt at articulation boundaries.")
    print("STOP: no acceptance rule is auto-selected; this is calibration evidence only.")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--e0-run-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    corpus_root = Path(args.corpus_root)
    e0_root = Path(args.e0_run_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    freezes = {}
    for idx, aid in enumerate(CALIBRATION_IDS, 1):
        print(f"[AUDIT] {idx}/8 {aid}", flush=True)
        asset_dir = corpus_root / "master" / "assets" / aid
        e0_dir = e0_root / aid
        freeze, ev = audit_asset(asset_dir, e0_dir)
        freezes[aid] = freeze
        rows.append(ev)
        atomic_json(out / aid / "E0_B_ADMISSION_FREEZE.json", freeze)
        atomic_json(out / aid / "E0_B_SAFETY_EVAL.json", ev)
    agg = aggregate(rows)
    result = {
        "schema": "RealSaS.E0BSafetyCalibration.v1",
        "status": "FIT_CALIBRATION_COMPLETE__NO_RULE_AUTO_SELECTED__NOT_GENERALIZATION",
        "asset_ids": list(CALIBRATION_IDS),
        "asset_count": 8,
        "source_e0_run": "E0_CALIBRATION8_GEOMETRY_V1_3",
        "teacher_identity_consumed_by_admission": False,
        "qualification_opened": False,
        "scientific_optimizer_steps": 0,
        "variants": agg,
        "per_asset": rows,
    }
    atomic_json(out / "E0_B_SAFETY_CALIBRATION_V1.json", result)
    print_tables(rows, agg)

if __name__ == "__main__":
    main()
