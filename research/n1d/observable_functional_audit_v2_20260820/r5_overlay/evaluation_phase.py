from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from observable_phase import decorate_pb_observable, sha256_file
from mechanics_metrics import effect_blocks, truth_block_errors

HERE = Path(__file__).resolve().parent
DEPS = HERE / "frozen_deps"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gfdr = _load("realsas_gfdr_v2_frozen", DEPS / "realsas_gfdr_v2.py")


def local_scale(P, k=4):
    P = np.asarray(P, np.float64)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = np.partition(D, k - 1, axis=1)[:, :k]
    return np.median(nn, axis=1)


def _truth(sidecar: Path):
    z = np.load(sidecar, allow_pickle=False)
    center = np.asarray(z["camera_center"], np.float64)
    half = float(z["camera_half_extent"])
    TA = (np.asarray(z["surface_points_A"], np.float64) - center) / (2 * half)
    TB = (np.asarray(z["surface_points_B"], np.float64) - center) / (2 * half)
    return {
        "P_A": TA.astype(np.float32), "P_B": TB.astype(np.float32),
        "N_A": np.asarray(z["surface_normals_A"], np.float32), "N_B": np.asarray(z["surface_normals_B"], np.float32),
        "V_A": np.asarray(z["surface_visibility_A"], np.uint8), "V_B": np.asarray(z["surface_visibility_B"], np.uint8),
    }


def _map64(PA, t):
    D = cdist(np.asarray(PA, np.float64), np.asarray(t["P_A"], np.float64))
    rr, cc = linear_sum_assignment(D)
    order = np.argsort(rr); ids = cc[order]
    return ids.astype(np.int64), D[np.arange(len(PA)), ids]


def _subset_truth(t, ids):
    q = {}
    for k in ("P_A", "P_B", "N_A", "N_B"):
        q[k] = np.asarray(t[k])[ids]
    for k in ("V_A", "V_B"):
        q[k] = np.asarray(t[k])[:, ids]
    return q


def _pool(z, i):
    off = np.asarray(z["H_offsets"], np.int64); a, b = int(off[i]), int(off[i + 1])
    return np.asarray(z["H_xyz"])[a:b], np.asarray(z["H_reproj_px"])[a:b], np.asarray(z["H_desc_score"])[a:b]


def _rank01(x, descending=False):
    x = np.asarray(x, np.float64); order = np.argsort(-x if descending else x, kind="stable")
    r = np.empty(len(x), np.float64); r[order] = np.arange(len(x)); return r / max(len(x) - 1, 1)


def evaluate_family(root: Path, family: int, episode: str, obs_dir: Path, sidecar: Path):
    state_path = obs_dir / f"{family}_{episode}_OBSERVABLE_STATE.npz"
    state_meta_path = state_path.with_suffix(".json")
    meta = json.loads(state_meta_path.read_text())
    if meta.get("truth_access") != "NONE": raise RuntimeError("observable state truth contract invalid")
    if sha256_file(state_path) != meta["observable_state_sha256"]: raise RuntimeError("observable state SHA mismatch")
    z = np.load(state_path, allow_pickle=False)
    PA = np.asarray(z["P_A"], np.float32); PB0 = np.asarray(z["P_B"], np.float32)
    base_world = {k: np.asarray(z[k]) for k in ("P_A", "P_B", "N_A", "N_B", "V_A", "V_B")}
    base_g = gfdr.compute_gfdr_v2(**base_world, Z=None, U=None)

    t512 = _truth(sidecar)
    ids, map_err = _map64(PA, t512)
    t = _subset_truth(t512, ids)
    sA = local_scale(t512["P_A"], 4)[ids]
    sB = local_scale(t512["P_B"], 4)[ids]
    reliable = map_err <= 2 * sA
    active = np.linalg.norm(np.asarray(t["P_B"]) - np.asarray(t["P_A"]), axis=1) > .005
    base_err = np.linalg.norm(PB0 - np.asarray(t["P_B"]), axis=1)
    true_g = gfdr.compute_gfdr_v2(**t, Z=None, U=None)

    candidates = []
    for i in range(64):
        H, reproj, desc = _pool(z, i)
        if not len(H):
            candidates.append({"n": 0, "nearest": -1, "min_err": float("inf"), "contained2": False})
            continue
        er = np.linalg.norm(H - np.asarray(t["P_B"])[i][None, :], axis=1)
        j = int(np.argmin(er))
        rr = _rank01(reproj, False); dr = _rank01(desc, True)
        obs_rank = .75 * rr + .25 * dr
        candidates.append({
            "n": int(len(H)), "nearest": j, "min_err": float(er[j]), "contained2": bool(er[j] <= 2 * sB[i]),
            "nearest_reproj_rank01": float(rr[j]), "nearest_desc_rank01": float(dr[j]),
            "nearest_combined_rank01": float(_rank01(obs_rank, False)[j]),
        })

    witness_ids = [i for i in range(64) if reliable[i] and active[i] and candidates[i]["contained2"] and base_err[i] > 2 * sB[i]]
    rows = []
    for i in witness_ids:
        H, reproj, desc = _pool(z, i); j = int(candidates[i]["nearest"])
        PB = PB0.copy(); PB[i] = H[j]
        NB, VB, XYB = decorate_pb_observable(root, family, episode, meta["route"], PA, PB)
        swap_world = {"P_A": PA, "P_B": PB, "N_A": np.asarray(z["N_A"]), "N_B": NB, "V_A": np.asarray(z["V_A"]), "V_B": VB}
        swap_g = gfdr.compute_gfdr_v2(**swap_world, Z=None, U=None)
        effect = effect_blocks(base_g, swap_g, PA, i)
        btruth = truth_block_errors(base_g, true_g, PA, i); struth = truth_block_errors(swap_g, true_g, PA, i)
        bcomp = btruth["composite_median_capped10"]; scomp = struth["composite_median_capped10"]
        rel_improve = float((bcomp - scomp) / (abs(bcomp) + 1e-8)) if np.isfinite(bcomp) and np.isfinite(scomp) else float("nan")
        material_improve = bool(np.isfinite(rel_improve) and rel_improve >= .20 and (bcomp - scomp) >= .05)
        material_degrade = bool(np.isfinite(rel_improve) and rel_improve <= -.20 and (scomp - bcomp) >= .05)
        rows.append({
            "family": int(family), "carrier": int(i), "truth_surface_id": int(ids[i]),
            "mapping_error_over_scaleA": float(map_err[i] / max(sA[i], 1e-12)),
            "truth_motion": float(np.linalg.norm(np.asarray(t["P_B"])[i] - np.asarray(t["P_A"])[i])),
            "baseline_error_over_scaleB": float(base_err[i] / max(sB[i], 1e-12)),
            "alternate_error_over_scaleB": float(candidates[i]["min_err"] / max(sB[i], 1e-12)),
            "candidate_observable_rank": {k: v for k, v in candidates[i].items() if "rank01" in k},
            "effect": effect,
            "truth_relative": {"baseline": btruth, "swap": struth, "relative_improvement": rel_improve, "material_improve": material_improve, "material_degrade": material_degrade},
            "descriptive_geometry": {"F_delta_NRMS": float(np.linalg.norm(np.asarray(base_g["F_delta_normalized"])[i] - np.asarray(swap_g["F_delta_normalized"])[i]) / (np.linalg.norm(np.asarray(swap_g["F_delta_normalized"])[i]) + 1e-8))},
        })
    return {
        "family": int(family), "observable_state_sha256": meta["observable_state_sha256"],
        "sidecar_sha256": sha256_file(sidecar),
        "population": {"mapping_reliable": int(reliable.sum()), "active": int(active.sum()), "feasible_contained2": int(sum(c["contained2"] for c in candidates)), "witnesses": len(witness_ids), "witness_carriers": witness_ids},
        "rows": rows,
    }


def aggregate(fams):
    rows = [r for f in fams for r in f["rows"]]; n = len(rows)
    if n == 0:
        return {"n": 0, "verdict": "INSUFFICIENT_HARDTAIL_POPULATION"}
    eq = np.array([r["effect"]["equivalent"] for r in rows], bool)
    imp = np.array([r["truth_relative"]["material_improve"] for r in rows], bool)
    deg = np.array([r["truth_relative"]["material_degrade"] for r in rows], bool)
    per = {}
    for f in fams:
        rr = f["rows"]
        per[str(f["family"])] = {"n": len(rr), "equivalence": (float(np.mean([x["effect"]["equivalent"] for x in rr])) if rr else None), "material_improve": (float(np.mean([x["truth_relative"]["material_improve"] for x in rr])) if rr else None)}
    enough = bool(n >= 20 and sum(v["n"] > 0 for v in per.values()) >= 4)
    family_guard = all(v["equivalence"] >= .60 for v in per.values() if v["n"] >= 5)
    eqfrac = float(eq.mean()); impfrac = float(imp.mean()); degfrac = float(deg.mean())
    if not enough:
        verdict = "INSUFFICIENT_HARDTAIL_POPULATION"
    elif eqfrac >= .80 and family_guard:
        verdict = "GEOMETRIC_SINGLETON_NOT_REQUIRED_UNDER_REAL_OBSERVABLE_GFDR_V2"
    elif eqfrac <= .50 and impfrac >= .50:
        verdict = "MECHANICALLY_MATERIAL_ALTERNATIVES_EXIST__NEXT_FUNCTIONAL_CLASS_OBSERVABILITY_AUDIT"
    elif eqfrac <= .50:
        verdict = "MECHANICS_CHANGE_BUT_TEACHER_NEAR_UTILITY_NOT_DOMINANT__RESEARCH_REQUIRED"
    else:
        verdict = "FUNCTIONAL_QUOTIENT_AMBIGUOUS__RESEARCH_REQUIRED"
    return {"n": n, "equivalence_fraction": eqfrac, "material_truth_improvement_fraction": impfrac, "material_truth_degradation_fraction": degfrac, "per_family": per, "enough_population": enough, "family_guard": family_guard, "verdict": verdict}


def _json_safe(x):
    if isinstance(x, dict): return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list): return [_json_safe(v) for v in x]
    if isinstance(x, tuple): return [_json_safe(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x); return v if np.isfinite(v) else None
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sidecar-root", type=Path, required=True)
    ap.add_argument("--observable-dir", type=Path, required=True)
    ap.add_argument("--families", nargs="+", type=int, required=True)
    ap.add_argument("--episode", default="e00")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fams = []
    for f in a.families:
        padded = f"{int(f):05d}"
        sidecar = a.sidecar_root / padded / "B" / "observation_sidecar.npz"
        fams.append(evaluate_family(a.root, int(f), a.episode, a.observable_dir, sidecar))
    result = {"schema": "RealSaS.N1D.ObservableFunctionalAuditV2.Result.v1", "families": fams, "aggregate": aggregate(fams)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists(): raise RuntimeError(f"refusing to overwrite canonical result: {a.out}")
    a.out.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
