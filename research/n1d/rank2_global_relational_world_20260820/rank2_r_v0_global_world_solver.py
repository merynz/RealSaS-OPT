from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

SCHEMA = "RealSaS.N1D.Rank2GlobalRelationalWorld.V0"
TOP_FRAC = 0.10
K_MIN = 8
K_MAX = 32
ANCHORS = 6
LAMBDA_R = 1.0
ANCHOR_MULT = 2.0
N_RESTARTS = 16
MAX_SWEEPS = 50
HUBER_DELTA = 0.25


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rank01(x: np.ndarray, descending: bool = False) -> np.ndarray:
    x = np.asarray(x, np.float64)
    order = np.argsort(-x if descending else x, kind="stable")
    r = np.empty(len(x), np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r / max(len(x) - 1, 1)


def huber(x: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    x = np.abs(np.asarray(x, np.float64))
    return np.where(x <= delta, 0.5 * x * x, delta * (x - 0.5 * delta))


def robust_scene_scale(P: np.ndarray) -> float:
    P = np.asarray(P, np.float64)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    vals = D[np.triu_indices(len(P), 1)]
    vals = vals[np.isfinite(vals) & (vals > 1e-9)]
    if not len(vals):
        return 1.0
    return float(np.median(vals))


def relation_features_from_delta(delta: np.ndarray, scale: float) -> np.ndarray:
    d = np.asarray(delta, np.float64)
    r = np.linalg.norm(d, axis=-1, keepdims=True)
    u = d / np.maximum(r, 1e-9)
    # radial, signed canonical components, unit direction
    return np.concatenate([r / scale, d / scale, u], axis=-1)


def relation_loss(feat_a: np.ndarray, feat_b: np.ndarray) -> np.ndarray:
    # Radial is most invariant under articulation; canonical signed components and
    # direction retain the historical address semantics but carry lower weight.
    w = np.array([1.0, 0.25, 0.25, 0.25, 0.20, 0.20, 0.20], dtype=np.float64)
    return np.sum(w * huber(feat_b - feat_a), axis=-1) / np.sum(w)


def load_family(state_path: Path):
    z = np.load(state_path, allow_pickle=False)
    required = {"P_A","P_B","H_xyz","H_reproj_px","H_desc_score","H_offsets"}
    missing = sorted(required - set(z.files))
    if missing:
        raise RuntimeError(f"missing observable arrays: {missing}")
    PA = np.asarray(z["P_A"], np.float64)
    PB = np.asarray(z["P_B"], np.float64)
    H = np.asarray(z["H_xyz"], np.float64)
    reproj = np.asarray(z["H_reproj_px"], np.float64)
    desc = np.asarray(z["H_desc_score"], np.float64)
    off = np.asarray(z["H_offsets"], np.int64)
    if PA.shape != (64,3) or PB.shape != (64,3) or off.shape != (65,):
        raise RuntimeError((PA.shape, PB.shape, off.shape))
    return PA, PB, H, reproj, desc, off


def build_domains(PA, PB, H, reproj, desc, off):
    domains = []
    scale = robust_scene_scale(PA)
    for i in range(64):
        a,b = int(off[i]), int(off[i+1])
        pts = H[a:b]
        rp = reproj[a:b]
        ds = desc[a:b]
        n = len(pts)
        if n == 0:
            raise RuntimeError(f"empty H for carrier {i}")
        obs = 0.75 * rank01(rp, False) + 0.25 * rank01(ds, True)
        k = min(K_MAX, max(K_MIN, int(np.ceil(TOP_FRAC * n))))
        k = min(k, n)
        keep_local = np.argsort(obs, kind="stable")[:k]
        kpts = pts[keep_local]
        kobs = obs[keep_local]
        # normalized G unary inside retained set; preserve absolute rank information.
        g = kobs.astype(np.float64)
        # Sharpness of retained G support.
        center = np.median(kpts, axis=0)
        if len(kpts) > 1:
            cov = np.cov((kpts-center).T)
            eig = np.maximum(np.linalg.eigvalsh(cov), 0.0)
            spread = float(np.sqrt(eig[-1]) / max(scale, 1e-9))
            linearity = float((eig[-2] + 1e-12) / (eig[-1] + 1e-12))
        else:
            spread, linearity = 0.0, 0.0
        # Anchor candidate uses observable frozen baseline as a G witness, projected
        # to the retained feasible domain; no external geometry is read.
        anchor_local = int(np.argmin(np.linalg.norm(kpts - PB[i][None,:], axis=1)))
        domains.append({
            "points": kpts,
            "g": g,
            "orig_local": keep_local.astype(np.int64),
            "orig_global": (a + keep_local).astype(np.int64),
            "sharpness": spread,
            "linearity": linearity,
            "anchor_choice": anchor_local,
            "raw_n": n,
        })
    return domains, scale


def choose_anchors(domains):
    keys = [(domains[i]["sharpness"], float(np.min(domains[i]["g"])), i) for i in range(64)]
    return np.array([x[2] for x in sorted(keys)[:ANCHORS]], dtype=np.int64)


def build_A_features(PA, scale):
    D = PA[:,None,:] - PA[None,:,:]
    return relation_features_from_delta(D, scale)


def pair_weight(i, j, anchors_set):
    return ANCHOR_MULT if (i in anchors_set or j in anchors_set) else 1.0


def unary_energy(domains, choice):
    return float(np.mean([domains[i]["g"][int(choice[i])] for i in range(64)]))


def relational_energy(PA_feat, points, scale, anchors_set):
    total = 0.0; wsum = 0.0
    for i in range(64):
        d = points[i] - points[i+1:]
        if not len(d):
            continue
        fb = relation_features_from_delta(d, scale)
        fa = PA_feat[i, i+1:]
        loss = relation_loss(fa, fb)
        pair_w = np.array([pair_weight(i,j,anchors_set) for j in range(i+1,64)], np.float64)
        total += float(np.sum(pair_w * loss)); wsum += float(np.sum(pair_w))
    return total / max(wsum, 1e-12)


def objective(PA_feat, domains, choice, scale, anchors_set):
    pts = np.stack([domains[i]["points"][int(choice[i])] for i in range(64)])
    eg = unary_energy(domains, choice)
    er = relational_energy(PA_feat, pts, scale, anchors_set)
    return eg + LAMBDA_R * er, eg, er


def coordinate_costs(i, PA_feat, domains, choice, scale, anchors_set):
    cand = domains[i]["points"]
    K = len(cand)
    # Unary mean contribution. Other unary terms are constant, so this is enough
    # for argmin; scale by 1/64 to match global objective.
    c = domains[i]["g"].astype(np.float64) / 64.0
    for j in range(64):
        if j == i:
            continue
        pj = domains[j]["points"][int(choice[j])]
        if i < j:
            delta = cand - pj[None,:]
            fa = np.repeat(PA_feat[i,j][None,:], K, axis=0)
        else:
            # Global energy stores pair (j,i) as point_j-point_i. Preserve sign.
            delta = pj[None,:] - cand
            fa = np.repeat(PA_feat[j,i][None,:], K, axis=0)
        fb = relation_features_from_delta(delta, scale)
        w = pair_weight(i,j,anchors_set)
        c += (LAMBDA_R * w * relation_loss(fa, fb))
    # Relational denominator is constant across candidate choices; divide by
    # exact full pair weight sum so coordinate argmin and global energy agree.
    denom = 0.0
    for a in range(64):
        for b in range(a+1,64):
            denom += pair_weight(a,b,anchors_set)
    # relational contributions above are unnormalized; unary already mean-normalized.
    c = domains[i]["g"].astype(np.float64)/64.0 + (c - domains[i]["g"].astype(np.float64)/64.0)/denom
    return c


def initial_choice(domains, anchors, restart, family):
    c = np.zeros(64, np.int64)
    if restart == 0:
        for i,d in enumerate(domains): c[i] = int(np.argmin(d["g"]))
    else:
        rng = np.random.default_rng(np.uint64((int(family)*1009 + restart*9176) & 0xFFFFFFFF))
        for i,d in enumerate(domains):
            g = d["g"]
            prob = np.exp(-4.0 * (g - np.min(g)))
            prob /= prob.sum()
            c[i] = int(rng.choice(len(g), p=prob))
    for i in anchors:
        c[int(i)] = int(domains[int(i)]["anchor_choice"])
    return c


def optimize_one(PA_feat, domains, scale, anchors, family, restart):
    anchors_set = set(map(int, anchors.tolist()))
    c = initial_choice(domains, anchors, restart, family)
    sweeps = 0
    for sweep in range(MAX_SWEEPS):
        changed = 0
        for i in range(64):
            if i in anchors_set:
                continue
            costs = coordinate_costs(i, PA_feat, domains, c, scale, anchors_set)
            best = int(np.argmin(costs))
            if best != int(c[i]):
                c[i] = best; changed += 1
        sweeps = sweep + 1
        if changed == 0:
            break
    total, eg, er = objective(PA_feat, domains, c, scale, anchors_set)
    return c, {"restart": restart, "sweeps": sweeps, "total": total, "E_G": eg, "E_R": er}


def local_counterfactual_margin(PA_feat, domains, choice, scale, anchors):
    aset = set(map(int, anchors.tolist()))
    best_margin = float("inf"); best_info = None
    for i in range(64):
        if i in aset: continue
        costs = coordinate_costs(i, PA_feat, domains, choice, scale, aset)
        cur = int(choice[i]); base = float(costs[cur])
        for j,val in enumerate(costs):
            if j == cur: continue
            margin = float(val-base)
            if margin < best_margin:
                best_margin = margin; best_info = (i,cur,j)
    return best_margin, best_info


def solve_family(state_path: Path, family: int):
    PA, PB, H, reproj, desc, off = load_family(state_path)
    domains, scale = build_domains(PA, PB, H, reproj, desc, off)
    anchors = choose_anchors(domains)
    PA_feat = build_A_features(PA, scale)
    runs = []
    choices = []
    for r in range(N_RESTARTS):
        c, info = optimize_one(PA_feat, domains, scale, anchors, family, r)
        choices.append(c.copy()); runs.append(info)
    order = sorted(range(len(runs)), key=lambda r:(runs[r]["total"], tuple(choices[r].tolist())))
    best_i = order[0]; best = choices[best_i]; best_info = runs[best_i]
    second = runs[order[1]]["total"] if len(order)>1 else float("inf")
    exact_restart_agreement = int(sum(np.array_equal(c,best) for c in choices))
    local_margin, local_alt = local_counterfactual_margin(PA_feat, domains, best, scale, anchors)
    chosen_points = np.stack([domains[i]["points"][int(best[i])] for i in range(64)])
    anchor_set = set(map(int,anchors.tolist()))
    baseline_total = None
    # Descriptive relational energy of the frozen baseline PB. No G unary is assigned
    # because PB need not be a member of the retained domain.
    baseline_R = relational_energy(PA_feat, PB, scale, anchor_set)
    return {
        "schema": SCHEMA,
        "family": int(family),
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
        "parameters": {"top_frac":TOP_FRAC,"k_min":K_MIN,"k_max":K_MAX,"anchors":ANCHORS,"lambda_R":LAMBDA_R,"anchor_mult":ANCHOR_MULT,"restarts":N_RESTARTS,"max_sweeps":MAX_SWEEPS,"huber_delta":HUBER_DELTA},
        "scene_scale": scale,
        "anchors": [int(x) for x in anchors],
        "anchor_sharpness": [float(domains[int(x)]["sharpness"]) for x in anchors],
        "candidate_counts_raw": [int(d["raw_n"]) for d in domains],
        "candidate_counts_retained": [int(len(d["points"])) for d in domains],
        "selected_retained_index": [int(x) for x in best],
        "selected_original_local_index": [int(domains[i]["orig_local"][int(best[i])]) for i in range(64)],
        "selected_original_global_index": [int(domains[i]["orig_global"][int(best[i])]) for i in range(64)],
        "selected_points": chosen_points.tolist(),
        "objective": {**best_info, "baseline_R_descriptive": baseline_R, "second_restart_total":float(second), "best_second_restart_margin":float(second-best_info["total"])},
        "stability": {"exact_restart_agreement":exact_restart_agreement,"restart_count":N_RESTARTS,"all_restarts_exact":bool(exact_restart_agreement==N_RESTARTS),"local_counterfactual_min_margin":float(local_margin),"local_counterfactual_arg":None if local_alt is None else [int(x) for x in local_alt],"locally_strict":bool(local_margin>1e-12)},
        "restarts": runs,
        "truth_access": "NONE",
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a=ap.parse_args()
    if a.out.exists(): raise RuntimeError(f"refusing overwrite: {a.out}")
    result=solve_family(a.state,a.family)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps({"family":result['family'],"objective":result['objective'],"stability":result['stability'],"anchors":result['anchors']},indent=2,sort_keys=True))

if __name__=='__main__': main()
