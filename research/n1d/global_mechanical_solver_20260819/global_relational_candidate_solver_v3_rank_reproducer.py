"""RealSaS N1D Global Relational Candidate Solver V3 rank-calibrated reproducer.

This is the compact canonical reproducer for the single V3 semantic change.
It intentionally reuses the exact persisted V2 execution source rather than duplicating
F16/M256 construction code.

Preparation:
  cat source_bundle_b64/global_relational_candidate_solver_v2_exec.py.b64.part00 \
      source_bundle_b64/global_relational_candidate_solver_v2_exec.py.b64.part01 \
      source_bundle_b64/global_relational_candidate_solver_v2_exec.py.b64.part02 \
    | base64 -d > global_relational_candidate_solver_v2_exec.py

The reassembled V2 source must have SHA256
047c264e5a59ff05386cb801e23f0fb8a083828817a329737c50b407093ae8ba.

V3 changes only V2 cost calibration from median/IQR z-scores to empirical
mid-rank percentiles. Candidate domains, raw factors, graph, degree weights,
ICM mechanics and evaluator sets remain V2-identical.
"""
from __future__ import annotations
import json
import numpy as np
import global_relational_candidate_solver_v2_exec as v2


def empirical_midrank_percentile(cost):
    """P(c_k) = (#<c_k + .5 #==c_k) / N, exact floating equality."""
    a = np.asarray(cost, dtype=np.float64)
    flat = a.reshape(-1)
    ordered = np.sort(flat, kind="stable")
    left = np.searchsorted(ordered, flat, side="left")
    right = np.searchsorted(ordered, flat, side="right")
    return ((left + 0.5 * (right - left)) / len(flat)).reshape(a.shape)


def solve_family_rank(f):
    # Monotonic transform: unary argmin remains exactly V2 U_ONLY.
    U = [empirical_midrank_percentile(n["Uraw"]) for n in f["nodes"]]
    x = np.array([int(np.argmin(u)) for u in U], dtype=int)
    x0 = x.copy()

    edges = v2.build_graph(f)
    deg = np.zeros(64, dtype=int)
    for i, j, _ in edges:
        deg[i] += 1
        deg[j] += 1

    adj = [[] for _ in range(64)]
    mats = {}
    for i, j, views in edges:
        # v2.pair_R returns median/IQR calibrated R, so reconstruct the raw relation
        # directly from the same frozen candidate projections and DIS observations.
        ni, nj = f["nodes"][i], f["nodes"][j]
        Mi, Mj = ni["M"], nj["M"]
        PA = f["PA"]
        vals = []
        for view in views:
            di = v2.project_np(Mi, view) - v2.project_np(PA[i][None], view)[0]
            dj = v2.project_np(Mj, view) - v2.project_np(PA[j][None], view)[0]
            obs = f["dis"][view, i] - f["dis"][view, j]
            residual = di[:, None, :] - dj[None, :, :] - obs[None, None, :]
            vals.append(np.linalg.norm(residual, axis=2))
        raw = np.median(np.stack(vals, axis=2), axis=2)
        R = empirical_midrank_percentile(raw)
        w = 1.0 / max(deg[i], 1) + 1.0 / max(deg[j], 1)
        mats[(i, j)] = (R, w)
        adj[i].append((j, R, w, True))
        adj[j].append((i, R, w, False))

    changes = []
    for sweep in range(20):
        changed = 0
        order = range(64) if sweep % 2 == 0 else range(63, -1, -1)
        for i in order:
            cost = U[i].copy()
            for j, R, w, forward in adj[i]:
                cost += w * (R[:, x[j]] if forward else R[x[j], :])
            q = int(np.argmin(cost))  # lowest index is deterministic tie rule
            if q != x[i]:
                x[i] = q
                changed += 1
        changes.append(changed)
        if changed == 0:
            break

    return x0, x, {
        "edges": len(edges),
        "sweeps": len(changes),
        "changed_nodes": changes,
    }


def run():
    fams = {fid: v2.build_family(fid) for fid in v2.FAMILIES}
    pf = v2.preflight(fams)

    # Frozen V3 parity gate.
    assert pf["full_F16_parity_all"]
    assert pf["den"] == 492
    assert abs(pf["pooled_2x"] - 0.975609756097561) < 1e-12
    assert abs(pf["worst_family_2x"] - 0.9322033898305084) < 1e-12
    assert pf["families_ge_090"] == 8
    assert abs(pf["best_worst_gap"] - 0.06779661016949157) < 1e-12

    per = {}
    xs = {}
    for fid, f in fams.items():
        u, g, diag = solve_family_rank(f)
        xs[fid] = (u, g)
        per[fid] = {
            "diag": diag,
            "primary": {
                "U_RANK_ONLY": v2.eval_arm(f, u, True),
                "G_RANK_REL": v2.eval_arm(f, g, True),
            },
            "secondary": {
                "U_RANK_ONLY": v2.eval_arm(f, u, False),
                "G_RANK_REL": v2.eval_arm(f, g, False),
            },
        }

    agg = {}
    for key in ("primary", "secondary"):
        agg[key] = {}
        for arm in ("U_RANK_ONLY", "G_RANK_REL"):
            q = v2.aggregate_evals(per, arm, key)
            if key == "primary":
                errs = []
                for fid in v2.FAMILIES:
                    x = xs[fid][0 if arm == "U_RANK_ONLY" else 1]
                    errs += v2.all_errors(fams[fid], x, True)
                q["median_norm_err"] = float(np.median(errs))
                q["p90_norm_err"] = float(np.quantile(errs, .9))
            agg[key][arm] = q

    U = agg["primary"]["U_RANK_ONLY"]
    G = agg["primary"]["G_RANK_REL"]
    hard = [11032, 13203, 15290]
    gains = {
        fid: per[fid]["primary"]["G_RANK_REL"]["contain2"]
             - per[fid]["primary"]["U_RANK_ONLY"]["contain2"]
        for fid in hard
    }

    absolute = (
        G["contain2"] >= .75
        and G["worst_family_contain2"] >= .60
        and all(per[f]["primary"]["G_RANK_REL"]["contain2"] >= .60 for f in hard)
        and G["contain1"] >= .50
        and G["worst_family_contain1"] >= .35
        and G["median_norm_err"] <= 1.00
        and G["best_worst_gap"] <= .30
    )

    # Use a tiny numeric tolerance only to represent the exact frozen >=.08 gate;
    # 11032 is exactly 2/25 = .08.
    eps = 1e-12
    causal = (
        G["contain2"] - U["contain2"] + eps >= .08
        and (
            G["worst_family_contain2"] - U["worst_family_contain2"] + eps >= .08
            or (
                sum(g + eps >= .10 for g in gains.values()) >= 2
                and all(g + eps >= -.05 for g in gains.values())
            )
        )
    )
    safety = (
        agg["secondary"]["G_RANK_REL"]["contain2"]
        >= agg["secondary"]["U_RANK_ONLY"]["contain2"] - .03
    )

    verdict = (
        "GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V3_RANK_PASS"
        if absolute and causal and safety
        else "RANK_CALIBRATION_INSUFFICIENT_V3"
    )
    result = {
        "preflight": pf,
        "per_family": per,
        "aggregate": agg,
        "hardtail_gains": gains,
        "gates": {"absolute": absolute, "causal": causal, "safety": safety},
        "verdict": verdict,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
