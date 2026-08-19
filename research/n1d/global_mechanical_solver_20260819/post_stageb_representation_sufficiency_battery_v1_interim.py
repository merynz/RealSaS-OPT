from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

FEATURES = [
    "predicted_flow_mean",
    "moved_fraction_005",
    "transport_gate_mean",
    "transport_edge_mass_mean",
    "transport_entropy_mean",
    "transport_peak_mean",
    "persistent_count",
]


def run(source: Path) -> dict:
    rows = pd.DataFrame(json.loads(source.read_text())["rows"])
    rows["active_target"] = (rows["stratum"] != "near_zero").astype(int)
    rows["logamp_target"] = np.log(1e-6 + rows["true_flow_rms"].astype(float))

    p_active = np.full(len(rows), np.nan)
    logamp = np.full(len(rows), np.nan)

    for family in sorted(rows.family.unique()):
        test = (rows.family == family).to_numpy()
        train = ~test

        clf = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=0)),
        ])
        clf.fit(rows.loc[train, FEATURES].astype(float), rows.loc[train, "active_target"])
        p_active[test] = clf.predict_proba(rows.loc[test, FEATURES].astype(float))[:, 1]

        train_active = train & (rows.active_target.to_numpy() == 1)
        reg = Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        reg.fit(rows.loc[train_active, FEATURES].astype(float), rows.loc[train_active, "logamp_target"])
        logamp[test] = reg.predict(rows.loc[test, FEATURES].astype(float))

    rows["lofo_p_active"] = p_active
    rows["lofo_logamp"] = logamp

    active = rows[rows.active_target == 1]
    per_family = {}
    for family, group in rows.groupby("family"):
        ga = group[group.active_target == 1]
        raw_rho = float(spearmanr(ga.predicted_flow_mean, ga.true_flow_rms).statistic) if len(ga) >= 3 else None
        ridge_rho = float(spearmanr(ga.lofo_logamp, ga.logamp_target).statistic) if len(ga) >= 3 else None
        auc = float(roc_auc_score(group.active_target, group.lofo_p_active)) if group.active_target.nunique() == 2 else None
        pred = (group.lofo_p_active >= 0.5).astype(int)
        per_family[str(int(family))] = {
            "active_n": int(group.active_target.sum()),
            "near_zero_n": int((group.active_target == 0).sum()),
            "raw_amp_spearman": raw_rho,
            "lofo_ridge_amp_spearman": ridge_rho,
            "lofo_p_active_auroc": auc,
            "lofo_p_active_balanced_accuracy": float(balanced_accuracy_score(group.active_target, pred)),
        }

    # Coarse cross-family collision proxy. This is deliberately not the final
    # carrier-level representation collision test.
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(rows[FEATURES].astype(float))
    family = rows.family.to_numpy()
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    D[family[:, None] == family[None, :]] = np.inf
    nearest = np.argmin(D, axis=1)
    nearest_dist = D[np.arange(len(rows)), nearest]
    mismatch = rows.active_target.to_numpy() != rows.iloc[nearest].active_target.to_numpy()
    logamp_diff = np.abs(rows.logamp_target.to_numpy() - rows.iloc[nearest].logamp_target.to_numpy())

    collision = {}
    for q in (0.05, 0.10):
        threshold = float(np.quantile(nearest_dist, q))
        mask = nearest_dist <= threshold
        collision[str(q)] = {
            "count": int(mask.sum()),
            "active_silent_mismatch_fraction": float(mismatch[mask].mean()),
            "median_abs_log_true_amp_diff": float(np.median(logamp_diff[mask])),
            "max_abs_log_true_amp_diff": float(logamp_diff[mask].max()),
        }

    family_raw = [v["raw_amp_spearman"] for v in per_family.values() if v["raw_amp_spearman"] is not None]
    family_ridge = [v["lofo_ridge_amp_spearman"] for v in per_family.values() if v["lofo_ridge_amp_spearman"] is not None]

    return {
        "episodes": int(len(rows)),
        "families": int(rows.family.nunique()),
        "strata": {str(k): int(v) for k, v in rows.stratum.value_counts().items()},
        "overall_p_active_balanced_accuracy": float(balanced_accuracy_score(rows.active_target, rows.lofo_p_active >= 0.5)),
        "overall_p_active_auroc": float(roc_auc_score(rows.active_target, rows.lofo_p_active)),
        "overall_lofo_ridge_logamp_spearman": float(spearmanr(active.lofo_logamp, active.logamp_target).statistic),
        "median_family_lofo_ridge_logamp_spearman": float(np.median(family_ridge)),
        "overall_raw_predicted_flow_amp_spearman": float(spearmanr(active.predicted_flow_mean, active.true_flow_rms).statistic),
        "median_family_raw_predicted_flow_amp_spearman": float(np.median(family_raw)),
        "min_family_raw_predicted_flow_amp_spearman": float(np.min(family_raw)),
        "families_raw_amp_spearman_ge_0_50": int(sum(x >= 0.5 for x in family_raw)),
        "families_raw_amp_spearman_negative": int(sum(x < 0 for x in family_raw)),
        "collision_proxy": collision,
        "per_family": per_family,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.source)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text)
    print(text)
