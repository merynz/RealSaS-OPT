from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np

SCHEMA = "RealSaS.KnightArachneV6SamplerExposureCourt.v1"
SEED = 20260926
DECODER_SEED = SEED + 202
ROWS_PER_STEP = 384
CLOSURE_STEPS = 8192
THRESHOLDS = (0.0, 0.01, 0.05, 0.10)

def q(x, p):
    return float(np.quantile(np.asarray(x, dtype=np.float64), p)) if len(x) else None

def schedule_uniform(clean):
    rng = np.random.default_rng(DECODER_SEED)
    out = np.empty((CLOSURE_STEPS, min(ROWS_PER_STEP, len(clean))), dtype=np.int32)
    for i in range(CLOSURE_STEPS):
        out[i] = rng.choice(clean, size=out.shape[1], replace=False)
    return out

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def bone_stats(W, valid, sched, thr):
    rows = []
    for j in range(W.shape[1]):
        active = valid & (W[:, j] > thr)
        c = active[sched].sum(axis=1)
        rows.append({
            "target_column": j,
            "active_valid_row_count": int(active.sum()),
            "active_valid_fraction": float(active.sum() / max(1, valid.sum())),
            "zero_active_step_count": int((c == 0).sum()),
            "zero_active_step_fraction": float(np.mean(c == 0)),
            "active_rows_per_step_min": int(c.min()),
            "active_rows_per_step_p05": q(c, .05),
            "active_rows_per_step_p50": q(c, .50),
            "active_rows_per_step_p95": q(c, .95),
            "active_rows_per_step_max": int(c.max()),
            "active_exposure_total": int(c.sum()),
        })
    return rows

def active_balanced_schedule(W, valid, k_per_bone, thr=0.0):
    clean = np.flatnonzero(valid)
    active = [np.flatnonzero(valid & (W[:, j] > thr)) for j in range(W.shape[1])]
    rng = np.random.default_rng(DECODER_SEED + 1000 + k_per_bone)
    out = np.empty((CLOSURE_STEPS, min(ROWS_PER_STEP, len(clean))), dtype=np.int32)
    for s in range(CLOSURE_STEPS):
        chosen, used = [], set()
        order = rng.permutation(W.shape[1])
        for j in order:
            pool = active[int(j)]
            if len(pool) == 0:
                continue
            perm = rng.permutation(pool)
            got = 0
            for x in perm:
                ix = int(x)
                if ix in used:
                    continue
                chosen.append(ix)
                used.add(ix)
                got += 1
                if got >= k_per_bone or len(chosen) >= out.shape[1]:
                    break
            if len(chosen) >= out.shape[1]:
                break
        if len(chosen) < out.shape[1]:
            remain = np.asarray([x for x in clean.tolist() if int(x) not in used], dtype=np.int64)
            need = out.shape[1] - len(chosen)
            fill = rng.choice(remain, size=need, replace=False)
            chosen.extend(map(int, fill))
        out[s] = np.asarray(chosen[:out.shape[1]], dtype=np.int32)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with np.load(a.teacher_bank, allow_pickle=False) as z:
        W = np.asarray(z["weights"], dtype=np.float64)
        valid = np.asarray(z["teacher_valid_mask"], dtype=np.uint8).astype(bool)
        keys = sorted(z.files)
        names = None
        for k in ("target_joint_ids", "canonical_joint_ids", "joint_ids", "target_names"):
            if k in z.files:
                names = [str(x) for x in z[k].tolist()]
                break
    if W.ndim != 2 or valid.shape != (W.shape[0],):
        raise RuntimeError("TEACHER_BANK_SHAPE_DRIFT")
    clean = np.flatnonzero(valid)
    base = schedule_uniform(clean)
    arms = {
        "UNIFORM_VALID_EXACT_V6": base,
        "ACTIVE_BALANCED_K1": active_balanced_schedule(W, valid, 1),
        "ACTIVE_BALANCED_K4": active_balanced_schedule(W, valid, 4),
    }
    arm_reports = {}
    for name, sched in arms.items():
        th = {}
        for t in THRESHOLDS:
            rs = bone_stats(W, valid, sched, t)
            if names is not None and len(names) == W.shape[1]:
                for r in rs:
                    r["target_name"] = names[r["target_column"]]
            th[str(t)] = {
                "bones": rs,
                "bones_with_any_zero_active_step": int(sum(r["zero_active_step_count"] > 0 for r in rs)),
                "worst_zero_active_step_fraction": float(max(r["zero_active_step_fraction"] for r in rs)),
                "median_zero_active_step_fraction": float(np.median([r["zero_active_step_fraction"] for r in rs])),
                "min_active_valid_row_count": int(min(r["active_valid_row_count"] for r in rs)),
            }
        arm_reports[name] = {"schedule_sha256": digest(sched), "thresholds": th}
    rep = {
        "schema": SCHEMA,
        "status": "DIAGNOSTIC_ONLY__NO_MODEL_MUTATION",
        "teacher_bank_keys": keys,
        "teacher_bank_shape": list(W.shape),
        "surface_row_count": int(W.shape[0]),
        "teacher_valid_row_count": int(valid.sum()),
        "teacher_invalid_row_count": int((~valid).sum()),
        "teacher_coverage": float(valid.mean()),
        "schedule": {"seed": DECODER_SEED, "rows_per_step": int(base.shape[1]), "steps": CLOSURE_STEPS},
        "arms": arm_reports,
        "interpretation_boundary": [
            "UNIFORM_VALID_EXACT_V6 exactly replays the V6 row-index schedule through closure step 8192.",
            "ACTIVE_BALANCED arms are exposure-only counterfactuals; they do not train a model and are not claimed to reproduce SkinTokens implementation exactly.",
            "All arms sample only teacher-valid rows; this court cannot solve or evaluate values on teacher-invalid rows.",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_ARACHNE_V6_SAMPLER_EXPOSURE_COURT_PASS", json.dumps({
        "valid": rep["teacher_valid_row_count"],
        "invalid": rep["teacher_invalid_row_count"],
        "uniform_t0": {k:v for k,v in rep["arms"]["UNIFORM_VALID_EXACT_V6"]["thresholds"]["0.0"].items() if k != "bones"},
        "uniform_t001": {k:v for k,v in rep["arms"]["UNIFORM_VALID_EXACT_V6"]["thresholds"]["0.01"].items() if k != "bones"},
    }, sort_keys=True))

if __name__ == "__main__":
    main()
