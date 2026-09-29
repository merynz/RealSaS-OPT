from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXPERIMENT_SCHEMA = "RealSaS.KnightArachneV6CoverageSampling2x2Preregistration.v1"
EXPECTED_EXPERIMENT_STATUS = "FROZEN_BEFORE_ANY_2X2_OPTIMIZER_STEP"
EXPECTED_V6_SOURCE_SHA256 = "39ff4f7a2051f0d9c1251a5999f472b9926e83deacbedcedc8e92eea424dce01"
EXPECTED_V6_PREREG_SHA256 = "187f5c1bac1e813ebdfd1bbb76fa6def3d6543b4c7abc5609d217b3ece5fd597"
EXPECTED_BANK_SHA256 = "26b6891ff81b9bfc3405461394159547fa76b417852a44518425b4e1de7e67d3"
EXPECTED_UNIFORM_VALID_SCHEDULE_SHA256 = "910e79a11dd70210108e164ed6e2b0c1a4b1fbb69632b80f546c68a9bb5331e9"
ARMS = {
    "UNIFORM_VALID",
    "ACTIVE_BALANCED_VALID",
    "UNIFORM_ALL_PROJECTED",
    "ACTIVE_BALANCED_ALL_PROJECTED",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def schedule_digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("realsas_exact_knight_v6", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("V6_IMPORT_SPEC_FAIL")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def uniform_schedule(pool: np.ndarray, *, steps: int, rows: int, seed: int) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    n = min(int(rows), len(pool))
    if n <= 0:
        raise RuntimeError("EMPTY_TRAINING_POOL")
    rng = np.random.default_rng(int(seed))
    out = np.empty((int(steps), n), dtype=np.int32)
    for i in range(int(steps)):
        out[i] = rng.choice(pool, size=n, replace=False)
    return out


def active_balanced_schedule(
    weights: np.ndarray,
    pool_mask: np.ndarray,
    *,
    steps: int,
    rows: int,
    seed: int,
    per_target: int = 4,
) -> np.ndarray:
    pool = np.flatnonzero(np.asarray(pool_mask, dtype=bool))
    n = min(int(rows), len(pool))
    if n <= 0:
        raise RuntimeError("EMPTY_TRAINING_POOL")
    positive = [
        np.flatnonzero(np.asarray(pool_mask, dtype=bool) & (weights[:, j] > 0.0))
        for j in range(weights.shape[1])
    ]
    rng = np.random.default_rng(int(seed))
    out = np.empty((int(steps), n), dtype=np.int32)
    for step in range(int(steps)):
        chosen = []
        used = set()
        for j in rng.permutation(weights.shape[1]).tolist():
            p = positive[int(j)]
            if len(p) == 0:
                continue
            order = rng.permutation(p)
            got = 0
            for x in order.tolist():
                ix = int(x)
                if ix in used:
                    continue
                chosen.append(ix)
                used.add(ix)
                got += 1
                if got >= int(per_target) or len(chosen) >= n:
                    break
            if len(chosen) >= n:
                break
        if len(chosen) < n:
            remain = np.asarray([int(x) for x in pool.tolist() if int(x) not in used], dtype=np.int64)
            need = n - len(chosen)
            chosen.extend(map(int, rng.choice(remain, size=need, replace=False).tolist()))
        out[step] = np.asarray(chosen[:n], dtype=np.int32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--experiment-prereg", required=True)
    ap.add_argument("--v6-source", required=True)
    ap.add_argument("--v6-prereg", required=True)
    ap.add_argument("--surface-json", required=True)
    ap.add_argument("--skeleton-json", required=True)
    ap.add_argument("--teacher-bank", required=True)
    ap.add_argument("--source-progress", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    arm = str(args.arm)
    experiment_prereg = Path(args.experiment_prereg).resolve()
    source_path = Path(args.v6_source).resolve()
    v6_prereg = Path(args.v6_prereg).resolve()
    bank_path = Path(args.teacher_bank).resolve()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    exp = json.loads(experiment_prereg.read_text())
    if exp.get("schema") != EXPERIMENT_SCHEMA or exp.get("status") != EXPECTED_EXPERIMENT_STATUS:
        raise RuntimeError("ARACHNE_2X2_PREREG_DRIFT")
    if sha256(source_path) != EXPECTED_V6_SOURCE_SHA256:
        raise RuntimeError("ARACHNE_2X2_V6_SOURCE_DRIFT")
    if sha256(v6_prereg) != EXPECTED_V6_PREREG_SHA256:
        raise RuntimeError("ARACHNE_2X2_V6_PREREG_DRIFT")
    if sha256(bank_path) != EXPECTED_BANK_SHA256:
        raise RuntimeError("ARACHNE_2X2_TEACHER_BANK_DRIFT")

    with np.load(bank_path, allow_pickle=False) as z:
        weights = np.asarray(z["weights"], dtype=np.float64)
        valid = np.asarray(z["teacher_valid_mask"], dtype=np.uint8).astype(bool)
    if weights.shape[0] != len(valid):
        raise RuntimeError("ARACHNE_2X2_BANK_SHAPE_DRIFT")

    mod = load_module(source_path)
    original_sample_schedule = mod._sample_schedule
    all_mask = np.ones_like(valid, dtype=bool)

    if arm == "UNIFORM_VALID":
        clean = np.flatnonzero(valid)
        sanity = original_sample_schedule(clean)
        got = schedule_digest(sanity)
        if got != EXPECTED_UNIFORM_VALID_SCHEDULE_SHA256:
            raise RuntimeError(f"ARACHNE_2X2_BASELINE_SCHEDULE_DRIFT::{got}")
        def patched(clean_arg):
            out = original_sample_schedule(np.asarray(clean_arg, dtype=np.int64))
            if schedule_digest(out) != EXPECTED_UNIFORM_VALID_SCHEDULE_SHA256:
                raise RuntimeError("ARACHNE_2X2_BASELINE_RUNTIME_SCHEDULE_DRIFT")
            return out
    elif arm == "ACTIVE_BALANCED_VALID":
        frozen = active_balanced_schedule(
            weights, valid,
            steps=int(mod.MAX_STEPS),
            rows=int(mod.ROWS_PER_STEP),
            seed=int(mod.DECODER_SEED),
            per_target=4,
        )
        def patched(clean_arg):
            del clean_arg
            return frozen.copy()
    elif arm == "UNIFORM_ALL_PROJECTED":
        frozen = uniform_schedule(
            np.flatnonzero(all_mask),
            steps=int(mod.MAX_STEPS),
            rows=int(mod.ROWS_PER_STEP),
            seed=int(mod.DECODER_SEED),
        )
        def patched(clean_arg):
            del clean_arg
            return frozen.copy()
    else:
        frozen = active_balanced_schedule(
            weights, all_mask,
            steps=int(mod.MAX_STEPS),
            rows=int(mod.ROWS_PER_STEP),
            seed=int(mod.DECODER_SEED),
            per_target=4,
        )
        def patched(clean_arg):
            del clean_arg
            return frozen.copy()

    mod._sample_schedule = patched

    schedule_preview = patched(np.flatnonzero(valid))
    manifest = {
        "schema": "RealSaS.KnightArachneV6CoverageSampling2x2Arm.v1",
        "status": "FROZEN_ARM_READY_BEFORE_OPTIMIZER_STEP_1",
        "arm": arm,
        "exact_v6_source_sha256": sha256(source_path),
        "exact_v6_prereg_sha256": sha256(v6_prereg),
        "teacher_bank_sha256": sha256(bank_path),
        "schedule_sha256": schedule_digest(schedule_preview),
        "schedule_shape": list(schedule_preview.shape),
        "teacher_valid_count": int(valid.sum()),
        "teacher_invalid_count": int((~valid).sum()),
        "training_pool_count": int(valid.sum()) if arm.endswith("_VALID") else int(len(valid)),
        "sampling_mode": "ACTIVE_BALANCED_K4_POSITIVE_SUPPORT" if arm.startswith("ACTIVE_") else "UNIFORM_WITHOUT_REPLACEMENT",
        "model_architecture_mutated": False,
        "objective_mutated": False,
        "compiler_gate_mutated": False,
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }
    (outdir / "ARACHNE_2X2_ARM_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("ARACHNE_2X2_ARM_READY=" + json.dumps(manifest, sort_keys=True), flush=True)

    run_args = SimpleNamespace(
        prereg=str(v6_prereg),
        surface_json=str(Path(args.surface_json).resolve()),
        skeleton_json=str(Path(args.skeleton_json).resolve()),
        teacher_bank=str(bank_path),
        source_progress=str(Path(args.source_progress).resolve()),
        output_dir=str(outdir),
    )
    result = mod.run(run_args)

    result_path = outdir / "ARACHNE_KNIGHT_V6_RESULT.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text())
        payload["coverage_sampling_2x2"] = {
            "arm": arm,
            "experiment_prereg_sha256": sha256(experiment_prereg),
            "schedule_sha256": manifest["schedule_sha256"],
            "sampling_mode": manifest["sampling_mode"],
            "training_pool_count": manifest["training_pool_count"],
        }
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("ARACHNE_2X2_ARM_COMPLETE=" + json.dumps({
        "arm": arm,
        "closure_step": result.get("closure_step"),
        "status": result.get("status"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
