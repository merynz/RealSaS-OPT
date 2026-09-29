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
EXPECTED_CONDITIONING_AUTHORITY_SHA256 = "a0a3e652e323c6f2042354eb2095f3f66ff6168bf5ab51b53c83f728b8a00499"
EXPECTED_CANONICAL_TO_TARGET_INDEX = np.asarray([6,27,8,0,12,15,5,3,16,18,1,19,7,13,20,11,23,25,24,22,26,21,9,10,17,4,14,2], dtype=np.int64)
EXPECTED_UNIFORM_VALID_SCHEDULE_SHA256 = "4c5731cf5b2b28a3985458d555c7ef545d78054523cf92ce3edb0e959eee82fb"
EXPECTED_MAX_STEPS = 32768
EXPECTED_ROWS_PER_STEP = 384
EXPECTED_DECODER_SEED = 20261128
EXPECTED_SCHEDULES = {
    "UNIFORM_VALID": {
        "full_32768": "4c5731cf5b2b28a3985458d555c7ef545d78054523cf92ce3edb0e959eee82fb",
        "prefix_8192": "7c9a91a2e8051d3ff1ce534ccf908d5a0b25ca0620f15f0ab04efdfb228fa5e5",
    },
    "ACTIVE_BALANCED_VALID": {
        "full_32768": "1c055e2ff76b65ead96094eaac613efbeeba1f552488c144d98d0d2bce13a308",
        "prefix_8192": "5f8216c0e758d8d6830294ea765308ce66ee3648dc72335c6fba79a70fb0b934",
    },
    "UNIFORM_ALL_PROJECTED": {
        "full_32768": "f2943931d70134509fbd24826ecd11388ccb5f4ce29a1d801626f6ffcd9ced4b",
        "prefix_8192": "c1575c6e6260753b68e055873295b5c74da6486fc1911bcd61fef1e626098aa4",
    },
    "ACTIVE_BALANCED_ALL_PROJECTED": {
        "full_32768": "24eaa6ab7faccf1fdff6bd2fc78b369f7ad19ca7841ef65a9721c775bc597a88",
        "prefix_8192": "a2a0725af466f4af85bd15faac1236736c7876bd7146967378c80390120ee018",
    },
}
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
    decoder_seed: int,
    per_target: int = 4,
) -> np.ndarray:
    """Exact K4 exposure schedule used by the sealed sampler court."""
    weights = np.asarray(weights, dtype=np.float64)
    pool_mask = np.asarray(pool_mask, dtype=bool)
    pool = np.flatnonzero(pool_mask).astype(np.int64)
    n = min(int(rows), len(pool))
    if n <= 0:
        raise RuntimeError("EMPTY_TRAINING_POOL")
    positive = [
        np.flatnonzero(pool_mask & (weights[:, j] > 0.0)).astype(np.int64)
        for j in range(weights.shape[1])
    ]
    rng = np.random.default_rng(int(decoder_seed) + 1000 + int(per_target))
    out = np.empty((int(steps), n), dtype=np.int32)
    used_mask = np.zeros(len(weights), dtype=bool)
    for step in range(int(steps)):
        chosen = []
        touched = []
        for j in rng.permutation(weights.shape[1]):
            candidates = positive[int(j)]
            if len(candidates) == 0:
                continue
            got = 0
            for x in rng.permutation(candidates):
                ix = int(x)
                if used_mask[ix]:
                    continue
                chosen.append(ix)
                used_mask[ix] = True
                touched.append(ix)
                got += 1
                if got >= int(per_target) or len(chosen) >= n:
                    break
            if len(chosen) >= n:
                break
        if len(chosen) < n:
            remain = pool[~used_mask[pool]]
            need = n - len(chosen)
            chosen.extend(map(int, rng.choice(remain, size=need, replace=False).tolist()))
        out[step] = np.asarray(chosen[:n], dtype=np.int32)
        if touched:
            used_mask[np.asarray(touched, dtype=np.int64)] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--experiment-prereg", required=True)
    ap.add_argument("--v6-source", required=True)
    ap.add_argument("--v6-prereg", required=True)
    ap.add_argument("--surface-json")
    ap.add_argument("--skeleton-json")
    ap.add_argument("--teacher-bank", required=True)
    ap.add_argument("--conditioning-authority-npz", required=True)
    ap.add_argument("--source-progress")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    arm = str(args.arm)
    experiment_prereg = Path(args.experiment_prereg).resolve()
    source_path = Path(args.v6_source).resolve()
    v6_prereg = Path(args.v6_prereg).resolve()
    bank_path = Path(args.teacher_bank).resolve()
    conditioning_authority_path = Path(args.conditioning_authority_npz).resolve()
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
    if sha256(conditioning_authority_path) != EXPECTED_CONDITIONING_AUTHORITY_SHA256:
        raise RuntimeError("ARACHNE_2X2_CONDITIONING_AUTHORITY_DRIFT")

    with np.load(bank_path, allow_pickle=False) as z:
        weights0 = np.asarray(z["weights"], dtype=np.float64)
        valid0 = np.asarray(z["teacher_valid_mask"], dtype=np.uint8).astype(bool)
        bank_surface_ids = tuple(map(str, z["surface_ids"].tolist()))
    if weights0.shape[0] != len(valid0) or len(bank_surface_ids) != len(valid0):
        raise RuntimeError("ARACHNE_2X2_BANK_SHAPE_DRIFT")

    with np.load(conditioning_authority_path, allow_pickle=False) as z:
        authority_surface_ids = tuple(map(str, z["surface_ids"].tolist()))
        authority_joint_ids = tuple(map(str, z["canonical_joint_ids"].tolist()))
    if len(authority_surface_ids) != len(bank_surface_ids) or len(authority_joint_ids) != weights0.shape[1]:
        raise RuntimeError("ARACHNE_2X2_CONDITIONING_AUTHORITY_SHAPE_DRIFT")
    row_of = {sid: i for i, sid in enumerate(bank_surface_ids)}
    if len(row_of) != len(bank_surface_ids) or any(sid not in row_of for sid in authority_surface_ids):
        raise RuntimeError("ARACHNE_2X2_CONDITIONING_SURFACE_BINDING_DRIFT")
    rows = np.asarray([row_of[sid] for sid in authority_surface_ids], dtype=np.int64)
    if sorted(rows.tolist()) != list(range(len(rows))):
        raise RuntimeError("ARACHNE_2X2_CONDITIONING_ROW_PERMUTATION_DRIFT")
    cols = EXPECTED_CANONICAL_TO_TARGET_INDEX.copy()
    if sorted(cols.tolist()) != list(range(weights0.shape[1])):
        raise RuntimeError("ARACHNE_2X2_CONDITIONING_COLUMN_PERMUTATION_DRIFT")

    weights = np.maximum(weights0[rows][:, cols], 0.0)
    sums = weights.sum(1, keepdims=True)
    if np.any(sums <= 1e-12):
        raise RuntimeError("ARACHNE_2X2_TEACHER_ZERO_ROW_AFTER_REINDEX")
    weights /= sums
    valid = valid0[rows]
    raw_vs_reindexed_valid_diff = int(np.count_nonzero(valid0 != valid))
    moved_rows = int(np.count_nonzero(rows != np.arange(len(rows), dtype=np.int64)))
    all_mask = np.ones_like(valid, dtype=bool)

    if arm == "UNIFORM_VALID":
        frozen = uniform_schedule(
            np.flatnonzero(valid),
            steps=EXPECTED_MAX_STEPS,
            rows=EXPECTED_ROWS_PER_STEP,
            seed=EXPECTED_DECODER_SEED,
        )
    elif arm == "ACTIVE_BALANCED_VALID":
        frozen = active_balanced_schedule(
            weights,
            valid,
            steps=EXPECTED_MAX_STEPS,
            rows=EXPECTED_ROWS_PER_STEP,
            decoder_seed=EXPECTED_DECODER_SEED,
            per_target=4,
        )
    elif arm == "UNIFORM_ALL_PROJECTED":
        frozen = uniform_schedule(
            np.flatnonzero(all_mask),
            steps=EXPECTED_MAX_STEPS,
            rows=EXPECTED_ROWS_PER_STEP,
            seed=EXPECTED_DECODER_SEED,
        )
    else:
        frozen = active_balanced_schedule(
            weights,
            all_mask,
            steps=EXPECTED_MAX_STEPS,
            rows=EXPECTED_ROWS_PER_STEP,
            decoder_seed=EXPECTED_DECODER_SEED,
            per_target=4,
        )

    def patched(clean_arg):
        del clean_arg
        return frozen.copy()


    schedule_preview = patched(np.flatnonzero(valid))
    schedule_sha = schedule_digest(schedule_preview)
    schedule_prefix_sha = schedule_digest(schedule_preview[:8192])
    expected_schedule = EXPECTED_SCHEDULES[arm]
    if schedule_sha != expected_schedule["full_32768"]:
        raise RuntimeError(f"ARACHNE_2X2_FULL_SCHEDULE_SHA_DRIFT::{arm}::{schedule_sha}")
    if schedule_prefix_sha != expected_schedule["prefix_8192"]:
        raise RuntimeError(f"ARACHNE_2X2_PREFIX_SCHEDULE_SHA_DRIFT::{arm}::{schedule_prefix_sha}")
    manifest = {
        "schema": "RealSaS.KnightArachneV6CoverageSampling2x2Arm.v1",
        "status": "FROZEN_ARM_READY_BEFORE_OPTIMIZER_STEP_1",
        "arm": arm,
        "exact_v6_source_sha256": sha256(source_path),
        "exact_v6_prereg_sha256": sha256(v6_prereg),
        "teacher_bank_sha256": sha256(bank_path),
        "conditioning_authority_sha256": sha256(conditioning_authority_path),
        "teacher_row_reindexed": True,
        "teacher_column_reindexed": True,
        "conditioning_row_permutation_moved_count": moved_rows,
        "raw_vs_reindexed_valid_mask_diff_count": raw_vs_reindexed_valid_diff,
        "schedule_sha256": schedule_sha,
        "schedule_prefix8192_sha256": schedule_prefix_sha,
        "schedule_shape": list(schedule_preview.shape),
        "teacher_valid_count": int(valid.sum()),
        "teacher_invalid_count": int((~valid).sum()),
        "training_pool_count": int(valid.sum()) if arm.endswith("_VALID") else int(len(valid)),
        "sampling_mode": "ACTIVE_BALANCED_K4_POSITIVE_SUPPORT" if arm.startswith("ACTIVE_") else "UNIFORM_WITHOUT_REPLACEMENT",
        "model_architecture_mutated": False,
        "objective_mutated": False,
        "compiler_gate_mutated": False,
        "evaluation_teacher_valid_mask_mutated": False,
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }
    (outdir / "ARACHNE_2X2_ARM_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("ARACHNE_2X2_ARM_READY=" + json.dumps(manifest, sort_keys=True), flush=True)

    if args.preflight_only:
        print("ARACHNE_2X2_PREFLIGHT_ONLY_PASS", flush=True)
        return

    mod = load_module(source_path)
    if int(mod.MAX_STEPS) != EXPECTED_MAX_STEPS:
        raise RuntimeError("ARACHNE_2X2_MAX_STEPS_DRIFT")
    if int(mod.ROWS_PER_STEP) != EXPECTED_ROWS_PER_STEP:
        raise RuntimeError("ARACHNE_2X2_ROWS_PER_STEP_DRIFT")
    if int(mod.DECODER_SEED) != EXPECTED_DECODER_SEED:
        raise RuntimeError("ARACHNE_2X2_DECODER_SEED_DRIFT")
    mod._sample_schedule = patched

    for raw, name in (
        (args.surface_json, "surface-json"),
        (args.skeleton_json, "skeleton-json"),
        (args.source_progress, "source-progress"),
    ):
        if not raw:
            raise RuntimeError(f"ARACHNE_2X2_REQUIRED_ARG_MISSING::{name}")

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
