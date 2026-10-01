from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if pred.shape != truth.shape or len(mask) != len(pred):
        raise RuntimeError("ARACHNE_2X2_METRIC_SHAPE_DRIFT")
    p = pred[mask]
    t = truth[mask]
    if len(p) == 0:
        raise RuntimeError("ARACHNE_2X2_EMPTY_METRIC_MASK")
    l1 = np.abs(p - t).sum(1)
    return {
        "row_count": int(len(l1)),
        "row_l1_mean": float(l1.mean()),
        "row_l1_p50": float(np.quantile(l1, 0.50)),
        "row_l1_p90": float(np.quantile(l1, 0.90)),
        "row_l1_p95": float(np.quantile(l1, 0.95)),
        "row_l1_p99": float(np.quantile(l1, 0.99)),
        "row_l1_max": float(l1.max()),
        "row_l1_gt_0p1": int(np.count_nonzero(l1 > 0.1)),
        "row_l1_gt_0p5": int(np.count_nonzero(l1 > 0.5)),
        "row_l1_gt_1": int(np.count_nonzero(l1 > 1.0)),
        "row_l1_gt_1p5": int(np.count_nonzero(l1 > 1.5)),
        "dominant_joint_accuracy": float(np.mean(np.argmax(p, 1) == np.argmax(t, 1))),
    }


def load_arm(path: Path, bank: dict[str, np.ndarray]) -> dict:
    result_path = path / "ARACHNE_KNIGHT_V6_RESULT.json"
    weights_path = path / "ARACHNE_KNIGHT_V6_CANONICAL_SKIN_WEIGHTS_F64.npz"
    if not result_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"ARACHNE_2X2_ARM_OUTPUT_MISSING::{path}")

    result = json.loads(result_path.read_text())
    with np.load(weights_path, allow_pickle=False) as z:
        pred = np.asarray(z["weights"], dtype=np.float64)
        surface_ids = np.asarray(z["surface_ids"]).astype(str)

    bank_ids = np.asarray(bank["surface_ids"]).astype(str)
    lookup = {sid: i for i, sid in enumerate(bank_ids.tolist())}
    try:
        bank_index = np.asarray([lookup[sid] for sid in surface_ids.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"ARACHNE_2X2_SURFACE_ID_ALIGNMENT_FAIL::{exc}") from exc

    mapping = np.asarray(
        result["teacher_binding"]["canonical_joint_to_target_index"],
        dtype=np.int64,
    )
    truth = np.asarray(bank["weights"], dtype=np.float64)[bank_index][:, mapping]
    valid = np.asarray(bank["teacher_valid_mask"], dtype=np.uint8).astype(bool)[bank_index]

    if pred.shape != truth.shape:
        raise RuntimeError(f"ARACHNE_2X2_ARM_WEIGHT_SHAPE_DRIFT::{pred.shape}::{truth.shape}")

    return {
        "status": result.get("status"),
        "closure_step": result.get("closure_step"),
        "terminal_check_steps": result.get("terminal_check_steps"),
        "wall_seconds": result.get("wall_seconds"),
        "final_valid_gate": result.get("final_evaluation", {}).get("full_gate"),
        "result_sha256": sha256(result_path),
        "weights_sha256": sha256(weights_path),
        "valid": metrics(pred, truth, valid),
        "invalid": metrics(pred, truth, ~valid),
        "all": metrics(pred, truth, np.ones(len(valid), dtype=bool)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-bank", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    with np.load(args.teacher_bank, allow_pickle=False) as z:
        bank = {k: np.asarray(z[k]) for k in z.files}

    arms = (
        "UNIFORM_VALID",
        "ACTIVE_BALANCED_VALID",
        "UNIFORM_ALL_PROJECTED",
        "ACTIVE_BALANCED_ALL_PROJECTED",
    )
    rows = {}
    missing = []
    for arm in arms:
        path = args.root / arm
        try:
            rows[arm] = load_arm(path, bank)
        except RuntimeError as exc:
            if not args.allow_partial:
                raise
            missing.append({"arm": arm, "error": str(exc)})

    report = {
        "schema": "RealSaS.KnightArachneV6CoverageSampling2x2ResultAudit.v1",
        "status": "COMPLETE" if len(rows) == len(arms) else "PARTIAL",
        "teacher_bank_sha256": sha256(args.teacher_bank),
        "arms": rows,
        "missing": missing,
        "decision_boundary": [
            "No arm is selected from training loss or the historical valid-only science gate.",
            "Teacher-invalid row metrics are diagnostic because projected invalid rows are not product teacher authority.",
            "Any promotion requires downstream topology/mechanical replay on the exact arm weights.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("ARACHNE_V6_2X2_RESULT_AUDIT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
