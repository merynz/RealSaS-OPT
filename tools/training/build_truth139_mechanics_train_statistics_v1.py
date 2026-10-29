#!/usr/bin/env python3
"""Build the tiny TRAIN-only mechanics statistics artifact used by Truth139.

This artifact is safe to hand to the sealed UNSEEN evaluator: it contains only two transformed
channel medians and two robust fixed scales, never train subject ids, row addresses, geometry,
features or target samples.  It makes the marginal-prior court exact without mounting TRAIN data
inside the UNSEEN campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TRAIN_SCHEMA = "RealSaS.IRISV31Truth139TrainManifest.v1"
OUTPUT_SCHEMA = "RealSaS.IRISV31Truth139MechanicsTrainStatistics.v1"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_partition(subject: str, count: int, seed: int, fraction: float) -> np.ndarray:
    """Exact mechanics-train side of presentation_partition_v1."""
    train, val = [], []
    for index in range(int(count)):
        token = f"{seed}|{subject}|mechanics|{index}".encode()
        bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(1 << 64)
        (train if bucket < float(fraction) else val).append(index)
    if count >= 2 and not val:
        val.append(train.pop())
    if count >= 2 and not train:
        train.append(val.pop())
    return np.asarray(train, dtype=np.int64)


def build_statistics(
    train_manifest: dict,
    *,
    partition_seed: int,
    train_fraction: float,
    minimum_scale: float,
) -> dict:
    if train_manifest.get("schema") != TRAIN_SCHEMA:
        raise ValueError("TRUTH139_MECHANICS_STATS_MANIFEST_SCHEMA")
    if train_manifest.get("contains_unseen_subject_ids") is not False or train_manifest.get("contains_unseen_file_refs") is not False:
        raise ValueError("TRUTH139_MECHANICS_STATS_UNSEEN_FIREWALL")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("TRUTH139_MECHANICS_STATS_PARTITION_FRACTION")
    if not np.isfinite(float(minimum_scale)) or float(minimum_scale) <= 0.0:
        raise ValueError("TRUTH139_MECHANICS_STATS_SCALE_FLOOR")

    values = []
    for row in train_manifest.get("subjects") or ():
        subject = str(row["subject_id"])
        ref = dict(row.get("presentation_targets") or {})
        path = Path(str(ref.get("path") or "")).resolve()
        if not path.is_file() or _file_hash(path) != str(ref.get("sha256") or ""):
            raise ValueError("TRUTH139_MECHANICS_STATS_TARGET_DRIFT:" + subject)
        with np.load(path, allow_pickle=False) as z:
            target = np.asarray(z["mechanical_target"], dtype=np.float64)
            valid = np.asarray(z["mechanical_valid"], dtype=np.bool_)
        if target.ndim != 2 or target.shape[1] != 2 or valid.shape != target.shape[:1]:
            raise ValueError("TRUTH139_MECHANICS_STATS_TARGET_LAYOUT:" + subject)
        valid_rows = np.flatnonzero(valid)
        local = _row_partition(subject, len(valid_rows), int(partition_seed), float(train_fraction))
        selected = valid_rows[local]
        if len(selected):
            raw = target[selected]
            if not np.isfinite(raw).all() or np.any(raw < 0.0):
                raise ValueError("TRUTH139_MECHANICS_STATS_TARGET_INVALID:" + subject)
            values.append(np.log1p(raw))
    if not values:
        raise ValueError("TRUTH139_MECHANICS_STATS_NO_TRAIN_ROWS")

    transformed = np.concatenate(values, axis=0)
    median = np.median(transformed, axis=0)
    mad = np.median(np.abs(transformed - median[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, float(minimum_scale))
    return {
        "schema": OUTPUT_SCHEMA,
        "split_identity_sha256": train_manifest["split_identity_sha256"],
        "train_manifest_identity_sha256": train_manifest["manifest_identity_sha256"],
        "target_transform": "LOG1P_NONNEGATIVE_MAGNITUDES",
        "partition_seed": int(partition_seed),
        "train_fraction": float(train_fraction),
        "train_row_count": int(len(transformed)),
        "channel_order": ["nonrigid_response", "response_energy"],
        "transformed_median": [float(x) for x in median],
        "fixed_scale": [float(x) for x in scale],
        "scale_estimator": "1P4826_MAD_WITH_FLOOR",
        "minimum_scale": float(minimum_scale),
        "contains_train_subject_ids": False,
        "contains_train_row_addresses": False,
        "contains_train_target_samples": False,
        "unseen_opened": False,
    }


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition-seed", type=int, default=261029139)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--minimum-scale", type=float, default=0.001)
    args = parser.parse_args()
    manifest = json.loads(Path(args.train_manifest).resolve().read_text())
    value = build_statistics(
        manifest,
        partition_seed=args.partition_seed,
        train_fraction=args.train_fraction,
        minimum_scale=args.minimum_scale,
    )
    output = Path(args.output).resolve()
    _write(output, value)
    print(json.dumps({
        "output": str(output),
        "train_row_count": value["train_row_count"],
        "fixed_scale": value["fixed_scale"],
        "transformed_median": value["transformed_median"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
