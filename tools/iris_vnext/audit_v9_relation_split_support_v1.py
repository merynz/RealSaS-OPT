from __future__ import annotations

"""Exact sidecar-level support court for the focused V9 relation experiment.

This audit runs before optimizer step 1.  It reconstructs the preregistered V9
pair-group split directly from the immutable Relation Truth V2 sidecars, rather
than trusting aggregate manifest statistics.  Optimized families must have both
classes in TRAIN and HELD-OUT support for every aggregate view.  Subject/view
cells remain diagnostic because rare boundaries can legitimately have zero
positives in one cell.  Positive-only occlusion is explicitly metric-only.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.cohort_v9 import (
    V9_HOLDOUT_FRACTION,
    V9_SPLIT_POLICY,
    V9_SPLIT_SEED,
    _pair_rank,
)
from models.iris.vnext.cohort_v9_balanced_fp32 import (
    V9_METRIC_ONLY_FAMILIES,
    V9_OPTIMIZED_FAMILIES,
    V9_SUPERVISION_POLICY,
)
from models.iris.vnext.graphics_truth_v2 import (
    STATIC_RELATION_KEYS_V2,
    VIEW_RELATION_KEYS_V2,
)


SCHEMA = "RealSaS.IRISVNextV9RelationSplitSupport.v1"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("VNEXT_V9_SUPPORT_JSON_OBJECT_REQUIRED")
    return value


def _sha40(value: str) -> str:
    out = str(value).strip().lower()
    if len(out) != 40 or any(c not in "0123456789abcdef" for c in out):
        raise ValueError("VNEXT_V9_SUPPORT_SOURCE_SHA_INVALID")
    return out


def _counts(target: torch.Tensor, valid: torch.Tensor) -> dict[str, int]:
    mask = valid.to(dtype=torch.bool)
    values = target.detach().float()[mask]
    if len(values) == 0:
        return {"valid": 0, "positive": 0, "negative": 0}
    if not bool(((values == 0.0) | (values == 1.0)).all()):
        raise ValueError("VNEXT_V9_SUPPORT_NONBINARY_TARGET")
    positive = int((values == 1.0).sum().item())
    return {"valid": int(len(values)), "positive": positive, "negative": int(len(values) - positive)}


def _add(dst: dict[str, int], src: dict[str, int]) -> None:
    for key in ("valid", "positive", "negative"):
        dst[key] += int(src[key])


def _empty_family_table(families: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {name: {"valid": 0, "positive": 0, "negative": 0} for name in families}


def audit_split_support(
    *,
    expected_source_sha: str,
    relation_root: Path,
    relation_manifest_path: Path,
) -> dict[str, Any]:
    source_sha = _sha40(expected_source_sha)
    root = relation_root.resolve()
    manifest_path = relation_manifest_path.resolve()
    manifest = _load_json(manifest_path)
    blockers: list[str] = []
    if manifest.get("schema") != "RealSaS.IRISVNextFit8RelationTruthManifest.v2":
        blockers.append("RELATION_MANIFEST_SCHEMA_INVALID")
    if tuple(manifest.get("subject_order") or ()) != SUBJECT_IDS:
        blockers.append("RELATION_SUBJECT_ORDER_DRIFT")

    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        blockers.append("RELATION_SUBJECT_ROWS_DRIFT")

    view_families = tuple(VIEW_RELATION_KEYS_V2)
    global_train = _empty_family_table(view_families)
    global_held = _empty_family_table(view_families)
    per_view: dict[str, Any] = {
        str(v): {
            "train": _empty_family_table(view_families),
            "held_out": _empty_family_table(view_families),
        }
        for v in range(8)
    }
    subject_view_diagnostics: list[dict[str, Any]] = []
    static_valid_total = {name: 0 for name in STATIC_RELATION_KEYS_V2}

    for subject_index, row in enumerate(rows):
        subject_id = SUBJECT_IDS[subject_index]
        path = (root / str(row.get("file") or "")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            blockers.append("RELATION_SIDECAR_PATH_INVALID:" + subject_id)
            continue
        if _sha(path) != str(row.get("sha256") or ""):
            blockers.append("RELATION_SIDECAR_SHA_DRIFT:" + subject_id)
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "RealSaS.IRISVNextFit8RelationTruth.v2" or payload.get("subject_id") != subject_id:
            blockers.append("RELATION_SIDECAR_IDENTITY_DRIFT:" + subject_id)
            continue
        views = list(payload.get("views") or ())
        if len(views) != 8:
            blockers.append("RELATION_VIEW_COUNT_DRIFT:" + subject_id)
            continue

        for view_index, view in enumerate(views):
            point_a = view.get("point_a")
            point_b = view.get("point_b")
            if not isinstance(point_a, torch.Tensor) or not isinstance(point_b, torch.Tensor):
                blockers.append(f"RELATION_POINTS_MISSING:{subject_id}:V{view_index}")
                continue
            row_count = int(point_a.shape[0])
            if point_a.shape != point_b.shape or point_a.ndim != 2 or row_count < 4 or row_count % 2:
                blockers.append(f"RELATION_POINT_SHAPE_INVALID:{subject_id}:V{view_index}")
                continue
            pair_count = row_count // 2
            if not torch.equal(point_a[0::2], point_b[1::2]) or not torch.equal(point_b[0::2], point_a[1::2]):
                blockers.append(f"RELATION_ORIENTATION_DRIFT:{subject_id}:V{view_index}")
                continue

            static_validity = view.get("static_validity") or {}
            for family in STATIC_RELATION_KEYS_V2:
                valid = torch.as_tensor(static_validity.get(family), dtype=torch.bool)
                if tuple(valid.shape) != (row_count,):
                    blockers.append(f"STATIC_VALIDITY_SHAPE_DRIFT:{subject_id}:V{view_index}:{family}")
                    continue
                count = int(valid.sum().item())
                static_valid_total[family] += count
                if count:
                    blockers.append(f"STATIC_RELATION_MUST_ABSTAIN:{subject_id}:V{view_index}:{family}")

            ranked = sorted(range(pair_count), key=lambda i: _pair_rank(subject_id, view_index, i))
            holdout_count = int(round(pair_count * V9_HOLDOUT_FRACTION))
            holdout_count = max(1, min(pair_count - 1, holdout_count))
            held_pairs = torch.tensor(sorted(ranked[:holdout_count]), dtype=torch.long)
            train_pairs = torch.tensor(sorted(ranked[holdout_count:]), dtype=torch.long)

            def rows_from_pairs(pair_indices: torch.Tensor) -> torch.Tensor:
                out = torch.empty(len(pair_indices) * 2, dtype=torch.long)
                out[0::2] = pair_indices * 2
                out[1::2] = pair_indices * 2 + 1
                return out

            train_rows = rows_from_pairs(train_pairs)
            held_rows = rows_from_pairs(held_pairs)
            view_truth = view.get("view_truth") or {}
            view_validity = view.get("view_validity") or {}
            diag = {"subject_id": subject_id, "view_index": view_index, "families": {}}
            for family in view_families:
                target = torch.as_tensor(view_truth.get(family))
                valid = torch.as_tensor(view_validity.get(family), dtype=torch.bool)
                if tuple(target.shape) != (row_count,) or tuple(valid.shape) != (row_count,):
                    blockers.append(f"VIEW_TARGET_SHAPE_DRIFT:{subject_id}:V{view_index}:{family}")
                    continue
                train_count = _counts(target[train_rows], valid[train_rows])
                held_count = _counts(target[held_rows], valid[held_rows])
                diag["families"][family] = {"train": train_count, "held_out": held_count}
                _add(global_train[family], train_count)
                _add(global_held[family], held_count)
                _add(per_view[str(view_index)]["train"][family], train_count)
                _add(per_view[str(view_index)]["held_out"][family], held_count)
            subject_view_diagnostics.append(diag)

    for family in STATIC_RELATION_KEYS_V2:
        if static_valid_total[family] != 0:
            blockers.append("STATIC_RELATION_GLOBAL_ABSTENTION_FAILED:" + family)

    for family in V9_OPTIMIZED_FAMILIES:
        for split_name, table in (("TRAIN", global_train), ("HELD_OUT", global_held)):
            row = table[family]
            if row["positive"] <= 0 or row["negative"] <= 0:
                blockers.append(f"OPTIMIZED_FAMILY_GLOBAL_BINARY_SUPPORT_MISSING:{split_name}:{family}")
        for view_index in range(8):
            for split_key, split_label in (("train", "TRAIN"), ("held_out", "HELD_OUT")):
                row = per_view[str(view_index)][split_key][family]
                if row["positive"] <= 0 or row["negative"] <= 0:
                    blockers.append(
                        f"OPTIMIZED_FAMILY_VIEW_BINARY_SUPPORT_MISSING:{split_label}:{family}:V{view_index}"
                    )

    for family in V9_METRIC_ONLY_FAMILIES:
        for split_name, table in (("TRAIN", global_train), ("HELD_OUT", global_held)):
            row = table[family]
            if row["positive"] <= 0:
                blockers.append(f"METRIC_ONLY_POSITIVE_SUPPORT_MISSING:{split_name}:{family}")
            if row["negative"] != 0:
                blockers.append(f"METRIC_ONLY_NEGATIVE_AUTHORITY_CHANGED__REVIEW_POLICY:{split_name}:{family}")

    zero_positive_cells: dict[str, dict[str, int]] = {}
    for family in V9_OPTIMIZED_FAMILIES:
        zero_positive_cells[family] = {"train": 0, "held_out": 0}
    for diag in subject_view_diagnostics:
        for family in V9_OPTIMIZED_FAMILIES:
            row = diag["families"].get(family) or {}
            for split_key in ("train", "held_out"):
                if int((row.get(split_key) or {}).get("positive", 0)) == 0:
                    zero_positive_cells[family][split_key] += 1

    status = "PASS__SPLIT_SUPPORT_QUALIFIED__NO_TRAINING" if not blockers else "BLOCKED"
    return {
        "schema": SCHEMA,
        "status": status,
        "source_commit_sha": source_sha,
        "relation_truth_v2_manifest_sha256": _sha(manifest_path),
        "split": {
            "policy": V9_SPLIT_POLICY,
            "seed": V9_SPLIT_SEED,
            "holdout_fraction": V9_HOLDOUT_FRACTION,
            "orientation_grouped": True,
        },
        "supervision_policy": V9_SUPERVISION_POLICY,
        "optimized_binary_families": list(V9_OPTIMIZED_FAMILIES),
        "metric_only_families": list(V9_METRIC_ONLY_FAMILIES),
        "static_abstention_families": list(STATIC_RELATION_KEYS_V2),
        "global": {"train": global_train, "held_out": global_held},
        "per_view_aggregate": per_view,
        "subject_view_zero_positive_cells": zero_positive_cells,
        "subject_view_policy": "DIAGNOSTIC_ONLY__NO_PER_CELL_BINARY_PASS_REQUIREMENT",
        "static_valid_total": static_valid_total,
        "blockers": blockers,
        "training_steps_executed": 0,
        "scientific_fit_started": False,
        "scientific_fit_pass_claimed": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--relation-root", type=Path, required=True)
    p.add_argument("--relation-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-pass", action="store_true")
    args = p.parse_args()
    report = audit_split_support(
        expected_source_sha=args.expected_source_sha,
        relation_root=args.relation_root,
        relation_manifest_path=args.relation_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        "VNEXT_V9_RELATION_SPLIT_SUPPORT",
        f"status={report['status']}",
        f"blockers={len(report['blockers'])}",
        "training_steps=0",
    )
    if args.require_pass and report["blockers"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())