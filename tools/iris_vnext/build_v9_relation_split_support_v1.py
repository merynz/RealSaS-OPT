from __future__ import annotations

"""Build deterministic TRAIN/HELD-OUT support evidence for IRIS V9 relation truth.

This tool does not train. It replays the exact V9 base-pair split over immutable
Relation Truth V2 sidecars, keeps both orientations of every two-hit pair in the
same partition, and records binary support for the optimized/diagnostic relation
families before any optimizer step may execute.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.cohort_v9 import (
    V9_HOLDOUT_FRACTION,
    V9_SPLIT_POLICY,
    V9_SPLIT_SEED,
    _pair_rank,
)
from models.iris.vnext.cohort_v9_balanced_fp32 import (
    V9_BOUNDARY_FAMILIES,
    V9_METRIC_ONLY_FAMILIES,
    V9_OPTIMIZED_FAMILIES,
    V9_ORDERING_FAMILIES,
    V9_SUPERVISION_POLICY,
)


SCHEMA = "RealSaS.IRISVNextV9RelationSplitSupport.v1"
RELATION_SCHEMA = "RealSaS.IRISVNextFit8RelationTruthManifest.v2"
SIDECAR_SCHEMA = "RealSaS.IRISVNextFit8RelationTruth.v2"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"VNEXT_V9_SPLIT_SUPPORT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha40(value: str) -> str:
    out = value.strip().lower()
    if len(out) != 40 or any(c not in "0123456789abcdef" for c in out):
        raise ValueError("VNEXT_V9_SPLIT_SUPPORT_SOURCE_SHA_INVALID")
    return out


def _rows_from_pairs(pair_indices: torch.Tensor) -> torch.Tensor:
    rows = torch.empty(len(pair_indices) * 2, dtype=torch.long)
    rows[0::2] = pair_indices * 2
    rows[1::2] = pair_indices * 2 + 1
    return rows


def _partition_pair_indices(
    subject_id: str, view_index: int, pair_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if pair_count < 2:
        raise ValueError("VNEXT_V9_SPLIT_SUPPORT_PAIR_COUNT_TOO_SMALL")
    ranked = sorted(range(pair_count), key=lambda i: _pair_rank(subject_id, view_index, i))
    holdout_count = int(round(pair_count * V9_HOLDOUT_FRACTION))
    holdout_count = max(1, min(pair_count - 1, holdout_count))
    held = torch.tensor(sorted(ranked[:holdout_count]), dtype=torch.long)
    train = torch.tensor(sorted(ranked[holdout_count:]), dtype=torch.long)
    if set(train.tolist()).intersection(held.tolist()):
        raise RuntimeError("VNEXT_V9_SPLIT_SUPPORT_GROUP_LEAKAGE")
    if len(train) + len(held) != pair_count:
        raise RuntimeError("VNEXT_V9_SPLIT_SUPPORT_PARTITION_INCOMPLETE")
    return train, held


def _binary_counts(
    target: torch.Tensor, valid: torch.Tensor, rows: torch.Tensor
) -> dict[str, int]:
    t = torch.as_tensor(target).reshape(-1)[rows].float()
    v = torch.as_tensor(valid, dtype=torch.bool).reshape(-1)[rows]
    values = t[v]
    if len(values) and not bool(((values == 0.0) | (values == 1.0)).all()):
        raise ValueError("VNEXT_V9_SPLIT_SUPPORT_NONBINARY_TARGET")
    positive = int((values == 1.0).sum().item())
    return {
        "valid": int(len(values)),
        "positive": positive,
        "negative": int(len(values) - positive),
    }


def _zero_counts() -> dict[str, int]:
    return {"valid": 0, "positive": 0, "negative": 0}


def _add(dst: dict[str, int], src: Mapping[str, int]) -> None:
    for key in ("valid", "positive", "negative"):
        dst[key] += int(src[key])


def _binary_eligible(row: Mapping[str, int]) -> bool:
    return (
        int(row["valid"]) > 0
        and int(row["positive"]) > 0
        and int(row["negative"]) > 0
    )


def build_split_support(
    *,
    expected_source_sha: str,
    relation_root: Path,
    relation_manifest_path: Path,
) -> dict[str, Any]:
    source = _sha40(expected_source_sha)
    relation_root = relation_root.resolve()
    manifest_path = relation_manifest_path.resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != RELATION_SCHEMA:
        raise ValueError("VNEXT_V9_SPLIT_SUPPORT_MANIFEST_SCHEMA_INVALID")
    if tuple(manifest.get("subject_order") or ()) != SUBJECT_IDS:
        raise ValueError("VNEXT_V9_SPLIT_SUPPORT_SUBJECT_ORDER_DRIFT")
    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("VNEXT_V9_SPLIT_SUPPORT_SUBJECT_ROW_ORDER_DRIFT")

    families = V9_OPTIMIZED_FAMILIES + V9_METRIC_ONLY_FAMILIES
    global_counts = {
        "train": {name: _zero_counts() for name in families},
        "held_out": {name: _zero_counts() for name in families},
    }
    per_view = {
        str(v): {
            "train": {name: _zero_counts() for name in families},
            "held_out": {name: _zero_counts() for name in families},
        }
        for v in range(8)
    }
    per_subject_view: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    total_pair_groups = 0
    total_train_pair_groups = 0
    total_held_pair_groups = 0

    for subject_index, row in enumerate(rows):
        subject_id = SUBJECT_IDS[subject_index]
        path = (relation_root / str(row["file"])).resolve()
        if not path.is_relative_to(relation_root):
            raise ValueError("VNEXT_V9_SPLIT_SUPPORT_SIDECAR_PATH_ESCAPE")
        if _sha(path) != str(row["sha256"]):
            raise ValueError("VNEXT_V9_SPLIT_SUPPORT_SIDECAR_SHA_DRIFT:" + subject_id)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != SIDECAR_SCHEMA or payload.get("subject_id") != subject_id:
            raise ValueError("VNEXT_V9_SPLIT_SUPPORT_SIDECAR_IDENTITY_DRIFT:" + subject_id)
        views = list(payload.get("views") or ())
        if len(views) != 8:
            raise ValueError("VNEXT_V9_SPLIT_SUPPORT_VIEW_COUNT_DRIFT:" + subject_id)
        per_subject_view[subject_id] = {}

        for view_index, view in enumerate(views):
            point_a = view.get("point_a")
            point_b = view.get("point_b")
            if not isinstance(point_a, torch.Tensor) or not isinstance(point_b, torch.Tensor):
                raise ValueError("VNEXT_V9_SPLIT_SUPPORT_POINTS_REQUIRED")
            if point_a.shape != point_b.shape or point_a.ndim != 2 or point_a.shape[-1] != 3:
                raise ValueError("VNEXT_V9_SPLIT_SUPPORT_POINT_SHAPE_INVALID")
            oriented_rows = int(point_a.shape[0])
            if oriented_rows < 4 or oriented_rows % 2:
                raise ValueError("VNEXT_V9_SPLIT_SUPPORT_ORIENTED_ROW_COUNT_INVALID")
            if not torch.equal(point_a[0::2], point_b[1::2]) or not torch.equal(
                point_b[0::2], point_a[1::2]
            ):
                raise ValueError(
                    f"VNEXT_V9_SPLIT_SUPPORT_ORIENTATION_DRIFT:{subject_id}:V{view_index}"
                )
            pair_count = oriented_rows // 2
            train_pairs, held_pairs = _partition_pair_indices(
                subject_id, view_index, pair_count
            )
            train_rows = _rows_from_pairs(train_pairs)
            held_rows = _rows_from_pairs(held_pairs)
            if set(train_rows.tolist()).intersection(held_rows.tolist()):
                raise RuntimeError("VNEXT_V9_SPLIT_SUPPORT_ORIENTED_ROW_LEAKAGE")

            view_truth = view.get("view_truth") or {}
            view_validity = view.get("view_validity") or {}
            cell = {
                "pair_groups": pair_count,
                "train_pair_groups": int(len(train_pairs)),
                "held_out_pair_groups": int(len(held_pairs)),
                "train": {},
                "held_out": {},
            }
            for family in families:
                if family not in view_truth or family not in view_validity:
                    raise ValueError("VNEXT_V9_SPLIT_SUPPORT_TARGET_MISSING:" + family)
                train_count = _binary_counts(
                    view_truth[family], view_validity[family], train_rows
                )
                held_count = _binary_counts(
                    view_truth[family], view_validity[family], held_rows
                )
                cell["train"][family] = train_count
                cell["held_out"][family] = held_count
                _add(global_counts["train"][family], train_count)
                _add(global_counts["held_out"][family], held_count)
                _add(per_view[str(view_index)]["train"][family], train_count)
                _add(per_view[str(view_index)]["held_out"][family], held_count)
            per_subject_view[subject_id][str(view_index)] = cell
            total_pair_groups += pair_count
            total_train_pair_groups += int(len(train_pairs))
            total_held_pair_groups += int(len(held_pairs))

    # Optimized families must be identifiable globally and for every aggregated
    # camera view in both TRAIN and HELD-OUT. Subject×view cells remain diagnostic.
    for family in V9_OPTIMIZED_FAMILIES:
        for partition in ("train", "held_out"):
            if not _binary_eligible(global_counts[partition][family]):
                blockers.append(f"GLOBAL_BINARY_SUPPORT_MISSING:{partition}:{family}")
            for view_index in range(8):
                if not _binary_eligible(per_view[str(view_index)][partition][family]):
                    blockers.append(
                        f"PER_VIEW_BINARY_SUPPORT_MISSING:{partition}:{family}:V{view_index}"
                    )

    for family in V9_METRIC_ONLY_FAMILIES:
        for partition in ("train", "held_out"):
            counts = global_counts[partition][family]
            if int(counts["positive"]) <= 0:
                blockers.append(
                    f"METRIC_ONLY_POSITIVE_SUPPORT_MISSING:{partition}:{family}"
                )
            if int(counts["negative"]) != 0:
                blockers.append(
                    "METRIC_ONLY_NEGATIVE_AUTHORITY_APPEARED__POLICY_REVIEW_REQUIRED:"
                    f"{partition}:{family}"
                )

    court_eligibility = {
        "ordering": all(
            _binary_eligible(global_counts[p][family])
            and all(
                _binary_eligible(per_view[str(v)][p][family]) for v in range(8)
            )
            for family in V9_ORDERING_FAMILIES
            for p in ("train", "held_out")
        ),
        "projected_alpha_boundary_probability": all(
            _binary_eligible(global_counts[p][V9_BOUNDARY_FAMILIES[0]])
            and all(
                _binary_eligible(per_view[str(v)][p][V9_BOUNDARY_FAMILIES[0]])
                for v in range(8)
            )
            for p in ("train", "held_out")
        ),
        "projected_appearance_boundary_probability": all(
            _binary_eligible(global_counts[p][V9_BOUNDARY_FAMILIES[1]])
            and all(
                _binary_eligible(per_view[str(v)][p][V9_BOUNDARY_FAMILIES[1]])
                for v in range(8)
            )
            for p in ("train", "held_out")
        ),
        "occlusion_probability": False,
    }

    return {
        "schema": SCHEMA,
        "status": (
            "PASS__SPLIT_SUPPORT_QUALIFIED__NO_TRAINING"
            if not blockers
            else "BLOCKED__SPLIT_SUPPORT_INSUFFICIENT"
        ),
        "source_commit_sha": source,
        "relation_truth_v2_manifest_sha256": _sha(manifest_path),
        "supervision_policy": V9_SUPERVISION_POLICY,
        "split": {
            "policy": V9_SPLIT_POLICY,
            "seed": V9_SPLIT_SEED,
            "holdout_fraction": V9_HOLDOUT_FRACTION,
            "orientation_grouped": True,
            "no_group_leakage": True,
        },
        "pair_groups": {
            "total": total_pair_groups,
            "train": total_train_pair_groups,
            "held_out": total_held_pair_groups,
        },
        "global": global_counts,
        "per_view": per_view,
        "per_subject_view": per_subject_view,
        "court_eligibility": court_eligibility,
        "subject_view_boundary_courts": "DIAGNOSTIC_ONLY__SPARSE_POSITIVES_ALLOWED",
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
    report = build_split_support(
        expected_source_sha=args.expected_source_sha,
        relation_root=args.relation_root,
        relation_manifest_path=args.relation_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "VNEXT_V9_RELATION_SPLIT_SUPPORT",
        f"status={report['status']}",
        f"blockers={len(report['blockers'])}",
        f"train_pairs={report['pair_groups']['train']}",
        f"heldout_pairs={report['pair_groups']['held_out']}",
        "training_steps=0",
    )
    if (
        args.require_pass
        and report["status"] != "PASS__SPLIT_SUPPORT_QUALIFIED__NO_TRAINING"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
