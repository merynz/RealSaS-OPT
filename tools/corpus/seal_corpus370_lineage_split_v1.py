#!/usr/bin/env python3
from __future__ import annotations

"""Seal Corpus370 global product-unseen and model-specific TRAIN/VAL splits.

The global holdout is selected first at whole-lineage granularity and is forbidden
from every model's TRAIN/VAL pool. Model VAL sets are then selected independently
inside each model's remaining eligible lineage pool. The procedure is deterministic,
source-pack aware, and fail-closed when lineage review is unresolved or a holdout
would starve a small eligible roster. This artifact is a split seal only; it never
mints training readiness.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.Corpus370LineageSplit.v1"
LINEAGE_SCHEMA = "RealSaS.Corpus370LineageGroups.v1"
ROSTERS_SCHEMA = "RealSaS.Corpus370ModelRosters.v1"
POLICY_SCHEMA = "RealSaS.Corpus370LineageSplitPolicy.v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("CORPUS370_SPLIT_JSON_OBJECT")
    return value


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def stable_rank(seed: str, value: str) -> str:
    return hashlib.sha256((seed + "\0" + value).encode()).hexdigest()


def target_count(total: int, fraction: float, minimum: int, maximum: int | None = None) -> int:
    if total <= 0:
        return 0
    count = int(round(total * fraction))
    count = max(minimum, count)
    if maximum is not None:
        count = min(maximum, count)
    return min(total, count)


def lineage_maps(lineage: dict) -> tuple[dict[str, dict], dict[str, str]]:
    rows = lineage.get("groups") or ()
    require(len(rows) == int(lineage.get("lineage_count", -1)),
            "CORPUS370_SPLIT_LINEAGE_GROUP_CENSUS")
    groups = {str(row["lineage_id"]): dict(row) for row in rows}
    require(len(groups) == len(rows), "CORPUS370_SPLIT_LINEAGE_DUPLICATE_GROUP")
    subject_to_lineage = {str(k): str(v) for k, v in (lineage.get("subject_to_lineage") or {}).items()}
    require(len(subject_to_lineage) == int(lineage.get("subject_count", -1)) == 370, "CORPUS370_SPLIT_LINEAGE_SUBJECT_SET")
    require(set(subject_to_lineage.values()) == set(groups), "CORPUS370_SPLIT_LINEAGE_GROUP_REF")
    # A sealed map is not sufficient evidence of a lineage partition: its
    # reciprocal groups must cover each subject exactly once. Otherwise a
    # forged/stale group can silently split identical actors across partitions.
    seen = set()
    for lineage_id, group in groups.items():
        members = group.get("subject_ids")
        require(isinstance(members, list) and members
                and all(isinstance(sid, str) and sid for sid in members)
                and len(members) == len(set(members))
                and group.get("lineage_id") == lineage_id
                and group.get("subject_count") == len(members),
                "CORPUS370_SPLIT_LINEAGE_GROUP_MEMBERS")
        for sid in members:
            require(sid in subject_to_lineage
                    and subject_to_lineage[sid] == lineage_id
                    and sid not in seen,
                    "CORPUS370_SPLIT_LINEAGE_RECIPROCAL_PARTITION")
            seen.add(sid)
    require(seen == set(subject_to_lineage),
            "CORPUS370_SPLIT_LINEAGE_RECIPROCAL_COVERAGE")
    # Source-byte identity is a hard lineage link, not merely a publisher
    # stratification hint. Never allow an identical source SHA in two lineages:
    # that could put the exact same teacher asset in TRAIN and unseen.
    source_sha_owner = {}
    for lineage_id, group in groups.items():
        hashes = group.get("source_sha256")
        require(isinstance(hashes, list) and hashes
                and all(isinstance(h, str) and len(h) == 64
                        and all(c in "0123456789abcdef" for c in h)
                        for h in hashes)
                and len(hashes) == len(set(hashes)),
                "CORPUS370_SPLIT_LINEAGE_SOURCE_SHA_CENSUS")
        for source_sha in hashes:
            previous = source_sha_owner.setdefault(source_sha, lineage_id)
            require(previous == lineage_id,
                    "CORPUS370_SPLIT_SOURCE_SHA_CROSS_LINEAGE_LEAKAGE")
    return groups, subject_to_lineage


def model_eligible_lineages(rosters: dict, subject_to_lineage: dict[str, str]) -> dict[str, set[str]]:
    result = {}
    for model, payload in (rosters.get("models") or {}).items():
        ids = [str(row["subject_id"]) for row in payload.get("eligible_subjects") or ()]
        require(len(ids) == len(set(ids)),
                "CORPUS370_SPLIT_MODEL_DUPLICATE_SUBJECT:" + model)
        require(all(sid in subject_to_lineage for sid in ids), "CORPUS370_SPLIT_MODEL_UNKNOWN_SUBJECT:" + model)
        result[model] = {subject_to_lineage[sid] for sid in ids}
    return result


def primary_source_id(group: dict) -> str:
    values = sorted(str(x) for x in group.get("source_ids") or () if str(x))
    return values[0] if values else "__unknown__"


def select_global_unseen(
    groups: dict[str, dict],
    eligible_by_model: dict[str, set[str]],
    policy: dict,
) -> tuple[list[str], dict[str, Any]]:
    cfg = policy["global_unseen"]
    target = target_count(
        len(groups),
        float(cfg["target_lineage_fraction"]),
        int(cfg["minimum_lineage_count"]),
        int(cfg["maximum_lineage_count"]),
    )
    seed = str(cfg["seed"])
    min_remaining = int(cfg["minimum_remaining_model_eligible_lineages"])
    max_fraction = float(cfg["maximum_model_eligible_holdout_fraction"])

    eligible_counts = {model: len(values) for model, values in eligible_by_model.items()}
    holdout_model_counts = Counter()
    selected: list[str] = []
    selected_set: set[str] = set()
    source_counts = Counter()
    all_source_counts = Counter(primary_source_id(group) for group in groups.values())

    def allowed(lineage_id: str) -> bool:
        for model, eligible in eligible_by_model.items():
            if lineage_id not in eligible:
                continue
            total = eligible_counts[model]
            projected = holdout_model_counts[model] + 1
            remaining = total - projected
            if total <= min_remaining:
                return False
            if remaining < min_remaining:
                return False
            allowed_count = max(1, int(math.floor(total * max_fraction)))
            if projected > allowed_count:
                return False
        return True

    def add(lineage_id: str) -> bool:
        if lineage_id in selected_set or not allowed(lineage_id):
            return False
        selected.append(lineage_id)
        selected_set.add(lineage_id)
        source_counts[primary_source_id(groups[lineage_id])] += 1
        for model, eligible in eligible_by_model.items():
            if lineage_id in eligible:
                holdout_model_counts[model] += 1
        return True

    # Pass 1: source-pack breadth. One lineage from each source_id, prioritized
    # by broad model evaluability then stable seed. This makes the unseen set less
    # likely to collapse onto one prolific publisher pack.
    by_source = defaultdict(list)
    for lineage_id, group in groups.items():
        by_source[primary_source_id(group)].append(lineage_id)
    source_order = sorted(
        by_source,
        key=lambda source: (
            all_source_counts[source],
            stable_rank(seed + ":source", source),
        ),
    )
    for source in source_order:
        candidates = sorted(
            by_source[source],
            key=lambda lineage_id: (
                -sum(lineage_id in values for values in eligible_by_model.values()),
                stable_rank(seed, lineage_id),
            ),
        )
        for lineage_id in candidates:
            if add(lineage_id):
                break
        if len(selected) >= target:
            break

    # Pass 2: fill target while balancing current source holdout rate.
    while len(selected) < target:
        candidates = [lineage_id for lineage_id in groups if lineage_id not in selected_set and allowed(lineage_id)]
        if not candidates:
            break
        candidates.sort(
            key=lambda lineage_id: (
                source_counts[primary_source_id(groups[lineage_id])] / max(1, all_source_counts[primary_source_id(groups[lineage_id])]),
                -sum(lineage_id in values for values in eligible_by_model.values()),
                stable_rank(seed, lineage_id),
            )
        )
        add(candidates[0])

    require(len(selected) == target, "CORPUS370_GLOBAL_UNSEEN_POLICY_UNSATISFIED")
    diagnostics = {
        "target_lineage_count": target,
        "selected_lineage_count": len(selected),
        "source_id_counts": dict(sorted(source_counts.items())),
        "eligible_lineage_counts_by_model": dict(sorted(eligible_counts.items())),
        "heldout_eligible_lineage_counts_by_model": dict(sorted(holdout_model_counts.items())),
    }
    return sorted(selected), diagnostics


def select_model_val(
    model: str,
    dev_lineages: set[str],
    groups: dict[str, dict],
    policy: dict,
) -> set[str]:
    cfg = policy["model_validation"]
    if not dev_lineages:
        return set()
    minimum = int(cfg["minimum_lineage_count_when_eligible_ge_10"]) if len(dev_lineages) >= 10 else 0
    target = target_count(len(dev_lineages), float(cfg["target_lineage_fraction"]), minimum)
    if target == 0:
        return set()
    seed = str(cfg["seed_prefix"]) + ":" + model
    selected: set[str] = set()
    source_counts = Counter()
    all_source_counts = Counter(primary_source_id(groups[lineage_id]) for lineage_id in dev_lineages)
    by_source = defaultdict(list)
    for lineage_id in dev_lineages:
        by_source[primary_source_id(groups[lineage_id])].append(lineage_id)

    for source in sorted(by_source, key=lambda value: (all_source_counts[value], stable_rank(seed + ":source", value))):
        if len(selected) >= target:
            break
        candidate = min(by_source[source], key=lambda lineage_id: stable_rank(seed, lineage_id))
        selected.add(candidate)
        source_counts[source] += 1

    while len(selected) < target:
        candidates = [lineage_id for lineage_id in dev_lineages if lineage_id not in selected]
        require(bool(candidates), "CORPUS370_MODEL_VAL_POLICY_UNSATISFIED:" + model)
        candidates.sort(
            key=lambda lineage_id: (
                source_counts[primary_source_id(groups[lineage_id])] / max(1, all_source_counts[primary_source_id(groups[lineage_id])]),
                stable_rank(seed, lineage_id),
            )
        )
        chosen = candidates[0]
        selected.add(chosen)
        source_counts[primary_source_id(groups[chosen])] += 1
    return selected


def build(lineage_path: Path, rosters_path: Path, policy_path: Path) -> dict:
    lineage = load(lineage_path)
    rosters = load(rosters_path)
    policy = load(policy_path)
    require(lineage.get("schema") == LINEAGE_SCHEMA, "CORPUS370_SPLIT_LINEAGE_SCHEMA")
    require(lineage.get("status") == "PASS_LINEAGE_GROUPS_SEALED" and lineage.get("split_authorized") is True, "CORPUS370_SPLIT_LINEAGE_NOT_SEALED")
    require(int(lineage.get("review_required_count", -1)) == 0, "CORPUS370_SPLIT_LINEAGE_REVIEW_OPEN")
    require(rosters.get("schema") == ROSTERS_SCHEMA and rosters.get("training_authorized") is False, "CORPUS370_SPLIT_ROSTERS_SCHEMA")
    require(policy.get("schema") == POLICY_SCHEMA and policy.get("training_authorized") is False, "CORPUS370_SPLIT_POLICY_SCHEMA")

    groups, subject_to_lineage = lineage_maps(lineage)
    eligible_by_model = model_eligible_lineages(rosters, subject_to_lineage)
    global_lineages, global_diag = select_global_unseen(groups, eligible_by_model, policy)
    global_set = set(global_lineages)
    global_subjects = sorted(sid for sid, lineage_id in subject_to_lineage.items() if lineage_id in global_set)

    models = {}
    for model, model_payload in sorted((rosters.get("models") or {}).items()):
        eligible_subjects = {str(row["subject_id"]): row for row in model_payload.get("eligible_subjects") or ()}
        eligible_lineages = eligible_by_model[model]
        dev_lineages = eligible_lineages - global_set
        val_lineages = select_model_val(model, dev_lineages, groups, policy)
        train_lineages = dev_lineages - val_lineages
        train_subjects = sorted(sid for sid in eligible_subjects if subject_to_lineage[sid] in train_lineages)
        val_subjects = sorted(sid for sid in eligible_subjects if subject_to_lineage[sid] in val_lineages)
        unseen_subjects = sorted(sid for sid in eligible_subjects if subject_to_lineage[sid] in global_set)
        require(not (set(train_subjects) & set(val_subjects)), "CORPUS370_SPLIT_MODEL_TRAIN_VAL_OVERLAP:" + model)
        require(not ((set(train_subjects) | set(val_subjects)) & set(global_subjects)), "CORPUS370_SPLIT_GLOBAL_LEAKAGE:" + model)
        require(train_lineages.isdisjoint(val_lineages) and train_lineages.isdisjoint(global_set) and val_lineages.isdisjoint(global_set), "CORPUS370_SPLIT_LINEAGE_LEAKAGE:" + model)
        models[model] = {
            "eligible_subject_count": len(eligible_subjects),
            "eligible_lineage_count": len(eligible_lineages),
            "train_lineage_ids": sorted(train_lineages),
            "val_lineage_ids": sorted(val_lineages),
            "global_unseen_lineage_ids": sorted(eligible_lineages & global_set),
            "train_subject_ids": train_subjects,
            "val_subject_ids": val_subjects,
            "global_unseen_subject_ids": unseen_subjects,
            "train_subject_count": len(train_subjects),
            "val_subject_count": len(val_subjects),
            "global_unseen_subject_count": len(unseen_subjects),
        }

    return {
        "schema": SCHEMA,
        "status": "PASS_GLOBAL_PRODUCT_UNSEEN_AND_MODEL_TRAIN_VAL_SPLIT_SEALED",
        "subject_count": 370,
        "lineage_count": len(groups),
        "global_unseen": {
            "lineage_ids": global_lineages,
            "subject_ids": global_subjects,
            "lineage_count": len(global_lineages),
            "subject_count": len(global_subjects),
            "diagnostics": global_diag,
        },
        "models": models,
        "parents": {
            "lineage_groups": {"path": str(lineage_path.resolve()), "sha256": sha256(lineage_path)},
            "model_rosters": {"path": str(rosters_path.resolve()), "sha256": sha256(rosters_path)},
            "split_policy": {"path": str(policy_path.resolve()), "sha256": sha256(policy_path)},
        },
        "rules": {
            "global_unseen_isolated_before_model_train_val": True,
            "whole_lineage_assignment": True,
            "cross_model_global_unseen_leakage": False,
            "historical_truth139_split_inherited": False,
        },
        "split_authorized": True,
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-groups", type=Path, required=True)
    parser.add_argument("--model-rosters", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path(__file__).resolve().parents[2] / "canonical/CORPUS370_LINEAGE_SPLIT_POLICY_V1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.lineage_groups, args.model_rosters, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "global_unseen_lineages": result["global_unseen"]["lineage_count"],
        "global_unseen_subjects": result["global_unseen"]["subject_count"],
        "model_counts": {model: {k: row[k] for k in ("train_subject_count", "val_subject_count", "global_unseen_subject_count")} for model, row in result["models"].items()},
        "training_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
