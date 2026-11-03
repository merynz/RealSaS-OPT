"""Consume only the sealed Post139 corpus authority for training/split planning.

The source/audit/finalizer JSONs are evidence inputs.  Training code must not
reinterpret them.  This tool accepts only a verified
``POST139_CORPUS_AUTHORITY_V1.json`` bundle and emits a deterministic consumer
manifest for immutable Base139 rows plus admitted Post139 primary rows.

It also validates fit/unseen/validation plans against the sealed lineage graph.
A lineage component is atomic: if two training rows are connected by exact rig
hierarchy or exact canonical surface identity they may not cross split
boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.corpus.post139_corpus_authority_v1 import read_json, verify_authority_dir, write_json

MANIFEST_SCHEMA = "RealSaS.Post139.TrainingConsumerManifest.v1"
SPLIT_SCHEMA = "RealSaS.Post139.TrainingSplitPlan.v1"
BASE_STATUS = "REFERENCE_BASE139_ROW"
POST_STATUS = "ADMIT_PRIMARY"
TRAINING_STATUSES = {BASE_STATUS, POST_STATUS}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _load_authority_ledgers(authority_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = verify_authority_dir(authority_dir)
    ledgers: dict[str, dict[str, Any]] = {}
    for key, ref in root["ledgers"].items():
        ledgers[key] = read_json(authority_dir / ref["file"])
    return root, ledgers


def _unique(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"POST139_TRAINING_CONSUMER_KEY_MISSING:{label}:{key}")
        if value in result:
            raise ValueError(f"POST139_TRAINING_CONSUMER_DUPLICATE_KEY:{label}:{value}")
        result[value] = row
    return result


def build_training_manifest(root: dict[str, Any], ledgers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    identities = _unique(ledgers["identity"].get("subjects", []), "subject_id", "identity")
    admissions = _unique(ledgers["admission"].get("subjects", []), "subject_id", "admission")
    materialization = _unique(ledgers["materialization"].get("primary_subjects", []), "subject_id", "materialization-primary")

    component_by_subject: dict[str, str] = {}
    for component in ledgers["lineage"].get("components", []):
        component_id = component.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise ValueError("POST139_TRAINING_CONSUMER_COMPONENT_ID_MISSING")
        for subject_id in component.get("members", []):
            previous = component_by_subject.setdefault(subject_id, component_id)
            if previous != component_id:
                raise ValueError(f"POST139_TRAINING_CONSUMER_MULTI_COMPONENT:{subject_id}")

    rows = []
    for subject_id, admission in sorted(admissions.items()):
        status = admission.get("status")
        if status not in TRAINING_STATUSES:
            continue
        identity = identities.get(subject_id)
        if identity is None:
            raise ValueError(f"POST139_TRAINING_CONSUMER_IDENTITY_MISSING:{subject_id}")
        component_id = component_by_subject.get(subject_id)
        if component_id is None:
            raise ValueError(f"POST139_TRAINING_CONSUMER_LINEAGE_MISSING:{subject_id}")
        fingerprints = identity.get("fingerprints", {})
        row = {
            "subject_id": subject_id,
            "source_id": identity.get("source_id"),
            "source_family": identity.get("source_family"),
            "pack": identity.get("pack"),
            "corpus_partition": "BASE139" if status == BASE_STATUS else "POST139",
            "admission_status": status,
            "lineage_component_id": component_id,
            "source_sha256": identity.get("source_sha256"),
            "relative_source_path": identity.get("relative_source_path"),
            "structural_subject": fingerprints.get("structural_subject"),
            "geometry_topology": fingerprints.get("geometry_topology"),
            "rig_rest": fingerprints.get("rig_rest"),
            "rig_hierarchy": fingerprints.get("rig_hierarchy"),
            "skin_weights": fingerprints.get("skin_weights"),
            "appearance": fingerprints.get("appearance"),
            "animation_inventory": fingerprints.get("animation_inventory"),
            "truth_scope": "FULL_ANIMATED_TRUTH",
        }
        if status == POST_STATUS:
            material = materialization.get(subject_id)
            if material is None:
                raise ValueError(f"POST139_TRAINING_CONSUMER_MATERIALIZATION_MISSING:{subject_id}")
            if material.get("lineage_component_id") != component_id:
                raise ValueError(f"POST139_TRAINING_CONSUMER_MATERIALIZATION_LINEAGE_MISMATCH:{subject_id}")
            if material.get("model_sha256") != row["source_sha256"]:
                raise ValueError(f"POST139_TRAINING_CONSUMER_MATERIALIZATION_HASH_MISMATCH:{subject_id}")
            row["materialization_authority"] = "POST139_MATERIALIZATION_LEDGER_V1"
            row["workspace_relative_model"] = material.get("workspace_relative_model")
            row["source_closure_ledger_sha256"] = material.get("source_closure_ledger_sha256")
            row["publisher_tree_digest"] = material.get("publisher_tree_digest")
        else:
            # Base139 is intentionally not rematerialized by Post139.  Its training
            # truth is the already-frozen Truth139 authority; Post139 only seals its
            # identity/lineage participation so leakage checks cover the full corpus.
            row["materialization_authority"] = "EXISTING_IMMUTABLE_TRUTH139_AUTHORITY"
        rows.append(row)

    counts = dict(sorted(Counter(row["corpus_partition"] for row in rows).items()))
    roster = ledgers["roster"]
    expected_base = int(roster.get("base139_training_row_count", -1))
    expected_post = int(roster.get("post139_primary_materialization_count", -1))
    if counts.get("BASE139", 0) != expected_base:
        raise ValueError(f"POST139_TRAINING_CONSUMER_BASE_COUNT_MISMATCH:{counts.get('BASE139', 0)}:{expected_base}")
    if counts.get("POST139", 0) != expected_post:
        raise ValueError(f"POST139_TRAINING_CONSUMER_POST_COUNT_MISMATCH:{counts.get('POST139', 0)}:{expected_post}")
    if len(rows) != int(roster.get("expanded_full_training_row_count", -1)):
        raise ValueError("POST139_TRAINING_CONSUMER_EXPANDED_COUNT_MISMATCH")

    core = {
        "authority_digest": root["authority_digest"],
        "split_rule": "LINEAGE_COMPONENT_ID_IS_ATOMIC_FOR_FIT_UNSEEN_SPLITS",
        "subjects": rows,
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "READY",
        "authority_digest": root["authority_digest"],
        "training_row_count": len(rows),
        "partition_counts": counts,
        "lineage_component_count": len({row["lineage_component_id"] for row in rows}),
        "split_rule": core["split_rule"],
        "subjects": rows,
        "manifest_digest": digest(core),
    }


def validate_split_plan(manifest: dict[str, Any], split_plan: dict[str, Any]) -> dict[str, Any]:
    if split_plan.get("schema") != SPLIT_SCHEMA:
        raise ValueError(f"POST139_TRAINING_SPLIT_SCHEMA_MISMATCH:{split_plan.get('schema')}")
    subjects = _unique(manifest.get("subjects", []), "subject_id", "training-manifest")
    assignments = _unique(split_plan.get("assignments", []), "subject_id", "split-plan")
    if set(assignments) != set(subjects):
        missing = sorted(set(subjects) - set(assignments))
        extra = sorted(set(assignments) - set(subjects))
        raise ValueError(f"POST139_TRAINING_SPLIT_SUBJECT_SET_MISMATCH:missing={missing}:extra={extra}")

    component_splits: dict[str, set[str]] = {}
    split_counts: Counter[str] = Counter()
    for subject_id, assignment in sorted(assignments.items()):
        split = assignment.get("split")
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"POST139_TRAINING_SPLIT_NAME_INVALID:{subject_id}")
        split = split.strip().upper()
        component_id = subjects[subject_id]["lineage_component_id"]
        component_splits.setdefault(component_id, set()).add(split)
        split_counts[split] += 1
    leaking = {
        component_id: sorted(splits)
        for component_id, splits in sorted(component_splits.items())
        if len(splits) > 1
    }
    if leaking:
        raise ValueError(f"POST139_TRAINING_SPLIT_LINEAGE_LEAKAGE:{json.dumps(leaking, sort_keys=True)}")

    core = {
        "authority_digest": manifest["authority_digest"],
        "training_manifest_digest": manifest["manifest_digest"],
        "assignments": [
            {"subject_id": subject_id, "split": assignments[subject_id]["split"].strip().upper()}
            for subject_id in sorted(assignments)
        ],
    }
    return {
        "schema": "RealSaS.Post139.ValidatedTrainingSplit.v1",
        "status": "PASS_LINEAGE_ATOMIC_SPLIT",
        "authority_digest": manifest["authority_digest"],
        "training_manifest_digest": manifest["manifest_digest"],
        "split_counts": dict(sorted(split_counts.items())),
        "lineage_component_count": len(component_splits),
        "assignments": core["assignments"],
        "split_digest": digest(core),
    }


def emit(args: argparse.Namespace) -> int:
    authority_dir = Path(args.authority_dir).resolve()
    root, ledgers = _load_authority_ledgers(authority_dir)
    manifest = build_training_manifest(root, ledgers)
    write_json(Path(args.out).resolve(), manifest)
    print(
        "POST139_TRAINING_CONSUMER_MANIFEST_PASS",
        manifest["training_row_count"],
        manifest["partition_counts"],
        manifest["manifest_digest"],
    )
    return 0


def validate(args: argparse.Namespace) -> int:
    authority_dir = Path(args.authority_dir).resolve()
    root, ledgers = _load_authority_ledgers(authority_dir)
    manifest = build_training_manifest(root, ledgers)
    plan = read_json(Path(args.split_plan).resolve())
    validated = validate_split_plan(manifest, plan)
    write_json(Path(args.out).resolve(), validated)
    print("POST139_TRAINING_SPLIT_PASS", validated["split_counts"], validated["split_digest"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_emit = sub.add_parser("emit")
    p_emit.add_argument("--authority-dir", required=True)
    p_emit.add_argument("--out", required=True)
    p_emit.set_defaults(func=emit)
    p_validate = sub.add_parser("validate-split")
    p_validate.add_argument("--authority-dir", required=True)
    p_validate.add_argument("--split-plan", required=True)
    p_validate.add_argument("--out", required=True)
    p_validate.set_defaults(func=validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
