#!/usr/bin/env python3
"""Seal physically separate TRAIN and UNSEEN_EVAL manifests for Truth139.

The input prepared-subject index may be created only by the corpus-preparation lane after source
qualification.  This tool verifies every referenced byte once, binds the exact lineage-safe split,
and emits two manifests.  The TRAIN manifest contains no unseen subject ids or unseen file paths;
it carries only a hash commitment to the unseen id set.  The evaluator manifest is a separate
artifact and must not be mounted in the fitting worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INDEX_SCHEMA = "RealSaS.Truth139PreparedSubjectIndex.v1"
SPLIT_SCHEMA = "RealSaS.LineageSafeSubjectSplit.v1"
TRAIN_SCHEMA = "RealSaS.IRISV31Truth139TrainManifest.v1"
EVAL_SCHEMA = "RealSaS.IRISV31Truth139UnseenEvalManifest.v1"
REQUIRED_REFS = (
    "geometry_shard",
    "presentation_targets",
    "surface_normal_truth",
    "frozen_feature_cache",
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _value_hash(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_ref(ref: dict, subject_id: str, role: str) -> dict:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64:
        raise ValueError(f"TRUTH139_PREPARED_REF_INVALID:{subject_id}:{role}")
    if _file_hash(path) != expected:
        raise ValueError(f"TRUTH139_PREPARED_REF_HASH_DRIFT:{subject_id}:{role}")
    return {"path": str(path), "sha256": expected}


def build_manifests(index: dict, split: dict) -> tuple[dict, dict]:
    if index.get("schema") != INDEX_SCHEMA:
        raise ValueError("TRUTH139_PREPARED_INDEX_SCHEMA")
    if split.get("schema") != SPLIT_SCHEMA:
        raise ValueError("TRUTH139_SPLIT_SCHEMA")
    if str(index.get("split_identity_sha256") or "") != str(split.get("split_identity_sha256") or ""):
        raise ValueError("TRUTH139_PREPARED_INDEX_SPLIT_DRIFT")

    rows = {str(row.get("subject_id") or ""): dict(row) for row in index.get("subjects") or ()}
    if "" in rows or len(rows) != len(index.get("subjects") or ()):
        raise ValueError("TRUTH139_PREPARED_SUBJECT_ID_INVALID_OR_DUPLICATE")
    train_ids = list(map(str, split.get("train_subjects") or ()))
    unseen_ids = list(map(str, split.get("unseen_subjects") or ()))
    all_ids = set(train_ids) | set(unseen_ids)
    if set(train_ids) & set(unseen_ids) or set(rows) != all_ids:
        raise ValueError("TRUTH139_PREPARED_INDEX_ROSTER_DRIFT")

    verified: dict[str, dict] = {}
    for subject_id in sorted(rows):
        row = rows[subject_id]
        refs = {}
        for role in REQUIRED_REFS:
            refs[role] = _verify_ref(dict(row.get(role) or {}), subject_id, role)
        observation_binding = str(row.get("observation_binding") or "")
        source_sha256 = str(row.get("source_sha256") or "")
        if len(observation_binding) != 64 or len(source_sha256) != 64:
            raise ValueError("TRUTH139_PREPARED_BINDING_INVALID:" + subject_id)
        verified[subject_id] = {
            "subject_id": subject_id,
            "source_sha256": source_sha256,
            "observation_binding": observation_binding,
            **refs,
        }

    unseen_commitment = _value_hash(sorted(unseen_ids))
    common = {
        "split_identity_sha256": split["split_identity_sha256"],
        "prepared_index_sha256": _value_hash(index),
        "teacher_inputs_in_forward": False,
        "semantic_part_taxonomy_used": False,
        "dense_carrier_family": "DENSE_3D",
    }
    train = {
        "schema": TRAIN_SCHEMA,
        **common,
        "subject_count": len(train_ids),
        "subjects": [verified[subject] for subject in train_ids],
        "unseen_subject_set_commitment_sha256": unseen_commitment,
        "contains_unseen_subject_ids": False,
        "contains_unseen_file_refs": False,
        "optimizer_scope": "TRAIN_ONLY",
    }
    unseen = {
        "schema": EVAL_SCHEMA,
        **common,
        "subject_count": len(unseen_ids),
        "subjects": [verified[subject] for subject in unseen_ids],
        "unseen_subject_set_commitment_sha256": unseen_commitment,
        "optimizer_updates_allowed": False,
        "checkpoint_selection_allowed": False,
        "evaluation_scope": "SEALED_UNSEEN_ONLY",
    }
    train["manifest_identity_sha256"] = _value_hash(train)
    unseen["manifest_identity_sha256"] = _value_hash(unseen)
    return train, unseen


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-index", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--unseen-output", required=True)
    args = parser.parse_args()
    index = json.loads(Path(args.prepared_index).resolve().read_text())
    split = json.loads(Path(args.split).resolve().read_text())
    train, unseen = build_manifests(index, split)
    _write(Path(args.train_output).resolve(), train)
    _write(Path(args.unseen_output).resolve(), unseen)
    print(json.dumps({
        "train_subject_count": train["subject_count"],
        "unseen_subject_count": unseen["subject_count"],
        "train_manifest_identity_sha256": train["manifest_identity_sha256"],
        "unseen_manifest_identity_sha256": unseen["manifest_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
