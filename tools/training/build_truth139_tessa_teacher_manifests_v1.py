#!/usr/bin/env python3
"""Create physically separate TESSA teacher-target manifests from the sealed Truth139 split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INDEX_SCHEMA = "RealSaS.Truth139TESSATeacherIndex.v1"
SPLIT_SCHEMA = "RealSaS.LineageSafeSubjectSplit.v1"
TRAIN_SCHEMA = "RealSaS.Truth139TESSATeacherTrainManifest.v1"
EVAL_SCHEMA = "RealSaS.Truth139TESSATeacherUnseenManifest.v1"


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _value_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _verified_ref(ref: dict, subject: str, role: str) -> dict:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or _hash(path) != expected:
        raise ValueError(f"TRUTH139_TESSA_TEACHER_REF_DRIFT:{subject}:{role}")
    return {"path": str(path), "sha256": expected}


def build_manifests(index: dict, split: dict) -> tuple[dict, dict]:
    if index.get("schema") != INDEX_SCHEMA:
        raise ValueError("TRUTH139_TESSA_TEACHER_INDEX_SCHEMA")
    if split.get("schema") != SPLIT_SCHEMA:
        raise ValueError("TRUTH139_TESSA_TEACHER_SPLIT_SCHEMA")
    split_identity = str(split.get("split_identity_sha256") or "")
    if str(index.get("split_identity_sha256") or "") != split_identity:
        raise ValueError("TRUTH139_TESSA_TEACHER_SPLIT_DRIFT")
    rows = {str(row.get("subject_id") or ""): dict(row) for row in index.get("subjects") or ()}
    if "" in rows or len(rows) != len(index.get("subjects") or ()):
        raise ValueError("TRUTH139_TESSA_TEACHER_SUBJECT_IDS")
    train_ids = list(map(str, split.get("train_subjects") or ()))
    unseen_ids = list(map(str, split.get("unseen_subjects") or ()))
    if set(rows) != set(train_ids) | set(unseen_ids) or set(train_ids) & set(unseen_ids):
        raise ValueError("TRUTH139_TESSA_TEACHER_ROSTER_DRIFT")

    verified = {}
    for subject, row in rows.items():
        source_sha256 = str(row.get("source_sha256") or "")
        if len(source_sha256) != 64:
            raise ValueError("TRUTH139_TESSA_TEACHER_SOURCE_IDENTITY:" + subject)
        verified[subject] = {
            "subject_id": subject,
            "source_sha256": source_sha256,
            "teacher_mesh": _verified_ref(dict(row.get("teacher_mesh") or {}), subject, "mesh"),
            "teacher_receipt": _verified_ref(dict(row.get("teacher_receipt") or {}), subject, "receipt"),
        }

    common = {
        "split_identity_sha256": split_identity,
        "teacher_index_sha256": _value_hash(index),
        "teacher_topology_target_only": True,
        "student_conditioning_allowed": False,
    }
    train = {
        "schema": TRAIN_SCHEMA,
        **common,
        "subject_count": len(train_ids),
        "subjects": [verified[subject] for subject in train_ids],
        "contains_unseen_subject_ids": False,
        "contains_unseen_file_refs": False,
        "scope": "TRAIN_TEACHER_TARGETS_ONLY",
    }
    unseen = {
        "schema": EVAL_SCHEMA,
        **common,
        "subject_count": len(unseen_ids),
        "subjects": [verified[subject] for subject in unseen_ids],
        "optimizer_updates_allowed": False,
        "checkpoint_selection_allowed": False,
        "scope": "SEALED_UNSEEN_TEACHER_TARGETS_ONLY",
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
    parser.add_argument("--teacher-index", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--unseen-output", required=True)
    args = parser.parse_args()
    index = json.loads(Path(args.teacher_index).resolve().read_text())
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
