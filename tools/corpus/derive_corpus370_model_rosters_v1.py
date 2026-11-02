#!/usr/bin/env python3
from __future__ import annotations

"""Derive model-specific Corpus370 corpus-eligibility rosters from the authority matrix.

This tool never invents a split or training authority. It only projects subjects
whose required family cells are PASS. Optional IRIS families are reported as
coverage masks so global-SDF/graphics losses can be enabled per sample without
turning absent targets into zeros.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

MATRIX_SCHEMA = "RealSaS.Corpus370SubjectFamilyAuthorityMatrix.v1"
SCHEMA = "RealSaS.Corpus370ModelRosters.v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def derive(matrix_path: Path) -> dict:
    matrix = json.loads(matrix_path.read_text())
    require(matrix.get("schema") == MATRIX_SCHEMA, "CORPUS370_ROSTER_MATRIX_SCHEMA")
    require(matrix.get("training_authorized") is False, "CORPUS370_MATRIX_MUST_NOT_PREAUTHORIZE_TRAINING")
    subjects = list(matrix.get("subjects") or ())
    require(len(subjects) == int(matrix.get("subject_count", -1)), "CORPUS370_ROSTER_SUBJECT_COUNT")
    model_names = sorted({name for row in subjects for name in (row.get("model_eligibility") or {})})
    rosters = {}
    for model in model_names:
        admitted = []
        blocked = []
        for row in subjects:
            state = row["model_eligibility"][model]
            record = {
                "subject_id": row["subject_id"],
                "source_sha256": row["source_sha256"],
                "origin": row.get("origin"),
                "source_id": row.get("source_id"),
                "required_pass": list(state.get("required_pass") or ()),
                "family_status": {family: cell["status"] for family, cell in row["families"].items()},
            }
            if bool(state.get("corpus_eligible")):
                admitted.append(record)
            else:
                record["blocking_families"] = list(state.get("blocking_families") or ())
                blocked.append(record)
        admitted_ids = {x["subject_id"] for x in admitted}
        coverage = {}
        for family in sorted(subjects[0]["families"] if subjects else ()):
            counter = Counter(
                row["families"][family]["status"]
                for row in subjects
                if row["subject_id"] in admitted_ids
            )
            coverage[family] = dict(sorted(counter.items()))
        rosters[model] = {
            "status": "CORPUS_ELIGIBILITY_ONLY__NO_SPLIT__NO_TRAINING_AUTHORITY",
            "eligible_count": len(admitted),
            "blocked_count": len(blocked),
            "eligible_subjects": admitted,
            "blocked_subjects": blocked,
            "eligible_family_coverage": coverage,
        }
    return {
        "schema": SCHEMA,
        "status": "OPEN__GLOBAL_LINEAGE_SPLIT_AND_READINESS_REQUIRED",
        "training_authorized": False,
        "authority_matrix": {"path": str(matrix_path), "sha256": sha256(matrix_path)},
        "subject_count": len(subjects),
        "models": rosters,
        "split": None,
        "global_unseen": None,
        "next_required_authority": [
            "global_lineage_dedup",
            "global_end_to_end_unseen_isolation",
            "model_specific_train_val_split",
            "materialization_seal",
            "training_readiness_receipt"
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = derive(args.matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": out["schema"], "subject_count": out["subject_count"], "counts": {k: v["eligible_count"] for k, v in out["models"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
