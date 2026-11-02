#!/usr/bin/env python3
from __future__ import annotations

"""Rescue only proven pipeline failures from a sealed Corpus370 V2 sweep.

This is the Corpus370 analogue of Truth139 Rescue-19: parent successes are never
recomputed. The parent V2 index remains immutable; only failures matching the
explicit uint8-index overflow signature are eligible for this V3 rescue. The
output is a composite index binding the parent receipt and the repaired worker
binding. Remaining scientific/admission failures are preserved verbatim.

No family authority, split, readiness, or training authorization is minted here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import traceback

from tools.corpus.prepare_corpus370_observation_packets_v2 import prepare

PARENT_SCHEMA = "RealSaS.Corpus370PreparedObservationPackets.v2"
SCHEMA = "RealSaS.Corpus370PreparedObservationPackets.v3"
OVERFLOW_RE = re.compile(r"^Python integer \d+ out of bounds for uint8$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def rescue(
    *,
    parent_index_path: Path,
    expected_parent_sha256: str,
    roster_path: Path,
    extracted_source_root: Path,
    output_root: Path,
    blender: Path,
    probe: Path,
    policy_path: Path,
    runtime_request_path: Path,
    expected_rescue_count: int,
) -> dict:
    parent_index_path = parent_index_path.resolve()
    roster_path = roster_path.resolve()
    output_root = output_root.resolve()
    require(parent_index_path.is_file(), "CORPUS370_RESCUE_PARENT_MISSING")
    require(sha256(parent_index_path) == expected_parent_sha256, "CORPUS370_RESCUE_PARENT_SHA_DRIFT")
    parent = json.loads(parent_index_path.read_text())
    require(parent.get("schema") == PARENT_SCHEMA, "CORPUS370_RESCUE_PARENT_SCHEMA")
    require(int(parent.get("subject_count", -1)) == 370, "CORPUS370_RESCUE_PARENT_COUNT")
    require(int(parent.get("prepared_subject_count", -1)) + int(parent.get("failure_count", -1)) == 370,
            "CORPUS370_RESCUE_PARENT_PARTITION")
    require(parent.get("training_authorized") is False, "CORPUS370_RESCUE_PARENT_PREAUTHORIZED")

    roster = json.loads(roster_path.read_text())
    require(roster.get("schema") == "RealSaS.Corpus370ExactSourceRoster.v1", "CORPUS370_RESCUE_ROSTER_SCHEMA")
    roster_rows = {str(row["subject_id"]): row for row in roster.get("subjects") or ()}
    require(len(roster_rows) == 370, "CORPUS370_RESCUE_ROSTER_COUNT")

    parent_subjects = {str(row["subject_id"]): row for row in parent.get("subjects") or ()}
    parent_failures = {str(row["subject_id"]): row for row in parent.get("failures") or ()}
    require(not (set(parent_subjects) & set(parent_failures)), "CORPUS370_RESCUE_PARENT_OVERLAP")
    require(set(parent_subjects) | set(parent_failures) == set(roster_rows), "CORPUS370_RESCUE_PARENT_SUBJECT_SET")

    selected = []
    preserved_failures = []
    for sid, failure in sorted(parent_failures.items()):
        if failure.get("error_type") == "OverflowError" and OVERFLOW_RE.fullmatch(str(failure.get("reason") or "")):
            selected.append(failure)
        else:
            preserved_failures.append(failure)
    require(len(selected) == expected_rescue_count,
            f"CORPUS370_RESCUE_SELECTION_COUNT:{len(selected)}!={expected_rescue_count}")
    require(all(str(row.get("source_format") or "").lower() == ".glb" for row in selected),
            "CORPUS370_RESCUE_OVERFLOW_SCOPE_NOT_GLB")

    output_root.mkdir(parents=True, exist_ok=True)
    failure_root = output_root / "rescue_failures_v3"
    failure_root.mkdir(parents=True, exist_ok=True)
    rescued = []
    rescue_failures = []
    rescue_binding = None

    for index, parent_failure in enumerate(selected, 1):
        sid = str(parent_failure["subject_id"])
        row = roster_rows[sid]
        try:
            result = prepare(
                roster_path,
                extracted_source_root,
                output_root,
                blender,
                probe,
                policy_path,
                runtime_request_path,
                sid,
            )
            require(result.get("schema") == PARENT_SCHEMA and len(result.get("subjects") or ()) == 1,
                    "CORPUS370_RESCUE_SINGLE_RESULT:" + sid)
            subject = dict(result["subjects"][0])
            require(subject.get("subject_id") == sid and subject.get("source_sha256") == row["source_sha256"],
                    "CORPUS370_RESCUE_SUBJECT_BINDING:" + sid)
            current_binding = result.get("binding")
            if rescue_binding is None:
                rescue_binding = current_binding
            else:
                require(rescue_binding == current_binding, "CORPUS370_RESCUE_WORKER_BINDING_DRIFT")
            subject["rescue_v3"] = {
                "parent_failure": {
                    "error_type": parent_failure.get("error_type"),
                    "reason": parent_failure.get("reason"),
                    "receipt": parent_failure.get("receipt"),
                },
                "repair_scope": "GLTF_GLΒ_UINT8_INDEX_STORAGE_WIDEN_BEFORE_SCENE_OFFSET",
            }
            rescued.append(subject)
            print(json.dumps({"event": "CORPUS370_OBSERVATION_RESCUE_V3_PASS", "index": index,
                              "total": len(selected), "subject_id": sid}), flush=True)
        except Exception as exc:
            failure = {
                "subject_id": sid,
                "source_sha256": row["source_sha256"],
                "source_format": row.get("source_format"),
                "source_id": row.get("source_id"),
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
                "parent_failure": parent_failure,
                "family_authority_minted": False,
                "training_authorized": False,
            }
            path = failure_root / (sid + ".json")
            path.write_text(json.dumps(failure, sort_keys=True, indent=2) + "\n")
            failure["receipt"] = {"path": str(path), "sha256": sha256(path)}
            rescue_failures.append(failure)
            print(json.dumps({"event": "CORPUS370_OBSERVATION_RESCUE_V3_FAIL", "index": index,
                              "total": len(selected), "subject_id": sid,
                              "error_type": type(exc).__name__, "reason": str(exc)}), flush=True)

    subjects = {**parent_subjects, **{str(row["subject_id"]): row for row in rescued}}
    failures = {str(row["subject_id"]): row for row in preserved_failures}
    failures.update({str(row["subject_id"]): row for row in rescue_failures})
    require(not (set(subjects) & set(failures)), "CORPUS370_RESCUE_COMPOSITE_OVERLAP")
    require(set(subjects) | set(failures) == set(roster_rows), "CORPUS370_RESCUE_COMPOSITE_SUBJECT_SET")

    result = {
        "schema": SCHEMA,
        "status": "PASS_ALL_370_OBSERVATION_INPUT_PACKETS_PREPARED" if not failures
                  else "COMPOSITE_OBSERVATION_PREPARATION__PARTIAL_FAMILY_EVIDENCE_PRESERVED",
        "subject_count": 370,
        "prepared_subject_count": len(subjects),
        "failure_count": len(failures),
        "full_candidate_count": 370,
        "parent": {
            "schema": parent["schema"],
            "path": str(parent_index_path),
            "sha256": expected_parent_sha256,
            "prepared_subject_count": int(parent["prepared_subject_count"]),
            "failure_count": int(parent["failure_count"]),
        },
        "rescue": {
            "selection_rule": "OverflowError + exact uint8 out-of-bounds signature + GLB source",
            "selected_subject_count": len(selected),
            "rescued_subject_count": len(rescued),
            "rescue_failure_count": len(rescue_failures),
            "selected_subject_ids": [str(row["subject_id"]) for row in selected],
            "worker_binding": rescue_binding,
            "successful_parent_packets_recomputed": False,
        },
        "subjects": [subjects[sid] for sid in sorted(subjects)],
        "failures": [failures[sid] for sid in sorted(failures)],
        "authority": {
            "family_authority_minted": False,
            "split_minted": False,
            "training_authorized": False,
        },
        "training_authorized": False,
    }
    output = output_root / "CORPUS370_PREPARED_OBSERVATION_PACKETS_V3.json"
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "prepared_subject_count": len(subjects), "failure_count": len(failures),
                      "rescued_subject_count": len(rescued), "index_sha256": sha256(output)}, sort_keys=True))
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-index", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--extracted-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=root / "canonical/CORPUS370_IRIS_OBSERVATION_POLICY_V1.json")
    parser.add_argument("--runtime-request", type=Path, default=root / "canonical/CORPUS370_TRUTH_OPERATOR_REQUEST_V1.json")
    parser.add_argument("--expected-rescue-count", type=int, default=23)
    args = parser.parse_args()
    rescue(
        parent_index_path=args.parent_index,
        expected_parent_sha256=args.expected_parent_sha256,
        roster_path=args.exact_source_roster,
        extracted_source_root=args.extracted_source_root,
        output_root=args.output_root,
        blender=args.blender,
        probe=args.probe,
        policy_path=args.policy,
        runtime_request_path=args.runtime_request,
        expected_rescue_count=args.expected_rescue_count,
    )


if __name__ == "__main__":
    main()
