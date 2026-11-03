#!/usr/bin/env python3
from __future__ import annotations

"""Run independently recorded Corpus370 V3 observation repair sweep."""

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from tools.corpus.prepare_corpus370_observation_packets_v4 import prepare

SCHEMA = "RealSaS.Corpus370PreparedObservationPackets.v4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def sweep(
    roster_path: Path,
    extracted_source_root: Path,
    output_root: Path,
    blender: Path,
    probe: Path,
    policy_path: Path,
    runtime_request_path: Path,
) -> dict:
    roster_path = roster_path.resolve()
    output_root = output_root.resolve()
    roster = json.loads(roster_path.read_text())
    require(roster.get("schema") == "RealSaS.Corpus370ExactSourceRoster.v1", "CORPUS370_OBSERVATION_SWEEP_ROSTER_SCHEMA")
    rows = list(roster.get("subjects") or ())
    require(int(roster.get("subject_count", -1)) == 370 and len(rows) == 370, "CORPUS370_OBSERVATION_SWEEP_ROSTER_COUNT")
    require(len({row["subject_id"] for row in rows}) == 370, "CORPUS370_OBSERVATION_SWEEP_ROSTER_IDS")

    output_root.mkdir(parents=True, exist_ok=True)
    failure_root = output_root / "failures_v4"
    failure_root.mkdir(parents=True, exist_ok=True)
    subjects = []
    failures = []
    binding = None

    for index, row in enumerate(sorted(rows, key=lambda item: item["subject_id"]), 1):
        sid = str(row["subject_id"])
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
            require(result.get("schema") == SCHEMA and len(result.get("subjects") or ()) == 1, "CORPUS370_OBSERVATION_SWEEP_SINGLE_RESULT:" + sid)
            subject = dict(result["subjects"][0])
            require(subject.get("subject_id") == sid and subject.get("source_sha256") == row["source_sha256"], "CORPUS370_OBSERVATION_SWEEP_SUBJECT_BINDING:" + sid)
            subjects.append(subject)
            current_binding = result.get("binding")
            if binding is None:
                binding = current_binding
            else:
                require(binding == current_binding, "CORPUS370_OBSERVATION_SWEEP_PRODUCER_BINDING_DRIFT")
            print(json.dumps({"event": "CORPUS370_OBSERVATION_SWEEP_V4_PASS", "index": index, "total": 370, "subject_id": sid}), flush=True)
        except Exception as exc:
            failure = {
                "subject_id": sid,
                "source_sha256": row["source_sha256"],
                "source_format": row.get("source_format"),
                "source_id": row.get("source_id"),
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
                "family_authority_minted": False,
                "training_authorized": False,
            }
            path = failure_root / (sid + ".json")
            path.write_text(json.dumps(failure, sort_keys=True, indent=2) + "\n")
            failure["receipt"] = {"path": str(path), "sha256": sha256(path)}
            failures.append(failure)
            print(json.dumps({"event": "CORPUS370_OBSERVATION_SWEEP_V4_FAIL", "index": index, "total": 370, "subject_id": sid, "error_type": type(exc).__name__, "reason": str(exc)}), flush=True)

    result = {
        "schema": SCHEMA,
        "status": "PASS_ALL_370_OBSERVATION_INPUT_PACKETS_PREPARED" if not failures else "INCOMPLETE_OBSERVATION_PREPARATION__FAILURES_PRESERVED",
        "subject_count": 370,
        "prepared_subject_count": len(subjects),
        "failure_count": len(failures),
        "full_candidate_count": 370,
        "exact_source_roster": {"path": str(roster_path), "sha256": sha256(roster_path)},
        "binding": binding,
        "subjects": sorted(subjects, key=lambda item: item["subject_id"]),
        "failures": sorted(failures, key=lambda item: item["subject_id"]),
        "authority": {
            "platform_attempts_created": False,
            "family_authority_minted": False,
            "split_minted": False,
            "training_authorized": False,
        },
        "training_authorized": False,
    }
    index_path = output_root / "CORPUS370_PREPARED_OBSERVATION_PACKETS_V4.json"
    index_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": SCHEMA, "status": result["status"], "prepared_subject_count": len(subjects), "failure_count": len(failures), "index_sha256": sha256(index_path)}, sort_keys=True))
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--extracted-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=root / "canonical/CORPUS370_IRIS_OBSERVATION_POLICY_V1.json")
    parser.add_argument("--runtime-request", type=Path, default=root / "canonical/CORPUS370_TRUTH_OPERATOR_REQUEST_V1.json")
    args = parser.parse_args()
    sweep(
        args.exact_source_roster,
        args.extracted_source_root,
        args.output_root,
        args.blender,
        args.probe,
        args.policy,
        args.runtime_request,
    )


if __name__ == "__main__":
    main()
