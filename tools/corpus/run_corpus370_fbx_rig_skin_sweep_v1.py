#!/usr/bin/env python3
from __future__ import annotations

"""Failure-preserving Corpus370 FBX rig/skin importer court sweep V1.

For the exact 111 FBX subjects this runner executes:
  1) primary animation-free pinned-Blender source inspection,
  2) separately implemented animation-free pinned-Blender replay,
  3) non-Blender semantic equivalence court.

Every selected subject is accounted as PASS_SOURCE_COURT or FAIL. Failures are
preserved and never zero-filled or silently removed. This sweep does not mint
final family, split, or training authority.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import traceback

from tools.corpus.collect_corpus370_rig_skin_court_index_v1 import collect as collect_courts
from tools.corpus.validate_fbx_imported_rig_skin_authority_v1 import validate as validate_fbx


SCHEMA = "RealSaS.Corpus370FbxRigSkinSourceSweep.v1"
ROSTER_SCHEMA = "RealSaS.Corpus370ExactSourceRoster.v1"
COURT_SCHEMA = "RealSaS.FbxImportedRigSkinAuthorityCourt.v1"
EXPECTED_ROSTER_SHA256 = "925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return path


def artifact_ref(path: Path, schema: str | None = None, status: str | None = None) -> dict:
    row = {"path": str(path.resolve()), "sha256": sha256(path)}
    if schema is not None:
        row["schema"] = schema
    if status is not None:
        row["status"] = status
    return row


def compact_reason(text: str, fallback: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return fallback
    for line in reversed(lines):
        if "Error:" in line or "ValueError:" in line or line.startswith("Traceback"):
            return line[-500:]
    return lines[-1][-500:]


def run_blender(
    blender: str,
    script: Path,
    source: Path,
    source_sha: str,
    subject_id: str,
    output_dir: Path,
    log_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(script),
        "--",
        "--source",
        str(source),
        "--expected-source-sha256",
        source_sha,
        "--subject-id",
        subject_id,
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout or "")
    if result.returncode != 0:
        raise RuntimeError(
            f"{script.stem.upper()}_EXIT_{result.returncode}:"
            + compact_reason(result.stdout or "", "BLENDER_PROCESS_FAILED")
        )


def failure(exc: Exception) -> dict:
    return {
        "status": "FAIL",
        "error_type": type(exc).__name__,
        "reason": str(exc),
        "traceback": traceback.format_exc(),
        "final_family_authority_minted": False,
    }


def run_subject(
    *,
    blender: str,
    repo_root: Path,
    source: Path,
    source_sha: str,
    subject_id: str,
    root: Path,
) -> dict:
    try:
        primary_script = repo_root / "tools/corpus/blender_fbx_source_inspection_v1.py"
        replay_script = repo_root / "tools/corpus/blender_fbx_rig_skin_replay_v1.py"
        require(primary_script.is_file(), "CORPUS370_FBX_PRIMARY_PRODUCER_MISSING")
        require(replay_script.is_file(), "CORPUS370_FBX_REPLAY_PRODUCER_MISSING")

        primary_root = root / "primary"
        replay_root = root / "replay"
        run_blender(
            blender,
            primary_script,
            source,
            source_sha,
            subject_id,
            primary_root,
            root / "PRIMARY_BLENDER.log",
        )
        run_blender(
            blender,
            replay_script,
            source,
            source_sha,
            subject_id,
            replay_root,
            root / "REPLAY_BLENDER.log",
        )

        inspection_path = primary_root / "IMPORTED_TEACHER_SOURCE_INSPECTION_V1.json"
        replay_path = replay_root / "FBX_RIG_SKIN_IMPORTER_REPLAY_V1.json"
        require(inspection_path.is_file(), "CORPUS370_FBX_PRIMARY_ARTIFACT_MISSING")
        require(replay_path.is_file(), "CORPUS370_FBX_REPLAY_ARTIFACT_MISSING")

        court = validate_fbx(inspection_path, replay_path)
        require(court.get("schema") == COURT_SCHEMA, "CORPUS370_FBX_COURT_SCHEMA")
        require(court.get("subject_id") == subject_id, "CORPUS370_FBX_COURT_SUBJECT")
        require((court.get("authority") or {}).get("training_authorized") is False, "CORPUS370_FBX_COURT_TRAINING_AUTHORITY")
        court_path = write(root / "FBX_IMPORTED_RIG_SKIN_AUTHORITY_COURT_V1.json", court)

        return {
            "status": "PASS_SOURCE_COURT",
            "inspection": artifact_ref(
                inspection_path,
                "RealSaS.ImportedTeacherSourceInspection.v1",
            ),
            "replay": artifact_ref(
                replay_path,
                "RealSaS.FbxRigSkinImporterReplay.v1",
            ),
            "court": artifact_ref(
                court_path,
                COURT_SCHEMA,
                str(court.get("status") or ""),
            ),
            "family_candidate_outcomes": dict(court.get("family_candidate_outcomes") or {}),
            "skin_quality": dict(court.get("skin_quality") or {}),
            "final_family_authority_minted": False,
        }
    except Exception as exc:
        return failure(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--extracted-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", default=None)
    args = parser.parse_args()

    roster_path = args.exact_source_roster.resolve()
    source_root = args.extracted_source_root.resolve()
    output_root = args.output_root.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    blender = args.blender or shutil.which("blender")
    require(bool(blender), "CORPUS370_FBX_BLENDER_NOT_FOUND")
    blender = str(Path(blender).resolve())

    require(roster_path.is_file(), "CORPUS370_FBX_ROSTER_MISSING")
    require(sha256(roster_path) == EXPECTED_ROSTER_SHA256, "CORPUS370_FBX_ROSTER_SHA")
    roster = json.loads(roster_path.read_text())
    require(roster.get("schema") == ROSTER_SCHEMA, "CORPUS370_FBX_ROSTER_SCHEMA")
    require(int(roster.get("subject_count", -1)) == 370, "CORPUS370_FBX_ROSTER_COUNT")
    rows = list(roster.get("subjects") or ())
    require(len(rows) == 370, "CORPUS370_FBX_ROSTER_ROWS")

    selected = [row for row in rows if str(row.get("source_format") or "").lower() == ".fbx"]
    require(len(selected) == 111, f"CORPUS370_FBX_EXPECTED_111:{len(selected)}")
    require(len({str(row["subject_id"]) for row in selected}) == 111, "CORPUS370_FBX_DUPLICATE_SUBJECT")

    results: list[dict] = []
    rollup = {"PASS_SOURCE_COURT": 0, "FAIL": 0}
    failure_reasons: dict[str, int] = {}
    candidate_outcomes = {
        "RIG_HIERARCHY": {},
        "SKIN_WEIGHTS": {},
    }

    for index, row in enumerate(sorted(selected, key=lambda value: str(value["subject_id"])), 1):
        sid = str(row["subject_id"])
        source = (source_root / str(row["source_relative_path"])).resolve()
        require(source.is_relative_to(source_root), "CORPUS370_FBX_SOURCE_ESCAPE")
        require(source.is_file(), "CORPUS370_FBX_SOURCE_MISSING:" + sid)
        source_sha = str(row["source_sha256"])
        require(sha256(source) == source_sha, "CORPUS370_FBX_SOURCE_SHA:" + sid)

        lane = run_subject(
            blender=blender,
            repo_root=repo_root,
            source=source,
            source_sha=source_sha,
            subject_id=sid,
            root=output_root / "subjects" / sid / "fbx_rig_skin_v1",
        )
        rollup[lane["status"]] += 1
        if lane["status"] == "FAIL":
            reason = str(lane.get("reason") or "UNKNOWN")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        else:
            for family, outcome in (lane.get("family_candidate_outcomes") or {}).items():
                bucket = candidate_outcomes.setdefault(str(family), {})
                key = str(outcome)
                bucket[key] = bucket.get(key, 0) + 1

        results.append(
            {
                "subject_id": sid,
                "source_sha256": source_sha,
                "source_format": str(row["source_format"]),
                "source_id": row.get("source_id"),
                "RIG_SKIN_SOURCE": lane,
            }
        )
        print(
            json.dumps(
                {
                    "event": "CORPUS370_FBX_RIG_SKIN_SOURCE_V1",
                    "index": index,
                    "total": len(selected),
                    "subject_id": sid,
                    "status": lane["status"],
                    "reason": lane.get("reason"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    require(sum(rollup.values()) == 111, "CORPUS370_FBX_ACCOUNTING")
    require(len(results) == 111, "CORPUS370_FBX_RESULT_ROWS")

    court_index = collect_courts(output_root)
    require(court_index.get("schema") == "RealSaS.Corpus370RigSkinCourtIndex.v1", "CORPUS370_FBX_INDEX_SCHEMA")
    require(int(court_index.get("subject_count", -1)) == rollup["PASS_SOURCE_COURT"], "CORPUS370_FBX_INDEX_PASS_COUNT")
    court_index_path = write(output_root / "CORPUS370_FBX_RIG_SKIN_COURT_INDEX_V1.json", court_index)

    result = {
        "schema": SCHEMA,
        "status": "PASS_FAILURE_PRESERVING_FBX_RIG_SKIN_SOURCE_SWEEP_V1__FINAL_ADMISSION_SEPARATE",
        "candidate_subject_count": 370,
        "selected_subject_count": 111,
        "format_counts": {"fbx": 111},
        "rollup": {"RIG_SKIN_SOURCE": rollup},
        "candidate_outcomes": candidate_outcomes,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "subjects": results,
        "parents": {
            "exact_source_roster": {
                "path": str(roster_path),
                "sha256": sha256(roster_path),
            }
        },
        "court_index": artifact_ref(
            court_index_path,
            "RealSaS.Corpus370RigSkinCourtIndex.v1",
        ),
        "authority": {
            "pinned_importer_primary_complete_for_passing_subjects": True,
            "independent_importer_replay_complete_for_passing_subjects": True,
            "non_blender_equivalence_court_complete_for_passing_subjects": True,
            "all_selected_subjects_accounted": True,
            "final_family_authority_minted": False,
            "split_minted": False,
            "training_authorized": False,
        },
        "training_authorized": False,
    }
    output = write(output_root / "CORPUS370_FBX_RIG_SKIN_SOURCE_SWEEP_V1.json", result)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "selected_subject_count": 111,
                "rollup": result["rollup"],
                "candidate_outcomes": result["candidate_outcomes"],
                "failure_reasons": result["failure_reasons"],
                "court_index_subject_count": court_index["subject_count"],
                "output_sha256": sha256(output),
                "training_authorized": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
