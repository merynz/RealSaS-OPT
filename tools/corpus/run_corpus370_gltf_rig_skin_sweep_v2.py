#!/usr/bin/env python3
from __future__ import annotations

"""Failure-preserving Corpus370 glTF2 rig/skin source-court sweep V2.

This is a narrow repair lane for the V1 systematic quaternion/TRS mismatch.  It
re-runs only RIG_SKIN_SOURCE on the exact 259 .gltf/.glb subjects.  Animation V1
artifacts remain immutable and are not recomputed here.  No importer equivalence,
final family authority, split or training authority is minted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from tools.corpus.build_gltf_rig_skin_authority_v2 import build as build_rig_skin
from tools.corpus.validate_gltf_rig_skin_authority_v2 import validate as validate_rig_skin


SCHEMA = "RealSaS.Corpus370Gltf2RigSkinSourceSweep.v2"
ROSTER_SCHEMA = "RealSaS.Corpus370ExactSourceRoster.v1"
RIG_MANIFEST_SCHEMA = "RealSaS.GltfRigSkinAuthority.v2"
RIG_COURT_SCHEMA = "RealSaS.GltfRigSkinAuthorityCourt.v2"


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


def ref(path: Path, schema: str, status: str | None = None) -> dict:
    row = {"path": str(path.resolve()), "sha256": sha256(path), "schema": schema}
    if status is not None:
        row["status"] = status
    return row


def failure(exc: Exception) -> dict:
    return {
        "status": "FAIL",
        "error_type": type(exc).__name__,
        "reason": str(exc),
        "traceback": traceback.format_exc(),
    }


def run_lane(source: Path, root: Path) -> dict:
    try:
        manifest = build_rig_skin(source, root)
        require(manifest.get("schema") == RIG_MANIFEST_SCHEMA, "CORPUS370_GLTF_RIG_V2_MANIFEST_SCHEMA")
        manifest_path = root / "GLTF_RIG_SKIN_AUTHORITY_MANIFEST_V2.json"
        court = validate_rig_skin(manifest)
        require(court.get("schema") == RIG_COURT_SCHEMA, "CORPUS370_GLTF_RIG_V2_COURT_SCHEMA")
        court_path = write(root / "GLTF_RIG_SKIN_AUTHORITY_COURT_V2.json", court)
        return {
            "status": "PASS_SOURCE_COURT",
            "manifest": ref(manifest_path, RIG_MANIFEST_SCHEMA, str(manifest.get("status") or "")),
            "court": ref(court_path, RIG_COURT_SCHEMA, str(court.get("status") or "")),
            "importer_equivalence_required": True,
            "final_family_authority_minted": False,
        }
    except Exception as exc:
        return failure(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--extracted-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    roster_path = args.exact_source_roster.resolve()
    source_root = args.extracted_source_root.resolve()
    output_root = args.output_root.resolve()
    roster = json.loads(roster_path.read_text())
    require(roster.get("schema") == ROSTER_SCHEMA, "CORPUS370_GLTF_RIG_V2_ROSTER_SCHEMA")
    require(int(roster.get("subject_count", -1)) == 370, "CORPUS370_GLTF_RIG_V2_ROSTER_COUNT")
    rows = list(roster.get("subjects") or ())
    require(len(rows) == 370, "CORPUS370_GLTF_RIG_V2_ROSTER_ROWS")

    selected = [row for row in rows if str(row.get("source_format") or "").lower() in {".gltf", ".glb"}]
    require(len(selected) == 259, f"CORPUS370_GLTF_RIG_V2_EXPECTED_259:{len(selected)}")

    results = []
    rollup = {"PASS_SOURCE_COURT": 0, "FAIL": 0}
    failure_reasons: dict[str, int] = {}
    for index, row in enumerate(sorted(selected, key=lambda value: str(value["subject_id"])), 1):
        sid = str(row["subject_id"])
        source = (source_root / str(row["source_relative_path"])).resolve()
        require(source.is_relative_to(source_root), "CORPUS370_GLTF_RIG_V2_SOURCE_ESCAPE")
        require(source.is_file(), "CORPUS370_GLTF_RIG_V2_SOURCE_MISSING:" + sid)
        require(sha256(source) == str(row["source_sha256"]), "CORPUS370_GLTF_RIG_V2_SOURCE_SHA:" + sid)
        lane = run_lane(source, output_root / "subjects" / sid / "rig_skin_source_v2")
        rollup[lane["status"]] += 1
        if lane["status"] == "FAIL":
            reason = str(lane.get("reason") or "UNKNOWN")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        results.append(
            {
                "subject_id": sid,
                "source_sha256": str(row["source_sha256"]),
                "source_format": str(row["source_format"]),
                "source_id": row.get("source_id"),
                "RIG_SKIN_SOURCE": lane,
            }
        )
        print(
            json.dumps(
                {
                    "event": "CORPUS370_GLTF2_RIG_SKIN_SOURCE_V2",
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

    result = {
        "schema": SCHEMA,
        "status": "PASS_FAILURE_PRESERVING_GLTF2_RIG_SKIN_SOURCE_SWEEP_V2__IMPORT_EQUIVALENCE_PENDING",
        "candidate_subject_count": 370,
        "selected_subject_count": len(selected),
        "format_counts": {
            "gltf": sum(str(row.get("source_format") or "").lower() == ".gltf" for row in selected),
            "glb": sum(str(row.get("source_format") or "").lower() == ".glb" for row in selected),
        },
        "rollup": {"RIG_SKIN_SOURCE": rollup},
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "subjects": results,
        "parents": {
            "exact_source_roster": {"path": str(roster_path), "sha256": sha256(roster_path)},
            "v1_failure_class": {
                "reason": "GLTF_RIG_SKIN_COURT_ARRAY_MISMATCH:joint_local_matrix",
                "observed_subject_count": 204,
                "historical_v1_artifacts_mutated": False,
            },
        },
        "authority": {
            "source_native_rig_skin_court_complete_for_selected_formats": True,
            "importer_equivalence_complete": False,
            "final_family_authority_minted": False,
            "split_minted": False,
            "training_authorized": False,
        },
        "training_authorized": False,
    }
    output = write(output_root / "CORPUS370_GLTF2_RIG_SKIN_SOURCE_SWEEP_V2.json", result)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "selected_subject_count": len(selected),
                "rollup": result["rollup"],
                "failure_reasons": result["failure_reasons"],
                "output_sha256": sha256(output),
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
