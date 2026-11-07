#!/usr/bin/env python3
"""Exact 370 source-rest -> StdGEN-like semantic SOURCE GROUP candidate sweep.

All 370 original subjects are accounted for. Only the 352 admitted source-rest
subjects may produce source-face groups; the other 18 remain observation FAIL.
Geometry is *re-used* from its independently courted Stage50 NPZ. Source names
suggest concepts for review, but no label/affordance/view mask/training PASS is
minted by this command.

The result contains private artist object names / per-face support, so use only
in the existing SHA-locked PRIVATE Drive evidence area. Never upload raw output
to a public GitHub Actions artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import traceback

import numpy as np

from models.iris.truth.semantic_prior_contract_v1 import validate_policy
from models.iris.truth.semantic_source_groups_v1 import (
    build_semantic_source_groups, validate_hint_policy,
)
from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import (
    INDEX_SHA as V5_SHA, row_input,
)

SCHEMA = "RealSaS.Corpus370SemanticSourceGroupSweep.v1"
REST_SHA = "384c92940e1ea8eec7bbb98cf319eeda5f89222c07b782b607998e2067f1d9c8"
ROSTER_SHA = "925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"
REST_INDEX_SCHEMA = "RealSaS.Corpus370V5SourceRestIndependentCourtSweep.v1"
REST_MANIFEST_SCHEMA = "RealSaS.AdmittedSourceRestGeometry.v1"


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok: bool, key: str) -> None:
    if not ok:
        raise ValueError("CORPUS370_SEMANTIC_SOURCE_SWEEP_" + key)


def read_exact(ref: dict, schema: str | None = None) -> dict:
    require(isinstance(ref, dict) and isinstance(ref.get("path"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(ref.get("sha256") or "")) is not None,
            "PARENT_REF")
    path = Path(ref["path"])
    require(path.is_file() and sha(path) == ref["sha256"],
            "PARENT_SHA_MISMATCH")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        require(payload.get("schema") == schema, "PARENT_SCHEMA")
    return payload


def preflight(v5: dict, roster: dict, rest: dict, policy: dict, hints: dict):
    require(v5.get("schema") == "RealSaS.Corpus370PreparedObservationPackets.v5"
            and (v5.get("subject_count"), v5.get("prepared_subject_count"),
                 v5.get("failure_count")) == (370, 352, 18)
            and v5.get("training_authorized") is False,
            "V5_COHORT")
    require(roster.get("schema") == "RealSaS.Corpus370ExactSourceRoster.v1"
            and roster.get("subject_count") == 370
            and roster.get("training_authorized") is False,
            "ROSTER")
    require(rest.get("schema") == REST_INDEX_SCHEMA
            and rest.get("observation_qualified_count") == 352
            and rest.get("observation_failed_count") == 18
            and rest.get("training_authorized") is False,
            "REST_INDEX")
    validate_policy(policy)
    validate_hint_policy(hints, concepts=set(policy["concepts"]))
    source = {r["subject_id"]: r for r in roster["subjects"]}
    prepared = {r["subject_id"]: r for r in v5["subjects"]}
    excluded = {r["subject_id"]: r for r in v5["failures"]}
    require(len(source) == 370 and len(prepared) == 352 and len(excluded) == 18
            and set(source) == set(prepared) | set(excluded)
            and not set(prepared) & set(excluded), "UNIVERSE_PARTITION")
    require(set(rest["subjects"]) == set(prepared), "REST_SUBJECT_SET")
    require(all(prepared[s]["source_sha256"] == source[s]["source_sha256"]
                for s in prepared), "SOURCE_SHA_MATCH")
    return source, prepared, excluded


def one_subject(row: dict, v5_row: dict, court: dict,
                policy: dict, hints: dict) -> dict:
    sid = row["subject_id"]
    require(court.get("status") == "PASS_SOURCE_REST_INDEPENDENT_COURT"
            and court.get("training_authorized") is False,
            "REST_NOT_INDEPENDENT_PASS")
    manifest = read_exact(court["manifest"], REST_MANIFEST_SCHEMA)
    require(manifest.get("subject_id") == sid
            and manifest.get("source_sha256") == row["source_sha256"]
            and manifest.get("status") == "PASS_ADMITTED_SOURCE_REST_CANDIDATE__FAMILY_COURTS_REQUIRED",
            "REST_MANIFEST_IDENTITY")
    # Admitted imported.json is independently referenced by the exact rest
    # manifest. It remains a source-to-group teacher-only feature.
    imported = read_exact(manifest["parents"]["admitted_import"])
    source_observation_refs = row_input(v5_row)
    require(source_observation_refs["import"]["sha256"] ==
            manifest["parents"]["admitted_import"]["sha256"],
            "ADMITTED_IMPORT_DRIFT")
    require(source_observation_refs["admission"]["sha256"] ==
            manifest["parents"]["observation_admission"]["sha256"],
            "ADMISSION_DRIFT")
    for parent in manifest["parents"].values():
        require(Path(parent["path"]).is_file() and
                sha(Path(parent["path"])) == parent["sha256"],
                "SOURCE_INPUT_PARENT_REOPEN")
    output_ref = manifest["output"]
    require(Path(output_ref["path"]).is_file()
            and sha(Path(output_ref["path"])) == output_ref["sha256"],
            "STAGE50_NPZ_DRIFT")
    with np.load(output_ref["path"], allow_pickle=False) as npz:
        require(set(npz.files) == {
            "rest_vertices_source_world", "faces", "face_owner_privileged"
        }, "STAGE50_PAYLOAD")
        vertices = np.asarray(npz["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(npz["faces"], dtype=np.int64)
        face_owner = np.asarray(npz["face_owner_privileged"], dtype=np.int64)
    owner_names = manifest["owner_order"]
    require(isinstance(owner_names, list) and len(owner_names) > 0
            and all(isinstance(x, str) for x in owner_names)
            and len(owner_names) == len(set(owner_names)), "OWNER_ORDER")
    original_object_names = sorted(o["name"] for o in imported["objects"])
    require(owner_names == original_object_names, "OWNER_IDENTITY_DRIFT")
    # Stage50 importer does not expose material slots. All faces use an
    # explicit unknown/default material 0: no fake material teacher labels.
    snapshot = {
        "vertices": vertices,
        "faces": faces,
        "face_owner": face_owner,
        "face_material": np.zeros(len(faces), dtype=np.int32),
        "owners": [
            {"anonymous_owner_index": i, "source_import_name": name,
             "material_slot_names": []}
            for i, name in enumerate(owner_names)
        ],
    }
    candidate = build_semantic_source_groups(
        subject_id=sid, source_sha256=row["source_sha256"],
        snapshot=snapshot, hint_policy=hints, semantic_prior_policy=policy,
    )
    candidate.update({
        "stage50_manifest_sha256": court["manifest"]["sha256"],
        "stage50_geometry_sha256": output_ref["sha256"],
        "admitted_source_import_sha256": source_observation_refs["import"]["sha256"],
        "stage50_importer_object_names_privileged": True,
        "source_material_names_available": False,
        "semantic_family_court_passed": False,
        "training_authorized": False,
    })
    return candidate


def sweep(v5_path: Path, roster_path: Path, rest_path: Path, policy_path: Path,
          hint_path: Path, output: Path, max_subjects: int = 0) -> dict:
    for path, expected in ((v5_path, V5_SHA), (roster_path, ROSTER_SHA),
                           (rest_path, REST_SHA)):
        require(sha(path) == expected, "FROZEN_PARENT_SHA")
    v5 = json.loads(v5_path.read_text())
    roster = json.loads(roster_path.read_text())
    rest = json.loads(rest_path.read_text())
    policy = json.loads(policy_path.read_text())
    hints = json.loads(hint_path.read_text())
    source, prepared, excluded = preflight(v5, roster, rest, policy, hints)
    require(type(max_subjects) is int and max_subjects >= 0, "MAX_SUBJECTS")
    sample_ids = sorted(prepared)[:max_subjects] if max_subjects else sorted(prepared)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    concept_counts: Counter = Counter()
    hint_counts: Counter = Counter()
    statuses: Counter = Counter()
    group_count = face_count = 0
    for index, sid in enumerate(sorted(source), 1):
        if sid in excluded:
            item = {"subject_id": sid, "status": "OBSERVATION_EXCLUDED",
                    "source_sha256": source[sid]["source_sha256"]}
        elif sid not in sample_ids:
            item = {"subject_id": sid, "status": "SOURCE_GROUP_UNMEASURED_IN_PILOT",
                    "source_sha256": source[sid]["source_sha256"]}
        else:
            try:
                result = one_subject(source[sid], prepared[sid],
                                     rest["subjects"][sid], policy, hints)
                out = output / "subjects" / sid / "SEMANTIC_SOURCE_GROUPS_CANDIDATE_V1.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                          allow_nan=False) + "\n")
                item = {"subject_id": sid, "status": "MEASURED_SOURCE_GROUP_CANDIDATE_ONLY",
                        "source_sha256": source[sid]["source_sha256"],
                        "private_receipt": {"path": str(out), "sha256": sha(out)},
                        "group_count": result["group_count"],
                        "source_face_count": result["source_face_count"]}
                group_count += result["group_count"]
                face_count += result["source_face_count"]
                for group in result["groups"]:
                    hint_counts[group["hint_status"]] += 1
                    concept_counts[group["suggested_concept"]] += 1
            except (KeyboardInterrupt, MemoryError):
                raise
            except Exception as exc:
                item = {"subject_id": sid, "status": "SOURCE_GROUP_FAILURE",
                        "source_sha256": source[sid]["source_sha256"],
                        "reason": str(exc), "error_type": type(exc).__name__}
                fp = output / "failures" / (sid + ".txt")
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(traceback.format_exc())
        rows.append(item)
        statuses[item["status"]] += 1
        if index % 20 == 0 or index == 370:
            print(json.dumps({"done": index, "total": 370,
                              "status_counts": dict(statuses)}, sort_keys=True),
                  flush=True)
    report = {
        "schema": SCHEMA,
        "status": "PILOT_ONLY__NO_SEMANTIC_AUTHORITY" if max_subjects else
                  "FULL_370_GROUP_CANDIDATE_CENSUS__NO_SEMANTIC_AUTHORITY",
        "subject_count": 370,
        "source_observation_qualified_count": 352,
        "excluded_observation_count": 18,
        "attempted_group_subject_count": len(sample_ids),
        "source_group_count": group_count,
        "source_face_count": face_count,
        "status_counts": dict(sorted(statuses.items())),
        "semantic_hint_status_counts": dict(sorted(hint_counts.items())),
        "semantic_concept_hint_counts": dict(sorted(concept_counts.items())),
        "parents": {
            "exact_source_roster_sha256": ROSTER_SHA,
            "v5_index_sha256": V5_SHA,
            "independent_source_rest_index_sha256": REST_SHA,
            "semantic_policy_sha256": sha(policy_path),
            "semantic_hint_policy_sha256": sha(hint_path),
        },
        "subjects": rows,
        "source_names_privileged_teacher_only": True,
        "semantic_source_material_names_qualified": False,
        "semantic_region_training_labels_authorized": False,
        "semantic_view_rasters_authorized": False,
        "family_court_passed": False,
        "training_authorized": False,
    }
    out = output / "CORPUS370_SEMANTIC_SOURCE_GROUP_SWEEP_V1.json"
    out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-index", required=True, type=Path)
    parser.add_argument("--exact-source-roster", required=True, type=Path)
    parser.add_argument("--source-rest-index", required=True, type=Path)
    parser.add_argument("--semantic-policy", required=True, type=Path)
    parser.add_argument("--semantic-hint-policy", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-subjects", type=int, default=0)
    args = parser.parse_args()
    result = sweep(args.v5_index, args.exact_source_roster,
                   args.source_rest_index, args.semantic_policy,
                   args.semantic_hint_policy, args.output_root,
                   max_subjects=args.max_subjects)
    print(json.dumps({
        "status": result["status"],
        "subjects": result["subject_count"],
        "attempted_group_subject_count": result["attempted_group_subject_count"],
        "status_counts": result["status_counts"],
        "source_group_count": result["source_group_count"],
        "training_authorized": result["training_authorized"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
