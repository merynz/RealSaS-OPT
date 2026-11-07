#!/usr/bin/env python3
"""Independent 352-subject component geometry candidate courts (NOT training).

Input: the exact V5 352-PASS/18-FAIL index and 352 original-admitted source-rest
geometry receipts. Produces canonical UDF, topology, per-triangle normals and
COMPACT component-sign candidates, then reruns the independent point-triangle/
solid-angle court. All source- and per-head training admission is separate.

Run a bounded pilot with --subject-id, or the full cohort with --all.
Resumability requires the input, producer, court and output identities to match.
Never fill missing signed queries with synthetic zero/positive targets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

from tools.corpus.build_component_geometry_authority_v1 import build as build_geometry
from tools.corpus.validate_component_geometry_authority_v1 import validate as court_geometry
from tools.corpus.validate_source_rest_geometry_v1 import validate as court_source_rest

SCHEMA = "RealSaS.Corpus370V5ComponentGeometryIndependentSweep.v1"
V5_SHA = "8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
REST_SHA = "384c92940e1ea8eec7bbb98cf319eeda5f89222c07b782b607998e2067f1d9c8"
SURFACE_SAMPLES = 32768
QUERY_SAMPLES = 8192
FAMILIES = ("SURFACE_GEOMETRY", "SURFACE_TOPOLOGY",
            "SIGNED_DISTANCE_OCCUPANCY", "ORIENTED_NORMALS")
ACCEPTABLE = ("QUALIFIED_CORE_CANDIDATE", "QUALIFIED_MASKED_CANDIDATE",
              "UNAVAILABLE_CANDIDATE")


def require(condition, code):
    if not condition:
        raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seal(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix(path.suffix + ".partial")
    stage.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    stage.replace(path)


def identity_for_run(rest_index, v5_index):
    require(sha(rest_index) == REST_SHA, "CORPUS370_GEOMETRY_REST_INDEX_SHA")
    require(sha(v5_index) == V5_SHA, "CORPUS370_GEOMETRY_V5_INDEX_SHA")
    return {
        "rest_index_sha256": REST_SHA,
        "v5_index_sha256": V5_SHA,
        "producer_sha256": sha(Path(build_geometry.__code__.co_filename)),
        "independent_court_sha256": sha(Path(court_geometry.__code__.co_filename)),
        "source_rest_court_sha256": sha(Path(court_source_rest.__code__.co_filename)),
        "coordinator_sha256": sha(Path(__file__)),
        "surface_samples": SURFACE_SAMPLES, "query_samples": QUERY_SAMPLES,
    }


def family_dispositions(court):
    require(court["status"] ==
            "PASS_COMPONENT_GEOMETRY_AUTHORITY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE",
            "CORPUS370_GEOMETRY_COURT_NOT_PASS")
    require(court["authority"]["independent_court_passed"] is True
            and court["authority"]["training_authorized"] is False
            and court["authority"]["final_subject_disposition_minted"] is False,
            "CORPUS370_GEOMETRY_COURT_AUTHORITY_OVERCLAIM")
    names = dict(court["family_candidate_outcomes"])
    require(set(names) == set(FAMILIES) and all(x in ACCEPTABLE for x in names.values()),
            "CORPUS370_GEOMETRY_FAMILY_CANDIDATE_SCHEMA")
    if names["SIGNED_DISTANCE_OCCUPANCY"] == "UNAVAILABLE_CANDIDATE":
        require(court["counts"]["signed_query_count"] == 0,
                "CORPUS370_GEOMETRY_FORGED_SIGN_UNAVAILABLE")
    else:
        require(0 < court["counts"]["signed_query_count"] <= QUERY_SAMPLES,
                "CORPUS370_GEOMETRY_SIGN_OVERFLOW_OR_ABSENCE")
    require(court["counts"]["query_count"] == QUERY_SAMPLES,
            "CORPUS370_GEOMETRY_QUERY_COUNT")
    return names


def conduct(rest_index_path, v5_index_path, output_root, subject_ids=None, *, all_subjects=False):
    rest_index_path = Path(rest_index_path).resolve()
    v5_index_path = Path(v5_index_path).resolve()
    output_root = Path(output_root).resolve()
    identity = identity_for_run(rest_index_path, v5_index_path)
    rest = json.loads(rest_index_path.read_text())
    v5 = json.loads(v5_index_path.read_text())
    require(rest.get("schema") ==
            "RealSaS.Corpus370V5SourceRestIndependentCourtSweep.v1"
            and rest["summary"]["independent_source_rest_pass"] == 352
            and rest["summary"]["not_proven_source_rest"] == 0
            and rest["summary"]["observation_excluded"] == 18
            and rest.get("training_authorized") is False,
            "CORPUS370_GEOMETRY_SOURCE_REST_PARTITION")
    require(v5.get("schema") == "RealSaS.Corpus370PreparedObservationPackets.v5"
            and (v5["prepared_subject_count"], v5["failure_count"]) == (352, 18)
            and v5.get("training_authorized") is False,
            "CORPUS370_GEOMETRY_OBSERVATION_PARTITION")
    rest_rows = dict(rest["subjects"])
    v5_rows = {r["subject_id"]: r for r in v5["subjects"]}
    excluded = {r["subject_id"] for r in v5["failures"]}
    require(len(rest_rows) == len(v5_rows) == 352
            and set(rest_rows) == set(v5_rows)
            and len(excluded) == 18
            and not set(rest_rows).intersection(excluded),
            "CORPUS370_GEOMETRY_SOURCE_REST_SCOPE")
    require(bool(subject_ids) != bool(all_subjects),
            "CORPUS370_GEOMETRY_EXPLICIT_COHORT_SELECTION_REQUIRED")
    selected = sorted(v5_rows) if all_subjects else sorted(set(subject_ids))
    require(bool(selected) and len(selected) == len(subject_ids or selected)
            and set(selected).issubset(v5_rows),
            "CORPUS370_GEOMETRY_SELECTED_NOT_OBSERVATION_PASS")
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = output_root / "COMPONENT_GEOMETRY_COURTS_INDEX.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        require(state.get("identity") == identity,
                "CORPUS370_GEOMETRY_RESUME_CODE_OR_SOURCE_DRIFT")
    else:
        state = {
            "schema": SCHEMA,
            "status": "PARTIAL_GEOMETRY_CANDIDATE_COURTS__NOT_TRAINING",
            "identity": identity, "universe_subject_count": 370,
            "observation_qualified_count": 352,
            "observation_excluded_count": 18,
            "excluded_observation_subject_ids": sorted(excluded),
            "subject_results": {},
            "authority": {
                "candidate_only": True, "model_targets_issued": False,
                "global_union_sign_proven": False,
                "family_admission_minted": False,
                "training_authorized": False,
            },
            "training_authorized": False,
        }
    require(state["excluded_observation_subject_ids"] == sorted(excluded)
            and state["observation_qualified_count"] == 352,
            "CORPUS370_GEOMETRY_RESUME_SCOPE")
    for i, sid in enumerate(selected, 1):
        old = rest_rows[sid]
        observation = v5_rows[sid]
        require(old["status"] == "PASS_SOURCE_REST_INDEPENDENT_COURT"
                and old["input"]["source_sha256"] == observation["source_sha256"],
                "CORPUS370_GEOMETRY_OLD_PARENT_DRIFT:" + sid)
        source_rest_manifest_path = Path(old["manifest"]["path"]).resolve()
        require(source_rest_manifest_path.is_file()
                and sha(source_rest_manifest_path) == old["manifest"]["sha256"],
                "CORPUS370_GEOMETRY_REST_MANIFEST_DRIFT:" + sid)
        src_manifest = json.loads(source_rest_manifest_path.read_text())
        require(src_manifest["source_sha256"] == observation["source_sha256"]
                and src_manifest["subject_id"] == sid,
                "CORPUS370_GEOMETRY_REST_SUBJECT_IDENTITY:" + sid)
        geometry_npz = Path(src_manifest["output"]["path"]).resolve()
        require(sha(geometry_npz) == src_manifest["output"]["sha256"],
                "CORPUS370_GEOMETRY_REST_ARRAY_DRIFT:" + sid)
        snapshot_ref = src_manifest["parents"]["source_snapshot"]
        camera_ref = observation["bindings"]["camera_bundle"]
        require(sha(Path(snapshot_ref["path"])) == snapshot_ref["sha256"]
                and sha(Path(camera_ref["path"])) == camera_ref["sha256"],
                "CORPUS370_GEOMETRY_SNAPSHOT_CAMERA_DRIFT:" + sid)
        # The source-rest court does not import any component-geometry producer.
        src_court = court_source_rest(source_rest_manifest_path)
        require(src_court["authority"]["independent_source_rest_reconstruction_passed"],
                "CORPUS370_GEOMETRY_REST_RECOURT_FAIL:" + sid)
        input_proof = {
            "source_sha256": observation["source_sha256"],
            "import_sha256": old["input"]["parent_import_sha256"],
            "source_rest_manifest_sha256": old["manifest"]["sha256"],
            "source_rest_arrays_sha256": src_manifest["output"]["sha256"],
            "source_snapshot_sha256": snapshot_ref["sha256"],
            "camera_bundle_sha256": camera_ref["sha256"],
        }
        root = output_root / "subjects" / sid
        manifest_path = root / "COMPONENT_GEOMETRY_AUTHORITY_MANIFEST_V1.json"
        court_path = root / "COMPONENT_GEOMETRY_INDEPENDENT_COURT_V1.json"
        previous = state["subject_results"].get(sid)
        started = time.monotonic()
        try:
            if previous and previous.get("status") == "PASS_COMPONENT_GEOMETRY_COURT":
                require(previous.get("input") == input_proof
                        and previous["manifest"]["sha256"] == sha(manifest_path)
                        and previous["court"]["sha256"] == sha(court_path),
                        "CORPUS370_GEOMETRY_RESUME_PROOF_DRIFT:" + sid)
                manifest = json.loads(manifest_path.read_text())
                court = court_geometry(manifest)
                require(court["family_candidate_outcomes"] == previous["family_candidate_outcomes"],
                        "CORPUS370_GEOMETRY_RESUME_FAMILY_DRIFT:" + sid)
                disposition = "RECOURTED_EXACT_CANDIDATE"
            else:
                root.mkdir(parents=True, exist_ok=True)
                manifest = build_geometry(
                    sid, geometry_npz, snapshot_ref["path"], camera_ref["path"],
                    root, surface_samples=SURFACE_SAMPLES, query_samples=QUERY_SAMPLES,
                )
                court = court_geometry(manifest)
                seal(court_path, court)
                disposition = "NEW_INDEPENDENT_COURT"
            families = family_dispositions(court)
            require(manifest["authority"]["training_authorized"] is False
                    and manifest["policy"]["union_occupancy_authority"] == "NOT_MINTED_IN_V1"
                    and manifest["counts"]["signed_query_count"] ==
                        court["counts"]["signed_query_count"],
                    "CORPUS370_GEOMETRY_PRODUCER_SIGN_AUTHORITY_OVERCLAIM:" + sid)
            state["subject_results"][sid] = {
                "status": "PASS_COMPONENT_GEOMETRY_COURT",
                "input": input_proof,
                "manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)},
                "court": {"path": str(court_path), "sha256": sha(court_path)},
                "family_candidate_outcomes": families,
                "coverage": court["coverage"],
                "counts": court["counts"],
                "disposition": disposition,
                "training_authorized": False,
            }
            print(json.dumps({
                "event": "CORPUS370_COMPONENT_COURT_PASS",
                "done": i, "selected_total": len(selected),
                "subject_id": sid, "elapsed_seconds": round(time.monotonic()-started, 3),
                "family_candidates": families,
                "signed_query_count": court["counts"]["signed_query_count"],
            }, sort_keys=True), flush=True)
        except (KeyboardInterrupt, MemoryError):
            raise
        except Exception as exc:
            path = output_root / "failures" / (sid + ".json")
            failure = {
                "schema": "RealSaS.Corpus370V5ComponentGeometryCourtFailure.v1",
                "subject_id": sid, "input": input_proof,
                "error_type": type(exc).__name__, "reason": str(exc),
                "traceback": traceback.format_exc(),
                "classification": "COURT_FAIL__NO_GEOMETRY_FAMILY_ADMISSION",
                "training_authorized": False,
            }
            seal(path, failure)
            state["subject_results"][sid] = {
                "status": "FAIL_COMPONENT_GEOMETRY_COURT", "input": input_proof,
                "failure": {"path": str(path), "sha256": sha(path)},
                "reason": str(exc), "training_authorized": False,
            }
            print(json.dumps({
                "event": "CORPUS370_COMPONENT_COURT_FAIL",
                "done": i, "selected_total": len(selected),
                "subject_id": sid, "elapsed_seconds": round(time.monotonic()-started, 3),
                "reason": str(exc),
            }, sort_keys=True), flush=True)
        finally:
            seal(checkpoint, state)
            gc.collect()
    outcomes = {k: state["subject_results"][k] for k in selected}
    passed = sum(r["status"] == "PASS_COMPONENT_GEOMETRY_COURT" for r in outcomes.values())
    failed = len(outcomes) - passed
    family_counts = {
        family: dict(Counter(r["family_candidate_outcomes"][family]
                             for r in outcomes.values()
                             if r["status"] == "PASS_COMPONENT_GEOMETRY_COURT"))
        for family in FAMILIES
    }
    summary = {
        "requested_subject_count": len(selected),
        "subject_outcome_count": len(outcomes),
        "independent_court_pass_count": passed,
        "independent_court_fail_count": failed,
        "unexamined_observation_pass_count": 352-len(state["subject_results"]),
        "observation_excluded_count": 18,
        "family_candidate_counts_for_selected": family_counts,
        "family_admission_minted": False,
        "training_authorized": False,
    }
    state["status"] = ("ALL_352_COURTS_ACCOUNTED__NOT_TRAINING" if
                       len(state["subject_results"]) == 352 else
                       "PARTIAL_COMPONENT_COURT_SWEEP__NOT_TRAINING")
    state["latest_selection_summary"] = summary
    seal(checkpoint, state)
    print(json.dumps({"event": "CORPUS370_COMPONENT_COURT_SUMMARY", **summary},
                     sort_keys=True), flush=True)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-rest-courts-index",type=Path,required=True)
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--subject-id",action="append",dest="subject_ids")
    choice.add_argument("--all",action="store_true",dest="all_subjects")
    a=p.parse_args()
    conduct(a.source_rest_courts_index,a.v5_index,a.output_root,
            a.subject_ids,all_subjects=a.all_subjects)


if __name__=="__main__":
    main()
