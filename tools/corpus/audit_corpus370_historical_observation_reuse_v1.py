"""Read-only SHA-complete reusability audit for historical Corpus370 PASS packets.

This does not turn historical V2/V3 packets into V4 admissions. It only asks
whether their exact admitted original import bytes and all eight source renders
still exist, match the already sealed strict old PASS, and could be transported
as separately typed legacy evidence without recomputation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SCHEMA = "RealSaS.Corpus370HistoricalObservationReuseAudit.v1"
PARENT_SCHEMA = "RealSaS.Corpus370PreparedObservationPackets.v3"
POLICY_SCHEMA = "RealSaS.IrisObservationPreparationPolicy.v1"


def sha(path):
    path = Path(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(cond, code):
    if not cond:
        raise ValueError(code)


def _check(path, expected, code):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(code + ":MISSING_HASH")
    if not path.is_file() or sha(path) != expected:
        raise ValueError(code + ":HASH_DRIFT")


def inspect(row, original_policy):
    sid = str(row["subject_id"])
    packet = Path(row["packet_path"]).resolve()
    try:
        require(packet.name == sid and packet.parent.name == "packet"
                and packet.parent.parent.parent.name == "objects",
                "REUSE_PACKET_LAYOUT")
        admission_path = packet / "IRIS_OBSERVATION_ADMISSION.json"
        admission = json.loads(admission_path.read_text())
        require(admission.get("schema") == "RealSaS.IrisBindObservationAdmission.v1"
                and admission.get("status") == "PASS_IRIS_BIND_OBSERVATIONS",
                "REUSE_NOT_HISTORICAL_PASS")
        require(admission.get("subject_id") == sid
                and admission.get("source_sha256") == row["source_sha256"],
                "REUSE_SOURCE_IDENTITY")
        policy = admission.get("policy") or {}
        for key in ("maximum_relative_vertex_error", "maximum_mask_gap_px",
                    "resolution", "view_count", "pose"):
            require(policy.get(key) == original_policy[key],
                    "REUSE_OLD_POLICY_DIFFERENT:" + key)
        candidate_path = packet / "SOURCE_REST_RENDER_CANDIDATE.json"
        candidate = json.loads(candidate_path.read_text())
        _check(candidate_path, admission.get("candidate_sha256"), "REUSE_CANDIDATE")
        require(candidate.get("subject_id") == sid
                and candidate.get("source_sha256") == row["source_sha256"],
                "REUSE_CANDIDATE_IDENTITY")
        camera = packet / "camera_bundle.json"
        snapshot = packet / "SOURCE_SCENE_SNAPSHOT.json"
        _check(camera, admission.get("camera_sha256"), "REUSE_CAMERA_ADMISSION")
        _check(camera, (candidate.get("camera_bundle") or {}).get("sha256"),
               "REUSE_CAMERA_CANDIDATE")
        _check(snapshot, (candidate.get("source_scene_snapshot") or {}).get("sha256"),
               "REUSE_SCENE_SNAPSHOT")
        imported = packet.parent.parent / "imported" / sid / "imported.json"
        _check(imported, admission.get("imported_geometry_sha256"),
               "REUSE_ORIGINAL_ADMITTED_IMPORT")
        source_geometry = json.loads(imported.read_text())
        require((source_geometry.get("import") or {}).get("source_sha256")
                == row["source_sha256"],
                "REUSE_IMPORTED_SOURCE_IDENTITY")
        indexed = admission.get("views") or []
        declared = candidate.get("views") or []
        count = original_policy["view_count"]
        require(len(indexed) == len(declared) == count,
                "REUSE_VIEW_COUNT")
        require([v.get("view_index") for v in indexed] == list(range(count))
                and [v.get("view_index") for v in declared] == list(range(count)),
                "REUSE_VIEW_IDS")
        for i in range(count):
            png = packet / f"V{i}.png"
            mask = packet / f"V{i}.mask.bin"
            _check(png, indexed[i].get("raster_sha256"), "REUSE_RASTER_ADMISSION")
            _check(png, (declared[i].get("raster") or {}).get("sha256"),
                   "REUSE_RASTER_CANDIDATE")
            _check(mask, indexed[i].get("mask_sha256"), "REUSE_MASK_ADMISSION")
            _check(mask, (declared[i].get("foreground_mask") or {}).get("sha256"),
                   "REUSE_MASK_CANDIDATE")
        return {"subject_id": sid, "status": "EXACT_HISTORICAL_V3_PASS_BYTES_REUSABLE",
                "legacy_admission_sha256": sha(admission_path),
                "original_import_sha256": sha(imported),
                "eight_view_raster_and_mask_hashes_verified": True}
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as e:
        return {"subject_id": sid, "status": "NOT_PROVEN",
                "reason": str(e).split(":")[0],
                "eight_view_raster_and_mask_hashes_verified": False}


def audit(index, roster, policy):
    require(index.get("schema") == PARENT_SCHEMA
            and index.get("training_authorized") is False,
            "REUSE_INDEX_SCOPE")
    require(roster.get("schema") == "RealSaS.Corpus370ExactSourceRoster.v1"
            and roster.get("subject_count") == 370
            and roster.get("training_authorized") is False,
            "REUSE_ROSTER_SCOPE")
    require(policy.get("schema") == POLICY_SCHEMA
            and policy.get("training_authorized") is False
            and policy.get("view_count") == 8,
            "REUSE_POLICY_SCOPE")
    candidates = {row["subject_id"]: row for row in roster["subjects"]}
    successes = {row["subject_id"]: row for row in index.get("subjects") or []}
    failures = {row["subject_id"]: row for row in index.get("failures") or []}
    require(len(candidates) == 370
            and len(successes) == index["prepared_subject_count"]
            and len(failures) == index["failure_count"]
            and not set(successes).intersection(failures)
            and set(successes).union(failures) == set(candidates),
            "REUSE_CANDIDATE_PARTITION")
    for sid, row in list(successes.items()) + list(failures.items()):
        require(row["source_sha256"] == candidates[sid]["source_sha256"],
                "REUSE_SOURCE_ROSTER_DRIFT:" + sid)
    outcomes = [inspect(successes[sid], policy) for sid in sorted(successes)]
    states = Counter(row["status"] for row in outcomes)
    return {
        "schema": SCHEMA,
        "status": "HISTORICAL_V3_PASS_BYTES_AUDITED__NO_NEW_ADMISSION",
        "candidate_count": 370, "original_success_count": len(successes),
        "original_failure_count": len(failures),
        "exact_old_pass_reusable_count": states["EXACT_HISTORICAL_V3_PASS_BYTES_REUSABLE"],
        "old_pass_not_proven_count": states["NOT_PROVEN"],
        "failure_subjects_unchanged": True,
        "subject_evidence": outcomes,
        "authority": {"new_v4_admission_minted": False,
                      "source_surface_family_qualified": False,
                      "platform_attempt_created": False,
                      "training_authorized": False},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-index", type=Path, required=True)
    p.add_argument("--expected-parent-sha256", required=True)
    p.add_argument("--roster", type=Path, required=True)
    p.add_argument("--expected-roster-sha256", required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    _check(args.parent_index, args.expected_parent_sha256, "REUSE_PARENT_SHA")
    _check(args.roster, args.expected_roster_sha256, "REUSE_ROSTER_SHA")
    parent = json.loads(args.parent_index.read_text())
    result = audit(parent, json.loads(args.roster.read_text()),
                   json.loads(args.policy.read_text()))
    result["parents"] = {
        "historical_index_sha256": sha(args.parent_index),
        "roster_sha256": sha(args.roster),
        "observation_policy_sha256": sha(args.policy),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in
          ("candidate_count","original_success_count","original_failure_count",
           "exact_old_pass_reusable_count","old_pass_not_proven_count")},
          sort_keys=True),flush=True)


if __name__=="__main__":main()
