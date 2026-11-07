#!/usr/bin/env python3
"""Independently reconstruct Corpus370 source-rest geometry from exact V5 PASS bytes.

Read-only wrt source packets. One scientific outcome per observation-PASS subject,
resumable with SHA-checked input fingerprint and independently rerun court. A
source-rest PASS is not a model geometry/sign/surface family PASS or training
authority. All 18 observation FAIL rows remain explicitly represented.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import traceback

from tools.corpus.source_rest_geometry_v1 import build
from tools.corpus.validate_source_rest_geometry_v1 import validate

SCHEMA = "RealSaS.Corpus370V5SourceRestIndependentCourtSweep.v1"
INDEX_SHA = "8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
ORIGINAL_V3_SHA = "5007b0863eddce70833ad48c4bc26af92a7abd79b618c8dc5c69ade7f43840a7"


def require(value, code):
    if not value:
        raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def seal(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def row_input(row):
    sid = row["subject_id"]
    packet = Path(row["packet_path"]).resolve()
    refs = dict(row.get("bindings") or {})
    if row["evidence_producer"] == "HISTORICAL_V3_PASS_BYTES_RECHECKED":
        import_ref = row["admitted_import_geometry"]
        require(row.get("new_v4_admission_claimed") is False,
                "SOURCE_COURTS_V5_LEGACY_V4_OVERCLAIM:" + sid)
    else:
        require(row["evidence_producer"] == "NEW_V4_REPAIR_FROM_ORIGINAL_FAIL",
                "SOURCE_COURTS_V5_UNKNOWN_PRODUCER:" + sid)
        import_ref = refs["admitted_import_geometry"]
        require(row.get("parent_failure_receipt"),
                "SOURCE_COURTS_V5_REPAIR_WITHOUT_OLD_FAIL:" + sid)
    paths = {
        "import": import_ref,
        "admission": refs["observation_admission"],
        "candidate": refs["source_rest_candidate"],
        "snapshot": refs["source_scene_snapshot"],
        "camera": refs["camera_bundle"],
    }
    for key,ref in paths.items():
        p=Path(str(ref.get("path") or "")).resolve()
        require(p.is_file() and sha(p)==ref.get("sha256"),
                "SOURCE_COURTS_V5_SOURCE_BYTES_DRIFT:"+sid+":"+key)
        if key == "import":
            if row["evidence_producer"] == "NEW_V4_REPAIR_FROM_ORIGINAL_FAIL":
                require(p == (packet/"imported.json").resolve(),
                        "SOURCE_COURTS_V5_NEW_IMPORT_PATH")
            else:
                require(p == (packet.parent.parent/"imported"/sid/"imported.json").resolve(),
                        "SOURCE_COURTS_V5_LEGACY_IMPORT_PATH")
        else:
            require(p == (packet / {
                "admission":"IRIS_OBSERVATION_ADMISSION.json",
                "candidate":"SOURCE_REST_RENDER_CANDIDATE.json",
                "snapshot":"SOURCE_SCENE_SNAPSHOT.json",
                "camera":"camera_bundle.json",
            }[key]).resolve(), "SOURCE_COURTS_V5_PACKET_PATH_DRIFT:"+key)
    return paths


def sweep(index_path, output_root):
    index_path=Path(index_path).resolve()
    output_root=Path(output_root).resolve()
    require(index_path.is_file() and sha(index_path)==INDEX_SHA,
            "SOURCE_COURTS_V5_PINNED_INDEX_DRIFT")
    d=json.loads(index_path.read_text())
    require(d.get("schema")=="RealSaS.Corpus370PreparedObservationPackets.v5"
            and d.get("subject_count")==370
            and (d.get("prepared_subject_count"),d.get("failure_count"))==(352,18)
            and (d.get("historical_v3_pass_preserved_count"),
                 d.get("new_v4_repaired_count"))==(294,58)
            and d.get("parent",{}).get("sha256")==ORIGINAL_V3_SHA
            and d.get("training_authorized") is False,
            "SOURCE_COURTS_V5_UNEXPECTED_UNIVERSE")
    candidate=list(d["subjects"])
    excluded=list(d["failures"])
    ids=[r["subject_id"] for r in candidate+excluded]
    require(len(ids)==len(set(ids))==370, "SOURCE_COURTS_V5_CANDIDATE_DUPLICATE")
    identity={
        "input_index_sha256":INDEX_SHA,
        "producer_code_sha256":sha(Path(__file__)),
        "geometry_builder_sha256":sha(Path(build.__code__.co_filename)),
        "independent_court_sha256":sha(Path(validate.__code__.co_filename)),
    }
    checkpoint=output_root/"SOURCE_REST_COURTS_INDEX.json"
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text())
        require(state.get("identity")==identity,
                "SOURCE_COURTS_V5_RESUME_CODE_OR_INDEX_DRIFT")
    else:
        state={
            "schema":SCHEMA,"status":"PARTIAL_SOURCE_REST_COURT_SWEEP__NO_TRAINING",
            "identity":identity,"candidate_count":370,
            "observation_qualified_count":352,"observation_failed_count":18,
            "subjects":{},"excluded_observation_failures":[
                {"subject_id":r["subject_id"],"reason":r.get("reason"),
                 "source_sha256":r["source_sha256"],
                 "status":"OBSERVATION_FAIL_NOT_SOURCE_REST_COURTED"}
                for r in sorted(excluded,key=lambda x:x["subject_id"])
            ],
            "authority":{"family_authority_minted":False,
                         "platform_attempt_created":False,
                         "training_authorized":False},
            "training_authorized":False,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    for i,row in enumerate(sorted(candidate,key=lambda r:r["subject_id"]),1):
        sid=row["subject_id"]
        src_sha=row["source_sha256"]
        previous=state["subjects"].get(sid)
        scope={"subject_id":sid,"source_sha256":src_sha,
               "producer_kind":row["evidence_producer"]}
        try:
            refs=row_input(row)
            scope["parent_import_sha256"]=refs["import"]["sha256"]
            root=output_root/"subjects"/sid
            root.mkdir(parents=True,exist_ok=True)
            manifest_path=root/"SOURCE_REST_GEOMETRY_MANIFEST_V1.json"
            if previous and previous.get("status")=="PASS_SOURCE_REST_INDEPENDENT_COURT":
                require(previous.get("input")==scope,
                        "SOURCE_COURTS_V5_RESUME_SUBJECT_DRIFT:"+sid)
                require(manifest_path.is_file()
                        and sha(manifest_path)==previous["manifest"]["sha256"],
                        "SOURCE_COURTS_V5_RESUME_MANIFEST_DRIFT:"+sid)
                court=validate(manifest_path)
                require(court["subject_id"]==sid and court["source_sha256"]==src_sha
                        and court["authority"]["independent_source_rest_reconstruction_passed"],
                        "SOURCE_COURTS_V5_RESUME_COURT_DRIFT:"+sid)
                status="REUSED_EXACT_COURTED_PASS"
            else:
                manifest=build(sid,src_sha,refs["import"]["path"],
                               refs["admission"]["path"],refs["candidate"]["path"],
                               refs["snapshot"]["path"],root)
                require(manifest["schema"]=="RealSaS.AdmittedSourceRestGeometry.v1",
                        "SOURCE_COURTS_V5_BUILDER_SCHEMA:"+sid)
                court=validate(manifest_path)
                require(court["subject_id"]==sid and court["source_sha256"]==src_sha
                        and court["authority"]["independent_source_rest_reconstruction_passed"]
                        and court["authority"]["training_authorized"] is False,
                        "SOURCE_COURTS_V5_INDEPENDENT_COURT_NOT_PASS:"+sid)
                status="MEASURED_INDEPENDENT_SOURCE_REST_PASS"
            state["subjects"][sid]={
                "input":scope,"status":"PASS_SOURCE_REST_INDEPENDENT_COURT",
                "disposition":status,
                "manifest":{"path":str(manifest_path),"sha256":sha(manifest_path)},
                "counts":court["counts"],"training_authorized":False,
            }
            print(json.dumps({"event":"SOURCE_REST_COURT_PASS","done":i,
                              "total":len(candidate),"subject_id":sid,
                              "disposition":status,"counts":court["counts"]}),flush=True)
        except (KeyboardInterrupt, MemoryError):
            raise
        except Exception as exc:
            failure={
                "schema":"RealSaS.Corpus370SourceRestCourtFailure.v1",
                "subject_id":sid,"source_sha256":src_sha,
                "error_type":type(exc).__name__,"reason":str(exc),
                "traceback":traceback.format_exc(),
                "classification":"NOT_QUALIFIED__REQUIRES_FAILURE_TRIAGE",
                "training_authorized":False,
            }
            path=output_root/"failures"/(sid+".json")
            seal(path,failure)
            state["subjects"][sid]={
                "input":scope,
                "status":"FAIL_SOURCE_REST_UNPROVEN",
                "reason":str(exc),
                "failure":{"path":str(path),"sha256":sha(path)},
                "training_authorized":False,
            }
            print(json.dumps({"event":"SOURCE_REST_COURT_FAIL","done":i,
                              "total":len(candidate),"subject_id":sid,
                              "error_type":type(exc).__name__,
                              "reason":str(exc)}),flush=True)
        finally:
            seal(checkpoint,state)
            gc.collect()
    pass_count=sum(r["status"]=="PASS_SOURCE_REST_INDEPENDENT_COURT"
                   for r in state["subjects"].values())
    fail_count=len(state["subjects"])-pass_count
    require(len(state["subjects"])==352, "SOURCE_COURTS_V5_INCOMPLETE_SWEEP")
    state["status"]="FULL_SOURCE_REST_COURT_SWEEP_COMPLETED__FAMILY_ADMISSION_SEPARATE"
    state["summary"]={
        "independent_source_rest_pass":pass_count,
        "not_proven_source_rest":fail_count,
        "observation_excluded":18,
        "candidate_count":370,
        "family_authority_minted":False,
        "training_authorized":False,
    }
    seal(checkpoint,state)
    print(json.dumps(state["summary"],sort_keys=True),flush=True)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    a=p.parse_args()
    sweep(a.v5_index,a.output_root)


if __name__=="__main__":
    main()
