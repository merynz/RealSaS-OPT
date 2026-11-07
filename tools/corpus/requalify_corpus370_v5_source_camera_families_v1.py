#!/usr/bin/env python3
"""Re-court Corpus370 V5 source import + 8-view camera from SHA-bound source truth.

Uses the complete 352/18 V5 partition, all 352 separately independent source
rest courts, and individually SHA-rechecked original admitted source packets.
Reuses the older camera geometry-independent observational auditor without
reusing the historical V2/V3 family decisions (which still say 341/29).
This produces CURRENT source/camera evidence for the existing family matrix,
not appearance/graphics authority and not any model training authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback

from tools.corpus.audit_corpus370_source_camera_authority_v2 import audit_camera
from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input
from tools.corpus.validate_source_rest_geometry_v1 import validate as independent_source_rest

SCHEMA = "RealSaS.Corpus370V5SourceCameraRequalification.v1"
EVIDENCE_SCHEMA = "RealSaS.Corpus370FamilyEvidenceIndex.v1"
SOURCE_COURT = "RealSaS.Corpus370V5SourceImportIndependentCourt.v1"
CAMERA_COURT = "RealSaS.Corpus370V5CameraObservationIndependentCourt.v1"
V5_SHA = "8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
REST_SHA = "384c92940e1ea8eec7bbb98cf319eeda5f89222c07b782b607998e2067f1d9c8"


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix(path.suffix + ".partial")
    stage.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    stage.replace(path)


def ref(path, schema, status=None):
    path = Path(path).resolve()
    require(path.is_file(), "V5_SOURCE_CAMERA_REF_MISSING:" + str(path))
    value = {"path": str(path), "sha256": sha(path), "schema": schema}
    if status:
        value["status"] = status
    return value


def claim(family, status, evidence, court, *, reasons=(), metadata=None):
    require(family in {"SOURCE_IMPORT", "CAMERA_OBSERVATION"}, "V5_SOURCE_CAMERA_FAMILY")
    require(status in {"PASS", "FAIL"}, "V5_SOURCE_CAMERA_STATUS")
    require(court["status"] == status, "V5_SOURCE_CAMERA_COURT_STATUS")
    require(bool(evidence) and not (status == "FAIL" and not reasons), "V5_SOURCE_CAMERA_CLAIM_EVIDENCE")
    return {
        "status": status, "reason_codes": list(reasons), "evidence_refs": list(evidence),
        "court": dict(court), "producer_refs": [],
        "metadata": {"source_version": "V5_352_PLUS_18", "family": family, **(metadata or {})},
    }


def _disposition(index_row, rest_row, source_row, roster, root, policy):
    sid = source_row["subject_id"]
    require(index_row["subject_id"] == sid and rest_row["status"] == "PASS_SOURCE_REST_INDEPENDENT_COURT",
            "V5_SOURCE_CAMERA_PARENT_SUBJECT:" + sid)
    require(rest_row["input"]["source_sha256"] == source_row["source_sha256"],
            "V5_SOURCE_CAMERA_REST_SOURCE_SHA:" + sid)
    refs = row_input(index_row)
    require(refs["import"]["sha256"] == rest_row["input"]["parent_import_sha256"],
            "V5_SOURCE_CAMERA_ADMITTED_IMPORT_SHA:" + sid)
    require(roster["source_sha256"] == source_row["source_sha256"],
            "V5_SOURCE_CAMERA_ROSTER_SHA:" + sid)
    raw = (root / roster["source_relative_path"]).resolve()
    require(raw.is_relative_to(root) and raw.is_file() and sha(raw) == source_row["source_sha256"],
            "V5_SOURCE_CAMERA_RAW_SOURCE_BYTES:" + sid)
    manifest_ref = rest_row["manifest"]
    manifest = json.loads(Path(manifest_ref["path"]).read_text())
    require(sha(Path(manifest_ref["path"])) == manifest_ref["sha256"],
            "V5_SOURCE_CAMERA_REST_MANIFEST_SHA:" + sid)
    require(manifest["source_sha256"] == source_row["source_sha256"] and manifest["subject_id"] == sid,
            "V5_SOURCE_CAMERA_REST_MANIFEST_ID:" + sid)
    court = independent_source_rest(Path(manifest_ref["path"]))
    require(court["subject_id"] == sid and court["source_sha256"] == source_row["source_sha256"]
            and court["authority"]["independent_source_rest_reconstruction_passed"] is True
            and court["authority"]["training_authorized"] is False,
            "V5_SOURCE_CAMERA_REST_NOT_INDEPENDENTLY_COURTED:" + sid)
    admission = json.loads(Path(refs["admission"]["path"]).read_text())
    require(admission["subject_id"] == sid and admission["source_sha256"] == source_row["source_sha256"]
            and admission.get("status") == "PASS_IRIS_BIND_OBSERVATIONS"
            and admission.get("policy",{}).get("maximum_mask_gap_px") == 2.0,
            "V5_SOURCE_CAMERA_ADMISSION_NOT_STRICT_PASS:" + sid)
    packet = Path(index_row["packet_path"]).resolve()
    camera_result = audit_camera(
        sid=sid, source_sha=source_row["source_sha256"],
        observation_dir=packet, policy=policy,
    )
    return {"refs": refs, "raw_path": raw, "rest_manifest": manifest_ref,
            "rest_court": court, "admission": admission, "camera": camera_result}


def process(v5_path, rest_path, roster_path, root, policy_path, output_dir):
    v5_path=Path(v5_path).resolve(); rest_path=Path(rest_path).resolve()
    roster_path=Path(roster_path).resolve(); root=Path(root).resolve()
    policy_path=Path(policy_path).resolve(); output_dir=Path(output_dir).resolve()
    require(sha(v5_path) == V5_SHA and sha(rest_path) == REST_SHA,
            "V5_SOURCE_CAMERA_PARENT_INDEX_SHA")
    v5=json.loads(v5_path.read_text()); rest=json.loads(rest_path.read_text())
    roster=json.loads(roster_path.read_text()); policy=json.loads(policy_path.read_text())
    require(v5.get("schema")=="RealSaS.Corpus370PreparedObservationPackets.v5"
            and v5.get("subject_count")==370 and v5.get("prepared_subject_count")==352
            and v5.get("failure_count")==18 and v5.get("training_authorized") is False,
            "V5_SOURCE_CAMERA_EXACT_V5_SCOPE")
    require(rest.get("schema")=="RealSaS.Corpus370V5SourceRestIndependentCourtSweep.v1"
            and rest.get("summary",{}).get("independent_source_rest_pass")==352
            and rest.get("summary",{}).get("observation_excluded")==18
            and rest.get("training_authorized") is False,
            "V5_SOURCE_CAMERA_REST_SCOPE")
    require(roster.get("schema")=="RealSaS.Corpus370ExactSourceRoster.v1"
            and roster.get("subject_count")==370 and roster.get("training_authorized") is False,
            "V5_SOURCE_CAMERA_ROSTER")
    require(policy.get("schema")=="RealSaS.Corpus370SourceCameraAuthorityPolicy.v2",
            "V5_SOURCE_CAMERA_POLICY")
    require(root.is_dir(), "V5_SOURCE_CAMERA_RAW_SOURCE_ROOT")
    by_source={x["subject_id"]:x for x in roster["subjects"]}
    qualified={x["subject_id"]:x for x in v5["subjects"]}
    failures={x["subject_id"]:x for x in v5["failures"]}
    require(len(by_source)==370 and len(qualified)==352 and len(failures)==18
            and not set(qualified).intersection(failures)
            and set(by_source)==set(qualified)|set(failures)
            and set(qualified)==set(rest["subjects"]),
            "V5_SOURCE_CAMERA_COHORT_IDENTITY")
    output_dir.mkdir(parents=True,exist_ok=True)
    evidence=[]; results=[]; totals=Counter()
    for i,sid in enumerate(sorted(by_source),1):
        source=by_source[sid]
        source_sha=source["source_sha256"]
        subset=qualified.get(sid)
        row_claims={}
        row_summary={"subject_id":sid,"source_sha256":source_sha}
        output=output_dir/"subjects"/sid
        if subset:
            # Independently re-verify every current parent before granting either PASS.
            try:
                result=_disposition(subset,rest["subjects"][sid],source,source,root,policy)
                source_paths=result["refs"]
                for family,payload,files,code in (
                    ("SOURCE_IMPORT",
                     {"source_sha256":source_sha, "import_sha256":source_paths["import"]["sha256"],
                      "rest_court_schema":result["rest_court"]["schema"],
                      "rest_court_identity":result["rest_court"].get("court_identity_sha256"),
                      "independent_source_rest_passed":True,
                      "current_v5_admission_passed":True},
                     [ref(result["raw_path"],"RealSaS.RawSourceBytes"),
                      ref(Path(source_paths["import"]["path"]),"RealSaS.ImportedBindGeometry.v1"),
                      ref(Path(source_paths["admission"]["path"]),"RealSaS.IrisBindObservationAdmission.v1"),
                      ref(Path(result["rest_manifest"]["path"]),"RealSaS.AdmittedSourceRestGeometry.v1")],
                     SOURCE_COURT),
                    ("CAMERA_OBSERVATION",
                     {"source_sha256":source_sha, "camera_measurements":result["camera"],
                      "current_v5_strict_admission_mask_gap_px":2.0,
                      "source_import_independently_verified":True},
                     [ref(Path(source_paths["camera"]["path"]),"RealSaS.CameraProjectionBundle.v1"),
                      ref(Path(source_paths["candidate"]["path"]),"RealSaS.Controlled8ViewSourceRestCandidate.v2"),
                      ref(Path(source_paths["admission"]["path"]),"RealSaS.IrisBindObservationAdmission.v1")],
                     CAMERA_COURT),
                ):
                    court_path=output/(family+"_COURT_V1.json")
                    write(court_path,{"schema":code, "status":"PASS", "subject_id":sid,
                        "family":family,"evidence":files,"measurements":payload,
                        "authority":{"independent_court_passed":True,
                                     "family_claim_only":True,
                                     "graphics_authority_minted":False,
                                     "model_targets_issued":False,
                                     "training_authorized":False}})
                    row_claims[family]=claim(family,"PASS",files,
                        ref(court_path,code,"PASS"),metadata={"source_sha256":source_sha})
                    totals[family+":PASS"]+=1
                row_summary.update({"SOURCE_IMPORT":"PASS","CAMERA_OBSERVATION":"PASS"})
            except (KeyboardInterrupt,MemoryError):
                raise
            except Exception as exc:
                path=output/"V5_REQUALIFICATION_FAILURE_V1.json"
                write(path,{"schema":"RealSaS.Corpus370V5SourceCameraRequalificationFailure.v1",
                    "subject_id":sid,"reason":str(exc),"traceback":traceback.format_exc(),
                    "training_authorized":False})
                # A new current-court failure cannot masquerade as a MASKED target.
                for family in ("SOURCE_IMPORT","CAMERA_OBSERVATION"):
                    court_path=output/(family+"_COURT_V1.json")
                    write(court_path,{"schema":SOURCE_COURT if family=="SOURCE_IMPORT" else CAMERA_COURT,
                        "status":"FAIL","subject_id":sid,"reason_codes":["INDEPENDENT_V5_REQUALIFICATION_FAILED"],
                        "failure":ref(path,"RealSaS.Corpus370V5SourceCameraRequalificationFailure.v1"),
                        "authority":{"training_authorized":False}})
                    row_claims[family]=claim(family,"FAIL",[ref(path,"RealSaS.Corpus370V5SourceCameraRequalificationFailure.v1")],
                        ref(court_path,SOURCE_COURT if family=="SOURCE_IMPORT" else CAMERA_COURT,"FAIL"),
                        reasons=["INDEPENDENT_V5_REQUALIFICATION_FAILED"],metadata={"reason":str(exc)[:200]})
                    totals[family+":FAIL"]+=1
                row_summary.update({"SOURCE_IMPORT":"FAIL","CAMERA_OBSERVATION":"FAIL","reason":str(exc)})
        else:
            # For 18 V5 observation failures, source import is untested at this gate.
            require(failures[sid]["source_sha256"] == source_sha, "V5_SOURCE_CAMERA_FAIL_ROSTER_SHA:" + sid)
            receipt=(failures[sid].get("receipt") or {})
            require(bool(receipt.get("path")) and bool(receipt.get("sha256"))
                    and sha(Path(receipt["path"])) == receipt["sha256"],
                    "V5_SOURCE_CAMERA_FAILED_RECEIPT_DRIFT:"+sid)
            p=output/"CAMERA_OBSERVATION_COURT_V1.json"
            write(p,{"schema":CAMERA_COURT,"status":"FAIL","subject_id":sid,
                "source_sha256":source_sha,
                "reason_codes":["V5_OBSERVATION_MASK_DRIFT_GREATER_THAN_2PX"],
                "failure_receipt":ref(receipt["path"],"RealSaS.Corpus370V5SourceObservationFailure.v1"),
                "source_import_independently_qualified":False,
                "authority":{"training_authorized":False}})
            row_claims["CAMERA_OBSERVATION"]=claim("CAMERA_OBSERVATION","FAIL",
                [ref(receipt["path"],"RealSaS.Corpus370V5SourceObservationFailure.v1")],
                ref(p,CAMERA_COURT,"FAIL"),
                reasons=["V5_OBSERVATION_MASK_DRIFT_GREATER_THAN_2PX"],
                metadata={"source_sha256":source_sha})
            totals["CAMERA_OBSERVATION:FAIL"]+=1
            totals["SOURCE_IMPORT:UNKNOWN"]+=1
            row_summary.update({"SOURCE_IMPORT":"UNKNOWN","CAMERA_OBSERVATION":"FAIL"})
        evidence.append({"subject_id":sid,"source_sha256":source_sha,"families":row_claims})
        results.append(row_summary)
        if i%10==0 or i==370:
            print(json.dumps({"event":"V5_SOURCE_CAMERA_REQUALIFICATION_PROGRESS",
                "done":i,"total":370,"totals":dict(sorted(totals.items()))},sort_keys=True),flush=True)
    result={"schema":EVIDENCE_SCHEMA,
            "status":"V5_ALL_370_CURRENT_SOURCE_CAMERA_INDEPENDENT_COURTS__MODEL_ADMISSION_SEPARATE",
            "subject_count":370,"subjects":evidence,
            "producer":{"schema":SCHEMA,"source_sha256":sha(Path(__file__))},
            "parents":{"v5_index":ref(v5_path,"RealSaS.Corpus370PreparedObservationPackets.v5"),
                "rest_index":ref(rest_path,"RealSaS.Corpus370V5SourceRestIndependentCourtSweep.v1"),
                "roster":ref(roster_path,"RealSaS.Corpus370ExactSourceRoster.v1")},
            "rollup":dict(sorted(totals.items())),
            "training_authorized":False}
    write(output_dir/"CORPUS370_V5_SOURCE_CAMERA_FAMILY_EVIDENCE_V1.json",result)
    write(output_dir/"CORPUS370_V5_SOURCE_CAMERA_REQUALIFICATION_V1.json",{
        "schema":SCHEMA,"status":"ALL_370_ACCOUNTED__FINAL_FAMILY_MATRIX_SEPARATE",
        "subjects":results,"rollup":dict(sorted(totals.items())),
        "training_authorized":False})
    print(json.dumps({"event":"V5_SOURCE_CAMERA_REQUALIFICATION_SUMMARY",**result["rollup"]},
                     sort_keys=True),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--source-rest-index",type=Path,required=True)
    p.add_argument("--exact-source-roster",type=Path,required=True)
    p.add_argument("--extracted-source-root",type=Path,required=True)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    process(a.v5_index,a.source_rest_index,a.exact_source_roster,
            a.extracted_source_root,a.policy,a.output_dir)

if __name__=="__main__":
    main()
