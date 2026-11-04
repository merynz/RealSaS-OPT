#!/usr/bin/env python3
"""SHA-exact, private-source-only joint court: native pose x visibility x rest.

For the same three preselected V5 subjects, bind independent rest geometry,
source-authored V4 teacher, source-render/visibility intervention evidence
and exact source->canonical observations. Consumer results are still
OBSERVATIONAL and cannot issue same/different drawable or causal response
supervision. Source face/barycentric IDs and authored teacher frames remain
private; public CI summaries contain counts only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np

from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input
from tools.corpus.build_corpus370_v5_source_native_canonical_animation_v1 import process
from tools.corpus.court_corpus370_visibility_oracle_consumer_v1 import consume
from tools.corpus.diagnose_corpus370_generic_pair_kinematics_v1 import analyze

SCHEMA = "RealSaS.Corpus370HostedNativeVisibilityPairJoin.v1"
V5_SHA = "8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
REST_INDEX_SHA = "384c92940e1ea8eec7bbb98cf319eeda5f89222c07b782b607998e2067f1d9c8"
ROSTER_SHA = "925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"
NATIVE_SUMMARY_SHA = "5aa3c4d9cc97ab669d2a7a951a4c9d18218ef5ea4d6f58f5b1ff1ac5b89cc272"
PALETTE_SUMMARY_SHA = "638579beb83edfa90614dd8db5c4da77406862a5cfaa3e8653d0cc67ad93e10e"
VISIBILITY_SUMMARY_SHA = "fbca92b37ed71975de31b77b91a1fba7871c40b03b4b0ebb5b1695669c43bcbd"
SOURCE_IDS = (
    "corpus_000ca1ae51f9582fde9e",
    "corpus_004e16e05b90505ca23c",
    "corpus_00a3688e42a29045056b",
)
NATIVE_PASS = "PASS_SOURCE_AUTHORED_TEMPORAL_GEOMETRY_CANDIDATE__CAUSAL_RESPONSE_SEPARATE"


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path, digest, code):
    p = Path(path).resolve()
    require(p.is_file() and sha(p) == digest, code)
    return p


def read(path, digest, code):
    return json.loads(checked(path,digest,code).read_text())


def ref(path):
    p = Path(path).resolve()
    return {"path":str(p), "sha256":sha(p)}


def seal(path, value):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+".partial")
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    temp.replace(p)


def run(*,v5_path,roster_path,rest_index_path,native_root,palette_root,
        visibility_root,output_root):
    v5=read(v5_path,V5_SHA,"JOINT_V5_PARENT_SHA")
    roster=read(roster_path,ROSTER_SHA,"JOINT_ROSTER_SHA")
    rests=read(rest_index_path,REST_INDEX_SHA,"JOINT_REST_INDEX_SHA")
    native_root=Path(native_root).resolve()
    palette_root=Path(palette_root).resolve()
    visibility_root=Path(visibility_root).resolve()
    output_root=Path(output_root).resolve()
    native=read(native_root/"CORPUS370_V5_AUTHORED_TEMPORAL_PILOT_V1.json",
                NATIVE_SUMMARY_SHA,"JOINT_NATIVE_SUMMARY_SHA")
    palette=read(palette_root/"CORPUS370_V5_PALETTE_INTERVENTION_PILOT_V1.json",
                 PALETTE_SUMMARY_SHA,"JOINT_PALETTE_SUMMARY_SHA")
    vis_batch=read(visibility_root/"CORPUS370_HOSTED_SOURCE_HIT_VISIBILITY_BATCH_V1.json",
                   VISIBILITY_SUMMARY_SHA,"JOINT_VISIBILITY_BATCH_SHA")
    ids=set(SOURCE_IDS)
    require(v5.get("prepared_subject_count")==352 and v5.get("failure_count")==18
            and rests.get("summary",{}).get("independent_source_rest_pass")==352
            and native.get("schema")=="RealSaS.Corpus370V5AuthoredTemporalTeacherPilot.v1"
            and native.get("status")=="PILOT_ALL_SELECTED_ACCOUNTED__CAUSAL_AND_MODEL_ADMISSION_SEPARATE"
            and set(native.get("selected_subject_ids",[]))==ids
            and set(x["subject_id"] for x in native.get("results",[]))==ids
            and native.get("training_authorized") is False
            and palette.get("schema")=="RealSaS.Corpus370V5SourcePaletteInterventionPilot.v1"
            and palette.get("failure_count")==0
            and set(palette.get("exact_subject_ids",[]))==ids
            and palette.get("training_authorized") is False
            and palette.get("graphics_family_authority_minted") is False
            and vis_batch.get("schema")=="RealSaS.Corpus370BoundedSourceHitVisibilityBatchCourt.v1"
            and vis_batch.get("parent_summary_sha256")==PALETTE_SUMMARY_SHA
            and vis_batch.get("subject_count")==3
            and set(vis_batch.get("subjects",{}))==ids
            and vis_batch.get("qualified_same_drawable_positives")==0
            and vis_batch.get("qualified_different_drawable_negatives")==0
            and vis_batch.get("training_authorized") is False,
            "JOINT_COHORT_OR_AUTHORITY_DRIFT")
    original={x["subject_id"]:x for x in v5["subjects"]}
    roster_rows={x["subject_id"]:x for x in roster["subjects"]}
    native_rows={x["subject_id"]:x for x in native["results"]}
    require(len(original)==352 and len(roster_rows)==370
            and len(native_rows)==3 and ids<=set(original),
            "JOINT_370_352_3_CENSUS")
    report={
        "schema":SCHEMA,
        "status":"PARTIAL_NATIVE_VISIBILITY_PAIR_JOIN__GRAPHICS_OWNER_UNKNOWN",
        "parents":{
            "v5_sha256":V5_SHA,"roster_sha256":ROSTER_SHA,
            "source_rest_index_sha256":REST_INDEX_SHA,
            "native_teacher_summary_sha256":NATIVE_SUMMARY_SHA,
            "palette_summary_sha256":PALETTE_SUMMARY_SHA,
            "visibility_batch_sha256":VISIBILITY_SUMMARY_SHA,
        },
        "expected_subject_count":3,
        "subjects":{},
        "same_drawable_positives":0,
        "different_drawable_negatives":0,
        "independently_courted_causal_controls":0,
        "graphic_owner_identity_minted":False,
        "tessa_teacher_authorized":False,
        "training_authorized":False,
    }
    output_root.mkdir(parents=True,exist_ok=True)
    result_path=output_root/"CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json"
    for sid in SOURCE_IDS:
        folder=output_root/"subjects"/sid
        try:
            require(roster_rows[sid]["source_sha256"]==original[sid]["source_sha256"]
                    and native_rows[sid].get("status")==NATIVE_PASS
                    and native_rows[sid].get("source_sha256")==original[sid]["source_sha256"],
                    "JOINT_NATIVE_SOURCE_OR_MOTION_NOT_POSITIVE")
            row=rests["subjects"][sid]
            require(row["status"]=="PASS_SOURCE_REST_INDEPENDENT_COURT"
                    and row["input"]["source_sha256"]==original[sid]["source_sha256"],
                    "JOINT_SOURCE_REST_NOT_COURTED")
            rest_manifest=read(row["manifest"]["path"],row["manifest"]["sha256"],
                               "JOINT_REST_MANIFEST_SHA")
            rest_npz=checked(rest_manifest["output"]["path"],
                             rest_manifest["output"]["sha256"],"JOINT_REST_ARRAY_SHA")
            bindings=row_input(original[sid])
            native_court_path=native_root/"subjects"/sid/"TEMPORAL_TEACHER_COURT_V1.json"
            native_court=json.loads(native_court_path.read_text())
            require(native_court["status"]==NATIVE_PASS
                    and native_court["source_sha256"]==original[sid]["source_sha256"]
                    and native_court["teacher_arrays"]["sha256"]==native_rows[sid]["teacher_arrays"]["sha256"]
                    and native_court["teacher_manifest"]["sha256"]==native_rows[sid]["teacher_manifest"]["sha256"],
                    "JOINT_NATIVE_TEACHER_COURT_PARENT_DRIFT")
            teacher_npz=checked(native_court["teacher_arrays"]["path"],
                                native_court["teacher_arrays"]["sha256"],"JOINT_TEACHER_ARRAY_SHA")
            receipt=read(native_court["teacher_manifest"]["path"],
                         native_court["teacher_manifest"]["sha256"],"JOINT_TEACHER_RECEIPT_SHA")
            require(receipt["subject_id"]==sid
                    and receipt["source"]["sha256"]==original[sid]["source_sha256"]
                    and receipt["output"]["sha256"]==sha(teacher_npz)
                    and receipt["teacher_only"] is True
                    and receipt["teacher_rig_skin_motion_allowed_as_inference_input"] is False,
                    "JOINT_TEACHER_RECEIPT_NOT_SOURCE_LOCKED")
            color_path=palette_root/"subjects"/sid/"SOURCE_PALETTE_INTERVENTION_V1.json"
            color=json.loads(color_path.read_text())
            require(sha(color_path)==vis_batch["subjects"][sid].get("source_render_sha256",
                     sha(color_path)), "JOINT_PALETTE_PARENT_SHA")
            vis_path=visibility_root/"subjects"/sid/"SOURCE_HIT_VISIBILITY_CANDIDATE_COURT_V1.json"
            vis=read(vis_path,vis_batch["subjects"][sid]["receipt_sha256"],
                     "JOINT_VISIBILITY_RECEIPT_SHA")
            require(vis["source_sha256"]==original[sid]["source_sha256"]
                    and vis["parents"]["source_manifest_sha256"]==sha(color_path)
                    and vis["parents"]["source_pilot_summary_sha256"]==PALETTE_SUMMARY_SHA
                    and len(vis["rows"])==2,
                    "JOINT_VISIBILITY_PARENT_CHAIN")
            oracle=consume(color,vis)
            require(oracle["occluded_edit_sources_recoverable"]==2
                    and oracle["qualified_same_drawable_positive_pairs"]==0
                    and oracle["qualified_different_drawable_negative_pairs"]==0,
                    "JOINT_VISIBILITY_ORACLE_NOT_QUALIFIED")
            canonical=process(
                subject_id=sid,source_sha=original[sid]["source_sha256"],
                rest_manifest=row["manifest"],teacher_court=ref(native_court_path),
                snapshot_ref=bindings["snapshot"],output_dir=folder/"canonical")
            require(canonical["training_authorized"] is False if "training_authorized" in canonical else
                    canonical["authority"]["training_authorized"] is False,
                    "JOINT_CANONICAL_AUTHORITY_INVALID")
            with np.load(rest_npz,allow_pickle=False) as rest, np.load(teacher_npz,allow_pickle=False) as teacher:
                pairs=[]
                for pixel in vis["rows"]:
                    t=pixel["targets"]
                    require(len(t)==2 and
                            t[0]["visibility_label"]=="POSITIVE_VISIBLE_CONTRIBUTOR" and
                            t[1]["visibility_label"]=="NEGATIVE_OPAQUE_OCCLUDED_CONTRIBUTOR" and
                            pixel["rest_geometry_sha256"]==sha(rest_npz) and
                            pixel["camera_sha256"]==bindings["camera"]["sha256"],
                            "JOINT_SOURCE_PIXEL_VISIBILITY_OR_REST_DRIFT")
                    pairs.append({
                        "locus_a":{
                            "face_index_teacher_only":t[0]["source_face_teacher_only"],
                            "barycentric_teacher_only":t[0]["barycentric_teacher_only"],
                        },
                        "locus_b":{
                            "face_index_teacher_only":t[1]["source_face_teacher_only"],
                            "barycentric_teacher_only":t[1]["barycentric_teacher_only"],
                        },
                        "view_index":pixel["view_index"],
                        "pixel_x":pixel["pixel_x"],"pixel_y":pixel["pixel_y"],
                    })
                joint=analyze(
                    rest["rest_vertices_source_world"],rest["faces"],
                    teacher["posed_vertices_source_world"],pairs,
                    source_sha256=original[sid]["source_sha256"],
                    rest_geometry_sha256=sha(rest_npz),
                    authored_teacher_sha256=sha(teacher_npz),
                    independent_visibility_sha256=sha(vis_path),
                    authored_rest_vertices=teacher["rest_vertices_source_world"],
                    authored_faces=teacher["faces"],
                )
            require(joint["training_authorized"] is False
                    and joint["qualified_same_drawable_positives"]==0
                    and joint["qualified_different_drawable_negatives"]==0
                    and joint["pair_count"]==2,
                    "JOINT_KINEMATIC_DIAGNOSTIC_SCOPE")
            joint["parents"]["authored_teacher_receipt_sha256"]=sha(
                Path(native_court["teacher_manifest"]["path"]))
            joint["parents"]["canonical_candidate_sha256"]=sha(
                Path(canonical["canonical_arrays"]["path"]))
            joint_path=folder/"SOURCE_HIT_NATIVE_KINEMATIC_PAIR_V1.json"
            seal(joint_path,joint)
            report["subjects"][sid]={
                "status":"PASS_PRIVATE_SOURCE_REST_AUTHORED_AND_VISIBILITY_JOIN__OWNER_UNKNOWN",
                "source_sha256":original[sid]["source_sha256"],
                "native_authored_frame_count":joint["authored_pose_count"],
                "observed_relative_deformation_pairs":joint["observed_relative_deformation_count"],
                "kinematic_pair_count":joint["pair_count"],
                "degenerate_abstain_count":joint["degenerate_abstain_count"],
                "canonical_status":canonical["status"],
                "canonical_candidate_receipt":ref(folder/"canonical"/"CANONICAL_NATIVE_ANIMATION_CANDIDATE_V1.json"),
                "pair_diagnostic_receipt":ref(joint_path),
                "graphics_owner_minted":False,
                "training_authorized":False,
            }
        except (KeyboardInterrupt,MemoryError):
            raise
        except Exception as exc:
            report["subjects"][sid]={
                "status":"FAIL_PRIVATE_NATIVE_VISIBILITY_OR_CANONICAL_JOIN",
                "error_type":type(exc).__name__,"reason":str(exc),
                "training_authorized":False,
            }
            seal(folder/"PAIR_JOIN_FAILURE_V1.json",{
                "subject_id":sid,
                "status":"FAIL_PRIVATE_NATIVE_VISIBILITY_OR_CANONICAL_JOIN",
                "exception_type":type(exc).__name__,"reason":str(exc),
                "traceback":traceback.format_exc(),
                "training_authorized":False,
            })
        finally:
            seal(result_path,report)
    counts=Counter(row["status"] for row in report["subjects"].values())
    report["status"]="ALL_THREE_NATIVE_VISIBILITY_PAIRS_ACCOUNTED__OWNER_AUTHORITY_SEPARATE"
    report["qualified_subject_count"]=counts["PASS_PRIVATE_SOURCE_REST_AUTHORED_AND_VISIBILITY_JOIN__OWNER_UNKNOWN"]
    report["failure_count"]=counts["FAIL_PRIVATE_NATIVE_VISIBILITY_OR_CANONICAL_JOIN"]
    report["measured_kinematic_pair_count"]=sum(r.get("kinematic_pair_count",0) for r in report["subjects"].values())
    report["observed_relative_deformation_pair_count"]=sum(r.get("observed_relative_deformation_pairs",0) for r in report["subjects"].values())
    report["degenerate_abstain_count"]=sum(r.get("degenerate_abstain_count",0) for r in report["subjects"].values())
    seal(result_path,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("subjects","parents")},
                     sort_keys=True),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--exact-roster",type=Path,required=True)
    p.add_argument("--source-rest-index",type=Path,required=True)
    p.add_argument("--native-root",type=Path,required=True)
    p.add_argument("--palette-root",type=Path,required=True)
    p.add_argument("--visibility-root",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    a=p.parse_args()
    result=run(v5_path=a.v5_index,roster_path=a.exact_roster,
               rest_index_path=a.source_rest_index,native_root=a.native_root,
               palette_root=a.palette_root,visibility_root=a.visibility_root,
               output_root=a.output_root)
    require(result["failure_count"]==0 and result["qualified_subject_count"]==3
            and result["measured_kinematic_pair_count"]==6,
            "JOINT_REAL_MEASURED_SUBJECT_FAILURES_REQUIRE_REPAIR")


if __name__ == "__main__":
    main()
