#!/usr/bin/env python3
"""Run real Blender source-locus palette perturbations on preselected V5 rays.

The exact source/observation/rest/camera provenance and independent geometric
ray-stack witness are required. Failure is explicit and preserved per subject.
The independent RGBA/reconstruction court reads emitted bytes; no owner labels
or model training authority are issued by this probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input
from tools.corpus.validate_corpus370_v5_palette_intervention_v1 import court
from tools.corpus.generic_source_locus_reciprocal_palette_v2 import court_document as court_v2

SCHEMA="RealSaS.Corpus370V5SourcePaletteInterventionPilot.v1"
V5_SHA="8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
ROSTER_SHA="925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"
RAY_SCHEMA="RealSaS.Corpus370V5GeometricRayPilotIndex.v1"
EXPECTED_SUBJECTS=[
    "corpus_000ca1ae51f9582fde9e",
    "corpus_004e16e05b90505ca23c",
    "corpus_00a3688e42a29045056b",
]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def require(ok,code):
    if not ok:
        raise ValueError(code)


def ref(path,schema):
    path=Path(path).resolve()
    require(path.is_file(),"PALETTE_PILOT_MISSING_SOURCE:"+str(path))
    return {"path":str(path),"sha256":sha(path),"schema":schema}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    stage=path.with_suffix(path.suffix+".partial")
    stage.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    stage.replace(path)


def run(v5_path,roster_path,ray_index_path,root,blender,output):
    v5_path=Path(v5_path).resolve();roster_path=Path(roster_path).resolve()
    ray_index_path=Path(ray_index_path).resolve();root=Path(root).resolve()
    blender=Path(blender).resolve();output=Path(output).resolve()
    require(sha(v5_path)==V5_SHA and sha(roster_path)==ROSTER_SHA,
            "PALETTE_PILOT_ORIGINAL_SOURCE_HASH_DRIFT")
    require(blender.is_file() and root.is_dir() and ray_index_path.is_file(),
            "PALETTE_PILOT_RUNTIME_MISSING")
    v5=json.loads(v5_path.read_text());roster=json.loads(roster_path.read_text())
    rays=json.loads(ray_index_path.read_text())
    require(rays.get("schema")==RAY_SCHEMA and
            rays.get("status")=="PILOT_SUBJECTS_ACCOUNTED__GRAPHICS_AUTHORITY_SEPARATE"
            and set(rays.get("subject_ids") or ())==set(EXPECTED_SUBJECTS)
            and rays.get("training_authorized") is False,
            "PALETTE_PILOT_EXACT_RAY_SOURCE_SELECTION")
    qualified={r["subject_id"]:r for r in v5["subjects"]}
    sources={r["subject_id"]:r for r in roster["subjects"]}
    require(len(qualified)==352 and len(sources)==370
            and set(EXPECTED_SUBJECTS).issubset(qualified),
            "PALETTE_PILOT_COHORT")
    output.mkdir(parents=True,exist_ok=True)
    summary={
        "schema":SCHEMA,
        "status":"PARTIAL_REAL_RENDER_INTERVENTION__NO_FAMILY_AUTHORITY",
        "v5_index_sha256":V5_SHA,
        "exact_roster_sha256":ROSTER_SHA,
        "ray_pilot_index_sha256":sha(ray_index_path),
        "exact_subject_ids":EXPECTED_SUBJECTS,
        "subjects":{},
        "teacher_only":True,
        "graphics_family_authority_minted":False,
        "training_authorized":False,
    }
    index_path=output/"CORPUS370_V5_PALETTE_INTERVENTION_PILOT_V1.json"
    for i,sid in enumerate(EXPECTED_SUBJECTS,1):
        started=time.monotonic()
        folder=output/"subjects"/sid
        folder.mkdir(parents=True,exist_ok=True)
        try:
            source=sources[sid];v5row=qualified[sid]
            require(source["source_sha256"]==v5row["source_sha256"],
                    "PALETTE_PILOT_V5_ROSTER_SHA:"+sid)
            rayrow=rays["subject_results"][sid]
            require(rayrow["source_sha256"]==source["source_sha256"]
                    and rayrow["status"]=="MEASURED_GEOMETRIC_RAY_CANDIDATE",
                    "PALETTE_PILOT_RAY_COURT_NOT_VALID:"+sid)
            ray_path=Path(rayrow["receipt"]["path"]).resolve()
            require(sha(ray_path)==rayrow["receipt"]["sha256"],
                    "PALETTE_PILOT_RAY_RECEIPT_DRIFT:"+sid)
            ray=json.loads(ray_path.read_text())
            require(ray.get("source_sha256")==source["source_sha256"]
                    and ray.get("subject_id")==sid
                    and ray.get("authority",{}).get("graphics_family_minted") is False,
                    "PALETTE_PILOT_RAY_NOT_MEASURED:"+sid)
            input_refs=row_input(v5row)
            source_file=(root/source["source_relative_path"]).resolve()
            require(source_file.is_relative_to(root)
                    and sha(source_file)==source["source_sha256"],
                    "PALETTE_PILOT_RAW_SOURCE_SHA:"+sid)
            geometry_ref=ray["source_rest_geometry"]
            geometry=Path(geometry_ref["path"]).resolve()
            require(sha(geometry)==geometry_ref["sha256"],
                    "PALETTE_PILOT_REST_GEOMETRY_SHA:"+sid)
            job={
                "schema":"RealSaS.Corpus370SourceLocusPaletteJob.v1",
                "subject_id":sid,
                "source":ref(source_file,"RealSaS.RawSourceBytes.v1"),
                "admission":input_refs["admission"],
                "candidate":input_refs["candidate"],
                "camera":input_refs["camera"],
                "geometry":geometry_ref,
                "geometric_rays":rayrow["receipt"],
                "palette_seed":370330,
                "cycles_samples":64,
                "maximum_rays":2,
                "training_authorized":False,
            }
            job_path=folder/"SOURCE_PALETTE_RENDER_JOB_V1.json"
            write(job_path,job)
            manifest_path=folder/"SOURCE_PALETTE_INTERVENTION_V1.json"
            cmd=[
                str(blender),"-b","--factory-startup","--python-exit-code","5",
                "--python","tools/truth_foundry/blender_corpus370_source_locus_palette_v1.py",
                "--","--job",str(job_path),"--output",str(manifest_path),
            ]
            result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,check=False,timeout=900)
            (folder/"BLENDER_LOG.txt").write_text(result.stdout[-280000:])
            require(result.returncode==0,"PALETTE_PILOT_BLENDER_EXIT:"+str(result.returncode))
            require(manifest_path.is_file(),"PALETTE_PILOT_NO_MANIFEST")
            value=json.loads(manifest_path.read_text())
            measured=court(value)
            # V2 independent court binds new reciprocal recolor arms, keeps
            # V1 receipts unchanged and forbids drawable/training authority.
            measured_v2=court_v2(value)
            require(measured_v2["subject_id"]==sid
                    and measured_v2["source_sha256"]==source["source_sha256"]
                    and measured_v2["training_authorized"] is False
                    and measured_v2["graphics_family_authority_minted"] is False,
                    "PALETTE_V2_INDEPENDENT_SOURCE_IDENTITY")
            require(measured["subject_id"]==sid
                    and measured["source_sha256"]==source["source_sha256"],
                    "PALETTE_PILOT_INDEPENDENT_COURT_IDENTITY")
            court_path=folder/"SOURCE_PALETTE_INDEPENDENT_COURT_V1.json"
            court_v2_path=folder/"SOURCE_PALETTE_RECIPROCAL_CAUSAL_COURT_V2.json"
            measured["parent"]={"path":str(manifest_path),"sha256":sha(manifest_path)}
            measured_v2["parent"]={"path":str(manifest_path),"sha256":sha(manifest_path)}
            write(court_path,measured)
            write(court_v2_path,measured_v2)
            summary["subjects"][sid]={
                "status":"PASS_BOUNDED_PALETTE_RENDER_COURT",
                "source_sha256":source["source_sha256"],
                "independent_court":ref(court_path,measured["schema"]),
                "reciprocal_causal_court_v2":ref(court_v2_path,measured_v2["schema"]),
                "v2_two_causal_locus_rows":measured_v2["two_locus_count"],
                "v2_one_causal_locus_rows":measured_v2["one_locus_count"],
                "v2_abstain_rows":measured_v2["abstain_count"],
                "measured_valid_palette_rows":measured["measured_valid_palette_rows"],
                "measured_abstain_rows":measured["abstained_rows"],
                "same_drawable_truth_minted":False,
                "training_authorized":False,
            }
        except (KeyboardInterrupt,MemoryError):raise
        except Exception as exc:
            failure={
                "schema":"RealSaS.Corpus370V5PaletteInterventionFailure.v1",
                "subject_id":sid,"reason":str(exc),
                "error_type":type(exc).__name__,"traceback":traceback.format_exc(),
                "graphics_family_authority_minted":False,
                "training_authorized":False,
            }
            fail_path=folder/"PALETTE_INTERVENTION_FAILURE_V1.json"
            write(fail_path,failure)
            summary["subjects"][sid]={
                "status":"FAIL_SOURCE_PALETTE_RENDER_OR_INDEPENDENT_COURT",
                "source_sha256":sources[sid]["source_sha256"],
                "failure":ref(fail_path,failure["schema"]),
                "training_authorized":False,
            }
        summary["subjects"][sid]["elapsed_seconds"]=round(time.monotonic()-started,3)
        write(index_path,summary)
        print(json.dumps({
            "event":"V5_PALETTE_SOURCE_INTERVENTION",
            "index":i,"total":3,
            "subject_id":sid,
            "status":summary["subjects"][sid]["status"],
            "valid_rows":summary["subjects"][sid].get("measured_valid_palette_rows"),
        },sort_keys=True),flush=True)
    summary["status"]="ALL_THREE_SOURCE_PALETTE_PILOT_SUBJECTS_ACCOUNTED__ADMISSION_SEPARATE"
    summary["failure_count"]=sum(x["status"].startswith("FAIL_") for x in summary["subjects"].values())
    summary["measured_valid_palette_rows"]=sum(x.get("measured_valid_palette_rows",0)
                                                for x in summary["subjects"].values())
    summary["v2_two_causal_locus_rows"]=sum(x.get("v2_two_causal_locus_rows",0)
                                            for x in summary["subjects"].values())
    summary["v2_one_causal_locus_rows"]=sum(x.get("v2_one_causal_locus_rows",0)
                                            for x in summary["subjects"].values())
    summary["v2_abstain_rows"]=sum(x.get("v2_abstain_rows",0)
                                  for x in summary["subjects"].values())
    summary["qualified_same_owner_pairs"]=0
    summary["qualified_different_owner_pairs"]=0
    write(index_path,summary)
    print(json.dumps({"event":"V5_PALETTE_INTERVENTION_SUMMARY",
                      "failures":summary["failure_count"],
                      "measured_valid_palette_rows":summary["measured_valid_palette_rows"],
                      "graphics_family_authority_minted":False,
                      "training_authorized":False},sort_keys=True),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--exact-roster",type=Path,required=True)
    p.add_argument("--geometric-ray-index",type=Path,required=True)
    p.add_argument("--extracted-source-root",type=Path,required=True)
    p.add_argument("--blender",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.v5_index,a.exact_roster,a.geometric_ray_index,
        a.extracted_source_root,a.blender,a.output)


if __name__=="__main__":main()
