#!/usr/bin/env python3
"""Measure actual V5 source-triangle/barycentric ray stacks for IRIS graphics truth.

This measures source geometry and exact V5 camera only. It does not infer
alpha-composited visibility, drawables, graphics ownership, relative motion or
material identity. Those need independent rendering and causal interventions.
The measured full stack and candidate geometrical pair direction are useful
inputs to a future RGB/alpha/StdGEN-style reversible palette intervention court.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np

from models.iris.truth.graphics_oracle_v1 import (
    camera_from_mapping, face_centroid_candidate_pixels_v1,
    measure_orthographic_ray_stack_v1,
)
from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input

SCHEMA="RealSaS.Corpus370V5SourceLocusGeometricRayCandidate.v1"
V5_SHA="8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
REST_SHA="384c92940e1ea8eec7bbb98cf319eeda5f89222c07b782b607998e2067f1d9c8"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def require(ok,code):
    if not ok:raise ValueError(code)


def seal(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    stage=path.with_suffix(path.suffix+".partial")
    stage.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    stage.replace(path)


def sample_rays(vertices,faces,bundle,*,max_pixels_per_view=8):
    """Fixed geometry-only proposal set, not chosen by observed graphic labels."""
    require(type(max_pixels_per_view) is int and 1<=max_pixels_per_view<=32,
            "SOURCE_RAY_BOUNDED_SAMPLING")
    vertices=np.asarray(vertices,np.float64)
    faces=np.asarray(faces,np.int64)
    require(vertices.ndim==2 and vertices.shape[1]==3
            and faces.ndim==2 and faces.shape[1]==3 and len(faces)>0
            and np.isfinite(vertices).all()
            and faces.min()>=0 and faces.max()<len(vertices),
            "SOURCE_RAY_EXACT_MESH")
    scale=float(np.linalg.norm(vertices.max(axis=0)-vertices.min(axis=0)))
    require(math.isfinite(scale) and scale>0,"SOURCE_RAY_GEOMETRY_SCALE")
    tie=max(scale*1e-7,1e-9)
    cameras=list(bundle.get("cameras") or [])
    require(bundle.get("schema")=="RealSaS.CameraProjectionBundle.v1"
            and len(cameras)==8 and sorted(int(x["view_index"]) for x in cameras)==list(range(8)),
            "SOURCE_RAY_EIGHT_FIXED_CAMERAS")
    rays=[]; stats=Counter()
    for row in sorted(cameras,key=lambda x:x["view_index"]):
        camera=camera_from_mapping(row)
        proposal=face_centroid_candidate_pixels_v1(
            vertices,faces,camera,maximum=max_pixels_per_view,
        )
        stats["proposed_pixels"]+=len(proposal)
        for px,py in proposal:
            ray=measure_orthographic_ray_stack_v1(
                vertices,faces,camera,px,py,tie_epsilon=tie,
            )
            status=(
                "VALID_GEOMETRIC_FULL_STACK_CANDIDATE"
                if ray.complete and not ray.has_depth_tie and ray.hits
                else "ABSTAIN_INCOMPLETE_OR_TIED_OR_NO_HITS"
            )
            hit_rows=[{
                "face_index_teacher_only":int(hit.face_index),
                "source_barycentric_teacher_only":list(map(float,hit.barycentric)),
                "depth_source_units":float(hit.depth),
            } for hit in ray.hits]
            if status=="VALID_GEOMETRIC_FULL_STACK_CANDIDATE":
                stats["qualified_geometric_rays"]+=1
                stats["qualified_geometric_hits"]+=len(hit_rows)
                stats["qualified_geometric_ordered_hit_pairs"]+=len(hit_rows)*(len(hit_rows)-1)//2
                if len(hit_rows)>=2:
                    stats["multihit_geometric_rays"]+=1
            else:
                stats["abstain_rays"]+=1
            rays.append({
                "view_index":int(row["view_index"]),
                "pixel_x":px,"pixel_y":py,
                "complete":ray.complete,"has_depth_tie":ray.has_depth_tie,
                "status":status,"tie_epsilon_source_units":tie,
                "hits":hit_rows,
                "source_alpha_compositing_qualified":False,
                "drawable_ownership_label_minted":False,
            })
    return rays,dict(sorted(stats.items()))


def run(v5_path,rest_path,output_root,*,subject_ids):
    v5_path=Path(v5_path).resolve();rest_path=Path(rest_path).resolve()
    output_root=Path(output_root).resolve()
    require(sha(v5_path)==V5_SHA and sha(rest_path)==REST_SHA,
            "SOURCE_RAY_PARENT_INDEX_SHA")
    v5=json.loads(v5_path.read_text());rest=json.loads(rest_path.read_text())
    require((v5.get("prepared_subject_count"),v5.get("failure_count"))==(352,18)
            and rest.get("summary",{}).get("independent_source_rest_pass")==352,
            "SOURCE_RAY_352_INPUT")
    rows={r["subject_id"]:r for r in v5["subjects"]}
    require(len(rows)==352 and bool(subject_ids) and len(set(subject_ids))==len(subject_ids)
            and set(subject_ids).issubset(rows),
            "SOURCE_RAY_EXACT_SELECTION")
    result={"schema":"RealSaS.Corpus370V5GeometricRayPilotIndex.v1",
        "status":"PILOT_EXECUTION_INCOMPLETE","subject_ids":sorted(subject_ids),
        "subject_results":{},"training_authorized":False,
        "graphics_owner_truth_minted":False,"appearance_authority_minted":False}
    out=output_root/"SOURCE_LOCUS_GEOMETRIC_RAYS_PILOT_INDEX_V1.json"
    for sid in sorted(subject_ids):
        row=rows[sid]
        try:
            refs=row_input(row)
            restrow=rest["subjects"][sid]
            require(restrow["status"]=="PASS_SOURCE_REST_INDEPENDENT_COURT"
                    and restrow["input"]["source_sha256"]==row["source_sha256"],
                    "SOURCE_RAY_SOURCE_REST_IDENTITY:"+sid)
            manifest_ref=restrow["manifest"]
            manifest_path=Path(manifest_ref["path"]).resolve()
            require(sha(manifest_path)==manifest_ref["sha256"],
                    "SOURCE_RAY_REST_MANIFEST_SHA:"+sid)
            manifest=json.loads(manifest_path.read_text())
            geometry=Path(manifest["output"]["path"])
            require(sha(geometry)==manifest["output"]["sha256"],
                    "SOURCE_RAY_REST_GEOMETRY_SHA:"+sid)
            camera=Path(refs["camera"]["path"])
            require(sha(camera)==refs["camera"]["sha256"],
                    "SOURCE_RAY_CAMERA_SHA:"+sid)
            with np.load(geometry,allow_pickle=False) as arrays:
                verts=np.asarray(arrays["rest_vertices_source_world"]).copy()
                faces=np.asarray(arrays["faces"]).copy()
            bundle=json.loads(camera.read_text())
            rays,counts=sample_rays(verts,faces,bundle,max_pixels_per_view=8)
            output={
                "schema":SCHEMA,
                "status":"SOURCE_GEOMETRIC_RAY_CANDIDATES_MEASURED__RENDERED_ALPHA_AND_OWNER_COURTS_REQUIRED",
                "subject_id":sid,"source_sha256":row["source_sha256"],
                "source_rest_geometry":manifest["output"],
                "v5_camera_bundle":refs["camera"],
                "source_rest_independent_court":manifest_ref,
                "counts":counts,"rays":rays,
                "measurement_semantics":{
                    "source_face_and_barycentric_are_teacher_provenance_only":True,
                    "geometry_only_depth_order_not_composited_visibility":True,
                    "missing_or_tied_ray_is_abstain_not_negative":True,
                    "alpha_and_drawable_source_renderer_courts_required":True,
                    "training_authorized":False,
                },"authority":{"source_geometry_ray_candidate_only":True,
                               "graphics_family_minted":False,
                               "student_owner_labels_minted":False,
                               "training_authorized":False},
            }
            path=output_root/"subjects"/sid/"SOURCE_LOCUS_GEOMETRIC_RAYS_V1.json"
            seal(path,output)
            result["subject_results"][sid]={
                "status":"MEASURED_GEOMETRIC_RAY_CANDIDATE",
                "source_sha256":row["source_sha256"],
                "receipt":{"path":str(path),"sha256":sha(path)},
                "counts":counts,"training_authorized":False,
            }
        except (KeyboardInterrupt,MemoryError):
            raise
        except Exception as exc:
            result["subject_results"][sid]={
                "status":"FAIL_GEOMETRIC_RAY_MEASUREMENT","reason":str(exc),
                "traceback":traceback.format_exc(),"training_authorized":False,
            }
        seal(out,result)
        print(json.dumps({"event":"CORPUS370_GEOMETRIC_RAY_PROGRESS",
            "subject_id":sid,"status":result["subject_results"][sid]["status"],
            "counts":result["subject_results"][sid].get("counts")},sort_keys=True),flush=True)
    result["status"]="PILOT_SUBJECTS_ACCOUNTED__GRAPHICS_AUTHORITY_SEPARATE"
    seal(out,result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--source-rest-index",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--subject-id",action="append",dest="subject_ids",default=[])
    p.add_argument("--all",action="store_true")
    a=p.parse_args()
    require(bool(a.subject_ids) != bool(a.all),"SOURCE_RAY_EXPLICIT_COHORT_CHOICE")
    if a.all:
        require(sha(a.v5_index)==V5_SHA,"SOURCE_RAY_V5_ALL_EXACT_PARENT")
        index=json.loads(a.v5_index.read_text())
        require((index.get("subject_count"),index.get("prepared_subject_count"),
                 index.get("failure_count"))==(370,352,18)
                and index.get("training_authorized") is False,
                "SOURCE_RAY_V5_ALL_INPUT_SCOPE")
        subject_ids=sorted(row["subject_id"] for row in index["subjects"])
        require(len(subject_ids)==len(set(subject_ids))==352,
                "SOURCE_RAY_ALL_NO_MISSING_SUBJECT")
    else:
        subject_ids=a.subject_ids
    result=run(a.v5_index,a.source_rest_index,a.output_dir,subject_ids=subject_ids)
    print(json.dumps({"status":result["status"],"subject_results":{
        k:v["status"] for k,v in result["subject_results"].items()
        },"training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":main()
