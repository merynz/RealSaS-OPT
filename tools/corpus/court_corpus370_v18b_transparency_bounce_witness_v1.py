#!/usr/bin/env python3
"""Independent pixel court for source-sealed V18b transparent bounce witness.

The three original-source camera variants are actual Blender renders, not
synthetic alpha math. The court independently reopens every pixel and tests
whether original Transparent BSDF coverage is due to transparency path length.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA="RealSaS.Corpus370V18bSourceEmptyTransparencyBounceWitness.v1"
RESULT="RealSaS.Corpus370IndependentV18bEmptyTransparencyCourt.v1"
STATUS="ORIGINAL_EMPTY_SOUP_BOUNCE_WITNESS_INDEPENDENTLY_MEASURED__NO_ARTIST_AUTHORITY"
PARENT="RealSaS.Corpus370V18bArchivedRenderParityDiagnosis.v1"
VIEWS=(0,2,4,6)
ORIGINAL_TOL=.025
VARIANTS=("transparent_shader_original_bounce_limit",
          "transparent_shader_high_bounce_limit",
          "physically_hidden_soup")


def require(condition, error):
    if not condition:raise ValueError(error)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def image_alpha_metrics(image):
    array=np.asarray(image,np.float32)
    require(array.shape==(192,192,4) and np.isfinite(array).all()
            and np.min(array[...,3])>=-1e-5
            and np.max(array[...,3])<=1.00001,
            "V18B_BOUNCE_COURT_INVALID_SOURCE_RGBA")
    alpha=array[...,3]
    return {
        "alpha_over_original_threshold_pixel_count":int(np.count_nonzero(alpha>ORIGINAL_TOL)),
        "max_alpha":float(alpha.max()),
        "alpha_p99":float(np.percentile(alpha,99)),
        "alpha_over_original_threshold":bool(np.any(alpha>ORIGINAL_TOL)),
    }


def measure(report_file,raw_file,v14_job_file,archived_file):
    report=json.loads(Path(report_file).read_text())
    job=json.loads(Path(v14_job_file).read_text())
    archived=json.loads(Path(archived_file).read_text())
    require(report.get("schema")==SCHEMA and
            report.get("status")=="REAL_SOURCE_TRANSPARENT_BOUNCE_VS_HIDDEN_EMPTY_RENDER_MEASURED__NO_AUTHORITY"
            and job.get("schema")=="RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"
            and archived.get("schema")==PARENT
            and archived.get("status")=="ARCHIVED_RENDER_PARITY_AND_PROBE_VISIBILITY_DIAGNOSED__NO_AUTHORITY"
            and archived.get("aggregate",{}).get("source_soup_parity_pass_views")==4
            and archived.get("aggregate",{}).get("original_v14_parity_pass_views")==4
            and archived.get("aggregate",{}).get("soup_noop_parity_pass_views")==4
            and archived.get("aggregate",{}).get("empty_soup_alpha_pass_views")<4
            and report.get("original_source_id_private")==job.get("subject_id")
                ==archived.get("source_subject_id_private")
            and report.get("original_source_sha256")==job.get("source",{}).get("sha256")
                ==archived.get("original_source_sha256")
            and report.get("parent_v14_job_sha256")==sha(v14_job_file)
            and report.get("parent_v18b_archived_diagnostic_sha256")==sha(archived_file)
            and report.get("independently_reopenable_render_rgba_npz_sha256")==sha(raw_file)
            and report.get("camera_views")==list(VIEWS)
            and report.get("original_heldout_pose_ordinal")==2
            and report.get("cycles_samples")==32
            and report.get("original_alpha_tolerance")==ORIGINAL_TOL
            and report.get("high_transparent_max_bounces")==256
            and 0<=report.get("original_transparent_max_bounces",-1)<256
            and report.get("new_renders_performed")==12
            and report.get("source_editability_qualified") is False
            and report.get("graphics_owner_authorized") is False
            and report.get("iris_training_authorized") is False
            and report.get("tessa_training_authorized") is False
            and report.get("training_authorized") is False
            and report.get("materialization_authorized") is False
            and job.get("training_authorized") is False,
            "V18B_BOUNCE_COURT_SOURCE_SHA_OR_AUTHORITY_FORGERY")
    with np.load(raw_file,allow_pickle=False) as archive:
        require(archive.files==["empty_variants_rgba"],
                "V18B_BOUNCE_COURT_UNEXPECTED_MEMBERS")
        frames=np.asarray(archive["empty_variants_rgba"],np.float32)
    require(frames.shape==(4,3,192,192,4) and np.isfinite(frames).all(),
            "V18B_BOUNCE_COURT_INVALID_RENDER_ARRAY")
    rows=[]
    for vi,view in enumerate(VIEWS):
        prior=archived["per_view"][vi]
        proposed=report["per_view"][vi]
        require(prior["original_source_camera_view"]==view
                and proposed["original_source_view"]==view,
                "V18B_BOUNCE_COURT_VIEW_BINDING")
        row={"original_source_view":view}
        for idx,key in enumerate(VARIANTS):
            recomputed=image_alpha_metrics(frames[vi,idx])
            claimed=proposed[key]
            require(set(claimed)==set(recomputed)
                    and all((claimed[name] is recomputed[name])
                           if isinstance(recomputed[name],bool)
                           else abs(float(claimed[name])-float(recomputed[name]))<=1e-7
                           for name in recomputed),
                    "V18B_BOUNCE_COURT_FAKE_PRODUCER_RENDER_METRICS")
            row[key]=recomputed
        # Original low-bounce rendering uses the same pose, camera, Cycles
        # sample count, original source geometry and original shader. If the
        # source context itself drifts, fail rather than claim root cause.
        require(row[VARIANTS[0]]["alpha_over_original_threshold_pixel_count"]==
                prior["parity_metrics"]["empty_soup_alpha_above_tolerance_pixel_count"],
                "V18B_BOUNCE_COURT_ORIGINAL_EMPTY_REPLAY_NOT_REPRODUCED")
        row["baseline_empty_alpha_observed"]=row[VARIANTS[0]]["alpha_over_original_threshold"]
        row["bounce_limit_fix_on_this_view"]=(
            row[VARIANTS[0]]["alpha_over_original_threshold"] and
            not row[VARIANTS[1]]["alpha_over_original_threshold"] and
            not row[VARIANTS[2]]["alpha_over_original_threshold"]
        )
        rows.append(row)
    bad=[x for x in rows if x["baseline_empty_alpha_observed"]]
    require(bad,"V18B_BOUNCE_COURT_NO_ORIGINAL_EMPTY_FAILURE")
    hidden_empty=all(not row[VARIANTS[2]]["alpha_over_original_threshold"] for row in rows)
    high_empty=all(not row[VARIANTS[1]]["alpha_over_original_threshold"] for row in rows)
    all_bad_repaired=all(x["bounce_limit_fix_on_this_view"] for x in bad)
    return {
        "schema":RESULT,"status":STATUS,
        "source_subject_id_private":report["original_source_id_private"],
        "original_source_sha256":report["original_source_sha256"],
        "producer_manifest_sha256":sha(report_file),
        "producer_frames_sha256":sha(raw_file),
        "original_parent_diagnostic_sha256":sha(archived_file),
        "original_parent_v14_job_sha256":sha(v14_job_file),
        "source_views":list(VIEWS),
        "per_view":rows,
        "original_empty_failed_views":len(bad),
        "high_transparent_bounces_empty_all_views":high_empty,
        "physically_hidden_empty_all_views":hidden_empty,
        "all_original_failed_views_repaired_by_increased_bounces":all_bad_repaired,
        "transparent_bounce_limit_hypothesis_supported":
            all_bad_repaired and high_empty and hidden_empty,
        "diagnostic_not_original_edit_authority":True,
        "artist_drawable_relation":"UNKNOWN",
        "graphics_owner_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for key in ("report","npz","v14_job","archived_diagnostic","output"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    d=measure(a.report,a.npz,a.v14_job,a.archived_diagnostic)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":d["status"],
        "original_empty_failed_views":d["original_empty_failed_views"],
        "higher_bounce_cleared":d["high_transparent_bounces_empty_all_views"],
        "physically_hidden_cleared":d["physically_hidden_empty_all_views"],
        "bounce_limit_hypothesis_supported":d["transparent_bounce_limit_hypothesis_supported"],
        "per_view_alpha_above_threshold_pixel_counts":[{
            "view":v["original_source_view"],
            "baseline":v[VARIANTS[0]]["alpha_over_original_threshold_pixel_count"],
            "higher_bounces":v[VARIANTS[1]]["alpha_over_original_threshold_pixel_count"],
            "hidden":v[VARIANTS[2]]["alpha_over_original_threshold_pixel_count"],
        } for v in d["per_view"]],
        "training_authorized":False
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
