#!/usr/bin/env python3
"""Independent original Cat full-factorial Möbius court.

Does NOT import source_intervention_mobius_truth_v1. Independently performs
the eight subset inclusion/exclusion identities directly, under immutable
original physical Cycles same-sample source and original independent court SHA.
This validates a finite original-source intervention truth, NOT artist layers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE_NPZ="d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114"
SOURCE_COURT="90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08"
SOURCE_MANIFEST="8a56e47a09f5c1216de2c97e6f13d5e04385f7fe7d2e580d0085b2ab69dc2da1"
ORIGINAL_CAT="1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
SCHEMA="RealSaS.IndependentCatOriginalMobiusInterventionCourt.v1"
MEASURED_TOL=0.025


def require(ok, reason):
    if not ok:
        raise ValueError("INDEPENDENT_MOBIUS_SOURCE_"+reason)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def independently_verify(root:Path, report_path:Path, output_path:Path):
    root=Path(root)
    docs={
        "source":(root/"original-physical-samples.npz",SOURCE_NPZ),
        "court":(root/"original-physical-independent-court.json",SOURCE_COURT),
        "manifest":(root/"original-physical-manifest.json",SOURCE_MANIFEST),
    }
    for key,(p,expected) in docs.items():
        require(sha(p)==expected,"PARENT_BYTES_DRIFT:"+key)
    report=json.loads(Path(report_path).read_text())
    original=json.loads(docs["court"][0].read_text())
    manifest=json.loads(docs["manifest"][0].read_text())
    require(report["schema"]=="RealSaS.SourcePhysicalInterventionMobiusTruth.v1"
            and report["status"] ==
            "ORIGINAL_SOURCE_FINITE_EDIT_INTERACTIONS_IDENTIFIED__ARTIST_OWNER_UNKNOWN"
            and report["original_source_sha256"]==ORIGINAL_CAT
            and manifest["original_source_sha256"]==ORIGINAL_CAT
            and original["source_original_glb_sha256"]==ORIGINAL_CAT
            and original["status"]==
               "PASS_ORIGINAL_CAT_LINKED_UV_EDIT_CONSEQUENCE_SUPERPOSITION__OWNER_UNKNOWN"
            and original["failed_obligations"]==[]
            and report["tolerance_from_original_independent_physical_source_court"]==MEASURED_TOL
            and report["original_native_pose14_edit_interactions"]=="UNKNOWN_NOT_MEASURED_IN_THIS_SOURCE"
            and report["source_physical_ray_to_original_face_edit_effect"]==
                "UNKNOWN_NOT_COURTED"
            and report["artist_drawable_ownership"]=="UNKNOWN"
            and all(report[k] is False for k in (
                "graphics_owner_authorized","iris_training_authorized",
                "tessa_training_authorized","materialization_authorized",
                "training_authorized")),
            "EXPERIMENT_SCOPE_OR_ARTIST_AUTHORITY_WRONG")
    with np.load(docs["source"][0],allow_pickle=False) as f:
        a=np.asarray(f["source_uv_edit_cycles_sample_rgba"],np.float64)
    require(a.shape==(4,8,8,192,192,4) and np.isfinite(a).all(),
            "ORIGINAL_PHYSICAL_ARRAY_SHAPE_DRIFT")
    a[...,:3]*=a[...,3:4]
    # Source archive is [VIEW,SAMPLE,ARM,Y,X,RGBA]. The following formula
    # treats all eight FULL actual physical arms as observations; no source
    # source-ray support inference or packet-generation assumption.
    b=a[:,:,0]
    x=a[:,:,1]
    y=a[:,:,2]
    z=a[:,:,3]
    xy=a[:,:,4]
    xz=a[:,:,5]
    yz=a[:,:,6]
    xyz=a[:,:,7]
    direct=(
        b,
        x-b,
        y-b,
        xy-x-y+b,
        z-b,
        xz-x-z+b,
        yz-y-z+b,
        xyz-xy-xz-yz+x+y+z-b,
    )
    prior=report["coefficients_by_exact_edit_subset"]
    require(len(prior)==8,"INTERACTION_TABLE_INCOMPLETE")
    records=[]
    for mask,coef in enumerate(direct):
        peak=np.max(np.abs(coef),axis=-1)
        count=int(np.count_nonzero(peak>MEASURED_TOL))
        max_error=float(np.max(peak))
        value=prior[mask]
        require(value["bitmask"]==mask and
                value["source_operation_subset_cardinality"]==mask.bit_count() and
                value["source_sample_operation_pixels_above_original_tolerance"]==count and
                abs(value["source_max_abs_linear_pm_rgba_effect"]-max_error)<1e-11
                and value["source_max_abs_alpha_effect"] is None if mask==0 else
                value["source_max_abs_alpha_effect"] is not None,
                "COEFFICIENT_RECOUNT_OR_SOURCE_MASK_DRIFT:"+str(mask))
        records.append({
            "bitmask":mask,"order":mask.bit_count(),
            "independently_measured_count_gt_original_tolerance":count,
            "independently_measured_max_rgba_interaction":max_error,
        })
    # Direct independent inversion: sum all interactions T subset S.
    for target,observed in (
        (0,b),(1,x),(2,y),(3,xy),(4,z),(5,xz),(6,yz),(7,xyz)
    ):
        reproduced=np.zeros_like(observed)
        for sub in range(8):
            if sub & target == sub:
                reproduced+=direct[sub]
        require(float(np.max(np.abs(reproduced-observed)))<1e-12,
                "DIRECT_INCLUSION_EXCLUSION_INVERSE_FAILED")
    require([records[m]["independently_measured_count_gt_original_tolerance"]
             for m in (1,2,4)] ==
            original["source_single_edit_observable_samples_per_operation"] ==
            [290979,264554,6048],
            "INDEPENDENT_PHYSICAL_SINGLE_EFFECT_COUNT_DRIFT")
    require(all(records[m]["independently_measured_count_gt_original_tolerance"]==0
                for m in (3,5,6,7)),
            "HIGHER_ORDER_SOURCE_INTERACTION_NOT_ZERO_UNDER_FROZEN_TOLERANCE")
    require(records[3]["independently_measured_max_rgba_interaction"]>0.0
            and records[5]["independently_measured_max_rgba_interaction"]==0.0
            and records[6]["independently_measured_max_rgba_interaction"]==0.0
            and records[7]["independently_measured_max_rgba_interaction"]==0.0,
            "PREDICTED_SOURCE_INTERACTION_SUPPORT_DIFFERENT")
    result={
        "schema":SCHEMA,
        "status":"EIGHT_ORIGINAL_PHYSICAL_SOURCE_ARMS_UNIQUELY_DECOMPOSED_BY_MOBIUS__NO_OWNER",
        "original_source_sha256":ORIGINAL_CAT,
        "original_physical_source_sample_sha256":SOURCE_NPZ,
        "original_physical_independent_court_sha256":SOURCE_COURT,
        "original_physical_manifest_sha256":SOURCE_MANIFEST,
        "producer_experiment_output_sha256":sha(report_path),
        "source_physical_sample_axes":[4,8,8,192,192,4],
        "original_sample_linear_pm_rgba_tolerance":MEASURED_TOL,
        "independent_exact_subset_coefficients":records,
        "no_high_order_interactions_above_inherited_tolerance":True,
        "nonzero_0_1_interaction_below_inherited_tolerance":True,
        "not_bitwise_zero_for_0_1":True,
        "source_causal_factors_are_original_edit_addresses_not_drawables":True,
        "original_author_drawable_ownership":"UNKNOWN",
        "student_inference_qualified":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }
    out=Path(output_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "status":result["status"],
        "source_0_1_interaction_max":records[3]["independently_measured_max_rgba_interaction"],
        "source_0_2_interaction_max":records[5]["independently_measured_max_rgba_interaction"],
        "source_1_2_interaction_max":records[6]["independently_measured_max_rgba_interaction"],
        "source_0_1_2_interaction_max":records[7]["independently_measured_max_rgba_interaction"],
        "artist_owner":"UNKNOWN","training_authorized":False,
    },sort_keys=True),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--producer",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    independently_verify(a.root,a.producer,a.output)
