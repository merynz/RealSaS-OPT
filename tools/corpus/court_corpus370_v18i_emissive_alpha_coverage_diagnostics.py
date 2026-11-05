#!/usr/bin/env python3
"""Read-only V18i: distinguish alpha coverage from draw-order RGB failures.

Important: sample-filtered isolated source renders are not generally
compositor-ready. For two alpha-separated images alpha(OVER) is order
invariant: aG + aR - aG*aR. A full-source alpha disagreement with this
identity cannot be repaired by reversing the draw order. This script
reopens SHA-locked original V18h source RGBA and independent V18h court;
it creates no artist-owner truth and authorizes no training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.court_corpus370_v18h_emissive_compositing_witness import (
    RENDER, SCHEMA as PARENT_COURT, STATUS as PARENT_PASS,
    THRESHOLDS, VIEWS, RGB_MEMBER, arms,
)
from tools.corpus.court_corpus370_v18g_operational_compositing_diagnostics import (
    TOL, compositor_probe
)

SCHEMA="RealSaS.Corpus370IndependentEmissiveAlphaCoverageDiagnosis.v18i"
STATUS="SOURCE_EMISSIVE_ALPHA_COVERAGE_VS_ORDER_DIAGNOSED__NO_DRAWABLE_TRUTH"


def require(ok,error):
    if not ok: raise ValueError(error)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def alpha_coverage_probe(full,group,remainder):
    """Distinguish unavoidable alpha mismatch from *RGB/order* failure.

    Inputs are immutable real original Blender samples. The original
    renderer's order-independent OVER alpha is necessary for BOTH layer
    orderings. An alpha mismatch falsifies this 2-layer alpha representation
    regardless of the source of the mismatch; it does NOT prove an artist
    component boundary or a specific antialiasing mechanism.
    """
    for im in (full,group,remainder):
        require(isinstance(im,np.ndarray)
            and im.shape==(192,192,4)
            and np.isfinite(im).all()
            and np.min(im[...,3])>=-1e-5
            and np.max(im[...,3])<=1.00001,
            "V18I_INVALID_SEALED_RGBA")
    a=np.clip(full[...,3],0,1)
    g=np.clip(group[...,3],0,1)
    r=np.clip(remainder[...,3],0,1)
    over_a=g+r-g*r
    delta=np.abs(a-over_a)
    # These inclusion-exclusion bounds are heuristic tests of independent
    # opaque coverage under a fixed sample kernel (not arbitrary transparent
    # or scattering materials). Do not admit them as hard truth.
    outside_union_bounds=(a<np.maximum(g,r)-TOL)|(a>np.minimum(1.,g+r)+TOL)
    two_orders=compositor_probe(full,group,remainder)
    # The per-pixel oracle is a diagnostic (not a render plan).
    failed_min=np.minimum(
        np.max(np.abs(np.concatenate(
            (full[...,:3]*a[...,None],a[...,None]),axis=-1)
            -np.concatenate((group[...,:3]*g[...,None]+
                remainder[...,:3]*r[...,None]*(1-g[...,None]),
                over_a[...,None]),axis=-1)),axis=-1),
        np.max(np.abs(np.concatenate(
            (full[...,:3]*a[...,None],a[...,None]),axis=-1)
            -np.concatenate((remainder[...,:3]*r[...,None]+
                group[...,:3]*g[...,None]*(1-r[...,None]),
                over_a[...,None]),axis=-1)),axis=-1))
    oracle_failed=failed_min>TOL
    assert int(oracle_failed.sum()) == two_orders["per_pixel_order_oracle"]["pixel_count_above_original_tolerance"]
    unavoidable_alpha=(delta>TOL)
    rgb_only=oracle_failed & ~unavoidable_alpha
    require(np.all(~unavoidable_alpha | oracle_failed),
            "V18I_MATHEMATICAL_ALPHA_CANNOT_PASS_RGB_ORACLE")
    support=(g>TOL)&(r>TOL)
    return {
        "source_full_visible_alpha_pixels":int(np.count_nonzero(a>TOL)),
        "group_visible_alpha_pixels":int(np.count_nonzero(g>TOL)),
        "remainder_visible_alpha_pixels":int(np.count_nonzero(r>TOL)),
        "both_complements_alpha_visible_pixels":int(np.count_nonzero(support)),
        "order_invariant_OVER_alpha_mismatch_pixels":int(np.count_nonzero(unavoidable_alpha)),
        "alpha_mismatch_p99":float(np.percentile(delta,99)),
        "alpha_mismatch_max":float(np.max(delta)),
        "inclusion_exclusion_union_bounds_warning_pixels":int(np.count_nonzero(outside_union_bounds)),
        "per_pixel_two_order_oracle_failed_pixels":int(np.count_nonzero(oracle_failed)),
        "per_pixel_oracle_fail_with_ALPHA_mismatch_pixels":int(np.count_nonzero(oracle_failed & unavoidable_alpha)),
        "per_pixel_oracle_fail_without_alpha_mismatch_pixels":int(np.count_nonzero(rgb_only)),
        "order_discriminating_overlap_pixels":two_orders["order_discriminating_overlap_pixels"],
        "alpha_disagreement_cannot_be_repaired_by_swapping_TWO_layer_order":True,
        "requires_pixelwise_source_subsample_proof_for_aa_causality":True,
    }


def measure(manifest_path,rgba_path,parent_court_path,expected_court_sha):
    require(isinstance(expected_court_sha,str) and
            len(expected_court_sha)==64 and
            sha(parent_court_path)==expected_court_sha,
            "V18I_EXACT_SOURCE_COURT_SHA_REQUIRED")
    manifest=load(manifest_path)
    court=load(parent_court_path)
    require(manifest.get("schema")==RENDER
        and manifest.get("status")==
          "V18H_ORIGINAL_EMISSIVE_COMPLEMENT_RENDERED__INDEPENDENT_COURT_REQUIRED"
        and court.get("schema")==PARENT_COURT
        and court.get("status")==PARENT_PASS
        and court.get("source_subject_id_private")==manifest.get("subject_id")
        and court.get("original_source_sha256")==manifest.get("original_source_sha256")
        and court.get("exact_source_v18h_render_manifest_sha256")==sha(manifest_path)
        and court.get("exact_source_v18h_render_rgba_npz_sha256")==sha(rgba_path)
        and manifest.get("new_source_actual_edited_rgba_npz_sha256")==sha(rgba_path)
        and manifest.get("source_edit_arms")==arms()
        and manifest.get("physical_face_visibility_operator")==
            "ACTUAL_TRIANGLE_REMOVAL_AND_OBJECT_RENDER_HIDE"
        and manifest.get("source_material_policy")==
            "FIXED_TWO_COLOR_OPAQUE_EMISSION_NO_MATERIAL_SHADING"
        and manifest.get("source_material_label_authority")=="NONE"
        and court.get("fixed_two_color_emission_only_diagnostic") is True
        and court.get("no_artist_semantic_or_original_layer_labels") is True
        and court.get("source_original_heldout_camera_views")==list(VIEWS)
        and court.get("original_v18f_run")=="42114401233-1"
        and court.get("original_v18g_run")=="42116712393-1"
        and set(court.get("per_threshold",{}))=={
          str(t) for t in THRESHOLDS}
        and len(court.get("per_camera_threshold",[]))==12
        and all(court.get(f) is False for f in (
          "graphics_owner_authorized","iris_training_authorized",
          "tessa_training_authorized","materialization_authorized",
          "training_authorized"))
        and all(manifest.get(f) is False for f in (
          "graphics_owner_authorized","iris_training_authorized",
          "tessa_training_authorized","materialization_authorized",
          "training_authorized")),
        "V18I_ARCHIVED_PARENT_SOURCE_AND_NONAUTHORITY_INVALID")
    with np.load(rgba_path,allow_pickle=False) as r:
        require(r.files==[RGB_MEMBER],"V18I_WRONG_RGBA_MEMBER")
        rgba=np.asarray(r[RGB_MEMBER],np.float32)
    require(rgba.shape==(len(VIEWS),len(arms()),192,192,4)
            and np.isfinite(rgba).all()
            and np.min(rgba[...,3])>=-1e-5
            and np.max(rgba[...,3])<=1.00001,
            "V18I_INCONSISTENT_RGBA_SOURCE_ARCHIVE")
    prior={(int(row["camera_view"]),float(row["threshold"])):row
           for row in court["per_camera_threshold"]}
    require(len(prior)==12,"V18I_PARENT_SOURCE_CAMERA_THRESHOLD_DUPLICATE")
    rows=[]
    sums={}
    for ti,t in enumerate(THRESHOLDS):
        entries=[]
        for vi,view in enumerate(VIEWS):
            full=rgba[vi,1+3*ti]
            group=rgba[vi,2+3*ti]
            remain=rgba[vi,3+3*ti]
            result=alpha_coverage_probe(full,group,remain)
            prev=prior[(int(view),float(t))]
            assert result["per_pixel_two_order_oracle_failed_pixels"]==prev["emissive"]["per_pixel_order_oracle"]["pixel_count_above_original_tolerance"]
            assert result["order_discriminating_overlap_pixels"]==prev["emissive"]["order_discriminating_overlap_pixels"]
            row={"view":int(view),"threshold":float(t),
                 "original_source_group_face_count":int(prev["source_group_faces"]),
                 "observations":result}
            rows.append(row)
            entries.append(row)
        sums[str(t)]={
            "original_source_group_count":court["per_threshold"][str(t)]["source_group_count"],
            "group_face_count":court["per_threshold"][str(t)]["examined_group_source_face_count"],
            "unique_view_cases":len(entries),
            "order_invariant_alpha_failure_pixel_instances":int(sum(
                z["observations"]["order_invariant_OVER_alpha_mismatch_pixels"] for z in entries)),
            "two_order_oracle_failure_pixel_instances":int(sum(
                z["observations"]["per_pixel_two_order_oracle_failed_pixels"] for z in entries)),
            "alpha_mismatch_explains_oracle_failure_pixel_instances":int(sum(
                z["observations"]["per_pixel_oracle_fail_with_ALPHA_mismatch_pixels"] for z in entries)),
            "RGB_or_order_failure_without_alpha_mismatch_pixel_instances":int(sum(
                z["observations"]["per_pixel_oracle_fail_without_alpha_mismatch_pixels"] for z in entries)),
            "independently_visible_both_complements_pixel_instances":int(sum(
                z["observations"]["both_complements_alpha_visible_pixels"] for z in entries)),
            "nontrivial_group_for_rendering":bool(
                court["per_threshold"][str(t)]["nontrivial_group_under_exact_source_geometry"]),
            "no_source_art_layer_truth_minted":True,
        }
        require(sums[str(t)]["two_order_oracle_failure_pixel_instances"]==
                court["per_threshold"][str(t)]["emissive_oracle_failed_pixel_instances"],
                "V18I_INDEPENDENT_PARENT_FAILED_PIXELS_NOT_REPRODUCED")
    return {
      "schema":SCHEMA,"status":STATUS,
      "source_subject_private":court["source_subject_id_private"],
      "original_source_sha256":court["original_source_sha256"],
      "original_v18h_run":"42120479356-1",
      "parent_v18h_court_sha256":sha(parent_court_path),
      "parent_v18h_render_sha256":sha(rgba_path),
      "parent_v18h_manifest_sha256":sha(manifest_path),
      "per_view_threshold":rows,
      "per_threshold":sums,
      "new_blender_renders":0,
      "anti_aliasing_mechanism_proved":False,
      "draw_order_truth_authorized":False,
      "source_artist_editable_layer_labels":"UNKNOWN",
      "source_operational_drawable_layers":"UNKNOWN",
      "graphics_owner_authorized":False,
      "iris_training_authorized":False,
      "tessa_training_authorized":False,
      "materialization_authorized":False,
      "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for n in ("render_manifest","render_npz","v18h_court","output"):
        p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
    p.add_argument("--expected-v18h-court-sha",required=True)
    a=p.parse_args()
    out=measure(a.render_manifest,a.render_npz,a.v18h_court,a.expected_v18h_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":out["status"],"per_threshold":out["per_threshold"],
                      "training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
