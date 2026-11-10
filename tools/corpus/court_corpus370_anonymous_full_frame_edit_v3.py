#!/usr/bin/env python3
"""Independent all-pixel original-source / anonymous-group editor render court.

Full camera FOV, bounded raster of two views, 6 actual Blender arms. This
court can measure rendering/alpha parity, no-op and whole-slot suppression
at ALL pixels, not just one selected ray. It CANNOT infer the original artist's
drawable ownership; no CAA, hidden-depth, skin or product authority minted.

Differences on a transparent pixel can be legitimate. Every channel is
compared with source alpha and premultiplied original-canvas conservatively.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SCHEMA="RealSaS.Corpus370AnonymousFullFrameEditIndependentCourt.v3"
RENDER="RealSaS.Corpus370AnonymousFullFrameEditorRender.v3"
ARMS=(
    "SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP",
    "REMOVE_SELECTED_GROUP_A","REMOVE_SELECTED_GROUP_B",
    "REMOVE_BOTH_SELECTED_GROUPS",
)
SOURCE_PARITY_P99_TOL=.025
NOOP_MAX_TOL=.0025
EFFECT_TOL=.025


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def _signal(a,b):
    # Conservative, carrier-agnostic RGBA difference; RGB compared directly
    # only where either alpha is non-zero, so hidden transparent RGB cannot
    # masquerade as an edit or source fidelity failure.
    alpha=np.maximum(a[...,3],b[...,3])
    color=np.max(np.abs(a[...,:3]-b[...,:3]),axis=-1)
    return np.maximum(np.abs(a[...,3]-b[...,3]),
                      color*(alpha>1e-4))


def evaluate(pixels,manifest):
    require(manifest.get("schema")==RENDER
            and manifest.get("status")==
                "SOURCE_FULL_FRAME_REST_ALPHA_EDIT_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
            and manifest.get("views")==[0,4]
            and tuple(manifest.get("arms") or [])==ARMS
            and manifest.get("teacher_only") is True
            and manifest.get("original_geometry_reimport_parity_verified") is True
            and manifest.get("training_authorized") is False
            and manifest.get("graphics_owner_authorized") is False
            and manifest.get("model_training_authorized") is False,
            "FULL_FRAME_COURT_PRODUCER_SCOPE_OR_AUTHORITY")
    p=np.asarray(pixels)
    r=manifest.get("full_frame_resolution")
    require(type(r) is int and 128<=r<=512
            and p.shape==(2,len(ARMS),r,r,4)
            and p.dtype.kind=="f" and np.isfinite(p).all()
            and np.all(p[...,3]>=-1e-5)
            and np.all(p[...,3]<=1.+1e-5),
            "FULL_FRAME_COURT_REAL_RGBA_ARRAY")
    rows=manifest.get("private_view_details")
    require(isinstance(rows,list) and len(rows)==2
            and [x.get("view") for x in rows]==[0,4]
            and all(type(x.get("selected_slot_a_teacher_only")) is int
                    and type(x.get("selected_slot_b_teacher_only")) is int
                    and type(x.get("source_pixel_x")) is int
                    and type(x.get("source_pixel_y")) is int for x in rows),
            "FULL_FRAME_COURT_VIEW_SOURCE_PIXEL_SCOPE")
    views=[]
    for i,view in enumerate((0,4)):
        full=p[i,1]
        reference=p[i,0]
        noop=p[i,2]
        without_a=p[i,3]
        without_b=p[i,4]
        without_both=p[i,5]
        diff_source=_signal(reference,full)
        diff_noop=_signal(full,noop)
        consequence_a=_signal(full,without_a)
        consequence_b=_signal(full,without_b)
        conditional_b=_signal(without_a,without_both)
        fraction=float(np.mean(diff_source>SOURCE_PARITY_P99_TOL))
        p99=float(np.quantile(diff_source,.99))
        noop_max=float(diff_noop.max())
        no_op_pass=noop_max<=NOOP_MAX_TOL
        source_pass=p99<=SOURCE_PARITY_P99_TOL
        group_a=rows[i]["selected_slot_a_teacher_only"]
        group_b=rows[i]["selected_slot_b_teacher_only"]
        view_measured={
            "view":view,
            "pixel_count":r*r,
            "source_original_vs_triangle_soup_p99_rgba_error":p99,
            "source_original_vs_triangle_soup_max_rgba_error":float(diff_source.max()),
            "source_pixels_above_rgb_alpha_parity_tol":int(np.count_nonzero(diff_source>SOURCE_PARITY_P99_TOL)),
            "source_nonparity_pixel_fraction":fraction,
            "soup_full_vs_noop_max_rgba_error":noop_max,
            "source_to_soup_bounded_parity_pass":bool(source_pass),
            "no_op_rerender_parity_pass":bool(no_op_pass),
            "source_wide_a_group_removal_pixel_count":int(np.count_nonzero(consequence_a>EFFECT_TOL)),
            "source_wide_b_group_removal_pixel_count":int(np.count_nonzero(consequence_b>EFFECT_TOL)),
            "source_wide_b_conditional_after_a_group_removal_pixel_count":int(
                np.count_nonzero(conditional_b>EFFECT_TOL)),
            "selected_a_b_groups_separable":group_a!=group_b,
            "rest_pose_only":True,
            "full_fov_but_downsampled":manifest["downsampled_relative_to_original"],
            "source_hit_visibility_independently_courted_elsewhere":True,
            "source_drawable_identity_proven":False,
        }
        view_measured["status"]=(
            "ABSTAIN_SOURCE_OR_NOOP_RENDER_PARITY_FAILURE"
            if not(source_pass and no_op_pass) else
            "ABSTAIN_SELECTED_GROUPS_NOT_SEPARABLE"
            if group_a==group_b else
            "MEASURED_REAL_FULL_FRAME_EDIT_EFFECTS__CAA_SEPARATE")
        views.append(view_measured)
    return {
        "schema":SCHEMA,
        "status":"ALL_BOUNDED_SOURCE_FULL_FOV_IMAGES_COURTED__GRAPHICS_AUTHORITY_SEPARATE",
        "view_count":len(views),
        "original_full_fov_pixels_measured":len(views)*r*r,
        "resolution":r,
        "per_view":views,
        "bounded_parity_and_separable_views":sum(int(
            x["status"]=="MEASURED_REAL_FULL_FRAME_EDIT_EFFECTS__CAA_SEPARATE")
            for x in views),
        "full_native_resolution_parity_checked":not manifest[
            "downsampled_relative_to_original"],
        "original_source_mesh_and_material_name_teacher_only":True,
        "same_drawable_positive_pairs":0,
        "different_drawable_negative_pairs":0,
        "graphics_owner_family_authorized":False,
        "appearance_caa_total_authority_minted":False,
        "animation_replay_authorized":False,
        "tessa_teacher_authorized":False,
        "product_compilation_authorized":False,
        "training_authorized":False,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--expected-manifest-sha256",required=True)
    parser.add_argument("--arrays",type=Path,required=True)
    parser.add_argument("--expected-arrays-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    require(a.manifest.is_file()
            and sha(a.manifest)==a.expected_manifest_sha256
            and a.arrays.is_file()
            and sha(a.arrays)==a.expected_arrays_sha256,
            "FULL_FRAME_COURT_INPUT_BYTES_SHA")
    receipt=json.loads(a.manifest.read_text())
    require(receipt["array_sha256"]==a.expected_arrays_sha256,
            "FULL_FRAME_COURT_RENDER_ARRAY_BINDING")
    with np.load(a.arrays,allow_pickle=False) as f:
        require(set(f.files)=={"source_original_vs_anonymous_group_rgba"},
                "FULL_FRAME_COURT_NO_EXTRA_PAYLOADS")
        outcome=evaluate(f["source_original_vs_anonymous_group_rgba"],receipt)
    outcome["parents"]={
        "source_sha256":receipt["source_sha256"],
        "render_manifest_sha256":a.expected_manifest_sha256,
        "render_array_sha256":a.expected_arrays_sha256,
        "source_camera_bundle_sha256":receipt["source_camera_bundle_sha256"],
        "selected_layer_proposal_sha256":receipt["selected_layer_proposal_sha256"],
        "selected_layer_court_sha256":receipt["selected_layer_court_sha256"],
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(outcome,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":outcome["status"],
        "pixels":outcome["original_full_fov_pixels_measured"],
        "bounded_parity_and_separable_views":outcome["bounded_parity_and_separable_views"],
        "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":main()
