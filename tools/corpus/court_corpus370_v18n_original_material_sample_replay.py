#!/usr/bin/env python3
"""V18n source-material compositor: measure, don't assume original appearance.

Only G/R isolates and independently acquired V18m original-source G/R depth
are predictor inputs. Native original and source-soup FULL are held out for
full-sample independent scoring. Material policy is the admitted source
appearance replay, not raw untransformed artist authored Cycles truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.court_corpus370_v18l_cycles_common_sample_source import (
    premul, over,
)
from tools.corpus.court_corpus370_v18m_depth_ordered_source_compositor import (
    DEPTH_TIE_EPS, TOL, SCHEMA as V18M_COURT, STATUS as V18M_STATUS,
    V18M_MANIFEST, V18M_STATUS as V18M_RENDER_STATUS,
)

SCHEMA="RealSaS.Corpus370IndependentSourceMaterialDepthCompositingCourt.v18n"
STATUS="SOURCE_MATERIAL_POLICY_RGBA_AND_DEPTH_ORDER_EVALUATED__ARTIST_OWNER_UNKNOWN"
MANIFEST="RealSaS.Corpus370OriginalArtistPolicyMaterialSampleReplay.v18n"
RENDER_STATUS="V18N_ORIGINAL_MATERIAL_CYCLES_SOURCE_REPLAYED__INDEPENDENT_COURT_REQUIRED"
CAMERAS=(0,2,4,6)
INDICES=(0,4,8,12,16,20,24,28)
ARMS=("SOURCE_NATIVE","SOUP_FULL","SOUP_ISOLATE","SOUP_REMOVE")
NONPROMOTION=("graphics_owner_authorized","iris_training_authorized",
              "tessa_training_authorized","materialization_authorized",
              "training_authorized")


def require(cond,reason):
    if not cond: raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def predict_material_from_source_isolates(colors,depth,epsilon=DEPTH_TIE_EPS):
    """Does not accept native or FULL image; colors = [8,2,192,192,4].

    Color alpha may vary under original source material, and depth alpha
    represents original *geometry* coverage. No color-alpha=depth-alpha
    equality is inferred. Material's stochastic opacity is not coerced to
    opaque. All subpixel scores explicitly allow UNKNOWN from depth ties.
    """
    colors=np.asarray(colors,dtype=np.float64)
    depth=np.asarray(depth,dtype=np.float64)
    require(colors.shape==(8,2,192,192,4)
        and depth.shape==colors.shape
        and np.isfinite(colors).all() and np.isfinite(depth).all()
        and 0<epsilon<=.001
        and np.min(colors[...,3])>=-1e-5
        and np.max(colors[...,3])<=1.00001,
        "V18N_SOURCE_MATERIAL_DEPTH_RGBA_SHAPE")
    cp=premul(colors)
    color_a=np.clip(colors[...,3],0.,1.)
    depth_a=np.clip(depth[...,3],0.,1.)
    require(np.all((depth_a<=1e-6)|(depth_a>=1-1e-6)),
        "V18N_SOURCE_DEPTH_1SPP_GEOMETRY_COVERAGE_NONBINARY")
    geo=depth_a>.5
    # Separate original source material may be invisible at a known 3D ray.
    # But it must not be visible where the *same ray* did not hit the source
    # geometry; otherwise source-material vs depth sample alignment failed.
    apparent_without_geo=(color_a>1e-6)&~geo
    require(not np.any(apparent_without_geo),
        "V18N_SOURCE_MATERIAL_VISIBLE_WHERE_GEOMETRY_NOT_HIT")
    g,r=geo[:,0],geo[:,1]
    zg,zr=depth[:,0,:,:,0],depth[:,1,:,:,0]
    valid_z=np.where(g,(zg>=.019)&(zg<=.981),True)&np.where(
        r,(zr>=.019)&(zr<=.981),True)
    tied=(g&r)&(np.abs(zg-zr)<=epsilon)
    unknown=(~valid_z)|tied
    gfirst=(~unknown)&g&(~r|(zg+epsilon<zr))
    rfirst=(~unknown)&r&(~g|(zr+epsilon<zg))
    require(np.all(~(gfirst&rfirst))
        and np.all(np.where(~unknown,gfirst|rfirst|(~g&~r),True)),
        "V18N_SOURCE_LOCAL_DEPTH_COVERAGE_NO_FRONTMOST")
    gcol=cp[:,0]
    rcol=cp[:,1]
    gr=over(gcol,rcol)
    rg=over(rcol,gcol)
    predicted=np.where(gfirst[...,None],gr,
         np.where(rfirst[...,None],rg,0.))
    valid=~unknown
    return {
        "predicted_sample_premultiplied_rgba":predicted,
        "source_valid_sample_mask":valid,
        "source_geometry_depth_tie_mask":tied,
        "source_invalid_depth_mask":~valid_z,
        "source_both_geometry_hit_mask":g&r,
        "source_gfront_sample_mask":gfirst,
        "source_rfront_sample_mask":rfirst,
        "source_material_nonbinary_alpha_mask":
            (color_a>1e-6)&(color_a<1.-1e-6),
        "source_geometry_hit_but_source_alpha_zero_mask":
            geo&(color_a<=1e-6),
        "source_visible_without_geometric_hit_count":
            int(np.count_nonzero(apparent_without_geo)),
        "source_compositor_used_full_or_native_to_decide_frontmost":False,
    }


def summary_view(material_frames,depth_frames):
    require(np.asarray(material_frames).shape==(8,4,192,192,4)
        and np.asarray(depth_frames).shape==(8,2,192,192,4),
        "V18N_ORIGINAL_SOURCE_MATERIAL_FULL_SAMPLE_LAYOUT")
    materials=np.asarray(material_frames,dtype=np.float64)
    assert materials.shape[-1]==4
    prediction=predict_material_from_source_isolates(
        materials[:,2:,:,:,:],depth_frames)
    native=premul(materials[:,0])
    soup=premul(materials[:,1])
    pix=prediction["predicted_sample_premultiplied_rgba"]
    valid=prediction["source_valid_sample_mask"]
    native_soup_error=np.max(np.abs(native-soup),axis=-1)
    pred_soup_error=np.max(np.abs(pix-soup),axis=-1)
    pred_native_error=np.max(np.abs(pix-native),axis=-1)
    valid_pixels=np.all(valid,axis=0)
    sample_diff_native_soup=int(np.count_nonzero(native_soup_error>TOL))
    sample_diff_pred_soup=int(np.count_nonzero(valid&(pred_soup_error>TOL)))
    sample_diff_pred_native=int(np.count_nonzero(valid&(pred_native_error>TOL)))
    ref_native=np.mean(native,axis=0)
    ref_soup=np.mean(soup,axis=0)
    pred_mean=np.mean(pix,axis=0)
    pix_err_source=np.max(np.abs(ref_native-pred_mean),axis=-1)
    pix_err_soup=np.max(np.abs(ref_soup-pred_mean),axis=-1)
    pix_err_native_soup=np.max(np.abs(ref_native-ref_soup),axis=-1)
    source_alpha=materials[...,3]
    # Baseline two-source-layer compositing cannot adapt to subpixel order:
    # calculate both fixed-order sample-first and independent postfilter.
    iso=premul(materials[:,2:])
    pre_gr=np.mean(over(iso[:,0],iso[:,1]),axis=0)
    pre_rg=np.mean(over(iso[:,1],iso[:,0]),axis=0)
    ga=np.mean(iso[:,0],axis=0)
    ra=np.mean(iso[:,1],axis=0)
    post_gr=over(ga,ra)
    post_rg=over(ra,ga)
    min_err=lambda a,b,ref:np.minimum(
        np.max(np.abs(a-ref),axis=-1),
        np.max(np.abs(b-ref),axis=-1))
    pre_soup_err=min_err(pre_gr,pre_rg,ref_soup)
    post_soup_err=min_err(post_gr,post_rg,ref_soup)
    require(int(np.count_nonzero(valid_pixels))+
            int(np.count_nonzero(~valid_pixels))==192*192,
            "V18N_ORIGINAL_APPEARANCE_FULL_PIXEL_UNKNOWN_CENSUS")
    return {
        "source_cycles_per_sample_instances":int(valid.size),
        "source_cycles_qualified_sample_instances":int(np.count_nonzero(valid)),
        "source_unknown_tied_sample_instances":int(
            np.count_nonzero(prediction["source_geometry_depth_tie_mask"])),
        "source_unknown_invalid_depth_sample_instances":int(
            np.count_nonzero(prediction["source_invalid_depth_mask"])),
        "source_8_sample_fully_qualified_pixels":int(
            np.count_nonzero(valid_pixels)),
        "source_8_sample_unknown_pixels":int(np.count_nonzero(~valid_pixels)),
        "original_source_material_native_vs_soup_sample_failures_full_domain":
            sample_diff_native_soup,
        "original_source_material_depth_compositor_vs_soup_sample_failures_qualified":
            sample_diff_pred_soup,
        "original_source_material_depth_compositor_vs_native_sample_failures_qualified":
            sample_diff_pred_native,
        "original_source_material_native_vs_soup_8sample_pixel_failures":
            int(np.count_nonzero(pix_err_native_soup>TOL)),
        "source_material_depth_compositor_vs_soup_8sample_pixel_failures_qualified":
            int(np.count_nonzero(valid_pixels&(pix_err_soup>TOL))),
        "source_material_depth_compositor_vs_native_8sample_pixel_failures_qualified":
            int(np.count_nonzero(valid_pixels&(pix_err_source>TOL))),
        "source_material_fixed_samplefirst_vs_soup_8sample_pixel_failures":
            int(np.count_nonzero(pre_soup_err>TOL)),
        "source_material_postfilter_vs_soup_8sample_pixel_failures":
            int(np.count_nonzero(post_soup_err>TOL)),
        "source_material_depth_compositor_recovers_fixed_order_failures":
            int(np.count_nonzero(valid_pixels&(pre_soup_err>TOL)
                                      &(pix_err_soup<=TOL))),
        "source_material_depth_compositor_regresses_fixed_order_passes":
            int(np.count_nonzero(valid_pixels&(pre_soup_err<=TOL)
                                      &(pix_err_soup>TOL))),
        "original_source_material_geometry_both_hit_sample_instances":int(
            np.count_nonzero(prediction["source_both_geometry_hit_mask"])),
        "original_source_material_geometry_gfront_sample_instances":int(
            np.count_nonzero(prediction["source_gfront_sample_mask"])),
        "original_source_material_geometry_rfront_sample_instances":int(
            np.count_nonzero(prediction["source_rfront_sample_mask"])),
        "original_source_native_nonbinary_1spp_alpha_instances":int(
            np.count_nonzero((source_alpha[:,0]>1e-6)&
                             (source_alpha[:,0]<1-1e-6))),
        "original_source_soup_nonbinary_1spp_alpha_instances":int(
            np.count_nonzero((source_alpha[:,1]>1e-6)&
                             (source_alpha[:,1]<1-1e-6))),
        "isolated_source_material_nonbinary_alpha_instances":int(
            np.count_nonzero(prediction["source_material_nonbinary_alpha_mask"])),
        "isolated_geometry_hit_but_color_transparent_instances":int(
            np.count_nonzero(prediction[
                "source_geometry_hit_but_source_alpha_zero_mask"])),
        "source_native_nonempty_sample_instances":int(
            np.count_nonzero(native[...,3]>1e-6)),
        "source_soup_nonempty_sample_instances":int(
            np.count_nonzero(soup[...,3]>1e-6)),
        "source_native_vs_soup_mean_max_premul_rgba_error":float(
            np.mean(native_soup_error)),
        "source_depth_compositor_vs_soup_mean_max_premul_rgba_error_qualified":
            float(np.mean(pred_soup_error[valid])) if np.any(valid) else None,
        "source_depth_compositor_predicted_without_native_full_lookup":True,
        "source_material_policy_is_admitted_replay_not_raw_original_artist_material":True,
        "source_unknown_is_not_positive_or_negative_drawable_edge":True,
    }


def audit(v18n_manifest,v18n_npz,v18m_manifest,v18m_npz,v18m_court,
          expected_v18m_court_sha):
    require(isinstance(expected_v18m_court_sha,str)
        and len(expected_v18m_court_sha)==64
        and sha(v18m_court)==expected_v18m_court_sha,
        "V18N_PARENT_EXACT_V18M_SOURCE_COURT_SHA_REQUIRED")
    n=load(v18n_manifest)
    m=load(v18m_manifest)
    c=load(v18m_court)
    require(n.get("schema")==MANIFEST
        and n.get("status")==RENDER_STATUS
        and m.get("schema")==V18M_MANIFEST
        and m.get("status")==V18M_RENDER_STATUS
        and c.get("schema")==V18M_COURT
        and c.get("status")==V18M_STATUS
        and c.get("oracle_free_source_depth_order_predictor") is True
        and c.get("source_original_v18m_manifest_sha256")==sha(v18m_manifest)
        and c.get("source_original_v18m_depth_rgba_sha256")==sha(v18m_npz)
        and n.get("source_depth_source_parent_v18m_court_sha256")==sha(v18m_court)
        and n.get("original_source_sha256")==m.get("original_source_sha256")==
            c.get("original_source_sha256")
        and n.get("subject_id")==m.get("subject_id")==
            c.get("original_source_subject_id_private")
        and n.get("source_original_v14_job_sha256")==m.get("source_original_v14_job_sha256")
        and n.get("source_reactive_stain_npz_sha256")==m.get("source_reactive_stain_npz_sha256")
        and n.get("original_source_camera_sha256")==m.get("original_source_camera_sha256")
        and n.get("heldout_source_camera_views")==list(CAMERAS)
        and n.get("original_source_cycles_sample_indices")==list(INDICES)
        and n.get("source_cycles_max_samples")==32
        and n.get("source_cycle_subset_length")==1
        and n.get("source_cycle_sampling_seed")==0
        and n.get("source_cycles_adaptive_sampling") is False
        and n.get("source_cycles_pixel_filter")=="BOX"
        and n.get("source_cycles_pixel_filter_width")==.01
        and n.get("source_material_policy")==
            "ORIGINAL_SEALED_SOURCE_MATERIAL_GRAPH_AND_UV_REPLAY"
        and n.get("source_appearance_material_policy")==
            "SEALED_SOURCE_APPEARANCE_POLICY_REPLAY"
        and n.get("source_artist_raw_untransformed_materials_not_claimed") is True
        and n.get("source_triangle_soup_uv_material_bindings_preserved") is True
        and n.get("source_edit_arms")==list(ARMS)
        and n.get("physical_face_visibility_operator")==
            "ACTUAL_TRIANGLE_REMOVAL_AND_OBJECT_RENDER_HIDE"
        and n.get("new_source_actual_edited_rgba_npz_sha256")==sha(v18n_npz)
        and 0<n.get("v18l_source_nontrivial_group_face_count",0)<
            n.get("v18l_original_source_face_count",0)
        and all(n.get(k) is False for k in NONPROMOTION)
        and all(c.get(k) is False for k in NONPROMOTION),
        "V18N_SOURCE_ORIGINAL_MATERIAL_EVIDENCE_PARENT_BINDING")
    with np.load(v18n_npz,allow_pickle=False) as z:
        require(z.files==["source_material_cycles_sample_rgba"],
                "V18N_SOURCE_ORIGINAL_MATERIAL_RGBA_MEMBER")
        rgba=np.asarray(z["source_material_cycles_sample_rgba"],np.float32)
    with np.load(v18m_npz,allow_pickle=False) as z:
        require(z.files==["source_cycles_depth_rgba"],
                "V18N_SOURCE_ORIGINAL_DEPTH_MEMBER")
        depth=np.asarray(z["source_cycles_depth_rgba"],np.float32)
    require(rgba.shape==(4,8,4,192,192,4)
        and depth.shape==(4,8,2,192,192,4)
        and np.isfinite(rgba).all() and np.isfinite(depth).all(),
        "V18N_SOURCE_RGBA_AND_DEPTH_FULL_CENSUS")
    views=[]
    for i,v in enumerate(CAMERAS):
        report=summary_view(rgba[i],depth[i])
        report["source_camera_view"]=v
        report["camera_partition"]=(
            "LOCKED_EVALUATION" if v in (4,6) else "EXPLORATORY")
        views.append(report)
    return {
        "schema":SCHEMA,"status":STATUS,
        "subject_id_private":n["subject_id"],
        "source_sha256":n["original_source_sha256"],
        "source_original_material_render_manifest_sha256":sha(v18n_manifest),
        "source_original_material_rgba_sha256":sha(v18n_npz),
        "source_original_depth_parent_manifest_sha256":sha(v18m_manifest),
        "source_original_depth_parent_rgba_sha256":sha(v18m_npz),
        "source_original_independent_depth_court_sha256":sha(v18m_court),
        "original_source_camera_reports":views,
        "source_appearance_replay_fidelity_measured_separately_from_compositor":True,
        "source_original_materials_not_replaced_by_constant_emission_colors":True,
        "source_material_alpha_and_rgb_disagreements_explicitly_counted":True,
        "source_native_full_is_scoring_only_not_predictor_input":True,
        "source_soup_full_is_scoring_only_not_predictor_input":True,
        "same_drawable_truth_minted":0,
        "different_drawable_truth_minted":0,
        "original_artist_drawable_identity":"UNKNOWN",
        "original_untransformed_artist_material_policy_qualified":False,
        "source_graphics_compiler_training_gold_authorized":False,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for x in ("v18n_manifest","v18n_npz","v18m_manifest","v18m_npz",
              "v18m_court","output"):
        p.add_argument("--"+x.replace("_","-"),type=Path,required=True)
    p.add_argument("--expected-v18m-court-sha",required=True)
    a=p.parse_args()
    out=audit(a.v18n_manifest,a.v18n_npz,a.v18m_manifest,a.v18m_npz,
              a.v18m_court,a.expected_v18m_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":out["status"],
        "source":out["subject_id_private"],
        "per_camera":out["original_source_camera_reports"],
        "source_native_full_is_scoring_only_not_predictor_input":True,
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
