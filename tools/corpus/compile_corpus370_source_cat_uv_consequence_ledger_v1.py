#!/usr/bin/env python3
"""Source-private Cat texture editing consequence ledger; NOT graphics owner truth.

Companion to original V18j source Graphics Evidence IR and Triceratops source
palette consequence ledger. Consumes independently SHA-courted physical source
texture edits and emits only source-conditioned candidate truth family facts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA="RealSaS.Corpus370SourceTextureUVEditConsequenceLedger.v1"
STATUS="SOURCE_CAT_LINKED_TEXTURE_EDIT_EVIDENCE_COURTED__FAMILY_AUTHORITY_PENDING"
PLAN="RealSaS.Corpus370OriginalSourceUVAtlasEditPlan.v2"
ASSETS="RealSaS.Corpus370OriginalSourceUVAtlasEditImages.v2"
RENDER="RealSaS.Corpus370OriginalCatSourceTextureSameCyclesEditEvidence.v2"
COURT="RealSaS.Corpus370IndependentOriginalCatSourceUVTextureEditCourt.v2"
PASS="PASS_ORIGINAL_CAT_LINKED_UV_EDIT_CONSEQUENCE_SUPERPOSITION__OWNER_UNKNOWN"
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","materialization_authorized","training_authorized")
CANDIDATES={
 "APPEARANCE_CAA":"SOURCE_LINKED_UV_ATLAS_FACE_ADDRESSABILITY_ONLY",
 "CAUSAL_RESPONSE":"SOURCE_PHYSICAL_RGB_TEXTURE_EDITS_AND_HELDOUT_JOINT_EFFECTS",
}
ARMS=("BASE","SINGLE_0","SINGLE_1","SINGLE_2",
      "PAIR_01","PAIR_02","PAIR_12","JOINT_012")
VIEWS=(1,3,5,7)
OFFSETS=(0,4,8,12,16,20,24,28)

def require(ok,code):
    if not ok:raise ValueError("CAT_GRAPHICS_SOURCE_LEDGER_"+code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def compile_evidence(plan,assets,renderer,court,contract,*,receipts):
    require(plan.get("schema")==PLAN and assets.get("schema")==ASSETS
        and renderer.get("schema")==RENDER and court.get("schema")==COURT
        and contract.get("schema")=="RealSaS.CorpusTruthAuthorityContract.v1",
        "SOURCE_FAMILY_SCHEMA")
    require(court.get("status")==PASS and court.get("failed_obligations")==[]
        and renderer.get("status")==
            "ORIGINAL_SOURCE_CAT_TEXTURE_UV_CYCLES_RENDERED__INDEPENDENT_COURT_REQUIRED",
        "INDEPENDENT_PHYSICAL_COURT_NOT_PASS")
    require(all(z.get(k) is False for z in (plan,assets,renderer,court)
                for k in FLAGS), "SOURCE_GRAPHICS_FORGED_MODEL_AUTHORITY")
    require(plan.get("source_uv_coordinate_origin")==
        "GLTF_2_IMAGE_TOP_LEFT__UV_V_INCREASES_DOWN"
        and plan.get("source_uv_to_png_y_flip_applied") is False
        and plan.get("original_source_texture_edit_rule")==
            "ATLAS_TILE_U8_RGB_INVERT__ALPHA_UNCHANGED"
        and plan.get("source_glb_triangle_count")==684
        and plan.get("source_texture_size_px")==[512,512]
        and plan.get("source_texture_source_image_alpha_all_opaque") is True
        and plan.get("selection_used_original_rendered_2d_image") is False
        and plan.get("operations_are_source_uv_texel_addresses_not_art_drawables") is True,
        "SOURCE_GLTF_UV_TEXTURE_DOMAIN")
    src=plan["source_original_glb_sha256"]
    atlas=plan["source_original_texture_sha256"]
    require(src==renderer.get("original_source_sha256")==
             court.get("source_original_glb_sha256")==
             assets.get("original_source_glb_sha256")==
             receipts["source_original_glb_sha256"] and
            atlas==assets.get("original_texture_source_sha256")==
             renderer.get("original_source_atlas_sha256")==
             court.get("original_source_image_sha256")==
             receipts["source_original_texture_sha256"],
            "ORIGINAL_SOURCE_IDENTITY")
    require(assets.get("source_operation_plan_sha256")==
        renderer.get("original_source_uv_edit_plan_sha256")==
        court.get("original_source_uv_edit_plan_sha256")==
        receipts["source_uv_plan_sha256"] and
        renderer.get("original_source_prepared_atlas_manifest_sha256")==
        court.get("original_source_atlas_edit_assets_manifest_sha256")==
        receipts["source_atlas_assets_sha256"] and
        court.get("original_source_blender_render_manifest_sha256")==
        receipts["original_render_manifest_sha256"] and
        renderer.get("source_original_sample_npz_sha256")==
        court.get("original_source_blender_sample_npz_sha256")==
        receipts["source_sample_npz_sha256_teacher_only"],
        "IMMUTABLE_SOURCE_PARENT_SHA_CHAIN")
    require(plan.get("source_original_camera_sha256")==
        renderer.get("original_camera_bundle_sha256") and
        plan.get("source_original_policy_sha256")==
        renderer.get("original_source_material_policy_sha256") and
        renderer.get("original_source_rest_only") is True and
        renderer.get("source_original_pose2_edit_court_passed") is False and
        renderer.get("all_texture_edit_sample_frames_independent_full_source_renders") is True
        and renderer.get("cycles_resolution")==192
        and renderer.get("cycles_seed")==0
        and renderer.get("cycles_configured_samples")==32
        and renderer.get("cycles_adaptive_sampling") is False
        and renderer.get("cycles_subset_length")==1
        and renderer.get("cycles_filter_width")==.01
        and renderer.get("cycles_pixel_filter")=="BOX"
        and renderer.get("cycles_subset_indices")==list(OFFSETS)
        and renderer.get("camera_indices")==list(VIEWS)
        and renderer.get("original_source_uv_texture_arms")==list(ARMS)
        and renderer.get("source_original_sample_npz_shape")==[4,8,8,192,192,4]
        and court.get("physical_original_sample_frames")==256
        and court.get("source_face_count")==684
        and court.get("source_pose_scope")=="ORIGINAL_CAT_REST_ONLY"
        and court.get("source_sampler_tolerance_rgba_premultiplied_max")==.025
        and court.get("source_unknown_alpha_drift")==0
        and court.get("source_atlas_region_is_art_drawable") is False
        and court.get("source_one_sample_edit_effect_from_raw_glb_without_single_render")==
            "NOT_ESTABLISHED",
        "ORIGINAL_SOURCE_RENDER_AND_EVAL_DOMAIN")
    require(contract.get("authority_boundary",{}).get("this_contract_mints_training_readiness")
        is False and "PENDING" in contract.get("family_authority_states",[])
        and set(CANDIDATES).issubset(set(contract.get("family_order",[]))),
        "CANONICAL_FAMILY_CONTRACT")
    edits=plan.get("selected_original_source_uv_edits")
    assetrows=assets.get("original_atlas_arm_rows")
    require(type(edits) is list and len(edits)==3 and
        type(assetrows) is list and len(assetrows)==8 and
        [r.get("arm") for r in assetrows]==list(ARMS) and
        [r.get("original_atlas_edited_rgb_pixel_count") for r in assetrows]==
            [0,4096,4096,4096,8192,8192,8192,12288] and
        all(z.get("source_texture_alpha_preserved_exactly") is True and
             z.get("atlas_pixel_edit_equals_original_declared_source_mask") is True
             for z in assetrows) and
        [z["source_atlas_rgba_sha256"] for z in assetrows]==
            renderer.get("source_atlas_texture_arm_sha256") and
        court.get("original_source_texture_uv_tile_operations")==
            [[z["uv_tile_u"],z["uv_tile_v"]] for z in edits],
        "EXACT_ORIGINAL_UV_EDIT_OPERATIONS")
    for i,z in enumerate(edits):
        u,v=z["uv_tile_u"],z["uv_tile_v"]
        require(z.get("source_uv_edit_operation_index")==i and
            z.get("source_rgba_png_pixel_rectangle_xyxy_exclusive")==
                [u*64,v*64,(u+1)*64,(v+1)*64]
            and z.get("source_artist_drawable_identity")=="UNKNOWN",
            "ORIGINAL_UV_EDIT_ADDRESS_DRIFT")
    rows=court.get("independent_court_per_camera")
    require(type(rows) is list and len(rows)==4 and
        [z.get("original_source_view") for z in rows]==list(VIEWS),
        "SOURCE_VIEW_COURT_CENSUS")
    all_effects=[0,0,0]
    all_overlap={"0_1":0,"0_2":0,"1_2":0}
    output_views=[]
    for v,z in zip(VIEWS,rows):
        counts=z.get("edited_single_source_visible_sample_count")
        pair=z.get("pairwise_single_edit_image_support_overlaps")
        checks=z.get("source_heldout_pair_joint_court")
        require(type(counts) is list and len(counts)==3 and
            all(type(x) is int and 0<=x<=8*192*192 for x in counts) and
            type(pair) is dict and set(pair)==set(all_overlap) and
            all(type(x) is int and 0<=x<=min(counts[int(key[0])],
                                              counts[int(key[-1])])
                for key,x in pair.items()) and
            z.get("all_source_arm_alpha_max_drift")==0 and
            type(checks) is list and len(checks)==4,
            "ORIGINAL_SOURCE_PER_VIEW_EFFECT_CENSUS")
        for got,name,selection in zip(checks,
            ("PAIR_01","PAIR_02","PAIR_12","JOINT_012"),
            ([0,1],[0,2],[1,2],[0,1,2])):
            require(got.get("actual_heldout_full_source_arm")==name and
                got.get("source_single_edited_arm_input_indices")==selection and
                got.get("prediction_uses_heldout_pair_or_joint_target") is False
                and got.get("source_per_sample_rgba_mismatch_count")==0
                and got.get("source_eight_sample_average_rgba_mismatch_count")==0
                and 0<=got.get("source_per_sample_rgba_max_delta",99)<=.025
                and 0<=got.get("source_eight_sample_average_rgba_max_delta",99)<=.025,
                "SOURCE_PAIR_JOINT_INDEPENDENT_HOLDOUT")
        all_effects=[x+y for x,y in zip(all_effects,counts)]
        for k in all_overlap:all_overlap[k]+=pair[k]
        output_views.append({
            "original_source_pose":"BIND_REST_ONLY",
            "source_view":v,
            "sample_offsets":list(OFFSETS),
            "effect_measurement_granularity":"SOURCE_SAMPLED_IMAGE_AGGREGATE_NOT_ART_DRAWABLE",
            "source_individual_uv_edit_sample_incidents":counts,
            "source_pairwise_visible_effect_support_overlap":pair,
            "source_edit_alpha_drift":0,
            "heldout_pair_joint_sample_mismatch_count":0,
            "heldout_pair_joint_eight_sample_mean_mismatch_count":0,
            "source_drawable_owner":"UNKNOWN",
        })
    require(all_effects==court.get("source_single_edit_observable_samples_per_operation")
        and all(x>=32 for x in all_effects)
        and all_overlap["0_1"]>0,
        "SOURCE_EFFECT_NONVACUITY_AND_OVERLAP")
    candidate_counts={"APPEARANCE_CAA":684,
                      "CAUSAL_RESPONSE":sum(all_effects)}
    return {
        "schema":SCHEMA,"status":STATUS,
        "original_source_teacher_private":True,
        "subject_id":plan["subject_id"],"original_source_sha256":src,
        "original_texture_sha256":atlas,"parent_receipts":receipts,
        "source_operation_domain":"ORIGINAL_GLTF_UV_TEXTURE_RGB_EDIT__NO_DRAWABLE_IDENTITY",
        "source_uv_edit_operations":edits,"source_uv_atlas_edits_are_art_drawables":False,
        "source_graphics_effect_per_view":output_views,
        "source_edit_consequence_totals":{
            "single_edit_sample_incidents_by_operation":all_effects,
            "pairwise_visible_sample_support_overlap":all_overlap,
            "heldout_source_pair_joint_sample_mismatches":0,
            "heldout_source_pair_joint_eight_sample_mean_mismatches":0,
            "source_alpha_drift":0,
        },
        "family_matrix_bridge":{
            family:{
              "candidate_source_observation":kind,
              "source_qualified_observation_count":candidate_counts[family],
              "final_family_authority_state":"PENDING",
              "final_training_validity_mask_sha256":None,
              "student_access_to_original_teacher_source":False,
              "training_target_minted":False,
            } for family,kind in CANDIDATES.items()
        },
        "source_visibility_occlusion_from_this_court":"UNKNOWN_NOT_MEASURED",
        "source_ray_depth_compositing_from_this_court":"UNKNOWN_NOT_MEASURED",
        "source_pose2_graphics_relations":"UNKNOWN_NOT_MEASURED",
        "source_artist_drawable_owner":"UNKNOWN",
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        "iris_student_training_targets":0,
        **{z:False for z in FLAGS},
    }

def compile_from_paths(plan_path,assets_path,render_path,court_path,contract_path,
                       *,expected_court_sha):
    require(sha(court_path)==expected_court_sha and len(expected_court_sha)==64,
            "INDEPENDENT_COURT_SHA_REQUIRED")
    p,a,r,c,t=(load(x) for x in (plan_path,assets_path,render_path,court_path,contract_path))
    receipts={
        "source_original_glb_sha256":p["source_original_glb_sha256"],
        "source_original_texture_sha256":p["source_original_texture_sha256"],
        "source_uv_plan_sha256":sha(plan_path),
        "source_atlas_assets_sha256":sha(assets_path),
        "original_render_manifest_sha256":sha(render_path),
        "source_original_independent_court_sha256":sha(court_path),
        "source_sample_npz_sha256_teacher_only":r["source_original_sample_npz_sha256"],
        "canonical_truth_family_contract_sha256":sha(contract_path),
    }
    return compile_evidence(p,a,r,c,t,receipts=receipts)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ("source-plan","atlas-assets","original-render-manifest",
              "independent-court","truth-contract","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--expected-independent-court-sha256",required=True)
    x=ap.parse_args()
    result=compile_from_paths(x.source_plan,x.atlas_assets,x.original_render_manifest,
        x.independent_court,x.truth_contract,
        expected_court_sha=x.expected_independent_court_sha256)
    x.output.parent.mkdir(parents=True,exist_ok=True)
    x.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],
        "original_source_texture_single_edit_sample_incidents":
            result["source_edit_consequence_totals"]["single_edit_sample_incidents_by_operation"],
        "owner_labels":0,"training_authorized":False},sort_keys=True))

if __name__=="__main__":main()
