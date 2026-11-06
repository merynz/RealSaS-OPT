"""Adversarial source-only Cat UV consequence ledger and independent court contracts."""
import copy
import pytest
from tools.corpus.compile_corpus370_source_cat_uv_consequence_ledger_v1 import (
    compile_evidence
)
from tools.corpus.court_corpus370_source_cat_uv_consequence_ledger_v1 import (
    verify
)

SOURCE="a"*64
ATLAS="b"*64
PARENTS={
 "source_original_glb_sha256":SOURCE,
 "source_original_texture_sha256":ATLAS,
 "source_uv_plan_sha256":"c"*64,
 "source_atlas_assets_sha256":"d"*64,
 "original_render_manifest_sha256":"e"*64,
 "source_original_independent_court_sha256":"f"*64,
 "source_sample_npz_sha256_teacher_only":"1"*64,
 "canonical_truth_family_contract_sha256":"2"*64,
}
F=("graphics_owner_authorized","iris_training_authorized","tessa_training_authorized",
   "materialization_authorized","training_authorized")
NEG={x:False for x in F}
ARMS=["BASE","SINGLE_0","SINGLE_1","SINGLE_2",
      "PAIR_01","PAIR_02","PAIR_12","JOINT_012"]

def make():
    actions=[{
        "source_uv_edit_operation_index":i,
        "uv_tile_u":u,"uv_tile_v":v,
        "source_rgba_png_pixel_rectangle_xyxy_exclusive":[u*64,v*64,(u+1)*64,(v+1)*64],
        "source_artist_drawable_identity":"UNKNOWN",
    } for i,(u,v) in enumerate(((6,6),(6,7),(7,6)))]
    plan={
        "schema":"RealSaS.Corpus370OriginalSourceUVAtlasEditPlan.v2",
        "subject_id":"cat_priv",
        "source_original_glb_sha256":SOURCE,
        "source_original_texture_sha256":ATLAS,
        "source_original_camera_sha256":"3"*64,
        "source_original_policy_sha256":"4"*64,
        "source_uv_coordinate_origin":"GLTF_2_IMAGE_TOP_LEFT__UV_V_INCREASES_DOWN",
        "source_uv_to_png_y_flip_applied":False,
        "original_source_texture_edit_rule":"ATLAS_TILE_U8_RGB_INVERT__ALPHA_UNCHANGED",
        "source_glb_triangle_count":684,"source_texture_size_px":[512,512],
        "source_texture_source_image_alpha_all_opaque":True,
        "selection_used_original_rendered_2d_image":False,
        "operations_are_source_uv_texel_addresses_not_art_drawables":True,
        "selected_original_source_uv_edits":actions,**NEG,
    }
    sets=((),(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))
    assets={
        "schema":"RealSaS.Corpus370OriginalSourceUVAtlasEditImages.v2",
        "original_source_glb_sha256":SOURCE,
        "original_texture_source_sha256":ATLAS,
        "source_operation_plan_sha256":PARENTS["source_uv_plan_sha256"],
        "original_atlas_arm_rows":[{
            "arm":name,
            "selected_source_uv_operation_indices":list(select),
            "source_atlas_rgba_sha256":str(i)*64,
            "original_atlas_edited_rgb_pixel_count":4096*len(select),
            "source_texture_alpha_preserved_exactly":True,
            "atlas_pixel_edit_equals_original_declared_source_mask":True,
        } for i,(name,select) in enumerate(zip(ARMS,sets))],
        **NEG,
    }
    render={
        "schema":"RealSaS.Corpus370OriginalCatSourceTextureSameCyclesEditEvidence.v2",
        "status":"ORIGINAL_SOURCE_CAT_TEXTURE_UV_CYCLES_RENDERED__INDEPENDENT_COURT_REQUIRED",
        "original_source_sha256":SOURCE,"original_source_atlas_sha256":ATLAS,
        "original_camera_bundle_sha256":plan["source_original_camera_sha256"],
        "original_source_material_policy_sha256":plan["source_original_policy_sha256"],
        "original_source_uv_edit_plan_sha256":PARENTS["source_uv_plan_sha256"],
        "original_source_prepared_atlas_manifest_sha256":PARENTS["source_atlas_assets_sha256"],
        "source_original_sample_npz_sha256":PARENTS["source_sample_npz_sha256_teacher_only"],
        "source_atlas_texture_arm_sha256":[r["source_atlas_rgba_sha256"]
                                          for r in assets["original_atlas_arm_rows"]],
        "camera_indices":[1,3,5,7],
        "cycles_subset_indices":[0,4,8,12,16,20,24,28],
        "cycles_resolution":192,"cycles_seed":0,"cycles_configured_samples":32,
        "cycles_adaptive_sampling":False,"cycles_subset_length":1,
        "cycles_filter_width":.01,"cycles_pixel_filter":"BOX",
        "original_source_uv_texture_arms":ARMS,
        "source_original_sample_npz_shape":[4,8,8,192,192,4],
        "original_source_rest_only":True,
        "source_original_pose2_edit_court_passed":False,
        "all_texture_edit_sample_frames_independent_full_source_renders":True,
        **NEG,
    }
    views=[]
    for view in [1,3,5,7]:
        views.append({
            "original_source_view":view,
            "edited_single_source_visible_sample_count":[25,50,0 if view in [3,5] else 40],
            "pairwise_single_edit_image_support_overlaps":{"0_1":3,"0_2":0,"1_2":0},
            "all_source_arm_alpha_max_drift":0,
            "source_heldout_pair_joint_court":[{
                "actual_heldout_full_source_arm":name,
                "source_single_edited_arm_input_indices":list(inputs),
                "prediction_uses_heldout_pair_or_joint_target":False,
                "source_per_sample_rgba_mismatch_count":0,
                "source_eight_sample_average_rgba_mismatch_count":0,
                "source_per_sample_rgba_max_delta":.003,
                "source_eight_sample_average_rgba_max_delta":.002,
            } for name,inputs in [
                ("PAIR_01",(0,1)),("PAIR_02",(0,2)),
                ("PAIR_12",(1,2)),("JOINT_012",(0,1,2))]],
        })
    court={
        "schema":"RealSaS.Corpus370IndependentOriginalCatSourceUVTextureEditCourt.v2",
        "status":"PASS_ORIGINAL_CAT_LINKED_UV_EDIT_CONSEQUENCE_SUPERPOSITION__OWNER_UNKNOWN",
        "failed_obligations":[],
        "source_original_glb_sha256":SOURCE,"original_source_image_sha256":ATLAS,
        "original_source_uv_edit_plan_sha256":PARENTS["source_uv_plan_sha256"],
        "original_source_atlas_edit_assets_manifest_sha256":PARENTS["source_atlas_assets_sha256"],
        "original_source_blender_render_manifest_sha256":PARENTS["original_render_manifest_sha256"],
        "original_source_blender_sample_npz_sha256":PARENTS["source_sample_npz_sha256_teacher_only"],
        "source_face_count":684,
        "original_source_texture_uv_tile_operations":[[6,6],[6,7],[7,6]],
        "physical_original_sample_frames":256,
        "source_pose_scope":"ORIGINAL_CAT_REST_ONLY",
        "source_sampler_tolerance_rgba_premultiplied_max":.025,
        "source_unknown_alpha_drift":0,
        "source_atlas_region_is_art_drawable":False,
        "source_one_sample_edit_effect_from_raw_glb_without_single_render":"NOT_ESTABLISHED",
        "source_single_edit_observable_samples_per_operation":[100,200,80],
        "independent_court_per_camera":views,
        **NEG,
    }
    contract={
        "schema":"RealSaS.CorpusTruthAuthorityContract.v1",
        "family_order":["APPEARANCE_CAA","CAUSAL_RESPONSE","RAY_COMPOSITING","VISIBILITY_OCCLUSION"],
        "family_authority_states":["PENDING"],
        "authority_boundary":{"this_contract_mints_training_readiness":False},
    }
    return plan,assets,render,court,contract

def complete():
    docs=make()
    ledger=compile_evidence(*docs,receipts=PARENTS)
    return docs,ledger,{**PARENTS,"source_ledger_sha256":"5"*64}

def test_source_uv_evidence_compiles_and_is_independently_checked():
    docs,ledger,parents=complete()
    result=verify(ledger,*docs,parents=parents)
    assert result["status"]=="PASS_CAT_SOURCE_UV_GRAPHICS_LEDGER_ACCOUNTING__TRUTH_AUTHORITY_PENDING"
    assert result["source_original_single_edit_sample_incidents_by_operation"]==[100,200,80]
    assert result["source_pairwise_visible_effect_overlap_sample_incidents"]["0_1"]==12
    assert set(ledger["family_matrix_bridge"])=={"APPEARANCE_CAA","CAUSAL_RESPONSE"}
    assert ledger["source_visibility_occlusion_from_this_court"]=="UNKNOWN_NOT_MEASURED"

@pytest.mark.parametrize("family",["APPEARANCE_CAA","CAUSAL_RESPONSE"])
def test_source_uv_court_cannot_mint_training_family(family):
    docs,ledger,parents=complete()
    ledger["family_matrix_bridge"][family]["final_family_authority_state"]="QUALIFIED_CORE"
    with pytest.raises(ValueError,match="CANDIDATE_FAMILY_ILLEGAL_PROMOTION"):
        verify(ledger,*docs,parents=parents)

def test_no_source_teacher_leakage_into_student_head():
    docs,ledger,parents=complete()
    ledger["family_matrix_bridge"]["CAUSAL_RESPONSE"][
        "student_access_to_original_teacher_source"]=True
    with pytest.raises(ValueError,match="CANDIDATE_FAMILY_ILLEGAL_PROMOTION"):
        verify(ledger,*docs,parents=parents)

def test_fake_drawable_pairs_rejected():
    docs,ledger,parents=complete()
    ledger["same_drawable_training_pairs_qualified"]=7
    with pytest.raises(ValueError,match="ILLEGAL_DRAWABLE_OR_TRAINING_AUTHORITY"):
        verify(ledger,*docs,parents=parents)

def test_tampered_camera_source_effect_rejected():
    docs,ledger,parents=complete()
    ledger["source_graphics_effect_per_view"][2][
        "source_individual_uv_edit_sample_incidents"][1]=20000
    with pytest.raises(ValueError,match="SOURCE_CAM_EDIT_EFFECT_PROJECTION"):
        verify(ledger,*docs,parents=parents)

def test_no_silent_source_texture_glb_sha_substitution():
    docs,ledger,parents=complete()
    docs[0]["source_original_glb_sha256"]="0"*64
    with pytest.raises(ValueError,match="SOURCE_SHA_COURT_LINEAGE"):
        verify(ledger,*docs,parents=parents)

def test_missing_heldout_joint_target_is_not_a_pass():
    docs=list(make())
    docs[3]["independent_court_per_camera"][0]["source_heldout_pair_joint_court"].pop()
    with pytest.raises(ValueError,match="ORIGINAL_SOURCE_PER_VIEW_EFFECT_CENSUS"):
        compile_evidence(*docs,receipts=PARENTS)

def test_no_silent_original_uv_v_axis_flip():
    docs=list(make())
    docs[0]["source_uv_to_png_y_flip_applied"]=True
    with pytest.raises(ValueError,match="SOURCE_GLTF_UV_TEXTURE_DOMAIN"):
        compile_evidence(*docs,receipts=PARENTS)

def test_no_reclassification_of_alpha_drift_as_safe():
    docs=list(make())
    docs[3]["source_unknown_alpha_drift"]=.01
    with pytest.raises(ValueError,match="ORIGINAL_SOURCE_RENDER_AND_EVAL_DOMAIN"):
        compile_evidence(*docs,receipts=PARENTS)

def test_missing_op_edit_nonvacuity_rejected():
    docs=list(make())
    docs[3]["source_single_edit_observable_samples_per_operation"]=[100,200,0]
    with pytest.raises(ValueError,match="SOURCE_EFFECT_NONVACUITY_AND_OVERLAP"):
        compile_evidence(*docs,receipts=PARENTS)

def test_fake_visibility_family_rejected():
    docs,ledger,parents=complete()
    ledger["family_matrix_bridge"]["VISIBILITY_OCCLUSION"]={
        "final_family_authority_state":"PENDING",
    }
    with pytest.raises(ValueError,match="CANDIDATE_FAMILY_SCOPE"):
        verify(ledger,*docs,parents=parents)

def test_wrong_original_source_npz_sha_not_accepted():
    docs,ledger,parents=complete()
    docs[2]["source_original_sample_npz_sha256"]="9"*64
    with pytest.raises(ValueError,match="SOURCE_SHA_COURT_LINEAGE"):
        verify(ledger,*docs,parents=parents)

def test_candidate_colour_source_edits_do_not_authorize_motion():
    docs,ledger,parents=complete()
    ledger["source_pose2_graphics_relations"]="QUALIFIED"
    with pytest.raises(ValueError,match="ILLEGAL_DRAWABLE_OR_TRAINING_AUTHORITY"):
        verify(ledger,*docs,parents=parents)
