"""Adversarial contracts for source graphics consequence ledger and independent court."""
import copy
import pytest

from tools.corpus.compile_corpus370_source_graphics_consequence_ledger_v1 import compose
from tools.corpus.court_corpus370_source_graphics_consequence_ledger_v1 import verify

SOURCE="a"*64
COLORS=[["0x1.0p-1"]*4,["0x1.8p-1"]*4,["0x1.0p-2"]*4]
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","materialization_authorized","training_authorized")
PARENTS={
    "original_source_sha256":SOURCE,
    "palette_facts_sha256":"b"*64,
    "renderer_manifest_sha256":"c"*64,
    "independent_render_court_sha256":"d"*64,
    "independent_native_support_court_sha256":"e"*64,
    "truth_authority_contract_sha256":"f"*64,
    "source_sample_npz_sha256_teacher_only":"1"*64,
    "original_rest_geometry_npz_sha256_teacher_only":"2"*64,
    "original_camera_bundle_sha256_teacher_only":"3"*64,
    "original_render_policy_sha256_teacher_only":"4"*64,
}

def sources():
    palette={
        "schema":"RealSaS.Corpus370OriginalShaderConstantPaletteFacts.v1",
        "original_source_sha256":SOURCE,
        "palette_operation_count":3,"source_face_count":6,
        "qualified_constant_face_count":6,"unsupported_face_count":0,
        "face_palette_validity":[True]*6,
        "face_palette_color_key_index_or_null":[0,1,2,0,1,2],
        "constant_palette_color_keys_hex_rgba":copy.deepcopy(COLORS),
        **{k:False for k in FLAGS},
    }
    render={
        "schema":"RealSaS.Corpus370OriginalThreePaletteCommonCyclesSampleEditEvidence.v1",
        "status":"SOURCE_ORIGINAL_K3_CYCLES_EDIT_SAMPLES_RENDERED__COURT_REQUIRED",
        "original_source_sha256":SOURCE,
        "source_original_material_palette_sha256":PARENTS["palette_facts_sha256"],
        "source_original_edited_sample_npz_sha256":PARENTS["source_sample_npz_sha256_teacher_only"],
        "source_cycle_camera_view_indices":[1,3],
        "source_cycle_sample_indices":[0,4,8,12,16,20,24,28],
        "source_cycle_resolution":4,
        "source_original_rest_pose_only":True,
        "source_original_artwork_heldout_pose_test":False,
        "physical_nonselected_faces_removed":True,
        "source_full_and_edited_full_are_court_targets_only":True,
        "source_original_triangles_per_color":[2,2,2],
        **{k:False for k in FLAGS},
    }
    rows=[]
    for view in (1,3):
        rows.append({
            "source_view_index":view,
            "source_full_alpha_intermediate_single_source_sample_count":0,
            "source_isolate_nonbinary_alpha_count":0,
            "source_invalid_depth_sample_count":0,
            "source_unknown_depth_tie_sample_count":0,
            "source_multi_group_overlapping_sample_count":12,
            "metric_rows":[{
                "arm":name,
                "qualified_source_sample_pixels":128,
                "unknown_source_samples":0,
                "fully_qualified_eight_sample_pixels":16,
                "sample_rgba_mismatch_count":0,
                "eight_sample_mean_rgba_mismatch_count":0,
                "physical_full_edit_changed_sample_pixel_count":0 if i==0 else [9,13,16][i-1],
            } for i,name in enumerate((
                "SOURCE_FULL","SOURCE_EDITED_FULL_0","SOURCE_EDITED_FULL_1",
                "SOURCE_EDITED_FULL_2"))],
        })
    court={
        "schema":"RealSaS.Corpus370IndependentOriginalThreePaletteCyclesEditCourt.v1",
        "status":"PASS_ORIGINAL_PHYSICAL_K3_SAME_CYCLES_SAMPLE_EDIT_COURT__OWNER_UNKNOWN",
        "failure_reasons":[],
        "original_source_sha256":SOURCE,
        "original_source_palette_facts_sha256":PARENTS["palette_facts_sha256"],
        "original_source_renderer_manifest_sha256":PARENTS["renderer_manifest_sha256"],
        "original_source_renderer_rgba_npz_sha256":PARENTS["source_sample_npz_sha256_teacher_only"],
        "original_camera_views":[1,3],
        "cycles_indices":[0,4,8,12,16,20,24,28],
        "source_group_count_K":3,
        "producer_court_source_full_not_used_to_choose_depth":True,
        "source_original_rest_only":True,
        "qualified_sample_total":256,
        "original_physical_overlapping_source_sample_count":24,
        "source_per_view_results":rows,
        **{k:False for k in FLAGS},
    }
    native={
        "schema":"RealSaS.Corpus370IndependentOperationalNativeSupportCourt.v1",
        "subject_id_private":"synthetic_subject",
        "original_source_sha256":SOURCE,"source_face_count":6,
        **{k:False for k in FLAGS},
    }
    contract={
        "schema":"RealSaS.CorpusTruthAuthorityContract.v1",
        "family_order":["APPEARANCE_CAA","VISIBILITY_OCCLUSION",
                         "RAY_COMPOSITING","CAUSAL_RESPONSE"],
        "family_authority_states":["PENDING"],
        "authority_boundary":{"this_contract_mints_training_readiness":False},
    }
    return palette,render,court,native,contract

def complete():
    inputs=sources()
    ledger=compose(*inputs,parents=PARENTS,subject_id="synthetic_subject")
    receipt={**PARENTS,"ledger_sha256":"5"*64}
    return inputs,ledger,receipt

def test_true_source_consequence_ledger_stays_pending_and_works_with_independent_court():
    inputs,ledger,receipt=complete()
    result=verify(ledger,*inputs,sha_index=receipt)
    assert result["status"].startswith("PASS_SOURCE_CONSEQUENCE_LEDGER_ACCOUNTING")
    assert result["source_qualified_sample_incidents"]==256
    assert result["source_overlapping_geometry_sample_incidents"]==24
    assert result["source_edit_changed_sample_incidents_by_color"]==[18,26,32]
    assert all(x["final_family_authority_state"]=="PENDING"
               for x in ledger["family_matrix_bridge"].values())
    assert ledger["same_drawable_training_pairs_qualified"]==0
    assert ledger["unmeasured_or_unqualified_domains"]["POSE_PERSISTENT_GRAPHICS_RELATIONS"].startswith("UNKNOWN")
    assert ledger["original_source_view_edit_consequence"][0]["spatial_granularity"].startswith("VIEW_AGGREGATE")

@pytest.mark.parametrize("family",[
    "APPEARANCE_CAA","VISIBILITY_OCCLUSION","RAY_COMPOSITING","CAUSAL_RESPONSE"])
def test_source_proof_cannot_mint_final_training_family(family):
    inputs,ledger,receipt=complete()
    ledger["family_matrix_bridge"][family]["final_family_authority_state"]="QUALIFIED_CORE"
    with pytest.raises(ValueError,match="FAMILY_AUTHORITY_UNLAWFUL_PROMOTION"):
        verify(ledger,*inputs,sha_index=receipt)

def test_source_owner_pair_forgery_rejected():
    inputs,ledger,receipt=complete()
    ledger["same_drawable_training_pairs_qualified"]=1
    with pytest.raises(ValueError,match="FALSE_OWNER"):
        verify(ledger,*inputs,sha_index=receipt)

def test_original_source_parent_tamper_rejected():
    inputs,ledger,receipt=complete()
    ledger["parent_receipts"]["independent_render_court_sha256"]="0"*64
    with pytest.raises(ValueError,match="SHA_AND_SUBJECT"):
        verify(ledger,*inputs,sha_index=receipt)

def test_missing_camera_or_edit_is_not_validated_as_success():
    inputs,ledger,receipt=complete()
    inputs[2]["source_per_view_results"][1]["metric_rows"].pop()
    with pytest.raises(ValueError,match="SOURCE_EDIT_ARMS"):
        compose(*inputs,parents=PARENTS,subject_id="synthetic_subject")

def test_nonbinary_alpha_does_not_enter_candidate_family():
    inputs,_,_=complete()
    inputs[2]["source_per_view_results"][0]["source_isolate_nonbinary_alpha_count"]=1
    with pytest.raises(ValueError,match="PER_VIEW_SOURCE_OR_ALPHA"):
        compose(*inputs,parents=PARENTS,subject_id="synthetic_subject")

def test_invalid_independent_source_court_status_refused():
    inputs,_,_=complete()
    inputs[2]["status"]="FAIL_ORIGINAL_PHYSICAL_K3_SAME_CYCLES_SAMPLE_EDIT_COURT"
    with pytest.raises(ValueError,match="INDEPENDENT_RENDER_COURT_NOT_PASS"):
        compose(*inputs,parents=PARENTS,subject_id="synthetic_subject")

def test_no_source_teacher_observation_into_student_target():
    inputs,ledger,receipt=complete()
    ledger["family_matrix_bridge"]["RAY_COMPOSITING"]["student_access_to_original_teacher_source"]=True
    with pytest.raises(ValueError,match="FAMILY_AUTHORITY_UNLAWFUL_PROMOTION"):
        verify(ledger,*inputs,sha_index=receipt)

def test_original_pose2_claim_cannot_be_minted_from_rest():
    inputs,ledger,receipt=complete()
    ledger["source_pose2_tested"]=True
    with pytest.raises(ValueError,match="FALSE_OWNER_OR_TRAINING_AUTHORITY"):
        verify(ledger,*inputs,sha_index=receipt)

def test_unknown_domains_cannot_be_backfilled_as_known():
    inputs,ledger,receipt=complete()
    ledger["unmeasured_or_unqualified_domains"]["ATTACHMENT_AND_DETACHMENT"]="PASS"
    with pytest.raises(ValueError,match="UNKNOWN_GRAPHICS_DOMAIN_RECLASSIFIED"):
        verify(ledger,*inputs,sha_index=receipt)
