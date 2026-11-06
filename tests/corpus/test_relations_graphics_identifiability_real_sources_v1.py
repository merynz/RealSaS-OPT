"""Strict source-court summary authority tests (synthetic JSONs, not source data)."""
from copy import deepcopy
import pytest
from tools.corpus.court_relations_graphics_identifiability_real_sources_v1 import (
    audit_original_courts, CAT, TRICERATOPS, PINS,
)


def faux_sealed_source_documents():
    flags={
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
    }
    local={
        "all":424,"uv_continuous":424,
        "at_least_one_mechanical_axis_divergent":59,
        "axis_divergent_but_uv_continuous":59,
        "pooled_mechanical_divergent":28,
    }
    lifted={
        "retrieval_candidates_total":1150,
        "same_vertex_pair_world_near_both_fit":497,
        "world_fit_near_with_coincident_uv":480,
        "world_fit_near_without_coincident_uv":17,
    }
    cat_src={
        "schema":"RealSaS.OriginalSourceUVWorldMechanicalRelationEvidence.v18s",
        "status":"SOURCE_UV_WORLD_MECHANICAL_RELATIONS_OBSERVED__OWNER_UNKNOWN",
        "original_source_sha256":CAT,
        "source_face_count":684,
        "source_graph_island_count":260,
        "source_local_relation_census":local,
        "source_lifted_relation_census":lifted,
        "source_face_independent_v18a_to_native_teacher_equal":True,
        "source_heldout_pose_ordinal_used_for_fit":False,
        "original_author_drawable_ownership":"UNKNOWN",
        "source_texture_rgba_continuity_across_uv":"UNKNOWN_NO_IMAGE_SAMPLE_COURT",
        **flags,
    }
    cat_geo_court={
        "schema":"RealSaS.IndependentOriginalUVWorldMechanicalRelationCourt.v18s",
        "status":"ORIGINAL_SOURCE_UV_WORLD_MECHANICS_RELATIONS_RECOMPUTED__OWNER_UNKNOWN",
        "original_source_sha256":CAT,
        "independently_replayed_source_v18s_sha256":PINS["v18s"],
        "source_original_face_count":684,
        "source_original_island_count":260,
        "source_local_evidence_recount":local,
        "source_lifted_evidence_recount":lifted,
        "heldout_pose2_not_consulted":True,
        **flags,
    }
    effects={
        "single_edit_sample_incidents_by_operation":[290979,264554,6048],
        "pairwise_visible_sample_support_overlap":{"0_1":2799,"0_2":0,"1_2":0},
        "heldout_source_pair_joint_sample_mismatches":0,
        "heldout_source_pair_joint_eight_sample_mean_mismatches":0,
        "source_alpha_drift":0,
    }
    cat_uv={
        "schema":"RealSaS.Corpus370SourceTextureUVEditConsequenceLedger.v1",
        "status":"SOURCE_CAT_LINKED_TEXTURE_EDIT_EVIDENCE_COURTED__FAMILY_AUTHORITY_PENDING",
        "original_source_sha256":CAT,
        "source_artist_drawable_owner":"UNKNOWN",
        "source_uv_atlas_edits_are_art_drawables":False,
        "source_visibility_occlusion_from_this_court":"UNKNOWN_NOT_MEASURED",
        "source_ray_depth_compositing_from_this_court":"UNKNOWN_NOT_MEASURED",
        "source_edit_consequence_totals":effects,
        "parent_receipts":{
            "source_original_independent_court_sha256":"e"*64},
        **flags,
    }
    cat_uv_court={
        "schema":"RealSaS.Corpus370IndependentCatSourceUVConsequenceLedgerCourt.v1",
        "status":"PASS_CAT_SOURCE_UV_GRAPHICS_LEDGER_ACCOUNTING__TRUTH_AUTHORITY_PENDING",
        "original_source_sha256":CAT,
        "source_ledger_sha256":PINS["cat_uv_ledger"],
        "original_independent_physical_court_sha256":"e"*64,
        "source_original_single_edit_sample_incidents_by_operation":[290979,264554,6048],
        "source_pairwise_visible_effect_overlap_sample_incidents":effects["pairwise_visible_sample_support_overlap"],
        "original_texture_linked_qualifying_source_faces":684,
        "source_original_texel_address_operations":3,
        "source_view_count":4,
        "source_family_authority":"PENDING",
        **flags,
    }
    tri_views=[]
    values=((1289,39010,2122),(820,39753,1848),
            (839,39753,1860),(1317,39010,2125))
    for view, changes in zip((1,3,5,7),values):
        tri_views.append({
            "source_view_index":view,
            "metric_rows":[
                {"arm":name,
                 "sample_rgba_mismatch_count":0,
                 "eight_sample_mean_rgba_mismatch_count":0,
                 "physical_full_edit_changed_sample_pixel_count":value}
                for name,value in zip((
                    "SOURCE_FULL","SOURCE_EDITED_FULL_0",
                    "SOURCE_EDITED_FULL_1","SOURCE_EDITED_FULL_2"
                ),(0,)+changes)
            ]
        })
    tri={
        "schema":"RealSaS.Corpus370IndependentOriginalThreePaletteCyclesEditCourt.v1",
        "status":"PASS_ORIGINAL_PHYSICAL_K3_SAME_CYCLES_SAMPLE_EDIT_COURT__OWNER_UNKNOWN",
        "original_source_sha256":TRICERATOPS,
        "source_artist_drawable_owner":"UNKNOWN",
        "source_group_count_K":3,
        "source_original_rest_only":True,
        "failure_reasons":[],
        "producer_court_source_full_not_used_to_choose_depth":True,
        "qualified_sample_total":1179646,
        "original_physical_overlapping_source_sample_count":14908,
        "original_camera_views":[1,3,5,7],
        "cycles_indices":[0,4,8,12,16,20,24,28],
        "source_per_view_results":tri_views,
        **flags,
    }
    return dict(v18s=cat_src,v18s_court=cat_geo_court,
                cat_uv_ledger=cat_uv,cat_uv_court=cat_uv_court,
                triceratops_k3_court=tri)


def test_valid_source_counterfactuals_are_distinct_from_face_edit_effect_joins():
    result=audit_original_courts(faux_sealed_source_documents())
    assert result["status"]=="INDEPENDENT_SOURCE_FAMILIES_RECONCILED__NO_SYNTHETIC_OWNER_LABEL"
    assert result["cat_local_source_graph_join_to_edited_sample_pixel"]["same_original_source"]
    assert not result["cat_local_source_graph_join_to_edited_sample_pixel"]["local_consequence_join_authorized"]
    assert result["cat_direct_face_to_edit_effect_truth"].startswith("UNKNOWN")
    assert result["original_subjects"][0]["qualified_source_relation_domains"][2]["count"]==561581
    assert result["original_subjects"][1]["qualified_source_relation_domains"][0]["count"]==1179646
    assert not result["iris_training_authorized"]


def test_forged_owner_gold_is_rejected_even_when_counts_match():
    evidence=faux_sealed_source_documents()
    evidence["v18s"]["graphics_owner_authorized"]=True
    with pytest.raises(ValueError,match="UNAUTHORIZED_OWNER"):
        audit_original_courts(evidence)


def test_dropped_independent_cat_source_court_is_not_a_valid_evidence_parent():
    evidence=faux_sealed_source_documents()
    evidence["cat_uv_court"]["original_independent_physical_court_sha256"]="f"*64
    with pytest.raises(ValueError,match="CAT_PHYSICAL_SOURCE_LEDGER"):
        audit_original_courts(evidence)


def test_court_cannot_impute_per_face_consequence_from_aggregate_rgb_pixels():
    evidence=faux_sealed_source_documents()
    x=audit_original_courts(evidence)
    assert x["cat_direct_face_to_edit_effect_truth"]=="UNKNOWN__NO_PIXEL_TO_SOURCE_FACE_EFFECT_BINDING"
    assert x["cat_local_source_graph_join_to_edited_sample_pixel"]["mismatched_scope_axes"]


def test_triceratops_single_collateral_failure_invalidates_physical_qualification():
    evidence=faux_sealed_source_documents()
    evidence["triceratops_k3_court"]["source_per_view_results"][1]["metric_rows"][2][
        "sample_rgba_mismatch_count"]=1
    with pytest.raises(ValueError,match="TRICERATOPS_SOURCE_PHYSICAL_PARITY"):
        audit_original_courts(evidence)


def test_actual_original_source_subject_sha_cannot_be_swapped():
    evidence=faux_sealed_source_documents()
    evidence["triceratops_k3_court"]["original_source_sha256"]=CAT
    with pytest.raises(ValueError,match="SOURCE_SUBJECT_SHA_JOIN"):
        audit_original_courts(evidence)
