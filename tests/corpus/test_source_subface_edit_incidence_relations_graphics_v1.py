"""Synthetic research checks: source face != uniform UV-edit response atom.

All synthetic addresses are invented. Actual original Cat facts require the
SHA-locked independent source court in GitHub Actions.
"""
import pytest

from tools.corpus.compile_source_subface_edit_incidence_relations_graphics_v1 import (
    build_cross_scope_source_profiles,
)


def pixel(face, view, x, y, edit):
    return {
        "face_index":face, "original_camera_view":view, "pixel_x":x,
        "pixel_y_top_first":y, "source_uv_edit_operation_index":edit,
    }


def local(a, b, axis=True):
    return {
        "original_face_a":a, "original_face_b":b,
        "source_uv_address_continuity":"SAME_ORIGINAL_UV_ENDPOINT_COORDINATES",
        "source_mechanical_axis_divergent":axis,
        "source_mechanical_pooled_divergent":False,
    }


def lifted(a,b,near=True):
    return {
        "source_island_a":a,"source_island_b":b,
        "original_fit_pose_world_near":near,
        "original_near_pairs_same_uv_endpoint_count":1 if near else 0,
    }


def test_same_original_face_two_edited_pixels_one_view_falsify_uniform_face_control():
    rows=[
        pixel(0,1,10,10,0), pixel(0,1,11,10,1),
        pixel(1,1,12,10,0), pixel(2,3,13,10,2),
    ]
    result=build_cross_scope_source_profiles(
        rows, [local(0,1)], [0,0,1,2], [lifted(0,1)])
    assert result["source_original_faces_with_any_physically_courted_edit_pixel"]==3
    assert result["source_face_count_with_two_different_original_edit_pixels_same_camera"]==1
    assert result["same_original_face_two_edits_one_camera_witness_count"]==1
    witness=result["same_original_face_two_independent_source_edits_one_camera_witnesses_PRIVATE"][0]
    assert witness["source_edit_0_physically_positive_pixel_xy"]==(10,10)
    assert witness["source_edit_1_physically_positive_pixel_xy"]==(11,10)
    assert witness["original_artist_layer_owner"]=="UNKNOWN"
    assert result["face_uniform_boolean_edit_handle_falsified_in_observed_source_scope"]
    assert result["source_edit_positive_pixels_by_operator"]==[2,1,1]
    assert result["source_local_edges_both_faces_physically_edit_observed"]==1
    assert result["source_local_edges_both_observed_axis_divergent"]==1
    assert result["source_local_edges_cross_scoped_relations_PRIVATE"][0][
        "source_mechanical_pose_vs_original_rest_edit_join"
    ]=="SAME_SOURCE_FACE_ID__DIFFERENT_POSE_SCOPE__NOT_PAIR_CAUSAL"


def test_different_views_do_not_fake_simultaneous_same_camera_witness():
    rows=[pixel(0,1,10,10,0),pixel(0,3,10,10,1)]
    result=build_cross_scope_source_profiles(rows, [], [0,1],[])
    assert result["source_face_count_with_two_different_original_edit_pixels_same_camera"]==0
    assert not result["face_uniform_boolean_edit_handle_falsified_in_observed_source_scope"]


def test_missing_face_edit_incidence_is_unknown_not_zero():
    result=build_cross_scope_source_profiles(
        [pixel(0,1,10,10,0)], [], [0,1,2], [])
    assert result["source_original_unwitnessed_face_count"]==2
    profile=result["observed_positive_edit_profiles_per_face_PRIVATE"][0]
    assert profile["observed_positive_source_edit_indices"]==[0]
    assert profile["unwitnessed_source_edit_control_negative_label"]=="UNKNOWN"
    assert result["source_mechanical_graphics_same_pose_causal_join"] == (
        "NOT_ADMITTED_DIFFERENT_POSE_SCOPES")


def test_duplicate_original_pixel_cannot_claim_two_face_owners():
    with pytest.raises(ValueError,match="PHYSICAL_PIXEL_MINTED_WITH_MULTIPLE_SOURCE_FACES"):
        build_cross_scope_source_profiles(
            [pixel(0,1,10,10,0),pixel(1,1,10,10,1)],
            [],[0,1],[])


def test_invalid_source_face_and_operation_rejected():
    with pytest.raises(ValueError,match="ORIGINAL_FACE_PIXEL_OR_EDIT_ADDRESS_INVALID"):
        build_cross_scope_source_profiles([pixel(3,1,10,10,0)],[],[0,1],[])
    with pytest.raises(ValueError,match="ORIGINAL_FACE_PIXEL_OR_EDIT_ADDRESS_INVALID"):
        build_cross_scope_source_profiles([pixel(0,1,10,10,3)],[],[0,1],[])


def test_local_edges_without_physical_witness_on_both_faces_are_not_negative():
    result=build_cross_scope_source_profiles(
        [pixel(0,1,10,10,0),pixel(2,1,11,10,1)],
        [local(0,1),local(0,2,axis=False)],
        [0,0,1],[])
    assert result["source_local_edges_both_faces_physically_edit_observed"]==1
    assert result["source_local_edges_exactly_one_face_physically_edit_observed"]==1
    assert result["source_local_edges_both_observed_no_common_positive_edit"]==1
    assert result["source_local_edges_cross_scoped_relations_PRIVATE"][0][
        "source_different_drawable"]=="UNKNOWN"


def test_island_near_and_uv_equality_only_cross_scoped_source_facts():
    result=build_cross_scope_source_profiles(
        [pixel(0,1,10,10,0),pixel(1,1,11,10,1)],
        [],[0,1],[lifted(0,1)])
    assert result["source_lifted_candidates_both_islands_physically_observed"]==1
    assert result["source_lifted_both_islands_observed_world_near"]==1
    assert result["source_lifted_both_islands_observed_world_uv_coincident"]==1
    row=result["source_lifted_candidates_cross_scope_PRIVATE"][0]
    assert row["join_scope"]=="SOURCE_ISLAND_ID_ONLY__POSE_CONTEXTS_DIFFER"
    assert row["source_same_artistic_drawable"]=="UNKNOWN"
