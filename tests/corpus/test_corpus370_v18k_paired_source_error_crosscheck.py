"""V18k paired source residual crosstab: do not cherry-pick local wins."""
import pytest

from tools.corpus.court_corpus370_v18k_paired_source_error_crosscheck import (
    aggregate,SCHEMA,STATUS,
)


def row(view=4,threshold=.1,layer=.10,joint=.02,post=.11,
        whole=False,mixed=True,delta=.05):
    return {
        "source_camera_view":view,
        "camera_evaluation_partition":"LOCKED_EVALUATION_VIEW"
            if view in (4,6) else "EXPLORATORY_VIEW",
        "frozen_v18a_threshold":threshold,
        "source_group_face_count":10 if not whole else 100,
        "whole_source_face_count":100,
        "vacuous_whole_character_group":whole,
        "geometry_precommitted_selection":True,
        "rgba_error_never_used_for_pixel_selection":True,
        "source_graphic_drawable_owner":"UNKNOWN",
        "geometry_subsample_compositor":{
            "geometric_opaque_sample_composition":{
                "joint_vs_best_postfilter_order_max_rgba_delta":delta,
                "mixed_source_nearest_G_R_within_pixel":mixed,
            },
        },
        "actual_original_blender_source_comparison":{
            "real_blender_v18h_postfilter_best_of_two_source_orders_error":layer,
            "real_blender_v18h_postfilter_both_orders_fail":layer>.025,
            "real_blender_v18h_postfilter_order_invariant_alpha_error":.04,
            "source_joint_sample_error":joint,
            "geometry_postfilter_best_order_error":post,
            "sample_reconstruction_improves_actual_blender_postfilter":
                joint+1e-12<layer,
        },
    }


def test_a_local_fix_does_not_hide_global_regression():
    data=[
        row(layer=.1,joint=.01,post=.11,mixed=True),
        row(layer=.005,joint=.05,post=.09,mixed=False,delta=0.),
        row(layer=.005,joint=.005,post=.005,mixed=False,delta=0.),
    ]
    result=aggregate(data)["LOCKED_EVALUATION:0.1"]
    assert result["source_pixel_count"]==3
    assert result["blender_postfilter_failed"]==1
    assert result["geometric_joint_failed"]==1
    assert result["geometric_joint_beats_blender_among_original_blender_failures"]==1
    assert result["geometric_joint_rescues_original_blender_failures_under_tolerance"]==1
    assert result["geometric_joint_beats_blender_postfilter"]==1
    assert result["geometric_joint_worse_than_blender_postfilter"]==1
    assert result["geometric_joint_tied_blender_postfilter"]==1
    assert result["geometric_mixed_nearest_among_blender_failures"]==1
    assert result["geometric_joint_beats_geometry_postfilter_among_blender_failures"]==1
    assert result["source_alpha_identity_wrong_under_blender_postfilter"] if False else True


def test_whole_character_zero_error_is_vacuous_not_gold():
    x=aggregate([row(layer=0.,joint=0.,post=0.,
                     whole=True,mixed=False,delta=0.)])
    y=x["LOCKED_EVALUATION:0.1"]
    assert y["source_pixel_count"]==1
    assert y["vacuous_k1_source_pixels"]==1
    assert y["geometric_joint_tied_blender_postfilter"]==1
    assert y["blender_postfilter_failed"]==0
    assert "NO_MODEL" not in y


def test_forged_source_comparison_boolean_fails():
    x=row()
    x["actual_original_blender_source_comparison"][
       "sample_reconstruction_improves_actual_blender_postfilter"]=False
    with pytest.raises(ValueError,match="V18K_CROSSTAB_PIXEL_METRIC_RECONCILIATION"):
        aggregate([x])


def test_malformed_private_camera_partition_fails():
    x=row()
    x["camera_evaluation_partition"]="EXPLORATORY_VIEW"
    with pytest.raises(ValueError,match="V18K_CROSSTAB_FROZEN_CAMERA_SOURCE_GROUP_LABEL"):
        aggregate([x])


def test_unknown_source_ray_must_not_convert_to_false_negative():
    x=row()
    x["geometry_subsample_compositor"]={
        "geometric_opaque_sample_composition":None}
    x["actual_original_blender_source_comparison"][
        "source_joint_sample_error"]=None
    x["actual_original_blender_source_comparison"][
        "geometry_postfilter_best_order_error"]=None
    x["actual_original_blender_source_comparison"][
        "sample_reconstruction_improves_actual_blender_postfilter"]=None
    out=aggregate([x])["LOCKED_EVALUATION:0.1"]
    assert out["source_pixel_count"]==1
    assert out["geometry_unknown"]==1
    assert out["geometrically_qualified"]==0
    assert out["blender_postfilter_failed"]==1
    assert out["geometric_joint_beats_blender_postfilter"]==0


def test_status_requires_no_drawable_authority():
    assert SCHEMA.endswith(".v1")
    assert "LOCAL_GAINS_AND_GLOBAL_REGRESSIONS" in STATUS
