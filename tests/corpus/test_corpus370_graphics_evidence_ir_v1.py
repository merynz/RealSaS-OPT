"""Graphics Evidence IR must preserve geometry vs drawable-truth boundary."""
import copy
import pytest

from tools.corpus.compile_corpus370_graphics_evidence_ir_v1 import (
    SCHEMA,STATUS,KNOWN_KINDS,project_ray_observation,
)


def sample(side="G",qualified=True):
    return {
        "offset_x":-.375,"offset_y":-.375,
        "stack_complete":bool(qualified),"has_depth_tie":False,
        "geometric_hit_count":1,
        "qualified_for_geometric_coverage":bool(qualified),
        "depth_sorted_source_hits_private":[{
            "source_face_teacher_only":1,
            "anonymized_candidate_side":side,"world_ray_depth":1.5
        }],
        "first_visible_geometric_side":side if qualified else "UNKNOWN",
        "group_present_any_depth":side=="G" if qualified else None,
        "remainder_present_any_depth":side=="R" if qualified else None,
        "depth_side_switch_count":0 if qualified else None,
    }


def record(samples=None):
    if samples is None:samples=[sample() for _ in range(16)]
    valid=[x for x in samples if x["qualified_for_geometric_coverage"]]
    ng=sum(x["first_visible_geometric_side"]=="G" for x in valid)
    nr=sum(x["first_visible_geometric_side"]=="R" for x in valid)
    return {
        "source_camera_view":0,"frozen_threshold":.1,
        "source_group_index_teacher_only":4,
        "group_source_face_count":2,
        "original_full_source_group_count":9,
        "retrospective_pixel_stratum":"RGB_ONLY_FAIL",
        "complete_stratum_population":30,
        "outcome_selected_not_blind_training":True,
        "source_ray_evidence_private":{
            "source_framebuffer_pixel_x":20,
            "source_framebuffer_pixel_y_bottom_up":80,
            "source_geometry_pixel_y_top_down":111,
            "geometric_subsamples":samples,
            "qualified_subsample_count":len(valid),
            "unknown_incomplete_or_tied_subsamples":16-len(valid),
            "qualified_first_group_subsamples":ng,
            "qualified_first_remainder_subsamples":nr,
            "qualified_nohit_subsamples":0,
            "qualified_with_both_sides_in_depth_stack":0,
            "qualified_with_three_or_more_depth_side_runs":0,
            "opposite_nearest_source_sides_across_subpixels":bool(ng and nr),
            "fully_qualified_4x4_coverage_fractions":{
                "geometric_group_subsample_fraction":ng/16,
                "geometric_remainder_subsample_fraction":nr/16,
                "geometric_joint_any_subsample_fraction":1.,
                "geometric_postfiltered_over_alpha":1.,
                "joint_minus_postfiltered_alpha":0.,
            } if len(valid)==16 else None,
            "source_graphic_part_identity":"UNKNOWN",
        }
    }


def test_typed_geometric_order_is_not_graphic_owner_relation():
    r=record([sample("G") for _ in range(8)]+
             [sample("R") for _ in range(8)])
    x=project_ray_observation(r)
    atom=x["observational_atoms"]["SOURCE_RAY_GEOMETRIC_ORDER"]
    assert atom["status"]=="MIXED_GEOMETRIC_NEAREST_SIDES_ACROSS_SUBPIXELS"
    assert atom["opposing_nearest_sides_across_subsamples"]
    assert x["operational_part_affinity"]=="UNKNOWN"
    assert x["operational_part_repulsion"]=="UNKNOWN"
    assert x["source_artist_part_identity"]=="UNKNOWN"
    assert x["graphics_compiler_draw_order"]=="UNKNOWN"
    assert not x["cannot_link_minted"] and not x["must_link_minted"]
    assert not x["student_model_target_admitted"]


def test_ambiguous_ray_yields_unknown_not_absent():
    samples=[sample("G") for _ in range(15)]+[sample("R",False)]
    out=project_ray_observation(record(samples))
    assert out["observational_atoms"]["RENDER_OBSERVABILITY"][
      "source_geometric_4x4_sampling"]=="PARTIAL_UNKNOWN"
    assert out["observational_atoms"]["GEOMETRIC_SUBPIXEL_COVERAGE"][
      "geometrical_4x4_coverage_fraction_diagnostic"] is None
    assert out["operational_part_repulsion"]=="UNKNOWN"


def test_no_artist_ownership_from_frozen_anonymous_group_count():
    out=project_ray_observation(record())
    assert out["proposal_lineage"]["source_proposal_group_count"]==9
    assert out["proposal_lineage"]["proposal_is_not_artist_component"] is True
    assert out["observational_atoms"]["CONTROLLED_RGBA_COMPOSITING_RESIDUAL_CLASS"][
        "alpha_or_rgb_class_does_not_imply_layer_boundary"] is True
    assert SCHEMA.endswith(".v1")
    assert "OWNERSHIP_UNKNOWN" in STATUS
    assert "SOURCE_RAY_GEOMETRIC_ORDER" in KNOWN_KINDS


@pytest.mark.parametrize("tamper",[
    "flip_framebuffer_y","forged_qualified_unknown","change_first_side",
    "negative_face","misorder_depth","positive_drawable",
])
def test_corrupt_source_ray_contracts_fail_or_stay_unknown(tamper):
    r=record()
    if tamper=="flip_framebuffer_y":
        r["source_ray_evidence_private"]["source_geometry_pixel_y_top_down"]=80
    elif tamper=="forged_qualified_unknown":
        r["source_ray_evidence_private"]["geometric_subsamples"][0][
            "qualified_for_geometric_coverage"]=False
    elif tamper=="change_first_side":
        r["source_ray_evidence_private"]["qualified_first_remainder_subsamples"]=4
    elif tamper=="negative_face":
        r["source_ray_evidence_private"]["geometric_subsamples"][0][
            "depth_sorted_source_hits_private"][0]["source_face_teacher_only"]=-1
    elif tamper=="misorder_depth":
        s=r["source_ray_evidence_private"]["geometric_subsamples"][0]
        s["geometric_hit_count"]=2
        s["depth_sorted_source_hits_private"].append({
            "source_face_teacher_only":2,"anonymized_candidate_side":"R",
            "world_ray_depth":.4
        })
    elif tamper=="positive_drawable":
        r["retrospective_pixel_stratum"]="DRAWABLE_GOLD"
    with pytest.raises(ValueError,match="GRAPHICS_IR_"):
        project_ray_observation(r)
