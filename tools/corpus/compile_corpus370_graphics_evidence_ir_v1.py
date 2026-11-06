#!/usr/bin/env python3
"""Generic operational Graphics Evidence IR — source-private observational v1.

This compiles *observations*, not artist layer labels or TRAINING TARGETS.
Three independent domains cannot be silently equated:
  - original 3D surface/ray geometry at pose and view (source teacher)
  - source 2D raster/appearance consequence under controlled rendering
  - source author's or user's operational editable component identity (UNKNOWN)

No fixed K, rig part ontology, semantic labels, or same/different drawable
training ground truth is derived from the V18a mechanical clustering proposal.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.measure_corpus370_v18j_source_subpixel_order_attribution import (
    SCHEMA as PARENT_RENDER, STATUS as PARENT_STATUS, STRATA, SAMPLE_COUNT,
)
from tools.corpus.court_corpus370_v18j_source_subpixel_order_attribution import (
    SCHEMA as PARENT_COURT, PASS as PARENT_PASS,
)

SCHEMA="RealSaS.Corpus370GenericOperationalGraphicsEvidenceIR.v1"
STATUS="SOURCE_GEOMETRIC_GRAPHICS_EVIDENCE_TYPED__ARTIST_AND_OPERATIONAL_OWNERSHIP_UNKNOWN"
KNOWN_KINDS=(
    "SOURCE_RAY_GEOMETRIC_ORDER",
    "GEOMETRIC_SUBPIXEL_COVERAGE",
    "CONTROLLED_RGBA_COMPOSITING_RESIDUAL_CLASS",
    "RENDER_OBSERVABILITY",
)
UNKNOWN_PART="UNKNOWN"
NO_LINK="NO_GRAPHIC_OWNER_RELATION_ADMITTED"


def require(value,reason):
    if not value:raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def project_ray_observation(row):
    require(row.get("retrospective_pixel_stratum") in STRATA
        and row.get("outcome_selected_not_blind_training") is True
        and row.get("source_camera_view") in (0,2,4,6)
        and row.get("group_source_face_count",0)>0
        and row.get("original_full_source_group_count",0)>0,
        "GRAPHICS_IR_ORIGINAL_VIEW_SOURCE_STRATUM_PROVENANCE")
    ray=row["source_ray_evidence_private"]
    samples=ray["geometric_subsamples"]
    require(len(samples)==SAMPLE_COUNT
        and ray["qualified_subsample_count"]+
            ray["unknown_incomplete_or_tied_subsamples"]==SAMPLE_COUNT
        and ray["source_geometry_pixel_y_top_down"]+
            ray["source_framebuffer_pixel_y_bottom_up"]==191
        and 0<=ray["source_framebuffer_pixel_x"]<192
        and 0<=ray["source_framebuffer_pixel_y_bottom_up"]<192,
        "GRAPHICS_IR_PIXEL_GEOMETRY_COORDINATE_CONTRACT")
    for sample in samples:
        require(sample["qualified_for_geometric_coverage"] is
                bool(sample["stack_complete"] and not sample["has_depth_tie"])
            and sample["geometric_hit_count"]==
                len(sample["depth_sorted_source_hits_private"])
            and (sample["first_visible_geometric_side"] in
                ("G","R","NONE") if sample["qualified_for_geometric_coverage"]
                else sample["first_visible_geometric_side"]=="UNKNOWN"),
            "GRAPHICS_IR_ORIGINAL_SOURCE_RAY_VALIDITY_MISREPRESENTED")
        depth=[hit["world_ray_depth"]
               for hit in sample["depth_sorted_source_hits_private"]]
        require(all(type(hit["source_face_teacher_only"]) is int
                    and hit["source_face_teacher_only"]>=0
                    and hit["anonymized_candidate_side"] in ("G","R")
                    for hit in sample["depth_sorted_source_hits_private"])
                and all(a<=b for a,b in zip(depth,depth[1:])),
                "GRAPHICS_IR_SOURCE_RAY_DEPTH_ORDER_FRAUD")
    all_qualified=ray["unknown_incomplete_or_tied_subsamples"]==0
    require((ray["fully_qualified_4x4_coverage_fractions"] is not None)
            == all_qualified,"GRAPHICS_IR_UNKNOWN_COVERAGE_FORCED_POSITIVE")
    near_group=ray["qualified_first_group_subsamples"]
    near_remain=ray["qualified_first_remainder_subsamples"]
    conflict=bool(ray["opposite_nearest_source_sides_across_subpixels"])
    require(conflict==(near_group>0 and near_remain>0)
            and near_group+near_remain+
                ray["qualified_nohit_subsamples"]==
                    ray["qualified_subsample_count"],
            "GRAPHICS_IR_RAY_FIRST_HIT_SIDE_COUNTS_INCONSISTENT")
    if not all_qualified:
        geometry_status="PARTIAL_GEOMETRIC_RAYS__COVERAGE_UNKNOWN"
    elif conflict:
        geometry_status="MIXED_GEOMETRIC_NEAREST_SIDES_ACROSS_SUBPIXELS"
    elif near_group+near_remain>0:
        geometry_status="CONSISTENT_OR_SINGLE_OBSERVED_NEAREST_SIDE"
    else:
        geometry_status="QUALIFIED_4X4_WITH_NO_GEOMETRIC_FIRST_HIT"
    return {
        "source_camera_view":int(row["source_camera_view"]),
        "original_framebuffer_pixel_bottom_up":[
            ray["source_framebuffer_pixel_x"],
            ray["source_framebuffer_pixel_y_bottom_up"]],
        "source_oracle_pixel_top_down":[
            ray["source_framebuffer_pixel_x"],
            ray["source_geometry_pixel_y_top_down"]],
        "source_pose_ordinal":2,
        "proposal_lineage":{
            "frozen_v18a_reactivity_threshold":float(row["frozen_threshold"]),
            "anonymous_component_proposal_id_teacher_only":
                int(row["source_group_index_teacher_only"]),
            "source_group_face_count":int(row["group_source_face_count"]),
            "source_proposal_group_count":
                int(row["original_full_source_group_count"]),
            "proposal_is_not_artist_component":True,
        },
        "observational_atoms":{
            "SOURCE_RAY_GEOMETRIC_ORDER":{
                "status":geometry_status,
                "frontmost_source_face_side_group_count":near_group,
                "frontmost_source_face_side_remainder_count":near_remain,
                "geometrically_empty_subsamples":ray["qualified_nohit_subsamples"],
                "opposing_nearest_sides_across_subsamples":conflict,
                "source_rays_with_three_or_more_depth_side_runs":
                    ray["qualified_with_three_or_more_depth_side_runs"],
                "raw_source_face_depth_stacks_teacher_only":samples,
                "visibility_relation_is_local_to_original_pose_camera_sample":True,
            },
            "GEOMETRIC_SUBPIXEL_COVERAGE":{
                "qualified_samples":ray["qualified_subsample_count"],
                "unknown_samples":ray["unknown_incomplete_or_tied_subsamples"],
                "fixed_source_sample_count":SAMPLE_COUNT,
                "geometrical_4x4_coverage_fraction_diagnostic":
                    ray["fully_qualified_4x4_coverage_fractions"],
                "not_exact_cycles_stochastic_sampling":True,
            },
            "CONTROLLED_RGBA_COMPOSITING_RESIDUAL_CLASS":{
                "retrospective_source_v18i_class":row["retrospective_pixel_stratum"],
                "source_class_population_in_view_threshold":
                    int(row["complete_stratum_population"]),
                "source_class_is_selected_using_observed_image":True,
                "alpha_or_rgb_class_does_not_imply_layer_boundary":True,
            },
            "RENDER_OBSERVABILITY":{
                "source_geometric_4x4_sampling":
                    "ALL_QUALIFIED" if all_qualified else "PARTIAL_UNKNOWN",
                "actual_source_art_alpha_from_geometric_hits":"UNKNOWN",
                "source_artist_editable_drawable":"UNKNOWN",
                "true_Cycles_filter_kernel":"UNKNOWN",
            },
        },
        "operational_part_affinity":"UNKNOWN",
        "operational_part_repulsion":"UNKNOWN",
        "source_artist_part_identity":"UNKNOWN",
        "graphics_compiler_draw_order":"UNKNOWN",
        "cannot_link_minted":False,
        "must_link_minted":False,
        "student_model_target_admitted":False,
    }


def compile_ir(proposal_file,independent_source_court,
               expected_independent_source_court_sha):
    require(isinstance(expected_independent_source_court_sha,str)
            and len(expected_independent_source_court_sha)==64
            and sha(independent_source_court)==
                expected_independent_source_court_sha,
            "GRAPHICS_IR_EXACT_INDEPENDENT_SOURCE_COURT_SHA_REQUIRED")
    proposal=load(proposal_file)
    court=load(independent_source_court)
    require(proposal.get("schema")==PARENT_RENDER
        and proposal.get("status")==PARENT_STATUS
        and court.get("schema")==PARENT_COURT
        and court.get("status")==PARENT_PASS
        and court.get("producer_sha256")==sha(proposal_file)
        and court.get("original_source_sha256")==proposal.get("original_source_sha256")
        and court.get("original_source_subject_id_private")==
            proposal.get("original_source_subject_private")
        and court.get("complete_private_source_ray_reread") is True
        and court.get("image_error_pixel_selection_is_retrospective_not_heldout") is True
        and court.get("unknown_is_not_a_negative_edge") is True
        and proposal.get("source_pixel_outcome_selected") is True
        and proposal.get("fixed_grid_is_not_blender_cycles_sample_pattern") is True
        and proposal.get("sample_geometry_is_not_drawable_ownership") is True
        and all(obj.get(k) is False for obj in (proposal,court) for k in (
           "graphics_owner_authorized","iris_training_authorized",
           "tessa_training_authorized","materialization_authorized",
           "training_authorized")),
        "GRAPHICS_IR_SOURCE_SHA_BINDING_OR_NONPROMOTION")
    observations=[project_ray_observation(z)
                  for z in proposal["source_private_pixel_and_ray_rows"]]
    require(len(observations)*SAMPLE_COUNT==proposal["source_rays_issued"],
            "GRAPHICS_IR_GEOMETRIC_RAY_CENSUS")
    distinct=set()
    for x in observations:
        key=(x["source_camera_view"],
             x["proposal_lineage"]["frozen_v18a_reactivity_threshold"],
             tuple(x["original_framebuffer_pixel_bottom_up"]))
        require(key not in distinct,"GRAPHICS_IR_DUPLICATE_SAMPLED_SOURCE_PIXEL")
        distinct.add(key)
    typed_counts={}
    for kind in STRATA:
        chosen=[x for x in observations if x["observational_atoms"]
               ["CONTROLLED_RGBA_COMPOSITING_RESIDUAL_CLASS"]
               ["retrospective_source_v18i_class"]==kind]
        typed_counts[kind]={
            "original_source_pixels":len(chosen),
            "source_geometrically_qualified_4x4_pixels":sum(
                x["observational_atoms"]["GEOMETRIC_SUBPIXEL_COVERAGE"]
                ["unknown_samples"]==0 for x in chosen),
            "opposing_nearest_source_side_pixels":sum(
                x["observational_atoms"]["SOURCE_RAY_GEOMETRIC_ORDER"]
                ["opposing_nearest_sides_across_subsamples"] for x in chosen),
            "pixels_with_geometrical_interleaving":sum(
                x["observational_atoms"]["SOURCE_RAY_GEOMETRIC_ORDER"]
                ["source_rays_with_three_or_more_depth_side_runs"]>0 for x in chosen),
            "proposed_same_drawable_positive_pairs":0,
            "proposed_different_drawable_negative_pairs":0,
        }
    return {
        "schema":SCHEMA,"status":STATUS,
        "source_subject_id_private":proposal["original_source_subject_private"],
        "original_source_sha256":proposal["original_source_sha256"],
        "parent_v18j_proposal_sha256":sha(proposal_file),
        "parent_v18j_independent_source_court_sha256":sha(independent_source_court),
        "evidence_atom_types":list(KNOWN_KINDS),
        "evidence_granularity":"SOURCE_POSE_VIEW_PIXEL_SUBSAMPLE",
        "observations_teacher_private":observations,
        "per_original_pixel_class_observation_counts":typed_counts,
        "original_2d_appearance_truth":"NOT_MEASURED_BY_GEOMETRIC_RAY_IR",
        "original_artist_layer_truth":"UNKNOWN",
        "consumer_editable_component_truth":"UNKNOWN",
        "no_anatomical_or_semantic_categories":True,
        "no_fixed_universal_drawable_count":True,
        "source_ray_order_is_not_global_drawable_order":True,
        "no_graph_signed_edges_minted":True,
        "evidence_is_retrospective_not_blind_heldout":True,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for key in ("proposal","v18j_court","output"):
        p.add_argument("--"+key.replace("_","-"),required=True,type=Path)
    p.add_argument("--expected-v18j-court-sha",required=True)
    a=p.parse_args()
    out=compile_ir(a.proposal,a.v18j_court,a.expected_v18j_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":out["status"],
        "source_pixel_classes":out["per_original_pixel_class_observation_counts"],
        "no_graph_signed_edges_minted":True,
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
