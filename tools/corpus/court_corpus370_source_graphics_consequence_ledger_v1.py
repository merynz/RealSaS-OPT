#!/usr/bin/env python3
"""Independent source receipt + typed ledger invariant court.

Does NOT import ledger compiler or its field projection. Rechecks original
source counts, action/view conservation, proof scope, unknown/nonpromotion.
Does NOT re-render source Cycles NPZ (parent independent K3 court did that).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

SCHEMA="RealSaS.Corpus370IndependentSourceGraphicsConsequenceLedgerCourt.v1"
PASS="PASS_SOURCE_CONSEQUENCE_LEDGER_ACCOUNTING__FINAL_AUTHORITY_PENDING"
PARENT="RealSaS.Corpus370SourceGraphicsEditConsequenceLedger.v1"
TARGET_FAMILIES={
    "APPEARANCE_CAA":"SOURCE_CONSTANT_SHADER_PALETTE_FACE_OBSERVATION",
    "VISIBILITY_OCCLUSION":"SOURCE_SAMPLE_MULTI_GROUP_GEOMETRIC_OVERLAP",
    "RAY_COMPOSITING":"SOURCE_ORIGINAL_K_DEPTH_ORDERED_RGBA_RECOMPOSITION",
    "CAUSAL_RESPONSE":"SOURCE_RGB_PALETTE_INTERVENTION_RENDER_CONSEQUENCE",
}
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","materialization_authorized","training_authorized")

def assert_true(x,label):
    if not x: raise ValueError("INDEPENDENT_GRAPHICS_LEDGER_"+label)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def verify(ir, palette, render, source_court, native_court, contract, *, sha_index):
    assert_true(ir.get("schema")==PARENT and
        ir.get("status")=="SOURCE_GRAPHICS_CONSEQUENCE_COURTED__FINAL_FAMILY_AUTHORITY_PENDING",
        "SCHEMA_OR_STATUS")
    assert_true(all(ir.get(x) is False for x in FLAGS) and
        ir.get("source_artist_drawable_owner")=="UNKNOWN" and
        ir.get("source_graphics_compiler_draw_order_global")=="UNKNOWN" and
        ir.get("same_drawable_training_pairs_qualified")==0 and
        ir.get("different_drawable_training_pairs_qualified")==0 and
        ir.get("student_model_target_count")==0 and
        ir.get("source_pose2_tested") is False and
        ir.get("nonbinary_alpha_tested") is False and
        ir.get("original_source_teacher_private") is True,
        "FALSE_OWNER_OR_TRAINING_AUTHORITY")
    expected={k:v for k,v in sha_index.items() if k!="ledger_sha256"}
    assert_true(ir.get("parent_receipts")==expected and
        sha_index["original_source_sha256"]==
        ir["original_source_sha256"]==palette["original_source_sha256"]==
        render["original_source_sha256"]==source_court["original_source_sha256"]==
        native_court["original_source_sha256"] and
        native_court["subject_id_private"]==ir["subject_id"],
        "SHA_AND_SUBJECT_CROSS_PARENT")
    assert_true(source_court["status"]==
       "PASS_ORIGINAL_PHYSICAL_K3_SAME_CYCLES_SAMPLE_EDIT_COURT__OWNER_UNKNOWN"
       and source_court.get("failure_reasons")==[]
       and source_court["original_source_renderer_manifest_sha256"]==
          sha_index["renderer_manifest_sha256"]
       and source_court["original_source_renderer_rgba_npz_sha256"]==
          render["source_original_edited_sample_npz_sha256"]
       and source_court["original_source_palette_facts_sha256"]==
          sha_index["palette_facts_sha256"]
       and source_court["producer_court_source_full_not_used_to_choose_depth"] is True,
       "INDEPENDENT_ORIGINAL_SOURCE_COURT_NOT_BOUND")
    assert_true(all(t.get(x) is False for t in (palette,render,source_court,native_court)
                    for x in FLAGS) and
       contract["schema"]=="RealSaS.CorpusTruthAuthorityContract.v1"
       and set(TARGET_FAMILIES).issubset(set(contract["family_order"])),
       "PARENT_AUTHORITY_OR_FAMILY_DRIFT")
    keys=palette["face_palette_color_key_index_or_null"]
    n=palette["source_face_count"]
    colors=palette["constant_palette_color_keys_hex_rgba"]
    k=palette["palette_operation_count"]
    assert_true(len(keys)==n and len(colors)==k and
        all(type(a) is int and 0<=a<k for a in keys) and
        all(palette["face_palette_validity"]) and
        palette["qualified_constant_face_count"]==n and
        palette["unsupported_face_count"]==0 and
        [sum(a==i for a in keys) for i in range(k)]==
        render["source_original_triangles_per_color"]==
        ir["original_source_palette_face_membership"]["source_triangles_per_palette_color"] and
        ir["original_source_palette_face_membership"]["qualified_source_faces"]==n and
        ir["original_source_palette_face_membership"]["unknown_source_faces"]==0 and
        ir["original_source_palette_face_membership"][
            "sample_local_color_groups_not_persistent_art_drawables"] is True and
        ir["original_source_group_count_K"]==k and
        ir["palette_group_kind"]=="EXACT_SOURCE_CONSTANT_COLOR_EQUIVALENCE_NOT_ART_PART",
        "SOURCE_PALETTE_FACE_CENSUS")
    offsets=render["source_cycle_sample_indices"]
    resolution=render["source_cycle_resolution"]
    views=render["source_cycle_camera_view_indices"]
    rows=source_court["source_per_view_results"]
    ir_rows=ir["original_source_view_edit_consequence"]
    assert_true([t["source_view_index"] for t in rows]==views and
        [t["source_camera_view"] for t in ir_rows]==views and
        len(rows)==len(ir_rows)>0 and len(offsets)>0,
        "ORIGINAL_VIEW_AND_CYCLES_INDICES")
    sumq=sumu=sumover=0
    changed=[0]*k
    for court_row,proposed in zip(rows,ir_rows):
        metrics=court_row["metric_rows"]
        assert_true(len(metrics)==k+1 and
            [a["arm"] for a in metrics]==["SOURCE_FULL"]+[
                f"SOURCE_EDITED_FULL_{i}" for i in range(k)] and
            proposed["source_pose_scope"]=="ORIGINAL_REST_ONLY" and
            proposed["sample_index_policy"]=="SAME_CYCLES_SOURCE_SAMPLE_INDICES" and
            proposed["original_cycles_sample_indices"]==offsets and
            proposed["spatial_granularity"]==
                "VIEW_AGGREGATE__NOT_PER_PIXEL_OWNER_OR_PER_RAY_MASK",
            "NO_FAKE_SAMPLE_OR_POSE_LOCALITY")
        base=metrics[0]
        qualified=base["qualified_source_sample_pixels"]
        unknown=base["unknown_source_samples"]
        overlap=court_row["source_multi_group_overlapping_sample_count"]
        assert_true(type(qualified) is int and type(unknown) is int and
            type(overlap) is int and
            qualified+unknown==len(offsets)*resolution*resolution and
            0<=overlap<=qualified and
            court_row["source_unknown_depth_tie_sample_count"]==unknown and
            court_row["source_invalid_depth_sample_count"]==0 and
            court_row["source_isolate_nonbinary_alpha_count"]==0 and
            court_row["source_full_alpha_intermediate_single_source_sample_count"]==0,
            "ORIGINAL_SOURCE_COVERAGE_OR_ALPHA")
        depth=proposed["source_depth_and_visibility"]
        comp=proposed["source_rgba_recomposition"]
        assert_true(depth=={
            "qualified_sample_count":qualified,
            "unknown_sample_count":unknown,
            "multi_group_source_geometry_overlap_samples":overlap,
            "individual_source_depth_stacks_in_this_ledger":"NOT_INCLUDED",
        } and
        comp=={
            "qualified_sample_count":qualified,
            "mismatch_at_original_court_tolerance":0,
            "sample_mean_mismatch_at_original_court_tolerance":0,
            "tolerance_source_parent":"INDEPENDENT_K3_COURT__0P025_MAX_PREMULTIPLIED_RGBA",
        },"SOURCE_COURTED_VIEW_PROJECTION")
        effects=proposed["source_palette_edits"]
        assert_true(len(effects)==k,"ONE_EDIT_PER_SOURCE_PALETTE")
        for i,metric in enumerate(metrics):
            assert_true(metric["qualified_source_sample_pixels"]==qualified and
                metric["unknown_source_samples"]==unknown and
                metric["sample_rgba_mismatch_count"]==0 and
                metric["eight_sample_mean_rgba_mismatch_count"]==0 and
                (i>0 or metric["physical_full_edit_changed_sample_pixel_count"]==0),
                "SOURCE_PHYSICAL_EDIT_SOURCE_SAMPLES_NOT_PASS")
            if i>0:
                changed[i-1]+=metric["physical_full_edit_changed_sample_pixel_count"]
                assert_true(effects[i-1]=={
                    "source_palette_local_index":i-1,
                    "source_color_hex_rgba":colors[i-1],
                    "observed_changed_source_sample_pixels":
                        metric["physical_full_edit_changed_sample_pixel_count"],
                    "full_edit_reconstruction_mismatches_at_court_tolerance":0,
                    "relation_to_art_drawable_owner":"UNKNOWN",
                },"SOURCE_EDIT_CONSEQUENCE_TAMPER")
        sumq+=qualified;sumu+=unknown;sumover+=overlap
    assert_true(sumq==source_court["qualified_sample_total"] and
        sumover==source_court["original_physical_overlapping_source_sample_count"] and
        ir["original_source_sample_totals"]=={
            "qualified":sumq,"unknown":sumu,"overlap":sumover,
            "independent_color_operations":k,
            "per_color_changed_sample_incidents":changed,
        } and all(x>=8 for x in changed),"ORIGINAL_SOURCE_GLOBAL_CONSERVATION")
    matrix=ir["family_matrix_bridge"]
    expected_counts={"APPEARANCE_CAA":n,"VISIBILITY_OCCLUSION":sumover,
                     "RAY_COMPOSITING":sumq,"CAUSAL_RESPONSE":sum(changed)}
    assert_true(set(matrix)==set(TARGET_FAMILIES),"FINAL_CONTRACT_FAMILY_SET")
    for family,kind in TARGET_FAMILIES.items():
        assert_true(matrix[family]=={
            "candidate_source_observation":kind,
            "source_qualified_observation_count":expected_counts[family],
            "final_family_authority_state":"PENDING",
            "final_training_validity_mask_sha256":None,
            "student_access_to_original_teacher_source":False,
            "training_target_minted":False,
        },"FAMILY_AUTHORITY_UNLAWFUL_PROMOTION_"+family)
    expected_unknown={
        key:"UNKNOWN_NOT_EVALUATED_BY_REST_POSE_CONSTANT_COLOR_COURT"
        for key in (
            "SOURCE_ARTIST_DRAWABLE_OWNERSHIP",
            "OPERATIONAL_SAME_DIFFERENT_DRAWABLE_EDGES",
            "ATTACHMENT_AND_DETACHMENT",
            "SECONDARY_MOTION_AND_SKIN_DEFORMATION",
            "POSE_PERSISTENT_GRAPHICS_RELATIONS",
            "TEXTURE_UV_LOCAL_RECOLOR",
            "NONBINARY_ALPHA_AND_SHADOW_INTERACTION",
            "ONE_VIEW_STUDENT_INFERENCE",
        )}
    assert_true(ir.get("unmeasured_or_unqualified_domains")==expected_unknown,
                "UNKNOWN_GRAPHICS_DOMAIN_RECLASSIFIED_AS_GOLD")
    return {
        "schema":SCHEMA,"status":PASS,
        "subject_id":ir["subject_id"],"original_source_sha256":ir["original_source_sha256"],
        "source_qualified_sample_incidents":sumq,
        "source_unknown_sample_incidents":sumu,
        "source_overlapping_geometry_sample_incidents":sumover,
        "source_independently_courted_palette_operations":k,
        "source_edit_changed_sample_incidents_by_color":changed,
        "family_matrix_proposals_checked":list(TARGET_FAMILIES),
        "source_court_parent_sha256":sha_index["independent_render_court_sha256"],
        "source_ledger_sha256":sha_index["ledger_sha256"],
        "all_source_family_authority_pending":True,
        "artist_drawable_owner":"UNKNOWN",
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for name in ("ledger","palette","render-manifest","independent-court",
                 "native-court","truth-contract","output"):
        a.add_argument("--"+name,type=Path,required=True)
    opts=a.parse_args()
    names={"palette_facts_sha256":opts.palette,"renderer_manifest_sha256":opts.render_manifest,
           "independent_render_court_sha256":opts.independent_court,
           "independent_native_support_court_sha256":opts.native_court,
           "truth_authority_contract_sha256":opts.truth_contract}
    receipt={k:sha(v) for k,v in names.items()}
    evidence=read(opts.ledger)
    receipt["ledger_sha256"]=sha(opts.ledger)
    receipt["original_source_sha256"]=evidence.get("original_source_sha256")
    receipt["source_sample_npz_sha256_teacher_only"]=read(
        opts.render_manifest)["source_original_edited_sample_npz_sha256"]
    receipt["original_rest_geometry_npz_sha256_teacher_only"]=read(
        opts.render_manifest)["source_original_rest_geometry_sha256"]
    receipt["original_camera_bundle_sha256_teacher_only"]=read(
        opts.render_manifest)["source_original_camera_bundle_sha256"]
    receipt["original_render_policy_sha256_teacher_only"]=read(
        opts.render_manifest)["source_original_render_policy_sha256"]
    result=verify(evidence, read(opts.palette), read(opts.render_manifest),
                  read(opts.independent_court),read(opts.native_court),
                  read(opts.truth_contract),sha_index=receipt)
    opts.output.parent.mkdir(parents=True,exist_ok=True)
    opts.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print(json.dumps({"status":result["status"],"family_proposals":len(TARGET_FAMILIES),
                      "training_authorized":False},sort_keys=True))

if __name__=="__main__":main()
