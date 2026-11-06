#!/usr/bin/env python3
"""Independent Cat original-source UV edit ledger admission/nonpromotion court.

Recomputes original-source per-view operation support and heldout result counts
from the SHA-courted original physical experiment. Does NOT import producer or
compilation implementation. Parent physical court, not this ledger, replayed
the 256 original Blender source Cycles sample arrays.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

SCHEMA="RealSaS.Corpus370IndependentCatSourceUVConsequenceLedgerCourt.v1"
PASS="PASS_CAT_SOURCE_UV_GRAPHICS_LEDGER_ACCOUNTING__TRUTH_AUTHORITY_PENDING"
IR="RealSaS.Corpus370SourceTextureUVEditConsequenceLedger.v1"
SOURCE_COURT="RealSaS.Corpus370IndependentOriginalCatSourceUVTextureEditCourt.v2"
OWNERS=("graphics_owner_authorized","iris_training_authorized",
        "tessa_training_authorized","materialization_authorized","training_authorized")
FAMILIES={
 "APPEARANCE_CAA":"SOURCE_LINKED_UV_ATLAS_FACE_ADDRESSABILITY_ONLY",
 "CAUSAL_RESPONSE":"SOURCE_PHYSICAL_RGB_TEXTURE_EDITS_AND_HELDOUT_JOINT_EFFECTS",
}
VIEWS=[1,3,5,7]
OFFSETS=list(range(0,32,4))
PAIRS=("0_1","0_2","1_2")

def req(ok,code):
    if not ok:raise ValueError("INDEPENDENT_CAT_UV_LEDGER_"+code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def verify(ledger,plan,assets,renderer,physical_court,canonical,*,parents):
    req(ledger.get("schema")==IR and
        ledger.get("status")=="SOURCE_CAT_LINKED_TEXTURE_EDIT_EVIDENCE_COURTED__FAMILY_AUTHORITY_PENDING"
        and physical_court.get("schema")==SOURCE_COURT
        and physical_court.get("status")==
            "PASS_ORIGINAL_CAT_LINKED_UV_EDIT_CONSEQUENCE_SUPERPOSITION__OWNER_UNKNOWN"
        and physical_court.get("failed_obligations")==[]
        and ledger.get("original_source_teacher_private") is True,
        "SCHEMA_OR_REAL_PHYSICAL_SOURCE_COURT")
    req(ledger.get("parent_receipts")=={k:v for k,v in parents.items()
                                          if k!="source_ledger_sha256"} and
        ledger.get("subject_id")==plan.get("subject_id") and
        ledger.get("original_source_sha256")==
            plan.get("source_original_glb_sha256")==
            renderer.get("original_source_sha256")==
            physical_court.get("source_original_glb_sha256")==
            parents["source_original_glb_sha256"] and
        ledger.get("original_texture_sha256")==
            plan.get("source_original_texture_sha256")==
            renderer.get("original_source_atlas_sha256")==
            physical_court.get("original_source_image_sha256")==
            parents["source_original_texture_sha256"] and
        renderer.get("original_source_uv_edit_plan_sha256")==
            physical_court.get("original_source_uv_edit_plan_sha256")==
            parents["source_uv_plan_sha256"] and
        physical_court.get("original_source_blender_render_manifest_sha256")==
            parents["original_render_manifest_sha256"] and
        physical_court.get("original_source_atlas_edit_assets_manifest_sha256")==
            parents["source_atlas_assets_sha256"] and
        physical_court.get("original_source_blender_sample_npz_sha256")==
            renderer.get("source_original_sample_npz_sha256")==
            parents["source_sample_npz_sha256_teacher_only"],
        "SOURCE_SHA_COURT_LINEAGE")
    req(all(z.get(name) is False for z in
            (ledger,plan,assets,renderer,physical_court) for name in OWNERS) and
        ledger.get("same_drawable_training_pairs_qualified")==0 and
        ledger.get("different_drawable_training_pairs_qualified")==0 and
        ledger.get("iris_student_training_targets")==0 and
        ledger.get("source_artist_drawable_owner")=="UNKNOWN" and
        ledger.get("source_uv_atlas_edits_are_art_drawables") is False and
        ledger.get("source_operation_domain")==
            "ORIGINAL_GLTF_UV_TEXTURE_RGB_EDIT__NO_DRAWABLE_IDENTITY" and
        ledger.get("source_pose2_graphics_relations")=="UNKNOWN_NOT_MEASURED" and
        ledger.get("source_visibility_occlusion_from_this_court")==
            "UNKNOWN_NOT_MEASURED" and
        ledger.get("source_ray_depth_compositing_from_this_court")==
            "UNKNOWN_NOT_MEASURED",
        "ILLEGAL_DRAWABLE_OR_TRAINING_AUTHORITY")
    req(canonical.get("schema")=="RealSaS.CorpusTruthAuthorityContract.v1"
        and canonical.get("authority_boundary",{}).get("this_contract_mints_training_readiness") is False
        and "PENDING" in canonical.get("family_authority_states",[])
        and set(FAMILIES).issubset(canonical.get("family_order",[]))
        and renderer.get("source_original_sample_npz_shape")==[4,8,8,192,192,4]
        and renderer.get("original_source_rest_only") is True
        and physical_court.get("source_pose_scope")=="ORIGINAL_CAT_REST_ONLY"
        and physical_court.get("source_sampler_tolerance_rgba_premultiplied_max")==.025
        and physical_court.get("source_unknown_alpha_drift")==0,
        "SOURCE_UNSUPPORTED_SCOPE_OR_CANONICAL_FAMILY")
    req(plan.get("source_uv_coordinate_origin")==
        "GLTF_2_IMAGE_TOP_LEFT__UV_V_INCREASES_DOWN" and
        plan.get("source_uv_to_png_y_flip_applied") is False and
        plan.get("source_glb_triangle_count")==684 and
        plan.get("selection_used_original_rendered_2d_image") is False,
        "SOURCE_GLTF_ORIGINAL_UV_DOMAIN")
    actions=plan.get("selected_original_source_uv_edits")
    names=("BASE","SINGLE_0","SINGLE_1","SINGLE_2",
           "PAIR_01","PAIR_02","PAIR_12","JOINT_012")
    asset_rows=assets.get("original_atlas_arm_rows")
    req(isinstance(actions,list) and len(actions)==3 and
        ledger.get("source_uv_edit_operations")==actions and
        isinstance(asset_rows,list) and
        [a.get("arm") for a in asset_rows]==list(names) and
        [a.get("original_atlas_edited_rgb_pixel_count") for a in asset_rows]==
            [0,4096,4096,4096,8192,8192,8192,12288] and
        all(a.get("atlas_pixel_edit_equals_original_declared_source_mask") is True
            and a.get("source_texture_alpha_preserved_exactly") is True
            for a in asset_rows),
        "ORIGINAL_SOURCE_TEXTURE_OPERATION_CENSUS")
    original=physical_court.get("independent_court_per_camera")
    projected=ledger.get("source_graphics_effect_per_view")
    req(isinstance(original,list) and isinstance(projected,list) and
        [z.get("original_source_view") for z in original]==VIEWS and
        [z.get("source_view") for z in projected]==VIEWS,
        "ALL_SOURCE_VIEWS_EXACT")
    by_action=[0,0,0];overlaps={k:0 for k in PAIRS}
    for evidence,row in zip(original,projected):
        counts=evidence.get("edited_single_source_visible_sample_count")
        pair=evidence.get("pairwise_single_edit_image_support_overlaps")
        checks=evidence.get("source_heldout_pair_joint_court")
        req(isinstance(counts,list) and len(counts)==3 and
            all(type(v) is int and 0<=v<=8*192*192 for v in counts) and
            isinstance(pair,dict) and set(pair)==set(PAIRS) and
            all(type(v) is int and v>=0 for v in pair.values()) and
            isinstance(checks,list) and len(checks)==4 and
            evidence.get("all_source_arm_alpha_max_drift")==0 and
            row.get("original_source_pose")=="BIND_REST_ONLY" and
            row.get("sample_offsets")==OFFSETS and
            row.get("effect_measurement_granularity")==
                "SOURCE_SAMPLED_IMAGE_AGGREGATE_NOT_ART_DRAWABLE" and
            row.get("source_individual_uv_edit_sample_incidents")==counts and
            row.get("source_pairwise_visible_effect_support_overlap")==pair and
            row.get("source_edit_alpha_drift")==0 and
            row.get("heldout_pair_joint_sample_mismatch_count")==0 and
            row.get("heldout_pair_joint_eight_sample_mean_mismatch_count")==0 and
            row.get("source_drawable_owner")=="UNKNOWN",
            "SOURCE_CAM_EDIT_EFFECT_PROJECTION")
        for check,name,inputs in zip(checks,
            ("PAIR_01","PAIR_02","PAIR_12","JOINT_012"),
            ([0,1],[0,2],[1,2],[0,1,2])):
            req(check.get("actual_heldout_full_source_arm")==name and
                check.get("source_single_edited_arm_input_indices")==inputs and
                check.get("source_per_sample_rgba_mismatch_count")==0 and
                check.get("source_eight_sample_average_rgba_mismatch_count")==0 and
                check.get("prediction_uses_heldout_pair_or_joint_target") is False and
                0<=check.get("source_per_sample_rgba_max_delta",99)<=.025,
                "ORIGINAL_SOURCE_PHYSICAL_PAIR_JOINT_NOT_COURTED")
        for i,v in enumerate(counts):by_action[i]+=v
        for k in PAIRS:overlaps[k]+=pair[k]
    req(by_action==physical_court.get("source_single_edit_observable_samples_per_operation")
        and all(x>=32 for x in by_action) and overlaps["0_1"]>0
        and ledger.get("source_edit_consequence_totals")=={
            "single_edit_sample_incidents_by_operation":by_action,
            "pairwise_visible_sample_support_overlap":overlaps,
            "heldout_source_pair_joint_sample_mismatches":0,
            "heldout_source_pair_joint_eight_sample_mean_mismatches":0,
            "source_alpha_drift":0,
        },"SOURCE_OPERATION_COUNTS_OR_NONVACUITY")
    matrix=ledger.get("family_matrix_bridge")
    req(isinstance(matrix,dict) and set(matrix)==set(FAMILIES),
        "CANDIDATE_FAMILY_SCOPE")
    for family,kind in FAMILIES.items():
        count=684 if family=="APPEARANCE_CAA" else sum(by_action)
        req(matrix[family]=={
            "candidate_source_observation":kind,
            "source_qualified_observation_count":count,
            "final_family_authority_state":"PENDING",
            "final_training_validity_mask_sha256":None,
            "student_access_to_original_teacher_source":False,
            "training_target_minted":False,
        },"CANDIDATE_FAMILY_ILLEGAL_PROMOTION_"+family)
    return {
        "schema":SCHEMA,"status":PASS,"original_source_sha256":ledger["original_source_sha256"],
        "subject_id":ledger["subject_id"],
        "source_ledger_sha256":parents["source_ledger_sha256"],
        "original_independent_physical_court_sha256":
            parents["source_original_independent_court_sha256"],
        "source_view_count":len(VIEWS),
        "source_original_texel_address_operations":3,
        "source_original_single_edit_sample_incidents_by_operation":by_action,
        "source_pairwise_visible_effect_overlap_sample_incidents":overlaps,
        "original_texture_linked_qualifying_source_faces":684,
        "source_candidates_checked":list(FAMILIES),
        "source_family_authority":"PENDING",
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for k in ("ledger","source-plan","atlas-assets","original-render-manifest",
              "independent-court","truth-contract","output"):
        a.add_argument("--"+k,type=Path,required=True)
    x=a.parse_args()
    paths=(x.source_plan,x.atlas_assets,x.original_render_manifest,
           x.independent_court,x.truth_contract)
    plan,assets,renderer,court,canonical=(read(p) for p in paths)
    receipts={
        "source_original_glb_sha256":plan["source_original_glb_sha256"],
        "source_original_texture_sha256":plan["source_original_texture_sha256"],
        "source_uv_plan_sha256":sha(x.source_plan),
        "source_atlas_assets_sha256":sha(x.atlas_assets),
        "original_render_manifest_sha256":sha(x.original_render_manifest),
        "source_original_independent_court_sha256":sha(x.independent_court),
        "source_sample_npz_sha256_teacher_only":renderer["source_original_sample_npz_sha256"],
        "canonical_truth_family_contract_sha256":sha(x.truth_contract),
        "source_ledger_sha256":sha(x.ledger),
    }
    result=verify(read(x.ledger),plan,assets,renderer,court,canonical,parents=receipts)
    x.output.parent.mkdir(parents=True,exist_ok=True)
    x.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],
        "source_edits":result["source_original_single_edit_sample_incidents_by_operation"],
        "owner_authority":False},sort_keys=True))

if __name__=="__main__":main()
