#!/usr/bin/env python3
"""Proof-carrying cross-family source evidence audit, original courts only.

Joins independently courted Cat V18s original source UV/world/response data,
Cat original physically sampled linked-atlas edits, and Triceratops K3 source
material/edit sample measurements. The join is SOURCE-SUBJECT-level only:
no Cat source face -> physical edited screen sample correspondence is present.
No learned labels or author drawable relations may be minted.

Requires original literal document bytes, SHA seals, independently verified
courts and exact experimentally qualified scope. Original source PRIVATE.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.relations_graphics_proof_contract_v1 import (
    SourceScope, compare_source_scopes, FALSE_AUTHORITY, digest,
)

SCHEMA = "RealSaS.CourtedRelationsGraphicsSourceTruthInventory.v1"
PINS = {
    "v18s": "cbf44fee069e79ea3a7591669fede47befe0fb0ff7eab02aa4d5b3479a75a829",
    "v18s_court": "c35ab620e3cf0955bb0b8ca445838d0320b50240bf357a706385eeeb1b67e09c",
    "cat_uv_ledger": "60ce8085c91282d43e2a16147d08dd6883764a5168270812f9f1be7e46a4b435",
    "cat_uv_court": "92afb720dff9ba6a6524901d7363e217ed499f1fdf22bbee0b09c8dabfda0b3e",
    "triceratops_k3_court": "021786e2bc9bc28d30d717c08eec3d92e64fca1ebc714a451ab85e2988b4882d",
}
FILE_NAMES = {
    "v18s": "PRIVATE_SOURCE_UV_WORLD_MECHANICAL_RELATION_EVIDENCE_V18S.json",
    "v18s_court": "PRIVATE_INDEPENDENT_SOURCE_UV_WORLD_RELATION_COURT_V18S.json",
    "cat_uv_ledger": "PRIVATE_SOURCE_CAT_UV_GRAPHICS_EDIT_LEDGER_V1.json",
    "cat_uv_court": "PRIVATE_INDEPENDENT_SOURCE_CAT_UV_GRAPHICS_LEDGER_COURT_V1.json",
    "triceratops_k3_court": "PRIVATE_INDEPENDENT_ORIGINAL_SOURCE_K3_SAME_INDEX_COURT_V1.json",
}
CAT = "1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
TRICERATOPS = "ad66b51cc5fc44c9a3429a5b1b98a61621af08d58d102d8a817499bd6a7f5b5a"


def require(ok, why):
    if not ok:
        raise ValueError("PROOF_SOURCE_COURT_" + why)


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _no_labels(*documents):
    for d in documents:
        require(d.get("graphics_owner_authorized") is False
                and d.get("iris_training_authorized") is False
                and d.get("training_authorized") is False
                and d.get("materialization_authorized") is False
                and d.get("same_drawable_training_pairs_qualified") in (None, 0)
                and d.get("different_drawable_training_pairs_qualified") in (None, 0),
                "UNAUTHORIZED_OWNER_OR_STUDENT_LABEL")


def audit_original_courts(files: dict[str, dict]) -> dict:
    require(set(files) == set(PINS), "MISSING_ORIGINAL_COURT")
    v, vc, c, cc, tri = (
        files["v18s"], files["v18s_court"],
        files["cat_uv_ledger"], files["cat_uv_court"],
        files["triceratops_k3_court"],
    )
    _no_labels(v, vc, c, cc, tri)
    require(
        v.get("schema") ==
            "RealSaS.OriginalSourceUVWorldMechanicalRelationEvidence.v18s" and
        vc.get("schema") ==
            "RealSaS.IndependentOriginalUVWorldMechanicalRelationCourt.v18s" and
        v.get("status") ==
            "SOURCE_UV_WORLD_MECHANICAL_RELATIONS_OBSERVED__OWNER_UNKNOWN" and
        vc.get("status") ==
            "ORIGINAL_SOURCE_UV_WORLD_MECHANICS_RELATIONS_RECOMPUTED__OWNER_UNKNOWN" and
        vc.get("independently_replayed_source_v18s_sha256") == PINS["v18s"] and
        vc.get("heldout_pose2_not_consulted") is True and
        v.get("source_heldout_pose_ordinal_used_for_fit") is False and
        v.get("source_face_independent_v18a_to_native_teacher_equal") is True and
        v.get("original_author_drawable_ownership") == "UNKNOWN" and
        v.get("source_texture_rgba_continuity_across_uv") ==
            "UNKNOWN_NO_IMAGE_SAMPLE_COURT",
        "V18S_ORIGINAL_COURT_SEAL_OR_SCOPE_INVALID")
    rloc = v["source_local_relation_census"]
    rlift = v["source_lifted_relation_census"]
    require(
        vc["source_local_evidence_recount"] == rloc and
        vc["source_lifted_evidence_recount"] == rlift and
        v["source_face_count"] == vc["source_original_face_count"] == 684 and
        v["source_graph_island_count"] == vc["source_original_island_count"] == 260 and
        rloc["all"] == rloc["uv_continuous"] == 424 and
        rloc["axis_divergent_but_uv_continuous"] == 59 and
        rloc["pooled_mechanical_divergent"] == 28 and
        rlift["retrieval_candidates_total"] == 1150 and
        rlift["same_vertex_pair_world_near_both_fit"] == 497 and
        rlift["world_fit_near_with_coincident_uv"] == 480 and
        rlift["world_fit_near_without_coincident_uv"] == 17,
        "V18S_RELATIONS_COURT_RECOUNT_DRIFT")
    require(
        c.get("schema") ==
            "RealSaS.Corpus370SourceTextureUVEditConsequenceLedger.v1" and
        cc.get("schema") ==
            "RealSaS.Corpus370IndependentCatSourceUVConsequenceLedgerCourt.v1" and
        c.get("status") ==
            "SOURCE_CAT_LINKED_TEXTURE_EDIT_EVIDENCE_COURTED__FAMILY_AUTHORITY_PENDING" and
        cc.get("status") ==
            "PASS_CAT_SOURCE_UV_GRAPHICS_LEDGER_ACCOUNTING__TRUTH_AUTHORITY_PENDING" and
        cc.get("source_ledger_sha256") == PINS["cat_uv_ledger"] and
        c.get("parent_receipts",{}).get("source_original_independent_court_sha256") ==
            cc.get("original_independent_physical_court_sha256") and
        c.get("source_artist_drawable_owner") == "UNKNOWN" and
        c.get("source_uv_atlas_edits_are_art_drawables") is False and
        cc.get("source_family_authority") == "PENDING" and
        c.get("source_visibility_occlusion_from_this_court") == "UNKNOWN_NOT_MEASURED" and
        c.get("source_ray_depth_compositing_from_this_court") == "UNKNOWN_NOT_MEASURED",
        "CAT_PHYSICAL_SOURCE_LEDGER_SCOPE_OR_INDEPENDENT_COURT_INVALID")
    effects=c["source_edit_consequence_totals"]
    require(
        effects["single_edit_sample_incidents_by_operation"] ==
            cc["source_original_single_edit_sample_incidents_by_operation"] ==
            [290979, 264554, 6048] and
        effects["pairwise_visible_sample_support_overlap"] ==
            cc["source_pairwise_visible_effect_overlap_sample_incidents"] and
        effects["pairwise_visible_sample_support_overlap"]["0_1"] == 2799 and
        effects["heldout_source_pair_joint_sample_mismatches"] == 0 and
        effects["heldout_source_pair_joint_eight_sample_mean_mismatches"] == 0 and
        effects["source_alpha_drift"] == 0 and
        cc["source_original_texel_address_operations"] == 3 and
        cc["source_view_count"] == 4 and
        cc["original_texture_linked_qualifying_source_faces"] == 684,
        "CAT_SOURCE_EDIT_EFFECTS_INDEPENDENT_COURT_DISAGREE")
    require(
        tri.get("schema") ==
            "RealSaS.Corpus370IndependentOriginalThreePaletteCyclesEditCourt.v1" and
        tri.get("status") ==
            "PASS_ORIGINAL_PHYSICAL_K3_SAME_CYCLES_SAMPLE_EDIT_COURT__OWNER_UNKNOWN" and
        tri.get("failure_reasons") == [] and
        tri.get("source_artist_drawable_owner") == "UNKNOWN" and
        tri.get("source_original_rest_only") is True and
        tri.get("producer_court_source_full_not_used_to_choose_depth") is True and
        tri.get("source_group_count_K") == 3 and
        tri.get("qualified_sample_total") == 1179646 and
        tri.get("original_physical_overlapping_source_sample_count") == 14908 and
        tri.get("original_camera_views") == [1,3,5,7] and
        tri.get("cycles_indices") == [0,4,8,12,16,20,24,28],
        "TRICERATOPS_ORIGINAL_PHYSICAL_K3_COURT_UNQUALIFIED")
    per_view = tri["source_per_view_results"]
    require(len(per_view) == 4
            and [row["source_view_index"] for row in per_view] == [1,3,5,7],
            "TRICERATOPS_PHYSICAL_VIEW_SCOPE")
    changed = [0,0,0]
    for row in per_view:
        arms = row["metric_rows"]
        require([x["arm"] for x in arms] == [
            "SOURCE_FULL", "SOURCE_EDITED_FULL_0",
            "SOURCE_EDITED_FULL_1", "SOURCE_EDITED_FULL_2"] and
            all(x["sample_rgba_mismatch_count"] == 0 and
                x["eight_sample_mean_rgba_mismatch_count"] == 0
                for x in arms),
            "TRICERATOPS_SOURCE_PHYSICAL_PARITY_OR_ARM_INVALID")
        for i in range(3):
            changed[i] += int(arms[i+1]["physical_full_edit_changed_sample_pixel_count"])
    require(changed == [4265,157526,7955],
            "TRICERATOPS_ACTUAL_EDIT_CONSEQUENCE_DRIFT")
    require(v["original_source_sha256"] ==
            vc["original_source_sha256"] ==
            c["original_source_sha256"] ==
            cc["original_source_sha256"] == CAT and
            tri["original_source_sha256"] == TRICERATOPS,
            "SOURCE_SUBJECT_SHA_JOIN_INVALID")

    # This is only subject-level comparability. V18s observes ORIGINAL indexed
    # face/vertex and fit poses; source UV physical court stores aggregate
    # edit/pixel incidences at rest pose. It DOES NOT give a pixel->face map
    # or matching pose/sample operation for a local causal join.
    g=SourceScope(
        CAT, "ORIGINAL_INDEXED_FACE_VERTEX_EDGES",
        PINS["v18s"], "SOURCE_FIT_POSES_0_1",
        "SOURCE_WORLD_NO_CAMERA", "SOURCE_VERTEX_GEOMETRY",
        "MECHANICAL_AXIS_SOURCE_RESPONSE", PINS["v18s_court"],
    )
    e=SourceScope(
        CAT, "ORIGINAL_CYCLES_SAMPLE_IMAGE_AGGREGATE",
        PINS["cat_uv_ledger"], "ORIGINAL_BIND_REST_ONLY",
        "ORIGINAL_SOURCE_V1_V3_V5_V7",
        "CYCLES_OFFSETS_0_4_8_12_16_20_24_28",
        "LINKED_ATLAS_3_RGB_INVERT_EDITS",
        PINS["cat_uv_court"],
    )
    join=compare_source_scopes(g, e)
    require(join["status"] == "SUBJECT_LEVEL_ONLY__NO_LOCAL_CONSEQUENCE_JOIN" and
            join["same_original_source"] is True and
            join["local_consequence_join_authorized"] is False,
            "FORGED_FACE_LEVEL_CAUSAL_JOIN")
    return {
        "schema": SCHEMA,
        "status": "INDEPENDENT_SOURCE_FAMILIES_RECONCILED__NO_SYNTHETIC_OWNER_LABEL",
        "original_subjects": [
            {"original_sha256":CAT,"name":"CAT",
             "qualified_source_relation_domains":[
                {"family":"LOCAL_UV_ADDRESS_VS_MECHANICAL_RESPONSE",
                 "unit":"ORIGINAL_INDEXED_SHARED_EDGE", "count":424,
                 "uv_continuity_witnesses":424,
                 "mechanically_axis_divergent_and_uv_continuous":59,
                 "source_scope":"FIT_POSES_0_1__NO_RENDER_SAMPLES"},
                {"family":"WORLD_NEAR_SOURCE_UV_ENDPOINT_COINCIDENCE",
                 "unit":"V18Q_PRELOCKED_CROSS_ISLAND_CANDIDATE",
                 "count":1150,"source_fit_near":497,
                 "source_near_and_uv_endpoint_coincident":480,
                 "source_scope":"FIT_POSES_0_1__NO_RENDER_SAMPLES"},
                {"family":"PHYSICAL_LINKED_ATLAS_EDIT_CONSEQUENCE",
                 "unit":"SOURCE_CYCLES_SAMPLE_OPERATION_PIXEL_INCIDENCE",
                 "count":sum(effects["single_edit_sample_incidents_by_operation"]),
                 "per_operation":effects["single_edit_sample_incidents_by_operation"],
                 "pairwise_effect_support_overlap_0_1":2799,
                 "source_scope":"REST_V1_V3_V5_V7_SAME_CYCLES_SAMPLE",
                 "original_artist_parts":"UNKNOWN"}]},
            {"original_sha256":TRICERATOPS,"name":"TRICERATOPS",
             "qualified_source_relation_domains":[
                {"family":"SOURCE_DEPTH_AND_PALETTE_K3_EDIT_CONSEQUENCE",
                 "unit":"CYCLES_SOURCE_SAMPLE_PIXEL",
                 "count":tri["qualified_sample_total"],
                 "unknown_source_sample_count":2,
                 "source_overlap_sample_incidence":14908,
                 "per_original_palette_edit_changed_source_samples":changed,
                 "source_scope":"REST_V1_V3_V5_V7_SAME_CYCLES_SAMPLE",
                 "original_artist_parts":"UNKNOWN"}]}
        ],
        "cat_local_source_graph_join_to_edited_sample_pixel":join,
        "cat_direct_face_to_edit_effect_truth":"UNKNOWN__NO_PIXEL_TO_SOURCE_FACE_EFFECT_BINDING",
        "cross_subject_locus_pair_truth":"NOT_JOINABLE__DIFFERENT_SOURCE_SHA",
        "author_drawable_identifiability":"NOT_IDENTIFIED",
        "full_reference_runtime_parity":"NOT_QUALIFIED_BY_THESE_SOURCE_COURTS",
        "student_inference_and_unseen_generalization":"UNMEASURED",
        "evidence_parents_sha256":dict(PINS),
        **FALSE_AUTHORITY,
    }


def court(root: Path, out: Path) -> dict:
    paths = {k: Path(root) / name for k,name in FILE_NAMES.items()}
    for key,path in paths.items():
        require(file_sha(path) == PINS[key], "SOURCE_PARENT_SHA_MISMATCH:" + key)
    facts={key:json.loads(path.read_text()) for key,path in paths.items()}
    result=audit_original_courts(facts)
    result["receipt_sha256"] = digest(result)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status":result["status"],"cat_original_face_edges":424,
        "cat_atlas_edit_incidents":561581,
        "triceratops_qualified_same_index_samples":1179646,
        "cat_direct_face_to_edit_effect_truth":"UNKNOWN",
        "artist_owner":"UNKNOWN","training_authorized":False,
        "receipt_sha256":result["receipt_sha256"],
    }, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    court(args.root,args.output)
