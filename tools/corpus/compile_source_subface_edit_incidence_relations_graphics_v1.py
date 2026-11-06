#!/usr/bin/env python3
"""Source-only Relations×Graphics join: original faces are NOT edit atoms.

Join independently courted original Cat V18s SOURCE-POSE mechanical/UV
relations and original Cat V2 REST-POSE source Cycles face/pixel/UV physical
edit observations by exact original face-ID and original-source SHA.

Crucial: source face may carry physically distinct UV edit consequences
inside ONE triangle, even in one view. This falsifies a uniform binary edit
selector per source face, not author-owned drawing partition. Across fit/rest
contexts emit only a TYPED CROSS-SCOPE FACT, never same-pose causality.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path

SCHEMA="RealSaS.OriginalCrossScopedSubFaceRelationsGraphicsTruth.v1"
CAT="1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
PINS={
    "v18s.json":"cbf44fee069e79ea3a7591669fede47befe0fb0ff7eab02aa4d5b3479a75a829",
    "v18s-court.json":"c35ab620e3cf0955bb0b8ca445838d0320b50240bf357a706385eeeb1b67e09c",
    "v18q.json":"ad88115d2cf0ffc00d241da30218a60589c25983e5d1df5d8391f07089bce0d2",
    "v2.json":"c2ab94ae6250159b046286efaa2f062c857a7557ad38d5942491dbf89ac16116",
    "v2-court.json":"a34a3050917fe563321f5e57c1ba1556844dc516965abdff201a6930913d015d",
}
NO_AUTHORITY={
    "graphics_owner_authorized":False,
    "same_drawable_training_pairs_qualified":0,
    "different_drawable_training_pairs_qualified":0,
    "iris_training_authorized":False,
    "tessa_training_authorized":False,
    "training_authorized":False,
    "materialization_authorized":False,
}


def req(ok,why):
    if not ok:raise ValueError("SUBFACE_GRAPHICS_TRUTH_"+why)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def build_cross_scope_source_profiles(v2_rows,local_edges,island_by_face,lifted):
    """Pure face-ID join; never treat UNOBSERVED edits as negative face facts."""
    ops_by_face=defaultdict(set)
    ops_by_face_view=defaultdict(set)
    first_witness={}
    pixels_by_edit=Counter()
    pixel_keys=set()
    for r in v2_rows:
        f=int(r["face_index"]);view=int(r["original_camera_view"])
        x=int(r["pixel_x"]);y=int(r["pixel_y_top_first"])
        op=int(r["source_uv_edit_operation_index"])
        req(0<=f<len(island_by_face) and view in (1,3,5,7)
            and 0<=x<192 and 0<=y<192 and op in (0,1,2),
            "ORIGINAL_FACE_PIXEL_OR_EDIT_ADDRESS_INVALID")
        key=(view,y,x)
        req(key not in pixel_keys,"PHYSICAL_PIXEL_MINTED_WITH_MULTIPLE_SOURCE_FACES")
        pixel_keys.add(key)
        ops_by_face[f].add(op)
        ops_by_face_view[(f,view)].add(op)
        first_witness.setdefault((f,view,op),(x,y))
        pixels_by_edit[op]+=1
    # A face-constant per-operation edit response B[f,k] would have to be
    # BOTH 1 and 0 for an originally measured pair of pixels here.
    dual_same_view={}
    for (f,view),ops in sorted(ops_by_face_view.items()):
        if 0 in ops and 1 in ops:
            dual_same_view[(f,view)]={
                "original_source_face_id":f,
                "original_rest_camera_view":view,
                "source_edit_0_physically_positive_pixel_xy":first_witness[(f,view,0)],
                "source_edit_1_physically_positive_pixel_xy":first_witness[(f,view,1)],
                "source_other_edit_negative_at_each_positive_pixel":True,
                "each_pixel_proved_first_hit_original_source_face":True,
                "original_artist_layer_owner":"UNKNOWN",
            }
    dual_faces={f for (f,_) in dual_same_view}
    local_both=[]
    local_one=0
    for e in local_edges:
        a=int(e["original_face_a"]);b=int(e["original_face_b"])
        req(0<=a<len(island_by_face) and 0<=b<len(island_by_face) and a!=b,
            "V18S_LOCAL_RELATION_ADDRESS_NOT_ORIGINAL_FACE")
        present_a=a in ops_by_face;present_b=b in ops_by_face
        if present_a!=present_b:
            local_one+=1
        if not present_a or not present_b:
            continue
        common=sorted(ops_by_face[a] & ops_by_face[b])
        local_both.append({
            "original_face_a":a,"original_face_b":b,
            "source_original_face_uv_continuity":
                e["source_uv_address_continuity"],
            "source_fit_axis_mechanical_divergence":
                bool(e["source_mechanical_axis_divergent"]),
            "source_fit_pooled_mechanical_divergence":
                bool(e["source_mechanical_pooled_divergent"]),
            "original_rest_both_faces_physically_edit_observed":True,
            "observed_positive_atlas_edits_face_a":sorted(ops_by_face[a]),
            "observed_positive_atlas_edits_face_b":sorted(ops_by_face[b]),
            "same_positive_edit_seen_on_both_faces_somewhere":bool(common),
            "common_source_atlas_edits_with_positive_witness":common,
            "source_mechanical_pose_vs_original_rest_edit_join":
                "SAME_SOURCE_FACE_ID__DIFFERENT_POSE_SCOPE__NOT_PAIR_CAUSAL",
            "source_same_drawable":"UNKNOWN","source_different_drawable":"UNKNOWN",
        })
    island_observed={}
    for f,edits in ops_by_face.items():
        island_observed.setdefault(int(island_by_face[f]),set()).update(edits)
    cross_lifted=[]
    for e in lifted:
        a=int(e["source_island_a"]);b=int(e["source_island_b"])
        if a not in island_observed or b not in island_observed:continue
        cross_lifted.append({
            "source_fit_island_a":a,
            "source_fit_island_b":b,
            "source_fit_same_vertex_pair_world_near":
                bool(e["original_fit_pose_world_near"]),
            "source_fit_world_near_uv_coincident":
                bool(e["original_near_pairs_same_uv_endpoint_count"]>0),
            "source_rest_physical_edit_observed_on_both_islands":True,
            "source_same_artistic_drawable":"UNKNOWN",
            "join_scope":"SOURCE_ISLAND_ID_ONLY__POSE_CONTEXTS_DIFFER",
        })
    profiles=Counter(tuple(sorted(edits)) for edits in ops_by_face.values())
    return {
        "source_original_faces_with_any_physically_courted_edit_pixel":len(ops_by_face),
        "source_original_source_face_count":len(island_by_face),
        "source_original_unwitnessed_face_count":len(island_by_face)-len(ops_by_face),
        "source_edit_positive_pixels_by_operator":[pixels_by_edit[i] for i in range(3)],
        "observed_positive_edit_profiles_per_face_PRIVATE":[
            {"source_face_id":f,"observed_positive_source_edit_indices":sorted(edits),
             "unwitnessed_source_edit_control_negative_label":"UNKNOWN"}
            for f,edits in sorted(ops_by_face.items())],
        "source_face_positive_edit_signature_census":{
            ",".join(map(str,p)):n for p,n in sorted(profiles.items())
        },
        "same_original_face_two_independent_source_edits_one_camera_witnesses_PRIVATE":
            list(dual_same_view.values()),
        "same_original_face_two_edits_one_camera_witness_count":len(dual_same_view),
        "source_face_count_with_two_different_original_edit_pixels_same_camera":
            len(dual_faces),
        "face_uniform_boolean_edit_handle_falsified_in_observed_source_scope":
            len(dual_faces)>0,
        "source_local_mechanical_uv_edges_original_count":len(local_edges),
        "source_local_edges_both_faces_physically_edit_observed":len(local_both),
        "source_local_edges_exactly_one_face_physically_edit_observed":local_one,
        "source_local_edges_both_observed_axis_divergent":sum(
            r["source_fit_axis_mechanical_divergence"] for r in local_both),
        "source_local_edges_both_observed_pooled_divergent":sum(
            r["source_fit_pooled_mechanical_divergence"] for r in local_both),
        "source_local_edges_both_observed_share_some_positive_edit":sum(
            r["same_positive_edit_seen_on_both_faces_somewhere"] for r in local_both),
        "source_local_edges_both_observed_no_common_positive_edit":sum(
            not r["same_positive_edit_seen_on_both_faces_somewhere"] for r in local_both),
        "source_local_edges_cross_scoped_relations_PRIVATE":local_both,
        "source_lifted_candidates_both_islands_physically_observed":len(cross_lifted),
        "source_lifted_both_islands_observed_world_near":sum(
            r["source_fit_same_vertex_pair_world_near"] for r in cross_lifted),
        "source_lifted_both_islands_observed_world_uv_coincident":sum(
            r["source_fit_world_near_uv_coincident"] for r in cross_lifted),
        "source_lifted_candidates_cross_scope_PRIVATE":cross_lifted,
        "source_mechanical_graphics_same_pose_causal_join":"NOT_ADMITTED_DIFFERENT_POSE_SCOPES",
        "original_source_artistic_drawable_ownership":"UNKNOWN",
        "positive_face_edit_incidence_is_not_complete_negative_support":True,
    }


def run(root,output):
    root=Path(root)
    for name,pin in PINS.items():
        req(sha(root/name)==pin,"SEALED_ORIGINAL_SOURCE_PARENT_BYTES_DRIFT:"+name)
    docs={k:json.loads((root/k).read_text()) for k in PINS}
    source,source_court,q,v2,v2court=(docs["v18s.json"],
        docs["v18s-court.json"],docs["v18q.json"],
        docs["v2.json"],docs["v2-court.json"])
    req(source["status"]==
        "SOURCE_UV_WORLD_MECHANICAL_RELATIONS_OBSERVED__OWNER_UNKNOWN"
        and source_court["status"]==
        "ORIGINAL_SOURCE_UV_WORLD_MECHANICS_RELATIONS_RECOMPUTED__OWNER_UNKNOWN"
        and source_court["independently_replayed_source_v18s_sha256"]==PINS["v18s.json"]
        and v2["status"]==
        "EXACT_LOCAL_SOURCE_DEPTH_PIXEL_UV_EDIT_TRUTH_PRODUCED__INDEPENDENT_COURT_REQUIRED"
        and v2court["status"]==
        "PASS_ORIGINAL_FACE_PIXEL_UV_EDIT_CAUSALITY_EXACT_LOCAL_DEPTH__OWNER_UNKNOWN"
        and v2court["producer_source_witness_sha256"]==PINS["v2.json"]
        and source["original_source_sha256"]==source_court["original_source_sha256"]==
            q["original_source_sha256"]==v2["original_source_sha256"]==
            v2court["source_original_glb_sha256"]==CAT
        and source["parent_source_byte_sha256"]["teacher"]==
            v2["original_parent_sha256"]["teacher.npz"]
        and source["parent_source_byte_sha256"]["uv"]==
            v2["original_parent_sha256"]["face-uv.npz"]
        and source["parent_source_byte_sha256"]["v18q"]==PINS["v18q.json"]
        and source["source_face_independent_v18a_to_native_teacher_equal"] is True
        and source["source_fit_pose_ordinals"]==[0,1]
        and source["source_heldout_pose_ordinal_used_for_fit"] is False
        and v2["source_counts"]["qualified_original_source_face_pixels"]==
            v2court["original_source_pixel_face_uv_witnesses_independently_courted"]==49494
        and v2court["original_source_physically_observable_edit_sample_positive"]==395952
        and v2court["original_source_non_target_edit_sample_absence"]==791904
        and all(t.get("graphics_owner_authorized") is False
                and t.get("iris_training_authorized") is False
                and t.get("training_authorized") is False
                and t.get("materialization_authorized") is False
                for t in (source,source_court,v2,v2court)),
        "CROSS_SCOPE_SOURCE_FACE_JOIN_AUTHORITY_INVALID")
    face_island=q["source_private_graph"][
        "source_face_observation_ir_teacher_private"][
        "source_topology_island_per_face_teacher_private"]
    req(len(face_island)==source["source_face_count"]==684
        and len(source["source_local_face_edge_evidence_private"])==424
        and len(source["source_lifted_fit_world_uv_evidence_private"])==1150
        and source["source_uv_corner_origin"]=="GLTF_IMAGE_TOP_LEFT",
        "SOURCE_FACE_NATIVE_INDEX_AND_V18S_RELATION_SCOPE")
    outcome=build_cross_scope_source_profiles(
        v2["source_face_pixel_uv_edit_private_rows"],
        source["source_local_face_edge_evidence_private"],
        face_island,
        source["source_lifted_fit_world_uv_evidence_private"])
    report={
        "schema":SCHEMA,
        "status":"SOURCE_SUBFACE_UV_PHYSICAL_EDIT_INCIDENCE_RECONCILED_WITH_V18S__ARTIST_OWNER_UNKNOWN",
        "original_source_sha256":CAT,
        "parent_source_receipts_sha256":dict(PINS),
        "evidence_scope_original_rest_edit":
            "ORIGINAL_BIND_REST__V1_V3_V5_V7__EIGHT_SAME_CYCLES_SOURCE_SAMPLES",
        "evidence_scope_mechanical_relations":
            "ORIGINAL_AUTHORED_FIT_POSES_0_1__ORIGINAL_FACE_WORLD_RELATIONS",
        "source_subface_locus_contract":
            "SOURCE_GLTF_FACE_INDEX_PLUS_BARYCENTRIC_UV_SUBADDRESS_PLUS_EDIT_PLUS_POSE_VIEW_SAMPLE",
        "source_face_uniform_control_counterexample_is_capability_relative":True,
        "source_edit_operator_absence_on_UNWITNESSED_FACE":"UNKNOWN_NOT_FALSE",
        "source_locus_address_reconciled_positively_only":True,
        **outcome,
        **NO_AUTHORITY,
    }
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],
        "source_faces_with_physical_edit_witnesses":
            report["source_original_faces_with_any_physically_courted_edit_pixel"],
        "source_faces_with_two_edit_operators_same_view":
            report["source_face_count_with_two_different_original_edit_pixels_same_camera"],
        "source_adjacent_face_edges_with_both_source_physical_witnesses":
            report["source_local_edges_both_faces_physically_edit_observed"],
        "source_cross_scope_axis_divergent_edges":
            report["source_local_edges_both_observed_axis_divergent"],
        "artist_owner":"UNKNOWN","training_authorized":False,
    },sort_keys=True),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.root,a.output)
