#!/usr/bin/env python3
"""Independent source sub-face/edit incidence+V18s relation court V1.

Independent implementation (no producer import) joins pre-admitted original
source V18s and physical V2 by strict original GLB+face/UV native SHA, and
rechecks every source face view/edit pixel incidence. Source edit absences
outside witnessed pixels are UNKNOWN, never false or artist-drawable labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict,Counter
import hashlib,json
from pathlib import Path

SCHEMA="RealSaS.IndependentOriginalSourceSubFaceRelationsGraphicsCourt.v1"
PASS="PASS_SOURCE_SUBFACE_EDIT_INCIDENCE_V18S_CROSS_SCOPE_RECOUNT__NO_ARTIST_OWNER"
CAT="1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
PINS={
    "v18s.json":"cbf44fee069e79ea3a7591669fede47befe0fb0ff7eab02aa4d5b3479a75a829",
    "v18s-court.json":"c35ab620e3cf0955bb0b8ca445838d0320b50240bf357a706385eeeb1b67e09c",
    "v18q.json":"ad88115d2cf0ffc00d241da30218a60589c25983e5d1df5d8391f07089bce0d2",
    "v2.json":"c2ab94ae6250159b046286efaa2f062c857a7557ad38d5942491dbf89ac16116",
    "v2-court.json":"a34a3050917fe563321f5e57c1ba1556844dc516965abdff201a6930913d015d",
}


def require(ok,code):
    if not ok:raise ValueError("INDEPENDENT_SUBFACE_REL_GRAPH_"+code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def recompute_independent(root,producer_path,output):
    root=Path(root)
    for name,pin in PINS.items():
        require(sha(root/name)==pin,"ORIGINAL_PARENT_BYTES_MISMATCH:"+name)
    original={k:json.loads((root/k).read_text()) for k in PINS}
    report=json.loads(Path(producer_path).read_text())
    v18s=original["v18s.json"]
    v18s_court=original["v18s-court.json"]
    v18q=original["v18q.json"]
    v2=original["v2.json"]
    v2_court=original["v2-court.json"]
    require(report["schema"]=="RealSaS.OriginalCrossScopedSubFaceRelationsGraphicsTruth.v1"
        and report["status"]==
          "SOURCE_SUBFACE_UV_PHYSICAL_EDIT_INCIDENCE_RECONCILED_WITH_V18S__ARTIST_OWNER_UNKNOWN"
        and report["parent_source_receipts_sha256"]==PINS
        and report["original_source_sha256"]==
            v18s["original_source_sha256"]==v2["original_source_sha256"]==
            v18s_court["original_source_sha256"]==
            v2_court["source_original_glb_sha256"]==CAT
        and v2_court["status"]==
            "PASS_ORIGINAL_FACE_PIXEL_UV_EDIT_CAUSALITY_EXACT_LOCAL_DEPTH__OWNER_UNKNOWN"
        and v2_court["producer_source_witness_sha256"]==PINS["v2.json"]
        and v18s_court["status"]==
            "ORIGINAL_SOURCE_UV_WORLD_MECHANICS_RELATIONS_RECOMPUTED__OWNER_UNKNOWN"
        and v18s_court["independently_replayed_source_v18s_sha256"]==PINS["v18s.json"]
        and v18s["parent_source_byte_sha256"]["teacher"]==
            v2["original_parent_sha256"]["teacher.npz"]
        and v18s["parent_source_byte_sha256"]["uv"]==
            v2["original_parent_sha256"]["face-uv.npz"]
        and report["evidence_scope_original_rest_edit"]==
            "ORIGINAL_BIND_REST__V1_V3_V5_V7__EIGHT_SAME_CYCLES_SOURCE_SAMPLES"
        and report["evidence_scope_mechanical_relations"]==
            "ORIGINAL_AUTHORED_FIT_POSES_0_1__ORIGINAL_FACE_WORLD_RELATIONS"
        and report["source_mechanical_graphics_same_pose_causal_join"]==
            "NOT_ADMITTED_DIFFERENT_POSE_SCOPES"
        and report["source_edit_operator_absence_on_UNWITNESSED_FACE"]==
            "UNKNOWN_NOT_FALSE"
        and report["source_subface_locus_contract"]==
            "SOURCE_GLTF_FACE_INDEX_PLUS_BARYCENTRIC_UV_SUBADDRESS_PLUS_EDIT_PLUS_POSE_VIEW_SAMPLE"
        and report["source_original_artistic_drawable_ownership"]=="UNKNOWN"
        and all(report[k] is False for k in (
            "graphics_owner_authorized","iris_training_authorized",
            "tessa_training_authorized","materialization_authorized",
            "training_authorized"))
        and report["same_drawable_training_pairs_qualified"]==0
        and report["different_drawable_training_pairs_qualified"]==0,
        "SOURCE_LINEAGE_OR_FALSE_OWNER_PROMOTION")
    rows=v2["source_face_pixel_uv_edit_private_rows"]
    require(len(rows)==v2_court["original_source_pixel_face_uv_witnesses_independently_courted"]==49494,
            "SOURCE_PHYSICAL_PIXEL_COUNT_OR_INDEPENDENT_COURT_DRIFT")
    incidence=defaultdict(lambda:defaultdict(set))
    example=defaultdict(dict)
    pix=set();witness_by_edit=Counter()
    for row in rows:
        face=row["face_index"];view=row["original_camera_view"]
        x=row["pixel_x"];y=row["pixel_y_top_first"];edit=row["source_uv_edit_operation_index"]
        require(type(face)==int and type(edit)==int and 0<=face<684 and
                0<=edit<3 and view in (1,3,5,7),
                "INVALID_SOURCE_ORIGINAL_FACE_PIXEL_OPERATOR")
        key=(view,y,x)
        require(key not in pix,"ONE_PIXEL_MULTIPLE_ORIGINAL_SOURCE_FACES")
        pix.add(key)
        incidence[face][view].add(edit)
        example[(face,view)].setdefault(edit,(x,y))
        witness_by_edit[edit]+=1
    require([witness_by_edit[i] for i in range(3)]==
            v2_court["original_source_face_pixels_by_edit"]==[24851,24400,243],
            "SOURCE_PHYSICAL_FACE_PIXEL_OP_COURT_MISMATCH")
    unique_faces=set(incidence)
    common_face_view=[]
    for face in sorted(unique_faces):
        for view in sorted(incidence[face]):
            ops=incidence[face][view]
            if 0 in ops and 1 in ops:
                common_face_view.append((face,view))
    dual_faces={face for face,view in common_face_view}
    profiles=Counter(tuple(sorted({
        op for ops in incidence[face].values() for op in ops
    })) for face in unique_faces)
    require(
        len(unique_faces)==427 and len(dual_faces)==90 and
        len(common_face_view)==130 and profiles==Counter({
            (0,):203,(1,):113,(0,1):90,(2,):21}),
        "SOURCE_SUBFACE_UNIFORM_EDIT_COUNTEREXAMPLE_DISAGREEMENT")
    # Every proposed same-face different-UV-edit physical witness must have
    # two ACTUAL distinct original pixel addresses in the SAME original view,
    # with original V2 independent court guaranteeing true/other effects.
    producer_witness={
        (w["original_source_face_id"],w["original_rest_camera_view"]):w
        for w in report[
            "same_original_face_two_independent_source_edits_one_camera_witnesses_PRIVATE"]
    }
    require(set(producer_witness)==set(common_face_view),
        "DUAL_SOURCE_FACE_VIEW_WITNESS_IDS_DRIFT")
    for face,view in common_face_view:
        w=producer_witness[(face,view)]
        p0=tuple(w["source_edit_0_physically_positive_pixel_xy"])
        p1=tuple(w["source_edit_1_physically_positive_pixel_xy"])
        require(p0!=p1 and p0==example[(face,view)][0]
                and p1==example[(face,view)][1]
                and w["source_other_edit_negative_at_each_positive_pixel"] is True,
                "ORIGINAL_FACE_EDIT_CONTRADICTION_WITNESS_FRAUD")
    g=v18q["source_private_graph"]
    mapping=g["source_face_observation_ir_teacher_private"][
        "source_topology_island_per_face_teacher_private"]
    require(len(mapping)==684,"SOURCE_ORIGINAL_FACE_ISLAND_COUNT_DRIFT")
    local=v18s["source_local_face_edge_evidence_private"]
    require(len(local)==424 and all(row["source_uv_address_continuity"]==
            "SAME_ORIGINAL_UV_ENDPOINT_COORDINATES" for row in local),
            "ORIGINAL_SOURCE_V18S_LOCAL_UV_RELATIONS_CHANGED")
    local_both=0;local_one=0;local_mech=0;local_pooled=0;common=0;disjoint=0
    lkeys=set()
    for e in local:
        a=e["original_face_a"];b=e["original_face_b"]
        key=tuple(sorted((a,b)))
        require(key not in lkeys and a!=b,"DUPLICATE_LOCAL_SOURCE_RELATION")
        lkeys.add(key)
        in_a=a in unique_faces;in_b=b in unique_faces
        local_one+=int(in_a!=in_b)
        if not(in_a and in_b):continue
        local_both+=1
        local_mech+=int(e["source_mechanical_axis_divergent"])
        local_pooled+=int(e["source_mechanical_pooled_divergent"])
        a_ops={z for ops in incidence[a].values() for z in ops}
        b_ops={z for ops in incidence[b].values() for z in ops}
        common+=int(bool(a_ops&b_ops))
        disjoint+=int(not bool(a_ops&b_ops))
    observed_islands={int(mapping[f]) for f in unique_faces}
    lifted=v18s["source_lifted_fit_world_uv_evidence_private"]
    both_lifted=[e for e in lifted if e["source_island_a"] in observed_islands
                 and e["source_island_b"] in observed_islands]
    near=sum(e["original_fit_pose_world_near"] for e in both_lifted)
    coincident=sum(e["original_near_pairs_same_uv_endpoint_count"]>0 for e in both_lifted)
    require(local_both==218 and local_one==47 and
            local_mech==31 and local_pooled==14 and common==216 and
            disjoint==2 and len(both_lifted)==725 and near==346 and
            coincident==346,
            "SOURCE_RELATIONS_CROSS_SCOPED_CENSUS_DISAGREEMENT")
    require(
        report["source_original_faces_with_any_physically_courted_edit_pixel"]==
            len(unique_faces)
        and report["source_original_source_face_count"]==684
        and report["source_original_unwitnessed_face_count"]==684-len(unique_faces)
        and report["source_face_count_with_two_different_original_edit_pixels_same_camera"]==
            len(dual_faces)
        and report["same_original_face_two_edits_one_camera_witness_count"]==
            len(common_face_view)
        and report["source_local_edges_both_faces_physically_edit_observed"]==local_both
        and report["source_local_edges_exactly_one_face_physically_edit_observed"]==local_one
        and report["source_local_edges_both_observed_axis_divergent"]==local_mech
        and report["source_local_edges_both_observed_pooled_divergent"]==local_pooled
        and report["source_local_edges_both_observed_share_some_positive_edit"]==common
        and report["source_local_edges_both_observed_no_common_positive_edit"]==disjoint
        and report["source_lifted_candidates_both_islands_physically_observed"]==
            len(both_lifted)
        and report["source_lifted_both_islands_observed_world_near"]==near
        and report["source_lifted_both_islands_observed_world_uv_coincident"]==coincident,
        "PRODUCER_CROSS_SCOPE_RELATION_LABEL_OR_COUNT_INVALID")
    result={
        "schema":SCHEMA,
        "status":PASS,
        "original_source_sha256":CAT,
        "source_cross_scope_producer_sha256":sha(producer_path),
        "original_parent_receipts_sha256":PINS,
        "source_distinct_face_count_with_positive_rest_edit_witness":len(unique_faces),
        "original_source_face_count":684,
        "source_unwitnessed_faces":684-len(unique_faces),
        "source_pixels_original_v2_physically_courted":len(rows),
        "source_face_dual_edit_same_view_counterexamples":len(dual_faces),
        "source_face_camera_dual_edit_witnesses":len(common_face_view),
        "physical_original_source_face_positive_edit_profile_census":{
            ",".join(map(str,k)):v for k,v in sorted(profiles.items())},
        "mechanical_source_local_edges_both_rest_physical_observed":local_both,
        "mechanical_axis_divergent_source_local_edges_both_observed":local_mech,
        "source_lifted_graph_pairs_both_islands_rest_physical_observed":len(both_lifted),
        "source_fit_mechanics_and_rest_graphics_observations":"DIFFERENT_SOURCE_POSE_SCOPES",
        "uniform_boolean_source_face_selector_falsified_under_declared_rest_edit_contract":True,
        "face_internal_uv_masks_or_subface_addresses_needed_to_preserve_observed_independent_edits":True,
        "source_artist_drawable_ownership":"UNKNOWN",
        "source_unobserved_edit_controls_for_faces":"UNKNOWN_NOT_FALSE",
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":PASS,"observed_original_faces":len(unique_faces),
        "same_face_dual_edit_counterexample_count":len(dual_faces),
        "same_view_dual_edit_witnesses":len(common_face_view),
        "both_physically_observed_local_edges":local_both,
        "axis_divergent_both_observed_edges":local_mech,
        "artist_owner":"UNKNOWN","training_authorized":False
    },sort_keys=True),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    recompute_independent(a.root,a.report,a.output)
