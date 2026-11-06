#!/usr/bin/env python3
"""Independent V18s source court (does not import V18s producer).

Rereads original independently-courted V18a/V18q/V18r/native-Cat-UV parents,
verifies exact topology index provenance and recomputes EVERY local edge and
lifted original-source proposal with a separate nested-pair calculation.
Retains source-local scope and zero artist/IRIS/training admission.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "RealSaS.IndependentOriginalUVWorldMechanicalRelationCourt.v18s"
PASS = "ORIGINAL_SOURCE_UV_WORLD_MECHANICS_RELATIONS_RECOMPUTED__OWNER_UNKNOWN"
SOURCE_SCHEMA = "RealSaS.OriginalSourceUVWorldMechanicalRelationEvidence.v18s"
NO_AUTHORITY_KEYS = (
    "graphics_owner_authorized", "iris_training_authorized",
    "tessa_training_authorized", "materialization_authorized",
    "training_authorized"
)


def require(ok, why):
    if not ok:
        raise ValueError("INDEPENDENT_V18S_SOURCE_" + why)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_source(*, root, pins, report_path, court_path):
    root = Path(root)
    pins = json.loads(Path(pins).read_text())
    report_path = Path(report_path)
    original = json.loads(report_path.read_text())
    required_files = {
        "v18r": "v18r.json",
        "v18r_court": "v18r-court.json",
        "v18q": "v18q.json",
        "v18q_court": "v18q-court.json",
        "v18a": "v18a.npz",
        "v13b": "v13b.npz",
        "teacher": "teacher.npz",
        "uv": "face-uv.npz",
        "uv_binding": "uv-alignment.json",
    }
    require(set(pins) == set(required_files), "PIN_SET")
    for key, name in required_files.items():
        require(sha(root / name) == pins[key], "IMMUTABLE_PARENT_SHA:" + key)
    r = json.loads((root / "v18r.json").read_text())
    rc = json.loads((root / "v18r-court.json").read_text())
    q = json.loads((root / "v18q.json").read_text())
    qc = json.loads((root / "v18q-court.json").read_text())
    witness = json.loads((root / "uv-alignment.json").read_text())
    source = r["original_source_sha256"]
    require(source == q["original_source_sha256"] ==
            rc["original_source_sha256"] == witness["original_glb_sha256"] ==
            original["original_source_sha256"],
            "SOURCE_ORIGINAL_GLTF_IDENTITY")
    require(rc["status"] ==
            "SOURCE_GEOMETRY_SPATIAL_AND_CREASE_COURTED__OWNER_UNKNOWN"
            and witness["source_to_teacher_face_uv_binding"] ==
            "VERIFIED_NATIVE_BLENDER_FACE_ORDER_AND_RAW_GLTF_UV_MULTISET"
            and witness["source_uv_comparison"]["matched"] is True
            and witness["original_teacher_sha256"] == pins["teacher"]
            and r["parents_sha256"]["v18q_source_ir"] == pins["v18q"]
            and r["parents_sha256"]["v18q_independent_court"] == pins["v18q_court"]
            and r["parents_sha256"]["v18a_original_arrays"] == pins["v18a"]
            and r["parents_sha256"]["v13b_actual_source_pose_vertices"] == pins["v13b"]
            and r["source_heldout_pose2_only_court_no_fit"] is True,
            "ORIGINAL_SOURCE_PARENT_LINEAGE")
    with np.load(root / "v18a.npz", allow_pickle=False) as a:
        original_faces = np.asarray(a["faces"], dtype=np.int64)
        source_edges = np.asarray(a["graph_edges"], dtype=np.int64)
    with np.load(root / "teacher.npz", allow_pickle=False) as t:
        source_teacher_faces = np.asarray(t["faces"], dtype=np.int64)
    with np.load(root / "face-uv.npz", allow_pickle=False) as u:
        corner = np.asarray(u["face_uv"], dtype=np.float64)
    with np.load(root / "v13b.npz", allow_pickle=False) as p:
        xyz = np.asarray(p["baseline_vertices"], dtype=np.float64)
    require(np.array_equal(original_faces, source_teacher_faces)
            and corner.shape == (len(original_faces), 3, 2)
            and np.isfinite(corner).all()
            and xyz.ndim == 3 and xyz.shape[0] == 3
            and np.isfinite(xyz).all()
            and sha(root / "v13b.npz") == rc[
                "original_parent_source_sha256"]["v13b_actual_source_pose_vertices"],
            "ORIGINAL_FACE_POSE_UV_ORDER")
    uv_hash = hashlib.sha256(
        np.ascontiguousarray(corner.astype("<f8")).tobytes()).hexdigest()
    require(uv_hash == witness["face_uv_array_sha256"],
            "NATIVE_ORIGINAL_GLTF_CORNER_UV_WITNESS")

    island_for_face = q["source_private_graph"][
        "source_face_observation_ir_teacher_private"][
        "source_topology_island_per_face_teacher_private"]
    local = r["original_source_world_evidence_teacher_private"][
        "source_local_geometry_edge_evidence_private"]
    lifted = r["original_source_world_evidence_teacher_private"][
        "source_lifted_world_proximity_evidence_private"]
    qlift = q["source_private_graph"][
        "source_nonlocal_graph_affinity_hypotheses_teacher_private"]
    require(len(island_for_face) == len(original_faces)
            and len(local) == len(source_edges)
            and len(qlift) == len(lifted)
            and original["source_fit_pose_ordinals"] == [0,1]
            and original["source_heldout_pose_ordinal_used_for_fit"] is False
            and original["uv_endpoint_coordinate_threshold"] == 1e-6
            and original["fit_near_vertex_pair_threshold_per_extent"] == .005,
            "FIT_SCOPE_AND_THRESHOLDS")

    local_map = {}
    for row in local:
        a = row["face_a_source_teacher_only"]
        b = row["face_b_source_teacher_only"]
        key = tuple(sorted((a, b)))
        require(key not in local_map, "LOCAL_DUPLICATE")
        local_map[key] = row
    expect_local_edges = {tuple(sorted(map(int, pair))) for pair in source_edges}
    require(set(local_map) == expect_local_edges and
            len(original["source_local_face_edge_evidence_private"]) ==
            len(local_map), "ORIGINAL_LOCAL_FACE_EDGE_SET")

    # Recompute direct face-corner edge endpoints from V18a source faces,
    # never reuse producer's source-vertex -> UV address map.
    lc = Counter()
    expected_local = []
    for key in sorted(local_map):
        a,b=key
        same = sorted(set(original_faces[a].tolist()) &
                      set(original_faces[b].tolist()))
        require(len(same)==2,"LOCAL_NOT_SHARED_ORIGINAL_SOURCE_EDGE")
        delta=[]
        for v in same:
            ia=int(np.flatnonzero(original_faces[a]==v)[0])
            ib=int(np.flatnonzero(original_faces[b]==v)[0])
            delta.append(float(np.linalg.norm(corner[a,ia]-corner[b,ib])))
        continuity = max(delta)<=1e-6
        evidence=local_map[key]
        row={
            "original_face_a":a,"original_face_b":b,
            "original_source_shared_vertex_ids":same,
            "source_shared_edge_uv_endpoint_delta_max":max(delta),
            "source_uv_address_continuity":
                "SAME_ORIGINAL_UV_ENDPOINT_COORDINATES" if continuity else
                "DISCONTINUOUS_ORIGINAL_UV_ENDPOINT_COORDINATES",
            "source_mechanical_axis_divergent":
                bool(evidence["source_mechanical_any_axis_divergent"]),
            "source_mechanical_pooled_divergent":
                bool(evidence["source_mechanical_pooled_divergent"]),
            "source_local_normal_crease_stable_fit":
                bool(evidence["fit_consistent_geometric_normal_crease_30deg"]),
            "same_drawable":"UNKNOWN","different_drawable":"UNKNOWN",
            "true_source_material_datablock_identity":"UNKNOWN",
        }
        expected_local.append(row)
        lc["all"]+=1
        lc["uv_continuous" if continuity else "uv_endpoint_discontinuous"]+=1
        if row["source_mechanical_axis_divergent"]:
            lc["at_least_one_mechanical_axis_divergent"]+=1
            if continuity:lc["axis_divergent_but_uv_continuous"]+=1
        if row["source_mechanical_pooled_divergent"]:
            lc["pooled_mechanical_divergent"]+=1
    # Producer preserves V18r local order (not sorted); independent court maps
    # rows by immutable source face keys so changing order cannot hide an error.
    actual_local = {
        (x["original_face_a"],x["original_face_b"]):x
        for x in original["source_local_face_edge_evidence_private"]
    }
    require(len(actual_local)==len(expected_local) and all(
        actual_local[(x["original_face_a"],x["original_face_b"])]==x
        for x in expected_local), "LOCAL_FACE_UV_TRUTH_SUBSTITUTION")
    require(dict(lc)==original["source_local_relation_census"],
            "LOCAL_SOURCE_CENSUS_DRIFT")

    # Recompute lifted candidate neighborhoods via explicit *same original
    # vertex IDs* at both fit poses, not independent V18r per-pose minima.
    islands = np.asarray(island_for_face, dtype=np.int64)
    groups = {
        int(k): np.unique(original_faces[islands==k].ravel())
        for k in sorted(set(islands.tolist()))
    }
    from_q = {tuple(sorted((r["topology_island_a_source_teacher_only"],
                           r["topology_island_b_source_teacher_only"])))
              for r in qlift}
    from_r = {tuple(sorted((r["source_island_a_teacher_only"],
                           r["source_island_b_teacher_only"])))
              for r in lifted}
    require(from_q == from_r and len(from_q) == len(qlift) == len(lifted),
            "FROZEN_LIFTED_CANDIDATE_SET")
    original_lifted = original["source_lifted_fit_world_uv_evidence_private"]
    require(len(original_lifted) == len(lifted), "LIFTED_SOURCE_ROW_CARDINALITY")
    original_map = {
        (r["source_island_a"],r["source_island_b"]):r
        for r in original_lifted
    }
    require(len(original_map) == len(lifted), "LIFTED_OUTPUT_DUPLICATE")
    extent=float(np.sqrt(np.sum(
        (xyz[0].max(axis=0)-xyz[0].min(axis=0))**2)))
    sc=Counter()
    for row in lifted:
        a=int(row["source_island_a_teacher_only"])
        b=int(row["source_island_b_teacher_only"])
        va,vb=groups[a],groups[b]
        # Direct independent loops; original source vertex IDs, UV corners.
        n=0
        same_uv=0
        different_uv=0
        min_uv=None
        for i in va:
            for j in vb:
                d0=float(np.linalg.norm(xyz[0,i]-xyz[0,j]))/extent
                d1=float(np.linalg.norm(xyz[1,i]-xyz[1,j]))/extent
                if d0 > .005 or d1 > .005:
                    continue
                n+=1
                uv_a = [
                    corner[fi,c] for fi in np.flatnonzero(
                        islands==a) for c in range(3)
                    if original_faces[fi,c] == i]
                uv_b = [
                    corner[fi,c] for fi in np.flatnonzero(
                        islands==b) for c in range(3)
                    if original_faces[fi,c] == j]
                require(uv_a and uv_b, "SOURCE_CORNER_VERTEX_ABSENT")
                gap=min(float(np.linalg.norm(x-y))
                        for x in uv_a for y in uv_b)
                min_uv=gap if min_uv is None else min(gap,min_uv)
                if gap<=1e-6:same_uv+=1
                else:different_uv+=1
        expected={
            "source_island_a":a,"source_island_b":b,
            "original_fit_pose_same_pair_near_vertex_count":n,
            "original_fit_pose_world_near":bool(n),
            "world_fit_near_original_extent_threshold":.005,
            "original_near_pairs_same_uv_endpoint_count":same_uv,
            "original_near_pairs_different_uv_endpoint_count":different_uv,
            "original_near_pair_minimum_uv_gap":min_uv,
            "source_world_uv_joint_relation":
                "WORLD_FIT_NEAR_AND_AT_LEAST_ONE_UV_ENDPOINT_COINCIDENT"
                if same_uv else "WORLD_FIT_NEAR_BUT_NO_UV_ENDPOINT_COINCIDENT"
                if n else "NOT_NEAR_IN_SAME_FIT_VERTEX_PAIR",
            "source_artistic_owner_merge":"UNKNOWN",
            "source_material_datablock_identity":"UNKNOWN",
            "source_actual_visual_appearance_continuity":"UNKNOWN",
            "source_hypothesis_is_not_merge_label":True,
        }
        require(original_map[(a,b)]==expected,
                "SOURCE_LIFTED_UV_WORLD_RELATION_FABRICATION")
        sc["retrieval_candidates_total"]+=1
        if n:
            sc["same_vertex_pair_world_near_both_fit"]+=1
            sc["world_fit_near_with_coincident_uv"]+=int(same_uv>0)
            sc["world_fit_near_without_coincident_uv"]+=int(same_uv==0)
    require(dict(sc)==original["source_lifted_relation_census"],
            "SOURCE_LIFTED_CENSUS_DRIFT")
    require(original["schema"] == SOURCE_SCHEMA and
            original["status"] ==
                "SOURCE_UV_WORLD_MECHANICAL_RELATIONS_OBSERVED__OWNER_UNKNOWN"
            and original["original_author_drawable_ownership"]=="UNKNOWN"
            and original["source_material_datablock_identification"] ==
                "UNKNOWN_PER_MESH_SLOT_ZERO_NOT_IDENTITY"
            and original["source_texture_rgba_continuity_across_uv"] ==
                "UNKNOWN_NO_IMAGE_SAMPLE_COURT"
            and original["claimed_IRIS_training_target"] is False
            and original["independent_court_required"] is True
            and original["parent_source_byte_sha256"]==pins
            and original["same_drawable_training_pairs_qualified"]==0
            and original["different_drawable_training_pairs_qualified"]==0
            and all(original.get(k) is False for k in NO_AUTHORITY_KEYS),
            "UNAUTHORIZED_STUDENT_OR_SOURCE_OWNER_TRUTH")
    output={
        "schema":SCHEMA,"status":PASS,
        "original_source_sha256":source,
        "independently_replayed_source_v18s_sha256":sha(report_path),
        "independent_parent_byte_sha256":pins,
        "source_original_face_count":len(original_faces),
        "source_original_island_count":len(groups),
        "source_local_evidence_recount":dict(lc),
        "source_lifted_evidence_recount":dict(sc),
        "heldout_pose2_not_consulted":True,
        "source_artist_drawable_owner":"UNKNOWN",
        "source_actual_texture_appearance_continuity":"UNKNOWN",
        **{k:False for k in NO_AUTHORITY_KEYS},
    }
    dest=Path(court_path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(output,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":PASS, "independent_local_edges":len(local_map),
        "independent_lifted":len(lifted),
        "count_near_both_fit":sc.get("same_vertex_pair_world_near_both_fit",0),
        "count_uv_coincident":sc.get("world_fit_near_with_coincident_uv",0),
        "training_authorized":False,
    },sort_keys=True))
    return output


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs",type=Path,required=True)
    ap.add_argument("--pins",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    verify_source(root=a.inputs,pins=a.pins,
                  report_path=a.report,court_path=a.output)
