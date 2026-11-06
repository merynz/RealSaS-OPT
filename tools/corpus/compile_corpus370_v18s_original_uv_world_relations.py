#!/usr/bin/env python3
"""V18s original source UV/world/mechanics relation evidence, NOT drawable labels.

Joins three *independently archived* source identities:
V18a mechanical source face graph, V18q/r typed graph and original 3D poses,
and SHA-qualified Blender-native original glTF face-corner UV. The 684-face
Cat source is a real-world experiment, not a hardcoded subject rule.

No source author ownership, material datablock identity, same/different
drawable training pair, student inference, candidate ranking or training
admission is ever produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "RealSaS.OriginalSourceUVWorldMechanicalRelationEvidence.v18s"
STATUS = "SOURCE_UV_WORLD_MECHANICAL_RELATIONS_OBSERVED__OWNER_UNKNOWN"
UV_TOL = 1e-6  # normalized glTF top-left coordinate
WORLD_GAP_TOL = 0.005  # original V18r frozen source extent fraction
NO_AUTHORITY = {
    "graphics_owner_authorized": False,
    "iris_training_authorized": False,
    "tessa_training_authorized": False,
    "training_authorized": False,
    "materialization_authorized": False,
    "same_drawable_training_pairs_qualified": 0,
    "different_drawable_training_pairs_qualified": 0,
}


def require(condition, code):
    if not condition:
        raise ValueError("V18S_SOURCE_RELATION_" + code)


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _original_uv_corner_vertex_map(faces, face_uv):
    """Never force a single UV per mesh vertex when a genuine seam exists."""
    vertex_uv = {}
    face_vertex_uv = {}
    for face_idx, vertices in enumerate(faces):
        for cidx, vid in enumerate(vertices):
            key = (int(face_idx), int(vid))
            uv = tuple(map(float, face_uv[face_idx, cidx]))
            face_vertex_uv[key] = uv
            vertex_uv.setdefault(int(vid), set()).add(uv)
    return vertex_uv, face_vertex_uv


def measure_relations(*, faces, face_uv, baseline_vertices, island_for_face,
                      local_rows, lifted_rows, original_source_hash):
    faces = np.asarray(faces, dtype=np.int64)
    face_uv = np.asarray(face_uv, dtype=np.float64)
    xyz = np.asarray(baseline_vertices, dtype=np.float64)
    island = np.asarray(island_for_face, dtype=np.int64)
    require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
            and face_uv.shape == (len(faces), 3, 2)
            and np.isfinite(face_uv).all()
            and np.min(face_uv) >= -1e-6 and np.max(face_uv) <= 1 + 1e-6,
            "UV_FACE_CORNER_DOMAIN")
    require(xyz.ndim == 3 and xyz.shape[0] >= 3 and xyz.shape[-1] == 3
            and np.isfinite(xyz).all()
            and np.min(faces) >= 0 and np.max(faces) < xyz.shape[1]
            and island.shape == (len(faces),) and np.min(island) >= 0,
            "POSED_SOURCE_GEOMETRY_FACE_BINDING")
    require(isinstance(original_source_hash, str) and
            len(original_source_hash) == 64,
            "SOURCE_SHA_REQUIRED")
    require(len(local_rows) > 0 and len(lifted_rows) >= 0,
            "SOURCE_GRAPH_RELATIONS_MISSING")
    per_vertex, face_vertex_uv = _original_uv_corner_vertex_map(faces, face_uv)
    source_vertex_uv_ambiguity = sum(len(values) > 1 for values in per_vertex.values())
    local = []
    local_keys = set()
    for evidence in local_rows:
        a = int(evidence["face_a_source_teacher_only"])
        b = int(evidence["face_b_source_teacher_only"])
        key = (min(a, b), max(a, b))
        require(a != b and 0 <= a < len(faces) and 0 <= b < len(faces)
                and key not in local_keys,
                "DUPLICATE_OR_INVALID_LOCAL_FACE_RELATION")
        local_keys.add(key)
        shared = sorted(set(map(int, faces[a])) & set(map(int, faces[b])))
        require(len(shared) == 2, "SOURCE_LOCAL_EDGE_NOT_SHARED_BY_TWO_VERTICES")
        corner_gaps = [
            float(np.linalg.norm(
                np.asarray(face_vertex_uv[(a, v)]) -
                np.asarray(face_vertex_uv[(b, v)])))
            for v in shared
        ]
        continuous = max(corner_gaps) <= UV_TOL
        local.append({
            "original_face_a": a,
            "original_face_b": b,
            "original_source_shared_vertex_ids": shared,
            "source_shared_edge_uv_endpoint_delta_max": max(corner_gaps),
            "source_uv_address_continuity": (
                "SAME_ORIGINAL_UV_ENDPOINT_COORDINATES"
                if continuous else "DISCONTINUOUS_ORIGINAL_UV_ENDPOINT_COORDINATES"
            ),
            "source_mechanical_axis_divergent":
                bool(evidence["source_mechanical_any_axis_divergent"]),
            "source_mechanical_pooled_divergent":
                bool(evidence["source_mechanical_pooled_divergent"]),
            "source_local_normal_crease_stable_fit":
                bool(evidence["fit_consistent_geometric_normal_crease_30deg"]),
            "same_drawable": "UNKNOWN",
            "different_drawable": "UNKNOWN",
            "true_source_material_datablock_identity": "UNKNOWN",
        })

    unique_islands = set(map(int, island.tolist()))
    by_island = {
        k: np.unique(faces[island == k].reshape(-1))
        for k in unique_islands
    }
    extent = float(np.linalg.norm(np.ptp(xyz[0], axis=0)))
    require(np.isfinite(extent) and extent > 1e-9, "SOURCE_FIT_WORLD_EXTENT_ZERO")
    lifted = []
    lifted_keys = set()
    for evidence in lifted_rows:
        a = int(evidence["source_island_a_teacher_only"])
        b = int(evidence["source_island_b_teacher_only"])
        key = (min(a, b), max(a, b))
        require(a != b and a in by_island and b in by_island and
                key not in lifted_keys,
                "DUPLICATE_OR_INVALID_LIFTED_ISLAND_RELATION")
        lifted_keys.add(key)
        va, vb = by_island[a], by_island[b]
        # Same original vertex-pair identity must be near on BOTH fit poses.
        # No original heldout pose/paint/edit target contributes to selection.
        p0 = np.linalg.norm(
            xyz[0, va][:, None, :] - xyz[0, vb][None, :, :], axis=-1)
        p1 = np.linalg.norm(
            xyz[1, va][:, None, :] - xyz[1, vb][None, :, :], axis=-1)
        require(bool(np.min(p0) / extent <= WORLD_GAP_TOL) ==
                bool(evidence["source_fit_both_pose_min_vertex_gap_le_0p005"])
                or np.min(p1) / extent > WORLD_GAP_TOL,
                "V18R_FIT_SOURCE_GAP_JOINT_PROVENANCE_DRIFT")
        joint = np.argwhere(
            (p0 <= WORLD_GAP_TOL * extent) &
            (p1 <= WORLD_GAP_TOL * extent))
        uv_coincident = 0
        uv_disagree = 0
        # UV across vertex-disjoint source islands can have multiple
        # loop-corner values. Compare all exact original loop-corner UV
        # values rather than inventing a unique per-vertex material chart.
        min_gap = None
        for ai, bi in joint:
            aa = int(va[ai])
            bb = int(vb[bi])
            gaps = [
                float(np.linalg.norm(np.asarray(u) - np.asarray(v)))
                for u in per_vertex[aa] for v in per_vertex[bb]
            ]
            gap = min(gaps)
            min_gap = gap if min_gap is None else min(min_gap, gap)
            if gap <= UV_TOL:
                uv_coincident += 1
            else:
                uv_disagree += 1
        near_any = bool(len(joint))
        lifted.append({
            "source_island_a": a,
            "source_island_b": b,
            "original_fit_pose_same_pair_near_vertex_count": len(joint),
            "original_fit_pose_world_near": near_any,
            "world_fit_near_original_extent_threshold": WORLD_GAP_TOL,
            "original_near_pairs_same_uv_endpoint_count": uv_coincident,
            "original_near_pairs_different_uv_endpoint_count": uv_disagree,
            "original_near_pair_minimum_uv_gap":
                min_gap if near_any else None,
            "source_world_uv_joint_relation": (
                "WORLD_FIT_NEAR_AND_AT_LEAST_ONE_UV_ENDPOINT_COINCIDENT"
                if uv_coincident > 0 else
                "WORLD_FIT_NEAR_BUT_NO_UV_ENDPOINT_COINCIDENT"
                if near_any else "NOT_NEAR_IN_SAME_FIT_VERTEX_PAIR"
            ),
            "source_artistic_owner_merge": "UNKNOWN",
            "source_material_datablock_identity": "UNKNOWN",
            "source_actual_visual_appearance_continuity": "UNKNOWN",
            "source_hypothesis_is_not_merge_label": True,
        })
    require(len(local_keys) == len(local_rows) and
            len(lifted_keys) == len(lifted_rows), "SOURCE_RELATION_DUPLICATE")
    lc = Counter()
    for x in local:
        lc["all"] += 1
        lc["uv_continuous" if x["source_uv_address_continuity"].startswith("SAME")
           else "uv_endpoint_discontinuous"] += 1
        if x["source_mechanical_axis_divergent"]:
            lc["at_least_one_mechanical_axis_divergent"] += 1
            if x["source_uv_address_continuity"].startswith("SAME"):
                lc["axis_divergent_but_uv_continuous"] += 1
        if x["source_mechanical_pooled_divergent"]:
            lc["pooled_mechanical_divergent"] += 1
    sc = Counter()
    for x in lifted:
        sc["retrieval_candidates_total"] += 1
        if x["original_fit_pose_world_near"]:
            sc["same_vertex_pair_world_near_both_fit"] += 1
            sc["world_fit_near_with_coincident_uv"] += int(
                x["original_near_pairs_same_uv_endpoint_count"] > 0)
            sc["world_fit_near_without_coincident_uv"] += int(
                x["original_near_pairs_same_uv_endpoint_count"] == 0)
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "original_source_sha256": original_source_hash,
        "source_face_count": len(faces),
        "source_vertex_count": xyz.shape[1],
        "source_graph_island_count": len(unique_islands),
        "source_uv_corner_origin": "GLTF_IMAGE_TOP_LEFT",
        "source_fit_pose_ordinals": [0, 1],
        "source_heldout_pose_ordinal_used_for_fit": False,
        "fit_near_vertex_pair_threshold_per_extent": WORLD_GAP_TOL,
        "uv_endpoint_coordinate_threshold": UV_TOL,
        "source_vertex_multiple_original_uv_addresses": source_vertex_uv_ambiguity,
        "source_local_face_edge_evidence_private": local,
        "source_lifted_fit_world_uv_evidence_private": lifted,
        "source_local_relation_census": dict(lc),
        "source_lifted_relation_census": dict(sc),
        "source_material_datablock_identification": "UNKNOWN_PER_MESH_SLOT_ZERO_NOT_IDENTITY",
        "source_texture_rgba_continuity_across_uv": "UNKNOWN_NO_IMAGE_SAMPLE_COURT",
        "original_author_drawable_ownership": "UNKNOWN",
        "claimed_IRIS_training_target": False,
        **NO_AUTHORITY,
    }


FILES = {
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


def load_source(root, pins):
    root = Path(root)
    require(set(pins) == set(FILES) and
            all(isinstance(value, str) and len(value) == 64
                for value in pins.values()),
            "PIN_SET_INCOMPLETE")
    for role, filename in FILES.items():
        require(digest(root / filename) == pins[role],
                "PARENT_BYTES_SHA:" + role)
    j = lambda role: json.loads((root / FILES[role]).read_text())
    v18r, court, v18q, qcourt, align = (
        j("v18r"), j("v18r_court"), j("v18q"),
        j("v18q_court"), j("uv_binding"))
    src = v18r["original_source_sha256"]
    require(src == v18q["original_source_sha256"] ==
            court["original_source_sha256"] == align["original_glb_sha256"]
            and v18r["status"] ==
            "SOURCE_POSE_GEOMETRY_BOUND_TO_TYPED_GRAPH__NO_GRAPHIC_OWNER_PROOF"
            and court["status"] ==
            "SOURCE_GEOMETRY_SPATIAL_AND_CREASE_COURTED__OWNER_UNKNOWN"
            and align["source_to_teacher_face_uv_binding"] ==
            "VERIFIED_NATIVE_BLENDER_FACE_ORDER_AND_RAW_GLTF_UV_MULTISET"
            and align["source_uv_comparison"]["matched"] is True
            and align["original_teacher_sha256"] == pins["teacher"]
            and v18r["parents_sha256"]["v18q_source_ir"] == pins["v18q"]
            and v18r["parents_sha256"]["v18q_independent_court"] ==
                pins["v18q_court"]
            and v18r["parents_sha256"]["v18a_original_arrays"] == pins["v18a"]
            and v18r["parents_sha256"]["v13b_actual_source_pose_vertices"] ==
                pins["v13b"],
            "SAME_SOURCE_AND_PARENT_COURT_BINDING_UNQUALIFIED")
    require(v18q["source_original_face_uv_material_relation"] ==
            "NOT_PRESENT_IN_SHA_BOUND_V18A_ARRAYS"
            and court["original_source_material_and_uv_boundary_not_measured"] is True
            and all(qcourt.get(k) is False for k in (
                "graphics_owner_authorized", "iris_training_authorized",
                "tessa_training_authorized", "training_authorized",
                "materialization_authorized"))
            and all(align.get(k) is False for k in (
                "graphics_owner_authorized", "iris_training_authorized",
                "tessa_training_authorized", "training_authorized",
                "materialization_authorized")),
            "SOURCE_EDITOR_OWNER_OR_TRAINING_PROMOTION")
    with np.load(root / FILES["v18a"], allow_pickle=False) as a, \
         np.load(root / FILES["v13b"], allow_pickle=False) as p, \
         np.load(root / FILES["teacher"], allow_pickle=False) as t, \
         np.load(root / FILES["uv"], allow_pickle=False) as u:
        faces = np.asarray(a["faces"], dtype=np.int64)
        graph_edges = np.asarray(a["graph_edges"], dtype=np.int64)
        baseline = np.asarray(p["baseline_vertices"], dtype=np.float64)
        teacher_faces = np.asarray(t["faces"], dtype=np.int64)
        uv = np.asarray(u["face_uv"], dtype=np.float64)
    require(np.array_equal(faces, teacher_faces) and
            faces.shape == (v18q["source_private_graph"]["source_total_faces"], 3)
            and v18r["original_source_world_evidence_teacher_private"][
                "source_total_faces"] == len(faces),
            "EXACT_FACE_ORDER_NOT_VERIFIED")
    expect_uv_hash = hashlib.sha256(
        np.ascontiguousarray(uv.astype("<f8")).tobytes()).hexdigest()
    require(expect_uv_hash == align["face_uv_array_sha256"],
            "ORIGINAL_UV_WITNESS_CONTENT_DRIFT")
    g = v18q["source_private_graph"]
    r = v18r["original_source_world_evidence_teacher_private"]
    local = r["source_local_geometry_edge_evidence_private"]
    lifted = r["source_lifted_world_proximity_evidence_private"]
    qlifted = g["source_nonlocal_graph_affinity_hypotheses_teacher_private"]
    source_edges = {tuple(sorted(map(int, e))) for e in graph_edges}
    graph_local_edges = {
        tuple(sorted((int(e["face_a_source_teacher_only"]),
                      int(e["face_b_source_teacher_only"]))))
        for e in local
    }
    locked_lifted = {
        tuple(sorted((int(e["topology_island_a_source_teacher_only"]),
                      int(e["topology_island_b_source_teacher_only"]))))
        for e in qlifted
    }
    r_lifted = {
        tuple(sorted((int(e["source_island_a_teacher_only"]),
                      int(e["source_island_b_teacher_only"]))))
        for e in lifted
    }
    require(source_edges == graph_local_edges and
            len(graph_local_edges) == len(local) == len(graph_edges)
            and locked_lifted == r_lifted and
            len(locked_lifted) == len(qlifted) == len(lifted),
            "V18A_V18Q_V18R_RELATION_KEY_DRIFT")
    require(r["source_fit_pose_ordinals"] == [0, 1] and
            r["source_original_heldout_pose_ordinal_COURT_ONLY"] == 2
            and r["heldout_pose_not_used_to_generate_candidates"] is True,
            "SOURCE_FIT_HELDOUT_BARRIER")
    return {
        "original_source_hash": src,
        "faces": faces,
        "face_uv": uv,
        "baseline_vertices": baseline,
        "island_for_face":
            g["source_face_observation_ir_teacher_private"][
                "source_topology_island_per_face_teacher_private"],
        "local_rows": local,
        "lifted_rows": lifted,
    }


def run(root, pins_path, output):
    pins = json.loads(Path(pins_path).read_text())
    params = load_source(root, pins)
    result = measure_relations(**params)
    result["parent_source_byte_sha256"] = pins
    result["source_face_independent_v18a_to_native_teacher_equal"] = True
    result["independent_court_required"] = True
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "source_face_count": result["source_face_count"],
        "local_relation_census": result["source_local_relation_census"],
        "lifted_relation_census": result["source_lifted_relation_census"],
        "artist_owner": "UNKNOWN",
        "iris_training_authorized": False,
    }, sort_keys=True))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--pins", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    run(a.inputs, a.pins, a.output)
