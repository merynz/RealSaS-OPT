from __future__ import annotations

"""Rebuild Mage FIT2 view meshes where topology and W share one canonical anchor.

The historical/current P1 raster vertices are used only as view-space sampling
witnesses.  Their topology and their old support bindings are *not* reused.
Each raster sample is remapped into one safe relation-supported GSA triangle,
receives barycentric canonical support there, and is triangulated only with samples
from that same parent.  The fresh FIT2 skin is then derived from those exact support
coordinates.

This is the causal treatment for the false-mechanical-adjacency failure exposed by
the dynamic proof.  It performs no training, no model query, no historical weight
transfer, and claims no PRODUCT PASS by itself.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, QhullError

from compiler.realsas_compiler_core.mesh.anchor_unified import (
    ANCHOR_UNIFIED_TOPOLOGY_METHOD,
    require_anchor_unified_topology,
)
from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_candidate_lineage_hash,
    qualify_supported_mesh,
)
from compiler.realsas_compiler_core.mesh.mwb2 import build_mwb2_candidate
from compiler.realsas_compiler_core.mesh.mwb2_skin import bind_mwb2_mesh_skin
from compiler.realsas_compiler_core.mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
)
from compiler.realsas_compiler_core.types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    QualificationError,
    SurfaceSupportBinding,
)
from compiler.realsas_compiler_core.v4 import build_mechanical_state

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_demo_fit1_v5_p1.materialize_p1_b2_g10_v1 as p1base
import experiments.mage_demo_fit1_v5_p1.run_v5_direct_p1_binding_v1 as v5base
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2


SCHEMA = "RealSaS.MageFIT2.AnchorUnifiedP1.v1"
_BIN = 32
_BARY_EPS = 1.0e-7
_BINDING_EPS = 1.0e-10
_POINT_EPS = 1.0e-8


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"EXPECTED_JSON_OBJECT:{path}")
    return value


def _barycentric(point, tri):
    px, py = map(float, point)
    (ax, ay), (bx, by), (cx, cy) = tri
    v0x, v0y = bx - ax, by - ay
    v1x, v1y = cx - ax, cy - ay
    v2x, v2y = px - ax, py - ay
    denom = v0x * v1y - v1x * v0y
    if abs(denom) <= 1.0e-14:
        return None
    wb = (v2x * v1y - v1x * v2y) / denom
    wc = (v0x * v2y - v2x * v0y) / denom
    wa = 1.0 - wb - wc
    raw = [float(wa), float(wb), float(wc)]
    if min(raw) < -_BARY_EPS or max(raw) > 1.0 + _BARY_EPS:
        return None
    clipped = [0.0 if abs(w) <= _BINDING_EPS else max(0.0, w) for w in raw]
    total = float(sum(clipped))
    if total <= 0.0:
        return None
    return tuple(float(w / total) for w in clipped)


def _binding(parent_surface_ids, weights, *, view: int):
    acc: dict[str, float] = {}
    for sid, weight in zip(parent_surface_ids, weights):
        if float(weight) <= _BINDING_EPS:
            continue
        acc[str(sid)] = acc.get(str(sid), 0.0) + float(weight)
    total = float(sum(acc.values()))
    if total <= 0.0:
        raise QualificationError("ANCHOR_UNIFIED_BINDING_EMPTY")
    coeff = tuple((sid, value / total) for sid, value in sorted(acc.items()))
    if len(coeff) == 1:
        return SurfaceSupportBinding(
            "IDENTITY_SURFACE_NODE",
            ((coeff[0][0], 1.0),),
            metadata={"observed_view": int(view), "anchor_unified": True},
        )
    return SurfaceSupportBinding(
        "LOCAL_CONVEX_INTERPOLATION",
        coeff,
        metadata={"observed_view": int(view), "anchor_unified": True},
    )


def _binding_key(binding):
    return tuple((str(sid), round(float(weight), 13)) for sid, weight in binding.coefficients if float(weight) > _BINDING_EPS)


def _derive(surface_nodes, surface_raster, binding):
    P = np.zeros(3, dtype=np.float64)
    xy = np.zeros(2, dtype=np.float64)
    for sid, coefficient in binding.coefficients:
        c = float(coefficient)
        P += c * np.asarray(surface_nodes[str(sid)].P, dtype=np.float64)
        xy += c * np.asarray(surface_raster[str(sid)], dtype=np.float64)
    return tuple(map(float, P)), tuple(map(float, xy))


def _safe_parents(surface, *, view: int, camera_hash: str):
    base = build_mwb2_candidate(surface, view_index=int(view), camera_binding_hash=str(camera_hash))
    sid_by_candidate = {}
    xy_by_candidate = {}
    for vertex in base.vertices:
        coeff = tuple(vertex.support_binding.coefficients)
        if vertex.support_binding.mode != "IDENTITY_SURFACE_NODE" or len(coeff) != 1:
            raise RuntimeError("ANCHOR_UNIFIED_BASE_PARENT_NOT_IDENTITY")
        sid_by_candidate[str(vertex.candidate_vertex_id)] = str(coeff[0][0])
        xy_by_candidate[str(vertex.candidate_vertex_id)] = tuple(map(float, vertex.metadata["raster_xy"]))
    parents = []
    for index, face in enumerate(base.faces):
        ids = tuple(sid_by_candidate[str(v)] for v in face)
        xy = tuple(xy_by_candidate[str(v)] for v in face)
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        parents.append({
            "index": int(index),
            "surface_ids": ids,
            "xy": xy,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
        })
    return tuple(parents)


def _parent_bins(parents):
    bins = {}
    for parent in parents:
        xmin, ymin, xmax, ymax = parent["bbox"]
        for by in range(int(math.floor(ymin / _BIN)), int(math.floor(ymax / _BIN)) + 1):
            for bx in range(int(math.floor(xmin / _BIN)), int(math.floor(xmax / _BIN)) + 1):
                bins.setdefault((bx, by), []).append(int(parent["index"]))
    return {key: tuple(sorted(value)) for key, value in bins.items()}


def _map_to_parent(point, *, parents, bins):
    x, y = map(float, point)
    bx, by = int(math.floor(x / _BIN)), int(math.floor(y / _BIN))
    candidate_ids = set(bins.get((bx, by), ()))
    if not candidate_ids:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidate_ids.update(bins.get((bx + dx, by + dy), ()))
    best = None
    best_min = -float("inf")
    for parent_index in sorted(candidate_ids):
        parent = parents[int(parent_index)]
        xmin, ymin, xmax, ymax = parent["bbox"]
        if x < xmin - _POINT_EPS or x > xmax + _POINT_EPS or y < ymin - _POINT_EPS or y > ymax + _POINT_EPS:
            continue
        weights = _barycentric((x, y), parent["xy"])
        if weights is None:
            continue
        score = min(weights)
        if best is None or score > best_min + 1.0e-15 or (
            abs(score - best_min) <= 1.0e-15 and int(parent_index) < int(best[0])
        ):
            best = (int(parent_index), weights)
            best_min = score
    return best


def _build_candidate_from_p1_samples(surface, p1_mesh, domain, *, view: int, camera_hash: str):
    parents = _safe_parents(surface, view=int(view), camera_hash=str(camera_hash))
    if not parents:
        raise QualificationError(f"ANCHOR_UNIFIED_NO_SAFE_PARENT_V{view}")
    bins = _parent_bins(parents)
    surface_nodes = {str(node.surface_id): node for node in surface.surface_nodes}
    surface_raster = {}
    for sid, node in surface_nodes.items():
        rows = [tuple(map(float, xy)) for v, xy in node.raster_bindings if int(v) == int(view)]
        if len(rows) > 1:
            raise QualificationError(f"ANCHOR_UNIFIED_DUPLICATE_RASTER_BINDING:V{view}:{sid}")
        if rows:
            surface_raster[sid] = rows[0]

    vertex_payload: dict[tuple, dict] = {}
    parent_vertex_keys: dict[int, set[tuple]] = {int(parent["index"]): set() for parent in parents}
    mapped = 0
    unmapped = 0
    max_raster_replay_error = 0.0

    def admit(binding, *, parent_index: int, source_vertex_id: str, requested_xy):
        nonlocal max_raster_replay_error
        P, xy = _derive(surface_nodes, surface_raster, binding)
        error = float(np.linalg.norm(np.asarray(xy) - np.asarray(requested_xy, dtype=np.float64)))
        max_raster_replay_error = max(max_raster_replay_error, error)
        if error > 1.0e-5:
            raise QualificationError(f"ANCHOR_UNIFIED_RASTER_REPLAY_DRIFT:V{view}:{error}")
        key = _binding_key(binding)
        if key not in vertex_payload:
            vertex_payload[key] = {
                "binding": binding,
                "P": P,
                "xy": xy,
                "source_vertex_ids": [str(source_vertex_id)] if source_vertex_id else [],
            }
        elif source_vertex_id:
            vertex_payload[key]["source_vertex_ids"].append(str(source_vertex_id))
        parent_vertex_keys[int(parent_index)].add(key)
        return key

    # Re-anchor current P1 raster samples; old P1 topology/support are provenance only.
    for vertex in p1_mesh.vertices:
        raw = dict(vertex.metadata or {}).get("raster_xy")
        if raw is None or len(tuple(raw)) != 2:
            raise QualificationError(f"ANCHOR_UNIFIED_P1_RASTER_WITNESS_MISSING:V{view}")
        xy = tuple(map(float, raw))
        hit = _map_to_parent(xy, parents=parents, bins=bins)
        if hit is None:
            unmapped += 1
            continue
        parent_index, weights = hit
        parent = parents[int(parent_index)]
        binding = _binding(parent["surface_ids"], weights, view=int(view))
        admit(
            binding,
            parent_index=int(parent_index),
            source_vertex_id=str(vertex.canonical_mesh_vertex_id),
            requested_xy=xy,
        )
        mapped += 1

    # Every active parent receives its exact canonical corners so local triangulation
    # cannot invent a cross-parent boundary diagonal.
    active_parents = [parent for parent in parents if parent_vertex_keys[int(parent["index"])]]
    for parent in active_parents:
        for sid, xy in zip(parent["surface_ids"], parent["xy"]):
            binding = SurfaceSupportBinding(
                "IDENTITY_SURFACE_NODE",
                ((str(sid), 1.0),),
                metadata={"observed_view": int(view), "anchor_unified": True, "parent_corner": True},
            )
            admit(binding, parent_index=int(parent["index"]), source_vertex_id="", requested_xy=xy)

    ordered_keys = sorted(vertex_payload)
    candidate_id = {key: f"AUP1:V{view}:{i:06d}" for i, key in enumerate(ordered_keys)}
    vertices = tuple(
        MeshVertexCandidate(
            candidate_id[key],
            tuple(vertex_payload[key]["P"]),
            vertex_payload[key]["binding"],
            metadata={
                "raster_xy": tuple(vertex_payload[key]["xy"]),
                "source_p1_vertex_ids": tuple(sorted(set(vertex_payload[key]["source_vertex_ids"]))),
                "anchor_unified": True,
                "source_p1_topology_used": False,
                "source_p1_support_binding_used": False,
            },
        )
        for key in ordered_keys
    )

    face_set = set()
    qhull_fail = 0
    alpha_reject = 0
    degenerate_reject = 0
    for parent in active_parents:
        keys = sorted(parent_vertex_keys[int(parent["index"])])
        if len(keys) < 3:
            continue
        coords = np.asarray([vertex_payload[key]["xy"] for key in keys], dtype=np.float64)
        if np.linalg.matrix_rank(coords - coords.mean(axis=0, keepdims=True)) < 2:
            continue
        try:
            tri = Delaunay(coords, qhull_options="Qbb Qc Qz Q12")
        except QhullError:
            qhull_fail += 1
            continue
        for simplex in np.asarray(tri.simplices, dtype=np.int64):
            if len(simplex) != 3 or np.any(simplex < 0) or np.any(simplex >= len(keys)):
                continue
            local_keys = tuple(keys[int(i)] for i in simplex.tolist())
            ids = tuple(candidate_id[key] for key in local_keys)
            if len(set(ids)) != 3:
                degenerate_reject += 1
                continue
            pts = tuple(vertex_payload[key]["xy"] for key in local_keys)
            area2 = (
                (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
                - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0])
            )
            if abs(float(area2)) <= 1.0e-12:
                degenerate_reject += 1
                continue
            face = ids if area2 > 0.0 else (ids[0], ids[2], ids[1])
            if not domain.triangle_inside(pts):
                alpha_reject += 1
                continue
            face_set.add(tuple(face))

    if not face_set:
        raise QualificationError(f"ANCHOR_UNIFIED_NO_ADMITTED_FACE_V{view}")
    faces = tuple(sorted(face_set))
    used = {vertex_id for face in faces for vertex_id in face}
    vertices = tuple(vertex for vertex in vertices if vertex.candidate_vertex_id in used)
    edge_set = set()
    for a, b, c in faces:
        edge_set.update({tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))})
    edges = tuple(sorted(edge_set))

    xy_by_id = {vertex.candidate_vertex_id: tuple(map(float, vertex.metadata["raster_xy"])) for vertex in vertices}
    coverage = domain.coverage(tuple(tuple(xy_by_id[v] for v in face) for face in faces))
    residual = {
        **coverage,
        "source_p1_vertex_count": len(p1_mesh.vertices),
        "mapped_source_p1_vertex_count": int(mapped),
        "unmapped_source_p1_vertex_count": int(unmapped),
        "mapped_source_p1_fraction": float(mapped / max(len(p1_mesh.vertices), 1)),
        "safe_parent_triangle_count": len(parents),
        "active_parent_triangle_count": len(active_parents),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "edge_count": len(edges),
        "qhull_parent_failure_count": int(qhull_fail),
        "alpha_rejected_face_count": int(alpha_reject),
        "degenerate_rejected_face_count": int(degenerate_reject),
        "max_raster_replay_error_px": float(max_raster_replay_error),
    }
    value = MeshDiscretizationCandidateIR(
        vertices=vertices,
        faces=faces,
        edges=edges,
        surface_binding_hash=str(surface.geometry_lineage_hash),
        view_index=int(view),
        camera_binding_hash=str(camera_hash),
        candidate_lineage_hash="",
        boundary_constraints=tuple(p1_mesh.boundary_constraints),
        coverage_classification="OBSERVATION_DOMAIN_GSA_VIEW_COMMON_ANCHOR_V1",
        solver_provenance={
            "solver": "GSA_SAFE_PARENT_LOCAL_DELAUNAY_FROM_P1_RASTER_SAMPLES_V1",
            "topology_authority": "SAFE_GSA_RELATION_TRIANGLES",
            "view_sample_authority": "CURRENT_P1_RASTER_WITNESS_ONLY",
            "source_p1_topology_used": False,
            "source_p1_support_binding_used": False,
            "source_mesh_used": False,
            "teacher_topology_used": False,
            "historical_skin_used": False,
        },
        residual_report=residual,
        metadata={
            "producer": SCHEMA,
            "anchor_unified_topology_method": ANCHOR_UNIFIED_TOPOLOGY_METHOD,
            "surface_lineage_hash": str(surface.geometry_lineage_hash),
            "source_p1_mesh_lineage_hash": str(p1_mesh.mesh_lineage_hash),
            "source_p1_topology_used": False,
            "source_p1_support_binding_used": False,
            "source_mesh_used": False,
            "target_view_winding": "CCW",
        },
    )
    return replace(value, candidate_lineage_hash=mesh_candidate_lineage_hash(value))


def run(args) -> dict:
    p1_dir = Path(args.p1_dir)
    skeleton_path = Path(args.skeleton)
    skin_path = Path(args.fit2_skin)
    handoff_path = Path(args.fit2_handoff)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p1_manifest_path = p1_dir / "P1_B2_G10_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
    if _sha(p1_manifest_path) != str(args.expected_p1_manifest):
        raise RuntimeError("ANCHOR_UNIFIED_P1_MANIFEST_SHA_DRIFT")
    p1_manifest = _load_json(p1_manifest_path)
    if p1_manifest.get("current_canonical_gsa_lineage_hash") != fit2io.EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("ANCHOR_UNIFIED_P1_SURFACE_LINEAGE_DRIFT")

    surface, tensor, replay = ceiling_v2._preflight_surface(args)
    if not replay.get("gsa_lineage_exact_match"):
        raise RuntimeError("ANCHOR_UNIFIED_CURRENT_GSA_REPLAY_FAIL")
    if surface.geometry_lineage_hash != fit2io.EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("ANCHOR_UNIFIED_CURRENT_GSA_LINEAGE_DRIFT")
    if int(tensor.node_count) != fit2io.EXPECTED_SURFACE_NODES or int(tensor.edge_count) != fit2io.EXPECTED_SURFACE_RELATIONS:
        raise RuntimeError("ANCHOR_UNIFIED_GSA_CARDINALITY_DRIFT")

    skeleton = fit2io.load_skeleton(skeleton_path)
    skin = fit2io.load_skin(skin_path)
    build_mechanical_state(surface, skeleton, skin)
    handoff = _load_json(handoff_path)
    if handoff.get("status") != "AUTHORIZED__FIT2_ARACHNE_COMPILER_QUALIFIED" or handoff.get("skin_lineage") != fit2io.EXPECTED_SKIN_LINEAGE:
        raise RuntimeError("ANCHOR_UNIFIED_ARACHNE_HANDOFF_INVALID")

    obs = tuple(Path(p) for p in args.observations)
    cams = tuple(Path(p) for p in args.cameras)
    _baselines, domains = p1base._baseline_inputs(surface, obs, cams)
    p1_rows = {int(row["view"]): row for row in p1_manifest.get("views", ())}
    if set(p1_rows) != set(range(8)):
        raise RuntimeError("ANCHOR_UNIFIED_P1_VIEW_SET_INVALID")

    rows = []
    all_pass = True
    for view in range(8):
        old = p1_rows[view]
        p1_mesh = v5base._load_mesh(
            p1_dir / f"P1_B2_G10_CURRENT_V{view}_QUALIFIED_MESH_IR.json",
            surface,
            str(old["current_mesh_lineage_hash"]),
        )
        candidate = _build_candidate_from_p1_samples(
            surface,
            p1_mesh,
            domains[view],
            view=int(view),
            camera_hash=ceiling_v1.CAMERA_SHA256[view],
        )
        mesh = qualify_supported_mesh(surface, candidate)
        anchor = require_anchor_unified_topology(surface, mesh)
        mesh_skin = bind_mwb2_mesh_skin(
            surface,
            skeleton,
            skin,
            mesh,
            require_anchor_unified_topology=True,
        )
        quality = evaluate_mesh_quality(
            coverage=dict(candidate.residual_report),
            raster_report=mesh_raster_quality_report(mesh, surface=surface, view_index=view),
            policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
        )
        passed = bool(anchor["passed"] and quality["passed"])
        all_pass = all_pass and passed

        mesh_name = f"ANCHOR_UNIFIED_V{view}_QUALIFIED_MESH_IR.json"
        skin_name = f"ANCHOR_UNIFIED_V{view}_QUALIFIED_MESH_SKIN_IR.json"
        _write_json(output_dir / mesh_name, mesh.to_dict())
        _write_json(output_dir / skin_name, mesh_skin.to_dict())
        row = {
            "view": int(view),
            "passed": bool(passed),
            "source_p1_mesh_lineage_hash": str(p1_mesh.mesh_lineage_hash),
            "anchor_unified_mesh_lineage_hash": str(mesh.mesh_lineage_hash),
            "anchor_unified_mesh_skin_lineage_hash": str(mesh_skin.mesh_skin_lineage_hash),
            "anchor_topology": anchor,
            "mesh_quality_passed": bool(quality["passed"]),
            "mesh_quality_failure_invariants": list(quality["failure_invariants"]),
            "source_alpha_recall": float(quality["source_alpha_recall"]),
            "precision_inside_alpha": float(quality["precision_inside_alpha"]),
            "alpha_iou": float(quality["alpha_iou"]),
            "mapped_source_p1_fraction": float(candidate.residual_report["mapped_source_p1_fraction"]),
            "unmapped_source_p1_vertex_count": int(candidate.residual_report["unmapped_source_p1_vertex_count"]),
            "safe_parent_triangle_count": int(candidate.residual_report["safe_parent_triangle_count"]),
            "active_parent_triangle_count": int(candidate.residual_report["active_parent_triangle_count"]),
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "edge_count": len(mesh.edges),
            "mesh_file": mesh_name,
            "mesh_file_sha256": _sha(output_dir / mesh_name),
            "mesh_skin_file": skin_name,
            "mesh_skin_file_sha256": _sha(output_dir / skin_name),
        }
        rows.append(row)
        print("ANCHOR_UNIFIED_V" + str(view) + "=" + json.dumps({
            "pass": passed,
            "mapped": row["mapped_source_p1_fraction"],
            "recall": row["source_alpha_recall"],
            "precision": row["precision_inside_alpha"],
            "faces": row["face_count"],
            "anchor_fail_faces": anchor["face_without_common_parent_count"],
        }, sort_keys=True), flush=True)

    manifest = {
        "schema": SCHEMA,
        "status": (
            "PASS__FIT2_GSA_VIEW_COMMON_ANCHOR_MESH_AND_SKIN_V0_V7"
            if all_pass
            else "FAIL__FIT2_GSA_VIEW_COMMON_ANCHOR_NOT_PRODUCT_READY"
        ),
        "source_commit_expected_parent": "417a67cf9254bfbb1c8fb392a773825e626b00c6",
        "source_p1_manifest_sha256": _sha(p1_manifest_path),
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "anchor_unified_topology_method": ANCHOR_UNIFIED_TOPOLOGY_METHOD,
        "view_raster_and_gsa_meet_before_topology": True,
        "topology_and_w_derived_from_same_canonical_support": True,
        "source_p1_raster_samples_used": True,
        "source_p1_topology_used": False,
        "source_p1_support_binding_used": False,
        "historical_fit1_skin_used": False,
        "model_query_performed": False,
        "training_performed": False,
        "product_pass_claimed": False,
        "views": rows,
    }
    manifest_path = output_dir / "FIT2_ANCHOR_UNIFIED_P1_MANIFEST.json"
    _write_json(manifest_path, manifest)
    print("FIT2_ANCHOR_UNIFIED_P1=" + json.dumps({
        "status": manifest["status"],
        "manifest_sha256": _sha(manifest_path),
        "all_views_pass": bool(all_pass),
    }, sort_keys=True), flush=True)
    return manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--p1-dir", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit2-skin", required=True)
    p.add_argument("--fit2-handoff", required=True)
    p.add_argument("--expected-p1-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
