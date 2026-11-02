#!/usr/bin/env python3
from __future__ import annotations

"""Independent court for Corpus400 TESSA mechanical teacher V4.

The court independently reconstructs edge-connected surface components and proves
that the TESSA manifest is bound to that relation, not to sign/manifold components.
It also retains the source-rebind and selected mechanical-candidate checks from the
V3 court. Causal-response family qualification remains separate.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.training.truth139_v32_spatial_index_v1 import new_rtree3


PARENT_SCHEMA = "RealSaS.CorpusMechanicalTESSATeacher.v4"
PARENT_STATUS = "PASS_CORPUS_MECHANICAL_TESSA_TEACHER_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA = "RealSaS.CorpusMechanicalTESSATeacherCourt.v4"
STATUS = "PASS_CORPUS_MECHANICAL_TESSA_SURFACE_TOPOLOGY_CONSUMER_COURT__CAUSAL_RESPONSE_AUTHORITY_SEPARATE"
REQUIRED = (
    "mapping_pass",
    "component_pass",
    "topology_hole_crack_preservation_pass",
    "attachment_object_separation_pass",
    "nondegenerate_all_actions_pass",
    "inversion_foldthrough_pass",
    "self_intersection_pass",
    "response_pass",
    "strain_jacobian_tail_pass",
    "silhouette_pass",
    "mechanical_pass",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bound(row: dict | None, code: str) -> Path:
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    expected = str(row.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _edge_surface_components(faces: np.ndarray) -> np.ndarray:
    table = defaultdict(list)
    for fi, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            table[(min(int(a), int(b)), max(int(a), int(b)))].append(fi)
    graph = [set() for _ in range(len(faces))]
    for ids in table.values():
        unique = sorted(set(map(int, ids)))
        for index, left in enumerate(unique):
            for right in unique[index + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    labels = np.full(len(faces), -1, dtype=np.int32)
    cid = 0
    for start in range(len(faces)):
        if labels[start] >= 0:
            continue
        labels[start] = cid
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = cid
                    stack.append(neighbor)
        cid += 1
    return labels


def _partition(labels: np.ndarray) -> set[frozenset[int]]:
    labels = np.asarray(labels, dtype=np.int64)
    return {
        frozenset(map(int, np.flatnonzero(labels == cid)))
        for cid in sorted(set(map(int, labels.tolist())))
        if cid >= 0
    }


def _tri_intersect_sat(a, b, eps=1e-9):
    ea = np.asarray([a[1] - a[0], a[2] - a[1], a[0] - a[2]])
    eb = np.asarray([b[1] - b[0], b[2] - b[1], b[0] - b[2]])
    na = np.cross(ea[0], ea[1])
    nb = np.cross(eb[0], eb[1])
    axes = [na, nb]
    axes.extend(np.cross(x, y) for x in ea for y in eb)
    axes.extend(np.cross(na, x) for x in ea)
    axes.extend(np.cross(nb, y) for y in eb)
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= eps:
            continue
        unit = axis / norm
        pa = a @ unit
        pb = b @ unit
        if float(pa.max()) < float(pb.min()) - eps or float(pb.max()) < float(pa.min()) - eps:
            return False
    return True


def _rest_self_intersection(vertices, faces):
    tri = vertices[faces]
    index = new_rtree3()
    for fi, value in enumerate(tri):
        lo, hi = value.min(0), value.max(0)
        index.insert(fi, (float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2])))
    for fi, value in enumerate(tri):
        lo, hi = value.min(0), value.max(0)
        bounds = (float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2]))
        for other in index.intersection(bounds):
            if other <= fi or len(set(map(int, faces[fi])) & set(map(int, faces[other]))):
                continue
            if _tri_intersect_sat(value, tri[other]):
                return True
    return False


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("TESSA_V4_COURT_PARENT")
    parents = dict(manifest.get("parents") or {})
    gm_path = bound(parents.get("geometry_manifest"), "TESSA_V4_COURT_GEOMETRY_DRIFT")
    gc_path = bound(parents.get("geometry_court"), "TESSA_V4_COURT_GEOMETRY_COURT_DRIFT")
    gm = json.loads(gm_path.read_text())
    gc = json.loads(gc_path.read_text())
    if gm.get("schema") != "RealSaS.ComponentGeometryAuthority.v1":
        raise ValueError("TESSA_V4_COURT_GEOMETRY_SCHEMA")
    if gc.get("schema") != "RealSaS.ComponentGeometryAuthorityCourt.v1" or gc.get("status") != "PASS_COMPONENT_GEOMETRY_AUTHORITY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE":
        raise ValueError("TESSA_V4_COURT_GEOMETRY_COURT_STATUS")
    topology_path = bound((gm.get("outputs") or {}).get("topology"), "TESSA_V4_COURT_TOPOLOGY_DRIFT")
    mesh_path = bound(manifest.get("teacher_mesh"), "TESSA_V4_COURT_TEACHER_DRIFT")

    with np.load(topology_path, allow_pickle=False) as z:
        required = {
            "topology_vertices_source_world",
            "topology_faces",
            "face_valid",
            "surface_component_id",
            "sign_component_id",
        }
        if not required.issubset(z.files):
            raise ValueError("TESSA_V4_COURT_SOURCE_ARRAYS")
        source_v = np.asarray(z["topology_vertices_source_world"], dtype=np.float64)
        source_f = np.asarray(z["topology_faces"], dtype=np.int64)
        face_valid = np.asarray(z["face_valid"], dtype=np.bool_)
        surface_component = np.asarray(z["surface_component_id"], dtype=np.int32)
        sign_component = np.asarray(z["sign_component_id"], dtype=np.int32)
    if face_valid.shape != (len(source_f),) or not np.all(face_valid):
        raise ValueError("TESSA_V4_COURT_INVALID_SOURCE_FACE")
    independent_surface = _edge_surface_components(source_f)
    if _partition(independent_surface) != _partition(surface_component):
        raise ValueError("TESSA_V4_COURT_SURFACE_COMPONENT_PARTITION")

    source_topology = dict(manifest.get("source_topology") or {})
    if source_topology.get("component_relation") != "EDGE_CONNECTED_SURFACE_COMPONENT_ID":
        raise ValueError("TESSA_V4_COURT_COMPONENT_RELATION")
    if source_topology.get("sign_component_relation_consumed") is not False:
        raise ValueError("TESSA_V4_COURT_SIGN_COMPONENT_CONSUMED")
    if source_topology.get("sdf_or_occupancy_truth_consumed") is not False:
        raise ValueError("TESSA_V4_COURT_SDF_CONSUMED")
    if source_topology.get("render_vertex_identity_used_as_topology") is not False:
        raise ValueError("TESSA_V4_COURT_RENDER_IDENTITY")
    if int(source_topology.get("surface_component_count", -1)) != len(_partition(independent_surface)):
        raise ValueError("TESSA_V4_COURT_COMPONENT_COUNT")

    with np.load(mesh_path, allow_pickle=False) as z:
        required = {"vertices_source_world", "faces", "source_triangle_id", "source_barycentric"}
        if not required.issubset(z.files):
            raise ValueError("TESSA_V4_COURT_TEACHER_ARRAYS")
        vertices = np.asarray(z["vertices_source_world"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        face_id = np.asarray(z["source_triangle_id"], dtype=np.int64)
        barycentric = np.asarray(z["source_barycentric"], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("TESSA_V4_COURT_TEACHER_LAYOUT")
    if face_id.shape != (len(vertices),) or barycentric.shape != (len(vertices), 3):
        raise ValueError("TESSA_V4_COURT_ADDRESS_LAYOUT")
    if np.any(face_id < 0) or np.any(face_id >= len(source_f)):
        raise ValueError("TESSA_V4_COURT_ADDRESS_FACE")
    if not np.allclose(barycentric.sum(axis=1), 1.0, atol=1.0e-5, rtol=0) or np.any(barycentric < -1.0e-5):
        raise ValueError("TESSA_V4_COURT_BARYCENTRIC")
    reconstructed = np.sum(source_v[source_f[face_id]] * barycentric[..., None], axis=1)
    scale = max(float(np.linalg.norm(source_v.max(0) - source_v.min(0))), 1.0e-9)
    max_error = float(np.max(np.linalg.norm(vertices - reconstructed, axis=1), initial=0.0) / scale)
    if max_error > 0.0105:
        raise ValueError("TESSA_V4_COURT_SOURCE_REBIND_ERROR")

    selected = dict(manifest.get("selected") or {})
    if selected.get("mechanical_pass") is not True:
        raise ValueError("TESSA_V4_COURT_SELECTED_NOT_PASS")
    candidates = [dict(row) for row in manifest.get("candidates") or () if row.get("mechanical_pass") is True]
    if not candidates:
        raise ValueError("TESSA_V4_COURT_NO_PASS_CANDIDATE")
    expected = min(
        candidates,
        key=lambda row: (int(row.get("face_count", 1 << 60)), int(row.get("vertex_count", 1 << 60)), float(row.get("ratio", 1e30))),
    )
    for key in ("face_count", "vertex_count", "ratio", "kind"):
        if selected.get(key) != expected.get(key):
            raise ValueError("TESSA_V4_COURT_SELECTION_DRIFT:" + key)
    if selected.get("candidate_evaluation_contract") != "MEASURED_ALL_KINDS_NO_FALLBACK_OVERRIDE_V1":
        raise ValueError("TESSA_V4_COURT_LEGACY_CANDIDATE_REEVALUATION_REQUIRED")
    if selected.get("kind") not in {"SIMPLIFIED", "SOURCE_FALLBACK"}:
        raise ValueError("TESSA_V4_COURT_SELECTED_KIND")
    for key in REQUIRED:
        if selected.get(key) is not True:
            raise ValueError("TESSA_V4_COURT_DECLARED_PASS_MISSING:" + key)
    if _rest_self_intersection(vertices, faces):
        raise ValueError("TESSA_V4_COURT_REST_SELF_INTERSECTION")

    response_outcome = (manifest.get("family_candidate_outcomes") or {}).get("CAUSAL_RESPONSE")
    if response_outcome != "EXISTING_EVIDENCE_CONSUMED__NEW_FAMILY_COURT_STILL_REQUIRED":
        raise ValueError("TESSA_V4_COURT_RESPONSE_AUTHORITY_OVERCLAIM")

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": manifest.get("subject_id"),
        "teacher_identity_sha256": manifest.get("teacher_identity_sha256"),
        "selected_kind": selected.get("kind"),
        "selected_face_count": int(len(faces)),
        "source_rebind_max_error_over_scale": max_error,
        "surface_component_count": int(len(_partition(independent_surface))),
        "surface_and_sign_partitions_equal": bool(_partition(surface_component) == _partition(sign_component)),
        "mechanical_component_relation": "INDEPENDENT_EDGE_CONNECTED_SURFACE_COMPONENT",
        "sign_component_consumed": False,
        "sdf_or_occupancy_consumed": False,
        "selected_candidate_order_verified": True,
        "family_candidate_outcomes": {
            "SURFACE_TOPOLOGY": "CONSUMER_SEMANTICS_PASS_CANDIDATE",
            "CAUSAL_RESPONSE": "PENDING_NEW_FAMILY_COURT",
        },
        "authority": {
            "tessa_surface_topology_consumer_court_passed": True,
            "causal_response_family_authority_minted": False,
            "final_subject_disposition_minted": False,
            "training_authorized": False,
        },
    }
    payload["court_identity_sha256"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.teacher_manifest.resolve().read_text())
    payload = validate(manifest)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "subject_id": payload["subject_id"], "court_identity_sha256": payload["court_identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
