#!/usr/bin/env python3
"""Generic source-locus pair identifiability: topology + authored kinematics.

This diagnostic does NOT determine editable drawable or mechanical ownership.
An authored animation is an observational trajectory, NOT an isolated control.
Connected triangles may be different editable drawables; disconnected
triangles may belong to one drawable. A deforming single skinned drawable
can have non-rigid internal motion; common-root co-motion cannot prove
single ownership.

All face indices, barycentrics and pose arrays are teacher-only source
addresses. Labels, material/object/bone names and color are never used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.corpus.topology_partition_court_v1 import (
    edge_connected_face_partition,
    vertex_connected_face_partition,
)

SCHEMA = "RealSaS.Corpus370GenericPairKinematicDiagnostic.v1"
BATCH_SCHEMA = "RealSaS.Corpus370GenericPairKinematicDiagnosticBatch.v1"
DISTANCE_TOL_OVER_SCALE = 1e-3
ORIENTATION_TOL_RADIANS = 2e-2
REST_VERTEX_TOL_OVER_SCALE = 1e-6


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def sha(path: Path) -> str:
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def _address(address: dict[str, Any], face_count: int) -> tuple[int, np.ndarray]:
    require(isinstance(address, dict), "PAIR_ADDRESS_REQUIRED")
    face = address.get("face_index_teacher_only")
    bary = address.get("barycentric_teacher_only")
    require(type(face) is int and 0 <= face < face_count,
            "PAIR_FACE_ADDRESS_RANGE")
    require(isinstance(bary, (list, tuple)) and len(bary) == 3,
            "PAIR_BARYCENTRIC_SHAPE")
    w = np.asarray(bary, dtype=np.float64)
    require(np.isfinite(w).all() and (w >= -1e-6).all()
            and (w <= 1+1e-6).all() and abs(float(w.sum())-1) <= 1e-5,
            "PAIR_BARYCENTRIC_INVALID")
    return face, w


def _orientation(tri: np.ndarray) -> np.ndarray | None:
    """Right-handed face frame, no semantic axes or authored rig names."""
    e0 = tri[1] - tri[0]
    e1 = tri[2] - tri[0]
    n0 = float(np.linalg.norm(e0))
    if n0 <= 1e-12:
        return None
    x = e0 / n0
    z = np.cross(x, e1)
    nz = float(np.linalg.norm(z))
    if nz <= 1e-12:
        return None
    z = z / nz
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=1)


def _rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    r = left.T @ right
    cosine = max(-1.0, min(1.0, (float(np.trace(r)) - 1) / 2))
    return float(math.acos(cosine))


def _pair_geometry(
    rest: np.ndarray,
    faces: np.ndarray,
    posed: np.ndarray,
    face_a: int, bary_a: np.ndarray,
    face_b: int, bary_b: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    a = rest[faces[face_a]]
    b = rest[faces[face_b]]
    basis_a = _orientation(a)
    basis_b = _orientation(b)
    if basis_a is None or basis_b is None:
        return {"status": "ABSTAIN_DEGENERATE_REST_TRIANGLE"}
    ref_dist = float(np.linalg.norm(bary_a @ a - bary_b @ b))
    ref_basis = basis_a.T @ basis_b
    changes = []
    angles = []
    for vertices in posed:
        aa = vertices[faces[face_a]]
        bb = vertices[faces[face_b]]
        qa = _orientation(aa)
        qb = _orientation(bb)
        if qa is None or qb is None:
            return {"status": "ABSTAIN_DEGENERATE_POSED_TRIANGLE"}
        distance = float(np.linalg.norm(bary_a @ aa - bary_b @ bb))
        changes.append(abs(distance - ref_dist) / scale)
        relative_basis = qa.T @ qb
        angles.append(_rotation_angle(ref_basis, relative_basis))
    dist = max(changes)
    orient = max(angles)
    observed = (dist > DISTANCE_TOL_OVER_SCALE
                or orient > ORIENTATION_TOL_RADIANS)
    return {
        "status": (
            "OBSERVED_RELATIVE_DEFORMATION__DRAWABLE_OWNER_UNKNOWN"
            if observed else
            "NO_RELATIVE_DEFORMATION_OBSERVED__DRAWABLE_OWNER_UNKNOWN"
        ),
        "max_distance_change_over_source_scale": dist,
        "max_relative_face_orientation_change_radians": orient,
        "relative_deformation_observed": observed,
        "authored_motion_is_causal_control": False,
        "identification_from_co_motion_allowed": False,
    }


def analyze(
    rest_vertices: Any,
    faces: Any,
    posed_vertices: Any,
    pairs: list[dict[str, Any]],
    *,
    source_sha256: str,
    rest_geometry_sha256: str,
    authored_teacher_sha256: str,
    independent_visibility_sha256: str,
    authored_rest_vertices: Any | None = None,
    authored_faces: Any | None = None,
) -> dict[str, Any]:
    for digest in (source_sha256, rest_geometry_sha256, authored_teacher_sha256,
                   independent_visibility_sha256):
        require(isinstance(digest, str) and len(digest) == 64
                and set(digest) <= set("0123456789abcdef"), "PAIR_PARENT_SHA")
    v = np.asarray(rest_vertices, np.float64)
    f = np.asarray(faces)
    posed = np.asarray(posed_vertices, np.float64)
    require(v.ndim == 2 and v.shape[1:] == (3,) and len(v) > 0
            and np.isfinite(v).all(), "PAIR_REST_VERTICES")
    require(f.ndim == 2 and f.shape[1:] == (3,)
            and len(f) > 0 and f.dtype.kind in "iu"
            and (f >= 0).all() and (f < len(v)).all()
            and not np.any((f[:,0] == f[:,1]) | (f[:,1] == f[:,2])
                           | (f[:,2] == f[:,0])), "PAIR_FACE_TOPOLOGY")
    require(posed.ndim == 3 and posed.shape[1:] == v.shape
            and len(posed) >= 2 and np.isfinite(posed).all(),
            "PAIR_AUTHORED_POSE_SHAPE_OR_NONFINITE")
    require(isinstance(pairs, list) and pairs, "PAIR_MEASUREMENTS_REQUIRED")
    scale = float(np.linalg.norm(np.ptp(v, axis=0)))
    require(math.isfinite(scale) and scale > 1e-10, "PAIR_REST_GEOMETRY_SCALE")
    # Authored export must bind to exact original source rest mesh ordering.
    # Similar-looking or reindexed geometry is not enough for face IDs.
    require(authored_rest_vertices is not None and authored_faces is not None,
            "PAIR_INDEPENDENT_TEACHER_REST_EQUIVALENCE_REQUIRED")
    teacher_v = np.asarray(authored_rest_vertices, dtype=np.float64)
    teacher_f = np.asarray(authored_faces)
    require(teacher_v.shape == v.shape and teacher_f.shape == f.shape
            and np.array_equal(teacher_f, f)
            and np.isfinite(teacher_v).all()
            and float(np.max(np.abs(teacher_v - v))) <= REST_VERTEX_TOL_OVER_SCALE * scale,
            "PAIR_TEACHER_REST_OR_FACE_ORDER_DRIFT")
    vertex_partition = vertex_connected_face_partition(f)
    edge_partition = edge_connected_face_partition(f)
    manifold_partition = edge_connected_face_partition(f, manifold_edges_only=True)
    rows = []
    seen = set()
    for item in pairs:
        require(isinstance(item, dict), "PAIR_ITEM_OBJECT")
        a, wa = _address(item.get("locus_a"), len(f))
        b, wb = _address(item.get("locus_b"), len(f))
        require(a != b, "PAIR_SELF_FACE_NOT_DISCRIMINATIVE")
        view = item.get("view_index")
        px = item.get("pixel_x")
        py = item.get("pixel_y")
        require(type(view) is int and 0 <= view < 8
                and type(px) is int and px >= 0
                and type(py) is int and py >= 0, "PAIR_VIEW_PIXEL_REQUIRED")
        key = (view, px, py)
        require(key not in seen, "PAIR_DUPLICATE_SOURCE_PIXEL")
        seen.add(key)
        shared = len(set(map(int, f[a])) & set(map(int, f[b])))
        kinematic = _pair_geometry(v, f, posed, a, wa, b, wb, scale)
        rows.append({
            "schema": SCHEMA,
            "source_face_a_teacher_only": a,
            "source_face_b_teacher_only": b,
            "view_index": view, "pixel_x": px, "pixel_y": py,
            "topology": {
                "shared_vertex_count": shared,
                "vertex_connected": bool(vertex_partition[a] == vertex_partition[b]),
                "surface_edge_connected": bool(edge_partition[a] == edge_partition[b]),
                "manifold_edge_connected": bool(manifold_partition[a] == manifold_partition[b]),
                "source_connection_not_drawable_identity": True,
            },
            "authored_kinematics": kinematic,
            "same_drawable": "UNKNOWN",
            "different_drawable": "UNKNOWN",
            "cut": "UNKNOWN",
            "attachment": "UNKNOWN",
            "mechanical_owner": "UNKNOWN",
            "source_hit_visibility_owner": "SOURCE_LOCUS_ONLY",
            "causal_control_qualified": False,
            "student_teacher_ids_exposed": False,
            "graphics_family_authority_minted": False,
            "training_authorized": False,
        })
    return {
        "schema": BATCH_SCHEMA,
        "status": "PAIRED_SOURCE_TOPOLOGY_AND_AUTHORED_KINEMATICS__DRAWABLE_UNKNOWN",
        "parents": {
            "source_sha256": source_sha256,
            "rest_geometry_sha256": rest_geometry_sha256,
            "authored_teacher_sha256": authored_teacher_sha256,
            "independent_visibility_sha256": independent_visibility_sha256,
        },
        "source_scale": scale,
        "authored_pose_count": len(posed),
        "pair_count": len(rows),
        "observed_relative_deformation_count": sum(
            x["authored_kinematics"].get("relative_deformation_observed") is True
            for x in rows),
        "degenerate_abstain_count": sum(
            x["authored_kinematics"]["status"].startswith("ABSTAIN_") for x in rows),
        "qualified_same_drawable_positives": 0,
        "qualified_different_drawable_negatives": 0,
        "rows": rows,
        "graphics_family_authority_minted": False,
        "training_authorized": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rest-geometry", type=Path, required=True)
    p.add_argument("--authored-teacher", type=Path, required=True)
    p.add_argument("--independent-visibility", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--expected-rest-sha256", required=True)
    p.add_argument("--expected-teacher-sha256", required=True)
    p.add_argument("--expected-visibility-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for path,expected in (
        (args.rest_geometry,args.expected_rest_sha256),
        (args.authored_teacher,args.expected_teacher_sha256),
        (args.independent_visibility,args.expected_visibility_sha256),
    ):
        require(path.is_file() and sha(path) == expected, "PAIR_PARENT_BYTES_SHA_DRIFT")
    vis = json.loads(args.independent_visibility.read_text())
    require(vis.get("schema") == "RealSaS.Corpus370BoundedSourceHitVisibilityCandidateCourt.v1"
            and vis.get("source_sha256") == args.source_sha256
            and vis.get("training_authorized") is False
            and vis.get("qualified_same_drawable_positives") == 0
            and vis.get("qualified_different_drawable_negatives") == 0,
            "PAIR_VISIBILITY_NOT_BOUNDED_SOURCE_EVIDENCE")
    pairs = []
    for row in vis.get("rows",[]):
        require(row.get("status") == "MEASURED_SOURCE_HIT_VISIBILITY_CANDIDATE_ONLY"
                and len(row.get("targets",[])) == 2
                and row.get("source_sha256") == args.source_sha256
                and row.get("rest_geometry_sha256") == args.expected_rest_sha256,
                "PAIR_VISIBILITY_ROW_SOURCE_IDENTITY")
        targets = row["targets"]
        require(targets[0]["visibility_label"] == "POSITIVE_VISIBLE_CONTRIBUTOR"
                and targets[1]["visibility_label"] == "NEGATIVE_OPAQUE_OCCLUDED_CONTRIBUTOR",
                "PAIR_SOURCE_VISIBILITY_CONTRADICTION")
        pairs.append({
            "locus_a": {
                "face_index_teacher_only": targets[0]["source_face_teacher_only"],
                "barycentric_teacher_only": targets[0]["barycentric_teacher_only"],
            },
            "locus_b": {
                "face_index_teacher_only": targets[1]["source_face_teacher_only"],
                "barycentric_teacher_only": targets[1]["barycentric_teacher_only"],
            },
            "view_index": row["view_index"], "pixel_x": row["pixel_x"],
            "pixel_y": row["pixel_y"],
        })
    with np.load(args.rest_geometry,allow_pickle=False) as r, np.load(
        args.authored_teacher,allow_pickle=False
    ) as t:
        require({"rest_vertices_source_world","faces"} <= set(r.files)
                and {"rest_vertices_source_world","faces","posed_vertices_source_world"}
                    <= set(t.files),
                "PAIR_REQUIRED_TEACHER_ARRAYS")
        result=analyze(
            r["rest_vertices_source_world"],r["faces"],
            t["posed_vertices_source_world"],pairs,
            source_sha256=args.source_sha256,
            rest_geometry_sha256=args.expected_rest_sha256,
            authored_teacher_sha256=args.expected_teacher_sha256,
            independent_visibility_sha256=args.expected_visibility_sha256,
            authored_rest_vertices=t["rest_vertices_source_world"],
            authored_faces=t["faces"],
        )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},sort_keys=True))


if __name__ == "__main__":
    main()
