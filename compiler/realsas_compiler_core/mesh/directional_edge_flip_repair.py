from __future__ import annotations

"""Coverage-preserving local quality repair for supported directional meshes.

The primary operator is an internal-edge flip over a convex two-triangle quad.
It changes only the diagonal: vertex positions, source raster support, mechanical
support bindings, and the union of the two raster triangles are unchanged.
Therefore source coverage is preserved geometrically while triangle conditioning may
improve. No new vertex, source pixel, weight, or geometry authority is introduced.
"""

from dataclasses import replace
import math

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_lineage_hash,
    validate_qualified_mesh,
)
from compiler.realsas_compiler_core.types import QualificationError

DIRECTIONAL_EDGE_FLIP_REPAIR_SCHEMA = "RealSaS.DirectionalEdgeFlipRepair.v1"


def _xy(mesh) -> dict[str, tuple[float, float]]:
    out = {}
    for vertex in mesh.vertices:
        md = dict(getattr(vertex, "metadata", {}) or {})
        p = md.get("source_raster_xy", md.get("raster_xy"))
        if p is None or len(p) != 2:
            raise QualificationError("DIRECTIONAL_REPAIR_RASTER_XY_REQUIRED")
        x, y = float(p[0]), float(p[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise QualificationError("DIRECTIONAL_REPAIR_RASTER_XY_NONFINITE")
        out[str(vertex.canonical_mesh_vertex_id)] = (x, y)
    return out


def _area2(a, b, c) -> float:
    return float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))


def _metrics(a, b, c) -> tuple[float, float, float]:
    """Return area, min angle degrees, aspect=max_edge/min_altitude."""
    area2 = abs(_area2(a, b, c))
    area = 0.5 * area2
    if area <= 1.0e-15:
        return area, 0.0, float("inf")
    lengths = (
        math.dist(a, b),
        math.dist(b, c),
        math.dist(c, a),
    )
    if min(lengths) <= 1.0e-15:
        return area, 0.0, float("inf")
    angles = []
    sides = lengths
    for i in range(3):
        x = sides[i]
        y = sides[(i+1)%3]
        z = sides[(i+2)%3]
        # angle opposite z
        cosv = max(-1.0, min(1.0, (x*x+y*y-z*z)/(2.0*x*y)))
        angles.append(math.degrees(math.acos(cosv)))
    max_edge = max(lengths)
    min_altitude = area2 / max_edge
    aspect = max_edge / max(min_altitude, 1.0e-15)
    return area, min(angles), aspect


def _face_score(face, xy, *, min_angle_deg: float, max_aspect: float) -> tuple[int, float, float]:
    a, b, c = (xy[str(v)] for v in face)
    area, angle, aspect = _metrics(a, b, c)
    fail = int(
        area <= 1.0e-12
        or angle + 1.0e-12 < float(min_angle_deg)
        or aspect - 1.0e-12 > float(max_aspect)
    )
    return fail, angle, aspect


def _orient(face, xy, sign: float):
    a, b, c = map(str, face)
    s = _area2(xy[a], xy[b], xy[c])
    if abs(s) <= 1.0e-12:
        raise QualificationError("DIRECTIONAL_REPAIR_DEGENERATE_FACE")
    if s * sign > 0.0:
        return (a, b, c)
    return (a, c, b)


def _edge_faces(faces):
    out: dict[tuple[str, str], list[int]] = {}
    for i, face in enumerate(faces):
        if len(face) != 3 or len(set(face)) != 3:
            raise QualificationError("DIRECTIONAL_REPAIR_REQUIRES_TRIANGULATED_MANIFOLD")
        a, b, c = map(str, face)
        for u, v in ((a,b),(b,c),(c,a)):
            key = tuple(sorted((u,v)))
            out.setdefault(key, []).append(i)
    if any(len(rows) > 2 for rows in out.values()):
        raise QualificationError("DIRECTIONAL_REPAIR_NONMANIFOLD_INPUT")
    return out


def _convex_quad(old_edge, opposite_a, opposite_b, xy) -> bool:
    u, v = old_edge
    # The opposite vertices must lie on opposite sides of the old diagonal.
    s1 = _area2(xy[u], xy[v], xy[opposite_a])
    s2 = _area2(xy[u], xy[v], xy[opposite_b])
    if s1 * s2 >= -1.0e-12:
        return False
    # Old edge endpoints must lie on opposite sides of the proposed diagonal.
    t1 = _area2(xy[opposite_a], xy[opposite_b], xy[u])
    t2 = _area2(xy[opposite_a], xy[opposite_b], xy[v])
    return t1 * t2 < -1.0e-12


def _pair_objective(faces, xy, *, min_angle_deg, max_aspect):
    scores = [
        _face_score(f, xy, min_angle_deg=min_angle_deg, max_aspect=max_aspect)
        for f in faces
    ]
    fail_count = sum(s[0] for s in scores)
    min_angle = min(s[1] for s in scores)
    max_aspect_value = max(s[2] for s in scores)
    # Lexicographic: fewer failing faces, then larger minimum angle, then lower aspect.
    return (fail_count, -min_angle, max_aspect_value)


def repair_directional_mesh_by_edge_flips_v1(
    mesh,
    surface,
    *,
    min_angle_deg: float = 0.25,
    max_aspect: float = 250.0,
    max_passes: int = 8,
):
    """Deterministically improve quality without changing raster coverage union."""

    if float(min_angle_deg) <= 0.0 or float(max_aspect) <= 1.0 or int(max_passes) <= 0:
        raise ValueError("invalid directional edge-flip repair policy")
    validate_qualified_mesh(mesh, surface)
    xy = _xy(mesh)
    faces = [tuple(map(str, face)) for face in mesh.faces]
    if not faces:
        raise QualificationError("DIRECTIONAL_REPAIR_EMPTY_FACE_SET")

    # Preserve the mesh's existing per-face orientation convention.
    signs = []
    for face in faces:
        a,b,c = (xy[v] for v in face)
        s = _area2(a,b,c)
        if abs(s) <= 1.0e-12:
            raise QualificationError("DIRECTIONAL_REPAIR_DEGENERATE_INPUT_FACE")
        signs.append(1.0 if s > 0.0 else -1.0)
    dominant_sign = 1.0 if sum(signs) >= 0 else -1.0

    before_bad = sum(
        _face_score(f, xy, min_angle_deg=min_angle_deg, max_aspect=max_aspect)[0]
        for f in faces
    )
    accepted = 0
    passes = 0

    for _pass in range(int(max_passes)):
        passes += 1
        changed = False
        edge_map = _edge_faces(faces)
        candidates = []
        for edge, incident in edge_map.items():
            if len(incident) != 2:
                continue
            i, j = incident
            fi, fj = faces[i], faces[j]
            oa = next(v for v in fi if v not in edge)
            ob = next(v for v in fj if v not in edge)
            if oa == ob or not _convex_quad(edge, oa, ob, xy):
                continue
            old_pair = (fi, fj)
            new_pair = (
                _orient((oa, ob, edge[0]), xy, dominant_sign),
                _orient((ob, oa, edge[1]), xy, dominant_sign),
            )
            old_obj = _pair_objective(
                old_pair, xy, min_angle_deg=min_angle_deg, max_aspect=max_aspect
            )
            new_obj = _pair_objective(
                new_pair, xy, min_angle_deg=min_angle_deg, max_aspect=max_aspect
            )
            if new_obj < old_obj:
                candidates.append((new_obj, old_obj, edge, i, j, new_pair))

        if not candidates:
            break
        candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))

        used_faces: set[int] = set()
        for _new_obj, _old_obj, _edge, i, j, new_pair in candidates:
            if i in used_faces or j in used_faces:
                continue
            # Re-check against current topology because earlier flips in this pass may
            # invalidate a candidate neighborhood.
            current_edges = _edge_faces(faces)
            incident = current_edges.get(_edge, ())
            if tuple(sorted(incident)) != tuple(sorted((i, j))):
                continue
            faces[i], faces[j] = new_pair
            used_faces.update((i, j))
            accepted += 1
            changed = True
        if not changed:
            break

    after_bad = sum(
        _face_score(f, xy, min_angle_deg=min_angle_deg, max_aspect=max_aspect)[0]
        for f in faces
    )

    # Edge flips over convex quads preserve the geometric union exactly. Vertex set,
    # rest positions, support bindings, and source UV/raster coordinates are untouched.
    report = dict(mesh.qualification_report)
    report["directional_edge_flip_repair"] = {
        "schema": DIRECTIONAL_EDGE_FLIP_REPAIR_SCHEMA,
        "accepted_flip_count": int(accepted),
        "passes": int(passes),
        "bad_face_count_before": int(before_bad),
        "bad_face_count_after": int(after_bad),
        "min_angle_deg": float(min_angle_deg),
        "max_aspect": float(max_aspect),
        "coverage_union_preserved_by_operator": True,
        "vertex_authority_changed": False,
        "support_binding_changed": False,
        "source_raster_binding_changed": False,
    }

    updated = replace(
        mesh,
        faces=tuple(faces),
        qualification_report=report,
        mesh_lineage_hash="",
    )
    updated = replace(updated, mesh_lineage_hash=mesh_lineage_hash(updated))
    validate_qualified_mesh(updated, surface)
    return updated


__all__ = [
    "DIRECTIONAL_EDGE_FLIP_REPAIR_SCHEMA",
    "repair_directional_mesh_by_edge_flips_v1",
]
