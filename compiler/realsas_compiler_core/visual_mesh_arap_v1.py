from __future__ import annotations

"""Source-art-owned 2D visual mesh and local/global ARAP deformation.

This module intentionally owns presentation geometry only. It must never become
mechanical/skeleton/skin authority. The design follows standard 2D ARAP
principles and the source-silhouette -> visual-mesh separation used by mature
2D skeletal animation systems, but is implemented independently for RealSaS.
"""

from dataclasses import asdict, dataclass, field, replace
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import factorized

from .hashing import content_sha256
from .mesh._historical_v05 import triangulate_production_cdt
from .types import QualificationError


@dataclass(frozen=True)
class VisualMesh2D:
    positions: np.ndarray  # [N,2] source-raster pixel coordinates
    faces: np.ndarray      # [F,3] uint32
    uv: np.ndarray         # [N,2] normalized source texture coordinates
    width: int
    height: int


def visual_mesh_semantic_hash(mesh: VisualMesh2D) -> str:
    positions = np.asarray(mesh.positions, dtype="<f8")
    faces = np.asarray(mesh.faces, dtype="<u4")
    uv = np.asarray(mesh.uv, dtype="<f8")
    h = hashlib.sha256()
    h.update(b"RealSaS.VisualMesh2D.semantic.v1\0")
    h.update(np.asarray((int(mesh.width), int(mesh.height)), dtype="<u4").tobytes())
    h.update(positions.tobytes(order="C"))
    h.update(faces.tobytes(order="C"))
    h.update(uv.tobytes(order="C"))
    return h.hexdigest()


@dataclass(frozen=True)
class VisualMeshViewIR:
    view_index: int
    direction_id: str
    width: int
    height: int
    vertex_count: int
    face_count: int
    mesh_npz_path: str
    mesh_npz_sha256: str
    source_raster_sha256: str
    source_foreground_mask_sha256: str
    mesh_hash: str
    schema_version: str = "RealSaS.VisualMeshViewIR.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class VisualMeshSetIR:
    observation_set_binding_hash: str
    output_direction_set_binding_hash: str
    views: tuple[VisualMeshViewIR, ...]
    set_hash: str
    schema_version: str = "RealSaS.VisualMeshSetIR.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def visual_mesh_set_hash(value: VisualMeshSetIR) -> str:
    payload = value.to_dict()
    payload.pop("set_hash", None)
    return content_sha256(payload)


def validate_visual_mesh_set(value: VisualMeshSetIR) -> None:
    rows = tuple(sorted(value.views, key=lambda row: int(row.view_index)))
    if len(rows) != 8 or tuple(int(row.view_index) for row in rows) != tuple(range(8)):
        raise QualificationError("VISUAL_MESH_SET_REQUIRES_V0_V7")
    if tuple(str(row.direction_id) for row in rows) != tuple(f"V{i}" for i in range(8)):
        raise QualificationError("VISUAL_MESH_SET_DIRECTION_ID_DRIFT")
    for row in rows:
        if (
            row.width <= 0
            or row.height <= 0
            or row.vertex_count < 3
            or row.face_count < 1
        ):
            raise QualificationError("VISUAL_MESH_VIEW_CARDINALITY_INVALID")
        for digest in (
            row.mesh_npz_sha256,
            row.source_raster_sha256,
            row.source_foreground_mask_sha256,
            row.mesh_hash,
        ):
            if len(str(digest)) != 64:
                raise QualificationError("VISUAL_MESH_VIEW_HASH_INVALID")
    if value.set_hash != visual_mesh_set_hash(value):
        raise QualificationError("VISUAL_MESH_SET_HASH_MISMATCH")


def visual_mesh_set_from_dict(payload: Mapping[str, Any]) -> VisualMeshSetIR:
    views = tuple(
        VisualMeshViewIR(
            view_index=int(row["view_index"]),
            direction_id=str(row["direction_id"]),
            width=int(row["width"]),
            height=int(row["height"]),
            vertex_count=int(row["vertex_count"]),
            face_count=int(row["face_count"]),
            mesh_npz_path=str(row["mesh_npz_path"]),
            mesh_npz_sha256=str(row["mesh_npz_sha256"]),
            source_raster_sha256=str(row["source_raster_sha256"]),
            source_foreground_mask_sha256=str(row["source_foreground_mask_sha256"]),
            mesh_hash=str(row["mesh_hash"]),
            schema_version=str(row.get("schema_version") or "RealSaS.VisualMeshViewIR.v1"),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("views") or ()
    )
    value = VisualMeshSetIR(
        observation_set_binding_hash=str(payload["observation_set_binding_hash"]),
        output_direction_set_binding_hash=str(payload["output_direction_set_binding_hash"]),
        views=views,
        set_hash=str(payload["set_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.VisualMeshSetIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    validate_visual_mesh_set(value)
    return value


def load_visual_mesh_view(value: VisualMeshViewIR) -> VisualMesh2D:
    path = Path(value.mesh_npz_path)
    if not path.is_file():
        raise QualificationError("VISUAL_MESH_VIEW_NPZ_MISSING")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != value.mesh_npz_sha256:
        raise QualificationError("VISUAL_MESH_VIEW_NPZ_BYTES_DRIFT")
    with np.load(path, allow_pickle=False) as data:
        required = {"positions", "faces", "uv"}
        if not required.issubset(data.files):
            raise QualificationError("VISUAL_MESH_VIEW_NPZ_ARRAY_MISSING")
        positions = np.asarray(data["positions"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.uint32)
        uv = np.asarray(data["uv"], dtype=np.float64)
    mesh = VisualMesh2D(
        positions=positions,
        faces=faces,
        uv=uv,
        width=int(value.width),
        height=int(value.height),
    )
    if (
        positions.shape != (value.vertex_count, 2)
        or faces.shape != (value.face_count, 3)
        or uv.shape != (value.vertex_count, 2)
        or not np.isfinite(positions).all()
        or not np.isfinite(uv).all()
        or np.any(faces >= value.vertex_count)
    ):
        raise QualificationError("VISUAL_MESH_VIEW_ARRAY_SHAPE_DRIFT")
    if visual_mesh_semantic_hash(mesh) != value.mesh_hash:
        raise QualificationError("VISUAL_MESH_VIEW_SEMANTIC_HASH_DRIFT")
    return mesh


@dataclass(frozen=True)
class HandleBinding2D:
    triangle_index: int
    barycentric: tuple[float, float, float]


@dataclass(frozen=True)
class ArapQa:
    flipped_triangles: int
    max_handle_residual_px: float
    p95_edge_stretch: float
    max_edge_stretch: float


def _signed_area(loop: list[tuple[int, int]]) -> float:
    area = 0.0
    for i, a in enumerate(loop):
        b = loop[(i + 1) % len(loop)]
        area += float(a[0] * b[1] - b[0] * a[1])
    return 0.5 * area


def _simplify_axis_aligned_loop(
    loop: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Remove only exactly collinear pixel-union boundary vertices.

    Coordinates are integer half-pixel units (2*x), so this operation is exact
    and cannot move the source silhouette.
    """
    if len(loop) < 3:
        return loop
    changed = True
    out = list(loop)
    while changed and len(out) >= 3:
        changed = False
        keep = []
        n = len(out)
        for i in range(n):
            a = out[(i - 1) % n]
            b = out[i]
            d = out[(i + 1) % n]
            ab = (b[0] - a[0], b[1] - a[1])
            bd = (d[0] - b[0], d[1] - b[1])
            if ab[0] * bd[1] - ab[1] * bd[0] == 0:
                # Same axis and same direction only. Never collapse a 180-degree
                # cusp because that would alter topology.
                if ab[0] * bd[0] + ab[1] * bd[1] > 0:
                    changed = True
                    continue
            keep.append(b)
        out = keep
    return out


def _trace_pixel_union_loops(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Exact closed loops of the union of foreground pixel cells.

    Pixel (x,y) is represented by a unit square centered at source-raster
    coordinate (x,y), hence corners are (x±0.5,y±0.5). We store doubled integer
    coordinates while tracing to avoid floating-point topology drift.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        x0, x1 = 2 * x - 1, 2 * x + 1
        y0, y1 = 2 * y - 1, 2 * y + 1
        if y == 0 or not mask[y - 1, x]:
            edges.add(((x0, y0), (x1, y0)))  # east
        if x + 1 >= w or not mask[y, x + 1]:
            edges.add(((x1, y0), (x1, y1)))  # south
        if y + 1 >= h or not mask[y + 1, x]:
            edges.add(((x1, y1), (x0, y1)))  # west
        if x == 0 or not mask[y, x - 1]:
            edges.add(((x0, y1), (x0, y0)))  # north

    if not edges:
        raise ValueError("VISUAL_MESH_BOUNDARY_EMPTY")

    outgoing: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for a, b in edges:
        outgoing.setdefault(a, set()).add(b)

    direction_code = {
        (2, 0): 0,   # east
        (0, 2): 1,   # south
        (-2, 0): 2,  # west
        (0, -2): 3,  # north
    }

    remaining = set(edges)
    loops: list[list[tuple[int, int]]] = []
    while remaining:
        first = min(remaining)
        start, nxt = first
        loop = [start]
        remaining.remove(first)
        cur = nxt
        prev = start
        guard = 0
        while cur != start:
            loop.append(cur)
            candidates = [
                b for b in outgoing.get(cur, ())
                if (cur, b) in remaining
            ]
            if not candidates:
                raise ValueError("VISUAL_MESH_BOUNDARY_OPEN_LOOP")
            incoming = (cur[0] - prev[0], cur[1] - prev[1])
            code = direction_code.get(incoming)
            if code is None:
                raise ValueError("VISUAL_MESH_BOUNDARY_NON_AXIS_EDGE")
            by_priority = []
            for b in candidates:
                delta = (b[0] - cur[0], b[1] - cur[1])
                out_code = direction_code.get(delta)
                if out_code is None:
                    continue
                turn = (out_code - code) % 4
                # Interior is on the right: prefer right turn, then straight,
                # then left, then back at a diagonal-touch ambiguity.
                rank = {1: 0, 0: 1, 3: 2, 2: 3}[turn]
                by_priority.append((rank, b))
            if not by_priority:
                raise ValueError("VISUAL_MESH_BOUNDARY_TRACE_FAILED")
            _, chosen = min(by_priority)
            remaining.remove((cur, chosen))
            prev, cur = cur, chosen
            guard += 1
            if guard > len(edges) + 4:
                raise ValueError("VISUAL_MESH_BOUNDARY_TRACE_GUARD")
        simplified = _simplify_axis_aligned_loop(loop)
        if len(simplified) >= 3 and abs(_signed_area(simplified)) > 0.0:
            loops.append(simplified)

    if not loops:
        raise ValueError("VISUAL_MESH_NO_CLOSED_BOUNDARY_LOOP")
    return loops


def _component_cdt(
    component_mask: np.ndarray,
    *,
    target_edge_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    loops_i2 = _trace_pixel_union_loops(component_mask)
    # For the directed pixel-union edges used above, the outer loop has the
    # largest absolute area. Treat every other enclosed loop as a hole and
    # normalize winding for the numerical kernel.
    outer_i2 = max(loops_i2, key=lambda row: abs(_signed_area(row)))
    holes_i2 = [row for row in loops_i2 if row is not outer_i2]

    def as_float(row):
        return [(0.5 * float(x), 0.5 * float(y)) for x, y in row]

    outer = as_float(outer_i2)
    if _signed_area(outer_i2) < 0.0:
        outer = list(reversed(outer))

    holes = []
    for row in holes_i2:
        value = as_float(row)
        if _signed_area(row) > 0.0:
            value = list(reversed(value))
        holes.append(value)

    h, w = component_mask.shape
    step = max(4, int(target_edge_px))
    support = []
    # Interior source-pixel centers are exact visual sample coordinates.
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if component_mask[y, x]:
                support.append((float(x), float(y)))

    boundary_vertices = sum(len(row) for row in (outer, *holes))
    result = triangulate_production_cdt(
        outer,
        hole_loops=holes or None,
        support_points=support or None,
        target_min_angle_deg=7.5,
        max_boundary_vertices=max(512, boundary_vertices * 2 + 64),
        max_support_vertices=max(128, len(support) + 64),
        max_constraint_recovery_iterations=192,
        max_quality_iterations=96,
        min_feature_spacing=1.0e-6,
    )
    if not bool(result.success):
        raise ValueError(f"VISUAL_MESH_CDT_FAIL:{result.reason}")
    if int(result.missing_constraint_count) != 0:
        raise ValueError("VISUAL_MESH_CDT_MISSING_CONSTRAINT")
    vertices = np.asarray(result.vertices, dtype=np.float64)
    faces = np.asarray(result.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("VISUAL_MESH_CDT_RESULT_SHAPE_INVALID")
    if len(vertices) < 3 or len(faces) < 1:
        raise ValueError("VISUAL_MESH_CDT_EMPTY")
    return vertices, faces


def build_visual_mesh_from_mask(
    mask: np.ndarray,
    *,
    target_edge_px: int = 20,
) -> VisualMesh2D:
    """Build a source-owned visual mesh with exact silhouette constraints.

    The source foreground mask is interpreted as a union of pixel cells.
    Closed outer/hole loops are extracted exactly, then triangulated by the
    hash-sealed historical RealSaS production CDT kernel. Mechanical relation
    faces are never used as visual topology.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("VISUAL_MESH_MASK_EMPTY_OR_INVALID")
    h, w = map(int, mask.shape)

    labels, count = ndimage.label(
        mask,
        structure=np.asarray(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            dtype=np.uint8,
        ),
    )
    all_vertices = []
    all_faces = []
    offset = 0
    for label in range(1, int(count) + 1):
        component = labels == label
        vertices, faces = _component_cdt(
            component,
            target_edge_px=target_edge_px,
        )
        all_vertices.append(vertices)
        all_faces.append(faces + offset)
        offset += len(vertices)

    points = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    uv = np.empty_like(points, dtype=np.float64)
    uv[:, 0] = points[:, 0] / max(1.0, float(w - 1))
    uv[:, 1] = points[:, 1] / max(1.0, float(h - 1))
    return VisualMesh2D(
        positions=points,
        faces=faces.astype(np.uint32),
        uv=uv,
        width=w,
        height=h,
    )


def _barycentric(point: np.ndarray, tri: np.ndarray) -> np.ndarray | None:
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    v2 = point - a
    den = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(den) <= 1.0e-12:
        return None
    inv = 1.0 / den
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) * inv)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) * inv)
    return np.asarray((1.0 - u - v, u, v), dtype=np.float64)


def bind_points_barycentric(
    mesh: VisualMesh2D,
    points_xy: np.ndarray,
    *,
    candidate_triangles: int = 64,
) -> tuple[HandleBinding2D, ...]:
    points = np.asarray(points_xy, dtype=np.float64)
    tris = mesh.positions[mesh.faces]
    centroids = tris.mean(axis=1)
    tree = cKDTree(centroids)
    k = min(max(1, int(candidate_triangles)), len(tris))
    result = []
    for point in points:
        _dist, idx = tree.query(point, k=k)
        candidates = np.atleast_1d(idx).astype(np.int64)
        best = None
        best_violation = float("inf")
        for tri_index in candidates:
            bary = _barycentric(point, tris[int(tri_index)])
            if bary is None:
                continue
            violation = float(max(0.0, -float(bary.min())))
            if violation < best_violation:
                best = (int(tri_index), bary)
                best_violation = violation
                if violation <= 1.0e-8:
                    break
        if best is None:
            raise ValueError("VISUAL_HANDLE_BINDING_FAILED")
        tri_index, bary = best
        if best_violation > 1.0e-8:
            bary = np.maximum(bary, 0.0)
            bary /= max(1.0e-12, float(bary.sum()))
        result.append(
            HandleBinding2D(
                triangle_index=tri_index,
                barycentric=tuple(map(float, bary)),
            )
        )
    return tuple(result)


def bind_visual_vertex_handles(
    mesh: VisualMesh2D,
    vertex_indices: np.ndarray | list[int] | tuple[int, ...],
) -> tuple[HandleBinding2D, ...]:
    """Bind selected visual vertices as exact one-hot ARAP constraints.

    This is used when RealSaS mechanical-surface correspondence, rather than
    the skeleton itself, owns presentation displacement. Each selected visual
    vertex is constrained through any incident visual triangle with a one-hot
    barycentric row, so the ARAP system remains unchanged.
    """
    indices = tuple(int(v) for v in vertex_indices)
    if not indices:
        raise ValueError("VISUAL_VERTEX_HANDLES_EMPTY")
    incident: dict[int, tuple[int, int]] = {}
    for tri_index, face in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
        for local, vertex_index in enumerate(face.tolist()):
            incident.setdefault(int(vertex_index), (int(tri_index), int(local)))
    rows = []
    seen = set()
    for vertex_index in indices:
        if vertex_index in seen:
            continue
        seen.add(vertex_index)
        if vertex_index < 0 or vertex_index >= len(mesh.positions):
            raise ValueError("VISUAL_VERTEX_HANDLE_INDEX_INVALID")
        if vertex_index not in incident:
            raise ValueError("VISUAL_VERTEX_HANDLE_NOT_INCIDENT")
        tri_index, local = incident[vertex_index]
        bary = [0.0, 0.0, 0.0]
        bary[local] = 1.0
        rows.append(
            HandleBinding2D(
                triangle_index=tri_index,
                barycentric=tuple(bary),
            )
        )
    return tuple(rows)


def sample_bone_handles(
    *,
    joint_ids: tuple[str, ...],
    parent_by_joint: dict[str, str | None],
    joint_xy: dict[str, tuple[float, float]],
    step: float = 0.25,
) -> tuple[tuple[str, str, float], ...]:
    rows: list[tuple[str, str, float]] = []
    emitted_joint = set()
    for joint_id in joint_ids:
        parent = parent_by_joint.get(joint_id)
        if parent is None:
            rows.append((joint_id, joint_id, 1.0))
            emitted_joint.add(joint_id)
            continue
        if parent not in joint_xy or joint_id not in joint_xy:
            continue
        if parent not in emitted_joint:
            rows.append((joint_id, parent, 0.0))
            emitted_joint.add(parent)
        t = float(step)
        while t < 1.0 - 1.0e-9:
            rows.append((joint_id, parent, t))
            t += float(step)
        rows.append((joint_id, parent, 1.0))
        emitted_joint.add(joint_id)
    if not rows:
        raise ValueError("VISUAL_BONE_HANDLES_EMPTY")
    return tuple(rows)


def evaluate_bone_handles(
    specs: tuple[tuple[str, str, float], ...],
    joint_xy: dict[str, tuple[float, float]],
) -> np.ndarray:
    out = np.zeros((len(specs), 2), dtype=np.float64)
    for i, (joint_id, parent_id, t) in enumerate(specs):
        j = np.asarray(joint_xy[joint_id], dtype=np.float64)
        p = np.asarray(joint_xy[parent_id], dtype=np.float64)
        out[i] = p + (j - p) * float(t)
    return out


def _cotangent(a: np.ndarray, b: np.ndarray) -> float:
    cross = float(a[0] * b[1] - a[1] * b[0])
    if abs(cross) <= 1.0e-12:
        return 0.0
    return float(np.dot(a, b) / abs(cross))


class Arap2D:
    def __init__(
        self,
        mesh: VisualMesh2D,
        bindings: tuple[HandleBinding2D, ...],
        *,
        constraint_scale: float = 1.0e3,
    ):
        self.mesh = mesh
        self.rest = np.asarray(mesh.positions, dtype=np.float64)
        self.cur = self.rest.copy()
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        self.bindings = tuple(bindings)
        n = len(self.rest)

        edge_weights: dict[tuple[int, int], float] = {}
        edge_tris: dict[tuple[int, int], list[int]] = {}
        for ti, (i, j, k) in enumerate(self.faces.tolist()):
            pi, pj, pk = self.rest[[i, j, k]]
            rows = (
                ((i, j), 0.5 * _cotangent(pi - pk, pj - pk)),
                ((j, k), 0.5 * _cotangent(pj - pi, pk - pi)),
                ((k, i), 0.5 * _cotangent(pk - pj, pi - pj)),
            )
            for (a, b), w in rows:
                key = (min(a, b), max(a, b))
                edge_weights[key] = edge_weights.get(key, 0.0) + max(1.0e-6, w)
                edge_tris.setdefault(key, []).append(ti)

        self.edges = []
        rr = []
        cc = []
        vv = []
        diag_sum = 0.0
        for (i, j), w in sorted(edge_weights.items()):
            self.edges.append((i, j, float(w), tuple(edge_tris[(i, j)])))
            rr.extend((i, j, i, j))
            cc.extend((i, j, j, i))
            vv.extend((w, w, -w, -w))
            diag_sum += 2.0 * w
        lap = coo_matrix((vv, (rr, cc)), shape=(n, n), dtype=np.float64).tocsr()

        cr = []
        cc2 = []
        cv = []
        for hi, binding in enumerate(self.bindings):
            tri = self.faces[binding.triangle_index]
            for vertex, weight in zip(tri.tolist(), binding.barycentric):
                cr.append(hi)
                cc2.append(int(vertex))
                cv.append(float(weight))
        C = coo_matrix(
            (cv, (cr, cc2)),
            shape=(len(self.bindings), n),
            dtype=np.float64,
        ).tocsr()
        self.C = C
        self.lambda_ = float(constraint_scale) * diag_sum / max(1, n)
        A = lap + self.lambda_ * (C.T @ C) + 1.0e-9 * eye(n, format="csr")
        self.solve_linear = factorized(A.tocsc())
        self.rest_area_sign = np.sign(self._triangle_areas(self.rest))

    def reset(self) -> None:
        self.cur = self.rest.copy()

    def _triangle_areas(self, pos: np.ndarray) -> np.ndarray:
        tri = pos[self.faces]
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def _triangle_rotations(self) -> np.ndarray:
        rotations = np.zeros((len(self.faces), 2), dtype=np.float64)
        for ti, face in enumerate(self.faces):
            r = self.rest[face]
            c = self.cur[face]
            s00 = s01 = s10 = s11 = 0.0
            for a, b in ((0, 1), (1, 2), (2, 0)):
                ce = c[a] - c[b]
                re = r[a] - r[b]
                s00 += ce[0] * re[0]
                s01 += ce[0] * re[1]
                s10 += ce[1] * re[0]
                s11 += ce[1] * re[1]
            angle = float(np.arctan2(s10 - s01, s00 + s11))
            rotations[ti] = (np.cos(angle), np.sin(angle))
        return rotations

    def solve(
        self,
        targets_xy: np.ndarray,
        *,
        iterations: int = 3,
    ) -> tuple[np.ndarray, ArapQa]:
        targets = np.asarray(targets_xy, dtype=np.float64)
        if targets.shape != (len(self.bindings), 2):
            raise ValueError("ARAP_TARGET_SHAPE_INVALID")
        for _ in range(max(1, int(iterations))):
            rotations = self._triangle_rotations()
            bx = np.zeros(len(self.rest), dtype=np.float64)
            by = np.zeros(len(self.rest), dtype=np.float64)
            for i, j, w, tri_ids in self.edges:
                c = float(sum(rotations[t, 0] for t in tri_ids))
                s = float(sum(rotations[t, 1] for t in tri_ids))
                norm = float(np.hypot(c, s))
                if norm <= 1.0e-12:
                    c, s = 1.0, 0.0
                else:
                    c, s = c / norm, s / norm
                ex, ey = self.rest[i] - self.rest[j]
                rx = w * (c * ex - s * ey)
                ry = w * (s * ex + c * ey)
                bx[i] += rx
                by[i] += ry
                bx[j] -= rx
                by[j] -= ry
            bx += self.lambda_ * np.asarray(self.C.T @ targets[:, 0]).reshape(-1)
            by += self.lambda_ * np.asarray(self.C.T @ targets[:, 1]).reshape(-1)
            self.cur[:, 0] = self.solve_linear(bx)
            self.cur[:, 1] = self.solve_linear(by)

        handle_xy = np.column_stack((self.C @ self.cur[:, 0], self.C @ self.cur[:, 1]))
        residual = np.linalg.norm(handle_xy - targets, axis=1)
        area = self._triangle_areas(self.cur)
        flipped = int(np.count_nonzero(
            (self.rest_area_sign != 0.0)
            & (np.sign(area) != 0.0)
            & (np.sign(area) != self.rest_area_sign)
        ))
        stretch = []
        for i, j, _w, _tri_ids in self.edges:
            a = float(np.linalg.norm(self.rest[i] - self.rest[j]))
            b = float(np.linalg.norm(self.cur[i] - self.cur[j]))
            if a > 1.0e-9 and b > 1.0e-9:
                ratio = b / a
                stretch.append(max(ratio, 1.0 / ratio))
        values = np.asarray(stretch or [1.0], dtype=np.float64)
        qa = ArapQa(
            flipped_triangles=flipped,
            max_handle_residual_px=float(residual.max(initial=0.0)),
            p95_edge_stretch=float(np.quantile(values, 0.95)),
            max_edge_stretch=float(values.max(initial=1.0)),
        )
        return self.cur.copy(), qa
