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


def _sample_binary_field(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
    return mask[y, x]


def build_visual_mesh_from_mask(
    mask: np.ndarray,
    *,
    target_edge_px: int = 20,
) -> VisualMesh2D:
    """Build a source-art-owned 2D mesh without borrowing mechanical faces.

    A dense-enough silhouette boundary plus an interior grid is Delaunay
    triangulated. Triangles are admitted only when vertices, edge interior
    probes and centroid remain inside the source foreground. This gives a
    silhouette-constrained visual carrier while keeping dependencies limited to
    SciPy already frozen in mainline CI.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("VISUAL_MESH_MASK_EMPTY_OR_INVALID")
    h, w = map(int, mask.shape)
    step = max(4, int(target_edge_px))

    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    boundary = mask & ~eroded
    by, bx = np.nonzero(boundary)
    if len(bx) < 3:
        raise ValueError("VISUAL_MESH_BOUNDARY_TOO_SMALL")

    # One representative boundary sample per step-sized raster bin.
    bins: dict[tuple[int, int], list[float]] = {}
    for x, y in zip(bx.tolist(), by.tolist()):
        key = (x // step, y // step)
        row = bins.setdefault(key, [0.0, 0.0, 0.0])
        row[0] += float(x)
        row[1] += float(y)
        row[2] += 1.0
    boundary_points = np.asarray(
        [[sx / n, sy / n] for sx, sy, n in bins.values()],
        dtype=np.float64,
    )

    ys = np.arange(step // 2, h, step, dtype=np.int64)
    xs = np.arange(step // 2, w, step, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    interior_xy = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    keep = mask[interior_xy[:, 1], interior_xy[:, 0]]
    interior_points = interior_xy[keep].astype(np.float64)

    points = np.concatenate((boundary_points, interior_points), axis=0)
    # Stable de-duplication after subpixel boundary averaging.
    quant = np.rint(points * 16.0).astype(np.int64)
    _, unique_idx = np.unique(quant, axis=0, return_index=True)
    points = points[np.sort(unique_idx)]
    if len(points) < 3:
        raise ValueError("VISUAL_MESH_POINT_SET_TOO_SMALL")

    tri = Delaunay(points)
    faces = np.asarray(tri.simplices, dtype=np.int64)
    p = points[faces]

    probes = np.stack(
        (
            p[:, 0, :],
            p[:, 1, :],
            p[:, 2, :],
            (p[:, 0, :] + p[:, 1, :]) * 0.5,
            (p[:, 1, :] + p[:, 2, :]) * 0.5,
            (p[:, 2, :] + p[:, 0, :]) * 0.5,
            p.mean(axis=1),
            p[:, 0, :] * 0.25 + p[:, 1, :] * 0.75,
            p[:, 1, :] * 0.25 + p[:, 2, :] * 0.75,
            p[:, 2, :] * 0.25 + p[:, 0, :] * 0.75,
        ),
        axis=1,
    )
    admitted = np.ones(len(faces), dtype=bool)
    for k in range(probes.shape[1]):
        admitted &= _sample_binary_field(mask, probes[:, k, :])
    faces = faces[admitted]
    if len(faces) == 0:
        raise ValueError("VISUAL_MESH_NO_ADMITTED_TRIANGLES")

    used = np.unique(faces.ravel())
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    points = points[used]
    faces = remap[faces]

    uv = np.empty_like(points, dtype=np.float64)
    uv[:, 0] = points[:, 0] / max(1.0, float(w - 1))
    uv[:, 1] = points[:, 1] / max(1.0, float(h - 1))
    return VisualMesh2D(
        positions=points,
        faces=faces.astype(np.uint32),
        uv=np.clip(uv, 0.0, 1.0),
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
