from __future__ import annotations

"""Compiler-owned bridge from sealed dense zero-surface topology to current GSA/W.

Dense zero-surface faces own adjacency. The compact RiggingSurfaceIR owns current
mechanical S/G/W. Product use is admitted only when the historical voxel compaction
can be replayed exactly, each render vertex resolves to that exact compact surface
row, and every render face is literally a source zero-surface face.
"""

from dataclasses import dataclass, replace
from hashlib import sha256
import math
from typing import Iterable

import numpy as np

from .hashing import content_sha256
from .mesh_binding import (
    mesh_candidate_lineage_hash,
    mesh_lineage_hash,
    qualify_supported_mesh,
    validate_qualified_mesh,
)
from .types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    QualificationError,
    RiggingSurfaceIR,
    SurfaceSupportBinding,
)

DENSE_ZERO_SURFACE_TOPOLOGY_METHOD = "ZERO_SURFACE_COMPACTION_TOPOLOGY_V1"
_REPLAY_POINT_TOL = 1.0e-9
_PROJECTION_TOL = 1.0e-9


def _array_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = sha256()
    h.update(str(arr.dtype).encode("ascii")); h.update(b"|")
    h.update("x".join(map(str, arr.shape)).encode("ascii")); h.update(b"|")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


@dataclass(frozen=True)
class DenseZeroSurfaceCompactionReplay:
    dense_vertex_to_compact_index: np.ndarray
    compact_surface_ids: tuple[str, ...]
    voxel_divisions: int
    dense_vertex_count: int
    compact_node_count: int
    max_compact_point_error: float
    inverse_sha256: str
    source_zero_surface_sha256: str
    surface_lineage_hash: str
    replay_hash: str

    def summary(self) -> dict:
        return {
            "method": DENSE_ZERO_SURFACE_TOPOLOGY_METHOD,
            "voxel_divisions": int(self.voxel_divisions),
            "dense_vertex_count": int(self.dense_vertex_count),
            "compact_node_count": int(self.compact_node_count),
            "max_compact_point_error": float(self.max_compact_point_error),
            "inverse_sha256": self.inverse_sha256,
            "source_zero_surface_sha256": self.source_zero_surface_sha256,
            "surface_lineage_hash": self.surface_lineage_hash,
            "replay_hash": self.replay_hash,
        }


def replay_dense_zero_surface_compaction(
    surface: RiggingSurfaceIR,
    dense_vertices,
    *,
    source_zero_surface_sha256: str,
    point_tolerance: float = _REPLAY_POINT_TOL,
) -> DenseZeroSurfaceCompactionReplay:
    p = np.asarray(dense_vertices, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise QualificationError("DENSE_ZERO_BRIDGE_INVALID_DENSE_VERTICES")
    if not str(source_zero_surface_sha256 or "").strip():
        raise QualificationError("DENSE_ZERO_BRIDGE_SOURCE_SHA_REQUIRED")
    if point_tolerance <= 0.0 or not math.isfinite(float(point_tolerance)):
        raise ValueError("point_tolerance must be finite and positive")

    md = dict(surface.metadata or {})
    try:
        divisions = int(md["compact_voxel_divisions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError("DENSE_ZERO_BRIDGE_COMPACTION_DIVISIONS_MISSING") from exc
    if divisions <= 0:
        raise QualificationError("DENSE_ZERO_BRIDGE_COMPACTION_DIVISIONS_INVALID")
    if md.get("source_dense_vertex_count") is not None and int(md["source_dense_vertex_count"]) != len(p):
        raise QualificationError("DENSE_ZERO_BRIDGE_DENSE_VERTEX_COUNT_DRIFT")
    md_sha = str(md.get("source_zero_surface_sha256", "") or "")
    if md_sha and md_sha != str(source_zero_surface_sha256):
        raise QualificationError("DENSE_ZERO_BRIDGE_SOURCE_ZERO_SURFACE_SHA_DRIFT")

    lo = p.min(axis=0)
    span = np.maximum(p.max(axis=0) - lo, 1.0e-12)
    keys = np.floor((p - lo) / span * divisions).astype(np.int64)
    keys = np.clip(keys, 0, divisions - 1)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64)
    count = int(inverse.max(initial=-1)) + 1
    if count != len(surface.surface_nodes):
        raise QualificationError(
            f"DENSE_ZERO_BRIDGE_COMPACT_CARDINALITY_DRIFT:{count}!={len(surface.surface_nodes)}"
        )

    counts = np.bincount(inverse, minlength=count).astype(np.float64)
    if np.any(counts <= 0.0):
        raise QualificationError("DENSE_ZERO_BRIDGE_EMPTY_COMPACTION_CELL")
    replay_points = np.zeros((count, 3), dtype=np.float64)
    np.add.at(replay_points, inverse, p)
    replay_points /= counts[:, None]
    surface_points = np.asarray([node.P for node in surface.surface_nodes], dtype=np.float64)
    if replay_points.shape != surface_points.shape or not np.isfinite(surface_points).all():
        raise QualificationError("DENSE_ZERO_BRIDGE_SURFACE_POINT_PAYLOAD_INVALID")
    max_error = float(np.linalg.norm(replay_points - surface_points, axis=1).max(initial=0.0))
    if max_error > float(point_tolerance):
        raise QualificationError(f"DENSE_ZERO_BRIDGE_COMPACT_POINT_REPLAY_DRIFT:{max_error}")

    surface_ids = tuple(str(node.surface_id) for node in surface.surface_nodes)
    if len(set(surface_ids)) != len(surface_ids):
        raise QualificationError("DENSE_ZERO_BRIDGE_DUPLICATE_SURFACE_ID")
    inverse_sha = _array_sha256(inverse)
    replay_hash = content_sha256({
        "schema": "RealSaS.DenseZeroSurfaceCompactionReplay.v1",
        "method": DENSE_ZERO_SURFACE_TOPOLOGY_METHOD,
        "surface_lineage_hash": str(surface.geometry_lineage_hash),
        "source_zero_surface_sha256": str(source_zero_surface_sha256),
        "voxel_divisions": divisions,
        "dense_vertex_count": len(p),
        "compact_node_count": count,
        "inverse_sha256": inverse_sha,
        "max_compact_point_error": max_error,
    })
    return DenseZeroSurfaceCompactionReplay(
        inverse, surface_ids, divisions, len(p), count, max_error, inverse_sha,
        str(source_zero_surface_sha256), str(surface.geometry_lineage_hash), replay_hash,
    )


def _validate_camera(camera: dict, view_index: int):
    if int(camera.get("view_index", -1)) != int(view_index):
        raise QualificationError("DENSE_ZERO_BRIDGE_CAMERA_VIEW_DRIFT")
    try:
        origin = np.asarray(camera["origin"], dtype=np.float64)
        right = np.asarray(camera["right"], dtype=np.float64)
        up = np.asarray(camera["screen_up"], dtype=np.float64)
        half = float(camera["half_extent"])
        resolution = int(camera["resolution"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError("DENSE_ZERO_BRIDGE_CAMERA_CONTRACT_INCOMPLETE") from exc
    if any(v.shape != (3,) for v in (origin, right, up)) or not all(np.isfinite(v).all() for v in (origin, right, up)):
        raise QualificationError("DENSE_ZERO_BRIDGE_CAMERA_VECTOR_INVALID")
    if half <= 0.0 or resolution <= 0 or not math.isfinite(half):
        raise QualificationError("DENSE_ZERO_BRIDGE_CAMERA_SCALE_INVALID")
    rn, un = float(np.linalg.norm(right)), float(np.linalg.norm(up))
    if rn <= 1e-12 or un <= 1e-12:
        raise QualificationError("DENSE_ZERO_BRIDGE_CAMERA_BASIS_DEGENERATE")
    return origin, right / rn, up / un, half, resolution


def project_dense_vertices(dense_vertices, camera: dict, *, view_index: int) -> np.ndarray:
    origin, right, up, half, resolution = _validate_camera(camera, int(view_index))
    p = np.asarray(dense_vertices, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise QualificationError("DENSE_ZERO_BRIDGE_INVALID_DENSE_VERTICES")
    d = p - origin[None, :]
    gx = (d @ right) / half
    gy = -(d @ up) / half
    return np.stack(((gx + 1.0) * 0.5 * resolution - 0.5, (gy + 1.0) * 0.5 * resolution - 0.5), axis=1)


def _selected_face_indices(selected_face_indices: Iterable[int], face_count: int) -> tuple[int, ...]:
    selected = tuple(sorted(set(int(x) for x in selected_face_indices)))
    if not selected:
        raise QualificationError("DENSE_ZERO_BRIDGE_EMPTY_FACE_SELECTION")
    if selected[0] < 0 or selected[-1] >= int(face_count):
        raise QualificationError("DENSE_ZERO_BRIDGE_FACE_INDEX_OUT_OF_RANGE")
    return selected


def build_dense_zero_surface_candidate(
    surface: RiggingSurfaceIR,
    dense_vertices,
    dense_faces,
    replay: DenseZeroSurfaceCompactionReplay,
    *,
    selected_face_indices: Iterable[int],
    camera: dict,
    view_index: int,
    camera_binding_hash: str,
    source_zero_surface_sha256: str,
    dynamic_witness_hash: str,
    coverage_witness_hash: str,
) -> MeshDiscretizationCandidateIR:
    p = np.asarray(dense_vertices, dtype=np.float64)
    f = np.asarray(dense_faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or np.any(f < 0) or np.any(f >= len(p)):
        raise QualificationError("DENSE_ZERO_BRIDGE_INVALID_DENSE_FACES")
    if replay.surface_lineage_hash != str(surface.geometry_lineage_hash):
        raise QualificationError("DENSE_ZERO_BRIDGE_REPLAY_SURFACE_DRIFT")
    if replay.source_zero_surface_sha256 != str(source_zero_surface_sha256):
        raise QualificationError("DENSE_ZERO_BRIDGE_REPLAY_SOURCE_DRIFT")
    if len(replay.dense_vertex_to_compact_index) != len(p):
        raise QualificationError("DENSE_ZERO_BRIDGE_REPLAY_VERTEX_COUNT_DRIFT")
    if not all(str(x or "").strip() for x in (camera_binding_hash, dynamic_witness_hash, coverage_witness_hash)):
        raise QualificationError("DENSE_ZERO_BRIDGE_EVIDENCE_HASH_REQUIRED")

    canonical = _selected_face_indices(selected_face_indices, len(f))
    origin, _right, _up, _half, _resolution = _validate_camera(camera, int(view_index))
    forward = np.asarray(camera.get("forward"), dtype=np.float64)
    if forward.shape != (3,) or not np.isfinite(forward).all() or np.linalg.norm(forward) <= 1e-12:
        raise QualificationError("DENSE_ZERO_BRIDGE_CAMERA_FORWARD_INVALID")
    forward = forward / np.linalg.norm(forward)
    ordered = tuple(sorted(canonical, key=lambda fi: (-float(np.mean((p[f[int(fi)]] - origin) @ forward)), int(fi))))
    source_faces = f[np.asarray(ordered, dtype=np.int64)]
    used_dense = np.unique(source_faces.reshape(-1))
    projected = project_dense_vertices(p[used_dense], camera, view_index=int(view_index))
    id_by_dense = {int(di): f"DZB:V{int(view_index)}:{int(di):06d}" for di in used_dense}
    xy_by_dense = {int(di): tuple(map(float, xy)) for di, xy in zip(used_dense, projected)}

    vertices = []
    for raw_di in used_dense:
        di = int(raw_di)
        ci = int(replay.dense_vertex_to_compact_index[di])
        if not 0 <= ci < len(surface.surface_nodes):
            raise QualificationError("DENSE_ZERO_BRIDGE_COMPACT_INDEX_OUT_OF_RANGE")
        node = surface.surface_nodes[ci]
        sid = str(node.surface_id)
        if sid != replay.compact_surface_ids[ci]:
            raise QualificationError("DENSE_ZERO_BRIDGE_COMPACT_SURFACE_ID_DRIFT")
        vertices.append(MeshVertexCandidate(
            id_by_dense[di], tuple(map(float, node.P)),
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((sid, 1.0),), metadata={"dense_zero_surface_bridge": True}),
            metadata={
                "raster_xy": xy_by_dense[di],
                "source_dense_vertex_index": di,
                "compact_surface_index": ci,
                "dense_zero_surface_topology": True,
                "source_mesh_used": False,
            },
        ))

    faces = []
    for src in source_faces:
        dense_ids = tuple(map(int, src))
        ids = [id_by_dense[x] for x in dense_ids]
        xy = [xy_by_dense[x] for x in dense_ids]
        signed = (xy[1][0]-xy[0][0])*(xy[2][1]-xy[0][1]) - (xy[1][1]-xy[0][1])*(xy[2][0]-xy[0][0])
        if abs(float(signed)) <= 1e-12:
            raise QualificationError("DENSE_ZERO_BRIDGE_SELECTED_FACE_PROJECTS_DEGENERATE")
        if signed < 0.0:
            ids[1], ids[2] = ids[2], ids[1]
        faces.append(tuple(ids))
    edge_set = set()
    for a, b, c in faces:
        edge_set.update((tuple(sorted((a,b))), tuple(sorted((b,c))), tuple(sorted((c,a)))))

    set_sha = _array_sha256(np.asarray(canonical, dtype=np.int64))
    order_sha = _array_sha256(np.asarray(ordered, dtype=np.int64))
    provisional = MeshDiscretizationCandidateIR(
        tuple(vertices), tuple(faces), tuple(sorted(edge_set)), str(surface.geometry_lineage_hash),
        int(view_index), str(camera_binding_hash), "", (), "DENSE_ZERO_SURFACE_DYNAMIC_SAFE_BODY",
        solver_provenance={
            "solver": "ORIGINAL_DENSE_ZERO_SURFACE_FACE_SUBSET",
            "topology_method": DENSE_ZERO_SURFACE_TOPOLOGY_METHOD,
            "new_adjacency_created": False,
            "delaunay_used": False,
            "source_mesh_used": False,
        },
        residual_report={
            "selected_dense_face_count": len(ordered),
            "selected_dense_vertex_count": len(used_dense),
            "selected_face_indices_sha256": set_sha,
        },
        metadata={
            "dense_zero_surface_bridge": True,
            "topology_method": DENSE_ZERO_SURFACE_TOPOLOGY_METHOD,
            "source_zero_surface_sha256": str(source_zero_surface_sha256),
            "compaction_replay_hash": replay.replay_hash,
            "compaction_inverse_sha256": replay.inverse_sha256,
            "selected_dense_face_indices": ordered,
            "selected_dense_face_set_sha256": set_sha,
            "selected_dense_face_order_sha256": order_sha,
            "face_draw_order": "FAR_TO_NEAR_CAMERA_DEPTH_THEN_SOURCE_FACE_ID",
            "dynamic_witness_hash": str(dynamic_witness_hash),
            "coverage_witness_hash": str(coverage_witness_hash),
            "source_mesh_used": False,
            "teacher_topology_used": False,
            "original_dense_zero_surface_faces_only": True,
            "new_adjacency_created": False,
            "camera_geometry_consumed": True,
        },
    )
    return replace(provisional, candidate_lineage_hash=mesh_candidate_lineage_hash(provisional))


def validate_dense_zero_surface_mesh(
    mesh,
    surface: RiggingSurfaceIR,
    dense_vertices,
    dense_faces,
    replay: DenseZeroSurfaceCompactionReplay,
    *,
    camera: dict,
    source_zero_surface_sha256: str,
    projection_tolerance: float = _PROJECTION_TOL,
) -> dict:
    validate_qualified_mesh(mesh, surface)
    p = np.asarray(dense_vertices, dtype=np.float64)
    f = np.asarray(dense_faces, dtype=np.int64)
    md_mesh = dict(mesh.metadata or {})
    if replay.surface_lineage_hash != str(surface.geometry_lineage_hash):
        raise QualificationError("DENSE_ZERO_BRIDGE_REPLAY_SURFACE_DRIFT")
    if md_mesh.get("dense_zero_surface_bridge") is not True:
        raise QualificationError("DENSE_ZERO_BRIDGE_MESH_METADATA_REQUIRED")
    if str(md_mesh.get("source_zero_surface_sha256", "")) != str(source_zero_surface_sha256):
        raise QualificationError("DENSE_ZERO_BRIDGE_MESH_SOURCE_SHA_DRIFT")
    if str(md_mesh.get("compaction_replay_hash", "")) != replay.replay_hash:
        raise QualificationError("DENSE_ZERO_BRIDGE_MESH_REPLAY_HASH_DRIFT")
    selected = tuple(int(x) for x in md_mesh.get("selected_dense_face_indices") or ())
    if not selected or len(selected) != len(mesh.faces) or len(set(selected)) != len(selected):
        raise QualificationError("DENSE_ZERO_BRIDGE_FACE_WITNESS_CARDINALITY_DRIFT")
    if min(selected) < 0 or max(selected) >= len(f):
        raise QualificationError("DENSE_ZERO_BRIDGE_FACE_WITNESS_OUT_OF_RANGE")
    if str(md_mesh.get("selected_dense_face_set_sha256", "")) != _array_sha256(np.asarray(sorted(selected), dtype=np.int64)):
        raise QualificationError("DENSE_ZERO_BRIDGE_FACE_SET_WITNESS_HASH_DRIFT")
    if str(md_mesh.get("selected_dense_face_order_sha256", "")) != _array_sha256(np.asarray(selected, dtype=np.int64)):
        raise QualificationError("DENSE_ZERO_BRIDGE_FACE_ORDER_WITNESS_HASH_DRIFT")

    dense_by_vertex = {}
    max_projection_error = 0.0
    for vertex in mesh.vertices:
        vmd = dict(vertex.metadata or {})
        try:
            di = int(vmd["source_dense_vertex_index"])
            ci = int(vmd["compact_surface_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QualificationError("DENSE_ZERO_BRIDGE_VERTEX_WITNESS_MISSING") from exc
        if not 0 <= di < len(p):
            raise QualificationError("DENSE_ZERO_BRIDGE_VERTEX_INDEX_OUT_OF_RANGE")
        expected_ci = int(replay.dense_vertex_to_compact_index[di])
        if ci != expected_ci:
            raise QualificationError("DENSE_ZERO_BRIDGE_VERTEX_COMPACTION_DRIFT")
        sid = str(surface.surface_nodes[expected_ci].surface_id)
        if vertex.support_binding.mode != "IDENTITY_SURFACE_NODE" or tuple(vertex.support_binding.coefficients) != ((sid, 1.0),):
            raise QualificationError("DENSE_ZERO_BRIDGE_VERTEX_SUPPORT_DRIFT")
        expected_xy = project_dense_vertices(p[di:di+1], camera, view_index=int(mesh.view_index))[0]
        actual_xy = np.asarray(vmd.get("raster_xy", ()), dtype=np.float64)
        if actual_xy.shape != (2,):
            raise QualificationError("DENSE_ZERO_BRIDGE_VERTEX_RASTER_WITNESS_MISSING")
        err = float(np.linalg.norm(expected_xy - actual_xy))
        max_projection_error = max(max_projection_error, err)
        if err > float(projection_tolerance):
            raise QualificationError(f"DENSE_ZERO_BRIDGE_VERTEX_RASTER_REPLAY_DRIFT:{err}")
        dense_by_vertex[str(vertex.canonical_mesh_vertex_id)] = di

    for local_i, (mesh_face, source_i) in enumerate(zip(mesh.faces, selected)):
        dense_face = tuple(dense_by_vertex[str(vid)] for vid in mesh_face)
        source_face = tuple(map(int, f[int(source_i)]))
        if tuple(sorted(dense_face)) != tuple(sorted(source_face)):
            raise QualificationError(f"DENSE_ZERO_BRIDGE_FACE_NOT_SOURCE_TOPOLOGY:{local_i}:{source_i}")

    return {
        "status": "PASS__DENSE_ZERO_SURFACE_COMPACTION_TOPOLOGY",
        "passed": True,
        "method": DENSE_ZERO_SURFACE_TOPOLOGY_METHOD,
        "view_index": int(mesh.view_index),
        "mesh_lineage_hash": str(mesh.mesh_lineage_hash),
        "surface_lineage_hash": str(surface.geometry_lineage_hash),
        "source_zero_surface_sha256": str(source_zero_surface_sha256),
        "compaction_replay_hash": replay.replay_hash,
        "compaction_inverse_sha256": replay.inverse_sha256,
        "selected_dense_face_count": len(selected),
        "mesh_vertex_count": len(mesh.vertices),
        "mesh_edge_count": len(mesh.edges),
        "max_dense_raster_projection_replay_error_px": max_projection_error,
        "original_dense_zero_surface_faces_only": True,
        "new_adjacency_created": False,
        "topology_and_W_share_exact_compaction_address": True,
        "source_mesh_used": False,
        "teacher_topology_used": False,
    }


def qualify_dense_zero_surface_mesh(
    surface: RiggingSurfaceIR,
    candidate: MeshDiscretizationCandidateIR,
    dense_vertices,
    dense_faces,
    replay: DenseZeroSurfaceCompactionReplay,
    *,
    camera: dict,
    source_zero_surface_sha256: str,
):
    mesh = qualify_supported_mesh(surface, candidate, allowed_binding_modes=("IDENTITY_SURFACE_NODE",))
    report = dict(mesh.qualification_report or {})
    metadata = {**dict(candidate.metadata or {}), **dict(mesh.metadata or {}), "source_candidate_lineage_hash": str(candidate.candidate_lineage_hash)}
    mesh = replace(mesh, qualification_report=report, metadata=metadata, mesh_lineage_hash="")
    mesh = replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))
    bridge = validate_dense_zero_surface_mesh(
        mesh, surface, dense_vertices, dense_faces, replay,
        camera=camera, source_zero_surface_sha256=source_zero_surface_sha256,
    )
    report.update({k: v for k, v in bridge.items() if k != "mesh_lineage_hash"})
    mesh = replace(mesh, qualification_report=report, mesh_lineage_hash="")
    mesh = replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))
    validate_qualified_mesh(mesh, surface)
    final = validate_dense_zero_surface_mesh(
        mesh, surface, dense_vertices, dense_faces, replay,
        camera=camera, source_zero_surface_sha256=source_zero_surface_sha256,
    )
    if not final["passed"]:
        raise QualificationError("DENSE_ZERO_BRIDGE_QUALIFICATION_NOT_PASS")
    return mesh, final


__all__ = [
    "DENSE_ZERO_SURFACE_TOPOLOGY_METHOD",
    "DenseZeroSurfaceCompactionReplay",
    "replay_dense_zero_surface_compaction",
    "project_dense_vertices",
    "build_dense_zero_surface_candidate",
    "validate_dense_zero_surface_mesh",
    "qualify_dense_zero_surface_mesh",
]
