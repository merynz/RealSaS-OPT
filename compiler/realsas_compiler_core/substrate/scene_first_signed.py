from __future__ import annotations

from dataclasses import replace
import math
import numpy as np

from .hashing import content_sha256
from .types import QualificationError, RiggingSurfaceIR, SurfaceNode, SurfaceRelation


ZERO_SURFACE_NORMAL_OPERATOR_ID = "RealSaS.GSA.ZeroSurfaceRobustLocalPCA.v1"
ZERO_SURFACE_COMPACTOR_ID = "RealSaS.GSA.ZeroSurfaceAdaptiveVoxel.v1"
ZERO_SURFACE_VISIBILITY_ID = "RealSaS.GSA.ZeroSurfaceSelfZBufferSupport.v1"


def zero_surface_normal_operator_identity_v1(*, k: int = 64) -> dict:
    return {
        "operator_id": ZERO_SURFACE_NORMAL_OPERATOR_ID,
        "method": "ROBUST_LOCAL_PCA_ON_DENSE_SIGNED_ZERO_SURFACE",
        "knn_k": int(k),
        "robust_gate": "neighbor_dist<=median+3*max(1.4826*MAD,1e-8)",
        "orientation_hint": "MARCHING_CUBES_IMPLICIT_NORMAL_SIGN_ONLY",
        "teacher_truth_used": False,
    }


def zero_surface_normal_operator_hash_v1(*, k: int = 64) -> str:
    return content_sha256(zero_surface_normal_operator_identity_v1(k=k))


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(~np.isfinite(a)) or np.any(n <= 1e-12):
        raise QualificationError("zero-surface normal contains invalid vector")
    return a / n


def robust_zero_surface_normals_v1(points, orientation_hints, *, k: int = 64) -> np.ndarray:
    """Deterministic robust local-plane normals on the dense decoded zero surface."""
    p = np.asarray(points, dtype=np.float64)
    hint = _normalize_rows(np.asarray(orientation_hints, dtype=np.float64))
    if p.ndim != 2 or p.shape[1] != 3 or hint.shape != p.shape or len(p) < 4:
        raise QualificationError("zero-surface normal input must be matching [N,3], N>=4")
    if not np.isfinite(p).all():
        raise QualificationError("zero-surface points must be finite")
    kk = min(max(3, int(k)), len(p) - 1)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required for zero-surface local geometry") from exc
    tree = cKDTree(p)
    out = np.empty_like(p)
    query_chunk = 8192
    for start in range(0, len(p), query_chunk):
        stop = min(len(p), start + query_chunk)
        _, rows = tree.query(p[start:stop], k=kk + 1, workers=-1)
        rows = np.asarray(rows[:, 1:], dtype=np.int64)
        x = p[rows]
        center = np.median(x, axis=1, keepdims=True)
        dist = np.linalg.norm(x - center, axis=2)
        med = np.median(dist, axis=1)
        mad = np.median(np.abs(dist - med[:, None]), axis=1)
        gate = dist <= med[:, None] + 3.0 * np.maximum(1.4826 * mad[:, None], 1e-8)
        w = gate.astype(np.float64)
        sw = w.sum(axis=1)
        bad = sw < 3.0
        if np.any(bad):
            w[bad] = 1.0
            sw[bad] = float(kk)
        mu = (w[:, :, None] * x).sum(axis=1) / sw[:, None]
        y = x - mu[:, None, :]
        cov = np.einsum("nki,nkj,nk->nij", y, y, w) / sw[:, None, None]
        evals, evecs = np.linalg.eigh(cov)
        n = evecs[:, :, 0]
        if not np.isfinite(evals).all() or not np.isfinite(n).all():
            raise QualificationError("non-finite zero-surface local PCA")
        h = hint[start:stop]
        flip = np.sum(n * h, axis=1) < 0.0
        n[flip] *= -1.0
        out[start:stop] = _normalize_rows(n)
    return out.astype(np.float32)


def mesh_connected_component_labels_v1(
    vertex_count: int,
    faces: np.ndarray,
    *,
    face_chunk_size: int = 524288,
) -> np.ndarray:
    """Exact mesh connectivity partition without Python per-face union loops.

    Labels are canonicalized by the minimum vertex id in each connected component.
    Only the partition is authority; union traversal order is deliberately not.
    """
    n = int(vertex_count)
    f = np.asarray(faces, dtype=np.int64)
    if n < 1 or f.ndim != 2 or f.shape[1] != 3:
        raise QualificationError("MESH_COMPONENT_LABEL_INPUT_INVALID")
    if np.any(f < 0) or np.any(f >= n):
        raise QualificationError("MESH_COMPONENT_LABEL_FACE_INDEX_INVALID")
    chunk = int(face_chunk_size)
    if chunk < 1:
        raise QualificationError("MESH_COMPONENT_LABEL_CHUNK_INVALID")

    parent = np.arange(n, dtype=np.int64)

    def compress() -> None:
        nonlocal parent
        while True:
            nxt = parent[parent]
            if np.array_equal(nxt, parent):
                return
            parent = nxt

    while True:
        compress()
        changed = False
        for start in range(0, len(f), chunk):
            rows = f[start : start + chunk]
            for left, right in ((0, 1), (1, 2), (2, 0)):
                ra = parent[rows[:, left]]
                rb = parent[rows[:, right]]
                hi = np.maximum(ra, rb)
                lo = np.minimum(ra, rb)
                active = hi != lo
                if np.any(active):
                    changed = True
                    np.minimum.at(parent, hi[active], lo[active])
        if not changed:
            break

    compress()
    roots = parent
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def _mesh_connected_component_labels(vertex_count:int, faces:np.ndarray)->np.ndarray:
    return mesh_connected_component_labels_v1(vertex_count, faces)


def _adaptive_voxel_compact(
    points,
    faces,
    dense_normals,
    *,
    target_nodes: int,
    preserve_connected_components: bool=False,
    precomputed_component_labels: np.ndarray | None = None,
):
    p = np.asarray(points, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.asarray(dense_normals, dtype=np.float64)
    if target_nodes < 64:
        raise QualificationError("scene-first target_nodes must be >=64")
    if len(p) <= target_nodes:
        inverse = np.arange(len(p), dtype=np.int64)
        divisions = 0
    else:
        lo = p.min(axis=0)
        span = np.maximum(p.max(axis=0) - lo, 1e-12)

        component_labels = None
        if bool(preserve_connected_components):
            if precomputed_component_labels is None:
                component_labels = _mesh_connected_component_labels(len(p), f)
            else:
                component_labels = np.asarray(
                    precomputed_component_labels, dtype=np.int64
                )
                if component_labels.shape != (len(p),):
                    raise QualificationError(
                        "PRECOMPUTED_COMPONENT_LABEL_SHAPE_INVALID"
                    )

        def labels_for(divisions: int):
            keys = np.floor((p - lo) / span * divisions).astype(np.int64)
            keys = np.clip(keys, 0, divisions - 1)
            if component_labels is not None:
                keys=np.column_stack((component_labels,keys))
            return np.unique(keys, axis=0, return_inverse=True)

        low, high = 1, 512
        best = None
        while low <= high:
            mid = (low + high) // 2
            unique, inv = labels_for(mid)
            if len(unique) <= target_nodes:
                best = (mid, unique, inv)
                low = mid + 1
            else:
                high = mid - 1
        if best is None:
            raise QualificationError("adaptive zero-surface compaction failed")
        divisions, _unique, inverse = best

    count = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=count).astype(np.float64)
    cp = np.zeros((count, 3), dtype=np.float64)
    cn = np.zeros((count, 3), dtype=np.float64)
    np.add.at(cp, inverse, p)
    np.add.at(cn, inverse, n)
    cp /= counts[:, None]
    if np.any(np.linalg.norm(cn, axis=1) <= 1e-12):
        raise QualificationError("zero-surface compaction produced degenerate normal")
    cn = _normalize_rows(cn)

    mapped = inverse[f]
    edges = np.concatenate([mapped[:, [0, 1]], mapped[:, [1, 2]], mapped[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
    if len(edges) == 0:
        raise QualificationError("zero-surface compaction removed all topology")
    return cp, cn, edges.astype(np.int64), int(divisions), inverse


def _camera_arrays(camera: dict):
    required = ("origin", "right", "screen_up", "forward", "half_extent", "resolution")
    if any(k not in camera for k in required):
        raise QualificationError("scene-first camera contract incomplete")
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    forward = np.asarray(camera["forward"], dtype=np.float64)
    if any(x.shape != (3,) for x in (origin, right, up, forward)):
        raise QualificationError("scene-first camera vectors must be 3D")
    half = float(camera["half_extent"])
    resolution = int(camera["resolution"])
    if not (math.isfinite(half) and half > 0 and resolution > 0):
        raise QualificationError("scene-first camera extent/resolution invalid")
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    forward = forward / np.linalg.norm(forward)
    return origin, right, up, forward, half, resolution


def _project(points: np.ndarray, camera: dict):
    origin, right, up, forward, half, resolution = _camera_arrays(camera)
    d = np.asarray(points, dtype=np.float64) - origin[None, :]
    gx = (d @ right) / half
    gy = -(d @ up) / half
    depth = d @ forward
    raster = np.stack(
        [
            (gx + 1.0) * 0.5 * resolution - 0.5,
            (gy + 1.0) * 0.5 * resolution - 0.5,
        ],
        axis=-1,
    )
    return raster, depth, resolution


def _self_zbuffer_support(dense_points, compact_points, cameras, *, depth_tolerance: float):
    if len(cameras) != 8:
        raise QualificationError("scene-first substrate requires exactly 8 cameras")
    support = np.zeros((len(compact_points), 8), dtype=bool)
    raster_all = np.zeros((len(compact_points), 8, 2), dtype=np.float64)
    view_counts = []
    for view, camera in enumerate(cameras):
        dr, dd, resolution = _project(dense_points, camera)
        cr, cd, cres = _project(compact_points, camera)
        if cres != resolution:
            raise QualificationError("camera resolution drift")
        raster_all[:, view] = cr
        dix = np.rint(dr[:, 0]).astype(np.int64)
        diy = np.rint(dr[:, 1]).astype(np.int64)
        dvalid = (
            (dix >= 0) & (dix < resolution) &
            (diy >= 0) & (diy < resolution) &
            np.isfinite(dd) & (dd > 0.0)
        )
        zbuf = np.full(resolution * resolution, np.inf, dtype=np.float64)
        flat = diy[dvalid] * resolution + dix[dvalid]
        np.minimum.at(zbuf, flat, dd[dvalid])

        cix = np.rint(cr[:, 0]).astype(np.int64)
        ciy = np.rint(cr[:, 1]).astype(np.int64)
        cvalid = (
            (cix >= 0) & (cix < resolution) &
            (ciy >= 0) & (ciy < resolution) &
            np.isfinite(cd) & (cd > 0.0)
        )
        zlocal = np.full(len(compact_points), np.inf, dtype=np.float64)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                xx = cix + ox
                yy = ciy + oy
                ok = cvalid & (xx >= 0) & (xx < resolution) & (yy >= 0) & (yy < resolution)
                rows = np.where(ok)[0]
                if len(rows):
                    zlocal[rows] = np.minimum(zlocal[rows], zbuf[yy[rows] * resolution + xx[rows]])
        vis = cvalid & np.isfinite(zlocal) & (cd <= zlocal + float(depth_tolerance))
        support[:, view] = vis
        view_counts.append(int(vis.sum()))
    return support, raster_all, tuple(view_counts)


def rigging_surface_from_scene_first_zero_mesh_v1(
    vertices_normalized,
    faces,
    implicit_normals,
    cameras,
    *,
    normalization_center,
    normalization_half_extent: float,
    authority_label: str,
    source_run_id: str,
    source_checkpoint_sha256: str,
    source_zero_surface_sha256: str,
    target_nodes: int = 1024,
    normal_k: int = 64,
    visibility_depth_tolerance_norm: float = 0.02,
    component_aware_compaction: bool = False,
    precomputed_dense_normals: np.ndarray | None = None,
    precomputed_component_labels: np.ndarray | None = None,
    metadata: dict | None = None,
) -> RiggingSurfaceIR:
    """Canonical GSA bridge from a predicted signed zero-surface to RiggingSurfaceIR.

    No hidden teacher mesh/rig/skin enters this API. Full reconstructed geometry is
    allowed as an intermediate representation; observation support is re-derived from
    the predicted surface itself and the exact product cameras.
    """
    vn = np.asarray(vertices_normalized, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    hint = np.asarray(implicit_normals, dtype=np.float64)
    center = np.asarray(normalization_center, dtype=np.float64)
    half = float(normalization_half_extent)
    if vn.ndim != 2 or vn.shape[1] != 3 or len(vn) < 4 or not np.isfinite(vn).all():
        raise QualificationError("scene-first zero-surface vertices must be finite [N,3]")
    if f.ndim != 2 or f.shape[1] != 3 or np.any(f < 0) or np.any(f >= len(vn)):
        raise QualificationError("scene-first zero-surface faces must be valid triangles")
    if hint.shape != vn.shape or not np.isfinite(hint).all():
        raise QualificationError("scene-first implicit normals must match vertices")
    if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(half) or half <= 0:
        raise QualificationError("scene-first normalization contract invalid")
    if visibility_depth_tolerance_norm <= 0:
        raise QualificationError("visibility tolerance must be positive")

    world = center[None, :] + vn * half
    if precomputed_dense_normals is None:
        dense_normals = robust_zero_surface_normals_v1(world, hint, k=normal_k)
    else:
        dense_normals = np.asarray(precomputed_dense_normals, dtype=np.float32)
        if dense_normals.shape != vn.shape or not np.isfinite(dense_normals).all():
            raise QualificationError("PRECOMPUTED_DENSE_NORMALS_INVALID")
        lengths = np.linalg.norm(dense_normals, axis=1)
        if np.any(lengths <= 1e-12):
            raise QualificationError("PRECOMPUTED_DENSE_NORMALS_DEGENERATE")
    points, normals, edges, divisions, _inverse = _adaptive_voxel_compact(
        world,
        f,
        dense_normals,
        target_nodes=int(target_nodes),
        preserve_connected_components=bool(component_aware_compaction),
        precomputed_component_labels=precomputed_component_labels,
    )
    support, raster, visible_counts = _self_zbuffer_support(
        world,
        points,
        tuple(cameras),
        depth_tolerance=float(visibility_depth_tolerance_norm) * half,
    )

    op_hash = zero_surface_normal_operator_hash_v1(k=normal_k)
    nodes = []
    for i, (p, n) in enumerate(zip(points, normals)):
        views = tuple(int(v) for v in range(8) if bool(support[i, v]))
        binds = tuple((v, (float(raster[i, v, 0]), float(raster[i, v, 1]))) for v in views)
        flags = ("OBSERVED_SIGNED_ZERO_SURFACE",) if views else ("MODEL_COMPLETED_SIGNED_ZERO_SURFACE",)
        sid = "SFS:" + content_sha256(
            {"index": i, "P": p.tolist(), "views": views, "source": source_zero_surface_sha256}
        )[:20]
        nodes.append(
            SurfaceNode(
                surface_id=sid,
                P=tuple(map(float, p)),
                support_views=views,
                provenance_refs=(str(authority_label),),
                source_observation_ids=(),
                raster_bindings=binds,
                persistence_group_id=f"SCENE_FIRST_SIGNED_ZERO:{i:05d}",
                derived_normal=tuple(map(float, n)),
                validity_flags=flags,
                metadata={
                    "normal_operator": ZERO_SURFACE_NORMAL_OPERATOR_ID,
                    "normal_operator_hash": op_hash,
                    "normal_implicit_hint_only": True,
                    "teacher_truth_used": False,
                },
            )
        )

    rel = []
    ids = tuple(n.surface_id for n in nodes)
    for a, b in edges.tolist():
        dist = float(np.linalg.norm(points[a] - points[b]))
        rel.append(
            SurfaceRelation(
                relation_id="SFSREL:" + content_sha256({"a": ids[a], "b": ids[b], "d": dist})[:20],
                a_surface_id=ids[a],
                b_surface_id=ids[b],
                relation_kind="SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",
                score=1.0,
                metadata={
                    "world_distance": dist,
                    "crosses_unknown": False,
                    "unknown_bridge": False,
                    "teacher_truth_used": False,
                },
            )
        )

    lineage = content_sha256(
        {
            "schema": "RealSaS.SceneFirstSignedToRiggingSurface.v1",
            "source_run_id": str(source_run_id),
            "source_checkpoint_sha256": str(source_checkpoint_sha256),
            "source_zero_surface_sha256": str(source_zero_surface_sha256),
            "normal_operator": zero_surface_normal_operator_identity_v1(k=normal_k),
            "compactor": {
                "id": ZERO_SURFACE_COMPACTOR_ID,
                "target_nodes": int(target_nodes),
                "voxel_divisions": int(divisions),
                "preserve_connected_components": bool(component_aware_compaction),
            },
            "visibility": {
                "id": ZERO_SURFACE_VISIBILITY_ID,
                "depth_tolerance_norm": float(visibility_depth_tolerance_norm),
            },
            "nodes": [n.to_dict() for n in nodes],
            "relations": [r.to_dict() for r in rel],
        }
    )
    meta = {
        **dict(metadata or {}),
        "scene_first_signed_geometry": True,
        "scene_first_decoder": "SIGNED_FIELD_ZERO_LEVEL_SURFACE",
        "source_dense_vertex_count": int(len(world)),
        "compact_surface_node_count": int(len(nodes)),
        "compact_voxel_divisions": int(divisions),
        "compact_target_nodes": int(target_nodes),
        "component_aware_compaction": bool(component_aware_compaction),
        "observed_node_count": int(np.any(support, axis=1).sum()),
        "completed_node_count": int((~np.any(support, axis=1)).sum()),
        "visibility_support_counts_by_view": visible_counts,
        "raster_coordinate_system": "PIXEL_CENTER_XY",
        "resolution": int(cameras[0]["resolution"]),
        "Nd_operator": ZERO_SURFACE_NORMAL_OPERATOR_ID,
        "Nd_operator_sha256": op_hash,
        "normal_implicit_hint_only": True,
        "source_run_id": str(source_run_id),
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "source_zero_surface_sha256": str(source_zero_surface_sha256),
        "teacher_truth_used": False,
        "character_gen_runtime_used": False,
        "full_hidden_mesh_completeness_hard_gate": False,
        "full_3d_intermediate_allowed": True,
    }
    return RiggingSurfaceIR(
        surface_nodes=tuple(nodes),
        local_relations=tuple(sorted(rel, key=lambda r: r.relation_id)),
        geometry_lineage_hash=lineage,
        builder_id="RealSaS.GeometricSubstrateAssembler.SceneFirstSigned.v1",
        schema_version="RealSaS.RiggingSurfaceIR.v1",
        metadata=meta,
    )


__all__ = [
    "ZERO_SURFACE_NORMAL_OPERATOR_ID",
    "ZERO_SURFACE_COMPACTOR_ID",
    "ZERO_SURFACE_VISIBILITY_ID",
    "zero_surface_normal_operator_identity_v1",
    "zero_surface_normal_operator_hash_v1",
    "robust_zero_surface_normals_v1",
    "rigging_surface_from_scene_first_zero_mesh_v1",
]
