from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import Delaunay

from realsas_compiler_core.hashing import content_sha256
from realsas_compiler_core.mesh_binding import mesh_candidate_lineage_hash
from realsas_compiler_core.types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    RiggingSurfaceIR,
    SurfaceSupportBinding,
)


@dataclass(frozen=True)
class DemoMeshConfig:
    max_vertices: int = 2048
    raster_round_digits: int = 5
    qhull_options: str = "QJ Qbb Qc"


def _binding_for_view(node, view_index: int):
    for view, xy in node.raster_bindings:
        if int(view) == int(view_index):
            return tuple(map(float, xy))
    return None


def _deterministic_subsample(records: list[tuple], max_vertices: int) -> list[tuple]:
    if len(records) <= max_vertices:
        return records
    # Uniform deterministic coverage in raster-sorted order. No learned or
    # specimen-specific selection enters this branch-local deformation carrier.
    ids = np.linspace(0, len(records) - 1, max_vertices, dtype=np.int64)
    return [records[int(i)] for i in ids]


def build_view_local_identity_mesh_candidate(
    surface: RiggingSurfaceIR,
    *,
    view_index: int,
    camera_contract: dict,
    cfg: DemoMeshConfig = DemoMeshConfig(),
) -> MeshDiscretizationCandidateIR:
    """Triangulate only admitted nodes that have a raster binding in one view.

    Every vertex is an exact identity binding to an existing surface node. Delaunay
    topology is explicitly a demo deformation discretization, not new geometry truth.
    """
    records = []
    seen_raster = set()
    for node in sorted(surface.surface_nodes, key=lambda n: n.surface_id):
        xy = _binding_for_view(node, view_index)
        if xy is None:
            continue
        key = (round(xy[0], cfg.raster_round_digits), round(xy[1], cfg.raster_round_digits))
        if key in seen_raster:
            continue
        seen_raster.add(key)
        records.append((key[1], key[0], node.surface_id, xy, tuple(map(float, node.P))))
    records.sort()
    records = _deterministic_subsample(records, cfg.max_vertices)
    if len(records) < 3:
        raise ValueError("DEMO_MESH_INSUFFICIENT_VIEW_BOUND_SURFACE_NODES")

    raster = np.asarray([[r[3][0], r[3][1]] for r in records], dtype=np.float64)
    if not np.isfinite(raster).all():
        raise ValueError("DEMO_MESH_NONFINITE_RASTER")
    try:
        tri = Delaunay(raster, qhull_options=cfg.qhull_options)
    except Exception as exc:
        raise ValueError(f"DEMO_MESH_DELAUNAY_FAIL:{type(exc).__name__}:{exc}") from exc

    vertices = []
    candidate_ids = []
    for i, rec in enumerate(records):
        surface_id = rec[2]
        cid = f"DMV:{view_index}:{i:05d}"
        candidate_ids.append(cid)
        vertices.append(
            MeshVertexCandidate(
                candidate_vertex_id=cid,
                P=rec[4],
                support_binding=SurfaceSupportBinding(
                    mode="IDENTITY_SURFACE_NODE",
                    coefficients=((surface_id, 1.0),),
                    metadata={"demo_view_index": int(view_index)},
                ),
                metadata={"raster_xy": rec[3], "source_surface_id": surface_id},
            )
        )

    faces = []
    edge_set: set[tuple[str, str]] = set()
    for simplex in tri.simplices.tolist():
        face = tuple(candidate_ids[int(i)] for i in simplex)
        if len(set(face)) != 3:
            continue
        faces.append(face)
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_set.add(tuple(sorted((a, b))))
    if not faces:
        raise ValueError("DEMO_MESH_EMPTY_TRIANGULATION")

    camera_hash = content_sha256({"demo_fixed_camera_contract": camera_contract, "view_index": int(view_index)})
    candidate = MeshDiscretizationCandidateIR(
        vertices=tuple(vertices),
        faces=tuple(faces),
        edges=tuple(sorted(edge_set)),
        surface_binding_hash=surface.geometry_lineage_hash,
        view_index=int(view_index),
        camera_binding_hash=camera_hash,
        candidate_lineage_hash="",
        boundary_constraints=(),
        coverage_classification="DEMO_VIEW_LOCAL_ADMITTED_SUPPORT_ONLY",
        solver_provenance={
            "solver": "SCIPY_DELAUNAY_VIEW_LOCAL_DEMO_V1",
            "qhull_options": cfg.qhull_options,
            "new_geometry_authority": False,
        },
        residual_report={
            "input_surface_nodes": len(surface.surface_nodes),
            "view_bound_vertices": len(vertices),
            "face_count": len(faces),
        },
        metadata={
            "experimental_demo_only": True,
            "hidden_surface_completion": False,
            "all_vertices_identity_bound": True,
        },
    )
    return MeshDiscretizationCandidateIR(
        **{**candidate.__dict__, "candidate_lineage_hash": mesh_candidate_lineage_hash(candidate)}
    )
