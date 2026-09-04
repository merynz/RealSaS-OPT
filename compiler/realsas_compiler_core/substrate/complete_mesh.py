from __future__ import annotations

"""Deterministic bridge from a predicted complete triangle mesh to RiggingSurfaceIR.

This module is intentionally *not* a teacher-truth adapter. It accepts geometry
produced by an external/image-conditioned reconstruction backend (for example
CharacterGen) and converts it into the same compiler-owned surface contract
consumed by Geppetto and Arachne.

The reconstructed mesh may contain completion outside direct image evidence. That
fact is preserved explicitly in metadata and never rewritten as observational
support.
"""

from hashlib import sha256
import math

import numpy as np

from .hashing import content_sha256
from .types import QualificationError, RiggingSurfaceIR, SurfaceNode


MESH_SURFACE_SAMPLER_SPEC = {
    "schema": "RealSaS.CompleteMeshSurfaceSampler.v1",
    "sampling": "AREA_CDF_PLUS_HALTON_BARYCENTRIC",
    "halton_bases": (2, 3),
    "normal": "SOURCE_TRIANGLE_GEOMETRIC_NORMAL",
    "observational_support_claim": False,
    "teacher_truth_used": False,
}


def mesh_surface_sampler_hash() -> str:
    return content_sha256(MESH_SURFACE_SAMPLER_SPEC)


def _radical_inverse(index: int, base: int) -> float:
    if index < 1 or base < 2:
        raise ValueError("radical inverse requires positive index and base>=2")
    inv_base = 1.0 / float(base)
    inv = inv_base
    value = 0.0
    n = int(index)
    while n:
        n, digit = divmod(n, base)
        value += float(digit) * inv
        inv *= inv_base
    return value


def _mesh_digest(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = sha256()
    h.update(np.asarray(vertices, dtype="<f4", order="C").tobytes(order="C"))
    h.update(np.asarray(faces, dtype="<i8", order="C").tobytes(order="C"))
    return h.hexdigest()


def _validated_triangles(vertices, faces) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3:
        raise QualificationError("complete mesh vertices must be finite non-empty [N,3]")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) < 1:
        raise QualificationError("complete mesh faces must be non-empty triangle indices [F,3]")
    if not np.isfinite(v).all():
        raise QualificationError("complete mesh contains non-finite vertices")
    if int(f.min(initial=0)) < 0 or int(f.max(initial=-1)) >= len(v):
        raise QualificationError("complete mesh face index out of bounds")

    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    keep = np.isfinite(double_area) & (double_area > 1e-12)
    if not bool(keep.any()):
        raise QualificationError("complete mesh has no non-degenerate triangles")
    f = f[keep]
    cross = cross[keep]
    double_area = double_area[keep]
    normals = cross / double_area[:, None]
    areas = 0.5 * double_area
    return v, f, normals, areas


def _sample_triangle_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(sample_count) < 4:
        raise ValueError("sample_count must be >=4")
    total = float(face_areas.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise QualificationError("complete mesh total surface area is invalid")
    cdf = np.cumsum(face_areas, dtype=np.float64) / total

    points = np.empty((sample_count, 3), dtype=np.float64)
    normals = np.empty((sample_count, 3), dtype=np.float64)
    face_ids = np.empty(sample_count, dtype=np.int64)
    tri = vertices[faces]

    # Midpoint quantiles give deterministic area-proportional face allocation;
    # Halton coordinates avoid RNG and repeated same-face samples.
    for i in range(sample_count):
        u_area = (float(i) + 0.5) / float(sample_count)
        face_idx = int(np.searchsorted(cdf, u_area, side="left"))
        face_idx = min(face_idx, len(faces) - 1)
        r1 = _radical_inverse(i + 1, 2)
        r2 = _radical_inverse(i + 1, 3)
        s = math.sqrt(r1)
        b0 = 1.0 - s
        b1 = s * (1.0 - r2)
        b2 = s * r2
        points[i] = b0 * tri[face_idx, 0] + b1 * tri[face_idx, 1] + b2 * tri[face_idx, 2]
        normals[i] = face_normals[face_idx]
        face_ids[i] = face_idx
    return points, normals, face_ids


def rigging_surface_from_complete_triangle_mesh(
    vertices,
    faces,
    *,
    provenance_ref: str,
    sample_count: int = 2048,
    backend_id: str = "EXTERNAL_COMPLETE_GEOMETRY",
    source_asset_id: str = "",
    extra_metadata: dict | None = None,
) -> RiggingSurfaceIR:
    """Convert a predicted complete mesh into a deterministic RiggingSurfaceIR.

    `support_views` is intentionally empty: generated/completed geometry is not
    silently promoted to observation truth. Geppetto still receives P, local
    covariance and valid normals; its eight support channels remain zero for these
    completion-derived samples.
    """
    provenance_ref = str(provenance_ref).strip()
    backend_id = str(backend_id).strip()
    if not provenance_ref or not backend_id:
        raise ValueError("provenance_ref and backend_id are required")

    v, f, face_normals, areas = _validated_triangles(vertices, faces)
    points, normals, sampled_face_ids = _sample_triangle_surface(
        v, f, face_normals, areas, int(sample_count)
    )
    mesh_sha = _mesh_digest(v, f)
    op_hash = mesh_surface_sampler_hash()

    nodes = []
    for i, (p, n, face_idx) in enumerate(zip(points, normals, sampled_face_ids)):
        sid = "CMESH:" + content_sha256(
            {
                "mesh_sha256": mesh_sha,
                "sample_index": int(i),
                "P": tuple(map(float, p)),
            }
        )[:20]
        nodes.append(
            SurfaceNode(
                surface_id=sid,
                P=tuple(map(float, p)),
                support_views=(),
                provenance_refs=(provenance_ref,),
                source_observation_ids=(),
                raster_bindings=(),
                persistence_group_id=f"COMPLETE_MESH_FACE:{int(face_idx)}",
                derived_normal=tuple(map(float, n)),
                validity_flags=("PREDICTED_COMPLETE_SURFACE",),
                metadata={
                    "backend_id": backend_id,
                    "predicted_mesh_sha256": mesh_sha,
                    "predicted_mesh_face_index": int(face_idx),
                    "observation_supported": False,
                    "teacher_truth_used": False,
                },
            )
        )

    metadata = {
        "surface_source": "PREDICTED_COMPLETE_TRIANGLE_MESH",
        "geometry_completion_authority": "MODEL_PREDICTION_NOT_OBSERVATION_TRUTH",
        "backend_id": backend_id,
        "source_asset_id": str(source_asset_id),
        "predicted_mesh_sha256": mesh_sha,
        "predicted_mesh_vertex_count": int(len(v)),
        "predicted_mesh_triangle_count": int(len(f)),
        "surface_sample_count": int(len(nodes)),
        "sampling_operator": MESH_SURFACE_SAMPLER_SPEC["schema"],
        "sampling_operator_sha256": op_hash,
        # Geppetto V2 requires deterministic normal lineage. These are direct
        # geometric normals of the predicted mesh, not teacher normals.
        "Nd_operator_sha256": op_hash,
        "Nd_operator_id": "PREDICTED_MESH_TRIANGLE_NORMAL_V1",
        "N_required": False,
        "raster_coordinate_system": "GRID_XY",
        "predicted_complete_mesh_used": True,
        "authoritative_source_mesh_used": False,
        "teacher_truth_used": False,
        "observational_support_claim": False,
    }
    if extra_metadata:
        extra = dict(extra_metadata)
        reserved = set(metadata).intersection(extra)
        if reserved:
            raise ValueError(
                f"extra_metadata may not override protected complete-mesh fields: {sorted(reserved)}"
            )
        metadata.update(extra)

    lineage = content_sha256(
        {
            "schema": "RealSaS.RiggingSurfaceIR.v1",
            "backend_id": backend_id,
            "mesh_sha256": mesh_sha,
            "sampling_operator_sha256": op_hash,
            "sample_count": int(sample_count),
            "provenance_ref": provenance_ref,
            "source_asset_id": str(source_asset_id),
        }
    )
    return RiggingSurfaceIR(
        surface_nodes=tuple(nodes),
        local_relations=(),
        geometry_lineage_hash=lineage,
        builder_id="RealSaS.GeometricSubstrateAssembler.complete_mesh_v1",
        metadata=metadata,
    )
