from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from ..hashing import content_sha256
from ..types import RiggingSurfaceIR, SurfaceNode


_RECONSTRUCTED_MESH_OPERATOR = {
    "schema": "RealSaS.ReconstructedMeshSurfaceSampler.v1",
    "sampling": "DETERMINISTIC_AREA_WEIGHTED_TRIANGLE_SEQUENCE",
    "normal": "SOURCE_TRIANGLE_GEOMETRIC_NORMAL",
    "observational_support_authority": False,
    "source_mesh_used": True,
}


def _array_sha256(a: np.ndarray) -> str:
    x = np.ascontiguousarray(a)
    h = sha256()
    h.update(str(x.dtype).encode("utf-8"))
    h.update(str(tuple(x.shape)).encode("utf-8"))
    h.update(x.tobytes())
    return h.hexdigest()


def load_obj_triangle_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only OBJ positions/faces and deterministically triangulate polygons.

    Texture/normal records are intentionally ignored. Geometry normals are recomputed
    from the emitted surface triangles so downstream conditioning never trusts stale
    or exporter-specific vertex-normal state.
    """
    path = Path(path)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"OBJ_BAD_VERTEX:{line_no}")
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
            continue
        if not line.startswith("f "):
            continue
        raw_idx = []
        for token in line.split()[1:]:
            head = token.split("/", 1)[0]
            if not head:
                raise ValueError(f"OBJ_BAD_FACE_INDEX:{line_no}")
            idx = int(head)
            if idx == 0:
                raise ValueError(f"OBJ_ZERO_FACE_INDEX:{line_no}")
            if idx < 0:
                idx = len(vertices) + idx
            else:
                idx -= 1
            if idx < 0 or idx >= len(vertices):
                raise ValueError(f"OBJ_FACE_INDEX_OOB:{line_no}:{idx}")
            raw_idx.append(idx)
        if len(raw_idx) < 3:
            raise ValueError(f"OBJ_FACE_TOO_SHORT:{line_no}")
        a = raw_idx[0]
        for i in range(1, len(raw_idx) - 1):
            faces.append((a, raw_idx[i], raw_idx[i + 1]))
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3:
        raise ValueError("OBJ_REQUIRES_FINITE_VERTEX_ARRAY")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) < 1:
        raise ValueError("OBJ_REQUIRES_TRIANGLES")
    if not np.isfinite(v).all():
        raise ValueError("OBJ_NONFINITE_VERTEX")
    return v, f


def _triangle_geometry(vertices: np.ndarray, faces: np.ndarray):
    tri = np.asarray(vertices, np.float64)[np.asarray(faces, np.int64)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1e-12)
    if not valid.any():
        raise ValueError("RECONSTRUCTED_MESH_HAS_NO_NONDEGENERATE_TRIANGLES")
    tri = tri[valid]
    face_ids = np.flatnonzero(valid).astype(np.int64)
    double_area = double_area[valid]
    normals = cross[valid] / double_area[:, None]
    areas = 0.5 * double_area
    return tri, face_ids, areas, normals


def _deterministic_surface_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(sample_count) < 1:
        raise ValueError("sample_count must be positive")
    tri, face_ids, areas, normals = _triangle_geometry(vertices, faces)
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("RECONSTRUCTED_MESH_BAD_AREA")
    cdf = np.cumsum(areas)
    targets = (np.arange(sample_count, dtype=np.float64) + 0.5) * total / float(sample_count)
    local_face = np.searchsorted(cdf, targets, side="left").clip(0, len(tri) - 1)

    # Two irrational rotations yield a deterministic low-discrepancy sequence. The
    # sqrt transform converts unit-square coordinates to uniform triangle barycentrics.
    i = np.arange(sample_count, dtype=np.float64) + 1.0
    r1 = np.mod(i * 0.7548776662466927, 1.0)
    r2 = np.mod(i * 0.5698402909980532, 1.0)
    s = np.sqrt(r1)
    bary = np.stack([1.0 - s, s * (1.0 - r2), s * r2], axis=1)
    chosen = tri[local_face]
    points = (chosen * bary[:, :, None]).sum(axis=1)
    sample_normals = normals[local_face]
    return points, sample_normals, face_ids[local_face], bary


def rigging_surface_from_reconstructed_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sample_count: int = 2048,
    source_ref: str = "RECONSTRUCTED_MESH",
    reconstruction_model: str = "UNKNOWN",
    reconstruction_checkpoint: str = "",
) -> RiggingSurfaceIR:
    """Compile a complete reconstructed mesh into the shared rigging substrate.

    This is intentionally a different epistemic lane from observation-only IRIS V2.
    Points are *reconstructed/completed* geometry, so support_views/raster_bindings are
    left empty rather than falsely declaring that every generated surface point was
    directly visible in the source artwork.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("reconstructed mesh must be vertices[N,3], faces[M,3]")
    if len(v) < 3 or len(f) < 1 or not np.isfinite(v).all():
        raise ValueError("reconstructed mesh is empty/nonfinite")
    if f.min(initial=0) < 0 or f.max(initial=-1) >= len(v):
        raise ValueError("reconstructed mesh face index out of range")

    points, normals, face_ids, bary = _deterministic_surface_samples(v, f, int(sample_count))
    mesh_sha = content_sha256({
        "vertices_sha256": _array_sha256(v.astype(np.float32)),
        "faces_sha256": _array_sha256(f.astype(np.int64)),
        "vertex_count": int(len(v)),
        "face_count": int(len(f)),
    })
    operator_hash = content_sha256({**_RECONSTRUCTED_MESH_OPERATOR, "sample_count": int(sample_count)})

    nodes = []
    for i, (p, n, face_id, bw) in enumerate(zip(points, normals, face_ids, bary)):
        sid = f"RCM:{i:06d}:{content_sha256({'mesh': mesh_sha, 'i': i, 'p': p.tolist()})[:12]}"
        nodes.append(SurfaceNode(
            surface_id=sid,
            P=tuple(map(float, p)),
            support_views=(),
            provenance_refs=(str(source_ref),),
            source_observation_ids=(),
            raster_bindings=(),
            persistence_group_id=f"RCM:{i:06d}",
            derived_normal=tuple(map(float, n)),
            validity_flags=("RECONSTRUCTED_COMPLETE_SURFACE",),
            metadata={
                "source_face_id": int(face_id),
                "source_face_barycentric": tuple(map(float, bw)),
                "observational_support_authority": False,
            },
        ))

    lineage = content_sha256({
        "schema": "RealSaS.RiggingSurfaceIR.v1",
        "builder": "RECONSTRUCTED_COMPLETE_MESH_V1",
        "mesh_sha256": mesh_sha,
        "operator_hash": operator_hash,
        "nodes": [n.to_dict() for n in nodes],
    })
    return RiggingSurfaceIR(
        tuple(nodes),
        local_relations=(),
        geometry_lineage_hash=lineage,
        builder_id="RealSaS.GeometricSubstrateAssembler.reconstructed_mesh_v1",
        metadata={
            "source_domain": "RECONSTRUCTED_COMPLETE_MESH",
            "full_3d_reconstruction_claim": True,
            "observational_support_authority": False,
            "reconstruction_model": str(reconstruction_model),
            "reconstruction_checkpoint": str(reconstruction_checkpoint),
            "source_ref": str(source_ref),
            "mesh_sha256": mesh_sha,
            "source_vertex_count": int(len(v)),
            "source_face_count": int(len(f)),
            "surface_sample_count": int(len(nodes)),
            "surface_sampling_operator": _RECONSTRUCTED_MESH_OPERATOR["schema"],
            "surface_sampling_operator_hash": operator_hash,
            # Existing Geppetto V2 contract asks for a local-geometry provenance hash.
            # Here normals come directly from reconstructed triangle geometry rather
            # than DTB-ND1, so the field carries this operator hash explicitly.
            "Nd_operator_sha256": operator_hash,
            "raster_coordinate_system": "NONE_RECONSTRUCTED_COMPLETE_MESH",
        },
    )


def rigging_surface_from_obj(
    path: str | Path,
    *,
    sample_count: int = 2048,
    source_ref: str | None = None,
    reconstruction_model: str = "UNKNOWN",
    reconstruction_checkpoint: str = "",
) -> RiggingSurfaceIR:
    path = Path(path)
    vertices, faces = load_obj_triangle_mesh(path)
    return rigging_surface_from_reconstructed_mesh(
        vertices,
        faces,
        sample_count=sample_count,
        source_ref=source_ref or f"OBJ:{path.name}",
        reconstruction_model=reconstruction_model,
        reconstruction_checkpoint=reconstruction_checkpoint,
    )
