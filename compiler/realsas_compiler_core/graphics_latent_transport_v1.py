from __future__ import annotations

"""Transport IRIS shared-scene H through TESSA repair lineage onto canonical M.

This module is deliberately *not* an admission court. It performs only explicit
lineage transport and reports the geometric residuals of the proposal-vertex ->
IRIS surface-witness association. No threshold is invented here.

Authority path:
  ShapeProposal.surface_h (learned evidence)
    -> original SHAPE_V2 proposal vertex (nearest evidence witness, measured)
    -> TESSAReferenceBinding coefficients (Compiler repair lineage)
    -> CanonicalMeshIR source_candidate_vertex_id (canonical M address)

The result is learned evidence aligned to canonical vertices. It does not mutate
canonical geometry and it does not claim Graphics or Mesh authority.
"""

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any

import numpy as np

from .types import QualificationError


@dataclass(frozen=True)
class GraphicsLatentTransportDiagnosticsV1:
    proposal_vertex_count: int
    surface_witness_count: int
    canonical_vertex_count: int
    latent_width: int
    nearest_distance_min: float
    nearest_distance_median: float
    nearest_distance_p95: float
    nearest_distance_max: float
    repaired_interpolation_row_count: int
    exact_identity_row_count: int
    transport_hash: str
    schema_version: str = "RealSaS.GraphicsLatentTransportDiagnostics.v1"


@dataclass(frozen=True)
class GraphicsLatentTransportV1:
    canonical_h: np.ndarray
    canonical_xyz_tessa: np.ndarray
    proposal_h: np.ndarray
    proposal_xyz_tessa: np.ndarray
    proposal_nearest_surface_index: np.ndarray
    proposal_nearest_surface_distance: np.ndarray
    diagnostics: GraphicsLatentTransportDiagnosticsV1


def _as_matrix(value: Any, *, columns: int | None, label: str, dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2 or not len(array):
        raise QualificationError(f"GRAPHICS_LATENT_{label}_MATRIX_INVALID")
    if columns is not None and array.shape[1] != columns:
        raise QualificationError(f"GRAPHICS_LATENT_{label}_WIDTH_INVALID")
    if not np.isfinite(array).all():
        raise QualificationError(f"GRAPHICS_LATENT_{label}_NONFINITE")
    return array


def _nearest_indices_and_distances(
    query_xyz: np.ndarray,
    witness_xyz: np.ndarray,
    *,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if chunk_size <= 0:
        raise QualificationError("GRAPHICS_LATENT_CHUNK_SIZE_INVALID")
    indices = np.empty((len(query_xyz),), dtype=np.int64)
    distances = np.empty((len(query_xyz),), dtype=np.float64)
    for start in range(0, len(query_xyz), chunk_size):
        stop = min(len(query_xyz), start + chunk_size)
        delta = query_xyz[start:stop, None, :] - witness_xyz[None, :, :]
        distance2 = np.einsum("qwi,qwi->qw", delta, delta, optimize=True)
        nearest = np.argmin(distance2, axis=1)
        indices[start:stop] = nearest
        distances[start:stop] = np.sqrt(
            distance2[np.arange(stop - start, dtype=np.int64), nearest]
        )
    return indices, distances


def _proposal_index(proposal_vertex_id: str, count: int) -> int:
    prefix = "SHAPE_V2:"
    value = str(proposal_vertex_id)
    if not value.startswith(prefix):
        raise QualificationError("GRAPHICS_LATENT_PROPOSAL_VERTEX_ID_INVALID")
    suffix = value[len(prefix):]
    if len(suffix) != 8 or not suffix.isdigit():
        raise QualificationError("GRAPHICS_LATENT_PROPOSAL_VERTEX_ID_INVALID")
    index = int(suffix)
    if index < 0 or index >= count:
        raise QualificationError("GRAPHICS_LATENT_PROPOSAL_VERTEX_INDEX_INVALID")
    return index


def _transport_hash(
    *,
    canonical_h: np.ndarray,
    canonical_xyz: np.ndarray,
    nearest_index: np.ndarray,
    nearest_distance: np.ndarray,
) -> str:
    digest = sha256()
    for array, dtype in (
        (canonical_h, "<f4"),
        (canonical_xyz, "<f8"),
        (nearest_index, "<i8"),
        (nearest_distance, "<f8"),
    ):
        value = np.asarray(array).astype(dtype, copy=False)
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def transport_shape_h_to_canonical_mesh_v1(
    *,
    shape: Any,
    canonical_mesh: Any,
    reference_binding: Any,
) -> GraphicsLatentTransportV1:
    metadata = dict(getattr(shape, "metadata", {}) or {})
    if metadata.get("schema") != "RealSaS.ShapeProposal.v2":
        raise QualificationError("GRAPHICS_LATENT_SHAPE_SCHEMA_DRIFT")
    if metadata.get("canonical_mesh_authority") is not False:
        raise QualificationError("GRAPHICS_LATENT_SHAPE_PREMATURE_MESH_AUTHORITY")

    proposal_xyz = _as_matrix(
        getattr(shape, "decoded_vertices_tessa", None),
        columns=3,
        label="PROPOSAL_XYZ",
        dtype=np.float64,
    )
    surface_xyz = _as_matrix(
        getattr(shape, "surface_xyz_tessa", None),
        columns=3,
        label="SURFACE_XYZ",
        dtype=np.float64,
    )
    surface_h = _as_matrix(
        getattr(shape, "surface_h", None),
        columns=None,
        label="SURFACE_H",
        dtype=np.float32,
    )
    if len(surface_xyz) != len(surface_h):
        raise QualificationError("GRAPHICS_LATENT_SURFACE_ACCOUNTING_DRIFT")

    nearest_index, nearest_distance = _nearest_indices_and_distances(
        proposal_xyz, surface_xyz
    )
    proposal_h = surface_h[nearest_index].astype(np.float32, copy=True)

    rows = tuple(getattr(reference_binding, "rows", ()) or ())
    if not rows:
        raise QualificationError("GRAPHICS_LATENT_REFERENCE_BINDING_EMPTY")
    row_by_candidate: dict[str, Any] = {}
    final_h: dict[str, np.ndarray] = {}
    final_xyz: dict[str, np.ndarray] = {}
    repaired_rows = 0
    identity_rows = 0

    for row in rows:
        candidate_id = str(getattr(row, "candidate_vertex_id", ""))
        if not candidate_id or candidate_id in row_by_candidate:
            raise QualificationError("GRAPHICS_LATENT_REFERENCE_ROW_ID_INVALID")
        coefficients = tuple(getattr(row, "coefficients", ()) or ())
        if not coefficients:
            raise QualificationError("GRAPHICS_LATENT_REFERENCE_COEFFICIENTS_EMPTY")
        weights: list[tuple[int, float]] = []
        total = 0.0
        for proposal_id, raw_weight in coefficients:
            index = _proposal_index(str(proposal_id), len(proposal_xyz))
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0.0:
                raise QualificationError("GRAPHICS_LATENT_REFERENCE_WEIGHT_INVALID")
            total += weight
            weights.append((index, weight))
        if abs(total - 1.0) > 1.0e-6:
            raise QualificationError("GRAPHICS_LATENT_REFERENCE_WEIGHT_SUM_DRIFT")
        h = np.zeros((surface_h.shape[1],), dtype=np.float64)
        xyz = np.zeros((3,), dtype=np.float64)
        for index, weight in weights:
            h += proposal_h[index].astype(np.float64) * weight
            xyz += proposal_xyz[index] * weight
        row_by_candidate[candidate_id] = row
        final_h[candidate_id] = h.astype(np.float32)
        final_xyz[candidate_id] = xyz
        if len(weights) == 1 and abs(weights[0][1] - 1.0) <= 1.0e-9:
            identity_rows += 1
        else:
            repaired_rows += 1

    canonical_vertices = tuple(getattr(canonical_mesh, "vertices", ()) or ())
    if not canonical_vertices:
        raise QualificationError("GRAPHICS_LATENT_CANONICAL_MESH_EMPTY")
    canonical_h_rows = []
    canonical_xyz_rows = []
    seen_source_ids: set[str] = set()
    for vertex in canonical_vertices:
        source_id = str(getattr(vertex, "source_candidate_vertex_id", ""))
        if not source_id or source_id in seen_source_ids:
            raise QualificationError("GRAPHICS_LATENT_CANONICAL_SOURCE_ID_INVALID")
        seen_source_ids.add(source_id)
        if source_id not in final_h:
            raise QualificationError("GRAPHICS_LATENT_REFERENCE_COVERAGE_INCOMPLETE")
        canonical_h_rows.append(final_h[source_id])
        canonical_xyz_rows.append(final_xyz[source_id])

    canonical_h = np.asarray(canonical_h_rows, dtype=np.float32)
    canonical_xyz = np.asarray(canonical_xyz_rows, dtype=np.float64)
    if not np.isfinite(canonical_h).all() or not np.isfinite(canonical_xyz).all():
        raise QualificationError("GRAPHICS_LATENT_CANONICAL_TRANSPORT_NONFINITE")

    distance = nearest_distance.astype(np.float64, copy=False)
    transport_hash = _transport_hash(
        canonical_h=canonical_h,
        canonical_xyz=canonical_xyz,
        nearest_index=nearest_index,
        nearest_distance=distance,
    )
    diagnostics = GraphicsLatentTransportDiagnosticsV1(
        proposal_vertex_count=len(proposal_xyz),
        surface_witness_count=len(surface_xyz),
        canonical_vertex_count=len(canonical_vertices),
        latent_width=int(surface_h.shape[1]),
        nearest_distance_min=float(np.min(distance)),
        nearest_distance_median=float(np.median(distance)),
        nearest_distance_p95=float(np.quantile(distance, 0.95)),
        nearest_distance_max=float(np.max(distance)),
        repaired_interpolation_row_count=int(repaired_rows),
        exact_identity_row_count=int(identity_rows),
        transport_hash=transport_hash,
    )
    return GraphicsLatentTransportV1(
        canonical_h=canonical_h,
        canonical_xyz_tessa=canonical_xyz,
        proposal_h=proposal_h,
        proposal_xyz_tessa=proposal_xyz.copy(),
        proposal_nearest_surface_index=nearest_index.copy(),
        proposal_nearest_surface_distance=distance.copy(),
        diagnostics=diagnostics,
    )


__all__ = [
    "GraphicsLatentTransportDiagnosticsV1",
    "GraphicsLatentTransportV1",
    "transport_shape_h_to_canonical_mesh_v1",
]
