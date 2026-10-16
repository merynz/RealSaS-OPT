from __future__ import annotations

"""Training-only canonicalization for authored topology supervision.

TESSA V1 identifies vertices by quantized XYZ. Authored assets frequently split a
mechanically identical vertex for UV/material seams.  IRIS vNext may collapse a
seam split for topology supervision only when all three facts are proven:

1. the vertices occur in the same face-connected source component;
2. their rest positions are exactly identical in normalized IRIS coordinates;
3. their complete sampled authored trajectories agree within a tight tolerance.

This does not alter appearance truth, source ownership, authored mechanical truth,
or product geometry authority.  A coincident pair that separates under motion is
never welded.  After the safe weld we explicitly census residual TESSA coordinate
collisions and fail the cohort builder rather than weakening the tokenizer guard.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from models.tessa.v1.tokenization_v1 import quantize_tessa_xyz_v1
from models.tessa.v1.training_data_v1 import connected_face_components_v1


@dataclass(frozen=True)
class VNextTopologyTeacherMeshV1:
    vertices_iris: np.ndarray
    posed_vertices_iris: np.ndarray
    faces: np.ndarray
    metadata: dict[str, Any]


def _validated_inputs(rest_iris, posed_iris, faces):
    rest = np.asarray(rest_iris, dtype=np.float64)
    posed = np.asarray(posed_iris, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if rest.ndim != 2 or rest.shape[1] != 3 or not np.isfinite(rest).all():
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_REST_INVALID")
    if posed.ndim != 3 or posed.shape[1:] != rest.shape or not np.isfinite(posed).all():
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_POSED_INVALID")
    if posed.shape[0] <= 0:
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_POSED_EMPTY")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_FACES_INVALID")
    if np.any(f < 0) or np.any(f >= len(rest)):
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_FACE_INDEX_INVALID")
    if float(np.max(np.abs(rest))) > 1.000001:
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_REST_DOMAIN_INVALID")
    return rest, posed, f


def _rest_key(row: np.ndarray) -> tuple[float, float, float]:
    return tuple(0.0 if float(x) == 0.0 else float(x) for x in row)


def _trajectory_agrees(posed: np.ndarray, a: int, b: int, tolerance: float) -> bool:
    return float(np.max(np.abs(posed[:, int(a)] - posed[:, int(b)]))) <= float(tolerance)


def _canonicalize_component(
    rest: np.ndarray,
    posed: np.ndarray,
    faces: np.ndarray,
    face_ids: tuple[int, ...],
    *,
    trajectory_tolerance: float,
):
    source_faces = faces[np.asarray(face_ids, dtype=np.int64)]
    vertex_ids = tuple(sorted(set(map(int, source_faces.reshape(-1).tolist()))))
    by_position: dict[tuple[float, float, float], list[int]] = {}
    for vi in vertex_ids:
        by_position.setdefault(_rest_key(rest[vi]), []).append(vi)

    representative: dict[int, int] = {vi: vi for vi in vertex_ids}
    exact_duplicate_vertices = 0
    welded_vertices = 0
    motion_distinct_coincident_pairs = 0
    candidate_groups = 0
    welded_groups = 0
    for group in by_position.values():
        if len(group) <= 1:
            continue
        group = sorted(group)
        candidate_groups += 1
        exact_duplicate_vertices += len(group) - 1
        clusters: list[list[int]] = []
        for vi in group:
            placed = False
            for cluster in clusters:
                if all(
                    _trajectory_agrees(posed, vi, other, trajectory_tolerance)
                    for other in cluster
                ):
                    cluster.append(vi)
                    placed = True
                    break
            if not placed:
                clusters.append([vi])
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                motion_distinct_coincident_pairs += len(clusters[i]) * len(clusters[j])
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            rep = min(cluster)
            welded_groups += 1
            welded_vertices += len(cluster) - 1
            for vi in cluster:
                representative[vi] = rep

    mapped_faces = [tuple(representative[int(v)] for v in row) for row in source_faces]
    kept_faces: list[tuple[int, int, int]] = []
    seen_face_keys: set[tuple[int, int, int]] = set()
    degenerate_removed = 0
    duplicate_removed = 0
    for face in mapped_faces:
        if len(set(face)) != 3:
            degenerate_removed += 1
            continue
        key = tuple(sorted(face))
        if key in seen_face_keys:
            duplicate_removed += 1
            continue
        seen_face_keys.add(key)
        kept_faces.append(face)
    if not kept_faces:
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_COMPONENT_EMPTIED_BY_SAFE_WELD")

    used = tuple(sorted(set(v for face in kept_faces for v in face)))
    remap = {source_vi: local_vi for local_vi, source_vi in enumerate(used)}
    local_faces = np.asarray(
        [[remap[v] for v in face] for face in kept_faces], dtype=np.int64
    )
    local_rest = rest[np.asarray(used, dtype=np.int64)]
    local_posed = posed[:, np.asarray(used, dtype=np.int64)]
    return local_rest, local_posed, local_faces, {
        "source_face_count": int(len(source_faces)),
        "source_vertex_count": int(len(vertex_ids)),
        "output_face_count": int(len(local_faces)),
        "output_vertex_count": int(len(local_rest)),
        "exact_rest_duplicate_vertex_count": int(exact_duplicate_vertices),
        "motion_equivalent_welded_vertex_count": int(welded_vertices),
        "motion_distinct_coincident_pair_count": int(motion_distinct_coincident_pairs),
        "candidate_duplicate_group_count": int(candidate_groups),
        "welded_group_count": int(welded_groups),
        "degenerate_face_count_removed": int(degenerate_removed),
        "duplicate_face_count_removed": int(duplicate_removed),
    }


def _quantized_collision_census(vertices_iris: np.ndarray, faces: np.ndarray, bins: int):
    q = quantize_tessa_xyz_v1(np.asarray(vertices_iris, dtype=np.float64) * 0.5, int(bins))
    components = connected_face_components_v1(faces)
    groups = []
    for component_index, face_ids in enumerate(components):
        vertex_ids = sorted(
            set(map(int, faces[np.asarray(face_ids, dtype=np.int64)].reshape(-1).tolist()))
        )
        by_q: dict[tuple[int, int, int], list[int]] = {}
        for vi in vertex_ids:
            by_q.setdefault(tuple(map(int, q[vi].tolist())), []).append(vi)
        for key, ids in sorted(by_q.items()):
            if len(ids) > 1:
                groups.append(
                    {
                        "component_index": int(component_index),
                        "quantized_xyz": list(key),
                        "vertex_indices": list(map(int, ids)),
                        "vertex_count": int(len(ids)),
                    }
                )
    return {
        "coordinate_bins": int(bins),
        "collision_group_count": int(len(groups)),
        "colliding_vertex_excess_count": int(sum(row["vertex_count"] - 1 for row in groups)),
        "examples": groups[:16],
    }


def canonicalize_vnext_topology_teacher_v1(
    rest_iris,
    posed_iris,
    faces,
    *,
    coordinate_bins: int,
    trajectory_tolerance: float = 1.0e-7,
) -> VNextTopologyTeacherMeshV1:
    """Return a training-only mechanically canonical topology teacher."""
    if int(coordinate_bins) < 2:
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_COORDINATE_BINS_INVALID")
    if not (0.0 <= float(trajectory_tolerance) <= 1.0e-3):
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_TRAJECTORY_TOLERANCE_INVALID")
    rest, posed, source_faces = _validated_inputs(rest_iris, posed_iris, faces)
    components = connected_face_components_v1(source_faces)
    if not components:
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_COMPONENTS_EMPTY")

    vertices_out = []
    posed_out = []
    faces_out = []
    component_rows = []
    vertex_offset = 0
    totals = {
        "exact_rest_duplicate_vertex_count": 0,
        "motion_equivalent_welded_vertex_count": 0,
        "motion_distinct_coincident_pair_count": 0,
        "candidate_duplicate_group_count": 0,
        "welded_group_count": 0,
        "degenerate_face_count_removed": 0,
        "duplicate_face_count_removed": 0,
    }
    for component_index, face_ids in enumerate(components):
        local_rest, local_posed, local_faces, row = _canonicalize_component(
            rest,
            posed,
            source_faces,
            face_ids,
            trajectory_tolerance=float(trajectory_tolerance),
        )
        faces_out.append(local_faces + int(vertex_offset))
        vertices_out.append(local_rest)
        posed_out.append(local_posed)
        vertex_offset += int(len(local_rest))
        row = {"component_index": int(component_index), **row}
        component_rows.append(row)
        for key in totals:
            totals[key] += int(row[key])

    canonical_rest = np.concatenate(vertices_out, axis=0)
    canonical_posed = np.concatenate(posed_out, axis=1)
    canonical_faces = np.concatenate(faces_out, axis=0)
    collision = _quantized_collision_census(
        canonical_rest, canonical_faces, int(coordinate_bins)
    )
    metadata = {
        "schema": "RealSaS.IRISVNextTopologyTeacherCanonicalization.v1",
        "training_only": True,
        "product_geometry_authority": False,
        "appearance_or_material_identity_used_for_weld": False,
        "weld_rule": "SAME_EDGE_CONNECTED_COMPONENT__EXACT_REST_XYZ__ALL_SAMPLED_AUTHORED_TRAJECTORIES_WITHIN_TOLERANCE",
        "trajectory_tolerance_iris": float(trajectory_tolerance),
        "source_vertex_count": int(len(rest)),
        "source_face_count": int(len(source_faces)),
        "output_vertex_count": int(len(canonical_rest)),
        "output_face_count": int(len(canonical_faces)),
        "component_count": int(len(components)),
        **{key: int(value) for key, value in totals.items()},
        "quantized_collision_census": collision,
        "components": component_rows,
    }
    return VNextTopologyTeacherMeshV1(
        vertices_iris=canonical_rest,
        posed_vertices_iris=canonical_posed,
        faces=canonical_faces,
        metadata=metadata,
    )
