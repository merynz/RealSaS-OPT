from __future__ import annotations

"""Training-only canonicalization for authored topology supervision.

TESSA V1 identifies vertices by quantized XYZ. Authored assets frequently split a
mechanically identical vertex for UV/material seams. IRIS vNext first collapses
only motion-equivalent exact seam duplicates. Any remaining within-component
TESSA coordinate collision is then resolved by a deterministic, bounded,
training-only identity lift into a nearby free quantization cell.

The lift preserves vertex count and face connectivity, applies the same tiny
translation to the complete sampled authored trajectory for a moved vertex, and
never changes appearance truth, source ownership, authored mechanical truth, or
product geometry authority. Residual coordinate collisions still fail closed.
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


def _project_iris_point_into_quantized_cell(
    point_iris: np.ndarray,
    qxyz: tuple[int, int, int],
    bins: int,
) -> np.ndarray:
    """Return the minimum robust IRIS-space move into one TESSA cell.

    Bounds are constructed directly in IRIS space and inset by a tiny floating
    margin. Doing ``nextafter`` in TESSA space before the IRIS x2 conversion can
    lose the one-ULP inset when the value is scaled back for quantization.
    """
    point = np.asarray(point_iris, dtype=np.float64)
    q = np.asarray(qxyz, dtype=np.int64)
    lo = 2.0 * (q.astype(np.float64) / float(bins) - 0.5)
    hi = 2.0 * ((q.astype(np.float64) + 1.0) / float(bins) - 0.5)
    margin = np.finfo(np.float64).eps * 32.0
    inner_lo = lo + margin
    inner_hi = hi - margin
    if np.any(inner_lo >= inner_hi):
        raise AssertionError("VNEXT_TOPOLOGY_TEACHER_IDENTITY_LIFT_CELL_COLLAPSED")
    out = np.minimum(np.maximum(point, inner_lo), inner_hi)
    check = quantize_tessa_xyz_v1(out.reshape(1, 3) * 0.5, int(bins))[0]
    if tuple(map(int, check.tolist())) != tuple(map(int, qxyz)):
        raise AssertionError("VNEXT_TOPOLOGY_TEACHER_IDENTITY_LIFT_QUANTIZATION_DRIFT")
    return out


def _candidate_quantized_cells(
    base: tuple[int, int, int],
    bins: int,
    max_radius_bins: int,
):
    bx, by, bz = map(int, base)
    for radius in range(1, int(max_radius_bins) + 1):
        shell = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy), abs(dz)) != radius:
                        continue
                    q = (bx + dx, by + dy, bz + dz)
                    if min(q) < 0 or max(q) >= int(bins):
                        continue
                    shell.append((dx, dy, dz, q))
        shell.sort(
            key=lambda row: (
                row[0] * row[0] + row[1] * row[1] + row[2] * row[2],
                abs(row[0]) + abs(row[1]) + abs(row[2]),
                row[3],
            )
        )
        for row in shell:
            yield radius, row[3]


def _injective_quantized_identity_lift(
    vertices_iris: np.ndarray,
    posed_vertices_iris: np.ndarray,
    faces: np.ndarray,
    *,
    bins: int,
    max_radius_bins: int = 2,
):
    """Make quantized vertex identity injective per face-connected component.

    Original quantized cells are reserved. Within each colliding group one anchor
    remains untouched; every additional vertex is translated by the minimum
    amount needed to enter a nearby unused cell. The same translation is applied
    to every sampled pose of that vertex so authored displacement trajectories
    remain unchanged.
    """
    if int(max_radius_bins) < 1:
        raise ValueError("VNEXT_TOPOLOGY_TEACHER_IDENTITY_LIFT_RADIUS_INVALID")
    rest = np.asarray(vertices_iris, dtype=np.float64).copy()
    posed = np.asarray(posed_vertices_iris, dtype=np.float64).copy()
    source_rest = rest.copy()
    source_q = quantize_tessa_xyz_v1(source_rest * 0.5, int(bins))
    components = connected_face_components_v1(faces)

    moved = []
    component_rows = []
    for component_index, face_ids in enumerate(components):
        vertex_ids = sorted(
            set(map(int, faces[np.asarray(face_ids, dtype=np.int64)].reshape(-1).tolist()))
        )
        by_q: dict[tuple[int, int, int], list[int]] = {}
        for vi in vertex_ids:
            by_q.setdefault(tuple(map(int, source_q[vi].tolist())), []).append(vi)
        collision_groups = [
            (key, sorted(ids)) for key, ids in sorted(by_q.items()) if len(ids) > 1
        ]
        used = set(by_q.keys())
        component_moved = 0
        component_radius = 0

        for base_key, ids in collision_groups:
            center_tessa = (np.asarray(base_key, dtype=np.float64) + 0.5) / float(bins) - 0.5
            center_iris = center_tessa * 2.0
            anchor = min(
                ids,
                key=lambda vi: (
                    float(np.max(np.abs(source_rest[vi] - center_iris))),
                    float(np.linalg.norm(source_rest[vi] - center_iris)),
                    int(vi),
                ),
            )
            for vi in ids:
                if vi == anchor:
                    continue
                best = None
                for radius, qxyz in _candidate_quantized_cells(
                    base_key, int(bins), int(max_radius_bins)
                ):
                    if qxyz in used:
                        continue
                    projected = _project_iris_point_into_quantized_cell(
                        source_rest[vi], qxyz, int(bins)
                    )
                    delta = projected - source_rest[vi]
                    score = (
                        float(np.max(np.abs(delta))),
                        float(np.linalg.norm(delta)),
                        int(radius),
                        tuple(map(int, qxyz)),
                    )
                    if best is None or score < best[0]:
                        best = (score, projected, delta, qxyz, radius)
                if best is None:
                    raise ValueError(
                        "VNEXT_TOPOLOGY_TEACHER_IDENTITY_LIFT_EXHAUSTED:"
                        f"component={component_index}:vertex={vi}:base={base_key}:"
                        f"radius={int(max_radius_bins)}"
                    )
                _, projected, delta, qxyz, radius = best
                rest[vi] = projected
                posed[:, vi] = posed[:, vi] + delta
                used.add(tuple(map(int, qxyz)))
                component_moved += 1
                component_radius = max(component_radius, int(radius))
                moved.append(
                    {
                        "component_index": int(component_index),
                        "vertex_index": int(vi),
                        "source_quantized_xyz": list(map(int, base_key)),
                        "lifted_quantized_xyz": list(map(int, qxyz)),
                        "max_abs_displacement_iris": float(np.max(np.abs(delta))),
                        "l2_displacement_iris": float(np.linalg.norm(delta)),
                    }
                )

        component_rows.append(
            {
                "component_index": int(component_index),
                "source_collision_group_count": int(len(collision_groups)),
                "moved_vertex_count": int(component_moved),
                "max_search_radius_bins_used": int(component_radius),
            }
        )

    final_q = quantize_tessa_xyz_v1(rest * 0.5, int(bins))
    for component_index, face_ids in enumerate(components):
        vertex_ids = sorted(
            set(map(int, faces[np.asarray(face_ids, dtype=np.int64)].reshape(-1).tolist()))
        )
        keys = [tuple(map(int, final_q[vi].tolist())) for vi in vertex_ids]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "VNEXT_TOPOLOGY_TEACHER_IDENTITY_LIFT_RESIDUAL_COLLISION:"
                f"component={component_index}"
            )

    return rest, posed, {
        "schema": "RealSaS.IRISVNextTopologyTeacherQuantizedIdentityLift.v1",
        "training_only": True,
        "product_geometry_authority": False,
        "face_connectivity_changed": False,
        "vertex_count_changed": False,
        "coordinate_bins": int(bins),
        "max_search_radius_bins": int(max_radius_bins),
        "moved_vertex_count": int(len(moved)),
        "max_abs_displacement_iris": float(
            max((row["max_abs_displacement_iris"] for row in moved), default=0.0)
        ),
        "max_l2_displacement_iris": float(
            max((row["l2_displacement_iris"] for row in moved), default=0.0)
        ),
        "components": component_rows,
        "examples": moved[:16],
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

    collision_before_lift = _quantized_collision_census(
        canonical_rest, canonical_faces, int(coordinate_bins)
    )
    lifted_rest, lifted_posed, lift_metadata = _injective_quantized_identity_lift(
        canonical_rest,
        canonical_posed,
        canonical_faces,
        bins=int(coordinate_bins),
    )
    collision = _quantized_collision_census(
        lifted_rest, canonical_faces, int(coordinate_bins)
    )
    if int(collision["collision_group_count"]) != 0:
        raise ValueError(
            "VNEXT_TOPOLOGY_TEACHER_RESIDUAL_QUANTIZED_COLLISION_AFTER_IDENTITY_LIFT"
        )

    metadata = {
        "schema": "RealSaS.IRISVNextTopologyTeacherCanonicalization.v2",
        "training_only": True,
        "product_geometry_authority": False,
        "appearance_or_material_identity_used_for_weld": False,
        "weld_rule": "SAME_EDGE_CONNECTED_COMPONENT__EXACT_REST_XYZ__ALL_SAMPLED_AUTHORED_TRAJECTORIES_WITHIN_TOLERANCE",
        "quantized_identity_lift_rule": "PER_COMPONENT__RESERVE_SOURCE_CELLS__MINIMUM_DISPLACEMENT_TO_NEARBY_FREE_TESSA_CELL__PRESERVE_FACES_AND_VERTEX_COUNT",
        "trajectory_tolerance_iris": float(trajectory_tolerance),
        "source_vertex_count": int(len(rest)),
        "source_face_count": int(len(source_faces)),
        "output_vertex_count": int(len(lifted_rest)),
        "output_face_count": int(len(canonical_faces)),
        "component_count": int(len(components)),
        **{key: int(value) for key, value in totals.items()},
        "quantized_collision_census_before_identity_lift": collision_before_lift,
        "quantized_identity_lift": lift_metadata,
        "quantized_collision_census": collision,
        "components": component_rows,
    }
    return VNextTopologyTeacherMeshV1(
        vertices_iris=lifted_rest,
        posed_vertices_iris=lifted_posed,
        faces=canonical_faces,
        metadata=metadata,
    )
