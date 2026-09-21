from __future__ import annotations

"""Quality evaluation for total Complete Appearance Authority assets."""

import math

import numpy as np
from scipy.spatial import cKDTree

from .appearance_authority_v2 import CAA_PROVENANCE
from .appearance_bake_v2 import straight_rgba_to_premultiplied_float
from .types import QualificationError


def rgba_l1_premultiplied(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = straight_rgba_to_premultiplied_float(np.asarray(a, dtype=np.uint8))
    bb = straight_rgba_to_premultiplied_float(np.asarray(b, dtype=np.uint8))
    if aa.shape != bb.shape or aa.shape[-1] != 4:
        raise QualificationError("CAA_QUALITY_RGBA_SHAPE_DRIFT")
    return np.mean(np.abs(aa - bb), axis=-1)


def _view_order(target: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            (view for view in range(8) if view != target),
            key=lambda view: (
                min(abs(view - target), 8 - abs(view - target)),
                view,
            ),
        )
    )


def _structured_band_mask(
    direct_valid: np.ndarray,
    source_xy: np.ndarray,
    *,
    view_index: int,
    band_fraction: float,
) -> np.ndarray:
    valid = np.asarray(direct_valid, dtype=bool)
    xy = np.asarray(source_xy, dtype=np.float64)
    if valid.ndim != 1 or xy.shape != (len(valid), 2):
        raise QualificationError("CAA_HOLDOUT_SOURCE_COORD_SHAPE_INVALID")
    indices = np.flatnonzero(valid & np.isfinite(xy).all(axis=1))
    mask = np.zeros(len(valid), dtype=bool)
    if len(indices) == 0:
        return mask
    axis = int(view_index) % 2
    values = xy[indices, axis]
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    if not math.isfinite(span) or span <= 0.0:
        return mask
    width = span * float(band_fraction)
    # Alternate low/high image-domain edge deterministically across views so the
    # holdout is silhouette-adjacent rather than an easy central interpolation
    # strip. The opposite edge remains available as source support.
    if (int(view_index) // 2) % 2 == 0:
        mask[indices] = values <= lo + width
    else:
        mask[indices] = values >= hi - width
    return mask


def structured_holdout_metrics(
    *,
    direct_valid: np.ndarray,
    direct_rgba: np.ndarray,
    source_xy: np.ndarray,
    sample_positions: np.ndarray,
    sample_component_index: np.ndarray,
    band_fraction: float,
) -> dict:
    direct_valid = np.asarray(direct_valid, dtype=bool)
    direct_rgba = np.asarray(direct_rgba, dtype=np.uint8)
    source_xy = np.asarray(source_xy, dtype=np.float32)
    positions = np.asarray(sample_positions, dtype=np.float64)
    component = np.asarray(sample_component_index, dtype=np.int32)
    if direct_valid.ndim != 2 or direct_valid.shape[0] != 8:
        raise QualificationError("CAA_HOLDOUT_DIRECT_VALID_SHAPE_INVALID")
    n = direct_valid.shape[1]
    if (
        direct_rgba.shape != (8, n, 4)
        or source_xy.shape != (8, n, 2)
        or positions.shape != (n, 3)
        or component.shape != (n,)
    ):
        raise QualificationError("CAA_HOLDOUT_ARRAY_SHAPE_DRIFT")
    fraction = float(band_fraction)
    if not math.isfinite(fraction) or not (0.02 <= fraction <= 0.5):
        raise QualificationError("CAA_HOLDOUT_BAND_FRACTION_INVALID")

    errors = []
    per_view = []
    for target in range(8):
        holdout = _structured_band_mask(
            direct_valid[target],
            source_xy[target],
            view_index=target,
            band_fraction=fraction,
        )
        held = np.flatnonzero(holdout)
        if len(held) == 0:
            per_view.append(
                {"view_index": target, "holdout_sample_count": 0}
            )
            continue

        available = direct_valid[target] & ~holdout
        predicted = np.zeros((n, 4), dtype=np.uint8)
        has = available.copy()
        predicted[available] = direct_rgba[target, available]

        for donor in _view_order(target):
            take = (~has) & direct_valid[donor]
            if np.any(take):
                predicted[take] = direct_rgba[donor, take]
                has[take] = True

        for component_id in np.unique(component):
            need = holdout & ~has & (component == component_id)
            if not np.any(need):
                continue
            donors = has & (component == component_id)
            if not np.any(donors):
                continue
            donor_indices = np.flatnonzero(donors)
            query_indices = np.flatnonzero(need)
            tree = cKDTree(positions[donor_indices])
            _distance, nearest = tree.query(positions[query_indices], k=1)
            predicted[query_indices] = predicted[
                donor_indices[np.asarray(nearest, dtype=np.int64)]
            ]
            has[query_indices] = True

        need = holdout & ~has
        if np.any(need):
            donors = np.flatnonzero(has)
            if len(donors) == 0:
                raise QualificationError("CAA_HOLDOUT_NO_AVAILABLE_DONOR")
            query = np.flatnonzero(need)
            tree = cKDTree(positions[donors])
            _distance, nearest = tree.query(positions[query], k=1)
            predicted[query] = predicted[
                donors[np.asarray(nearest, dtype=np.int64)]
            ]
            has[query] = True

        if not np.all(has[held]):
            raise QualificationError("CAA_HOLDOUT_PREDICTION_NOT_TOTAL")
        error = rgba_l1_premultiplied(
            predicted[held],
            direct_rgba[target, held],
        )
        errors.extend(map(float, error))
        per_view.append(
            {
                "view_index": target,
                "holdout_sample_count": int(len(held)),
                "mean_rgba_l1": float(np.mean(error)),
                "p95_rgba_l1": float(np.quantile(error, 0.95)),
            }
        )

    values = np.asarray(errors, dtype=np.float64)
    return {
        "mode": "TARGET_VIEW_SILHOUETTE_ADJACENT_EDGE_BAND_V2",
        "band_fraction": fraction,
        "sample_count": int(len(values)),
        "mean_rgba_l1": float(np.mean(values)) if len(values) else 0.0,
        "p95_rgba_l1": float(np.quantile(values, 0.95)) if len(values) else 0.0,
        "per_view": per_view,
    }


def cross_view_source_compatibility_metrics(
    *,
    direct_valid: np.ndarray,
    direct_rgba: np.ndarray,
    sample_component_index: np.ndarray,
    color_conflict_cut_rgba_l1: float = 0.35,
    alpha_conflict_cut: float = 0.25,
) -> dict:
    valid = np.asarray(direct_valid, dtype=bool)
    rgba = np.asarray(direct_rgba, dtype=np.uint8)
    component = np.asarray(sample_component_index, dtype=np.int32)
    color_cut = float(color_conflict_cut_rgba_l1)
    alpha_cut = float(alpha_conflict_cut)
    if valid.ndim != 2 or valid.shape[0] != 8:
        raise QualificationError("CAA_CROSS_VIEW_VALID_SHAPE_INVALID")
    sample_count = valid.shape[1]
    if rgba.shape != (8, sample_count, 4) or component.shape != (sample_count,):
        raise QualificationError("CAA_CROSS_VIEW_ARRAY_SHAPE_DRIFT")
    if not (0.0 <= color_cut <= 1.0 and 0.0 <= alpha_cut <= 1.0):
        raise QualificationError("CAA_CROSS_VIEW_CONFLICT_CUT_INVALID")

    def summarize(errors: np.ndarray, alpha: np.ndarray) -> dict:
        count = int(len(errors))
        return {
            "shared_direct_sample_count": count,
            "mean_premultiplied_rgba_l1": float(np.mean(errors)) if count else 0.0,
            "p95_premultiplied_rgba_l1": float(np.quantile(errors, 0.95)) if count else 0.0,
            "p99_premultiplied_rgba_l1": float(np.quantile(errors, 0.99)) if count else 0.0,
            "max_premultiplied_rgba_l1": float(np.max(errors)) if count else 0.0,
            "color_conflict_fraction": (
                float(np.mean(errors > color_cut)) if count else 0.0
            ),
            "mean_alpha_abs": float(np.mean(alpha)) if count else 0.0,
            "p95_alpha_abs": float(np.quantile(alpha, 0.95)) if count else 0.0,
            "p99_alpha_abs": float(np.quantile(alpha, 0.99)) if count else 0.0,
            "max_alpha_abs": float(np.max(alpha)) if count else 0.0,
            "alpha_conflict_fraction": (
                float(np.mean(alpha > alpha_cut)) if count else 0.0
            ),
        }

    pair_rows = []
    component_rows = []
    all_errors = []
    all_alpha = []
    for left in range(8):
        right = (left + 1) % 8
        shared = valid[left] & valid[right]
        indices = np.flatnonzero(shared)
        if len(indices):
            errors = rgba_l1_premultiplied(
                rgba[left, indices],
                rgba[right, indices],
            )
            alpha = np.abs(
                rgba[left, indices, 3].astype(np.float64)
                - rgba[right, indices, 3].astype(np.float64)
            ) / 255.0
            all_errors.extend(map(float, errors))
            all_alpha.extend(map(float, alpha))
        else:
            errors = np.asarray([], dtype=np.float64)
            alpha = np.asarray([], dtype=np.float64)
        row = {
            "left_view_index": left,
            "right_view_index": right,
            **summarize(errors, alpha),
        }
        pair_rows.append(row)
        for component_id in sorted(set(map(int, component))):
            local = shared & (component == component_id)
            local_indices = np.flatnonzero(local)
            if not len(local_indices):
                continue
            local_error = rgba_l1_premultiplied(
                rgba[left, local_indices],
                rgba[right, local_indices],
            )
            local_alpha = np.abs(
                rgba[left, local_indices, 3].astype(np.float64)
                - rgba[right, local_indices, 3].astype(np.float64)
            ) / 255.0
            component_rows.append(
                {
                    "left_view_index": left,
                    "right_view_index": right,
                    "component_index": component_id,
                    **summarize(local_error, local_alpha),
                }
            )

    values = np.asarray(all_errors, dtype=np.float64)
    alpha_values = np.asarray(all_alpha, dtype=np.float64)
    return {
        "mode": "ADJACENT_8VIEW_SHARED_CANONICAL_DIRECT_SOURCE_V2",
        "pair_count": 8,
        **summarize(values, alpha_values),
        "per_pair": pair_rows,
        "per_pair_component": component_rows,
        "color_conflict_cut_rgba_l1": color_cut,
        "alpha_conflict_cut": alpha_cut,
        "raw_rgb_equality_required": False,
        "measurement_is_compatibility_not_color_authority": True,
    }

def _triangle_lattice_neighbors(tile_resolution: int) -> tuple[tuple[int, int], ...]:
    resolution = int(tile_resolution)
    index = {}
    cursor = 0
    for j in range(resolution):
        for i in range(resolution - j):
            index[(i, j)] = cursor
            cursor += 1
    pairs = set()
    steps = ((1, 0), (0, 1), (1, -1))
    for (i, j), a in index.items():
        for di, dj in steps:
            key = (i + di, j + dj)
            if key in index:
                b = index[key]
                pairs.add((min(a, b), max(a, b)))
    return tuple(sorted(pairs))


def provenance_boundary_metrics(
    *,
    rgba: np.ndarray,
    provenance: np.ndarray,
    source_view: np.ndarray | None = None,
    sample_positions: np.ndarray,
    sample_face_index: np.ndarray,
    face_count: int,
    tile_resolution: int,
) -> dict:
    rgba = np.asarray(rgba, dtype=np.uint8)
    provenance = np.asarray(provenance, dtype=np.uint8)
    donor_view = (
        None
        if source_view is None
        else np.asarray(source_view, dtype=np.int16)
    )
    positions = np.asarray(sample_positions, dtype=np.float64)
    face_index = np.asarray(sample_face_index, dtype=np.int32)
    if rgba.ndim != 3 or rgba.shape[0] != 8 or rgba.shape[2] != 4:
        raise QualificationError("CAA_SEAM_RGBA_SHAPE_INVALID")
    n = rgba.shape[1]
    if provenance.shape != (8, n) or positions.shape != (n, 3) or face_index.shape != (n,):
        raise QualificationError("CAA_SEAM_ARRAY_SHAPE_DRIFT")
    if donor_view is not None and donor_view.shape != (8, n):
        raise QualificationError("CAA_SEAM_SOURCE_VIEW_SHAPE_DRIFT")
    per_face = int(tile_resolution) * (int(tile_resolution) + 1) // 2
    if n != int(face_count) * per_face:
        raise QualificationError("CAA_SEAM_FACE_SAMPLE_ACCOUNTING_DRIFT")

    local_pairs = _triangle_lattice_neighbors(tile_resolution)
    pair_set: set[tuple[int, int]] = set()
    shared_edge_pair_set: set[tuple[int, int]] = set()
    for face in range(int(face_count)):
        base = face * per_face
        for a, b in local_pairs:
            pair_set.add((base + a, base + b))

    # Shared-edge samples from adjacent faces occupy identical canonical positions.
    buckets: dict[tuple[int, int, int], list[int]] = {}
    scale = 1.0e8
    for index, point in enumerate(positions):
        key = tuple(int(round(float(value) * scale)) for value in point)
        buckets.setdefault(key, []).append(index)
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a = indices[i]
                b = indices[j]
                if face_index[a] != face_index[b]:
                    pair = (min(a, b), max(a, b))
                    pair_set.add(pair)
                    shared_edge_pair_set.add(pair)

    adjacency: dict[int, set[int]] = {index: set() for index in range(n)}
    for a, b in pair_set:
        adjacency[a].add(b)
        adjacency[b].add(a)

    errors = []
    gradient_jumps = []
    pairs_by_view = []
    for view in range(8):
        count = 0
        values = []
        view_gradient_jumps = []
        donor_switch_count = 0
        shared_edge_count = 0
        shared_edge_errors = []
        for a, b in pair_set:
            shared_edge = (a, b) in shared_edge_pair_set
            provenance_differs = provenance[view, a] != provenance[view, b]
            donor_differs = bool(
                donor_view is not None
                and int(donor_view[view, a]) != int(donor_view[view, b])
            )
            if not shared_edge and not provenance_differs and not donor_differs:
                continue
            error = float(
                rgba_l1_premultiplied(
                    rgba[view, [a]],
                    rgba[view, [b]],
                )[0]
            )
            values.append(error)
            errors.append(error)
            count += 1
            if shared_edge:
                shared_edge_count += 1
                shared_edge_errors.append(error)
            if donor_differs:
                donor_switch_count += 1

            same_a = [
                neighbor
                for neighbor in adjacency[a]
                if neighbor != b
                and provenance[view, neighbor] == provenance[view, a]
            ]
            same_b = [
                neighbor
                for neighbor in adjacency[b]
                if neighbor != a
                and provenance[view, neighbor] == provenance[view, b]
            ]
            if same_a and same_b:
                grad_a = float(
                    np.mean(
                        rgba_l1_premultiplied(
                            np.repeat(rgba[view, [a]], len(same_a), axis=0),
                            rgba[view, same_a],
                        )
                    )
                )
                grad_b = float(
                    np.mean(
                        rgba_l1_premultiplied(
                            np.repeat(rgba[view, [b]], len(same_b), axis=0),
                            rgba[view, same_b],
                        )
                    )
                )
                jump = abs(grad_a - grad_b)
                view_gradient_jumps.append(jump)
                gradient_jumps.append(jump)

        pairs_by_view.append(
            {
                "view_index": view,
                "boundary_pair_count": count,
                "mean_rgba_l1": float(np.mean(values)) if values else 0.0,
                "p95_rgba_l1": float(np.quantile(values, 0.95)) if values else 0.0,
                "gradient_pair_count": int(len(view_gradient_jumps)),
                "shared_edge_pair_count": int(shared_edge_count),
                "shared_edge_mean_rgba_l1": float(np.mean(shared_edge_errors)) if shared_edge_errors else 0.0,
                "shared_edge_p95_rgba_l1": float(np.quantile(shared_edge_errors, 0.95)) if shared_edge_errors else 0.0,
                "donor_view_switch_pair_count": int(donor_switch_count),
                "mean_gradient_jump": float(np.mean(view_gradient_jumps)) if view_gradient_jumps else 0.0,
                "p95_gradient_jump": float(np.quantile(view_gradient_jumps, 0.95)) if view_gradient_jumps else 0.0,
            }
        )
    arr = np.asarray(errors, dtype=np.float64)
    grad = np.asarray(gradient_jumps, dtype=np.float64)
    return {
        "boundary_pair_count": int(len(arr)),
        "mean_rgba_l1": float(np.mean(arr)) if len(arr) else 0.0,
        "p95_rgba_l1": float(np.quantile(arr, 0.95)) if len(arr) else 0.0,
        "gradient_pair_count": int(len(grad)),
        "mean_gradient_jump": float(np.mean(grad)) if len(grad) else 0.0,
        "p95_gradient_jump": float(np.quantile(grad, 0.95)) if len(grad) else 0.0,
        "per_view": pairs_by_view,
        "includes_shared_face_edges": True,
        "shared_face_edges_are_compared_even_when_provenance_matches": True,
        "source_view_identity_consumed": donor_view is not None,
        "donor_view_switch_pair_count": int(
            sum(row["donor_view_switch_pair_count"] for row in pairs_by_view)
        ),
    }
