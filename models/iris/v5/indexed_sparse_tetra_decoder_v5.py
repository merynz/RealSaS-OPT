from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Any

import numpy as np


TETRAHEDRA_V5 = np.asarray(
    [
        [0, 5, 1, 7],
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
    ],
    dtype=np.int64,
)

TETRA_EDGES_V5 = np.asarray(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
    dtype=np.int64,
)

TRI_TABLE_V5 = np.asarray(
    [
        [-1, -1, -1, -1, -1, -1],
        [1, 0, 2, -1, -1, -1],
        [4, 0, 3, -1, -1, -1],
        [1, 4, 2, 1, 3, 4],
        [3, 1, 5, -1, -1, -1],
        [2, 3, 0, 2, 5, 3],
        [1, 4, 0, 1, 5, 4],
        [4, 2, 5, -1, -1, -1],
        [4, 5, 2, -1, -1, -1],
        [4, 1, 0, 4, 5, 1],
        [3, 2, 0, 3, 5, 2],
        [1, 3, 5, -1, -1, -1],
        [4, 1, 2, 4, 3, 1],
        [3, 0, 4, -1, -1, -1],
        [2, 0, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ],
    dtype=np.int64,
)

NUM_TRIANGLES_V5 = np.asarray(
    [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0],
    dtype=np.int64,
)

CUBE8_V5 = np.asarray(
    [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ],
    dtype=np.int32,
)

SUBCUBE_OFFSETS_V5 = np.asarray(
    [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)],
    dtype=np.int32,
)

LOCAL27_V5 = np.asarray(
    [(a, b, c) for a in (0, 1, 2) for b in (0, 1, 2) for c in (0, 1, 2)],
    dtype=np.int32,
)


class SparseSurfaceTruncationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SparseRegularTetraPolicyV5:
    """F4-selected undeformed T512 -> sparse T1024 -> indexed MT contract."""

    base_cells: int = 512
    fine_cells: int = 1024
    query_chunk: int = 262144
    cell_chunk: int = 10000
    refine_slab_cells: int = 8
    fine_gid_cell_chunk: int = 50000
    normal_chunk: int = 131072
    refine_band_multiplier: float = math.sqrt(3.0)
    normal_epsilon_fine_steps: float = 0.5
    fail_on_boundary_crossing: bool = True

    def validate(self) -> None:
        if int(self.base_cells) < 2:
            raise ValueError("base_cells must be >= 2")
        if int(self.fine_cells) != 2 * int(self.base_cells):
            raise ValueError("fine_cells must equal 2*base_cells")
        for name in (
            "query_chunk",
            "cell_chunk",
            "refine_slab_cells",
            "fine_gid_cell_chunk",
            "normal_chunk",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            not math.isfinite(float(self.refine_band_multiplier))
            or float(self.refine_band_multiplier) <= 0.0
        ):
            raise ValueError("refine_band_multiplier must be finite and positive")
        if (
            not math.isfinite(float(self.normal_epsilon_fine_steps))
            or float(self.normal_epsilon_fine_steps) <= 0.0
        ):
            raise ValueError("normal_epsilon_fine_steps must be finite and positive")

    @property
    def base_step(self) -> float:
        return 2.0 / float(self.base_cells)

    @property
    def fine_step(self) -> float:
        return 2.0 / float(self.fine_cells)

    @property
    def refine_band(self) -> float:
        return float(self.refine_band_multiplier) * self.base_step

    @property
    def normal_epsilon(self) -> float:
        return float(self.normal_epsilon_fine_steps) * self.fine_step


@dataclass(frozen=True)
class IndexedSparseTetraMeshV5:
    vertices_normalized: np.ndarray
    faces: np.ndarray
    implicit_normals: np.ndarray
    diagnostics: dict[str, Any]
    decoder_id: str = (
        "RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT.v5"
    )


def _as_query_result(values: Any, expected: int) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    out = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(out) != int(expected):
        raise ValueError(
            f"query_fn returned {len(out)} values for {expected} query points"
        )
    if not np.isfinite(out).all():
        raise ValueError("query_fn returned non-finite values")
    return out


def _query_numpy(
    query_fn: Callable[[np.ndarray], Any],
    points: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    q = np.asarray(points, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("query points must be [N,3]")
    out = np.empty(len(q), dtype=np.float32)
    for start in range(0, len(q), int(chunk_size)):
        end = min(len(q), start + int(chunk_size))
        out[start:end] = _as_query_result(query_fn(q[start:end]), end - start)
    return out


def _grid_positions_from_linear_indices(
    linear_indices: np.ndarray,
    *,
    cells: int,
) -> np.ndarray:
    n = int(cells) + 1
    idx = np.asarray(linear_indices, dtype=np.int64).reshape(-1)
    total = n * n * n
    if idx.size == 0 or np.any(idx < 0) or np.any(idx >= total):
        raise ValueError("invalid grid linear indices")
    ii = idx // (n * n)
    rem = idx - ii * (n * n)
    jj = rem // n
    kk = rem - jj * n
    return (
        -1.0
        + 2.0
        * np.stack([ii, jj, kk], axis=1).astype(np.float32)
        / float(cells)
    )


def query_coarse_field_v5(
    query_fn: Callable[[np.ndarray], Any],
    *,
    policy: SparseRegularTetraPolicyV5,
    work_dir: str | Path | None = None,
) -> np.ndarray:
    policy.validate()
    n = int(policy.base_cells) + 1
    shape = (n, n, n)
    if work_dir is None:
        field = np.empty(shape, dtype=np.float32)
    else:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        field = np.memmap(
            root / "v5_coarse_field.f32",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
    flat = field.reshape(-1)
    total = flat.size
    for start in range(0, total, int(policy.query_chunk)):
        end = min(total, start + int(policy.query_chunk))
        idx = np.arange(start, end, dtype=np.int64)
        q = _grid_positions_from_linear_indices(idx, cells=int(policy.base_cells))
        flat[start:end] = _as_query_result(query_fn(q), end - start)
    if isinstance(field, np.memmap):
        field.flush()
    if not float(np.min(field)) < 0.0 < float(np.max(field)):
        raise ValueError("coarse field does not bracket the zero level")
    return field


def select_refined_cells_v5(
    coarse_field: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
) -> np.ndarray:
    policy.validate()
    base = int(policy.base_cells)
    n = base + 1
    field = np.asarray(coarse_field)
    if field.shape != (n, n, n):
        raise ValueError(f"coarse_field must be {(n, n, n)}")
    if not np.isfinite(field).all():
        raise ValueError("coarse_field contains non-finite values")

    parts: list[np.ndarray] = []
    slab = int(policy.refine_slab_cells)
    for x0 in range(0, base, slab):
        x1 = min(base, x0 + slab)
        corners = [
            field[x0:x1, 0:base, 0:base],
            field[x0:x1, 0:base, 1:n],
            field[x0:x1, 1:n, 0:base],
            field[x0:x1, 1:n, 1:n],
            field[x0 + 1 : x1 + 1, 0:base, 0:base],
            field[x0 + 1 : x1 + 1, 0:base, 1:n],
            field[x0 + 1 : x1 + 1, 1:n, 0:base],
            field[x0 + 1 : x1 + 1, 1:n, 1:n],
        ]
        minimum = corners[0].copy()
        maximum = corners[0].copy()
        minimum_abs = np.abs(corners[0]).copy()
        for corner in corners[1:]:
            np.minimum(minimum, corner, out=minimum)
            np.maximum(maximum, corner, out=maximum)
            np.minimum(minimum_abs, np.abs(corner), out=minimum_abs)
        mask = ((minimum <= 0.0) & (maximum >= 0.0)) | (
            minimum_abs <= float(policy.refine_band)
        )
        lx, jj, kk = np.nonzero(mask)
        if len(lx):
            parts.append(
                np.stack([lx + x0, jj, kk], axis=1).astype(np.int32)
            )
    if not parts:
        raise ValueError("refinement selected no coarse cells")
    refined = np.concatenate(parts, axis=0)
    keys = (
        (refined[:, 0].astype(np.int64) * base + refined[:, 1])
        * base
        + refined[:, 2]
    )
    keys = np.unique(keys)
    ii = keys // (base * base)
    rem = keys - ii * (base * base)
    jj = rem // base
    kk = rem - jj * base
    return np.stack([ii, jj, kk], axis=1).astype(np.int32)


def fine_vertex_gids_v5(
    refined_cells: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
) -> np.ndarray:
    policy.validate()
    refined = np.asarray(refined_cells, dtype=np.int64)
    if refined.ndim != 2 or refined.shape[1] != 3 or len(refined) == 0:
        raise ValueError("refined_cells must be non-empty [N,3]")
    n_fine = int(policy.fine_cells) + 1
    parts: list[np.ndarray] = []
    chunk = int(policy.fine_gid_cell_chunk)
    for start in range(0, len(refined), chunk):
        cc = refined[start : start + chunk]
        fine = (2 * cc[:, None, :] + LOCAL27_V5[None, :, :]).reshape(-1, 3)
        gid = ((fine[:, 0] * n_fine + fine[:, 1]) * n_fine + fine[:, 2])
        parts.append(np.unique(gid))
    gids = np.unique(np.concatenate(parts)).astype(np.int64)
    if len(gids) == 0:
        raise ValueError("fine sparse carrier has no vertices")
    return gids


def fine_positions_from_gids_v5(
    gids: np.ndarray,
    *,
    fine_cells: int,
) -> np.ndarray:
    n = int(fine_cells) + 1
    gid = np.asarray(gids, dtype=np.int64).reshape(-1)
    if gid.size == 0:
        raise ValueError("gids must be non-empty")
    ii = gid // (n * n)
    rem = gid - ii * (n * n)
    jj = rem // n
    kk = rem - jj * n
    return (
        -1.0
        + 2.0
        * np.stack([ii, jj, kk], axis=1).astype(np.float32)
        / float(fine_cells)
    )


def query_fine_scalar_v5(
    query_fn: Callable[[np.ndarray], Any],
    fine_gids: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
    work_dir: str | Path | None = None,
) -> np.ndarray:
    policy.validate()
    gids = np.asarray(fine_gids, dtype=np.int64).reshape(-1)
    if work_dir is None:
        scalar = np.empty(len(gids), dtype=np.float32)
    else:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        scalar = np.memmap(
            root / "v5_fine_scalar.f32",
            mode="w+",
            dtype=np.float32,
            shape=(len(gids),),
        )
    chunk = int(policy.query_chunk)
    for start in range(0, len(gids), chunk):
        end = min(len(gids), start + chunk)
        q = fine_positions_from_gids_v5(
            gids[start:end], fine_cells=int(policy.fine_cells)
        )
        scalar[start:end] = _as_query_result(query_fn(q), end - start)
    if isinstance(scalar, np.memmap):
        scalar.flush()
    if not float(np.min(scalar)) <= 0.0 <= float(np.max(scalar)):
        raise ValueError("fine sparse field does not bracket the zero level")
    return scalar


def boundary_parent_cells_v5(
    refined_cells: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
) -> np.ndarray:
    refined = np.asarray(refined_cells, dtype=np.int64)
    base = int(policy.base_cells)
    keys = ((refined[:, 0] * base + refined[:, 1]) * base + refined[:, 2])
    if np.any(np.diff(keys) < 0):
        raise ValueError("refined_cells must be in deterministic sorted order")
    boundary = np.zeros(len(refined), dtype=bool)
    neighbors = np.asarray(
        [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )
    for offset in neighbors:
        nn = refined + offset
        in_domain = ((nn >= 0) & (nn < base)).all(axis=1)
        nkeys = ((nn[:, 0] * base + nn[:, 1]) * base + nn[:, 2])
        pos = np.searchsorted(keys, nkeys)
        present = np.zeros(len(nn), dtype=bool)
        ok = in_domain & (pos < len(keys))
        ids = np.flatnonzero(ok)
        if len(ids):
            present[ids] = keys[pos[ids]] == nkeys[ids]
        boundary |= ~present
    return boundary


def _carrier_chunk(
    refined_cells: np.ndarray,
    fine_gids: np.ndarray,
    fine_scalar: np.ndarray,
    *,
    start: int,
    end: int,
    policy: SparseRegularTetraPolicyV5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cc = np.asarray(refined_cells[start:end], dtype=np.int64)
    n_fine = int(policy.fine_cells) + 1
    origins = 2 * cc[:, None, :] + SUBCUBE_OFFSETS_V5[None, :, :]
    fine_corners = origins[:, :, None, :] + CUBE8_V5[None, None, :, :]
    global_gid = (
        (fine_corners[..., 0] * n_fine + fine_corners[..., 1]) * n_fine
        + fine_corners[..., 2]
    )
    flat = global_gid.reshape(-1)
    local = np.searchsorted(fine_gids, flat)
    if (
        np.any(local >= len(fine_gids))
        or np.any(np.asarray(fine_gids[local]) != flat)
    ):
        raise RuntimeError("SPARSE_FINE_VERTEX_LOOKUP_MISS")
    cube_local = local.reshape(len(cc) * 8, 8)
    tet_local = cube_local[:, TETRAHEDRA_V5].reshape(-1, 4)
    scalar = np.asarray(fine_scalar[tet_local])
    codes = (
        (scalar > 0.0)
        * np.asarray([1, 2, 4, 8], dtype=np.int8)[None, :]
    ).sum(axis=1).astype(np.int8)
    return tet_local, scalar, codes


def count_sparse_mt_v5(
    refined_cells: np.ndarray,
    fine_gids: np.ndarray,
    fine_scalar: np.ndarray,
    boundary_parent: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
) -> dict[str, Any]:
    crossing = 0
    faces = 0
    boundary_crossing = 0
    histogram = np.zeros(16, dtype=np.int64)
    chunks: list[dict[str, int]] = []
    for start in range(0, len(refined_cells), int(policy.cell_chunk)):
        end = min(len(refined_cells), start + int(policy.cell_chunk))
        _tet, _sv, codes = _carrier_chunk(
            refined_cells,
            fine_gids,
            fine_scalar,
            start=start,
            end=end,
            policy=policy,
        )
        histogram += np.bincount(codes, minlength=16)
        valid = (codes > 0) & (codes < 15)
        chunk_crossing = int(valid.sum())
        chunk_faces = int(NUM_TRIANGLES_V5[codes].sum())
        crossing += chunk_crossing
        faces += chunk_faces
        parent_boundary = np.repeat(
            np.asarray(boundary_parent[start:end], dtype=bool), 8 * 6
        )
        boundary_crossing += int((valid & parent_boundary).sum())
        chunks.append(
            {
                "start": int(start),
                "end": int(end),
                "crossing_tets": chunk_crossing,
                "faces": chunk_faces,
            }
        )
    return {
        "candidate_fine_tets": int(len(refined_cells) * 8 * 6),
        "crossing_tets": int(crossing),
        "exact_faces": int(faces),
        "boundary_crossing_tets": int(boundary_crossing),
        "case_histogram": histogram.tolist(),
        "chunks": chunks,
    }


def _emit_edge_keys_for_chunk(
    refined_cells: np.ndarray,
    fine_gids: np.ndarray,
    fine_scalar: np.ndarray,
    *,
    start: int,
    end: int,
    policy: SparseRegularTetraPolicyV5,
) -> np.ndarray:
    tet_local, _sv, codes = _carrier_chunk(
        refined_cells,
        fine_gids,
        fine_scalar,
        start=start,
        end=end,
        policy=policy,
    )
    valid = (codes > 0) & (codes < 15)
    if not np.any(valid):
        return np.empty(0, dtype=np.uint64)
    tv = tet_local[valid]
    cv = codes[valid]
    edge_pairs = tv[:, TETRA_EDGES_V5]
    pieces: list[np.ndarray] = []
    for ntri in (1, 2):
        selection = np.flatnonzero(NUM_TRIANGLES_V5[cv] == ntri)
        if len(selection) == 0:
            continue
        triangle_edges = TRI_TABLE_V5[cv[selection], : 3 * ntri].reshape(
            -1, ntri, 3
        )
        selected_pairs = edge_pairs[selection]
        for triangle_index in range(ntri):
            edge_ids = triangle_edges[:, triangle_index, :]
            pair = np.take_along_axis(
                selected_pairs,
                edge_ids[:, :, None].repeat(2, axis=2),
                axis=1,
            )
            gids = np.asarray(fine_gids[pair], dtype=np.int64)
            lo = np.minimum(gids[:, :, 0], gids[:, :, 1]).astype(np.uint64)
            hi = np.maximum(gids[:, :, 0], gids[:, :, 1]).astype(np.uint64)
            if np.any(lo >= 2**32) or np.any(hi >= 2**32):
                raise ValueError("fine-grid gid exceeds uint32 edge-key packing")
            pieces.append(((lo << np.uint64(32)) | hi).reshape(-1))
    if not pieces:
        return np.empty(0, dtype=np.uint64)
    return np.concatenate(pieces)


def _one_sided_orientation_fallback_v5(
    query_fn: Callable[[np.ndarray], Any],
    points: np.ndarray,
    *,
    epsilon: float,
    policy: SparseRegularTetraPolicyV5,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Resolve non-smooth MAX-field kinks without pretending a classical gradient exists.

    The canonical V5 source field is a MAX over per-view signed fields.  At an active-view
    ridge a symmetric central stencil can be exactly zero even though the zero set is
    perfectly well-defined.  Stage14 consumes these vectors only as orientation hints for
    its robust local-PCA normal operator, so for those non-smooth points we select a
    deterministic outward one-sided secant.  A genuinely flat/unoriented neighborhood
    still fails closed.
    """

    q_all = np.asarray(points, dtype=np.float32)
    if q_all.ndim != 2 or q_all.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    if len(q_all) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.zeros(0, dtype=bool),
            {},
        )

    eye = np.eye(3, dtype=np.float32)
    out = np.zeros_like(q_all)
    resolved = np.zeros(len(q_all), dtype=bool)
    scale_counts: dict[str, int] = {}
    f0_all = _query_numpy(
        query_fn,
        q_all,
        chunk_size=int(policy.query_chunk),
    ).astype(np.float64)

    # Start at the exact frozen normal stencil, then widen only unresolved points.
    for scale in (1.0, 2.0, 4.0, 8.0):
        pending = np.flatnonzero(~resolved)
        if len(pending) == 0:
            break
        q = q_all[pending]
        f0 = f0_all[pending]
        step = float(epsilon) * float(scale)
        components = np.zeros((len(q), 3), dtype=np.float64)
        best_axis_slope = np.zeros((len(q), 3), dtype=np.float64)

        for axis in range(3):
            qp = np.clip(q + step * eye[axis], -1.0, 1.0)
            qm = np.clip(q - step * eye[axis], -1.0, 1.0)
            fp = _query_numpy(
                query_fn, qp, chunk_size=int(policy.query_chunk)
            ).astype(np.float64)
            fm = _query_numpy(
                query_fn, qm, chunk_size=int(policy.query_chunk)
            ).astype(np.float64)
            dp = qp[:, axis].astype(np.float64) - q[:, axis].astype(np.float64)
            dm = q[:, axis].astype(np.float64) - qm[:, axis].astype(np.float64)

            plus = np.full(len(q), -np.inf, dtype=np.float64)
            minus = np.full(len(q), -np.inf, dtype=np.float64)
            valid_plus = dp > 1e-12
            valid_minus = dm > 1e-12
            plus[valid_plus] = (fp[valid_plus] - f0[valid_plus]) / dp[valid_plus]
            minus[valid_minus] = (fm[valid_minus] - f0[valid_minus]) / dm[valid_minus]

            choose_plus = plus >= minus
            chosen = np.maximum(plus, minus)
            positive = np.maximum(chosen, 0.0)
            components[:, axis] = np.where(choose_plus, positive, -positive)
            best_axis_slope[:, axis] = positive

        norm = np.linalg.norm(components, axis=1)
        good = np.isfinite(norm) & (norm > 1e-12) & (
            np.max(best_axis_slope, axis=1) > 1e-12
        )
        if np.any(good):
            local = pending[good]
            out[local] = (
                components[good] / norm[good, None]
            ).astype(np.float32)
            resolved[local] = True
            scale_counts[f"{scale:g}x"] = int(np.count_nonzero(good))

    return out.astype(np.float32), resolved, scale_counts


def implicit_normals_with_diagnostics_from_query_v5(
    query_fn: Callable[[np.ndarray], Any],
    vertices_normalized: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices_normalized, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices_normalized must be non-empty [N,3]")
    epsilon = float(policy.normal_epsilon)
    eye = np.eye(3, dtype=np.float32)
    out = np.empty_like(vertices)
    central_count = 0
    fallback_count = 0
    fallback_scale_counts: dict[str, int] = {}

    for start in range(0, len(vertices), int(policy.normal_chunk)):
        end = min(len(vertices), start + int(policy.normal_chunk))
        q = vertices[start:end]
        gradients = []
        for axis in range(3):
            qp = np.clip(q + epsilon * eye[axis], -1.0, 1.0)
            qm = np.clip(q - epsilon * eye[axis], -1.0, 1.0)
            fp = _query_numpy(query_fn, qp, chunk_size=int(policy.query_chunk))
            fm = _query_numpy(query_fn, qm, chunk_size=int(policy.query_chunk))
            denom = np.maximum(qp[:, axis] - qm[:, axis], 1e-12)
            gradients.append(
                (fp.astype(np.float64) - fm.astype(np.float64))
                / denom.astype(np.float64)
            )
        gradient = np.stack(gradients, axis=1)
        norm = np.linalg.norm(gradient, axis=1)
        if not np.isfinite(norm).all():
            raise ValueError("implicit field gradient is non-finite on extracted surface")

        good = norm > 1e-12
        if np.any(good):
            out[start:end][good] = (
                gradient[good] / norm[good, None]
            ).astype(np.float32)
            central_count += int(np.count_nonzero(good))

        bad = ~good
        if np.any(bad):
            fallback, resolved, scale_counts = _one_sided_orientation_fallback_v5(
                query_fn,
                q[bad],
                epsilon=epsilon,
                policy=policy,
            )
            if not np.all(resolved):
                unresolved = q[bad][~resolved]
                sample = unresolved[:8].astype(np.float64).tolist()
                raise ValueError(
                    "implicit field orientation is unresolved on extracted surface:"
                    f"count={len(unresolved)} sample={sample}"
                )
            out[start:end][bad] = fallback
            nfb = int(np.count_nonzero(bad))
            fallback_count += nfb
            for key, value in scale_counts.items():
                fallback_scale_counts[key] = (
                    fallback_scale_counts.get(key, 0) + int(value)
                )

    final_norm = np.linalg.norm(out.astype(np.float64), axis=1)
    if (
        not np.isfinite(out).all()
        or not np.isfinite(final_norm).all()
        or np.any(final_norm <= 1e-12)
    ):
        raise ValueError("implicit orientation hint contains invalid vector")

    diagnostics = {
        "method": (
            "CENTRAL_DIFFERENCE_GRADIENT_WITH_DETERMINISTIC_"
            "ONE_SIDED_OUTWARD_SECANT_FALLBACK"
        ),
        "normal_epsilon": float(epsilon),
        "vertex_count": int(len(vertices)),
        "central_gradient_vertex_count": int(central_count),
        "one_sided_fallback_vertex_count": int(fallback_count),
        "one_sided_fallback_fraction": float(fallback_count / len(vertices)),
        "fallback_scale_counts": dict(sorted(fallback_scale_counts.items())),
        "unresolved_vertex_count": 0,
        "source_field_only": True,
        "teacher_truth_used": False,
    }
    return out.astype(np.float32), diagnostics


def implicit_normals_from_query_v5(
    query_fn: Callable[[np.ndarray], Any],
    vertices_normalized: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
) -> np.ndarray:
    normals, _diagnostics = implicit_normals_with_diagnostics_from_query_v5(
        query_fn,
        vertices_normalized,
        policy=policy,
    )
    return normals


def build_indexed_mt_from_sparse_carrier_v5(
    refined_cells: np.ndarray,
    fine_gids: np.ndarray,
    fine_scalar: np.ndarray,
    *,
    query_fn: Callable[[np.ndarray], Any],
    policy: SparseRegularTetraPolicyV5,
    work_dir: str | Path | None = None,
) -> IndexedSparseTetraMeshV5:
    policy.validate()
    refined = np.asarray(refined_cells, dtype=np.int32)
    gids = np.asarray(fine_gids, dtype=np.int64)
    scalar = np.asarray(fine_scalar, dtype=np.float32)
    if refined.ndim != 2 or refined.shape[1] != 3 or len(refined) == 0:
        raise ValueError("refined_cells must be non-empty [N,3]")
    if gids.ndim != 1 or scalar.shape != gids.shape or len(gids) == 0:
        raise ValueError("fine_gids/fine_scalar must be matching non-empty vectors")
    if np.any(np.diff(gids) <= 0):
        raise ValueError("fine_gids must be strictly increasing")
    if not np.isfinite(scalar).all():
        raise ValueError("fine_scalar contains non-finite values")

    boundary = boundary_parent_cells_v5(refined, policy=policy)
    count = count_sparse_mt_v5(
        refined, gids, scalar, boundary, policy=policy
    )
    if count["exact_faces"] <= 0:
        raise ValueError("sparse MT emitted no faces")
    if (
        bool(policy.fail_on_boundary_crossing)
        and int(count["boundary_crossing_tets"]) != 0
    ):
        raise SparseSurfaceTruncationError(
            "SPARSE_SHELL_TRUNCATES_ZERO_SURFACE:"
            f"{count['boundary_crossing_tets']} boundary crossing tetrahedra"
        )

    raw_edge_count = 3 * int(count["exact_faces"])
    if work_dir is None:
        raw_edges = np.empty(raw_edge_count, dtype=np.uint64)
        raw_edge_path = None
    else:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        raw_edge_path = root / "v5_mt_raw_edges.u64"
        raw_edges = np.memmap(
            raw_edge_path,
            mode="w+",
            dtype=np.uint64,
            shape=(raw_edge_count,),
        )

    cursor = 0
    for row in count["chunks"]:
        keys = _emit_edge_keys_for_chunk(
            refined,
            gids,
            scalar,
            start=int(row["start"]),
            end=int(row["end"]),
            policy=policy,
        )
        expected = 3 * int(row["faces"])
        if len(keys) != expected:
            raise RuntimeError(
                f"EDGE_EMIT_FACE_COUNT_MISMATCH:{len(keys)}!={expected}"
            )
        raw_edges[cursor : cursor + len(keys)] = keys
        cursor += len(keys)
    if cursor != raw_edge_count:
        raise RuntimeError("RAW_EDGE_FINAL_COUNT_MISMATCH")
    if isinstance(raw_edges, np.memmap):
        raw_edges.flush()

    unique_keys, inverse = np.unique(np.asarray(raw_edges), return_inverse=True)
    vertex_count = int(len(unique_keys))
    if vertex_count < 4:
        raise ValueError("indexed MT produced fewer than four vertices")
    if vertex_count >= 2**31:
        raise ValueError("indexed MT vertex count exceeds int32")
    faces = inverse.reshape(-1, 3).astype(np.int32, copy=False)
    del inverse
    if len(faces) != int(count["exact_faces"]):
        raise RuntimeError("INDEXED_MT_FACE_COUNT_MISMATCH")
    if np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    ):
        raise ValueError("indexed MT emitted a degenerate index triangle")

    lo = (unique_keys >> np.uint64(32)).astype(np.int64)
    hi = (unique_keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
    left = np.searchsorted(gids, lo)
    right = np.searchsorted(gids, hi)
    if (
        np.any(left >= len(gids))
        or np.any(right >= len(gids))
        or np.any(gids[left] != lo)
        or np.any(gids[right] != hi)
    ):
        raise RuntimeError("UNIQUE_EDGE_ENDPOINT_LOOKUP_MISS")

    p0 = fine_positions_from_gids_v5(lo, fine_cells=int(policy.fine_cells))
    p1 = fine_positions_from_gids_v5(hi, fine_cells=int(policy.fine_cells))
    s0 = scalar[left].astype(np.float64)
    s1 = scalar[right].astype(np.float64)
    denominator = s0 - s1
    if np.any(np.abs(denominator) < 1e-15):
        raise ValueError("ZERO_DENOMINATOR_UNIQUE_MT_EDGE")
    alpha = (s0 / denominator).astype(np.float32)
    vertices = (p0 + alpha[:, None] * (p1 - p0)).astype(np.float32)
    if not np.isfinite(vertices).all() or np.max(np.abs(vertices)) > 1.00001:
        raise ValueError("indexed MT produced invalid normalized vertices")

    normals, normal_diagnostics = implicit_normals_with_diagnostics_from_query_v5(
        query_fn,
        vertices,
        policy=policy,
    )
    diagnostics = {
        "base_cells": int(policy.base_cells),
        "fine_cells": int(policy.fine_cells),
        "refined_cell_count": int(len(refined)),
        "fine_vertex_count": int(len(gids)),
        "boundary_parent_cell_count": int(boundary.sum()),
        "candidate_fine_tets": int(count["candidate_fine_tets"]),
        "crossing_tets": int(count["crossing_tets"]),
        "boundary_crossing_tets": int(count["boundary_crossing_tets"]),
        "raw_edge_reference_count": int(raw_edge_count),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "global_shared_edge_dedup": True,
        "deformation": False,
        "whole_cell_halo": False,
        "raw_edge_path": str(raw_edge_path) if raw_edge_path is not None else None,
        "implicit_normal_diagnostics": normal_diagnostics,
    }
    return IndexedSparseTetraMeshV5(
        vertices_normalized=vertices,
        faces=faces,
        implicit_normals=normals,
        diagnostics=diagnostics,
    )


def decode_indexed_sparse_tetra_mt_v5(
    query_fn: Callable[[np.ndarray], Any],
    *,
    policy: SparseRegularTetraPolicyV5 = SparseRegularTetraPolicyV5(),
    work_dir: str | Path | None = None,
) -> IndexedSparseTetraMeshV5:
    """Full selected decoder: T512 -> sparse T1024 -> undeformed indexed MT."""

    policy.validate()
    coarse = query_coarse_field_v5(
        query_fn,
        policy=policy,
        work_dir=work_dir,
    )
    refined = select_refined_cells_v5(coarse, policy=policy)
    gids = fine_vertex_gids_v5(refined, policy=policy)
    scalar = query_fine_scalar_v5(
        query_fn,
        gids,
        policy=policy,
        work_dir=work_dir,
    )
    mesh = build_indexed_mt_from_sparse_carrier_v5(
        refined,
        gids,
        scalar,
        query_fn=query_fn,
        policy=policy,
        work_dir=work_dir,
    )
    merged = dict(mesh.diagnostics)
    merged.update(
        {
            "coarse_field_min": float(np.min(coarse)),
            "coarse_field_max": float(np.max(coarse)),
            "refine_band": float(policy.refine_band),
            "normal_epsilon": float(policy.normal_epsilon),
            "surface_evaluation": "INDEXED_MARCHING_TETRAHEDRA",
            "representation": "SPARSE_REGULAR_TETRA_UNDEFORMED",
        }
    )
    return IndexedSparseTetraMeshV5(
        vertices_normalized=mesh.vertices_normalized,
        faces=mesh.faces,
        implicit_normals=mesh.implicit_normals,
        diagnostics=merged,
    )


def write_stage12_npz_v5(
    mesh: IndexedSparseTetraMeshV5,
    path: str | Path,
) -> Path:
    """Write the exact array names consumed by iris_geometry_v2 Stage12."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        vertices_normalized=np.asarray(mesh.vertices_normalized, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        implicit_normals=np.asarray(mesh.implicit_normals, dtype=np.float32),
    )
    return out
