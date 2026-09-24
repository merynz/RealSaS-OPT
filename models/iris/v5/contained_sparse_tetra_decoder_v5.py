from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .indexed_sparse_tetra_decoder_v5 import (
    IndexedSparseTetraMeshV5,
    SparseRegularTetraPolicyV5,
    SparseSurfaceTruncationError,
    build_indexed_mt_from_sparse_carrier_v5,
    fine_vertex_gids_v5,
    query_coarse_field_v5,
    query_fine_scalar_v5,
    select_refined_cells_v5,
)
from .shell_escape_diagnostics_v5 import (
    FACE_NAMES_V5,
    ShellEscapeDiagnosticPolicyV5,
    exact_missing_face_escape_report_v5,
    face_zero_crossing_from_scalar_v5,
    parent_face_fine_ijk_v5,
)


CONTAINED_DECODER_ID_V5 = (
    "RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT."
    "ExactShellContainment.v5_1"
)


def _fine_gid_from_ijk_v5(ijk: np.ndarray, *, fine_cells: int) -> np.ndarray:
    p = np.asarray(ijk, dtype=np.int64)
    n = int(fine_cells) + 1
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("ijk must be [N,3]")
    if np.any(p < 0) or np.any(p > int(fine_cells)):
        raise ValueError("fine-grid ijk outside domain")
    return ((p[:, 0] * n + p[:, 1]) * n + p[:, 2]).astype(np.int64)


def domain_boundary_face_escape_report_v5(
    refined_cells: np.ndarray,
    fine_gids: np.ndarray,
    fine_scalar: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
    diagnostic_policy: ShellEscapeDiagnosticPolicyV5 = ShellEscapeDiagnosticPolicyV5(),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure actual zero-set contact with the outer [-1,1]^3 carrier boundary.

    The in-domain omitted-neighbor diagnostic intentionally excludes domain faces.
    A containment gate must cover both classes of carrier boundary:
    (1) a refined cell adjacent to an omitted in-domain cell and
    (2) a refined cell adjacent to the global normalized-domain boundary.

    A zero on an outer face is treated conservatively as contact/escape. Edge- or
    corner-only zero contact is therefore not silently ignored.
    """

    policy.validate()
    diagnostic_policy.validate()
    refined = np.asarray(refined_cells, dtype=np.int32)
    gids = np.asarray(fine_gids, dtype=np.int64).reshape(-1)
    scalar = np.asarray(fine_scalar, dtype=np.float32).reshape(-1)
    if refined.ndim != 2 or refined.shape[1] != 3 or len(refined) == 0:
        raise ValueError("refined_cells must be non-empty [N,3]")
    if scalar.shape != gids.shape or len(gids) == 0 or np.any(np.diff(gids) <= 0):
        raise ValueError("fine_gids/fine_scalar must be matching sorted vectors")

    base = int(policy.base_cells)
    face_specs = (
        (0, 0, 0),
        (0, base - 1, 1),
        (1, 0, 2),
        (1, base - 1, 3),
        (2, 0, 4),
        (2, base - 1, 5),
    )
    parent_ids: list[int] = []
    face_ids: list[int] = []
    face_min: list[float] = []
    face_max: list[float] = []

    for axis, coord, face_index in face_specs:
        for pi in np.flatnonzero(refined[:, axis] == coord):
            ijk = parent_face_fine_ijk_v5(refined[int(pi)], face_index)
            fgids = _fine_gid_from_ijk_v5(
                ijk,
                fine_cells=int(policy.fine_cells),
            )
            pos = np.searchsorted(gids, fgids)
            if np.any(pos >= len(gids)) or np.any(gids[pos] != fgids):
                raise RuntimeError("DOMAIN_FACE_FINE_VERTEX_LOOKUP_MISS")
            vals = scalar[pos]
            grid = vals.reshape(3, 3)
            if face_zero_crossing_from_scalar_v5(
                grid,
                epsilon=float(diagnostic_policy.zero_epsilon),
            ):
                parent_ids.append(int(pi))
                face_ids.append(int(face_index))
                face_min.append(float(np.min(vals)))
                face_max.append(float(np.max(vals)))

    parent_ids_np = np.asarray(parent_ids, dtype=np.int64)
    face_ids_np = np.asarray(face_ids, dtype=np.int8)
    unique_parents = (
        np.unique(parent_ids_np)
        if len(parent_ids_np)
        else np.empty((0,), dtype=np.int64)
    )
    report = {
        "actual_domain_boundary_zero_crossing_count": int(len(face_ids_np)),
        "actual_domain_boundary_parent_cell_count": int(len(unique_parents)),
        "face_name_counts": {
            FACE_NAMES_V5[i]: int(np.count_nonzero(face_ids_np == i))
            for i in range(6)
        },
    }
    arrays = {
        "domain_escape_parent_indices": parent_ids_np,
        "domain_escape_parent_cells": (
            refined[parent_ids_np].astype(np.int32)
            if len(parent_ids_np)
            else np.empty((0, 3), dtype=np.int32)
        ),
        "domain_escape_face_indices": face_ids_np,
        "domain_escape_face_min": np.asarray(face_min, dtype=np.float32),
        "domain_escape_face_max": np.asarray(face_max, dtype=np.float32),
    }
    return report, arrays


def exact_sparse_shell_containment_report_v5(
    refined_cells: np.ndarray,
    fine_gids: np.ndarray,
    fine_scalar: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
    diagnostic_policy: ShellEscapeDiagnosticPolicyV5 = ShellEscapeDiagnosticPolicyV5(),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Exact carrier-boundary containment gate for the piecewise-linear MT field.

    For the tetrahedral piecewise-linear zero set to leave the union of refined
    carrier cells, it must intersect the topological boundary of that union.
    That boundary is exhausted by omitted in-domain neighbor faces plus global
    normalized-domain faces. This gate tests the fine-grid face interpolation on
    both classes and fails on any zero contact.

    The historical boundary_parent / boundary_crossing_tets count is retained
    only as conservative telemetry; it is not a truncation proof.
    """

    omitted_report, omitted_arrays = exact_missing_face_escape_report_v5(
        refined_cells,
        fine_gids,
        fine_scalar,
        policy=policy,
        diagnostic_policy=diagnostic_policy,
    )
    domain_report, domain_arrays = domain_boundary_face_escape_report_v5(
        refined_cells,
        fine_gids,
        fine_scalar,
        policy=policy,
        diagnostic_policy=diagnostic_policy,
    )
    omitted = int(omitted_report["actual_missing_face_zero_crossing_count"])
    domain = int(domain_report["actual_domain_boundary_zero_crossing_count"])
    report = {
        "containment_gate": "EXACT_REFINED_CARRIER_BOUNDARY_FACE_ZERO_CONTACT_V1",
        "actual_missing_face_zero_crossing_count": omitted,
        "actual_escape_parent_cell_count": int(
            omitted_report["actual_escape_parent_cell_count"]
        ),
        "actual_domain_boundary_zero_crossing_count": domain,
        "actual_domain_boundary_parent_cell_count": int(
            domain_report["actual_domain_boundary_parent_cell_count"]
        ),
        "actual_total_boundary_zero_crossing_count": int(omitted + domain),
        "passed": bool(omitted == 0 and domain == 0),
        "conservative_boundary_parent_cell_count": int(
            omitted_report["boundary_parent_cell_count_conservative"]
        ),
        "conservative_boundary_parent_false_positive_count": int(
            omitted_report["conservative_boundary_parent_false_positive_count"]
        ),
        "omitted_face_name_counts": omitted_report["face_name_counts"],
        "domain_face_name_counts": domain_report["face_name_counts"],
        "edge_or_corner_zero_contact_policy": "FAIL_CLOSED_AS_BOUNDARY_CONTACT",
    }
    arrays = dict(omitted_arrays)
    arrays.update(domain_arrays)
    return report, arrays


def build_indexed_mt_from_sparse_carrier_contained_v5(
    refined_cells: np.ndarray,
    fine_gids: np.ndarray,
    fine_scalar: np.ndarray,
    *,
    query_fn: Callable[[np.ndarray], Any],
    policy: SparseRegularTetraPolicyV5,
    work_dir: str | Path | None = None,
) -> IndexedSparseTetraMeshV5:
    """Build indexed MT only after exact shell containment is established."""

    containment, _arrays = exact_sparse_shell_containment_report_v5(
        refined_cells,
        fine_gids,
        fine_scalar,
        policy=policy,
    )
    if not bool(containment["passed"]):
        raise SparseSurfaceTruncationError(
            "SPARSE_SHELL_ACTUAL_ZERO_SURFACE_ESCAPE:"
            f"omitted_faces={containment['actual_missing_face_zero_crossing_count']}:"
            f"domain_faces={containment['actual_domain_boundary_zero_crossing_count']}"
        )

    # The legacy boundary-crossing proxy is intentionally disabled only after the
    # exact containment proof above passes. All geometry/extraction parameters stay
    # byte-for-byte identical.
    extraction_policy = replace(policy, fail_on_boundary_crossing=False)
    mesh = build_indexed_mt_from_sparse_carrier_v5(
        refined_cells,
        fine_gids,
        fine_scalar,
        query_fn=query_fn,
        policy=extraction_policy,
        work_dir=work_dir,
    )
    diagnostics = dict(mesh.diagnostics)
    diagnostics.update(containment)
    diagnostics.update(
        {
            "legacy_boundary_crossing_tets_role": "TELEMETRY_ONLY",
            "legacy_boundary_crossing_tets_is_truncation_gate": False,
            "exact_shell_containment_required": True,
        }
    )
    return IndexedSparseTetraMeshV5(
        vertices_normalized=mesh.vertices_normalized,
        faces=mesh.faces,
        implicit_normals=mesh.implicit_normals,
        diagnostics=diagnostics,
        decoder_id=CONTAINED_DECODER_ID_V5,
    )


def decode_indexed_sparse_tetra_mt_contained_v5(
    query_fn: Callable[[np.ndarray], Any],
    *,
    policy: SparseRegularTetraPolicyV5 = SparseRegularTetraPolicyV5(),
    work_dir: str | Path | None = None,
) -> IndexedSparseTetraMeshV5:
    """T512 -> sparse T1024 -> exact containment gate -> indexed MT."""

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
    mesh = build_indexed_mt_from_sparse_carrier_contained_v5(
        refined,
        gids,
        scalar,
        query_fn=query_fn,
        policy=policy,
        work_dir=work_dir,
    )
    diagnostics = dict(mesh.diagnostics)
    diagnostics.update(
        {
            "coarse_field_min": float(np.min(coarse)),
            "coarse_field_max": float(np.max(coarse)),
            "refine_band": float(policy.refine_band),
            "normal_epsilon": float(policy.normal_epsilon),
            "surface_evaluation": "INDEXED_MARCHING_TETRAHEDRA",
            "representation": "SPARSE_REGULAR_TETRA_UNDEFORMED",
            "gate_correction_only": True,
        }
    )
    return IndexedSparseTetraMeshV5(
        vertices_normalized=mesh.vertices_normalized,
        faces=mesh.faces,
        implicit_normals=mesh.implicit_normals,
        diagnostics=diagnostics,
        decoder_id=CONTAINED_DECODER_ID_V5,
    )
