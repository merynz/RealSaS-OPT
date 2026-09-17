from __future__ import annotations

"""Generic source-backed appearance authority for 8-direction playback.

UNSEEN has one meaning only: the drawable locus has no qualified source evidence in
any required observation view. A target view that lacks direct evidence must use a
stable qualified donor from another view when one exists.

This module resolves authority only. It does not render, complete, inpaint, delete
geometry, or inspect motion output.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_runtime_v3 import AppearanceProvenance
from compiler.realsas_compiler_core.types import QualificationError

GLOBAL_SOURCE_APPEARANCE_SCHEMA = "RealSaS.GlobalSourceAppearanceAuthority.v2"
DEFAULT_VIEWS = tuple(f"V{i}" for i in range(8))


@dataclass(frozen=True)
class GlobalSourceAppearanceAuthorityV2:
    view_ids: tuple[str, ...]
    provenance_codes_by_view: Mapping[str, np.ndarray]
    donor_view_indices_by_view: Mapping[str, np.ndarray]
    globally_unseen_mask: np.ndarray
    authority_hash: str
    schema_version: str = GLOBAL_SOURCE_APPEARANCE_SCHEMA


_PROV_CODE = {
    AppearanceProvenance.DIRECT_SOURCE: 0,
    AppearanceProvenance.OTHER_VIEW_SOURCE: 1,
    AppearanceProvenance.UNDER_RIGID_SOURCE: 2,
    AppearanceProvenance.UNSEEN: 3,
    AppearanceProvenance.COMPLETION: 4,
}


def _normalize_forwards(
    camera_forwards_by_view: Mapping[str, Sequence[float]],
    view_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if set(camera_forwards_by_view) != set(view_ids):
        raise QualificationError("GLOBAL_APPEARANCE_CAMERA_VIEW_SET_MISMATCH")
    out: dict[str, np.ndarray] = {}
    for view_id in view_ids:
        row = np.asarray(camera_forwards_by_view[view_id], dtype=np.float64)
        if row.shape != (3,) or not np.isfinite(row).all():
            raise QualificationError("GLOBAL_APPEARANCE_CAMERA_FORWARD_INVALID")
        norm = float(np.linalg.norm(row))
        if norm <= 1.0e-12:
            raise QualificationError("GLOBAL_APPEARANCE_CAMERA_FORWARD_DEGENERATE")
        out[view_id] = row / norm
    return out


def _direct_matrix(
    direct_source_by_view: Mapping[str, Sequence[bool] | np.ndarray],
    view_ids: tuple[str, ...],
) -> np.ndarray:
    if set(direct_source_by_view) != set(view_ids):
        raise QualificationError("GLOBAL_APPEARANCE_DIRECT_VIEW_SET_MISMATCH")
    rows = [np.asarray(direct_source_by_view[v], dtype=np.bool_) for v in view_ids]
    if not rows or rows[0].ndim != 1 or len(rows[0]) == 0:
        raise QualificationError("GLOBAL_APPEARANCE_LOCUS_SET_EMPTY")
    locus_count = len(rows[0])
    if any(row.shape != (locus_count,) for row in rows):
        raise QualificationError("GLOBAL_APPEARANCE_DIRECT_CARDINALITY_MISMATCH")
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.bool_)


def resolve_global_source_appearance_authority_v2(
    direct_source_by_view: Mapping[str, Sequence[bool] | np.ndarray],
    camera_forwards_by_view: Mapping[str, Sequence[float]],
    *,
    view_ids: Sequence[str] = DEFAULT_VIEWS,
) -> GlobalSourceAppearanceAuthorityV2:
    """Resolve stable DIRECT/OTHER_VIEW/UNSEEN authority for arbitrary loci.

    A locus is UNSEEN iff no required view has direct source evidence for it.
    Donor choice is compile-time deterministic: maximum camera-forward dot product,
    then lowest donor view index. Motion never participates in donor selection.
    """

    views = tuple(map(str, view_ids))
    if not views or len(views) != len(set(views)):
        raise QualificationError("GLOBAL_APPEARANCE_REQUIRED_VIEW_SET_INVALID")
    direct = _direct_matrix(direct_source_by_view, views)
    forwards = _normalize_forwards(camera_forwards_by_view, views)
    globally_seen = np.any(direct, axis=0)
    globally_unseen = np.ascontiguousarray(~globally_seen, dtype=np.bool_)

    provenance: dict[str, np.ndarray] = {}
    donors: dict[str, np.ndarray] = {}
    donor_orders: dict[str, tuple[int, ...]] = {}

    for target_index, target_view in enumerate(views):
        target_forward = forwards[target_view]
        order = sorted(
            (
                (-float(np.dot(target_forward, forwards[donor_view])), donor_index)
                for donor_index, donor_view in enumerate(views)
                if donor_index != target_index
            ),
            key=lambda row: (row[0], row[1]),
        )
        donor_order = tuple(index for _score, index in order)
        donor_orders[target_view] = donor_order

        prov = np.full(
            direct.shape[1],
            _PROV_CODE[AppearanceProvenance.UNSEEN],
            dtype=np.uint8,
        )
        donor = np.full(direct.shape[1], -1, dtype=np.int16)
        own = direct[target_index]
        prov[own] = _PROV_CODE[AppearanceProvenance.DIRECT_SOURCE]
        donor[own] = target_index

        missing_target = (~own) & globally_seen
        for locus_index in np.flatnonzero(missing_target):
            selected = next(
                (idx for idx in donor_order if bool(direct[idx, locus_index])),
                None,
            )
            if selected is None:
                raise QualificationError("GLOBAL_APPEARANCE_DONOR_RESOLUTION_INCOMPLETE")
            prov[locus_index] = _PROV_CODE[AppearanceProvenance.OTHER_VIEW_SOURCE]
            donor[locus_index] = int(selected)

        if np.any((prov == _PROV_CODE[AppearanceProvenance.UNSEEN]) != globally_unseen):
            raise QualificationError("GLOBAL_APPEARANCE_UNSEEN_SEMANTIC_VIOLATION")
        provenance[target_view] = np.ascontiguousarray(prov)
        donors[target_view] = np.ascontiguousarray(donor)

    payload = {
        "schema": GLOBAL_SOURCE_APPEARANCE_SCHEMA,
        "view_ids": list(views),
        "locus_count": int(direct.shape[1]),
        "direct_source_counts": {
            view_id: int(np.count_nonzero(direct[i]))
            for i, view_id in enumerate(views)
        },
        "globally_unseen_count": int(np.count_nonzero(globally_unseen)),
        "donor_orders": {k: list(v) for k, v in donor_orders.items()},
        "provenance": {
            view_id: provenance[view_id].tolist()
            for view_id in views
        },
        "donors": {
            view_id: donors[view_id].tolist()
            for view_id in views
        },
        "completion_used": False,
        "motion_used_for_donor_selection": False,
    }
    return GlobalSourceAppearanceAuthorityV2(
        view_ids=views,
        provenance_codes_by_view=provenance,
        donor_view_indices_by_view=donors,
        globally_unseen_mask=globally_unseen,
        authority_hash=content_sha256(payload),
    )


__all__ = [
    "DEFAULT_VIEWS",
    "GLOBAL_SOURCE_APPEARANCE_SCHEMA",
    "GlobalSourceAppearanceAuthorityV2",
    "resolve_global_source_appearance_authority_v2",
]
