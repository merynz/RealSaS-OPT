"""Independent relational checks used by the scoped Stage45 gate.

Coincident source coordinates are candidates for investigation, not permission
to weld surfaces or a substitute for qualified material/contact intent.
"""
import numpy as np

from .types import QualificationError


def source_cut_pairs(rest_positions, domain_id, vertex_owner):
    rest = np.asarray(rest_positions, dtype=np.float64)
    domains = np.asarray(domain_id)
    owners = np.asarray(vertex_owner)
    if (rest.ndim != 2 or rest.shape[1] != 2 or not np.isfinite(rest).all()
            or domains.shape != (len(rest),) or owners.shape != (len(rest),)):
        raise QualificationError("PRESENTATION_SOURCE_CUT_INPUT_INVALID")
    # Exact equality only. Nearby separate silhouettes must not become seams.
    _, inverse = np.unique(rest, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    splits = np.flatnonzero(np.diff(inverse[order])) + 1
    pairs = []
    for group in np.split(order, splits):
        for owner in np.unique(owners[group]):
            local = group[owners[group] == owner]
            if len(local) > 1:
                pairs.extend((int(local[0]), int(i)) for i in local[1:]
                             if domains[i] != domains[local[0]])
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def contact_residuals(positions, pairs, *, tolerance_px=1e-7):
    positions = np.asarray(positions, dtype=np.float64)
    pairs = np.asarray(pairs)
    if (positions.ndim != 2 or positions.shape[1] != 2
            or not np.isfinite(positions).all() or pairs.ndim != 2 or pairs.shape[1] != 2
            or pairs.dtype.kind not in "iu" or np.any(pairs < 0)
            or np.any(pairs >= len(positions)) or not np.isfinite(tolerance_px) or tolerance_px < 0):
        raise QualificationError("PRESENTATION_CONTACT_PROBE_INVALID")
    distances = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    return {"contact_pair_count": len(pairs),
            "contact_failure_count": int(np.count_nonzero(distances > tolerance_px)),
            "maximum_contact_residual_px": float(np.max(distances, initial=0))}


def relational_verdict(proof):
    """Missing qualification is an explicit blocker, never an implicit PASS."""
    required = ("connected_palette_relations_passed", "qualified_contact_relations_passed",
                "qualified_dynamic_coverage_passed", "semantic_occlusion_passed",
                "setup_identity_passed", "frame0_relations_passed", "temporal_relations_passed")
    blockers = [key for key in required if proof.get(key) is not True]
    return {"relational_presentation_passed": not blockers,
            "relational_presentation_blockers": blockers}
