from __future__ import annotations

from .hashing import content_sha256
from .mesh_binding import validate_qualified_mesh, validate_qualified_mesh_skin
from .types import QualifiedMeshSkinIR, QualifiedMeshSkinRow, QualificationError


_MESH_SKIN_TRANSFER_REPAIR_L1 = 1e-9
SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD = "SURFACE_SUPPORT_CONVEX_TRANSFER_V1"


def bind_mwb2_mesh_skin(
    surface,
    skeleton,
    skin,
    mesh,
    *,
    max_transfer_repair_l1: float = _MESH_SKIN_TRANSFER_REPAIR_L1,
    max_total_transfer_correction_l1: float = _MESH_SKIN_TRANSFER_REPAIR_L1,
) -> QualifiedMeshSkinIR:
    """Bind exact mesh rows from qualified surface skin through admitted support coefficients.

    This is a deterministic Compiler derivation, not a learned direct mesh query and
    not a historical cross-substrate transfer.  Each mesh vertex already owns a
    qualified SurfaceSupportBinding against ``surface``; weights are the exact convex
    combination of the current QualifiedSkinIR rows referenced by that binding.
    """
    if max_transfer_repair_l1 < 0.0 or max_total_transfer_correction_l1 < 0.0:
        raise ValueError("mesh-skin transfer repair budgets must be nonnegative")
    validate_qualified_mesh(mesh, surface)
    if skin.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("MESH_WEIGHT_SKIN_LINEAGE_MISMATCH: surface")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("MESH_WEIGHT_SKIN_LINEAGE_MISMATCH: skeleton")

    source = {r.surface_id: r for r in skin.rows}
    rows = []
    total_correction = 0.0
    max_residual = 0.0
    for vertex in sorted(mesh.vertices, key=lambda v: v.canonical_mesh_vertex_id):
        accum: dict[str, float] = {}
        provenance = []
        for sid, coeff in vertex.support_binding.coefficients:
            if sid not in source:
                raise QualificationError(f"MESH_WEIGHT_UNSUPPORTED_SURFACE:{sid}")
            c = float(coeff)
            provenance.append((sid, c))
            for jid, w in source[sid].influences:
                accum[jid] = accum.get(jid, 0.0) + c * float(w)
        total = sum(accum.values())
        if total <= 0.0:
            raise QualificationError("MESH_WEIGHT_ZERO_MASS")
        residual = abs(total - 1.0)
        if residual > max_transfer_repair_l1 + 1e-15:
            raise QualificationError(f"MESH_WEIGHT_TRANSFER_REPAIR_BUDGET_EXCEEDED:{residual}")
        normalized = tuple(sorted((jid, w / total) for jid, w in accum.items() if w > 0.0))
        final = dict(normalized)
        correction = sum(
            abs(final.get(jid, 0.0) - accum.get(jid, 0.0))
            for jid in set(accum) | set(final)
        )
        if correction > max_transfer_repair_l1 + 1e-15:
            raise QualificationError(f"MESH_WEIGHT_TRANSFER_CORRECTION_BUDGET_EXCEEDED:{correction}")
        total_correction += correction
        if total_correction > max_total_transfer_correction_l1 + 1e-15:
            raise QualificationError(f"MESH_WEIGHT_TOTAL_TRANSFER_CORRECTION_BUDGET_EXCEEDED:{total_correction}")
        max_residual = max(max_residual, residual)
        rows.append(
            QualifiedMeshSkinRow(
                vertex.canonical_mesh_vertex_id,
                normalized,
                tuple(provenance),
                residual,
                correction,
            )
        )

    report = {
        "status": "PASS_MWB2_CONVEX_SKIN_TRANSFER",
        "row_count": len(rows),
        "transfer_method": SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD,
        "semantic_skin_synthesis": False,
        "source_skin_rows_consumed": True,
        "direct_model_query": False,
        "historical_weight_transfer_used": False,
        "max_simplex_residual_before": max_residual,
        "total_correction_l1": total_correction,
        "bounded_transfer_repair_l1_per_row": max_transfer_repair_l1,
        "bounded_transfer_correction_l1_total": max_total_transfer_correction_l1,
        "silent_normalization_forbidden": True,
    }
    value = QualifiedMeshSkinIR(
        tuple(rows),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        skin.skin_lineage_hash,
        mesh.mesh_lineage_hash,
        SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD,
        report,
        "",
        metadata={
            "source_mesh_used": False,
            "surface_support_convex_transfer": True,
            "source_skin_rows_consumed": True,
            "direct_model_query": False,
            "historical_weight_transfer_used": False,
            "semantic_skin_synthesis": False,
        },
    )
    payload = value.to_dict()
    payload.pop("mesh_skin_lineage_hash", None)
    value = QualifiedMeshSkinIR(
        **{
            **value.__dict__,
            "mesh_skin_lineage_hash": content_sha256(payload),
        }
    )
    validate_qualified_mesh_skin(value, surface=surface, skeleton=skeleton, skin=skin, mesh=mesh)
    return value


__all__ = ["SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD", "bind_mwb2_mesh_skin"]
