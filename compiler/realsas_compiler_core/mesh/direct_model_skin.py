from __future__ import annotations

"""Compiler qualification for weights predicted directly on an exact mesh.

This path is deliberately distinct from historical mesh<-surface skin transfer.
A learned model supplies one row-simplex weight vector for every exact mesh vertex;
the Compiler only validates IDs, legality, bounded numerical normalization and
lineage, then emits QualifiedMeshSkinIR bound to the exact M/G/model-skin authority.
"""

from math import isfinite
from typing import Mapping, Sequence

from .mesh_binding import (
    mesh_skin_lineage_hash,
    validate_qualified_mesh,
    validate_qualified_mesh_skin,
)
from .types import (
    QualifiedEditableMeshIR,
    QualifiedMeshSkinIR,
    QualifiedMeshSkinRow,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualificationError,
    RiggingSurfaceIR,
)


DIRECT_MODEL_TRANSFER_METHOD = "ARACHNE_V5_DIRECT_QUERY_EXACT_MESH_V1"


def qualify_direct_model_mesh_skin(
    surface: RiggingSurfaceIR,
    skeleton: QualifiedSkeletonIR,
    source_skin_authority: QualifiedSkinIR,
    mesh: QualifiedEditableMeshIR,
    *,
    joint_ids: Sequence[str],
    predicted_weights_by_vertex: Mapping[str, Sequence[float]],
    model_provenance: Mapping[str, object],
    max_simplex_repair_l1: float = 1.0e-5,
    max_total_correction_l1: float = 2.0e-3,
    negative_tolerance: float = 1.0e-8,
) -> QualifiedMeshSkinIR:
    """Qualify direct neural weights on the exact mesh; no source-skin row transfer.

    ``source_skin_authority`` contributes only the sealed learned-skin lineage and
    skeleton/model provenance. Its surface rows are never read or interpolated.
    ``predicted_weights_by_vertex`` must contain exactly one vector for every
    canonical vertex in ``mesh`` and no extra IDs.
    """
    if max_simplex_repair_l1 < 0.0 or max_total_correction_l1 < 0.0 or negative_tolerance < 0.0:
        raise ValueError("direct mesh-skin repair budgets/tolerance must be nonnegative")

    validate_qualified_mesh(mesh, surface)

    if source_skin_authority.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("DIRECT_MODEL_SKIN_AUTHORITY_SKELETON_MISMATCH")

    ordered_joint_ids = tuple(map(str, joint_ids))
    if not ordered_joint_ids or len(ordered_joint_ids) != len(set(ordered_joint_ids)):
        raise QualificationError("DIRECT_MODEL_JOINT_IDS_INVALID")
    legal_joint_ids = {str(j.canonical_joint_id) for j in skeleton.joints}
    if set(ordered_joint_ids) != legal_joint_ids:
        raise QualificationError("DIRECT_MODEL_JOINT_SET_MISMATCH")

    ordered_vertices = tuple(sorted(mesh.vertices, key=lambda v: str(v.canonical_mesh_vertex_id)))
    mesh_vertex_ids = tuple(str(v.canonical_mesh_vertex_id) for v in ordered_vertices)
    submitted_ids = set(map(str, predicted_weights_by_vertex))
    if submitted_ids != set(mesh_vertex_ids):
        missing = sorted(set(mesh_vertex_ids) - submitted_ids)
        extra = sorted(submitted_ids - set(mesh_vertex_ids))
        raise QualificationError(f"DIRECT_MODEL_WEIGHT_VERTEX_SET_MISMATCH:missing={missing[:8]}:extra={extra[:8]}")

    rows = []
    corrected_rows = 0
    max_residual = 0.0
    max_row_correction = 0.0
    total_correction = 0.0
    tiny_negative_count = 0
    tiny_negative_mass = 0.0

    for vertex in ordered_vertices:
        vertex_id = str(vertex.canonical_mesh_vertex_id)
        raw = tuple(float(x) for x in predicted_weights_by_vertex[vertex_id])
        if len(raw) != len(ordered_joint_ids):
            raise QualificationError(f"DIRECT_MODEL_WEIGHT_WIDTH_MISMATCH:{vertex_id}:{len(raw)}:{len(ordered_joint_ids)}")
        if any(not isfinite(x) for x in raw):
            raise QualificationError(f"DIRECT_MODEL_WEIGHT_NONFINITE:{vertex_id}")
        if any(x < -negative_tolerance for x in raw):
            raise QualificationError(f"DIRECT_MODEL_WEIGHT_MATERIAL_NEGATIVE:{vertex_id}")

        clipped = []
        for x in raw:
            if x < 0.0:
                tiny_negative_count += 1
                tiny_negative_mass += -x
                clipped.append(0.0)
            else:
                clipped.append(x)
        raw_sum = sum(raw)
        residual = abs(raw_sum - 1.0)
        max_residual = max(max_residual, residual)
        clipped_sum = sum(clipped)
        if clipped_sum <= 1.0e-12:
            raise QualificationError(f"DIRECT_MODEL_WEIGHT_ZERO_ROW:{vertex_id}")
        normalized = tuple(x / clipped_sum for x in clipped)
        correction = sum(abs(a - b) for a, b in zip(normalized, raw))
        if correction > max_simplex_repair_l1 + 1.0e-15:
            raise QualificationError(f"DIRECT_MODEL_SIMPLEX_REPAIR_BUDGET_EXCEEDED:{vertex_id}:{correction}")
        total_correction += correction
        if total_correction > max_total_correction_l1 + 1.0e-15:
            raise QualificationError(f"DIRECT_MODEL_TOTAL_REPAIR_BUDGET_EXCEEDED:{total_correction}")
        max_row_correction = max(max_row_correction, correction)
        if correction > 1.0e-12:
            corrected_rows += 1

        influences = tuple(
            (joint_id, float(weight))
            for joint_id, weight in zip(ordered_joint_ids, normalized)
            if float(weight) > 0.0
        )
        if not influences:
            raise QualificationError(f"DIRECT_MODEL_WEIGHT_EMPTY_ADMITTED_ROW:{vertex_id}")
        rows.append(
            QualifiedMeshSkinRow(
                vertex_id,
                influences,
                tuple((str(sid), float(coeff)) for sid, coeff in vertex.support_binding.coefficients),
                float(residual),
                float(correction),
            )
        )

    provenance = {str(k): v for k, v in model_provenance.items()}
    report = {
        "status": "PASS_DIRECT_MODEL_EXACT_MESH_SKIN_QUALIFICATION",
        "transfer_method": DIRECT_MODEL_TRANSFER_METHOD,
        "row_count": len(rows),
        "joint_count": len(ordered_joint_ids),
        "joint_ids": list(ordered_joint_ids),
        "source_skin_rows_consumed": False,
        "historical_barycentric_weight_transfer_used": False,
        "semantic_skin_synthesis": True,
        "direct_model_query": True,
        "model_provenance": provenance,
        "corrected_row_count": int(corrected_rows),
        "max_simplex_residual_before": float(max_residual),
        "max_row_correction_l1": float(max_row_correction),
        "total_correction_l1": float(total_correction),
        "bounded_repair_limit_l1_per_row": float(max_simplex_repair_l1),
        "bounded_repair_limit_l1_total": float(max_total_correction_l1),
        "tiny_negative_clipped_count": int(tiny_negative_count),
        "tiny_negative_clipped_mass": float(tiny_negative_mass),
        "mesh_lineage_hash": str(mesh.mesh_lineage_hash),
        "surface_lineage_hash": str(surface.geometry_lineage_hash),
        "skeleton_lineage_hash": str(skeleton.skeleton_lineage_hash),
        "source_skin_authority_lineage_hash": str(source_skin_authority.skin_lineage_hash),
    }

    bound = QualifiedMeshSkinIR(
        tuple(rows),
        str(surface.geometry_lineage_hash),
        str(skeleton.skeleton_lineage_hash),
        str(source_skin_authority.skin_lineage_hash),
        str(mesh.mesh_lineage_hash),
        DIRECT_MODEL_TRANSFER_METHOD,
        report,
        "",
        metadata={
            "direct_model_query": True,
            "source_skin_rows_consumed": False,
            "historical_weight_transfer_used": False,
            "model_provenance": provenance,
        },
    )
    bound = QualifiedMeshSkinIR(**{**bound.__dict__, "mesh_skin_lineage_hash": mesh_skin_lineage_hash(bound)})
    validate_qualified_mesh_skin(
        bound,
        surface=surface,
        skeleton=skeleton,
        skin=source_skin_authority,
        mesh=mesh,
    )
    return bound


__all__ = ["DIRECT_MODEL_TRANSFER_METHOD", "qualify_direct_model_mesh_skin"]
