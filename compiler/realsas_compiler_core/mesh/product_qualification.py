from __future__ import annotations

from dataclasses import replace

from .mesh_binding import mesh_lineage_hash, validate_qualified_mesh
from .mwb2_cdt import qualify_mwb2_observation_cdt_mesh
from .observation_domain import ObservationRasterDomain
from .quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    MeshQualityPolicyV1,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
)
from .types import MeshDiscretizationCandidateIR, QualificationError, RiggingSurfaceIR


FIT2_PRODUCT_MESH_QUALIFIER_ID = "RealSaS.MeshProductQualification.FIT2.v1"


def _exact_mesh_raster_triangles(mesh, *, surface: RiggingSurfaceIR, view_index: int):
    """Re-derive exact mesh triangles in the admitted surface raster frame.

    Product coverage must not trust candidate-reported residual metrics.  Every mesh
    vertex is projected from its qualified SurfaceSupportBinding using the exact
    RiggingSurfaceIR raster bindings for this view, including supported local-convex
    inserted vertices.
    """
    surface_raster: dict[str, tuple[float, float]] = {}
    for node in surface.surface_nodes:
        rows = [tuple(map(float, xy)) for v, xy in node.raster_bindings if int(v) == int(view_index)]
        if len(rows) > 1:
            raise QualificationError(
                f"FIT2_PRODUCT_MESH_DUPLICATE_SURFACE_RASTER:{node.surface_id}:{view_index}"
            )
        if rows:
            surface_raster[str(node.surface_id)] = rows[0]

    by_id: dict[str, tuple[float, float]] = {}
    for vertex in mesh.vertices:
        x = y = total = 0.0
        for sid, coefficient in vertex.support_binding.coefficients:
            sid = str(sid)
            if sid not in surface_raster:
                raise QualificationError(
                    f"FIT2_PRODUCT_MESH_SUPPORT_NOT_RASTER_BOUND:{view_index}:{sid}"
                )
            c = float(coefficient)
            x += c * surface_raster[sid][0]
            y += c * surface_raster[sid][1]
            total += c
        if abs(total - 1.0) > 1.0e-8:
            raise QualificationError("FIT2_PRODUCT_MESH_SUPPORT_SIMPLEX_RESIDUAL")
        cached_xy = (getattr(vertex, "metadata", {}) or {}).get("raster_xy")
        if cached_xy is not None:
            cx, cy = map(float, cached_xy)
            if max(abs(cx - x), abs(cy - y)) > 1.0e-7:
                raise QualificationError(
                    f"FIT2_PRODUCT_MESH_CACHED_RASTER_BINDING_DRIFT:{vertex.canonical_mesh_vertex_id}"
                )
        by_id[str(vertex.canonical_mesh_vertex_id)] = (float(x), float(y))

    triangles = []
    for face in mesh.faces:
        if len(face) != 3:
            raise QualificationError("FIT2_PRODUCT_MESH_REQUIRES_TRIANGLES")
        ids = tuple(map(str, face))
        if any(vertex_id not in by_id for vertex_id in ids):
            raise QualificationError("FIT2_PRODUCT_MESH_FACE_REFERENCES_UNKNOWN_VERTEX")
        triangles.append((by_id[ids[0]], by_id[ids[1]], by_id[ids[2]]))
    return tuple(triangles)


def _validate_observation_binding(candidate, observation_domain: ObservationRasterDomain) -> None:
    observation_domain.validate()
    if int(observation_domain.view_index) != int(candidate.view_index):
        raise QualificationError("FIT2_PRODUCT_MESH_OBSERVATION_VIEW_MISMATCH")
    if candidate.metadata.get("observation_mask_sha256") != observation_domain.mask_sha256:
        raise QualificationError("FIT2_PRODUCT_MESH_OBSERVATION_MASK_HASH_MISMATCH")
    if candidate.metadata.get("source_alpha_sha256") != observation_domain.source_alpha_sha256:
        raise QualificationError("FIT2_PRODUCT_MESH_SOURCE_ALPHA_HASH_MISMATCH")

    matching = [
        row for row in candidate.boundary_constraints
        if row.get("kind") == "EXACT_OBSERVATION_ALPHA_DOMAIN"
        and int(row.get("view_index", -1)) == int(candidate.view_index)
    ]
    if len(matching) != 1:
        raise QualificationError("FIT2_PRODUCT_MESH_EXACT_ALPHA_BOUNDARY_AUTHORITY_MISSING")
    boundary = matching[0]
    if boundary.get("mask_sha256") != observation_domain.mask_sha256:
        raise QualificationError("FIT2_PRODUCT_MESH_BOUNDARY_MASK_HASH_MISMATCH")
    if boundary.get("source_alpha_sha256") != observation_domain.source_alpha_sha256:
        raise QualificationError("FIT2_PRODUCT_MESH_BOUNDARY_SOURCE_ALPHA_HASH_MISMATCH")
    if int(boundary.get("width", -1)) != int(observation_domain.width) or int(
        boundary.get("height", -1)
    ) != int(observation_domain.height):
        raise QualificationError("FIT2_PRODUCT_MESH_BOUNDARY_DIMENSION_MISMATCH")


def qualify_fit2_product_mwb2_observation_cdt_mesh(
    surface: RiggingSurfaceIR,
    candidate: MeshDiscretizationCandidateIR,
    *,
    observation_domain: ObservationRasterDomain,
    policy: MeshQualityPolicyV1 = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
):
    """Promote a directional CDT mesh only after the frozen FIT2 product gates pass.

    ``qualify_mwb2_observation_cdt_mesh`` is the typed CDT compatibility/admission
    gate. It proves support lineage and deterministic-kernel authority, but its
    historical coarse coverage floor is not product proof.

    FIT2 product admission binds the exact observation authority, rebuilds the
    qualified mesh's raster triangles from admitted surface support, recomputes
    coverage from those exact triangles, remeasures topology, and applies the frozen
    product policy fail-closed. Candidate residual metrics are retained only as
    diagnostics and never trusted as the product coverage measurement.
    """
    policy.validate()
    _validate_observation_binding(candidate, observation_domain)

    # Legal current-authority CDT/surface support remains a prerequisite.
    mesh = qualify_mwb2_observation_cdt_mesh(surface, candidate)

    exact_triangles = _exact_mesh_raster_triangles(
        mesh,
        surface=surface,
        view_index=int(candidate.view_index),
    )
    coverage = observation_domain.coverage(exact_triangles)
    raster_report = mesh_raster_quality_report(
        mesh,
        surface=surface,
        view_index=int(candidate.view_index),
    )
    quality = evaluate_mesh_quality(
        coverage=coverage,
        raster_report=raster_report,
        policy=policy,
    )
    if not bool(quality.get("passed", False)):
        failures = tuple(map(str, quality.get("failure_invariants") or ()))
        suffix = ",".join(failures) if failures else "unspecified"
        raise QualificationError(f"FIT2_PRODUCT_MESH_QUALITY_GATE_FAIL:{suffix}")

    report = dict(mesh.qualification_report)
    report.update(
        {
            "status": "PASS_FIT2_PRODUCT_MESH_QUALIFICATION",
            "product_mesh_qualifier": FIT2_PRODUCT_MESH_QUALIFIER_ID,
            "fit2_product_mesh_quality_pass": True,
            "fit2_product_mesh_quality": quality,
            "candidate_reported_coverage_diagnostics": dict(candidate.residual_report or {}),
            "coverage_recomputed_from_exact_observation_authority": True,
            "exact_directional_raster_remeasurement": True,
            "observation_mask_sha256": observation_domain.mask_sha256,
            "source_alpha_sha256": observation_domain.source_alpha_sha256,
            "lower_level_cdt_pass_is_not_product_pass": True,
        }
    )
    updated = replace(
        mesh,
        qualification_report=report,
        metadata={
            **mesh.metadata,
            "product_mesh_qualifier": FIT2_PRODUCT_MESH_QUALIFIER_ID,
            "product_mesh_policy_schema": policy.schema_version,
            "product_observation_mask_sha256": observation_domain.mask_sha256,
            "product_source_alpha_sha256": observation_domain.source_alpha_sha256,
        },
        mesh_lineage_hash="",
    )
    updated = replace(updated, mesh_lineage_hash=mesh_lineage_hash(updated))
    validate_qualified_mesh(updated, surface)
    return updated
