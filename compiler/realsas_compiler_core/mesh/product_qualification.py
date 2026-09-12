from __future__ import annotations

from dataclasses import replace

from .mesh_binding import mesh_lineage_hash, validate_qualified_mesh
from .mwb2_cdt import qualify_mwb2_observation_cdt_mesh
from .quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    MeshQualityPolicyV1,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
)
from .types import MeshDiscretizationCandidateIR, QualificationError, RiggingSurfaceIR


FIT2_PRODUCT_MESH_QUALIFIER_ID = "RealSaS.MeshProductQualification.FIT2.v1"


def qualify_fit2_product_mwb2_observation_cdt_mesh(
    surface: RiggingSurfaceIR,
    candidate: MeshDiscretizationCandidateIR,
    *,
    policy: MeshQualityPolicyV1 = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
):
    """Promote a directional CDT mesh only after the frozen FIT2 product gates pass.

    ``qualify_mwb2_observation_cdt_mesh`` is the historical/current typed CDT
    admission gate.  It proves support lineage, deterministic kernel authority and
    a coarse observation-domain floor.  Product admission is deliberately stricter:
    the exact directional mesh is re-measured in its authoritative raster frame and
    must satisfy the frozen FIT2 coverage, connected-hole and topology policy.

    This wrapper is the FIT2 product entry point.  Calling the lower-level CDT
    qualifier directly is not evidence of product-mesh closure.
    """
    policy.validate()

    # First prove that the candidate is a legal current-authority CDT mesh.  The
    # lower-level 0.90 recall floor is intentionally only a compatibility/sanity
    # floor; the frozen product policy below is authoritative and stricter.
    mesh = qualify_mwb2_observation_cdt_mesh(surface, candidate)

    coverage = dict(candidate.residual_report or {})
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
            "exact_directional_raster_remeasurement": True,
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
        },
        mesh_lineage_hash="",
    )
    updated = replace(updated, mesh_lineage_hash=mesh_lineage_hash(updated))
    validate_qualified_mesh(updated, surface)
    return updated
