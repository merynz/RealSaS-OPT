from __future__ import annotations

"""Representation-aware MESH_QUALITY authority for product proof rebinds.

External directional render meshes carry two different coordinate meanings:
``QualifiedMeshVertex.P`` is the qualified mechanical support address used to replay
fresh S/G/W, while ``metadata.raster_xy`` is the exact directional render topology
consumed by appearance and raster deformation.  The product mesh-quality domain must
therefore measure nondegeneracy/area in raster space for *qualified external render
support*, without changing the generic support-space geometry semantics used by
canonical mechanical meshes.

This module intentionally does not rebuild product state or motion bakes.  It can
replace only a failing MESH_QUALITY domain in an already product-bound proof bundle;
all other required domains must already be PASS and are preserved byte-for-byte at
the IR level.
"""

from compiler.realsas_compiler_services.proof.failure_signatures import (
    derive_failure_signatures,
    no_owner_attribution,
)

from . import proof_engine as v7
from .deformation import mesh_triangle_area_report
from .mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
    raster_quality_gate_failures,
)
from .product_external_render import validate_external_renderable_component
from .types import QualificationError
from .v4 import (
    bind_domain_proof,
    bind_measurement_report,
    bind_product_proof_bundle,
    bind_proof_plan,
    required_proof_domains,
    validate_product_ontology,
)

_EXTERNAL_RASTER_AUTHORITY = "QUALIFIED_EXTERNAL_COMPONENT_CACHED_RASTER_WITNESS"
_CANONICAL_SUPPORT_AREA_AUTHORITY = "CANONICAL_MESH_VERTEX_P_SUPPORT_SPACE"
_ENGINE = "RealSaS.ProofEngine.v8.representation_aware_mesh_area_authority"
_OPERATOR = "RealSaS.ProofEngine.MESH_QUALITY.v8.representation_aware_area_authority"


def _authoritative_area_summary(*, support_area: dict, raster: dict, external: bool) -> dict:
    """Return the area/nondegeneracy fields that own the MESH_QUALITY verdict."""
    if external:
        return {
            "face_count": int(raster["face_count"]),
            "degenerate_faces": int(raster["degenerate_faces"]),
            "min_area": float(raster["min_raster_triangle_area"]),
            "mesh_area_authority": _EXTERNAL_RASTER_AUTHORITY,
        }
    return {
        "face_count": int(support_area["face_count"]),
        "degenerate_faces": int(support_area["degenerate_faces"]),
        "min_area": float(support_area["min_area"]),
        "mesh_area_authority": _CANONICAL_SUPPORT_AREA_AUTHORITY,
    }


def measure_mesh_quality(product) -> dict:
    """Measure product MESH_QUALITY without relabelling mechanical support geometry."""
    validate_product_ontology(product)
    external_set = v7._external_render_set_claim(product.directional_renderables)
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    rows = []
    raster_reports = []
    authoritative_area_reports = []
    support_area_reports = []
    cdt_rows = []
    external_component_count = 0

    for direction in product.directional_renderables.directions:
        for component in direction.components:
            external = v7._external_component_claim(component)
            if bool(external) != bool(external_set):
                raise QualificationError("MESH_V8_EXTERNAL_SET_COMPONENT_AUTHORITY_DRIFT")

            support_area = mesh_triangle_area_report(component.mesh)
            if external:
                validate_external_renderable_component(component, product.mechanical_state)
                raster = mesh_raster_quality_report(
                    component.mesh,
                    surface=None,
                    view_index=int(direction.view_index),
                )
                raster_authority = _EXTERNAL_RASTER_AUTHORITY
                external_component_count += 1
            else:
                raster = mesh_raster_quality_report(
                    component.mesh,
                    surface=product.mechanical_state.surface,
                    view_index=int(direction.view_index),
                )
                raster_authority = v7._SCIENTIFIC_RASTER_AUTHORITY

            authoritative_area = _authoritative_area_summary(
                support_area=support_area,
                raster=raster,
                external=external,
            )
            authoritative_area_reports.append(authoritative_area)
            support_area_reports.append(support_area)
            raster_reports.append(raster)

            qreport = dict(component.mesh.qualification_report or {})
            coverage = dict(qreport.get("candidate_residual_report") or {})
            row = {
                "view_index": int(direction.view_index),
                "component_id": str(component.component_id),
                "coverage_classification": str(component.mesh.support_coverage_classification),
                "raster_authority": raster_authority,
                "external_render_support": bool(external),
                "support_space_face_count": int(support_area["face_count"]),
                "support_space_degenerate_faces": int(support_area["degenerate_faces"]),
                "support_space_min_area": float(support_area["min_area"]),
                **raster,
                **authoritative_area,
            }

            if component.mesh.support_coverage_classification == "OBSERVATION_DOMAIN_CDT":
                for key in ("source_alpha_recall", "precision_inside_alpha"):
                    if key not in coverage and key in qreport:
                        coverage[key] = qreport[key]
                evaluated = evaluate_mesh_quality(
                    coverage=coverage,
                    raster_report=raster,
                    policy=policy,
                )
                row.update({
                    "strict_product_mesh_gate": True,
                    "mesh_quality_passed": bool(evaluated["passed"]),
                    "mesh_quality_failure_invariants": tuple(evaluated["failure_invariants"]),
                    "coverage": coverage,
                })
                cdt_rows.append(evaluated)
            else:
                failures = raster_quality_gate_failures(raster, policy=policy)
                row.update({
                    "strict_product_mesh_gate": False,
                    "mesh_quality_passed": not failures,
                    "mesh_quality_failure_invariants": tuple(failures),
                })
            rows.append(row)

    return {
        "component_count": len(rows),
        "external_render_support_component_count": external_component_count,
        "external_render_support_fail_closed": True,
        "face_count": sum(int(r["face_count"]) for r in authoritative_area_reports),
        "degenerate_faces": sum(int(r["degenerate_faces"]) for r in authoritative_area_reports),
        "min_area": min((float(r["min_area"]) for r in authoritative_area_reports), default=0.0),
        "mesh_area_authority": (
            _EXTERNAL_RASTER_AUTHORITY if external_set else _CANONICAL_SUPPORT_AREA_AUTHORITY
        ),
        "support_space_area_is_diagnostic": bool(external_set),
        "support_space_degenerate_faces": sum(int(r["degenerate_faces"]) for r in support_area_reports),
        "support_space_min_area": min((float(r["min_area"]) for r in support_area_reports), default=0.0),
        "duplicate_faces": sum(int(r["duplicate_faces"]) for r in raster_reports),
        "nonmanifold_edges": sum(int(r["nonmanifold_edges"]) for r in raster_reports),
        "min_raster_triangle_angle_deg": min(
            (float(r["min_raster_triangle_angle_deg"]) for r in raster_reports),
            default=0.0,
        ),
        "max_raster_triangle_aspect_ratio": max(
            (float(r["max_raster_triangle_aspect_ratio"]) for r in raster_reports),
            default=0.0,
        ),
        "strict_cdt_component_count": len(cdt_rows),
        "strict_cdt_pass_count": sum(bool(r["passed"]) for r in cdt_rows),
        "min_source_alpha_recall": min(
            (float(r.get("source_alpha_recall", 1.0)) for r in cdt_rows), default=1.0
        ),
        "min_precision_inside_alpha": min(
            (float(r.get("precision_inside_alpha", 1.0)) for r in cdt_rows), default=1.0
        ),
        "min_alpha_iou": min((float(r.get("alpha_iou", 1.0)) for r in cdt_rows), default=1.0),
        "max_largest_uncovered_component_fraction": max(
            (float(r.get("largest_uncovered_component_fraction", 0.0)) for r in cdt_rows),
            default=0.0,
        ),
        "min_large_alpha_component_recall": min(
            (float(r.get("min_large_alpha_component_recall", 1.0)) for r in cdt_rows),
            default=1.0,
        ),
        "mesh_quality_policy": policy.to_dict(),
        "component_reports": rows,
    }


def rebind_failed_mesh_domain(product, existing_bundle):
    """Replace only a failing MESH_QUALITY report; preserve all other PASS domains."""
    validate_product_ontology(product)
    if existing_bundle.source_product_state_hash != product.product_state_hash:
        raise QualificationError("MESH_V8_EXISTING_BUNDLE_PRODUCT_HASH_DRIFT")

    required_now = tuple(sorted(required_proof_domains(product.capability_contract)))
    required_old = tuple(sorted(existing_bundle.required_domains))
    if required_old != required_now:
        raise QualificationError("MESH_V8_EXISTING_BUNDLE_REQUIRED_DOMAIN_DRIFT")

    reports = tuple(existing_bundle.domain_reports)
    by_domain = {str(report.proof_domain): report for report in reports}
    if len(by_domain) != len(reports) or set(by_domain) != set(required_now):
        raise QualificationError("MESH_V8_EXISTING_BUNDLE_DOMAIN_SET_INVALID")
    if "MESH_QUALITY" not in by_domain:
        raise QualificationError("MESH_V8_MESH_DOMAIN_MISSING")
    if by_domain["MESH_QUALITY"].status != "FAIL":
        raise QualificationError("MESH_V8_REBIND_REQUIRES_FAILING_MESH_DOMAIN")

    non_mesh_failures = {
        domain: report.status
        for domain, report in by_domain.items()
        if domain != "MESH_QUALITY" and report.status != "PASS"
    }
    if non_mesh_failures:
        raise QualificationError(f"MESH_V8_NON_MESH_DOMAIN_NOT_PASS:{sorted(non_mesh_failures.items())}")

    plan = bind_proof_plan(
        product,
        proof_domain="MESH_QUALITY",
        operator_policy_hashes=(_OPERATOR,),
        probe_specification={
            "deterministic": True,
            "domain": "MESH_QUALITY",
            "area_authority": "REPRESENTATION_AWARE",
        },
    )
    measurements = measure_mesh_quality(product)
    status = v7._status("MESH_QUALITY", measurements)
    failures = derive_failure_signatures("MESH_QUALITY", measurements, status=status)
    measurement_report = bind_measurement_report(product, plan, measurements=measurements)
    mesh_metadata = {
        "causal_mutation_gate": "SOURCE_TEST_REQUIRED",
        "diagnostic_service": "RealSaS.CompilerServices.ProofFailureSignatures.v1",
        "causal_owner_attribution": "NOT_PERFORMED",
        "owner_attribution_requires_controlled_fault_experiment": True,
        "alpha_domain_coverage_required_for_observation_cdt": True,
        "large_uncovered_region_gate_required": True,
        "mesh_quality_policy": FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.to_dict(),
        "external_render_support_fail_closed": True,
        "external_render_support_uses_scientific_surface_fallback": False,
        "external_raster_authority": _EXTERNAL_RASTER_AUTHORITY,
        "mesh_area_authority": measurements["mesh_area_authority"],
        "support_space_area_is_diagnostic": bool(measurements["support_space_area_is_diagnostic"]),
        "threshold_relaxation_used": False,
    }
    replacement = bind_domain_proof(
        product,
        plan,
        measurement_report,
        status=status,
        failure_signatures=failures,
        owner_attribution=no_owner_attribution(),
        metadata=mesh_metadata,
    )

    rebound_reports = tuple(
        replacement if report.proof_domain == "MESH_QUALITY" else report
        for report in reports
    )
    preserved = tuple(sorted(domain for domain in by_domain if domain != "MESH_QUALITY"))
    return bind_product_proof_bundle(product, rebound_reports, metadata={
        "engine": _ENGINE,
        "source_proof_bundle_hash": existing_bundle.proof_bundle_hash,
        "mesh_only_rebind": True,
        "preserved_exact_product_bound_pass_domains": preserved,
        "motion_bakes_recomputed": False,
        "product_state_rebuilt": False,
        "threshold_relaxation_used": False,
        "post_export_native_interlock_required": bool(
            (existing_bundle.metadata or {}).get("post_export_native_interlock_required", True)
        ),
    })


__all__ = [
    "measure_mesh_quality",
    "rebind_failed_mesh_domain",
]
