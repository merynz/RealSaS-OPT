from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_services.proof.failure_signatures import (
    derive_failure_signatures,
    no_owner_attribution,
)

from .deformation import (
    effective_motion_track_count,
    mesh_triangle_area_report,
    verified_mesh_lbs_measurement,
)
from .types import QualificationError
from .v4 import (
    bind_domain_proof,
    bind_measurement_report,
    bind_product_proof_bundle,
    bind_proof_plan,
    required_proof_domains,
    validate_directional_renderable_set,
    validate_motion_against_mechanical,
    validate_product_ontology,
)

_REQUIRED = {
    "MECHANICAL_STRUCTURE",
    "MESH_QUALITY",
    "DEFORMATION",
    "DIRECTIONAL_VISUAL",
    "MOTION",
    "RUNTIME_CONSUMPTION",
}


def _mechanical(product):
    sk = product.mechanical_state.skeleton
    ids = {j.canonical_joint_id for j in sk.joints}
    bad_parent = sum(
        j.parent_canonical_id is not None and j.parent_canonical_id not in ids
        for j in sk.joints
    )
    unsupported = sum(not j.support_surface_ids for j in sk.joints)
    return {
        "joint_count": len(sk.joints),
        "deform_root_count": len(sk.deform_root_ids),
        "illegal_parent_count": int(bad_parent),
        "unsupported_joint_count": int(unsupported),
    }


def _mesh(product):
    reports = [
        mesh_triangle_area_report(c.mesh)
        for d in product.directional_renderables.directions
        for c in d.components
    ]
    return {
        "component_count": len(reports),
        "face_count": sum(r["face_count"] for r in reports),
        "degenerate_faces": sum(r["degenerate_faces"] for r in reports),
        "min_area": min((r["min_area"] for r in reports), default=0.0),
    }


def _visual(product):
    validate_directional_renderable_set(
        product.directional_renderables,
        product.mechanical_state,
    )
    corners = sum(
        len(c.appearance.corner_bindings)
        for d in product.directional_renderables.directions
        for c in d.components
    )
    return {
        "direction_count": len(product.directional_renderables.directions),
        "corner_binding_count": corners,
        "view_order": [d.view_index for d in product.directional_renderables.directions],
    }


def _motion(product):
    validate_motion_against_mechanical(product.motion_state, product.mechanical_state)
    return {
        "clip_count": len(product.motion_state.clips),
        "joint_track_count": len(product.motion_state.joint_tracks),
        "effective_joint_track_count": effective_motion_track_count(product.motion_state),
    }


def _runtime(product):
    return {
        "representation_class": product.representation_class,
        "full_3d_reconstruction_authority": bool(product.full_3d_reconstruction_authority),
        "direction_count": len(product.directional_renderables.directions),
        "motion_representation": product.motion_state.representation_class,
    }


def _status(domain, m):
    if domain == "MECHANICAL_STRUCTURE":
        ok = (
            m["joint_count"] > 0
            and m["deform_root_count"] > 0
            and m["illegal_parent_count"] == 0
            and m["unsupported_joint_count"] == 0
        )
    elif domain == "MESH_QUALITY":
        ok = (
            m["component_count"] >= 8
            and m["face_count"] > 0
            and m["degenerate_faces"] == 0
            and m["min_area"] > 1e-12
        )
    elif domain == "DIRECTIONAL_VISUAL":
        ok = (
            m["direction_count"] == 8
            and m["view_order"] == list(range(8))
            and m["corner_binding_count"] > 0
        )
    elif domain == "MOTION":
        ok = m["clip_count"] > 0 and m["effective_joint_track_count"] > 0
    elif domain == "RUNTIME_CONSUMPTION":
        ok = (
            m["representation_class"] == "DIRECTIONAL_2D_2P5D_PUPPET"
            and not m["full_3d_reconstruction_authority"]
            and m["direction_count"] == 8
            and m["motion_representation"] == "DIRECTIONAL_2D_2P5D_PUPPET_MOTION"
        )
    elif domain == "DEFORMATION":
        ok = (
            m.get("rms", float("inf")) <= float(m.get("rms_threshold", 1e-6))
            and m.get("p95", float("inf")) <= float(m.get("p95_threshold", 1e-6))
        )
    else:
        raise ValueError(f"unknown proof domain:{domain}")
    return "PASS" if ok else "FAIL"


def evaluate_product_proof(product, *, deformation_fixture: dict | None = None):
    """Evaluate current-compiler proof domains and bind diagnostic evidence.

    Proof status remains canonical-core authority. Diagnostic services may
    explain a FAIL/ABSTAIN, but geometry/measurement failure alone is never
    treated as causal owner attribution.
    """
    validate_product_ontology(product)
    required = set(required_proof_domains(product.capability_contract))
    unknown = required - _REQUIRED
    if unknown:
        raise QualificationError(f"UNIMPLEMENTED_REQUIRED_PROOF_DOMAIN:{sorted(unknown)}")

    reports = []
    for domain in sorted(required):
        plan = bind_proof_plan(
            product,
            proof_domain=domain,
            operator_policy_hashes=(f"RealSaS.ProofEngine.{domain}.v1",),
            probe_specification={"deterministic": True, "domain": domain},
        )
        if domain == "MECHANICAL_STRUCTURE":
            measurements = _mechanical(product)
        elif domain == "MESH_QUALITY":
            measurements = _mesh(product)
        elif domain == "DIRECTIONAL_VISUAL":
            measurements = _visual(product)
        elif domain == "MOTION":
            measurements = _motion(product)
        elif domain == "RUNTIME_CONSUMPTION":
            measurements = _runtime(product)
        elif domain == "DEFORMATION":
            if deformation_fixture is None:
                measurements = {"status": "MISSING_FIXTURE"}
            else:
                component = product.directional_renderables.directions[
                    int(deformation_fixture.get("view_index", 0))
                ].components[int(deformation_fixture.get("component_index", 0))]
                measurements = verified_mesh_lbs_measurement(
                    component.mesh,
                    component.mesh_skin,
                    product.mechanical_state.skeleton,
                    deformation_fixture["transforms"],
                    deformation_fixture["expected"],
                )
                measurements.update(
                    {
                        "rms_threshold": float(deformation_fixture.get("rms_threshold", 1e-6)),
                        "p95_threshold": float(deformation_fixture.get("p95_threshold", 1e-6)),
                    }
                )

        status = (
            "ABSTAIN"
            if domain == "DEFORMATION" and deformation_fixture is None
            else _status(domain, measurements)
        )
        failures = derive_failure_signatures(domain, measurements, status=status)

        # Causal owner attribution is intentionally fail-closed. A later
        # controlled mutation/fault experiment may bind independently proven
        # ownership, but a domain failure is not itself owner evidence.
        owners = no_owner_attribution()

        measurement_report = bind_measurement_report(
            product,
            plan,
            measurements=measurements,
        )
        reports.append(
            bind_domain_proof(
                product,
                plan,
                measurement_report,
                status=status,
                failure_signatures=failures,
                owner_attribution=owners,
                metadata={
                    "causal_mutation_gate": "SOURCE_TEST_REQUIRED",
                    "diagnostic_service": "RealSaS.CompilerServices.ProofFailureSignatures.v1",
                    "causal_owner_attribution": "NOT_PERFORMED",
                    "owner_attribution_requires_controlled_fault_experiment": True,
                },
            )
        )
    return bind_product_proof_bundle(
        product,
        tuple(reports),
        metadata={"engine": "RealSaS.ProofEngine.v1", "heavy_solver_promoted": False},
    )


def mutation_worsens_measurement(domain: str, baseline: dict, mutated: dict) -> bool:
    if domain == "MESH_QUALITY":
        return (
            mutated.get("degenerate_faces", 0) > baseline.get("degenerate_faces", 0)
            or mutated.get("min_area", 0) < baseline.get("min_area", 0)
        )
    if domain == "DEFORMATION":
        return (
            mutated.get("rms", 0) > baseline.get("rms", 0)
            or mutated.get("p95", 0) > baseline.get("p95", 0)
        )
    if domain == "DIRECTIONAL_VISUAL":
        return (
            mutated.get("direction_count", 8) < baseline.get("direction_count", 8)
            or mutated.get("corner_binding_count", 0) < baseline.get("corner_binding_count", 0)
        )
    if domain == "MOTION":
        return mutated.get("effective_joint_track_count", 0) < baseline.get(
            "effective_joint_track_count", 0
        )
    if domain == "MECHANICAL_STRUCTURE":
        return (
            mutated.get("illegal_parent_count", 0) > baseline.get("illegal_parent_count", 0)
            or mutated.get("deform_root_count", 0) < baseline.get("deform_root_count", 0)
            or mutated.get("unsupported_joint_count", 0)
            > baseline.get("unsupported_joint_count", 0)
        )
    if domain == "RUNTIME_CONSUMPTION":
        return (
            mutated.get("representation_class") != baseline.get("representation_class")
            or bool(mutated.get("full_3d_reconstruction_authority"))
        )
    raise ValueError(domain)
