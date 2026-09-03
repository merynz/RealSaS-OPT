from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from compiler.realsas_compiler_services.proof.failure_signatures import (
    derive_failure_signatures,
    no_owner_attribution,
)
from compiler.realsas_compiler_services.proof.motion_probe import (
    AuthoredMotionProbePolicyV1,
    SERVICE_ID as AUTHORED_MOTION_PROBE_SERVICE_ID,
    authored_motion_measurement_passes_v1,
    measure_authored_motion_v1,
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

@dataclass(frozen=True)
class ProductProofEvaluationV1:
    proof_bundle: object
    measurement_reports: tuple[object, ...]
    schema_version: str = "RealSaS.ProductProofEvaluation.v1"

    def motion_measurement_report(self):
        reports = [r for r in self.proof_bundle.domain_reports if r.proof_domain == "MOTION"]
        if len(reports) != 1:
            raise QualificationError("PROOF_EVALUATION_REQUIRES_ONE_MOTION_DOMAIN")
        target = reports[0].measurement_report_hash
        matches = [r for r in self.measurement_reports if r.measurement_report_hash == target]
        if len(matches) != 1:
            raise QualificationError("PROOF_EVALUATION_MOTION_MEASUREMENT_MISSING_OR_AMBIGUOUS")
        return matches[0]


_REQUIRED = {
    "MECHANICAL_STRUCTURE", "MESH_QUALITY", "DEFORMATION",
    "DIRECTIONAL_VISUAL", "MOTION", "RUNTIME_CONSUMPTION",
}


def _mechanical(product):
    sk = product.mechanical_state.skeleton
    ids = {j.canonical_joint_id for j in sk.joints}
    return {
        "joint_count": len(sk.joints),
        "deform_root_count": len(sk.deform_root_ids),
        "illegal_parent_count": int(sum(j.parent_canonical_id is not None and j.parent_canonical_id not in ids for j in sk.joints)),
        "unsupported_joint_count": int(sum(not j.support_surface_ids for j in sk.joints)),
    }


def _mesh(product):
    reports = [mesh_triangle_area_report(c.mesh) for d in product.directional_renderables.directions for c in d.components]
    return {
        "component_count": len(reports),
        "face_count": sum(r["face_count"] for r in reports),
        "degenerate_faces": sum(r["degenerate_faces"] for r in reports),
        "min_area": min((r["min_area"] for r in reports), default=0.0),
    }


def _visual(product):
    validate_directional_renderable_set(product.directional_renderables, product.mechanical_state)
    return {
        "direction_count": len(product.directional_renderables.directions),
        "corner_binding_count": sum(len(c.appearance.corner_bindings) for d in product.directional_renderables.directions for c in d.components),
        "view_order": [d.view_index for d in product.directional_renderables.directions],
    }


def _motion(product, policy: AuthoredMotionProbePolicyV1):
    validate_motion_against_mechanical(product.motion_state, product.mechanical_state)
    measurements = measure_authored_motion_v1(product, policy=policy)
    measurements.update({
        "joint_track_count": len(product.motion_state.joint_tracks),
        "effective_joint_track_count": effective_motion_track_count(product.motion_state),
        "dynamic_probe_required_for_pass": True,
    })
    return measurements


def _runtime(product, motion_measurements: dict):
    atlas_ok = True
    uv_ok = True
    for direction in product.directional_renderables.directions:
        hashes = {str(c.appearance.atlas_payload_hash) for c in direction.components}
        if len(hashes) != 1 or "" in hashes:
            atlas_ok = False
        for component in direction.components:
            if dict(component.appearance.metadata or {}).get("material_uv_convention") != "NATIVE_PIXEL_CENTER_TO_NORMALIZED_UV_V1":
                uv_ok = False
    return {
        "representation_class": product.representation_class,
        "full_3d_reconstruction_authority": bool(product.full_3d_reconstruction_authority),
        "direction_count": len(product.directional_renderables.directions),
        "motion_representation": product.motion_state.representation_class,
        "proof_owned_runtime_bake_status": str(motion_measurements.get("runtime_bake_status") or "MISSING"),
        "runtime_v2_single_atlas_per_direction": bool(atlas_ok),
        "runtime_v2_material_uv_projection_supported": bool(uv_ok),
        "runtime_texture_payload_binding_required": True,
        "runtime_export_solver_replay": False,
    }


def _status(domain, m):
    if domain == "MECHANICAL_STRUCTURE":
        ok = m["joint_count"] > 0 and m["deform_root_count"] > 0 and m["illegal_parent_count"] == 0 and m["unsupported_joint_count"] == 0
    elif domain == "MESH_QUALITY":
        ok = m["component_count"] >= 8 and m["face_count"] > 0 and m["degenerate_faces"] == 0 and m["min_area"] > 1e-12
    elif domain == "DIRECTIONAL_VISUAL":
        ok = m["direction_count"] == 8 and m["view_order"] == list(range(8)) and m["corner_binding_count"] > 0
    elif domain == "MOTION":
        ok = authored_motion_measurement_passes_v1(m)
    elif domain == "RUNTIME_CONSUMPTION":
        ok = (
            m["representation_class"] == "DIRECTIONAL_2D_2P5D_PUPPET"
            and not m["full_3d_reconstruction_authority"]
            and m["direction_count"] == 8
            and m["motion_representation"] == "DIRECTIONAL_2D_2P5D_PUPPET_MOTION"
            and str(m.get("proof_owned_runtime_bake_status") or "").startswith("PASS_")
            and bool(m.get("runtime_v2_single_atlas_per_direction"))
            and bool(m.get("runtime_v2_material_uv_projection_supported"))
            and m.get("runtime_export_solver_replay") is False
        )
    elif domain == "DEFORMATION":
        ok = m.get("rms", float("inf")) <= float(m.get("rms_threshold", 1e-6)) and m.get("p95", float("inf")) <= float(m.get("p95_threshold", 1e-6))
    else:
        raise ValueError(f"unknown proof domain:{domain}")
    return "PASS" if ok else "FAIL"


def _plan_binding(domain: str, motion_policy: AuthoredMotionProbePolicyV1):
    if domain == "MOTION":
        return (
            (motion_policy.policy_hash,),
            {
                "deterministic": True,
                "domain": "MOTION",
                "service_id": AUTHORED_MOTION_PROBE_SERVICE_ID,
                "policy": asdict(motion_policy),
                "requested_authored_motion_is_measured": True,
                "causal_owner_attribution": "NOT_PERFORMED",
            },
        )
    return ((f"RealSaS.ProofEngine.{domain}.v1",), {"deterministic": True, "domain": domain})


def evaluate_product_proof_evidence(
    product,
    *,
    deformation_fixture: dict | None = None,
    motion_probe_policy: AuthoredMotionProbePolicyV1 = AuthoredMotionProbePolicyV1(),
):
    """Evaluate current proof domains and preserve exact measurement evidence."""
    validate_product_ontology(product)
    motion_probe_policy.validate()
    required = set(required_proof_domains(product.capability_contract))
    unknown = required - _REQUIRED
    if unknown:
        raise QualificationError(f"UNIMPLEMENTED_REQUIRED_PROOF_DOMAIN:{sorted(unknown)}")

    reports = []
    measurement_reports = []
    measurements_by_domain = {}
    for domain in sorted(required):
        policy_hashes, probe_spec = _plan_binding(domain, motion_probe_policy)
        plan = bind_proof_plan(product, proof_domain=domain, operator_policy_hashes=policy_hashes, probe_specification=probe_spec)
        if domain == "MECHANICAL_STRUCTURE": measurements = _mechanical(product)
        elif domain == "MESH_QUALITY": measurements = _mesh(product)
        elif domain == "DIRECTIONAL_VISUAL": measurements = _visual(product)
        elif domain == "MOTION": measurements = _motion(product, motion_probe_policy)
        elif domain == "RUNTIME_CONSUMPTION": measurements = _runtime(product, measurements_by_domain.get("MOTION", {}))
        elif domain == "DEFORMATION":
            if deformation_fixture is None: measurements = {"status": "MISSING_FIXTURE"}
            else:
                component = product.directional_renderables.directions[int(deformation_fixture.get("view_index", 0))].components[int(deformation_fixture.get("component_index", 0))]
                measurements = verified_mesh_lbs_measurement(component.mesh, component.mesh_skin, product.mechanical_state.skeleton, deformation_fixture["transforms"], deformation_fixture["expected"])
                measurements.update({"rms_threshold": float(deformation_fixture.get("rms_threshold", 1e-6)), "p95_threshold": float(deformation_fixture.get("p95_threshold", 1e-6))})

        measurements_by_domain[domain] = measurements
        if domain == "DEFORMATION" and deformation_fixture is None: status = "ABSTAIN"
        elif domain == "RUNTIME_CONSUMPTION" and str(measurements.get("proof_owned_runtime_bake_status") or "").startswith("ABSTAIN_"): status = "ABSTAIN"
        else: status = _status(domain, measurements)
        failures = derive_failure_signatures(domain, measurements, status=status)
        measurement_report = bind_measurement_report(product, plan, measurements=measurements)
        measurement_reports.append(measurement_report)
        reports.append(bind_domain_proof(
            product, plan, measurement_report, status=status,
            failure_signatures=failures, owner_attribution=no_owner_attribution(),
            metadata={"causal_mutation_gate":"SOURCE_TEST_REQUIRED","diagnostic_service":"RealSaS.CompilerServices.ProofFailureSignatures.v1","causal_owner_attribution":"NOT_PERFORMED","owner_attribution_requires_controlled_fault_experiment":True,"dynamic_authored_motion_probe":domain == "MOTION"},
        ))
    bundle = bind_product_proof_bundle(product, tuple(reports), metadata={"engine":"RealSaS.ProofEngine.v3","heavy_solver_promoted":False,"authored_motion_dynamic_probe_promoted":True,"authored_motion_probe_service":AUTHORED_MOTION_PROBE_SERVICE_ID,"measurement_reports_externalized":True})
    return ProductProofEvaluationV1(bundle, tuple(measurement_reports))


def evaluate_product_proof(product, *, deformation_fixture: dict | None = None, motion_probe_policy: AuthoredMotionProbePolicyV1 = AuthoredMotionProbePolicyV1()):
    """Backward-compatible bundle-only facade over evidence-preserving evaluation."""
    return evaluate_product_proof_evidence(product, deformation_fixture=deformation_fixture, motion_probe_policy=motion_probe_policy).proof_bundle


def mutation_worsens_measurement(domain: str, baseline: dict, mutated: dict) -> bool:
    if domain == "MESH_QUALITY": return mutated.get("degenerate_faces", 0) > baseline.get("degenerate_faces", 0) or mutated.get("min_area", 0) < baseline.get("min_area", 0)
    if domain == "DEFORMATION": return mutated.get("rms", 0) > baseline.get("rms", 0) or mutated.get("p95", 0) > baseline.get("p95", 0)
    if domain == "DIRECTIONAL_VISUAL": return mutated.get("direction_count", 8) < baseline.get("direction_count", 8) or mutated.get("corner_binding_count", 0) < baseline.get("corner_binding_count", 0)
    if domain == "MOTION":
        try: base_pass = authored_motion_measurement_passes_v1(baseline); mut_pass = authored_motion_measurement_passes_v1(mutated)
        except ValueError: return True
        return ((base_pass and not mut_pass) or mutated.get("max_edge_relative_change",0)>baseline.get("max_edge_relative_change",0) or mutated.get("min_triangle_area_ratio",1)<baseline.get("min_triangle_area_ratio",1) or mutated.get("max_loop_seam_normalized",0)>baseline.get("max_loop_seam_normalized",0) or mutated.get("effective_clip_count",0)<baseline.get("effective_clip_count",0))
    if domain == "MECHANICAL_STRUCTURE": return mutated.get("illegal_parent_count",0)>baseline.get("illegal_parent_count",0) or mutated.get("deform_root_count",0)<baseline.get("deform_root_count",0) or mutated.get("unsupported_joint_count",0)>baseline.get("unsupported_joint_count",0)
    if domain == "RUNTIME_CONSUMPTION": return mutated.get("representation_class") != baseline.get("representation_class") or bool(mutated.get("full_3d_reconstruction_authority"))
    raise ValueError(domain)
