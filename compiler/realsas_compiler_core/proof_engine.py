from __future__ import annotations

from compiler.realsas_compiler_services.proof.failure_signatures import (
    derive_failure_signatures,
    no_owner_attribution,
)
from compiler.realsas_compiler_services.proof.motion_bake import assert_motion_bake_binding
from compiler.realsas_compiler_services.proof.motion_frame_metrics import (
    RESTORED_V05_POLICY_V1,
    evaluate_motion_bake_metrics,
    measure_motion_bake_geometry,
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


def _motion(product, plan, *, motion_bake_provider=None, motion_policy=None):
    validate_motion_against_mechanical(product.motion_state, product.mechanical_state)
    base = {
        "clip_count": len(product.motion_state.clips),
        "joint_track_count": len(product.motion_state.joint_tracks),
        "effective_joint_track_count": effective_motion_track_count(product.motion_state),
    }
    required_clips = tuple(
        clip for clip in product.motion_state.clips
        if clip.clip_kind == "PRESET" or "PRESET_MOTION" in set(clip.required_capabilities)
    )
    base["required_dynamic_clip_ids"] = [clip.clip_id for clip in required_clips]
    if motion_bake_provider is None:
        return {
            **base,
            "status": "MISSING_QUALIFICATION_OWNED_BAKE",
            "dynamic_frame_evidence_status": "MISSING_QUALIFICATION_OWNED_BAKE",
            "dynamic_motion_passed": False,
            "qualified_bake_count": 0,
            "clip_measurements": [],
        }, {}

    bakes = {}
    rows = []
    for clip in required_clips:
        bake = motion_bake_provider(product, plan, clip)
        if bake is None:
            rows.append({"clip_id": clip.clip_id, "passed": False, "failure_invariants": ["missing_qualification_owned_bake"]})
            continue
        assert_motion_bake_binding(
            bake,
            source_product_state_hash=product.product_state_hash,
            proof_plan_hash=plan.proof_plan_hash,
        )
        if bake.clip_id != clip.clip_id:
            raise QualificationError("MOTION_BAKE_CLIP_ID_MISMATCH")
        metrics = evaluate_motion_bake_metrics(measure_motion_bake_geometry(bake), policy=motion_policy)
        rows.append(metrics)
        bakes[clip.clip_id] = bake

    measured = bool(required_clips) and len(bakes) == len(required_clips)
    passed = measured and all(bool(row.get("passed", False)) for row in rows)
    return {
        **base,
        "status": "MEASURED" if measured else "INCOMPLETE_QUALIFICATION_OWNED_BAKE",
        "dynamic_frame_evidence_status": "MEASURED" if measured else "INCOMPLETE_QUALIFICATION_OWNED_BAKE",
        "dynamic_motion_passed": bool(passed),
        "qualified_bake_count": len(bakes),
        "clip_measurements": rows,
        "max_motion01": max((float(row.get("max_motion01", 0.0)) for row in rows), default=0.0),
        "max_edge_stretch_ratio": max((float(row.get("max_edge_stretch_ratio", 1.0)) for row in rows), default=1.0),
        "max_area_change_ratio": max((float(row.get("max_area_change_ratio", 1.0)) for row in rows), default=1.0),
        "flipped_triangles": sum(int(row.get("flipped_triangles", 0)) for row in rows),
        "max_loop_seam_error01": max((float(row.get("loop_seam_error01", 0.0)) for row in rows), default=0.0),
        "return_to_rest_error01": max((float(row.get("return_to_rest_error01", 0.0)) for row in rows), default=0.0),
        "max_return_to_rest_error01": max((float(row.get("return_to_rest_error01", 0.0)) for row in rows), default=0.0),
        "frozen_policy_thresholds": {**RESTORED_V05_POLICY_V1, **dict(motion_policy or {})},
    }, bakes


def _runtime(product):
    return {
        "representation_class": product.representation_class,
        "full_3d_reconstruction_authority": bool(product.full_3d_reconstruction_authority),
        "direction_count": len(product.directional_renderables.directions),
        "motion_representation": product.motion_state.representation_class,
    }


def _status(domain, m):
    if domain == "MECHANICAL_STRUCTURE":
        ok = m["joint_count"] > 0 and m["deform_root_count"] > 0 and m["illegal_parent_count"] == 0 and m["unsupported_joint_count"] == 0
    elif domain == "MESH_QUALITY":
        ok = m["component_count"] >= 8 and m["face_count"] > 0 and m["degenerate_faces"] == 0 and m["min_area"] > 1e-12
    elif domain == "DIRECTIONAL_VISUAL":
        ok = m["direction_count"] == 8 and m["view_order"] == list(range(8)) and m["corner_binding_count"] > 0
    elif domain == "MOTION":
        ok = (
            m["clip_count"] > 0
            and m["effective_joint_track_count"] > 0
            and m.get("dynamic_frame_evidence_status") == "MEASURED"
            and bool(m.get("dynamic_motion_passed", False))
        )
    elif domain == "RUNTIME_CONSUMPTION":
        ok = m["representation_class"] == "DIRECTIONAL_2D_2P5D_PUPPET" and not m["full_3d_reconstruction_authority"] and m["direction_count"] == 8 and m["motion_representation"] == "DIRECTIONAL_2D_2P5D_PUPPET_MOTION"
    elif domain == "DEFORMATION":
        ok = m.get("rms", float("inf")) <= float(m.get("rms_threshold", 1e-6)) and m.get("p95", float("inf")) <= float(m.get("p95_threshold", 1e-6))
    else:
        raise ValueError(f"unknown proof domain:{domain}")
    return "PASS" if ok else "FAIL"


def evaluate_product_proof(product, *, deformation_fixture: dict | None = None, motion_bake_provider=None, motion_policy: dict | None = None, artifacts_out: dict | None = None):
    """Evaluate current proof domains without manufacturing directional frames.

    MOTION can PASS only when a separately qualified evaluator supplies a bake
    bound to this exact product state and this exact proof plan. Missing frame
    evidence is ABSTAIN. Direct mechanical-joint/P.xy evaluation is forbidden.
    """
    validate_product_ontology(product)
    required = set(required_proof_domains(product.capability_contract))
    unknown = required - _REQUIRED
    if unknown:
        raise QualificationError(f"UNIMPLEMENTED_REQUIRED_PROOF_DOMAIN:{sorted(unknown)}")

    reports = []
    for domain in sorted(required):
        domain_motion_bakes = {}
        plan = bind_proof_plan(product, proof_domain=domain, operator_policy_hashes=(f"RealSaS.ProofEngine.{domain}.v1",), probe_specification={"deterministic": True, "domain": domain})
        if domain == "MECHANICAL_STRUCTURE": measurements = _mechanical(product)
        elif domain == "MESH_QUALITY": measurements = _mesh(product)
        elif domain == "DIRECTIONAL_VISUAL": measurements = _visual(product)
        elif domain == "MOTION":
            measurements, domain_motion_bakes = _motion(product, plan, motion_bake_provider=motion_bake_provider, motion_policy=motion_policy)
            if artifacts_out is not None:
                artifacts_out.setdefault("motion_bakes", {}).update(domain_motion_bakes)
        elif domain == "RUNTIME_CONSUMPTION": measurements = _runtime(product)
        elif domain == "DEFORMATION":
            if deformation_fixture is None:
                measurements = {"status": "MISSING_FIXTURE"}
            else:
                component = product.directional_renderables.directions[int(deformation_fixture.get("view_index", 0))].components[int(deformation_fixture.get("component_index", 0))]
                measurements = verified_mesh_lbs_measurement(component.mesh, component.mesh_skin, product.mechanical_state.skeleton, deformation_fixture["transforms"], deformation_fixture["expected"])
                measurements.update({"rms_threshold": float(deformation_fixture.get("rms_threshold", 1e-6)), "p95_threshold": float(deformation_fixture.get("p95_threshold", 1e-6))})

        status = "ABSTAIN" if (domain == "DEFORMATION" and deformation_fixture is None) or (domain == "MOTION" and measurements.get("dynamic_frame_evidence_status") != "MEASURED") else _status(domain, measurements)
        failures = derive_failure_signatures(domain, measurements, status=status)
        measurement_report = bind_measurement_report(product, plan, measurements=measurements)
        metadata = {
            "causal_mutation_gate": "SOURCE_TEST_REQUIRED",
            "diagnostic_service": "RealSaS.CompilerServices.ProofFailureSignatures.v1",
            "causal_owner_attribution": "NOT_PERFORMED",
            "owner_attribution_requires_controlled_fault_experiment": True,
        }
        if domain == "MOTION":
            metadata.update({
                "qualification_owned_bake_hashes": {clip_id: bake.bake_hash for clip_id, bake in sorted(domain_motion_bakes.items())},
                "qualification_owned_bake_hash": next(iter(domain_motion_bakes.values())).bake_hash if len(domain_motion_bakes) == 1 else "",
                "export_solver_replay_forbidden": True,
                "dynamic_frame_proof_required": True,
                "directional_joint_view_binding_required": True,
            })
        reports.append(bind_domain_proof(product, plan, measurement_report, status=status, failure_signatures=failures, owner_attribution=no_owner_attribution(), metadata=metadata))
    return bind_product_proof_bundle(product, tuple(reports), metadata={"engine": "RealSaS.ProofEngine.v3.fail_closed_motion_bake", "heavy_solver_promoted": False})


def mutation_worsens_measurement(domain: str, baseline: dict, mutated: dict) -> bool:
    if domain == "MESH_QUALITY": return mutated.get("degenerate_faces", 0) > baseline.get("degenerate_faces", 0) or mutated.get("min_area", 0) < baseline.get("min_area", 0)
    if domain == "DEFORMATION": return mutated.get("rms", 0) > baseline.get("rms", 0) or mutated.get("p95", 0) > baseline.get("p95", 0)
    if domain == "DIRECTIONAL_VISUAL": return mutated.get("direction_count", 8) < baseline.get("direction_count", 8) or mutated.get("corner_binding_count", 0) < baseline.get("corner_binding_count", 0)
    if domain == "MOTION": return mutated.get("effective_joint_track_count", 0) < baseline.get("effective_joint_track_count", 0)
    if domain == "MECHANICAL_STRUCTURE": return mutated.get("illegal_parent_count", 0) > baseline.get("illegal_parent_count", 0) or mutated.get("deform_root_count", 0) < baseline.get("deform_root_count", 0) or mutated.get("unsupported_joint_count", 0) > baseline.get("unsupported_joint_count", 0)
    if domain == "RUNTIME_CONSUMPTION": return mutated.get("representation_class") != baseline.get("representation_class") or bool(mutated.get("full_3d_reconstruction_authority"))
    raise ValueError(domain)
