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
from compiler.realsas_compiler_services.proof.directional_motion_provider import (
    QualifiedDirectionalMotionBakeProviderV1,
)

from .deformation import (
    effective_motion_track_count,
    mesh_triangle_area_report,
    verified_mesh_lbs_measurement,
)
from .mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
    raster_quality_gate_failures,
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
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    rows = []
    area_reports = []
    raster_reports = []
    cdt_rows = []
    for direction in product.directional_renderables.directions:
        for component in direction.components:
            area = mesh_triangle_area_report(component.mesh)
            raster = mesh_raster_quality_report(
                component.mesh,
                surface=product.mechanical_state.surface,
                view_index=int(direction.view_index),
            )
            area_reports.append(area)
            raster_reports.append(raster)
            qreport = dict(component.mesh.qualification_report or {})
            coverage = dict(qreport.get("candidate_residual_report") or {})
            row = {
                "view_index": int(direction.view_index),
                "component_id": str(component.component_id),
                "coverage_classification": str(component.mesh.support_coverage_classification),
                **area,
                **raster,
            }
            if component.mesh.support_coverage_classification == "OBSERVATION_DOMAIN_CDT":
                # Older qualifier revisions also copy recall/precision to the top level;
                # candidate_residual_report is the exact complete coverage envelope.
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
        "face_count": sum(r["face_count"] for r in area_reports),
        "degenerate_faces": sum(r["degenerate_faces"] for r in area_reports),
        "min_area": min((r["min_area"] for r in area_reports), default=0.0),
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
            (float(r.get("source_alpha_recall", 1.0)) for r in cdt_rows),
            default=1.0,
        ),
        "min_precision_inside_alpha": min(
            (float(r.get("precision_inside_alpha", 1.0)) for r in cdt_rows),
            default=1.0,
        ),
        "min_alpha_iou": min(
            (float(r.get("alpha_iou", 1.0)) for r in cdt_rows),
            default=1.0,
        ),
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

    if not isinstance(motion_bake_provider, QualifiedDirectionalMotionBakeProviderV1):
        raise QualificationError("MOTION_BAKE_PROVIDER_NOT_QUALIFIED")
    motion_bake_provider.assert_for_product(product)
    base.update({
        "qualified_motion_provider_hash": motion_bake_provider.provider_hash,
        "directional_binding_set_hash": motion_bake_provider.directional_binding_set_hash,
        "directional_evaluator_policy_hash": motion_bake_provider.evaluator_policy_hash,
        "directional_evaluator_semantic_version": motion_bake_provider.evaluator_semantic_version,
    })

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
        if bake.evaluator_semantic_version != motion_bake_provider.evaluator_semantic_version:
            raise QualificationError("MOTION_BAKE_EVALUATOR_SEMANTIC_MISMATCH")
        bake_metadata = dict(bake.metadata or {})
        if bake_metadata.get("directional_binding_set_hash") != motion_bake_provider.directional_binding_set_hash:
            raise QualificationError("MOTION_BAKE_DIRECTIONAL_BINDING_MISMATCH")
        if bake_metadata.get("evaluator_policy_hash") != motion_bake_provider.evaluator_policy_hash:
            raise QualificationError("MOTION_BAKE_EVALUATOR_POLICY_MISMATCH")
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
    runtime_requirements = tuple(
        requirement for requirement in product.capability_contract.requirements
        if requirement.activation == "REQUIRED" and "RUNTIME_CONSUMPTION" in set(requirement.required_proof_domains)
    )
    implementation_hashes = tuple(sorted({str(row.implementation_binding_hash) for row in runtime_requirements if row.implementation_binding_hash}))
    return {
        "proof_scope": "PRE_EXPORT_RUNTIME_CONTRACT_COMPATIBILITY_ONLY",
        "runtime_contract_compatibility_status": "MEASURED_PRE_EXPORT",
        "native_package_execution_performed": False,
        "post_export_native_interlock_required": True,
        "sealed_native_runtime_consumer_required": True,
        "runtime_requirement_count": len(runtime_requirements),
        "runtime_implementation_binding_hashes": implementation_hashes,
        "representation_class": product.representation_class,
        "full_3d_reconstruction_authority": bool(product.full_3d_reconstruction_authority),
        "direction_count": len(product.directional_renderables.directions),
        "motion_representation": product.motion_state.representation_class,
    }


def _status(domain, m):
    if domain == "MECHANICAL_STRUCTURE":
        ok = m["joint_count"] > 0 and m["deform_root_count"] > 0 and m["illegal_parent_count"] == 0 and m["unsupported_joint_count"] == 0
    elif domain == "MESH_QUALITY":
        policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
        base_ok = (
            m["component_count"] >= 8
            and m["face_count"] > 0
            and m["degenerate_faces"] == 0
            and m["min_area"] > 1e-12
            and m["duplicate_faces"] <= policy.max_duplicate_faces
            and m["nonmanifold_edges"] <= policy.max_nonmanifold_edges
            and m["min_raster_triangle_angle_deg"] >= policy.min_raster_triangle_angle_deg
            and m["max_raster_triangle_aspect_ratio"] <= policy.max_raster_triangle_aspect_ratio
        )
        cdt_ok = (
            int(m.get("strict_cdt_component_count", 0)) == 0
            or int(m.get("strict_cdt_pass_count", 0)) == int(m.get("strict_cdt_component_count", 0))
        )
        ok = base_ok and cdt_ok
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
        ok = (
            m.get("proof_scope") == "PRE_EXPORT_RUNTIME_CONTRACT_COMPATIBILITY_ONLY"
            and m.get("runtime_contract_compatibility_status") == "MEASURED_PRE_EXPORT"
            and m.get("native_package_execution_performed") is False
            and m.get("post_export_native_interlock_required") is True
            and m.get("sealed_native_runtime_consumer_required") is True
            and int(m.get("runtime_requirement_count", 0)) > 0
            and bool(m.get("runtime_implementation_binding_hashes"))
            and m["representation_class"] == "DIRECTIONAL_2D_2P5D_PUPPET"
            and not m["full_3d_reconstruction_authority"]
            and m["direction_count"] == 8
            and m["motion_representation"] == "DIRECTIONAL_2D_2P5D_PUPPET_MOTION"
        )
    elif domain == "DEFORMATION":
        ok = m.get("rms", float("inf")) <= float(m.get("rms_threshold", 1e-6)) and m.get("p95", float("inf")) <= float(m.get("p95_threshold", 1e-6))
    else:
        raise ValueError(f"unknown proof domain:{domain}")
    return "PASS" if ok else "FAIL"


def evaluate_product_proof(product, *, deformation_fixture: dict | None = None, motion_bake_provider=None, motion_policy: dict | None = None, artifacts_out: dict | None = None):
    """Evaluate current product proof without manufacturing runtime evidence.

    MOTION can PASS only when a typed QualifiedDirectionalMotionBakeProviderV1
    supplies qualification-owned frames bound to this exact product, directional
    joint/view binding, evaluator policy and proof plan. Missing evidence is
    ABSTAIN; arbitrary callables are rejected. Direct mechanical-joint/P.xy
    evaluation is forbidden.

    MESH_QUALITY now measures the exact raster mesh, including alpha-domain coverage
    for ObservationDomainCDT components. Aggregate face existence can no longer hide
    a large untriangulated visible region.

    The historical domain name RUNTIME_CONSUMPTION is retained for contract
    compatibility, but this pre-export proof measures runtime-contract compatibility
    only. It explicitly records native_package_execution_performed=False. Actual
    .rss -> sealed C++ open/sample/render is a separate post-export interlock gate,
    avoiding circular proof<->export authority.
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
        if domain == "MESH_QUALITY":
            metadata.update({
                "alpha_domain_coverage_required_for_observation_cdt": True,
                "large_uncovered_region_gate_required": True,
                "mesh_quality_policy": FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.to_dict(),
            })
        elif domain == "MOTION":
            metadata.update({
                "qualification_owned_bake_hashes": {clip_id: bake.bake_hash for clip_id, bake in sorted(domain_motion_bakes.items())},
                "qualification_owned_bake_hash": next(iter(domain_motion_bakes.values())).bake_hash if len(domain_motion_bakes) == 1 else "",
                "qualified_motion_provider_hash": measurements.get("qualified_motion_provider_hash", ""),
                "directional_binding_set_hash": measurements.get("directional_binding_set_hash", ""),
                "directional_evaluator_policy_hash": measurements.get("directional_evaluator_policy_hash", ""),
                "directional_evaluator_semantic_version": measurements.get("directional_evaluator_semantic_version", ""),
                "export_solver_replay_forbidden": True,
                "dynamic_frame_proof_required": True,
                "directional_joint_view_binding_required": True,
                "arbitrary_motion_provider_callable_forbidden": True,
            })
        elif domain == "RUNTIME_CONSUMPTION":
            metadata.update({
                "proof_scope": "PRE_EXPORT_RUNTIME_CONTRACT_COMPATIBILITY_ONLY",
                "native_package_execution_performed": False,
                "post_export_native_interlock_required": True,
                "post_export_native_gate": "RSS_TO_SEALED_CPP_OPEN_SAMPLE_RENDER",
                "circular_proof_export_dependency_forbidden": True,
            })
        reports.append(bind_domain_proof(product, plan, measurement_report, status=status, failure_signatures=failures, owner_attribution=no_owner_attribution(), metadata=metadata))
    return bind_product_proof_bundle(product, tuple(reports), metadata={
        "engine": "RealSaS.ProofEngine.v6.mesh_coverage_typed_motion_preexport_runtime_compatibility",
        "heavy_solver_promoted": False,
        "post_export_native_interlock_required": "RUNTIME_CONSUMPTION" in required,
    })


def mutation_worsens_measurement(domain: str, baseline: dict, mutated: dict) -> bool:
    if domain == "MESH_QUALITY":
        return (
            mutated.get("degenerate_faces", 0) > baseline.get("degenerate_faces", 0)
            or mutated.get("min_area", 0) < baseline.get("min_area", 0)
            or mutated.get("nonmanifold_edges", 0) > baseline.get("nonmanifold_edges", 0)
            or mutated.get("duplicate_faces", 0) > baseline.get("duplicate_faces", 0)
            or mutated.get("min_source_alpha_recall", 1.0) < baseline.get("min_source_alpha_recall", 1.0)
            or mutated.get("max_largest_uncovered_component_fraction", 0.0) > baseline.get("max_largest_uncovered_component_fraction", 0.0)
        )
    if domain == "DEFORMATION": return mutated.get("rms", 0) > baseline.get("rms", 0) or mutated.get("p95", 0) > baseline.get("p95", 0)
    if domain == "DIRECTIONAL_VISUAL": return mutated.get("direction_count", 8) < baseline.get("direction_count", 8) or mutated.get("corner_binding_count", 0) < baseline.get("corner_binding_count", 0)
    if domain == "MOTION": return mutated.get("effective_joint_track_count", 0) < baseline.get("effective_joint_track_count", 0)
    if domain == "MECHANICAL_STRUCTURE": return mutated.get("illegal_parent_count", 0) > baseline.get("illegal_parent_count", 0) or mutated.get("deform_root_count", 0) < baseline.get("deform_root_count", 0) or mutated.get("unsupported_joint_count", 0) > baseline.get("unsupported_joint_count", 0)
    if domain == "RUNTIME_CONSUMPTION": return (
        mutated.get("representation_class") != baseline.get("representation_class")
        or bool(mutated.get("full_3d_reconstruction_authority"))
        or mutated.get("runtime_contract_compatibility_status") != baseline.get("runtime_contract_compatibility_status")
        or mutated.get("runtime_implementation_binding_hashes") != baseline.get("runtime_implementation_binding_hashes")
    )
    raise ValueError(domain)
