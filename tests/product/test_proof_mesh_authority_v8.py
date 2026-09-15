from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core import proof_engine as v7
from compiler.realsas_compiler_core.mesh_binding import mesh_lineage_hash
from compiler.realsas_compiler_core.motion import build_deterministic_preset_motion
from compiler.realsas_compiler_core.product_external_render import (
    assemble_product_v3_with_external_render_support,
    build_external_directional_renderable,
    build_external_directional_renderable_set,
    build_external_renderable_component,
)
from compiler.realsas_compiler_core.proof_mesh_authority_v8 import (
    _CANONICAL_SUPPORT_AREA_AUTHORITY,
    _EXTERNAL_RASTER_AUTHORITY,
    _authoritative_area_summary,
    measure_mesh_quality,
    rebind_failed_mesh_domain,
)
from compiler.realsas_compiler_core.v4 import (
    bind_domain_proof,
    bind_measurement_report,
    bind_product_proof_bundle,
    bind_proof_plan,
    build_capability_contract,
)
from compiler.realsas_compiler_core.v4_types import CapabilityRequirement
from tests.product.test_external_render_support_v1 import (
    _appearance,
    _mechanical,
    _p1_mesh,
    _p1_skin,
)


def _collapsed_support_mesh(view: int, *, raster_degenerate: bool = False):
    """External render triangle valid in raster space but collapsed in support P."""
    mesh = _p1_mesh(view)
    vertices = list(mesh.vertices)
    third = vertices[2]
    metadata = dict(third.metadata)
    if raster_degenerate:
        metadata["raster_xy"] = (15.0, 10.0)
    vertices[2] = replace(third, P=(0.15, 0.0, 0.0), metadata=metadata)
    mesh = replace(mesh, vertices=tuple(vertices), mesh_lineage_hash="")
    return replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))


def _external_mesh_quality_product(*, raster_degenerate: bool = False):
    mechanical = _mechanical()
    directions = []
    for view in range(8):
        mesh = _collapsed_support_mesh(view, raster_degenerate=raster_degenerate)
        mesh_skin = _p1_skin(mechanical, mesh)
        component = build_external_renderable_component(
            component_id="BODY",
            view_index=view,
            mesh=mesh,
            mesh_skin=mesh_skin,
            mechanical=mechanical,
            appearance=_appearance(mesh),
            setup_order=0,
            coverage_classification="P1_FULL_SUBJECT",
            materialization_manifest_sha256="sha256:v8-raster-authority-fixture",
            direct_binding_manifest_sha256="sha256:v8-direct-binding-fixture",
        )
        directions.append(build_external_directional_renderable(
            view_index=view,
            camera_binding_hash=mesh.camera_binding_hash,
            components=(component,),
            mechanical=mechanical,
        ))
    render_set = build_external_directional_renderable_set(tuple(directions), mechanical)
    contract = build_capability_contract(
        "V8_MESH_AUTHORITY_TEST",
        (
            CapabilityRequirement(
                "VISUAL_8_DIRECTION",
                "REQUIRED",
                "visual",
                "policy",
                ("MESH_QUALITY",),
            ),
        ),
    )
    motion = build_deterministic_preset_motion(mechanical)
    return assemble_product_v3_with_external_render_support(
        mechanical,
        render_set,
        contract,
        motion,
    )


def test_external_render_support_uses_raster_area_for_product_nondegeneracy():
    summary = _authoritative_area_summary(
        support_area={"face_count": 4, "degenerate_faces": 4, "min_area": 0.0},
        raster={"face_count": 4, "degenerate_faces": 0, "min_raster_triangle_area": 2.5},
        external=True,
    )
    assert summary == {
        "face_count": 4,
        "degenerate_faces": 0,
        "min_area": 2.5,
        "mesh_area_authority": _EXTERNAL_RASTER_AUTHORITY,
    }


def test_full_external_product_accepts_p_collapsed_but_raster_valid_mesh():
    product = _external_mesh_quality_product()
    report = measure_mesh_quality(product)

    assert report["component_count"] == 8
    assert report["external_render_support_component_count"] == 8
    assert report["mesh_area_authority"] == _EXTERNAL_RASTER_AUTHORITY
    assert report["support_space_area_is_diagnostic"] is True
    assert report["support_space_degenerate_faces"] == 8
    assert report["support_space_min_area"] == 0.0
    assert report["degenerate_faces"] == 0
    assert report["min_area"] > 1.0e-12
    assert all(
        row["mesh_area_authority"] == _EXTERNAL_RASTER_AUTHORITY
        and row["support_space_degenerate_faces"] == 1
        and row["degenerate_faces"] == 0
        for row in report["component_reports"]
    )
    assert v7._status("MESH_QUALITY", report) == "PASS"


def test_full_external_product_still_fails_when_raster_triangle_is_degenerate():
    product = _external_mesh_quality_product(raster_degenerate=True)
    report = measure_mesh_quality(product)

    assert report["support_space_degenerate_faces"] == 8
    assert report["degenerate_faces"] == 8
    assert report["min_area"] == 0.0
    assert v7._status("MESH_QUALITY", report) == "FAIL"


def test_mesh_only_rebind_changes_only_required_failing_mesh_domain_to_pass():
    product = _external_mesh_quality_product()
    legacy_measurements = dict(measure_mesh_quality(product))
    legacy_measurements["degenerate_faces"] = 8
    legacy_measurements["min_area"] = 0.0
    legacy_measurements["mesh_area_authority"] = _CANONICAL_SUPPORT_AREA_AUTHORITY

    plan = bind_proof_plan(
        product,
        proof_domain="MESH_QUALITY",
        operator_policy_hashes=("RealSaS.ProofEngine.MESH_QUALITY.legacy_fixture",),
        probe_specification={"deterministic": True, "domain": "MESH_QUALITY"},
    )
    measurement = bind_measurement_report(product, plan, measurements=legacy_measurements)
    failing = bind_domain_proof(
        product,
        plan,
        measurement,
        status="FAIL",
        metadata={"fixture": "legacy_support_space_area_authority"},
    )
    old_bundle = bind_product_proof_bundle(
        product,
        (failing,),
        metadata={"post_export_native_interlock_required": True},
    )
    assert old_bundle.overall_status == "FAIL"

    rebound = rebind_failed_mesh_domain(product, old_bundle)
    assert rebound.overall_status == "PASS"
    assert rebound.metadata["source_proof_bundle_hash"] == old_bundle.proof_bundle_hash
    assert rebound.metadata["mesh_only_rebind"] is True
    assert rebound.metadata["motion_bakes_recomputed"] is False
    assert rebound.metadata["product_state_rebuilt"] is False
    assert rebound.metadata["threshold_relaxation_used"] is False
    assert len(rebound.domain_reports) == 1
    assert rebound.domain_reports[0].proof_domain == "MESH_QUALITY"
    assert rebound.domain_reports[0].status == "PASS"
    assert rebound.domain_reports[0].metadata["mesh_area_authority"] == _EXTERNAL_RASTER_AUTHORITY


def test_canonical_mesh_keeps_support_space_area_authority():
    summary = _authoritative_area_summary(
        support_area={"face_count": 2, "degenerate_faces": 1, "min_area": 0.0},
        raster={"face_count": 2, "degenerate_faces": 0, "min_raster_triangle_area": 10.0},
        external=False,
    )
    assert summary == {
        "face_count": 2,
        "degenerate_faces": 1,
        "min_area": 0.0,
        "mesh_area_authority": _CANONICAL_SUPPORT_AREA_AUTHORITY,
    }
