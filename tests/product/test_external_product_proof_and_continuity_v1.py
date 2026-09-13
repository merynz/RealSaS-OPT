from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.component_attachment import (
    ComponentAttachmentEvidenceIR,
    qualify_component_assembly,
    qualify_component_attachment,
)
from compiler.realsas_compiler_core.continuity_underlay import (
    assemble_product_v3_with_continuity_underlay,
    bind_continuity_underlay_set_to_directional_renderables,
    build_continuity_underlay_proof,
    build_continuity_underlay_set,
    qualify_continuity_raster_measurement,
    qualify_continuity_underlay,
)
from compiler.realsas_compiler_core.mesh.direct_model_skin import DIRECT_MODEL_TRANSFER_METHOD
from compiler.realsas_compiler_core.mesh_binding import mesh_lineage_hash, mesh_skin_lineage_hash
from compiler.realsas_compiler_core.motion import build_deterministic_preset_motion
from compiler.realsas_compiler_core.product_external_render import (
    assemble_product_v3_with_external_render_support,
    build_external_directional_renderable,
    build_external_directional_renderable_set,
    build_external_renderable_component,
)
from compiler.realsas_compiler_core.proof_engine import _mesh, _visual
from compiler.realsas_compiler_core.types import (
    QualificationError,
    QualifiedEditableMeshIR,
    QualifiedJoint,
    QualifiedMeshSkinIR,
    QualifiedMeshSkinRow,
    QualifiedMeshVertex,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceSupportBinding,
)
from compiler.realsas_compiler_core.v4 import (
    build_appearance_binding,
    build_capability_contract,
    build_mechanical_state,
    qualified_skeleton_v2_lineage_hash,
)
from compiler.realsas_compiler_core.v4_types import (
    AppearanceCornerBinding,
    CapabilityRequirement,
    QualifiedSkeletonIRV2,
)


def _mechanical():
    nodes = (
        SurfaceNode(
            "FIT1:S0", (0.0, 0.0, 0.0), (0,), ("FIT1",), (),
            raster_bindings=((0, (10.0, 10.0)),),
        ),
        SurfaceNode(
            "FIT1:S1", (0.3, 0.0, 0.0), (0,), ("FIT1",), (),
            raster_bindings=((0, (20.0, 10.0)),),
        ),
    )
    surface = RiggingSurfaceIR(nodes, geometry_lineage_hash="FIT1:SCIENTIFIC:S")
    joints = (
        QualifiedJoint("J:ROOT", (0.0, 0.0, 0.0), None),
        QualifiedJoint("J:ARM", (0.3, 0.0, 0.0), "J:ROOT"),
    )
    skeleton = QualifiedSkeletonIRV2(joints, ("J:ROOT",), {}, {"status": "PASS"}, "")
    skeleton = replace(skeleton, skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(skeleton))
    skin = QualifiedSkinIR(
        (
            QualifiedSkinRow("FIT1:S0", (("J:ROOT", 1.0),), 0.0, 0.0),
            QualifiedSkinRow("FIT1:S1", (("J:ARM", 1.0),), 0.0, 0.0),
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        {"status": "PASS"},
        "FIT1:SOURCE:SKIN",
    )
    return build_mechanical_state(surface, skeleton, skin)


def _p1_mesh(view: int):
    verts = (
        QualifiedMeshVertex(
            "MV:0", (0.0, 0.0, 0.0),
            SurfaceSupportBinding("LOCAL_CONVEX_INTERPOLATION", (("P1:S0", 1.0),)),
            metadata={"raster_xy": (10.0, 10.0)},
        ),
        QualifiedMeshVertex(
            "MV:1", (0.3, 0.0, 0.0),
            SurfaceSupportBinding("LOCAL_CONVEX_INTERPOLATION", (("P1:S1", 1.0),)),
            metadata={"raster_xy": (20.0, 10.0)},
        ),
        QualifiedMeshVertex(
            "MV:2", (0.15, 0.0, 0.3),
            SurfaceSupportBinding("LOCAL_CONVEX_INTERPOLATION", (("P1:S2", 1.0),)),
            metadata={"raster_xy": (15.0, 20.0)},
        ),
    )
    mesh = QualifiedEditableMeshIR(
        verts,
        (("MV:0", "MV:1", "MV:2"),),
        (("MV:0", "MV:1"), ("MV:1", "MV:2"), ("MV:2", "MV:0")),
        "P1:RENDER:SUPPORT:S",
        view,
        f"CAM:{view}",
        {"status": "PASS_EXTERNAL_SUPPORT"},
        "",
        support_coverage_classification="P1_FULL_SUBJECT",
    )
    return replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))


def _p1_skin(mechanical, mesh):
    rows = (
        QualifiedMeshSkinRow("MV:0", (("J:ROOT", 1.0),), (("P1:S0", 1.0),), 0.0, 0.0),
        QualifiedMeshSkinRow("MV:1", (("J:ARM", 1.0),), (("P1:S1", 1.0),), 0.0, 0.0),
        QualifiedMeshSkinRow(
            "MV:2", (("J:ROOT", 0.5), ("J:ARM", 0.5)),
            (("P1:S2", 1.0),), 0.0, 0.0,
        ),
    )
    value = QualifiedMeshSkinIR(
        rows,
        mesh.surface_binding_hash,
        mechanical.skeleton.skeleton_lineage_hash,
        mechanical.skin.skin_lineage_hash,
        mesh.mesh_lineage_hash,
        DIRECT_MODEL_TRANSFER_METHOD,
        {"status": "PASS_DIRECT_MODEL_EXACT_MESH_SKIN_QUALIFICATION"},
        "",
        metadata={"direct_model_query": True, "historical_weight_transfer_used": False},
    )
    return replace(value, mesh_skin_lineage_hash=mesh_skin_lineage_hash(value))


def _appearance(mesh):
    by_id = {vertex.canonical_mesh_vertex_id: vertex for vertex in mesh.vertices}
    corners = []
    for face_index, face in enumerate(mesh.faces):
        for corner_index, vertex_id in enumerate(face):
            xy = by_id[vertex_id].metadata["raster_xy"]
            corners.append(
                AppearanceCornerBinding(
                    face_index,
                    corner_index,
                    (xy[0] / 31.0, xy[1] / 31.0),
                    mesh.view_index,
                    xy,
                    f"OBS:{mesh.view_index}",
                    "OBSERVED_LOCAL",
                )
            )
    return build_appearance_binding(
        target_view_index=mesh.view_index,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        camera_binding_hash=mesh.camera_binding_hash,
        corner_bindings=tuple(corners),
    )


def _render_set(mechanical):
    directions = []
    for view in range(8):
        mesh = _p1_mesh(view)
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
            materialization_manifest_sha256="sha256:materialized-p1",
            direct_binding_manifest_sha256="sha256:direct-v5",
        )
        directions.append(
            build_external_directional_renderable(
                view_index=view,
                camera_binding_hash=mesh.camera_binding_hash,
                components=(component,),
                mechanical=mechanical,
            )
        )
    return build_external_directional_renderable_set(tuple(directions), mechanical)


def _capability_contract():
    return build_capability_contract(
        "P1_CONTINUITY_TEST",
        (
            CapabilityRequirement(
                "VISUAL_8_DIRECTION", "REQUIRED", "visual", "policy",
                ("DIRECTIONAL_VISUAL",),
            ),
            CapabilityRequirement(
                "PRESET_MOTION", "REQUIRED", "motion", "policy",
                ("MOTION",),
            ),
        ),
    )


def _component_evidence(component_id: str, mechanical_class: str, *, parent: str = ""):
    one_hot = mechanical_class == "RIGID_SKINNED_COMPONENT"
    return ComponentAttachmentEvidenceIR(
        component_id=component_id,
        source_provenance_refs=(f"source:{component_id}",),
        mechanical_class=mechanical_class,
        directional_render_membership=tuple((view, "BODY") for view in range(8)),
        geometry_membership_refs=(f"P1:partition:{component_id}",),
        geometry_lineage_hash="P1Q:FULL_SUBJECT",
        skin_deformer_lineage_hash="FIT1:ARACHNE:V5:DIRECT",
        canonical_parent_joint_id=parent,
        socket_id="",
        bind_state_authority_hash=f"bind:{component_id}",
        detachability_class="FIXED_COMPONENT",
        visible_required=True,
        one_hot_carry_verified=one_hot,
        one_hot_carry_joint_id=parent if one_hot else "",
        visual_only_qualification_hash="",
        exclusion_reason="",
        qualification_evidence_hash=f"evidence:{component_id}",
        metadata={"fixture": True},
    )


def _component_assembly(mechanical, render_set):
    body = qualify_component_attachment(
        _component_evidence("body", "DEFORMABLE_COMPONENT"),
        mechanical.skeleton,
        qualification_report={
            "passed": True,
            "source_authority_verified": True,
            "classification_inferred_from_filename": False,
            "deformable_multi_joint_support": True,
        },
    )
    book = qualify_component_attachment(
        _component_evidence("book", "RIGID_SKINNED_COMPONENT", parent="J:ARM"),
        mechanical.skeleton,
        qualification_report={
            "passed": True,
            "source_authority_verified": True,
            "classification_inferred_from_filename": False,
        },
    )
    return qualify_component_assembly(
        (body, book),
        mechanical.skeleton,
        required_visible_component_ids=("body", "book"),
        directional_renderables=render_set,
        metadata={"fixture": "typed-source-partition"},
    )


def _underlay_set(mechanical, render_set, assembly):
    views = tuple(
        qualify_continuity_underlay(
            view_index=direction.view_index,
            substrate_component=direction.components[0],
            mechanical=mechanical,
            component_assembly=assembly,
            partition_authority_sha256="sha256:typed-source-partition-authority",
        )
        for direction in render_set.directions
    )
    return build_continuity_underlay_set(views, component_assembly=assembly)


def _external_product():
    mechanical = _mechanical()
    render_set = _render_set(mechanical)
    motion = build_deterministic_preset_motion(mechanical)
    product = assemble_product_v3_with_external_render_support(
        mechanical,
        render_set,
        _capability_contract(),
        motion,
    )
    return mechanical, render_set, product


def test_proof_mesh_uses_external_cached_raster_witness_not_scientific_surface():
    mechanical, _, product = _external_product()
    report = _mesh(product)

    assert mechanical.surface.geometry_lineage_hash == "FIT1:SCIENTIFIC:S"
    assert report["component_count"] == 8
    assert report["external_render_support_component_count"] == 8
    assert report["external_render_support_fail_closed"] is True
    assert {
        row["raster_authority"] for row in report["component_reports"]
    } == {"QUALIFIED_EXTERNAL_COMPONENT_CACHED_RASTER_WITNESS"}
    assert all(row["external_render_support"] is True for row in report["component_reports"])


def test_proof_visual_uses_external_validator_for_external_product():
    _, _, product = _external_product()
    report = _visual(product)
    assert report["direction_count"] == 8
    assert report["external_render_support"] is True
    assert report["visual_validation_authority"] == "QUALIFIED_EXTERNAL_RENDER_SUPPORT"


def test_external_proof_claim_cannot_fall_back_when_qualification_is_missing():
    _, render_set, product = _external_product()
    first = render_set.directions[0].components[0]
    bad_metadata = dict(first.metadata)
    bad_metadata.pop("external_render_support_qualification")
    bad_component = replace(first, metadata=bad_metadata)
    bad_direction = replace(render_set.directions[0], components=(bad_component,))
    bad_set = replace(render_set, directions=(bad_direction, *render_set.directions[1:]))
    bad_product = replace(product, directional_renderables=bad_set)

    with pytest.raises(QualificationError, match="EXTERNAL_RENDER_SUPPORT_METADATA_INCONSISTENT"):
        _mesh(bad_product)


def test_continuity_underlay_binds_exact_external_substrate_and_component_assembly():
    mechanical = _mechanical()
    render_set = _render_set(mechanical)
    assembly = _component_assembly(mechanical, render_set)
    underlays = _underlay_set(mechanical, render_set, assembly)

    assert len(underlays.views) == 8
    assert underlays.component_assembly_hash == assembly.component_assembly_hash
    assert {row.deformable_component_ids for row in underlays.views} == {("body",)}
    assert {row.foreground_component_ids for row in underlays.views} == {("book",)}

    bound = bind_continuity_underlay_set_to_directional_renderables(
        render_set,
        underlays,
        assembly,
        mechanical,
    )
    assert bound.metadata["continuity_underlay_qualified"] is True
    assert bound.metadata["continuity_underlay_set_hash"] == underlays.qualification_hash
    assert bound.metadata["component_assembly_hash"] == assembly.component_assembly_hash


def test_continuity_underlay_rejects_placeholder_partition_and_mutation_claims():
    mechanical = _mechanical()
    render_set = _render_set(mechanical)
    assembly = _component_assembly(mechanical, render_set)
    component = render_set.directions[0].components[0]

    with pytest.raises(QualificationError, match="INVALID_PARTITION_AUTHORITY_SHA256"):
        qualify_continuity_underlay(
            view_index=0,
            substrate_component=component,
            mechanical=mechanical,
            component_assembly=assembly,
            partition_authority_sha256="pending",
        )

    with pytest.raises(QualificationError, match="FORBIDDEN_MUTATION:weights_mutated"):
        qualify_continuity_underlay(
            view_index=0,
            substrate_component=component,
            mechanical=mechanical,
            component_assembly=assembly,
            partition_authority_sha256="sha256:typed-source-partition-authority",
            metadata={"weights_mutated": True},
        )


def test_continuity_underlay_rejects_tampered_component_assembly_hash():
    mechanical = _mechanical()
    render_set = _render_set(mechanical)
    assembly = _component_assembly(mechanical, render_set)
    tampered = replace(assembly, component_assembly_hash="sha256:tampered")

    with pytest.raises(QualificationError, match="COMPONENT_ASSEMBLY_HASH_MISMATCH"):
        qualify_continuity_underlay(
            view_index=0,
            substrate_component=render_set.directions[0].components[0],
            mechanical=mechanical,
            component_assembly=tampered,
            partition_authority_sha256="sha256:typed-source-partition-authority",
        )


def test_continuity_proof_requires_all_eight_views_to_pass_and_never_implies_product_pass():
    mechanical = _mechanical()
    render_set = _render_set(mechanical)
    assembly = _component_assembly(mechanical, render_set)
    underlays = _underlay_set(mechanical, render_set, assembly)
    bound = bind_continuity_underlay_set_to_directional_renderables(
        render_set,
        underlays,
        assembly,
        mechanical,
    )

    clean = tuple(
        qualify_continuity_raster_measurement(
            view_index=view,
            underlay=underlays.views[view],
            boundary_sample_set_sha256=f"sha256:boundary:{view}",
            composed_raster_sha256=f"sha256:raster:{view}",
            exposed_seam_pixel_count=0,
            evaluated_boundary_pixel_count=100,
            max_allowed_exposed_seam_fraction=0.01,
        )
        for view in range(8)
    )
    failed_view = qualify_continuity_raster_measurement(
        view_index=4,
        underlay=underlays.views[4],
        boundary_sample_set_sha256="sha256:boundary:4:failed",
        composed_raster_sha256="sha256:raster:4:failed",
        exposed_seam_pixel_count=2,
        evaluated_boundary_pixel_count=100,
        max_allowed_exposed_seam_fraction=0.01,
    )
    mixed = (*clean[:4], failed_view, *clean[5:])

    with pytest.raises(QualificationError, match="CONTINUITY_PROOF_VIEW_FAILED:4"):
        build_continuity_underlay_proof(
            directional_renderables=bound,
            continuity_underlays=underlays,
            component_assembly=assembly,
            measurements=mixed,
        )

    proof = build_continuity_underlay_proof(
        directional_renderables=bound,
        continuity_underlays=underlays,
        component_assembly=assembly,
        measurements=clean,
    )
    assert proof.metadata["status"] == "PASS"
    assert proof.metadata["all_eight_views_measured"] is True
    assert proof.metadata["product_pass_implied"] is False
    assert proof.proof_hash


def test_continuity_product_runtime_policy_is_hash_bound_and_forbids_hidden_repair():
    mechanical = _mechanical()
    render_set = _render_set(mechanical)
    assembly = _component_assembly(mechanical, render_set)
    underlays = _underlay_set(mechanical, render_set, assembly)
    motion = build_deterministic_preset_motion(mechanical)

    product = assemble_product_v3_with_continuity_underlay(
        mechanical,
        render_set,
        assembly,
        underlays,
        _capability_contract(),
        motion,
    )

    assert product.runtime_policy["mechanical_continuity_underlay_required"] is True
    assert product.runtime_policy["continuity_underlay_set_hash"] == underlays.qualification_hash
    assert product.runtime_policy["component_assembly_hash"] == assembly.component_assembly_hash
    assert product.runtime_policy["continuity_underlay_may_generate_new_art"] is False
    assert product.runtime_policy["continuity_underlay_may_mutate_topology"] is False
    assert product.runtime_policy["continuity_underlay_may_mutate_skin_weights"] is False
    assert product.runtime_policy["scientific_mechanical_surface_relabelled"] is False
