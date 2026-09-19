from __future__ import annotations

"""Build the qualified presentation aggregate from explicit subordinate authorities."""

from dataclasses import replace

from .hashing import content_sha256
from .presentation_structure_v1 import (
    build_presentation_structure,
    validate_presentation_structure,
)
from .product_appearance_v1 import (
    build_product_appearance_set,
    validate_product_appearance_set,
)
from .product_composition_v1 import (
    build_composition_set,
    validate_composition_set,
)
from .product_authority_v1 import (
    PresentationDecisionEvidenceIR,
    PresentationViewOverlayIR,
    QualifiedPresentationGraphIR,
    qualified_presentation_lineage_hash,
    validate_qualified_presentation_graph,
)


def build_qualified_presentation_bundle(
    *,
    surface,
    skeleton,
    skin,
    mesh,
    partition,
    carrier_policy,
    envelope,
    product_state,
    observation_set,
):
    structure=build_presentation_structure(
        surface=surface,skeleton=skeleton,skin=skin,mesh=mesh,partition=partition,
        carrier_policy=carrier_policy,product_state=product_state,
    )
    appearance=build_product_appearance_set(
        surface=surface,mesh=mesh,observation_set=observation_set
    )
    composition=build_composition_set(
        mesh=mesh,observation_set=observation_set,
        presentation_structure=structure,product_state=product_state,
    )

    app_by_view={int(row.target_view_index):row for row in appearance.bindings}
    comp_by_view={int(row.view_index):row for row in composition.views}
    obs_by_view={int(row.view_index):row for row in observation_set.views}
    overlays=tuple(
        PresentationViewOverlayIR(
            view_index=view,
            camera_binding_hash=obs_by_view[view].camera_binding_hash,
            appearance_binding_hash=app_by_view[view].appearance_lineage_hash,
            composition_binding_hash=comp_by_view[view].composition_binding_hash,
            metadata={
                "source_raster_sha256":obs_by_view[view].source_raster_sha256,
                "appearance_set_hash":appearance.appearance_set_hash,
                "composition_set_hash":composition.composition_set_hash,
            },
        )
        for view in range(8)
    )

    decisions=list(structure.decisions)
    for view in range(8):
        decisions.append(PresentationDecisionEvidenceIR(
            decision_id="DEC:"+content_sha256({
                "kind":"SOURCE_APPEARANCE",
                "view":view,
                "appearance":app_by_view[view].appearance_lineage_hash,
            })[:24],
            decision_kind="APPEARANCE_BINDING",
            authority_class="SOURCE_APPEARANCE",
            evidence_refs=(
                app_by_view[view].appearance_lineage_hash,
                observation_set.observation_set_hash,
                obs_by_view[view].source_raster_sha256,
            ),
            metadata={
                "view_index":view,
                "face_uniform_donor":True,
                "cross_view_color_blending":False,
            },
        ))
        decisions.append(PresentationDecisionEvidenceIR(
            decision_id="DEC:"+content_sha256({
                "kind":"POSED_GEOMETRY_COMPOSITION",
                "view":view,
                "composition":comp_by_view[view].composition_binding_hash,
            })[:24],
            decision_kind="SETUP_COMPOSITION",
            authority_class="POSED_GEOMETRY",
            evidence_refs=(
                comp_by_view[view].composition_binding_hash,
                str(mesh.qualification_report["g5_evidence_hash"]),
            ),
            metadata={
                "view_index":view,
                "physical_occlusion_rule":comp_by_view[view].physical_occlusion_rule,
                "slot_order_role":comp_by_view[view].slot_order_role,
            },
        ))

    graph=QualifiedPresentationGraphIR(
        slots=structure.slots,
        attachments=structure.attachments,
        view_overlays=overlays,
        decisions=tuple(decisions),
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        product_state_binding_hash=product_state.product_state_hash,
        presentation_structure_binding_hash=structure.structure_lineage_hash,
        appearance_set_binding_hash=appearance.appearance_set_hash,
        composition_set_binding_hash=composition.composition_set_hash,
        qualification_report={
            "status":"PASS",
            "slot_count":len(structure.slots),
            "attachment_count":len(structure.attachments),
            "view_overlay_count":len(overlays),
            "single_canonical_mesh":True,
            "source_raster_appearance_authority":True,
            "face_uniform_donor":True,
            "cross_view_color_blending":False,
            "physical_occlusion_authority":"CANONICAL_Z_BUFFER_VISIBLE_OWNER_V1",
            "slot_order_solves_physical_occlusion":False,
            "categorical_recognition_used":False,
            "detachability_inferred":False,
            "completion_used":False,
            "planar_proxy_used":False,
        },
        presentation_lineage_hash="",
        metadata={
            "producer":"RealSaS.QualifiedPresentationBundle.v1",
            "representation":"CANONICAL_MESH_PLUS_8_VIEW_PRESENTATION_OVERLAYS",
            "runtime_may_mint_presentation":False,
        },
    )
    graph=replace(graph,presentation_lineage_hash=qualified_presentation_lineage_hash(graph))

    validate_presentation_structure(
        structure,surface=surface,skeleton=skeleton,skin=skin,mesh=mesh,
        partition=partition,carrier_policy=carrier_policy,product_state=product_state,
    )
    validate_product_appearance_set(
        appearance,surface=surface,mesh=mesh,observation_set=observation_set
    )
    validate_composition_set(
        composition,mesh=mesh,observation_set=observation_set,
        presentation_structure=structure,product_state=product_state,
    )
    validate_qualified_presentation_graph(
        graph,carrier_policy=carrier_policy,puppet_state=product_state,
        skeleton=skeleton,partition=partition,envelope=envelope,mesh=mesh,
        presentation_structure=structure,appearance_set=appearance,
        composition_set=composition,
    )
    return structure,appearance,composition,graph
