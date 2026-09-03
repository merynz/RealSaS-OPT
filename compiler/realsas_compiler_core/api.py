from __future__ import annotations
from .surface import build_surface_from_persistence, rigging_surface_from_d2_arrays
from .rig import qualify_skeleton, qualify_skeleton_v2
from .skin import qualify_skin
from .mesh_binding import (
    mesh_candidate_lineage_hash, mesh_lineage_hash, mesh_skin_lineage_hash,
    validate_surface_support_binding, derive_bound_position, validate_mesh_candidate,
    validate_qualified_mesh, validate_qualified_mesh_skin, qualify_identity_subset_mesh,
    bind_identity_mesh_skin,
)
from .directional_binding import (
    qualify_directional_joint_view_binding,
    assert_directional_binding_for_product,
    project_mechanical_point,
    projection_for_view,
    joint_pivot,
)
from .product import assemble_product, assemble_product_v2, bind_proof, require_current_proof, project_runtime_package
from .v4 import (
    upgrade_qualified_skeleton_v2, validate_qualified_skeleton_v2,
    validate_mechanical_state, build_mechanical_state,
    build_appearance_binding, validate_appearance_binding, qualify_visual_completion,
    build_renderable_component, validate_renderable_component, build_directional_renderable,
    validate_directional_renderable, build_directional_renderable_set,
    validate_directional_renderable_set, build_capability_contract, validate_capability_contract,
    make_single_family_e2e_capability_contract, build_joint_track, build_order_track, build_visibility_track,
    build_motion_state, validate_motion_state, required_proof_domains,
    validate_product_ontology, assemble_product_v3,
    bind_proof_plan, bind_measurement_report, bind_domain_proof, bind_product_proof_bundle,
    require_current_proof_bundle, qualify_capability, project_runtime_package_v3,
)
from .bundle_routes import write_typed_artifact, route_for


class CompilerFacade:
    """Single current programmatic entrypoint for compiler qualification.

    V1/V2 compatibility remains available, but V4 composition authority terminates
    in a directional 2D/2.5D CanonicalPuppetGraph.v3 + ProductProofBundleIR.
    World/camera-space mechanical evidence never implies full-3D reconstruction authority.
    """
    build_surface_from_persistence=staticmethod(build_surface_from_persistence)
    rigging_surface_from_d2_arrays=staticmethod(rigging_surface_from_d2_arrays)
    qualify_skeleton=staticmethod(qualify_skeleton)
    qualify_skeleton_v2=staticmethod(qualify_skeleton_v2)
    qualify_skin=staticmethod(qualify_skin)
    mesh_candidate_lineage_hash=staticmethod(mesh_candidate_lineage_hash)
    mesh_lineage_hash=staticmethod(mesh_lineage_hash)
    mesh_skin_lineage_hash=staticmethod(mesh_skin_lineage_hash)
    validate_surface_support_binding=staticmethod(validate_surface_support_binding)
    derive_bound_position=staticmethod(derive_bound_position)
    validate_mesh_candidate=staticmethod(validate_mesh_candidate)
    validate_qualified_mesh=staticmethod(validate_qualified_mesh)
    validate_qualified_mesh_skin=staticmethod(validate_qualified_mesh_skin)
    qualify_identity_subset_mesh=staticmethod(qualify_identity_subset_mesh)
    bind_identity_mesh_skin=staticmethod(bind_identity_mesh_skin)
    qualify_directional_joint_view_binding=staticmethod(qualify_directional_joint_view_binding)
    assert_directional_binding_for_product=staticmethod(assert_directional_binding_for_product)
    project_mechanical_point=staticmethod(project_mechanical_point)
    projection_for_view=staticmethod(projection_for_view)
    joint_pivot=staticmethod(joint_pivot)
    assemble_product=staticmethod(assemble_product)
    assemble_product_v2=staticmethod(assemble_product_v2)
    bind_proof=staticmethod(bind_proof)
    require_current_proof=staticmethod(require_current_proof)
    project_runtime_package=staticmethod(project_runtime_package)

    upgrade_qualified_skeleton_v2=staticmethod(upgrade_qualified_skeleton_v2)
    validate_qualified_skeleton_v2=staticmethod(validate_qualified_skeleton_v2)
    validate_mechanical_state=staticmethod(validate_mechanical_state)
    build_mechanical_state=staticmethod(build_mechanical_state)
    build_appearance_binding=staticmethod(build_appearance_binding)
    validate_appearance_binding=staticmethod(validate_appearance_binding)
    qualify_visual_completion=staticmethod(qualify_visual_completion)
    build_renderable_component=staticmethod(build_renderable_component)
    validate_renderable_component=staticmethod(validate_renderable_component)
    build_directional_renderable=staticmethod(build_directional_renderable)
    validate_directional_renderable=staticmethod(validate_directional_renderable)
    build_directional_renderable_set=staticmethod(build_directional_renderable_set)
    validate_directional_renderable_set=staticmethod(validate_directional_renderable_set)
    build_capability_contract=staticmethod(build_capability_contract)
    validate_capability_contract=staticmethod(validate_capability_contract)
    make_single_family_e2e_capability_contract=staticmethod(make_single_family_e2e_capability_contract)
    build_joint_track=staticmethod(build_joint_track)
    build_order_track=staticmethod(build_order_track)
    build_visibility_track=staticmethod(build_visibility_track)
    build_motion_state=staticmethod(build_motion_state)
    validate_motion_state=staticmethod(validate_motion_state)
    required_proof_domains=staticmethod(required_proof_domains)
    validate_product_ontology=staticmethod(validate_product_ontology)
    assemble_product_v3=staticmethod(assemble_product_v3)
    bind_proof_plan=staticmethod(bind_proof_plan)
    bind_measurement_report=staticmethod(bind_measurement_report)
    bind_domain_proof=staticmethod(bind_domain_proof)
    bind_product_proof_bundle=staticmethod(bind_product_proof_bundle)
    require_current_proof_bundle=staticmethod(require_current_proof_bundle)
    qualify_capability=staticmethod(qualify_capability)
    project_runtime_package_v3=staticmethod(project_runtime_package_v3)

    write_typed_artifact=staticmethod(write_typed_artifact)
    route_for=staticmethod(route_for)
