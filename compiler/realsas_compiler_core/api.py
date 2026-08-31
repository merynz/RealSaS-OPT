from __future__ import annotations
from .surface import build_surface_from_persistence, rigging_surface_from_d2_arrays
from .rig import qualify_skeleton
from .skin import qualify_skin
from .mesh_binding import (
    mesh_candidate_lineage_hash,
    mesh_lineage_hash,
    mesh_skin_lineage_hash,
    validate_surface_support_binding,
    derive_bound_position,
    validate_mesh_candidate,
    validate_qualified_mesh,
    validate_qualified_mesh_skin,
    qualify_identity_subset_mesh,
    bind_identity_mesh_skin,
)
from .product import assemble_product, assemble_product_v2, bind_proof, require_current_proof, project_runtime_package
from .bundle_routes import write_typed_artifact, route_for

class CompilerFacade:
    """Single current programmatic entrypoint for compiler qualification.

    No model loading, teacher identity, source-rig truth or canonical-ID prediction
    is accepted by this facade.
    """
    build_surface_from_persistence=staticmethod(build_surface_from_persistence)
    rigging_surface_from_d2_arrays=staticmethod(rigging_surface_from_d2_arrays)
    qualify_skeleton=staticmethod(qualify_skeleton)
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
    assemble_product=staticmethod(assemble_product)
    assemble_product_v2=staticmethod(assemble_product_v2)
    bind_proof=staticmethod(bind_proof)
    require_current_proof=staticmethod(require_current_proof)
    project_runtime_package=staticmethod(project_runtime_package)
    write_typed_artifact=staticmethod(write_typed_artifact)
    route_for=staticmethod(route_for)
