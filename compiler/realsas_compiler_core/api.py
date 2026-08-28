from __future__ import annotations
from .surface import build_surface_from_persistence, rigging_surface_from_d2_arrays
from .rig import qualify_skeleton
from .skin import qualify_skin
from .product import assemble_product, bind_proof, require_current_proof, project_runtime_package
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
    assemble_product=staticmethod(assemble_product)
    bind_proof=staticmethod(bind_proof)
    require_current_proof=staticmethod(require_current_proof)
    project_runtime_package=staticmethod(project_runtime_package)
    write_typed_artifact=staticmethod(write_typed_artifact)
    route_for=staticmethod(route_for)
